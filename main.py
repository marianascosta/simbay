import gc
import math
import signal
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np

from src.estimation.mujoco_particle_filter import FrankaMuJoCoEnv
from src.estimation.particle_filter import ParticleFilter
from src.estimation.warp_filter import _uniform_weight_metrics
from src.estimation.warp_filter import _update_and_optionally_resample
from src.estimation.warp_filter import WarpParticleFilter
from src.estimation.warp_particle_filter import FrankaWarpEnv
from src.planning import FrankaSmartSolver
from src.planning import plan_linear_trajectory
from src.utils.logging_utils import setup_logging
from src.utils.metrics import LiftPhaseResult
from src.utils.metrics import apply_setup_observability
from src.utils.metrics import observed_stage
from src.utils.metrics import init_metrics
from src.utils.metrics import init_stage_state
from src.utils.metrics import finalize_phase_4_metrics
from src.utils.metrics import shutdown_metrics
from src.utils.metrics import stage_span_attrs
from src.utils.metrics import substage_span_attrs
from src.utils.metrics import phase_4_step_observability
from src.utils.mujoco_utils import initialize_mujoco_env
from src.utils.settings import BACKEND
from src.utils.settings import DEFAULT_OBJECT_PROPS
from src.utils.settings import FRANKA_HOME_QPOS
from src.utils.settings import HEADLESS
from src.utils.settings import NUM_PARTICLES
from src.utils.settings import RUN_ID
from src.utils.tracing import get_tracer
from src.utils.tracing import trace_call
from src.utils.tracing import set_span_attributes
from src.utils.tracing import setup_tracing
from src.utils.tracing import shutdown_tracing
from src.utils.tracing import span as tracing_span
from src.utils.profiling import annotate


shutdown_requested = False
shutdown_signal_name: str | None = None

PHASE_LOG_LABELS = {
    "phase_1_approach": "phase 1 (approach)",
    "phase_2_descend": "phase 2 (descent)",
    "phase_3_grip": "phase 3 (grip)",
    "phase_4_lift": "phase 4 (lift)",
}

SUBSTAGE_FINISHED_LABELS = {
    "robot_execute": "robot motion",
    "pf_replay": "particle filter replay",
    "pf_update": "particle filter update",
}


def init_runtime(run_id: str = RUN_ID) -> dict[str, Any]:
    runtime_log_data = {"run_id": run_id}
    setup_tracing(run_id=run_id)
    runtime_tracer = get_tracer("simbay.main")
    runtime_metrics = init_metrics(run_id=run_id)
    runtime_logger = setup_logging(run_id=run_id)
    exit_stack = ExitStack()
    exit_stack.callback(shutdown_tracing)
    exit_stack.callback(force_flush_tracing)
    exit_stack.callback(shutdown_metrics, runtime_metrics)
    set_span_attributes({"simbay.run_id": run_id})
    return {
        "run_id": run_id,
        "tracer": runtime_tracer,
        "metrics": runtime_metrics,
        "logger": runtime_logger,
        "log_data": runtime_log_data,
        "exit_stack": exit_stack,
        "started_at": time.perf_counter(),
    }


def install_signal_handlers(logger: Any, log_data: dict[str, object]) -> None:
    def _handle_shutdown_signal(signum, _frame) -> None:
        global shutdown_requested
        global shutdown_signal_name

        shutdown_requested = True
        shutdown_signal_name = signal.Signals(signum).name
        logger.info({**log_data, "event": "shutdown_requested", "msg": f"Received {shutdown_signal_name} and will finish the current run before shutting down.", "signal": shutdown_signal_name, "mode": "graceful", "action": "finish_current_run"})

    signal.signal(signal.SIGINT, _handle_shutdown_signal)
    signal.signal(signal.SIGTERM, _handle_shutdown_signal)

@observed_stage("setup")
@trace_call("simbay.main", span_name="setup")
def setup(
    *,
    backend: str,
    headless: bool,
    log_data: dict[str, object],
    logger: Any,
    metrics: Any,
    run_id: str,
) -> dict[str, Any]:
    real_robot = initialize_mujoco_env()
    viewer = None
    if not headless:
        viewer = mujoco.viewer.launch_passive(real_robot.model, real_robot.data)
    real_robot.viewer = viewer
    dt = real_robot.dt

    obj_pos = DEFAULT_OBJECT_PROPS["pos"]
    true_mass = DEFAULT_OBJECT_PROPS["mass"]
    num_particles = NUM_PARTICLES
    limits = (0.0, 3.0)

    if backend == "mujoco-warp":
        env = FrankaWarpEnv(limits, num_particles, logging_data=log_data)
        particle_filter = WarpParticleFilter(env, logging_data=log_data)
    elif backend == "cpu":
        env = FrankaMuJoCoEnv(limits, num_particles)
        particle_filter = ParticleFilter(env, logging_data=log_data)
    else:
        raise AssertionError(f"Unexpected backend: {backend}")

    memory_profile = particle_filter.memory_profile()
    env_memory_profile = env.memory_profile()
    execution_device = str(env_memory_profile.get("execution_device", backend))
    apply_setup_observability(
        run_id=run_id,
        metrics=metrics,
        backend_name=backend,
        num_particles=num_particles,
        dt=dt,
        execution_device=execution_device,
        true_mass=true_mass,
        env=env,
        env_memory_profile=env_memory_profile,
        memory_profile=memory_profile,
    )
    logger.info({**log_data, "event": "simulation_setup", "msg": "Completed simulation setup.", "backend": backend, "headless": headless})
    return {
        "obj_pos": obj_pos,
        "true_mass": true_mass,
        "num_particles": num_particles,
        "limits": limits,
        "real_robot": real_robot,
        "viewer": viewer,
        "dt": dt,
        "env": env,
        "particle_filter": particle_filter,
        "execution_device": execution_device,
    }


@observed_stage("ik_planning")
@trace_call("simbay.main", span_name="ik_planning")
def ik_planning(
    *,
    backend: str,
    obj_pos: np.ndarray,
    dt: float,
    particle_filter: Any,
    logger: Any,
    log_data: dict[str, object],
) -> dict[str, np.ndarray]:
    target_quat = np.array([0.0, 1.0, 0.0, 0.0])

    pre_grasp_pos = obj_pos + np.array([0.0, 0.0, 0.15])
    pre_grasp_q7 = FrankaSmartSolver.solve(FRANKA_HOME_QPOS, np.concatenate([pre_grasp_pos, target_quat]))
    grasp_q7 = FrankaSmartSolver.solve(pre_grasp_q7, np.concatenate([obj_pos, target_quat]))
    lift_pos = obj_pos + np.array([0.0, 0.0, 0.2])
    lift_q7 = FrankaSmartSolver.solve(grasp_q7, np.concatenate([lift_pos, target_quat]))

    q_home = np.append(FRANKA_HOME_QPOS, 255)
    q_pre_grasp = np.append(pre_grasp_q7, 255)
    q_grasp_open = np.append(grasp_q7, 255)
    q_grasp_closed = np.append(grasp_q7, 0)
    q_lift_closed = np.append(lift_q7, 0)

    traj1 = plan_linear_trajectory(q_home, q_pre_grasp, max_velocity=1.0, dt=dt)
    traj2 = plan_linear_trajectory(q_pre_grasp, q_grasp_open, max_velocity=0.5, dt=dt)
    traj3 = plan_linear_trajectory(q_grasp_closed, q_grasp_closed, max_velocity=500, dt=dt, settle_time=0.5)
    traj4 = plan_linear_trajectory(q_grasp_closed, q_lift_closed, max_velocity=0.5, dt=dt, settle_time=1.0)
    set_span_attributes(
        {
            "ik.target_quat_dim": int(target_quat.shape[0]),
            "ik.pre_grasp_height": float(pre_grasp_pos[2]),
            "ik.lift_height": float(lift_pos[2]),
        }
    )
    if backend == "mujoco-warp":
        warmed_rollout_lengths = particle_filter.warmup_runtime([len(traj1), len(traj2), len(traj3)])
        logger.info({**log_data, "event": "backend_runtime_warmup_summary", "msg": f"Finished backend runtime warm-up for the {backend} backend.", "backend": backend, "rollout_lengths": warmed_rollout_lengths, "phase4_step_warmup": 1})
    return {
        "traj1": traj1,
        "traj2": traj2,
        "traj3": traj3,
        "traj4": traj4,
    }


@trace_call("simbay.main", span_name="robot_execute")
def robot_execute(
    ctx: dict[str, Any],
    phase: str,
    trajectory: list[np.ndarray] | np.ndarray,
    real_robot: Any,
    viewer: Any,
) -> float:
    substage = "robot_execute"
    phase_label = PHASE_LOG_LABELS.get(phase, phase.replace("_", " "))
    set_span_attributes(
        substage_span_attrs(
            run_id=ctx["run_id"],
            backend=ctx["backend"],
            num_particles=ctx["num_particles"],
            dt=ctx["dt"],
            execution_device=ctx["execution_device"],
            stage=phase,
            substage=substage,
            steps=len(trajectory),
            execution_mode="sequential",
            parallel_units=1,
        )
    )
    stage_token = ctx["metrics"].start_substage(phase, substage)
    ctx["logger"].info(
        {
            **ctx["log_data"],
            "event": "substage_started",
            "msg": f"Started {substage.replace('_', ' ')} for {phase.replace('_', ' ')}.",
            "phase": phase,
            "substage": substage,
        }
    )
    for _, qpos in enumerate(trajectory):
        real_robot.move_joints(qpos)
        if viewer is not None:
            viewer.sync()
    duration = ctx["metrics"].finish_substage(stage_token)
    ctx["logger"].info(
        {
            **ctx["log_data"],
            "event": "substage_finished",
            "msg": f"Finished {SUBSTAGE_FINISHED_LABELS[substage]} for {phase_label}.",
            "phase": phase,
            "substage": substage,
        }
    )
    ctx["metrics"].set_substage_workload(phase, substage, len(trajectory), 1, duration)
    return duration


@trace_call("simbay.main", span_name="pf_replay")
def pf_replay(
    ctx: dict[str, Any],
    phase: str,
    trajectory: list[np.ndarray] | np.ndarray,
    particle_filter: Any,
) -> float:
    substage = "pf_replay"
    phase_label = PHASE_LOG_LABELS.get(phase, phase.replace("_", " "))
    set_span_attributes(
        substage_span_attrs(
            run_id=ctx["run_id"],
            backend=ctx["backend"],
            num_particles=ctx["num_particles"],
            dt=ctx["dt"],
            execution_device=ctx["execution_device"],
            stage=phase,
            substage=substage,
            steps=len(trajectory),
            execution_mode="batched_parallel" if ctx["backend"] == "mujoco-warp" else "sequential",
            parallel_units=particle_filter.N if ctx["backend"] == "mujoco-warp" else 1,
        )
    )
    stage_token = ctx["metrics"].start_substage(phase, substage)
    ctx["logger"].info(
        {
            **ctx["log_data"],
            "event": "substage_started",
            "msg": f"Started {substage.replace('_', ' ')} for {phase.replace('_', ' ')}.",
            "phase": phase,
            "substage": substage,
        }
    )
    if ctx["backend"] == "mujoco-warp":
        particle_filter.predict_trajectory(trajectory)
    else:
        for qpos in trajectory:
            particle_filter.predict(qpos)
    duration = ctx["metrics"].finish_substage(stage_token)
    ctx["logger"].info(
        {
            **ctx["log_data"],
            "event": "substage_finished",
            "msg": f"Finished {SUBSTAGE_FINISHED_LABELS[substage]} for {phase_label}.",
            "phase": phase,
            "substage": substage,
        }
    )
    ctx["metrics"].set_substage_workload(phase, substage, len(trajectory), particle_filter.N, duration)
    return duration


@observed_stage("phase_4_lift", env_arg="env")
@trace_call("simbay.main", span_name="phase_4_lift")
def run_phase_4_lift(
    ctx: dict[str, Any],
    *,
    steps: int,
    trajectory: list[np.ndarray] | np.ndarray,
    real_robot: Any,
    viewer: Any,
    particle_filter: Any,
    env: Any,
    true_mass: float,
    uniform_weight_metrics: Any,
    update_and_optionally_resample: Any,
) -> LiftPhaseResult:
    phase = "phase_4_lift"
    stage_state = init_stage_state(phase)
    if stage_state is not None:
        ctx["stage_state"] = stage_state
    with tracing_span(ctx["tracer"], phase):
        set_span_attributes(
            stage_span_attrs(
                run_id=ctx["run_id"],
                backend=ctx["backend"],
                num_particles=ctx["num_particles"],
                dt=ctx["dt"],
                execution_device=ctx["execution_device"],
                stage=phase,
                steps=len(trajectory),
            )
        )
        stage_state = ctx["stage_state"]
        for step, qpos in enumerate(trajectory):
            with tracing_span(ctx["tracer"], "step"):
                set_span_attributes(
                    {
                        "simbay.run_id": ctx["run_id"],
                        "simbay.stage": phase,
                        "simbay.backend": ctx["backend"],
                        "simbay.control_dt": ctx["dt"],
                        "simbay.execution_device": ctx["execution_device"],
                    }
                )
                set_span_attributes(
                    {
                        "simbay.step": step,
                        "simbay.execution_mode": ("batched_parallel" if ctx["backend"] == "mujoco-warp" else "sequential"),
                        "simbay.batched_particle_updates": (particle_filter.N if ctx["backend"] == "mujoco-warp" else 1),
                    }
                )
                step_wall_start = time.perf_counter()
                step_cpu_start = time.process_time()
                step_result = phase_4_step_logic(
                    backend=ctx["backend"],
                    stage_state=stage_state,
                    particle_filter=particle_filter,
                    real_robot=real_robot,
                    viewer=viewer,
                    qpos=qpos,
                    uniform_weight_metrics=uniform_weight_metrics,
                    update_and_optionally_resample=update_and_optionally_resample,
                )
                step_wall_duration = time.perf_counter() - step_wall_start
                step_cpu_duration = time.process_time() - step_cpu_start
                phase_4_step_observability(
                    ctx,
                    stage_state,
                    particle_filter=particle_filter,
                    step=step,
                    true_mass=true_mass,
                    step_result=step_result,
                    step_wall_duration=step_wall_duration,
                    step_cpu_duration=step_cpu_duration,
                )
        return finalize_phase_4_metrics(
            ctx,
            stage_state,
            trajectory=trajectory,
            particle_filter=particle_filter,
        )


@trace_call("simbay.main", span_name="main")
def main(runtime: dict[str, Any]) -> None:
    logger = runtime["logger"]
    metrics = runtime["metrics"]
    tracer = runtime["tracer"]
    log_data = runtime["log_data"]
    run_id = runtime["run_id"]

    headless = HEADLESS
    backend = BACKEND
    setup_result = setup(
        backend=backend,
        headless=headless,
        log_data=log_data,
        logger=logger,
        metrics=metrics,
        run_id=run_id,
    )

    obj_pos = setup_result["obj_pos"]
    true_mass = setup_result["true_mass"]
    num_particles = setup_result["num_particles"]
    real_robot = setup_result["real_robot"]
    viewer = setup_result["viewer"]
    dt = setup_result["dt"]
    env = setup_result["env"]
    particle_filter = setup_result["particle_filter"]
    execution_device = setup_result["execution_device"]

    ctx = {
        "run_id": run_id,
        "tracer": tracer,
        "metrics": metrics,
        "logger": logger,
        "log_data": log_data,
        "started_at": runtime["started_at"],
        "backend": backend,
        "num_particles": num_particles,
        "dt": dt,
        "execution_device": execution_device,
    }

    planning_result = ik_planning(
        backend=backend,
        obj_pos=obj_pos,
        dt=dt,
        particle_filter=particle_filter,
        logger=logger,
        log_data=log_data,
    )

    traj1 = planning_result["traj1"]
    traj2 = planning_result["traj2"]
    traj3 = planning_result["traj3"]
    traj4 = planning_result["traj4"]

    uniform_weight_metrics = _uniform_weight_metrics
    update_and_optionally_resample = _update_and_optionally_resample
    if backend == "cpu":
        uniform_weight_metrics = None
        update_and_optionally_resample = None

    robot_execute(ctx, "phase_1_approach", traj1, real_robot, viewer)
    pf_replay(ctx, "phase_1_approach", traj1, particle_filter)
    robot_execute(ctx, "phase_2_descend", traj2, real_robot, viewer)
    pf_replay(ctx, "phase_2_descend", traj2, particle_filter)
    robot_execute(ctx, "phase_3_grip", traj3, real_robot, viewer)
    pf_replay(ctx, "phase_3_grip", traj3, particle_filter)
    lift_result = run_phase_4_lift(
        ctx,
        steps=len(traj4),
        trajectory=traj4,
        real_robot=real_robot,
        viewer=viewer,
        particle_filter=particle_filter,
        env=env,
        true_mass=true_mass,
        uniform_weight_metrics=uniform_weight_metrics,
        update_and_optionally_resample=update_and_optionally_resample,
    )

    history_estimates = lift_result.history_estimates

    if backend == "cpu":
        env_memory_profile = env.memory_profile()
        metrics.update_warp_memory(
            stage="phase_4_lift",
            bytes_in_use=int(env_memory_profile["bytes_in_use"]),
            peak_bytes_in_use=int(env_memory_profile["peak_bytes_in_use"]),
            bytes_limit=int(env_memory_profile["bytes_limit"]),
            state_bytes_estimate=int(env_memory_profile.get("state_bytes_estimate", 0)),
        )
        logger.info({**log_data, "event": "particle_filter_completed", "msg": "Finished the particle filter run.", "backend": "cpu"})
    elif backend == "mujoco-warp":
        logger.info({**log_data, "event": "particle_filter_completed", "msg": "Finished the particle filter run.", "backend": "mujoco-warp"})

    logger.info({**log_data, "event": "sequence_complete", "msg": "Finished the execution sequence.", "awaiting_user_input": not headless})
    if not headless and not shutdown_requested:
        input()
    elif not headless and shutdown_requested:
        logger.info({**log_data, "event": "sequence_complete_skipping_user_input", "msg": "Skipped waiting for user input because shutdown was requested.", "signal": shutdown_signal_name})

    final_prediction = float(particle_filter.estimate())
    time_to_prediction_seconds = time.perf_counter() - runtime["started_at"]
    metrics.set_prediction_ready(
        total_wall_seconds=time_to_prediction_seconds,
        final_error_pct=abs(true_mass - final_prediction) * 100,
    )
    logger.info({**log_data, "event": "prediction_ready", "msg": "The prediction is ready."})

    logger.info({**log_data, "event": "plot_generation_start", "msg": "Started generating the output plots."})
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(history_estimates)), history_estimates, color="red", linewidth=3, label="Filter Estimate (Mean)")
    plt.axhline(y=true_mass, color="green", linestyle="--", linewidth=2, label=f"True Mass ({true_mass} kg)")
    plt.title("Particle Filter: Mass Estimation Evolution", fontsize=14, fontweight="bold")
    plt.xlabel("Simulation Step (Lifting Phase)", fontsize=12)
    plt.ylabel("Estimated Mass (kg)", fontsize=12)
    plt.ylim(env.min, env.max)
    plt.legend(loc="upper right")
    plt.grid(True, linestyle=":", alpha=0.7)
    output_dir = Path("temp")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "particle_filter_evolution.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info({**log_data, "event": "plot_saved", "msg": f"Saved the particle filter plot to {output_path}.", "path": str(output_path)})

    gc.collect()
    logger.info({**log_data, "event": "goodbye", "msg": "Finished the run and shut down cleanly.", "shutdown_requested": shutdown_requested, "signal": shutdown_signal_name or "none"})


if __name__ == "__main__":
    runtime = init_runtime()
    install_signal_handlers(runtime["logger"], runtime["log_data"])
    try:
        main(runtime)
    finally:
        if runtime is not None:
            runtime["exit_stack"].close()
