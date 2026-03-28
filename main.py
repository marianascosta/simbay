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
from src.utils.metrics import phase_4_step_observability
from src.utils.metrics import update_warp_memory_metrics
from src.utils.mujoco_utils import initialize_mujoco_env
from src.utils.settings import BACKEND
from src.utils.settings import DEFAULT_OBJECT_PROPS
from src.utils.settings import FRANKA_HOME_QPOS
from src.utils.settings import HEADLESS
from src.utils.settings import NUM_PARTICLES
from src.utils.settings import RUN_ID
from src.utils.tracing import get_tracer
from src.utils.tracing import trace_call
from src.utils.tracing import force_flush_tracing
from src.utils.tracing import set_span_attributes
from src.utils.tracing import setup_tracing
from src.utils.tracing import shutdown_tracing
from src.utils.tracing import span as tracing_span
from src.utils.profiling import annotate


shutdown_requested = False
shutdown_signal_name: str | None = None
LOGGER: Any | None = None
METRICS: Any | None = None

def init_runtime(run_id: str = RUN_ID) -> dict[str, Any]:
    global LOGGER
    global METRICS
    runtime_log_data = {"run_id": run_id}
    setup_tracing(run_id=run_id)
    runtime_tracer = get_tracer("simbay.main")
    runtime_metrics = init_metrics(run_id=run_id)
    runtime_logger = setup_logging(run_id=run_id)
    LOGGER = runtime_logger
    METRICS = runtime_metrics
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
        logger.info({**log_data, "msg": f"Received {shutdown_signal_name} and will finish the current run before shutting down.", "signal": shutdown_signal_name, "mode": "graceful", "action": "finish_current_run"})

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
    logger.info({**log_data, "msg": "Completed simulation setup.", "backend": backend, "headless": headless})
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
    metrics: Any,
    logger: Any,
    log_data: dict[str, object],
    span_attrs: dict[str, Any],
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
            **span_attrs,
            "ik.target_quat_dim": int(target_quat.shape[0]),
            "ik.pre_grasp_height": float(pre_grasp_pos[2]),
            "ik.lift_height": float(lift_pos[2]),
        }
    )
    if backend == "mujoco-warp":
        warmed_rollout_lengths = particle_filter.warmup_runtime([len(traj1), len(traj2), len(traj3)])
        logger.info({**log_data, "msg": f"Finished backend runtime warm-up for the {backend} backend.", "backend": backend})
    return {
        "traj1": traj1,
        "traj2": traj2,
        "traj3": traj3,
        "traj4": traj4,
    }


@trace_call("simbay.main", span_name="robot_execute")
def robot_execute(
    *,
    phase: str,
    trajectory: list[np.ndarray] | np.ndarray,
    real_robot: Any,
    viewer: Any,
    dt: float,
    span_attrs: dict[str, Any],
    log_data: dict[str, Any],
) -> float:
    if LOGGER is None or METRICS is None:
        raise RuntimeError("Runtime logger and metrics must be initialized before robot execution.")
    substage = "robot_execute"
    if phase == "phase_1_approach":
        started_msg = "Started robot execute for phase 1 (approach)."
        finished_msg = "Finished robot motion for phase 1 (approach)."
    elif phase == "phase_2_descend":
        started_msg = "Started robot execute for phase 2 (descent)."
        finished_msg = "Finished robot motion for phase 2 (descent)."
    elif phase == "phase_3_grip":
        started_msg = "Started robot execute for phase 3 (grip)."
        finished_msg = "Finished robot motion for phase 3 (grip)."
    elif phase == "phase_4_lift":
        started_msg = "Started robot execute for phase 4 (lift)."
        finished_msg = "Finished robot motion for phase 4 (lift)."
    else:
        started_msg = f"Started robot execute for {phase.replace('_', ' ')}."
        finished_msg = f"Finished robot motion for {phase.replace('_', ' ')}."
    span_attrs = {
        **span_attrs,
        "simbay.stage": phase,
        "simbay.trajectory_step_count": len(trajectory),
        "simbay.substage": substage,
        "simbay.substage_execution_strategy": "single_robot_control_loop",
        "simbay.particles_updated_at_the_same_time": 1,
        "simbay.commanded_robot_motion_seconds": len(trajectory) * dt,
    }
    set_span_attributes(span_attrs)
    stage_token = METRICS.start_substage(phase, substage)
    LOGGER.info(
        {
            **log_data,
            "msg": started_msg,
        }
    )
    for _, qpos in enumerate(trajectory):
        real_robot.move_joints(qpos)
        if viewer is not None:
            viewer.sync()
    duration = METRICS.finish_substage(stage_token)
    LOGGER.info(
        {
            **log_data,
            "msg": finished_msg,
        }
    )
    METRICS.set_substage_workload(phase, substage, len(trajectory), 1, duration)
    return duration


@trace_call("simbay.main", span_name="pf_replay")
def pf_replay(
    *,
    phase: str,
    trajectory: list[np.ndarray] | np.ndarray,
    particle_filter: Any,
    backend: str,
    span_attrs: dict[str, Any],
    log_data: dict[str, Any],
) -> float:
    if LOGGER is None or METRICS is None:
        raise RuntimeError("Runtime logger and metrics must be initialized before particle filter replay.")
    substage = "pf_replay"
    if phase == "phase_1_approach":
        started_msg = "Started particle filter replay for phase 1 (approach)."
        finished_msg = "Finished particle filter replay for phase 1 (approach)."
    elif phase == "phase_2_descend":
        started_msg = "Started particle filter replay for phase 2 (descent)."
        finished_msg = "Finished particle filter replay for phase 2 (descent)."
    elif phase == "phase_3_grip":
        started_msg = "Started particle filter replay for phase 3 (grip)."
        finished_msg = "Finished particle filter replay for phase 3 (grip)."
    elif phase == "phase_4_lift":
        started_msg = "Started particle filter replay for phase 4 (lift)."
        finished_msg = "Finished particle filter replay for phase 4 (lift)."
    else:
        started_msg = f"Started particle filter replay for {phase.replace('_', ' ')}."
        finished_msg = f"Finished particle filter replay for {phase.replace('_', ' ')}."
    span_attrs = {
        **span_attrs,
        "simbay.stage": phase,
        "simbay.trajectory_step_count": len(trajectory),
        "simbay.substage": substage,
        "simbay.substage_execution_strategy": (
            "trajectory_replay_across_all_particles_in_batch"
            if backend == "mujoco-warp"
            else "trajectory_replay_one_control_step_at_a_time"
        ),
        "simbay.particles_updated_at_the_same_time": particle_filter.N if backend == "mujoco-warp" else 1,
    }
    set_span_attributes(span_attrs)
    stage_token = METRICS.start_substage(phase, substage)
    LOGGER.info(
        {
            **log_data,
            "msg": started_msg,
        }
    )
    if backend == "mujoco-warp":
        particle_filter.predict_trajectory(trajectory)
    else:
        for qpos in trajectory:
            particle_filter.predict(qpos)
    duration = METRICS.finish_substage(stage_token)
    LOGGER.info(
        {
            **log_data,
            "msg": finished_msg,
        }
    )
    METRICS.set_substage_workload(phase, substage, len(trajectory), particle_filter.N, duration)
    return duration


def phase_4_warp_step_logic(
    particle_filter: Any,
    qpos: np.ndarray,
    noisy_ft_reading: np.ndarray,
    *,
    update_and_optionally_resample: Any,
    uniform_weight_metrics: Any,
    attempt: int = 1,
) -> dict[str, Any]:
    particle_filter.particles = particle_filter.env.propagate(particle_filter.particles, qpos)
    likelihoods = particle_filter.env.compute_likelihoods(particle_filter.particles, noisy_ft_reading)
    diagnostics = particle_filter.env.last_measurement_diagnostics()
    if not particle_filter._measurement_is_valid(diagnostics):
        return particle_filter._skip_invalid_update(diagnostics, attempt=attempt)
    if not particle_filter._measurement_is_informative(diagnostics):
        return particle_filter._skip_uninformative_update(diagnostics)

    offset = float(particle_filter._rng.uniform())
    (
        particle_filter.weights,
        particle_filter.particles,
        particle_filter._ess,
        indexes,
        did_resample,
    ) = update_and_optionally_resample(
        particle_filter.weights,
        particle_filter.particles,
        likelihoods,
        offset,
    )
    if did_resample:
        particle_filter.env.resample_states(indexes)
        particle_filter._resample_count += 1
    particle_filter._save_last_good_snapshot()
    uniform_weight_l1, uniform_weight_max_dev, collapsed_to_uniform = uniform_weight_metrics(particle_filter.weights)
    particle_filter._step_index += 1
    return {
        "ess": float(particle_filter._ess),
        "resampled": did_resample,
        "resample_count": particle_filter._resample_count,
        "uniform_weight_l1_distance": uniform_weight_l1,
        "uniform_weight_max_deviation": uniform_weight_max_dev,
        "collapsed_to_uniform": collapsed_to_uniform,
        "diagnostics": diagnostics,
        "skipped_invalid_update": False,
        "skipped_invalid_updates": particle_filter._skipped_invalid_updates,
        "bootstrap_attempts": attempt,
        "uninformative_update": False,
    }


def phase_4_step_logic(
    *,
    backend: str,
    stage_state: dict[str, Any],
    particle_filter: Any,
    real_robot: Any,
    viewer: Any,
    qpos: np.ndarray,
    uniform_weight_metrics: Any,
    update_and_optionally_resample: Any,
) -> dict[str, Any]:
    robot_execute_start = time.perf_counter()
    real_robot.move_joints(qpos)
    if viewer is not None:
        viewer.sync()
    stage_state["robot_execute_total"] += time.perf_counter() - robot_execute_start

    measurements = real_robot.get_sensor_reads()
    noisy_ft_reading = measurements + np.random.normal(0, 0.5, size=3)

    with annotate("phase4_particle_filter_step"):
        if backend == "mujoco-warp":
            if not stage_state["bootstrap_applied"]:
                last_result = None
                for attempt in range(1, 4):
                    step_result = phase_4_warp_step_logic(
                        particle_filter,
                        qpos,
                        noisy_ft_reading,
                        update_and_optionally_resample=update_and_optionally_resample,
                        uniform_weight_metrics=uniform_weight_metrics,
                        attempt=attempt,
                    )
                    last_result = step_result
                    if not bool(step_result.get("skipped_invalid_update", False)):
                        if attempt > 1:
                            step_result["recovered_first_update_attempts"] = attempt
                        break
                step_result = last_result if last_result is not None else {
                    "ess": float(particle_filter.effective_sample_size()),
                    "resampled": False,
                    "resample_count": particle_filter._resample_count,
                    "uniform_weight_l1_distance": 0.0,
                    "uniform_weight_max_deviation": 0.0,
                    "collapsed_to_uniform": False,
                    "skipped_invalid_update": True,
                    "skipped_invalid_updates": particle_filter._skipped_invalid_updates,
                    "bootstrap_attempts": 1,
                    "uninformative_update": False,
                }
                stage_state["bootstrap_applied"] = True
            else:
                step_result = phase_4_warp_step_logic(
                    particle_filter,
                    qpos,
                    noisy_ft_reading,
                    update_and_optionally_resample=update_and_optionally_resample,
                    uniform_weight_metrics=uniform_weight_metrics,
                )
        else:
            particle_filter.predict(qpos)
            particle_filter.update(noisy_ft_reading)
            did_resample = particle_filter.effective_sample_size() < (particle_filter.N / 2)
            if did_resample:
                particle_filter.resample()
            step_result = {
                "ess": particle_filter.effective_sample_size(),
                "resampled": did_resample,
                "resample_count": getattr(
                    particle_filter,
                    "_resample_count",
                    stage_state["resample_count"] + int(did_resample),
                ),
                "uniform_weight_l1_distance": 0.0,
                "uniform_weight_max_deviation": 0.0,
                "collapsed_to_uniform": False,
            }
    return step_result


@observed_stage("phase_4_lift", env_arg="env")
@trace_call("simbay.main", span_name="phase_4_lift")
def run_phase_4_lift(
    *,
    tracer: Any,
    run_id: str,
    started_at: float,
    span_attrs: dict[str, Any],
    backend: str,
    trajectory: list[np.ndarray] | np.ndarray,
    real_robot: Any,
    viewer: Any,
    particle_filter: Any,
    env: Any,
    true_mass: float,
    metrics: Any,
    logger: Any,
    log_data: dict[str, Any],
    uniform_weight_metrics: Any,
    update_and_optionally_resample: Any,
) -> LiftPhaseResult:
    phase = "phase_4_lift"
    stage_state = init_stage_state(phase)
    phase_started_at = time.perf_counter()
    with tracing_span(tracer, phase):
        set_span_attributes(
            {
                **span_attrs,
                "simbay.stage": phase,
                "simbay.phase_trajectory_step_count": len(trajectory),
            }
        )
        for step, qpos in enumerate(trajectory):
            metrics.set_stage_duration(phase, time.perf_counter() - phase_started_at)
            with tracing_span(tracer, "step"):
                set_span_attributes(
                    {
                        **span_attrs,
                        "simbay.stage": phase,
                    }
                )
                set_span_attributes(
                    {
                        "simbay.phase_step_index": step,
                        "simbay.particle_update_strategy": (
                            "single_control_step_across_all_particles_in_batch"
                            if backend == "mujoco-warp"
                            else "single_control_step_one_particle_update_at_a_time"
                        ),
                        "simbay.particle_updates_per_control_step": (particle_filter.N if backend == "mujoco-warp" else 1),
                    }
                )
                step_wall_start = time.perf_counter()
                step_cpu_start = time.process_time()
                step_result = phase_4_step_logic(
                    backend=backend,
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
                    run_id=run_id,
                    backend=backend,
                    started_at=started_at,
                    span_attrs=span_attrs,
                    log_data=log_data,
                    logger=logger,
                    metrics=metrics,
                    stage_state=stage_state,
                    particle_filter=particle_filter,
                    step=step,
                    true_mass=true_mass,
                    step_result=step_result,
                    step_wall_duration=step_wall_duration,
                    step_cpu_duration=step_cpu_duration,
                )
        metrics.set_stage_duration(phase, time.perf_counter() - phase_started_at)
        return finalize_phase_4_metrics(
            log_data=log_data,
            logger=logger,
            metrics=metrics,
            state=stage_state,
            trajectory=trajectory,
            particle_filter=particle_filter,
        )


def generate_particle_filter_plot(
    *,
    history_estimates: list[float],
    true_mass: float,
    env: Any,
) -> Path:
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
    return output_path


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
    span_attrs = {
        "simbay.run_id": run_id,
        "simbay.backend": backend,
        "simbay.particles": num_particles,
        "simbay.control_dt": dt,
        "simbay.execution_device": execution_device,
        "simbay.true_mass": float(true_mass),
    }

    set_span_attributes(span_attrs)

    planning_result = ik_planning(
        backend=backend,
        obj_pos=obj_pos,
        dt=dt,
        particle_filter=particle_filter,
        metrics=metrics,
        logger=logger,
        log_data=log_data,
        span_attrs=span_attrs,
    )

    traj1 = planning_result["traj1"]
    traj2 = planning_result["traj2"]
    traj3 = planning_result["traj3"]
    traj4 = planning_result["traj4"]

    phase_token = metrics.start_stage("phase_1_approach")
    logger.info({**log_data, "msg": "Started phase 1 (approach)."})
    try:
        with tracing_span(tracer, "phase_1_approach"):
            set_span_attributes(
                {
                    **span_attrs,
                    "simbay.stage": "phase_1_approach",
                    "simbay.phase_trajectory_step_count": len(traj1),
                }
            )
            robot_execute(phase="phase_1_approach", trajectory=traj1, real_robot=real_robot, viewer=viewer, dt=dt, span_attrs=span_attrs, log_data=log_data)
            pf_replay(phase="phase_1_approach", trajectory=traj1, particle_filter=particle_filter, backend=backend, span_attrs=span_attrs, log_data=log_data)
    finally:
        if backend == "mujoco-warp":
            update_warp_memory_metrics(env, metrics, stage="phase_1_approach")
        metrics.finish_stage(phase_token)
        logger.info({**log_data, "msg": "Finished phase 1 (approach)."})

    phase_token = metrics.start_stage("phase_2_descend")
    logger.info({**log_data, "msg": "Started phase 2 (descent)."})
    try:
        with tracing_span(tracer, "phase_2_descend"):
            set_span_attributes(
                {
                    **span_attrs,
                    "simbay.stage": "phase_2_descend",
                    "simbay.phase_trajectory_step_count": len(traj2),
                }
            )
            robot_execute(phase="phase_2_descend", trajectory=traj2, real_robot=real_robot, viewer=viewer, dt=dt, span_attrs=span_attrs, log_data=log_data)
            pf_replay(phase="phase_2_descend", trajectory=traj2, particle_filter=particle_filter, backend=backend, span_attrs=span_attrs, log_data=log_data)
    finally:
        if backend == "mujoco-warp":
            update_warp_memory_metrics(env, metrics, stage="phase_2_descend")
        metrics.finish_stage(phase_token)
        logger.info({**log_data, "msg": "Finished phase 2 (descent)."})

    phase_token = metrics.start_stage("phase_3_grip")
    logger.info({**log_data, "msg": "Started phase 3 (grip)."})
    try:
        with tracing_span(tracer, "phase_3_grip"):
            set_span_attributes(
                {
                    **span_attrs,
                    "simbay.stage": "phase_3_grip",
                    "simbay.phase_trajectory_step_count": len(traj3),
                }
            )
            robot_execute(phase="phase_3_grip", trajectory=traj3, real_robot=real_robot, viewer=viewer, dt=dt, span_attrs=span_attrs, log_data=log_data)
            pf_replay(phase="phase_3_grip", trajectory=traj3, particle_filter=particle_filter, backend=backend, span_attrs=span_attrs, log_data=log_data)
    finally:
        if backend == "mujoco-warp":
            update_warp_memory_metrics(env, metrics, stage="phase_3_grip")
        metrics.finish_stage(phase_token)
        logger.info({**log_data, "msg": "Finished phase 3 (grip)."})
    lift_result = run_phase_4_lift(
        tracer=tracer,
        run_id=run_id,
        started_at=runtime["started_at"],
        span_attrs=span_attrs,
        backend=backend,
        trajectory=traj4,
        real_robot=real_robot,
        viewer=viewer,
        particle_filter=particle_filter,
        env=env,
        true_mass=true_mass,
        metrics=metrics,
        logger=logger,
        log_data=log_data,
        uniform_weight_metrics=_uniform_weight_metrics,
        update_and_optionally_resample=_update_and_optionally_resample,
    )

    history_estimates = lift_result.history_estimates

    logger.info({**log_data, "msg": "Finished the particle filter run.", "backend": backend})

    awaiting_user_input = not headless and not shutdown_requested
    logger.info({**log_data, "msg": "Finished the execution sequence.", "awaiting_user_input": awaiting_user_input})
    if awaiting_user_input:
        input()
    elif not headless:
        logger.info({**log_data, "msg": "Skipped waiting for user input because shutdown was requested.", "signal": shutdown_signal_name})

    final_prediction = float(particle_filter.estimate())
    time_to_prediction_seconds = time.perf_counter() - runtime["started_at"]
    metrics.set_prediction_ready(
        total_wall_seconds=time_to_prediction_seconds,
        final_error_pct=abs(true_mass - final_prediction) * 100,
    )
    logger.info({**log_data, "msg": "The prediction is ready."})

    logger.info({**log_data, "msg": "Started generating the output plots."})
    output_path = generate_particle_filter_plot(
        history_estimates=history_estimates,
        true_mass=true_mass,
        env=env,
    )
    logger.info({**log_data, "msg": f"Saved the particle filter plot to {output_path}.", "path": str(output_path)})

    gc.collect()
    logger.info({**log_data, "msg": "Finished the run and shut down cleanly.", "shutdown_requested": shutdown_requested, "signal": shutdown_signal_name or "none"})


if __name__ == "__main__":
    runtime = init_runtime()
    install_signal_handlers(runtime["logger"], runtime["log_data"])
    try:
        main(runtime)
    finally:
        if runtime is not None:
            runtime["exit_stack"].close()
