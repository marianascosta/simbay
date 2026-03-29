import gc
import math
import signal
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

import mujoco
import mujoco.viewer
import numpy as np

from src.estimation.mujoco_particle_filter import FrankaMuJoCoEnv
from src.estimation.particle_filter import ParticleFilter
from src.estimation.warp_filter import WarpParticleFilter
from src.estimation.warp_particle_filter import FrankaWarpEnv
from src.planning import FrankaSmartSolver
from src.planning import plan_linear_trajectory
from src.utils import metrics
from src.utils.logging_utils import logger as simbay_logger, logger
from src.utils.mujoco_utils import initialize_mujoco_env
from src.utils.settings import BACKEND
from src.utils.settings import DEFAULT_OBJECT_PROPS
from src.utils.settings import ENVIRONMENT
from src.utils.settings import FRANKA_HOME_QPOS
from src.utils.settings import HEADLESS
from src.utils.settings import NUM_PARTICLES
from src.utils.settings import OBSERVABILITY_ENABLED
from src.utils.settings import RUN_ID
from src.utils.tracing import get_tracer
from src.utils.tracing import trace_call
from src.utils.tracing import force_flush_tracing
from src.utils.tracing import set_tracing_enabled
from src.utils.tracing import set_span_attributes
from src.utils.tracing import setup_tracing
from src.utils.tracing import shutdown_tracing
from src.utils.tracing import span as tracing_span
from src.utils.profiling import annotate


shutdown_requested = False
shutdown_signal_name: str | None = None
LOGGER: Any | None = None

def install_signal_handlers(logger: Any, log_data: dict[str, object]) -> None:
    def _handle_shutdown_signal(signum, _frame) -> None:
        global shutdown_requested
        global shutdown_signal_name

        shutdown_requested = True
        shutdown_signal_name = signal.Signals(signum).name
        logger.info({**log_data, "msg": f"Received {shutdown_signal_name} and will finish the current run before shutting down.", "signal": shutdown_signal_name, "mode": "graceful", "action": "finish_current_run"})

    signal.signal(signal.SIGINT, _handle_shutdown_signal)
    signal.signal(signal.SIGTERM, _handle_shutdown_signal)

@metrics.observed_stage("setup")
@trace_call("simbay.main", span_name="setup")
def setup(
    *,
    backend: str,
    headless: bool,
    log_data: dict[str, object],
    logger: Any,
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
    metrics.apply_setup_observability(
        run_id=run_id,
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


@metrics.observed_stage("ik_planning")
@trace_call("simbay.main", span_name="ik_planning")
def ik_planning(
    *,
    backend: str,
    obj_pos: np.ndarray,
    dt: float,
    particle_filter: Any,
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

    if backend in {"cpu", "mujoco-warp"}:
        traj1 = plan_linear_trajectory(q_home, q_pre_grasp, max_velocity=1.0, dt=dt, settle_time=0.0)
        traj2 = plan_linear_trajectory(q_pre_grasp, q_grasp_open, max_velocity=0.5, dt=dt, settle_time=0.0)
        traj3 = plan_linear_trajectory(q_grasp_closed, q_grasp_closed, max_velocity=500, dt=dt, settle_time=0.5)
        traj4 = plan_linear_trajectory(q_grasp_closed, q_lift_closed, max_velocity=0.5, dt=dt, settle_time=1.0)
    else:
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
        particle_filter.set_contact_mode(simplified_contacts=True)
        particle_filter.warmup_runtime([len(traj1), len(traj2)])
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
    if LOGGER is None:
        raise RuntimeError("Runtime logger must be initialized before robot execution.")
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
    stage_token = metrics.start_substage(phase, substage)
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
    duration = metrics.finish_substage(stage_token)
    LOGGER.info(
        {
            **log_data,
            "msg": finished_msg,
        }
    )
    metrics.set_substage_workload(phase, substage, len(trajectory), 1, duration)
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
    if LOGGER is None:
        raise RuntimeError("Runtime logger must be initialized before particle filter replay.")
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
    stage_token = metrics.start_substage(phase, substage)
    LOGGER.info(
        {
            **log_data,
            "msg": started_msg,
        }
    )
    if backend == "mujoco-warp":
        # Warp can replay phases 1-3 as a single batched rollout because these
        # phases only need state propagation, not per-step measurement updates.
        particle_filter.predict_trajectory(trajectory)
    else:
        for qpos in trajectory:
            particle_filter.predict(qpos)
    duration = metrics.finish_substage(stage_token)
    LOGGER.info(
        {
            **log_data,
            "msg": finished_msg,
        }
    )
    metrics.set_substage_workload(phase, substage, len(trajectory), particle_filter.N, duration)
    return duration


def phase_4_step_logic(
    *,
    backend: str,
    stage_state: dict[str, Any],
    particle_filter: Any,
    real_robot: Any,
    viewer: Any,
    qpos: np.ndarray,
) -> dict[str, Any]:
    robot_execute_start = time.perf_counter()
    real_robot.move_joints(qpos)
    if viewer is not None:
        viewer.sync()
    stage_state["robot_execute_total"] += time.perf_counter() - robot_execute_start

    measurements = np.asarray(real_robot.get_sensor_reads())
    noisy_ft_reading = measurements + np.random.normal(0, 0.5, size=measurements.shape)
    step_index = len(stage_state["history_estimates"])
    before_update = metrics.maybe_capture_particle_audit(
        stage_state,
        step=step_index,
        particle_filter=particle_filter,
        qpos=qpos,
        observation=noisy_ft_reading,
    )

    with annotate("phase4_particle_filter_step"):
        if backend == "mujoco-warp":
            if not stage_state["bootstrap_applied"]:
                step_result = particle_filter.bootstrap_first_update(qpos, noisy_ft_reading, max_attempts=3)
                stage_state["bootstrap_applied"] = True
            else:
                step_result = particle_filter.step(qpos, noisy_ft_reading)
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
    metrics.validate_particle_invariants(
        particle_filter,
        step=step_index,
        step_result=step_result,
    )
    step_result["before_update"] = before_update
    return step_result


@metrics.observed_stage("phase_4_lift", env_arg="env")
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
    logger: Any,
    log_data: dict[str, Any],
) -> metrics.LiftPhaseResult:
    phase = "phase_4_lift"
    stage_state = metrics.init_stage_state(phase)
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
                )
                step_wall_duration = time.perf_counter() - step_wall_start
                step_cpu_duration = time.process_time() - step_cpu_start
                metrics.phase_4_step_observability(
                    run_id=run_id,
                    backend=backend,
                    started_at=started_at,
                    span_attrs=span_attrs,
                    log_data=log_data,
                    logger=logger,
                    stage_state=stage_state,
                    particle_filter=particle_filter,
                    step=step,
                    true_mass=true_mass,
                    step_result=step_result,
                    step_wall_duration=step_wall_duration,
                    step_cpu_duration=step_cpu_duration,
                )
        metrics.set_stage_duration(phase, time.perf_counter() - phase_started_at)
        return metrics.finalize_phase_4_metrics(
            log_data=log_data,
            logger=logger,
            state=stage_state,
            trajectory=trajectory,
            particle_filter=particle_filter,
        )


def generate_particle_filter_plot(
    *,
    run_id: str,
    history_estimates: list[float],
    true_mass: float,
    env: Any,
    history_particles: list[np.ndarray] | None = None,
    num_particles: int | None = None,
    history_ess: list[float] | None = None,
    history_ci50_low: list[float] | None = None,
    history_ci50_high: list[float] | None = None,
    history_ci90_low: list[float] | None = None,
    history_ci90_high: list[float] | None = None,
    history_resampled: list[bool] | None = None,
) -> Path:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, (ax_mass, ax_ess) = plt.subplots(
        2,
        1,
        figsize=(11, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    particle_count_label = num_particles if num_particles is not None else "unknown"
    steps = list(range(len(history_estimates)))
    if history_particles is not None and num_particles is not None:
        num_steps = len(history_particles)
        for t in range(num_steps):
            ax_mass.scatter([t] * num_particles, history_particles[t], color="steelblue", alpha=0.05, s=15)
    if history_ci90_low is not None and history_ci90_high is not None and len(history_ci90_low) == len(history_estimates):
        ax_mass.fill_between(steps, history_ci90_low, history_ci90_high, color="cornflowerblue", alpha=0.18, label="90% credible interval")
    if history_ci50_low is not None and history_ci50_high is not None and len(history_ci50_low) == len(history_estimates):
        ax_mass.fill_between(steps, history_ci50_low, history_ci50_high, color="royalblue", alpha=0.28, label="50% credible interval")
    ax_mass.plot(steps, history_estimates, color="firebrick", linewidth=2.5, label="Filter estimate")
    ax_mass.axhline(y=true_mass, color="darkgreen", linestyle="--", linewidth=2, label=f"True mass ({true_mass} kg)")
    if history_resampled is not None:
        resample_steps = [step for step, resampled in enumerate(history_resampled) if resampled]
        if resample_steps:
            ax_mass.vlines(resample_steps, env.min, env.max, colors="black", alpha=0.12, linewidth=1, label="Resample")
    ax_mass.set_title(
        f"Particle Filter: Mass Estimation Evolution (N={particle_count_label})\nRun ID: {run_id}",
        fontsize=14,
        fontweight="bold",
    )
    ax_mass.set_ylabel("Mass (kg)", fontsize=12)
    ax_mass.set_ylim(env.min, env.max)
    ax_mass.legend(loc="upper right")
    ax_mass.grid(True, linestyle=":", alpha=0.7)
    if history_ess is not None and len(history_ess) == len(history_estimates):
        ax_ess.plot(steps, history_ess, color="darkorange", linewidth=2, label="ESS")
        if num_particles is not None:
            ax_ess.axhline(y=num_particles / 2.0, color="gray", linestyle="--", linewidth=1.5, label="Resample threshold")
            ax_ess.set_ylim(0.0, max(float(num_particles), max(history_ess, default=0.0) * 1.05))
        ax_ess.legend(loc="upper right")
    ax_ess.set_xlabel("Simulation Step (Lifting Phase)", fontsize=12)
    ax_ess.set_ylabel("ESS", fontsize=12)
    ax_ess.grid(True, linestyle=":", alpha=0.7)
    output_dir = Path("temp")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "particle_filter_evolution.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def run_simbay_mujoco_original(
    *,
    traj1: list[np.ndarray] | np.ndarray,
    traj2: list[np.ndarray] | np.ndarray,
    traj3: list[np.ndarray] | np.ndarray,
    traj4: list[np.ndarray] | np.ndarray,
    real_robot: Any,
    viewer: Any,
    particle_filter: Any,
    logger: Any,
    log_data: dict[str, object],
) -> tuple[list[float], list[np.ndarray]]:
    logger.info({**log_data, "msg": "Moving to Approach position..."})
    for qpos in traj1:
        real_robot.move_joints(qpos)
        if viewer is not None:
            viewer.sync()
        particle_filter.predict(qpos)

    logger.info({**log_data, "msg": "Descending to grasp..."})
    for qpos in traj2:
        real_robot.move_joints(qpos)
        if viewer is not None:
            viewer.sync()
        particle_filter.predict(qpos)

    logger.info({**log_data, "msg": "Closing Gripper..."})
    for qpos in traj3:
        real_robot.move_joints(qpos)
        if viewer is not None:
            viewer.sync()
        particle_filter.predict(qpos)

    logger.info({**log_data, "msg": "Lifting object and running Particle Filter..."})
    history_particles: list[np.ndarray] = []
    history_estimates: list[float] = []
    for _, qpos in enumerate(traj4):
        real_robot.move_joints(qpos)
        if viewer is not None:
            viewer.sync()
        particle_filter.predict(qpos)
        measurements = np.asarray(real_robot.get_sensor_reads())
        noisy_ft_reading = measurements + np.random.normal(0, 0.5, size=measurements.shape)
        particle_filter.update(noisy_ft_reading)
        particle_filter.resample()
        history_particles.append(particle_filter.particles.copy())
        history_estimates.append(float(particle_filter.estimate()))
    return history_estimates, history_particles


def run_simbay_mujoco_warp(
    *,
    tracer: Any,
    run_id: str,
    started_at: float,
    span_attrs: dict[str, Any],
    traj1: list[np.ndarray] | np.ndarray,
    traj2: list[np.ndarray] | np.ndarray,
    traj3: list[np.ndarray] | np.ndarray,
    traj4: list[np.ndarray] | np.ndarray,
    real_robot: Any,
    viewer: Any,
    dt: float,
    particle_filter: Any,
    env: Any,
    true_mass: float,
    backend: str,
    logger: Any,
    log_data: dict[str, object],
) -> metrics.LiftPhaseResult:
    particle_filter.set_contact_mode(simplified_contacts=True)
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
        metrics.update_warp_memory_metrics(env, stage="phase_1_approach")
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
        metrics.update_warp_memory_metrics(env, stage="phase_2_descend")
        metrics.finish_stage(phase_token)
        logger.info({**log_data, "msg": "Finished phase 2 (descent)."})

    particle_filter.set_contact_mode(simplified_contacts=False)
    particle_filter.warmup_runtime([len(traj3)])
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
        metrics.update_warp_memory_metrics(env, stage="phase_3_grip")
        metrics.finish_stage(phase_token)
        logger.info({**log_data, "msg": "Finished phase 3 (grip)."})

    lift_result = run_phase_4_lift(
        tracer=tracer,
        run_id=run_id,
        started_at=started_at,
        span_attrs=span_attrs,
        backend=backend,
        trajectory=traj4,
        real_robot=real_robot,
        viewer=viewer,
        particle_filter=particle_filter,
        env=env,
        true_mass=true_mass,
        logger=logger,
        log_data=log_data,
    )
    return lift_result


@trace_call("simbay.main", span_name="main")
def main(run_id: str = RUN_ID) -> None:
    tracer = get_tracer("simbay.main")
    log_data: dict[str, object] = {"run_id": run_id}
    started_at = time.perf_counter()

    headless = HEADLESS
    backend = BACKEND
    setup_result = setup(
        backend=backend,
        headless=headless,
        log_data=log_data,
        logger=logger,
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
        logger=logger,
        log_data=log_data,
        span_attrs=span_attrs,
    )

    traj1 = planning_result["traj1"]
    traj2 = planning_result["traj2"]
    traj3 = planning_result["traj3"]
    traj4 = planning_result["traj4"]

    history_ess: list[float] | None = None
    history_ci50_low: list[float] | None = None
    history_ci50_high: list[float] | None = None
    history_ci90_low: list[float] | None = None
    history_ci90_high: list[float] | None = None
    history_resampled: list[bool] | None = None

    if backend == "cpu":
        history_estimates, history_particles = run_simbay_mujoco_original(
            traj1=traj1,
            traj2=traj2,
            traj3=traj3,
            traj4=traj4,
            real_robot=real_robot,
            viewer=viewer,
            particle_filter=particle_filter,
            logger=logger,
            log_data=log_data,
        )
    else:
        lift_result = run_simbay_mujoco_warp(
            tracer=tracer,
            run_id=run_id,
            started_at=started_at,
            span_attrs=span_attrs,
            traj1=traj1,
            traj2=traj2,
            traj3=traj3,
            traj4=traj4,
            real_robot=real_robot,
            viewer=viewer,
            dt=dt,
            particle_filter=particle_filter,
            env=env,
            true_mass=true_mass,
            backend=backend,
            logger=logger,
            log_data=log_data,
        )
        history_estimates = lift_result.history_estimates
        history_particles = lift_result.history_particles
        history_ess = lift_result.history_ess
        history_ci50_low = lift_result.history_ci50_low
        history_ci50_high = lift_result.history_ci50_high
        history_ci90_low = lift_result.history_ci90_low
        history_ci90_high = lift_result.history_ci90_high
        history_resampled = lift_result.history_resampled

    logger.info({**log_data, "msg": "Finished the particle filter run.", "backend": backend})
    final_prediction = float(particle_filter.estimate())
    time_to_prediction_seconds = time.perf_counter() - started_at
    metrics.set_prediction_ready(
        total_wall_seconds=time_to_prediction_seconds,
        final_error_pct=abs(true_mass - final_prediction) * 100,
    )
    logger.info({**log_data, "msg": "The prediction is ready."})

    logger.info({**log_data, "msg": "Started generating the output plots."})
    output_path = generate_particle_filter_plot(
        run_id=run_id,
        history_estimates=history_estimates,
        true_mass=true_mass,
        env=env,
        history_particles=history_particles,
        num_particles=num_particles,
        history_ess=history_ess,
        history_ci50_low=history_ci50_low,
        history_ci50_high=history_ci50_high,
        history_ci90_low=history_ci90_low,
        history_ci90_high=history_ci90_high,
        history_resampled=history_resampled,
    )
    logger.info({**log_data, "msg": f"Saved the particle filter plot to {output_path}.", "path": str(output_path)})

    if viewer is not None:
        with suppress(Exception):
            viewer.close()
    gc.collect()
    logger.info({**log_data, "msg": "Finished the run and shut down cleanly.", "shutdown_requested": shutdown_requested, "signal": shutdown_signal_name or "none"})


if __name__ == "__main__":
    run_id = RUN_ID
    export_observability = OBSERVABILITY_ENABLED
    set_tracing_enabled(export_observability)
    setup_tracing(run_id=run_id)
    LOGGER = simbay_logger
    LOGGER.info(
        {
            "run_id": run_id,
            "msg": "Initialized runtime observability mode.",
            "environment": ENVIRONMENT,
            "observability_enabled": export_observability,
        }
    )
    metrics.init_metrics(run_id=run_id, enabled=export_observability)
    set_span_attributes({"simbay.run_id": run_id})
    install_signal_handlers(LOGGER, {"run_id": run_id})
    try:
        main(run_id=run_id)
    finally:
        metrics.shutdown_metrics()
        force_flush_tracing()
        shutdown_tracing()
