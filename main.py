import atexit
import gc
from datetime import datetime
import logging
import math
import os
from pathlib import Path
import signal
import time
from typing import Any

import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np

from src.estimation.mujoco_particle_filter import FrankaMuJoCoEnv
from src.estimation.particle_filter import ParticleFilter
from src.planning import FrankaSmartSolver
from src.planning import plan_linear_trajectory
from src.utils.constants import DEFAULT_OBJECT_PROPS
from src.utils.constants import FRANKA_HOME_QPOS
from src.utils.logging_utils import extend_logging_data
from src.utils.logging_utils import format_bytes
from src.utils.logging_utils import get_process_memory_bytes
from src.utils.mass_timeseries import ParticleMassTimeseriesCollector
from src.utils.mujoco_utils import initialize_mujoco_env
from src.utils.logging_utils import setup_logging
from src.utils.metrics import create_metrics_from_env
from src.utils.metrics import shutdown_metrics
from src.utils.profiling import annotate
from src.utils.tracing import add_exemplar
from src.utils.tracing import force_flush_tracing
from src.utils.tracing import get_tracer
from src.utils.tracing import set_span_attributes
from src.utils.tracing import setup_tracing
from src.utils.tracing import shutdown_tracing
from src.utils.tracing import span as tracing_span


shutdown_requested = False
shutdown_signal_name: str | None = None
logger = None
base_logging_data: dict[str, object] = {}
_shutdown_done = False


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    if values.size == 0:
        return 0.0
    if values.size != weights.size:
        raise ValueError("values and weights must have the same length")
    quantile = min(max(float(quantile), 0.0), 1.0)
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = np.clip(weights[order], 0.0, None)
    total_weight = float(np.sum(sorted_weights))
    if total_weight <= 0.0:
        return float(np.quantile(sorted_values, quantile))
    cumulative = np.cumsum(sorted_weights) / total_weight
    return float(np.interp(quantile, cumulative, sorted_values))


def _shutdown_runtime() -> None:
    global _shutdown_done

    if _shutdown_done:
        return
    _shutdown_done = True

    run_span_cm = globals().get("_run_span_cm")
    if run_span_cm is not None:
        run_span_cm.__exit__(None, None, None)

    metrics_obj = globals().get("metrics")
    if metrics_obj is not None:
        shutdown_metrics(metrics_obj)

    force_flush_tracing()
    shutdown_tracing()


def _stage_span_attrs(stage: str, *, steps: int | None = None) -> dict[str, str | int | float | bool]:
    attrs: dict[str, str | int | float | bool] = {
        "simbay.run_id": run_id,
        "simbay.stage": stage,
        "simbay.backend": backend,
        "simbay.particles": num_particles,
        "simbay.control_dt": dt,
        "simbay.execution_device": execution_device,
    }
    if steps is not None:
        attrs["simbay.steps"] = steps
    return attrs


def _substage_span_attrs(
    stage: str,
    substage: str,
    *,
    steps: int,
    execution_mode: str,
    parallel_units: int,
) -> dict[str, str | int | float | bool]:
    attrs = _stage_span_attrs(stage, steps=steps)
    attrs["simbay.substage"] = substage
    attrs["simbay.execution_mode"] = execution_mode
    attrs["simbay.parallel_units"] = parallel_units
    return attrs


def _log_setup_summary(
    logging_data: dict[str, object],
    backend_name: str,
    env_memory_profile: dict[str, Any],
    memory_profile: dict[str, Any],
    dt: float,
    true_mass: float,
    num_particles: int,
    cpu_cores: int,
    headless: bool,
) -> None:
    common_data = extend_logging_data(
        logging_data,
        event="simulation_setup",
        msg=(
            f"Completed simulation setup for the {backend_name} backend "
            f"with {num_particles} particles."
        ),
        dt=dt,
        true_mass=true_mass,
        particles=num_particles,
        cpu_cores=cpu_cores,
        headless=headless,
        backend=backend_name,
        state_memory_total_bytes=memory_profile["state_bytes_total"],
        state_memory_total=format_bytes(memory_profile["state_bytes_total"]),
        state_memory_per_particle_bytes=memory_profile["state_bytes_per_particle"],
        state_memory_per_particle=format_bytes(memory_profile["state_bytes_per_particle"]),
        process_memory_per_particle_estimate_bytes=(
            memory_profile["process_memory_per_particle_estimate_bytes"]
        ),
        process_memory_per_particle_estimate=format_bytes(
            memory_profile["process_memory_per_particle_estimate_bytes"]
        ),
    )

    if backend_name == "mujoco-warp":
        logger.info(
            extend_logging_data(
                common_data,
                execution_platform=env_memory_profile["execution_platform"],
                execution_device=env_memory_profile["execution_device"],
                default_jax_platform=env_memory_profile["default_jax_platform"],
                default_jax_device=env_memory_profile["default_jax_device"],
                device_fallback_applied=env_memory_profile["device_fallback_applied"],
                bytes_in_use=env_memory_profile["bytes_in_use"],
                peak_bytes_in_use=env_memory_profile["peak_bytes_in_use"],
                bytes_limit=env_memory_profile["bytes_limit"],
            )
        )
        return

    logger.info(
        extend_logging_data(
            common_data,
            mujoco_model_buffer_per_particle_bytes=(
                env_memory_profile["model_nbuffer_bytes_per_robot"]
            ),
            mujoco_model_buffer_per_particle=format_bytes(
                env_memory_profile["model_nbuffer_bytes_per_robot"]
            ),
            mujoco_data_buffer_per_particle_bytes=(
                env_memory_profile["data_nbuffer_bytes_per_robot"]
            ),
            mujoco_data_buffer_per_particle=format_bytes(
                env_memory_profile["data_nbuffer_bytes_per_robot"]
            ),
            mujoco_data_arena_per_particle_bytes=(
                env_memory_profile["data_narena_bytes_per_robot"]
            ),
            mujoco_data_arena_per_particle=format_bytes(
                env_memory_profile["data_narena_bytes_per_robot"]
            ),
            mujoco_native_memory_per_particle_bytes=env_memory_profile["native_bytes_per_robot"],
            mujoco_native_memory_per_particle=format_bytes(
                env_memory_profile["native_bytes_per_robot"]
            ),
            mujoco_native_memory_total_bytes=env_memory_profile["native_bytes_total"],
            mujoco_native_memory_total=format_bytes(env_memory_profile["native_bytes_total"]),
        )
    )


def _handle_shutdown_signal(signum, _frame) -> None:
    global shutdown_requested
    global shutdown_signal_name

    shutdown_requested = True
    shutdown_signal_name = signal.Signals(signum).name
    if logger is not None:
        logger.info(
            extend_logging_data(
                base_logging_data,
                event="shutdown_requested",
                msg=(
                    f"Received {shutdown_signal_name} and will finish the current run "
                    "before shutting down."
                ),
                signal=shutdown_signal_name,
                mode="graceful",
                action="finish_current_run",
            )
        )
    else:
        logging.getLogger("simbay").info(
            {
                "event": "shutdown_requested",
                "msg": (
                    f"Received {shutdown_signal_name} and will finish the current run "
                    "before shutting down."
                ),
                "run_id": base_logging_data.get("run_id", "unknown"),
                "signal": shutdown_signal_name,
                "mode": "graceful",
                "action": "finish_current_run",
            }
        )


def _log_substage_duration(
    logging_data: dict[str, object],
    phase: str,
    substage: str,
    duration_seconds: float,
    steps: int,
) -> None:
    phase_labels = {
        "phase_1_approach": "phase 1 (approach)",
        "phase_2_descend": "phase 2 (descent)",
        "phase_3_grip": "phase 3 (grip)",
        "phase_4_lift": "phase 4 (lift)",
    }
    substage_labels = {
        "robot_execute": "robot motion",
        "pf_replay": "particle filter replay",
        "pf_update": "particle filter update",
    }
    step_label = "step" if steps == 1 else "steps"
    phase_label = phase_labels.get(phase, phase.replace("_", " "))
    substage_label = substage_labels.get(substage, substage.replace("_", " "))

    logger.info(
        extend_logging_data(
            logging_data,
            event="substage_finished",
            msg=(
                f"Finished {substage_label} for {phase_label} "
                f"in {duration_seconds:.2f} seconds over {steps} {step_label}."
            ),
            phase=phase,
            substage=substage,
            steps=steps,
            duration_ms=duration_seconds * 1000.0,
        )
    )


def _log_stage_started(
    logging_data: dict[str, object],
    stage: str,
    *,
    steps: int | None = None,
) -> None:
    payload = extend_logging_data(
        logging_data,
        event="stage_started",
        msg=f"Started {stage.replace('_', ' ')}.",
        stage=stage,
    )
    if steps is not None:
        payload["steps"] = steps
    logger.info(payload)


def _log_stage_finished(
    logging_data: dict[str, object],
    stage: str,
    duration_seconds: float,
) -> None:
    logger.info(
        extend_logging_data(
            logging_data,
            event="stage_finished",
            msg=(
                f"Finished {stage.replace('_', ' ')} "
                f"in {duration_seconds:.2f} seconds."
            ),
            stage=stage,
            duration_ms=duration_seconds * 1000.0,
        )
    )


def _log_substage_started(
    logging_data: dict[str, object],
    phase: str,
    substage: str,
    *,
    steps: int,
) -> None:
    logger.info(
        extend_logging_data(
            logging_data,
            event="substage_started",
            msg=(
                f"Started {substage.replace('_', ' ')} for {phase.replace('_', ' ')} "
                f"over {steps} {'step' if steps == 1 else 'steps'}."
            ),
            phase=phase,
            substage=substage,
            steps=steps,
        )
    )


def _log_run_metadata(
    logging_data: dict[str, object],
    backend_name: str,
    execution_device: str,
    num_particles: int,
    dt: float,
) -> None:
    logger.info(
        extend_logging_data(
            logging_data,
            event="run_metadata",
            msg=(
                f"Recorded run configuration for the {backend_name} backend with "
                f"{num_particles} particles at dt {dt}."
            ),
            backend=backend_name,
            device=execution_device,
            particles=num_particles,
            control_dt=dt,
        )
    )


signal.signal(signal.SIGINT, _handle_shutdown_signal)
signal.signal(signal.SIGTERM, _handle_shutdown_signal)

# ==========================================
# 1. SETUP
# ==========================================
run_id = os.getenv("SIMBAY_RUN_ID") or datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
setup_tracing(run_id=run_id)
_tracer = get_tracer("simbay.main")
_run_span_cm = tracing_span(_tracer, "simbay.run")
_run_span_cm.__enter__()
set_span_attributes({"simbay.run_id": run_id})
atexit.register(_shutdown_runtime)
base_logging_data = {"run_id": run_id}
run_wall_start = time.perf_counter()
metrics = create_metrics_from_env(run_id=run_id)
metrics.register_stages(
    [
        "setup",
        "ik_planning",
        "phase_1_approach",
        "phase_2_descend",
        "phase_3_grip",
        "phase_4_lift",
        "plot_generation",
    ]
)
metrics.register_substages(
    [
        ("phase_1_approach", "robot_execute"),
        ("phase_1_approach", "pf_replay"),
        ("phase_2_descend", "robot_execute"),
        ("phase_2_descend", "pf_replay"),
        ("phase_3_grip", "robot_execute"),
        ("phase_3_grip", "pf_replay"),
        ("phase_4_lift", "robot_execute"),
        ("phase_4_lift", "pf_update"),
    ]
)
metrics.start()
setup_stage = metrics.start_stage("setup")
logger = setup_logging(run_id=run_id)
_log_stage_started(base_logging_data, "setup")
with tracing_span(_tracer, "setup"):
    headless = os.getenv("SIMBAY_HEADLESS", "true").lower() in {"1", "true", "yes", "on"}
    mass_series_enabled = os.getenv("SIMBAY_MASS_TIMESERIES_ENABLED", "1").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    _backend_env = os.getenv("SIMBAY_BACKEND", "").lower()
    if _backend_env in {"", "cpu"}:
        backend = "cpu"
    elif _backend_env == "mujoco-warp":
        backend = "mujoco-warp"
    else:
        raise SystemExit(
            "Unsupported backend. Set SIMBAY_BACKEND to `cpu` or `mujoco-warp`."
        )

    use_batched_backend = backend == "mujoco-warp"
    # Setup "real" robot
    real_robot = initialize_mujoco_env()
    viewer = None
    if not headless:
        viewer = mujoco.viewer.launch_passive(real_robot.model, real_robot.data)
    real_robot.viewer = viewer
    dt = real_robot.dt

    obj_pos = DEFAULT_OBJECT_PROPS["pos"]
    true_mass = DEFAULT_OBJECT_PROPS["mass"]

    num_particles = int(os.getenv("SIMBAY_PARTICLES", "100"))
    limits = ((0.0, 3.0))
    if backend == "mujoco-warp":
        try:
            from src.estimation.warp_filter import _uniform_weight_metrics
            from src.estimation.warp_filter import _update_and_optionally_resample
            from src.estimation.warp_filter import WarpParticleFilter
            from src.estimation.warp_particle_filter import FrankaWarpEnv
        except ModuleNotFoundError as exc:
            if exc.name in {"mujoco_warp", "warp"}:
                raise SystemExit(
                    "Warp backend requires optional dependencies. Run "
                    "`poetry install --extras warp` or `make install-warp` first."
                ) from exc
            raise

        env = FrankaWarpEnv(limits, num_particles, logging_data=base_logging_data)
        particle_filter = WarpParticleFilter(env, logging_data=base_logging_data)
    elif backend == "cpu":
        env = FrankaMuJoCoEnv(limits, num_particles)
        particle_filter = ParticleFilter(env, logging_data=base_logging_data)
    else:
        raise AssertionError(f"Unexpected backend: {backend}")
    memory_profile = particle_filter.memory_profile()
    env_memory_profile = env.memory_profile()
    cpu_cores = os.cpu_count() or 1
    execution_device = str(env_memory_profile.get("execution_device", backend))
    set_span_attributes(
        {
            "simbay.run_id": run_id,
            "simbay.stage": "setup",
            "simbay.backend": backend,
            "simbay.particles": num_particles,
            "simbay.control_dt": dt,
            "simbay.execution_device": execution_device,
            "simbay.cpu_cores": cpu_cores,
            "simbay.headless": headless,
            "simbay.use_batched_backend": use_batched_backend,
            "simbay.mass_series_enabled": mass_series_enabled,
            "simbay.true_mass": float(true_mass),
        }
    )
    metrics.set_particle_count(num_particles)
    metrics.set_backend(backend, execution_device)
    metrics.set_run_info(backend=backend, particles=num_particles, control_dt=dt)
    if backend == "mujoco-warp":
        metrics.update_warp_memory(
            stage="setup",
            bytes_in_use=int(env_memory_profile["bytes_in_use"]),
            peak_bytes_in_use=int(env_memory_profile["peak_bytes_in_use"]),
            bytes_limit=int(env_memory_profile["bytes_limit"]),
            state_bytes_estimate=int(env_memory_profile.get("state_bytes_estimate", 0)),
        )
    _log_setup_summary(
        base_logging_data,
        backend,
        env_memory_profile,
        memory_profile,
        dt,
        true_mass,
        num_particles,
        cpu_cores,
        headless,
    )
    _log_run_metadata(base_logging_data, backend, execution_device, num_particles, dt)
setup_duration = metrics.finish_stage(setup_stage)
_log_stage_finished(base_logging_data, "setup", setup_duration)

# ==========================================
# 2. TRAJECTORY PLANNING
# ==========================================
planning_stage = metrics.start_stage("ik_planning")
_log_stage_started(base_logging_data, "ik_planning")
with tracing_span(_tracer, "ik_planning"):
    set_span_attributes(_stage_span_attrs("ik_planning"))
    target_quat = np.array([0.0, 1.0, 0.0, 0.0])

    pre_grasp_pos = obj_pos + np.array([0.0, 0.0, 0.15])
    pre_grasp_q7 = FrankaSmartSolver.solve(
        FRANKA_HOME_QPOS, np.concatenate([pre_grasp_pos, target_quat])
    )
    grasp_q7 = FrankaSmartSolver.solve(pre_grasp_q7, np.concatenate([obj_pos, target_quat]))

    lift_pos = obj_pos + np.array([0.0, 0.0, 0.2])
    lift_q7 = FrankaSmartSolver.solve(grasp_q7, np.concatenate([lift_pos, target_quat]))

    OPEN = 255
    CLOSED = 0
    q_home = np.append(FRANKA_HOME_QPOS, OPEN)
    q_pre_grasp = np.append(pre_grasp_q7, OPEN)
    q_grasp_open = np.append(grasp_q7, OPEN)
    q_grasp_closed = np.append(grasp_q7, CLOSED)
    q_lift_closed = np.append(lift_q7, CLOSED)
    traj1 = plan_linear_trajectory(q_home, q_pre_grasp, max_velocity=1.0, dt=dt)
    traj2 = plan_linear_trajectory(q_pre_grasp, q_grasp_open, max_velocity=0.5, dt=dt)
    traj3 = plan_linear_trajectory(
        q_grasp_closed,
        q_grasp_closed,
        max_velocity=500,
        dt=dt,
        settle_time=0.5,
    )
    traj4 = plan_linear_trajectory(
        q_grasp_closed,
        q_lift_closed,
        max_velocity=0.5,
        dt=dt,
        settle_time=1.0,
    )
    set_span_attributes(
        {
            "ik.target_quat_dim": int(target_quat.shape[0]),
            "ik.pre_grasp_height": float(pre_grasp_pos[2]),
            "ik.lift_height": float(lift_pos[2]),
        }
    )
    if use_batched_backend:
        warmed_rollout_lengths = particle_filter.warmup_runtime(
            [
                len(traj1),
                len(traj2),
                len(traj3),
            ]
        )
        logger.info(
            extend_logging_data(
                base_logging_data,
                event="backend_runtime_warmup_summary",
                msg=(
                    f"Finished backend runtime warm-up for the {backend} backend "
                    f"with {particle_filter.N} particles."
                ),
                backend=backend,
                particles=particle_filter.N,
                rollout_lengths=warmed_rollout_lengths,
                phase4_step_warmup=1,
            )
        )
planning_duration = metrics.finish_stage(planning_stage)
_log_stage_finished(base_logging_data, "ik_planning", planning_duration)

# ==========================================
# 3. EXECUTION
# ==========================================

# Phase 1: Move ABOVE the object (No PF updates, just predict to stay synced)
approach_stage = metrics.start_stage("phase_1_approach")
_log_stage_started(base_logging_data, "phase_1_approach", steps=len(traj1))
with tracing_span(_tracer, "phase_1_approach"):
    set_span_attributes(_stage_span_attrs("phase_1_approach", steps=len(traj1)))
    with tracing_span(_tracer, "robot_execute"):
        set_span_attributes(
            _substage_span_attrs(
                "phase_1_approach",
                "robot_execute",
                steps=len(traj1),
                execution_mode="sequential",
                parallel_units=1,
            )
        )
        robot_execute_stage = metrics.start_substage("phase_1_approach", "robot_execute")
        _log_substage_started(
            base_logging_data,
            "phase_1_approach",
            "robot_execute",
            steps=len(traj1),
        )
        for _, qpos in enumerate(traj1):
            real_robot.move_joints(qpos)
            if viewer is not None:
                viewer.sync()
        robot_execute_duration = metrics.finish_substage(robot_execute_stage)
        _log_substage_duration(
            base_logging_data,
            "phase_1_approach",
            "robot_execute",
            robot_execute_duration,
            len(traj1),
        )
        metrics.set_substage_workload(
            "phase_1_approach",
            "robot_execute",
            len(traj1),
            1,
            robot_execute_duration,
        )
    with tracing_span(_tracer, "pf_replay"):
        set_span_attributes(
            _substage_span_attrs(
                "phase_1_approach",
                "pf_replay",
                steps=len(traj1),
                execution_mode="batched_parallel" if use_batched_backend else "sequential",
                parallel_units=particle_filter.N if use_batched_backend else 1,
            )
        )
        pf_replay_stage = metrics.start_substage("phase_1_approach", "pf_replay")
        _log_substage_started(
            base_logging_data,
            "phase_1_approach",
            "pf_replay",
            steps=len(traj1),
        )
        if use_batched_backend:
            particle_filter.predict_trajectory(traj1)
        else:
            for qpos in traj1:
                particle_filter.predict(qpos)
        pf_replay_duration = metrics.finish_substage(pf_replay_stage)
        _log_substage_duration(
            base_logging_data,
            "phase_1_approach",
            "pf_replay",
            pf_replay_duration,
            len(traj1),
        )
        metrics.set_substage_workload(
            "phase_1_approach",
            "pf_replay",
            len(traj1),
            particle_filter.N,
            pf_replay_duration,
        )
    if backend == "mujoco-warp":
        env_memory_profile = env.memory_profile()
        metrics.update_warp_memory(
            stage="phase_1_approach",
            bytes_in_use=int(env_memory_profile["bytes_in_use"]),
            peak_bytes_in_use=int(env_memory_profile["peak_bytes_in_use"]),
            bytes_limit=int(env_memory_profile["bytes_limit"]),
            state_bytes_estimate=int(env_memory_profile.get("state_bytes_estimate", 0)),
        )
approach_duration = metrics.finish_stage(approach_stage)
_log_stage_finished(base_logging_data, "phase_1_approach", approach_duration)


# Phase 2: Descend vertically to the object (No PF updates)
descend_stage = metrics.start_stage("phase_2_descend")
_log_stage_started(base_logging_data, "phase_2_descend", steps=len(traj2))
with tracing_span(_tracer, "phase_2_descend"):
    set_span_attributes(_stage_span_attrs("phase_2_descend", steps=len(traj2)))
    with tracing_span(_tracer, "robot_execute"):
        set_span_attributes(
            _substage_span_attrs(
                "phase_2_descend",
                "robot_execute",
                steps=len(traj2),
                execution_mode="sequential",
                parallel_units=1,
            )
        )
        robot_execute_stage = metrics.start_substage("phase_2_descend", "robot_execute")
        _log_substage_started(
            base_logging_data,
            "phase_2_descend",
            "robot_execute",
            steps=len(traj2),
        )
        for _, qpos in enumerate(traj2):
            real_robot.move_joints(qpos)
            if viewer is not None:
                viewer.sync()
        robot_execute_duration = metrics.finish_substage(robot_execute_stage)
        _log_substage_duration(
            base_logging_data,
            "phase_2_descend",
            "robot_execute",
            robot_execute_duration,
            len(traj2),
        )
        metrics.set_substage_workload(
            "phase_2_descend",
            "robot_execute",
            len(traj2),
            1,
            robot_execute_duration,
        )
    with tracing_span(_tracer, "pf_replay"):
        set_span_attributes(
            _substage_span_attrs(
                "phase_2_descend",
                "pf_replay",
                steps=len(traj2),
                execution_mode="batched_parallel" if use_batched_backend else "sequential",
                parallel_units=particle_filter.N if use_batched_backend else 1,
            )
        )
        pf_replay_stage = metrics.start_substage("phase_2_descend", "pf_replay")
        _log_substage_started(
            base_logging_data,
            "phase_2_descend",
            "pf_replay",
            steps=len(traj2),
        )
        if use_batched_backend:
            particle_filter.predict_trajectory(traj2)
        else:
            for qpos in traj2:
                particle_filter.predict(qpos)
        pf_replay_duration = metrics.finish_substage(pf_replay_stage)
        _log_substage_duration(
            base_logging_data,
            "phase_2_descend",
            "pf_replay",
            pf_replay_duration,
            len(traj2),
        )
        metrics.set_substage_workload(
            "phase_2_descend",
            "pf_replay",
            len(traj2),
            particle_filter.N,
            pf_replay_duration,
        )
    if backend == "mujoco-warp":
        env_memory_profile = env.memory_profile()
        metrics.update_warp_memory(
            stage="phase_2_descend",
            bytes_in_use=int(env_memory_profile["bytes_in_use"]),
            peak_bytes_in_use=int(env_memory_profile["peak_bytes_in_use"]),
            bytes_limit=int(env_memory_profile["bytes_limit"]),
            state_bytes_estimate=int(env_memory_profile.get("state_bytes_estimate", 0)),
        )
descend_duration = metrics.finish_stage(descend_stage)
_log_stage_finished(base_logging_data, "phase_2_descend", descend_duration)


# Phase 3: Close the Gripper (No PF updates)
grip_stage = metrics.start_stage("phase_3_grip")
_log_stage_started(base_logging_data, "phase_3_grip", steps=len(traj3))
with tracing_span(_tracer, "phase_3_grip"):
    set_span_attributes(_stage_span_attrs("phase_3_grip", steps=len(traj3)))
    with tracing_span(_tracer, "robot_execute"):
        set_span_attributes(
            _substage_span_attrs(
                "phase_3_grip",
                "robot_execute",
                steps=len(traj3),
                execution_mode="sequential",
                parallel_units=1,
            )
        )
        robot_execute_stage = metrics.start_substage("phase_3_grip", "robot_execute")
        _log_substage_started(
            base_logging_data,
            "phase_3_grip",
            "robot_execute",
            steps=len(traj3),
        )
        for _, qpos in enumerate(traj3):
            real_robot.move_joints(qpos)
            if viewer is not None:
                viewer.sync()
        robot_execute_duration = metrics.finish_substage(robot_execute_stage)
        _log_substage_duration(
            base_logging_data,
            "phase_3_grip",
            "robot_execute",
            robot_execute_duration,
            len(traj3),
        )
        metrics.set_substage_workload(
            "phase_3_grip",
            "robot_execute",
            len(traj3),
            1,
            robot_execute_duration,
        )
    with tracing_span(_tracer, "pf_replay"):
        set_span_attributes(
            _substage_span_attrs(
                "phase_3_grip",
                "pf_replay",
                steps=len(traj3),
                execution_mode="batched_parallel" if use_batched_backend else "sequential",
                parallel_units=particle_filter.N if use_batched_backend else 1,
            )
        )
        pf_replay_stage = metrics.start_substage("phase_3_grip", "pf_replay")
        _log_substage_started(
            base_logging_data,
            "phase_3_grip",
            "pf_replay",
            steps=len(traj3),
        )
        if use_batched_backend:
            particle_filter.predict_trajectory(traj3)
        else:
            for qpos in traj3:
                particle_filter.predict(qpos)
        pf_replay_duration = metrics.finish_substage(pf_replay_stage)
        _log_substage_duration(
            base_logging_data,
            "phase_3_grip",
            "pf_replay",
            pf_replay_duration,
            len(traj3),
        )
        metrics.set_substage_workload(
            "phase_3_grip",
            "pf_replay",
            len(traj3),
            particle_filter.N,
            pf_replay_duration,
        )
    if backend == "mujoco-warp":
        env_memory_profile = env.memory_profile()
        metrics.update_warp_memory(
            stage="phase_3_grip",
            bytes_in_use=int(env_memory_profile["bytes_in_use"]),
            peak_bytes_in_use=int(env_memory_profile["peak_bytes_in_use"]),
            bytes_limit=int(env_memory_profile["bytes_limit"]),
            state_bytes_estimate=int(env_memory_profile.get("state_bytes_estimate", 0)),
        )
grip_duration = metrics.finish_stage(grip_stage)
_log_stage_finished(base_logging_data, "phase_3_grip", grip_duration)


# Phase 4: Lift straight up (OBJECT IS GRASPED - START TRACKING MASS)
lft_stage = metrics.start_stage("phase_4_lift")
_log_stage_started(base_logging_data, "phase_4_lift", steps=len(traj4))

with tracing_span(_tracer, "phase_4_lift"):
    set_span_attributes(_stage_span_attrs("phase_4_lift", steps=len(traj4)))
    history_particles = []
    history_particle_steps = []
    history_estimates = []
    pf_wall_durations = []
    pf_cpu_durations = []
    phase_4_robot_execute_total = 0.0
    phase_4_pf_update_total = 0.0
    phase_4_resample_count = 0
    mass_series_sample_interval = int(
        os.getenv(
            "SIMBAY_MASS_TIMESERIES_INTERVAL",
            "1" if backend == "mujoco-warp" else "10",
        )
    )
    mass_series_flush_snapshots = int(os.getenv("SIMBAY_MASS_TIMESERIES_FLUSH_SNAPSHOTS", "64"))
    mass_series = ParticleMassTimeseriesCollector(
        run_id=run_id,
        phase="phase_4_lift",
        enabled=mass_series_enabled,
        sample_interval=mass_series_sample_interval,
        flush_snapshots=mass_series_flush_snapshots,
    )
    set_span_attributes(
        {
            "simbay.mass_series_enabled": mass_series_enabled,
            "simbay.mass_series_sample_interval": mass_series_sample_interval,
            "simbay.mass_series_flush_snapshots": mass_series_flush_snapshots,
        }
    )
    latest_particles_snapshot = None
    phase_4_bootstrap_applied = False
    phase_4_invalid_sensor_events = 0
    phase_4_invalid_state_events = 0
    phase_4_skipped_invalid_updates = 0
    phase_4_first_invalid_sensor_step = -1
    phase_4_first_invalid_state_step = -1
    phase_4_max_repaired_world_count = 0
    phase_4_abs_error_sum = 0.0
    phase_4_squared_error_sum = 0.0
    time_to_first_estimate_seconds = -1.0
    convergence_time_to_5pct_seconds = -1.0
    convergence_time_to_10pct_seconds = -1.0

    def _step_span_attrs(step: int) -> dict[str, str | int]:
        return {
            "simbay.run_id": run_id,
            "simbay.step": step,
            "simbay.backend": backend,
            "simbay.stage": "phase_4_lift",
            "simbay.substage": "pf_update",
        }

    def _warp_step(qpos, noisy_ft_reading, step: int, attempt: int = 1) -> dict[str, float | bool]:
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
        ) = _update_and_optionally_resample(
            particle_filter.weights,
            particle_filter.particles,
            likelihoods,
            offset,
        )
        if did_resample:
            particle_filter.env.resample_states(indexes)
            particle_filter._resample_count += 1
        particle_filter._save_last_good_snapshot()
        uniform_weight_l1, uniform_weight_max_dev, collapsed_to_uniform = _uniform_weight_metrics(
            particle_filter.weights
        )
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

    for step, qpos in enumerate(traj4):
        with tracing_span(_tracer, "step"):
            set_span_attributes(
                {
                    "simbay.run_id": run_id,
                    "simbay.stage": "phase_4_lift",
                    "simbay.backend": backend,
                    "simbay.control_dt": dt,
                    "simbay.execution_device": execution_device,
                }
            )
            set_span_attributes(
                {
                    "simbay.step": step,
                    "simbay.execution_mode": (
                        "batched_parallel" if backend == "mujoco-warp" else "sequential"
                    ),
                    "simbay.batched_particle_updates": (
                        particle_filter.N if backend == "mujoco-warp" else 1
                    ),
                }
            )

            with tracing_span(_tracer, "robot_execute"):
                set_span_attributes(
                    {
                        "simbay.run_id": run_id,
                        "simbay.stage": "phase_4_lift",
                        "simbay.substage": "robot_execute",
                        "simbay.backend": backend,
                        "simbay.execution_mode": "sequential",
                        "simbay.parallel_units": 1,
                    }
                )
                set_span_attributes({"simbay.step": step})
                robot_execute_start = time.perf_counter()
                real_robot.move_joints(qpos)
                if viewer is not None:
                    viewer.sync()
                phase_4_robot_execute_total += time.perf_counter() - robot_execute_start

            with tracing_span(_tracer, "pf_update"):
                previous_mass_estimate = float(particle_filter.estimate())
                set_span_attributes(
                    {
                        "simbay.run_id": run_id,
                        "simbay.stage": "phase_4_lift",
                        "simbay.substage": "pf_update",
                        "simbay.backend": backend,
                        "simbay.previous_mass_estimate_kg": previous_mass_estimate,
                        "simbay.execution_mode": (
                            "batched_parallel" if backend == "mujoco-warp" else "sequential"
                        ),
                        "simbay.parallel_units": (
                            particle_filter.N if backend == "mujoco-warp" else 1
                        ),
                    }
                )
                step_wall_start = time.perf_counter()
                step_cpu_start = time.process_time()

                measurements = real_robot.get_sensor_reads()
                real_ft_reading = measurements
                noisy_ft_reading = real_ft_reading + np.random.normal(0, 0.5, size=3)

                if backend == "mujoco-warp":
                    with annotate("phase4_particle_filter_step"):
                        if not phase_4_bootstrap_applied:
                            last_result = None
                            for attempt in range(1, 4):
                                step_result = _warp_step(qpos, noisy_ft_reading, step, attempt=attempt)
                                last_result = step_result
                                if not bool(step_result.get("skipped_invalid_update", False)):
                                    if attempt > 1:
                                        logger.info(
                                            extend_logging_data(
                                                base_logging_data,
                                                event="warp_first_update_recovered",
                                                msg=(
                                                    f"Recovered the first Warp update after "
                                                    f"{attempt} attempts."
                                                ),
                                                attempts=attempt,
                                                step=particle_filter._step_index - 1,
                                            )
                                        )
                                    break
                            step_result = last_result if last_result is not None else _warp_step(qpos, noisy_ft_reading, step)
                            phase_4_bootstrap_applied = True
                        else:
                            step_result = _warp_step(qpos, noisy_ft_reading, step)
                elif backend == "cpu":
                    with annotate("phase4_particle_filter_step"):
                        particle_filter.predict(qpos)
                        particle_filter.update(noisy_ft_reading)
                        did_resample = particle_filter.effective_sample_size() < (particle_filter.N / 2)
                        if did_resample:
                            particle_filter.resample()
                        step_result = {
                            "ess": particle_filter.effective_sample_size(),
                            "resampled": did_resample,
                            "resample_count": getattr(particle_filter, "_resample_count", phase_4_resample_count),
                            "uniform_weight_l1_distance": 0.0,
                            "uniform_weight_max_deviation": 0.0,
                            "collapsed_to_uniform": False,
                        }
                else:
                    particle_filter.predict(qpos)
                    particle_filter.update(noisy_ft_reading)
                    did_resample = particle_filter.effective_sample_size() < (particle_filter.N / 2)
                    if did_resample:
                        particle_filter.resample()
                    else:
                        particle_filter.resample()
                    step_result = {
                        "ess": particle_filter.effective_sample_size(),
                        "resampled": did_resample,
                        "resample_count": phase_4_resample_count + int(did_resample),
                        "uniform_weight_l1_distance": 0.0,
                        "uniform_weight_max_deviation": 0.0,
                        "collapsed_to_uniform": False,
                    }

                set_span_attributes(
                    {
                        "simbay.new_mass_estimate_kg": float(particle_filter.estimate()),
                    }
                )

        particles_snapshot = None
        if backend == "mujoco-warp":
            particles_snapshot = particle_filter.particles.copy()
        elif backend == "cpu":
            particles_snapshot = particle_filter.particles.copy()
        if particles_snapshot is not None:
            latest_particles_snapshot = particles_snapshot
            if mass_series.record(step, particles_snapshot, force=step == len(traj4) - 1):
                history_particle_steps.append(step)
                history_particles.append(particles_snapshot)
        current_estimate = float(particle_filter.estimate())
        history_estimates.append(current_estimate)
        if time_to_first_estimate_seconds < 0.0:
            time_to_first_estimate_seconds = time.perf_counter() - run_wall_start

        abs_error_kg = abs(current_estimate - true_mass)
        rel_error_pct = (abs_error_kg / true_mass * 100.0) if true_mass != 0 else 0.0
        phase_4_abs_error_sum += abs_error_kg
        phase_4_squared_error_sum += abs_error_kg**2
        phase_4_mae_kg = phase_4_abs_error_sum / (step + 1)
        phase_4_rmse_kg = math.sqrt(phase_4_squared_error_sum / (step + 1))
        elapsed_since_run_start = time.perf_counter() - run_wall_start
        if convergence_time_to_10pct_seconds < 0.0 and rel_error_pct <= 10.0:
            convergence_time_to_10pct_seconds = elapsed_since_run_start
        if convergence_time_to_5pct_seconds < 0.0 and rel_error_pct <= 5.0:
            convergence_time_to_5pct_seconds = elapsed_since_run_start

        step_wall_duration = time.perf_counter() - step_wall_start
        step_cpu_duration = time.process_time() - step_cpu_start
        phase_4_pf_update_total += step_wall_duration
        pf_wall_durations.append(step_wall_duration)
        pf_cpu_durations.append(step_cpu_duration)

        cpu_equivalent_cores_used = (
            step_cpu_duration / step_wall_duration if step_wall_duration > 0 else 0.0
        )
        cpu_percent_single_core = cpu_equivalent_cores_used * 100.0
        cpu_percent_total_machine = (
            (cpu_equivalent_cores_used / cpu_cores) * 100.0 if cpu_cores > 0 else 0.0
        )
        step_rate_hz = 1.0 / step_wall_duration if step_wall_duration > 0 else 0.0
        _ = add_exemplar(run_id, step)
        metrics.update_filter_state(
            ess=particle_filter.effective_sample_size(),
            estimate=current_estimate,
            wall_seconds=step_wall_duration,
            cpu_seconds=step_cpu_duration,
            cpu_equivalent_cores=cpu_equivalent_cores_used,
            particles=particle_filter.N,
        )
        phase_4_resample_count = int(step_result.get("resample_count", phase_4_resample_count))
        metrics.update_weight_health(
            uniform_weight_l1_distance=float(step_result.get("uniform_weight_l1_distance", 0.0)),
            uniform_weight_max_deviation=float(step_result.get("uniform_weight_max_deviation", 0.0)),
            collapsed_to_uniform=bool(step_result.get("collapsed_to_uniform", False)),
        )
        metrics.update_accuracy_metrics(
            mass_abs_error_kg=abs_error_kg,
            mass_rel_error_pct=rel_error_pct,
            phase4_mae_kg=phase_4_mae_kg,
            phase4_rmse_kg=phase_4_rmse_kg,
            mass_error_within_1pct=rel_error_pct <= 1.0,
            mass_error_within_5pct=rel_error_pct <= 5.0,
            mass_error_within_10pct=rel_error_pct <= 10.0,
            convergence_time_to_5pct_seconds=convergence_time_to_5pct_seconds,
            convergence_time_to_10pct_seconds=convergence_time_to_10pct_seconds,
            time_to_first_estimate_seconds=time_to_first_estimate_seconds,
        )
        if latest_particles_snapshot is not None:
            weights_snapshot = np.asarray(particle_filter.weights, dtype=np.float64).reshape(-1)
            particle_values = np.asarray(latest_particles_snapshot, dtype=np.float64).reshape(-1)
            particle_weight_sum = float(np.sum(weights_snapshot))
            if particle_weight_sum > 0.0:
                weights_snapshot = weights_snapshot / particle_weight_sum
            ci50_low = _weighted_quantile(particle_values, weights_snapshot, 0.25)
            ci50_high = _weighted_quantile(particle_values, weights_snapshot, 0.75)
            ci90_low = _weighted_quantile(particle_values, weights_snapshot, 0.05)
            ci90_high = _weighted_quantile(particle_values, weights_snapshot, 0.95)
            safe_weights = np.clip(weights_snapshot, np.finfo(np.float64).tiny, 1.0)
            weight_entropy = float(-np.sum(safe_weights * np.log(safe_weights)))
            max_entropy = math.log(len(safe_weights)) if len(safe_weights) > 0 else 0.0
            weight_entropy_normalized = (
                float(weight_entropy / max_entropy) if max_entropy > 0.0 else 0.0
            )
            weight_perplexity = float(np.exp(weight_entropy))
            metrics.update_resample_state(
                steps=step + 1,
                resample_count=phase_4_resample_count,
                resampled=bool(step_result.get("resampled", False)),
                particle_min=float(np.min(latest_particles_snapshot)),
                particle_max=float(np.max(latest_particles_snapshot)),
                particle_mean=float(np.mean(latest_particles_snapshot)),
                particle_std=float(np.std(latest_particles_snapshot)),
                particle_p10=float(np.percentile(latest_particles_snapshot, 10)),
                particle_p50=float(np.percentile(latest_particles_snapshot, 50)),
                particle_p90=float(np.percentile(latest_particles_snapshot, 90)),
            )
            metrics.update_uncertainty_metrics(
                credible_interval_50_width_kg=ci50_high - ci50_low,
                credible_interval_90_width_kg=ci90_high - ci90_low,
                credible_interval_50_contains_truth=ci50_low <= true_mass <= ci50_high,
                credible_interval_90_contains_truth=ci90_low <= true_mass <= ci90_high,
                weight_entropy=weight_entropy,
                weight_entropy_normalized=weight_entropy_normalized,
                weight_perplexity=weight_perplexity,
            )
        if backend == "mujoco-warp":
            diagnostics = step_result.get("diagnostics", {})
            phase_4_invalid_sensor_events = max(
                phase_4_invalid_sensor_events,
                int(diagnostics.get("invalid_sensor_events", 0.0)),
            )
            phase_4_invalid_state_events = max(
                phase_4_invalid_state_events,
                int(diagnostics.get("invalid_state_events", 0.0)),
            )
            phase_4_skipped_invalid_updates = max(
                phase_4_skipped_invalid_updates,
                int(step_result.get("skipped_invalid_updates", 0)),
            )
            current_first_invalid_sensor_step = int(
                diagnostics.get("first_invalid_sensor_step", -1.0)
            )
            current_first_invalid_state_step = int(
                diagnostics.get("first_invalid_state_step", -1.0)
            )
            if phase_4_first_invalid_sensor_step < 0 and current_first_invalid_sensor_step >= 0:
                phase_4_first_invalid_sensor_step = current_first_invalid_sensor_step
            if phase_4_first_invalid_state_step < 0 and current_first_invalid_state_step >= 0:
                phase_4_first_invalid_state_step = current_first_invalid_state_step
            phase_4_max_repaired_world_count = max(
                phase_4_max_repaired_world_count,
                int(diagnostics.get("repaired_world_count", 0.0)),
            )
            metrics.update_likelihood_health(
                sim_force_finite_ratio=float(diagnostics.get("sim_force_finite_ratio", 0.0)),
                diff_finite_ratio=float(diagnostics.get("diff_finite_ratio", 0.0)),
                likelihood_finite_ratio=float(diagnostics.get("likelihood_finite_ratio", 0.0)),
                sim_force_norm_mean=float(diagnostics.get("sim_force_norm_mean", 0.0)),
                diff_norm_mean=float(diagnostics.get("diff_norm_mean", 0.0)),
                likelihood_min=float(diagnostics.get("likelihood_min", 0.0)),
                likelihood_max=float(diagnostics.get("likelihood_max", 0.0)),
                likelihood_mean=float(diagnostics.get("likelihood_mean", 0.0)),
                likelihood_std=float(diagnostics.get("likelihood_std", 0.0)),
            )
            metrics.update_invalid_state_counts(
                invalid_sensor_events=int(diagnostics.get("invalid_sensor_events", 0.0)),
                invalid_state_events=int(diagnostics.get("invalid_state_events", 0.0)),
                skipped_invalid_updates=int(step_result.get("skipped_invalid_updates", 0)),
                skipped_invalid_update=bool(step_result.get("skipped_invalid_update", False)),
                bootstrap_attempts=int(step_result.get("bootstrap_attempts", 1)),
                first_invalid_sensor_step=int(diagnostics.get("first_invalid_sensor_step", -1.0)),
                first_invalid_state_step=int(diagnostics.get("first_invalid_state_step", -1.0)),
                sim_force_nonfinite_count=int(diagnostics.get("sim_force_nonfinite_count", 0.0)),
                diff_nonfinite_count=int(diagnostics.get("diff_nonfinite_count", 0.0)),
                likelihood_nonfinite_count=int(diagnostics.get("likelihood_nonfinite_count", 0.0)),
                qpos_nonfinite_count=int(diagnostics.get("qpos_nonfinite_count", 0.0)),
                qvel_nonfinite_count=int(diagnostics.get("qvel_nonfinite_count", 0.0)),
                sensordata_nonfinite_count=int(diagnostics.get("sensordata_nonfinite_count", 0.0)),
                ctrl_nonfinite_count=int(diagnostics.get("ctrl_nonfinite_count", 0.0)),
            )
            metrics.update_contact_health(
                contact_count_mean=float(diagnostics.get("contact_count_mean", 0.0)),
                contact_count_max=float(diagnostics.get("contact_count_max", 0.0)),
                active_contact_particle_ratio=float(
                    diagnostics.get("active_contact_particle_ratio", 0.0)
                ),
                contact_metric_available=bool(diagnostics.get("contact_metric_available", 0.0)),
                contact_force_mismatch=bool(diagnostics.get("contact_force_mismatch", 0.0)),
                valid_force_particle_ratio=float(
                    diagnostics.get("valid_force_particle_ratio", 0.0)
                ),
                sim_force_signal_particle_ratio=float(
                    diagnostics.get("sim_force_signal_particle_ratio", 0.0)
                ),
            )
        set_span_attributes(
            {
                "simbay.ess": float(particle_filter.effective_sample_size()),
                "simbay.resampled": bool(step_result.get("resampled", False)),
                "simbay.mass_estimate_kg": current_estimate,
                "simbay.step_wall_ms": step_wall_duration * 1000.0,
            }
        )
    if backend == "mujoco-warp":
        env_memory_profile = env.memory_profile()
        metrics.update_warp_memory(
            stage="phase_4_lift",
            bytes_in_use=int(env_memory_profile["bytes_in_use"]),
            peak_bytes_in_use=int(env_memory_profile["peak_bytes_in_use"]),
            bytes_limit=int(env_memory_profile["bytes_limit"]),
            state_bytes_estimate=int(env_memory_profile.get("state_bytes_estimate", 0)),
        )
    metrics.set_substage_duration("phase_4_lift", "robot_execute", phase_4_robot_execute_total)
    metrics.set_substage_duration("phase_4_lift", "pf_update", phase_4_pf_update_total)
    _log_substage_started(
        base_logging_data,
        "phase_4_lift",
        "robot_execute",
        steps=len(traj4),
    )
    _log_substage_duration(
        base_logging_data,
        "phase_4_lift",
        "robot_execute",
        phase_4_robot_execute_total,
        len(traj4),
    )
    _log_substage_started(
        base_logging_data,
        "phase_4_lift",
        "pf_update",
        steps=len(traj4),
    )
    _log_substage_duration(
        base_logging_data,
        "phase_4_lift",
        "pf_update",
        phase_4_pf_update_total,
        len(traj4),
    )
    metrics.set_substage_workload(
        "phase_4_lift",
        "robot_execute",
        len(traj4),
        1,
        phase_4_robot_execute_total,
    )
    metrics.set_substage_workload(
        "phase_4_lift",
        "pf_update",
        len(traj4),
        particle_filter.N,
        phase_4_pf_update_total,
    )
lft_duration = metrics.finish_stage(lft_stage)
_log_stage_finished(base_logging_data, "phase_4_lift", lft_duration)
force_flush_tracing()

avg_wall_duration = sum(pf_wall_durations) / len(pf_wall_durations) if pf_wall_durations else 0.0
avg_cpu_duration = sum(pf_cpu_durations) / len(pf_cpu_durations) if pf_cpu_durations else 0.0
avg_step_rate_hz = 1.0 / avg_wall_duration if avg_wall_duration > 0 else 0.0
avg_cpu_equivalent_cores = (
    avg_cpu_duration / avg_wall_duration if avg_wall_duration > 0 else 0.0
)
if backend == "cpu":
    env_memory_profile = env.memory_profile()
    metrics.update_warp_memory(
        stage="phase_4_lift",
        bytes_in_use=int(env_memory_profile["bytes_in_use"]),
        peak_bytes_in_use=int(env_memory_profile["peak_bytes_in_use"]),
        bytes_limit=int(env_memory_profile["bytes_limit"]),
        state_bytes_estimate=int(env_memory_profile.get("state_bytes_estimate", 0)),
    )
    logger.info(
        extend_logging_data(
            base_logging_data,
            event="particle_filter_summary",
            msg=(
                f"Finished the particle filter run over {len(pf_wall_durations)} steps "
                f"with a final estimate of {float(particle_filter.estimate()):.3f} kg."
            ),
            steps=len(pf_wall_durations),
            avg_wall_ms=avg_wall_duration * 1000.0,
            avg_step_rate_hz=avg_step_rate_hz,
            avg_cpu_ms=avg_cpu_duration * 1000.0,
            avg_cpu_equivalent_cores=avg_cpu_equivalent_cores,
            final_estimate=float(particle_filter.estimate()),
            final_error_pct=abs(true_mass - particle_filter.estimate()) * 100,
            final_rss_bytes=get_process_memory_bytes(),
            final_rss=format_bytes(get_process_memory_bytes()),
            backend="cpu",
            mujoco_model_buffer_per_particle_bytes=(
                env_memory_profile["model_nbuffer_bytes_per_robot"]
            ),
            mujoco_model_buffer_per_particle=format_bytes(
                env_memory_profile["model_nbuffer_bytes_per_robot"]
            ),
            mujoco_data_buffer_per_particle_bytes=(
                env_memory_profile["data_nbuffer_bytes_per_robot"]
            ),
            mujoco_data_buffer_per_particle=format_bytes(
                env_memory_profile["data_nbuffer_bytes_per_robot"]
            ),
            mujoco_data_arena_per_particle_bytes=(
                env_memory_profile["data_narena_bytes_per_robot"]
            ),
            mujoco_data_arena_per_particle=format_bytes(
                env_memory_profile["data_narena_bytes_per_robot"]
            ),
            mujoco_native_memory_per_particle_bytes=env_memory_profile["native_bytes_per_robot"],
            mujoco_native_memory_per_particle=format_bytes(
                env_memory_profile["native_bytes_per_robot"]
            ),
            mujoco_native_memory_total_bytes=env_memory_profile["native_bytes_total"],
            mujoco_native_memory_total=format_bytes(env_memory_profile["native_bytes_total"]),
        )
    )
elif backend == "mujoco-warp":
    env_memory_profile = env.memory_profile()
    logger.info(
        extend_logging_data(
            base_logging_data,
            event="particle_filter_summary",
            msg=(
                f"Finished the particle filter run over {len(pf_wall_durations)} steps "
                f"with a final estimate of {float(particle_filter.estimate()):.3f} kg."
            ),
            steps=len(pf_wall_durations),
            avg_wall_ms=avg_wall_duration * 1000.0,
            avg_step_rate_hz=avg_step_rate_hz,
            avg_cpu_ms=avg_cpu_duration * 1000.0,
            avg_cpu_equivalent_cores=avg_cpu_equivalent_cores,
            final_estimate=float(particle_filter.estimate()),
            final_error_pct=abs(true_mass - particle_filter.estimate()) * 100,
            final_rss_bytes=get_process_memory_bytes(),
            final_rss=format_bytes(get_process_memory_bytes()),
            backend="mujoco-warp",
            execution_platform=env_memory_profile["execution_platform"],
            execution_device=env_memory_profile["execution_device"],
            invalid_sensor_events=phase_4_invalid_sensor_events,
            invalid_state_events=phase_4_invalid_state_events,
            skipped_invalid_updates=phase_4_skipped_invalid_updates,
            first_invalid_sensor_step=phase_4_first_invalid_sensor_step,
            first_invalid_state_step=phase_4_first_invalid_state_step,
            max_repaired_world_count=phase_4_max_repaired_world_count,
        )
    )
else:
    logger.info(
        extend_logging_data(
            base_logging_data,
            event="particle_filter_summary",
            msg=(
                f"Finished the particle filter run over {len(pf_wall_durations)} steps "
                f"with a final estimate of {float(particle_filter.estimate()):.3f} kg."
            ),
            steps=len(pf_wall_durations),
            avg_wall_ms=avg_wall_duration * 1000.0,
            avg_step_rate_hz=avg_step_rate_hz,
            avg_cpu_ms=avg_cpu_duration * 1000.0,
            avg_cpu_equivalent_cores=avg_cpu_equivalent_cores,
            final_estimate=float(particle_filter.estimate()),
            final_error_pct=abs(true_mass - particle_filter.estimate()) * 100,
            final_rss_bytes=get_process_memory_bytes(),
            final_rss=format_bytes(get_process_memory_bytes()),
            mujoco_native_memory_total_bytes=env_memory_profile["native_bytes_total"],
            mujoco_native_memory_total=format_bytes(env_memory_profile["native_bytes_total"]),
        )
    )

logger.info(
    extend_logging_data(
        base_logging_data,
        event="sequence_complete",
        msg="Finished the execution sequence.",
        awaiting_user_input=not headless,
    )
)
if not headless and not shutdown_requested:
    input()
elif not headless and shutdown_requested:
    logger.info(
        extend_logging_data(
            base_logging_data,
            event="sequence_complete_skipping_user_input",
            msg="Skipped waiting for user input because shutdown was requested.",
            signal=shutdown_signal_name,
        )
    )
final_prediction = float(particle_filter.estimate())
time_to_prediction_seconds = time.perf_counter() - run_wall_start
metrics.set_prediction_ready(
    total_wall_seconds=time_to_prediction_seconds,
    final_error_pct=abs(true_mass - final_prediction) * 100,
)
logger.info(
    extend_logging_data(
        base_logging_data,
        event="prediction_ready",
        msg=(
            f"The prediction is ready after {time_to_prediction_seconds:.2f} seconds."
        ),
        total_wall_s=time_to_prediction_seconds,
        final_mass_prediction_kg=final_prediction,
    )
)
logger.info(
    extend_logging_data(
        base_logging_data,
        event="final_mass_prediction",
        msg=f"Recorded the final mass prediction as {final_prediction:.3f} kg.",
        final_mass_prediction_kg=final_prediction,
    )
)
logger.info(
    extend_logging_data(
        base_logging_data,
        event="final_error",
        msg=(
            "Recorded the final percentage error for the mass prediction."
        ),
        final_error_pct=abs(true_mass - final_prediction) * 100,
    )
)
history_particle_steps_array, history_particles, mass_series_artifacts = mass_series.finalize()
history_particle_steps = history_particle_steps_array.tolist()
if mass_series_artifacts:
    logger.info(
        extend_logging_data(
            base_logging_data,
            event="mass_timeseries_export_complete",
            msg=(
                f"Exported {len(history_particles)} particle-mass snapshots to "
                f"{mass_series_artifacts[0].parent}."
            ),
            snapshots=len(history_particles),
            artifacts=len(mass_series_artifacts),
            path=str(mass_series_artifacts[0].parent),
            sample_interval=mass_series_sample_interval,
        )
    )
# You should also print the real mass here to see if the filter got it right!


# ==========================================
# 4. GRAPH GENERATION
# ==========================================
plot_stage = metrics.start_stage("plot_generation")
_log_stage_started(base_logging_data, "plot_generation")
logger.info(
    extend_logging_data(
        base_logging_data,
        event="plot_generation_start",
        msg="Started generating the output plots.",
    )
)
with tracing_span(_tracer, "plot_generation"):
    set_span_attributes(
        {
            **_stage_span_attrs("plot_generation", steps=len(history_estimates)),
            "history_particle_snapshots": len(history_particles),
            "history_estimates": len(history_estimates),
            "true_mass": float(true_mass),
        }
    )
    plt.figure(figsize=(10, 6))

    for step_index, particle_masses in zip(history_particle_steps, history_particles):
        plt.scatter([step_index] * len(particle_masses), particle_masses, color="blue", alpha=0.05, s=15)

    plt.plot(
        range(len(history_estimates)),
        history_estimates,
        color="red",
        linewidth=3,
        label="Filter Estimate (Mean)",
    )
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
    logger.info(
        extend_logging_data(
            base_logging_data,
            event="plot_saved",
            msg=f"Saved the particle filter plot to {output_path}.",
            path=str(output_path),
        )
    )
plot_duration = metrics.finish_stage(plot_stage)
_log_stage_finished(base_logging_data, "plot_generation", plot_duration)

gc.collect()

logger.info(
    extend_logging_data(
        base_logging_data,
        event="goodbye",
        msg="Finished the run and shut down cleanly.",
        shutdown_requested=shutdown_requested,
        signal=shutdown_signal_name or "none",
    )
)
_shutdown_runtime()
