import gc
import logging
import os
from pathlib import Path
import signal
import time
from typing import Any
from uuid import uuid4

# Apply JAX/XLA settings before importing modules that may load JAX.
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"

import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np

from src.estimation.mujoco_particle_filter import FrankaMuJoCoEnv
from src.estimation.particle_filter import ParticleFilter
from src.planning import FrankaSmartSolver
from src.planning import plan_linear_trajectory
from src.utils import DEFAULT_OBJECT_PROPS
from src.utils import FRANKA_HOME_QPOS
from src.utils import initialize_mujoco_env
from src.utils.logging_utils import extend_logging_data
from src.utils.logging_utils import format_bytes
from src.utils.logging_utils import get_process_memory_bytes
from src.utils.mass_timeseries import ParticleMassTimeseriesCollector
from src.utils.logging_utils import setup_logging
from src.utils.metrics import create_metrics_from_env
from src.utils.metrics import shutdown_metrics


shutdown_requested = False
shutdown_signal_name: str | None = None
logger = None
base_logging_data: dict[str, object] = {}


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

    if backend_name in {"cpu", "warp"}:
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
                signal=shutdown_signal_name,
                mode="graceful",
                action="finish_current_run",
            )
        )
    else:
        logging.getLogger("simbay").info(
            {
                "event": "shutdown_requested",
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
    logger.info(
        extend_logging_data(
            logging_data,
            event="phase_substage_duration",
            phase=phase,
            substage=substage,
            steps=steps,
            duration_ms=duration_seconds * 1000.0,
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
run_id = os.getenv("SIMBAY_RUN_ID", str(uuid4()))
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
headless = os.getenv("SIMBAY_HEADLESS", "true").lower() in {"1", "true", "yes", "on"}
use_mjx = os.getenv("SIMBAY_USE_MJX", "true").lower() in {"1", "true", "yes", "on"}
mass_series_enabled = os.getenv("SIMBAY_MASS_TIMESERIES_ENABLED", "1").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
_backend_env = os.getenv("SIMBAY_BACKEND", "").lower()
if _backend_env in {"cpu", "warp"}:
    backend = _backend_env
elif use_mjx:
    backend = "cpu"
else:
    backend = "mujoco"

use_batched_backend = backend in {"cpu", "warp"}

if backend == "cpu":
    import jax

    from src.estimation.mjx_filter import MJXParticleFilter
    from src.estimation.mjx_particle_filter import FrankaMJXEnv

    try:
        device = jax.devices("gpu")[0]
        memory_stats = device.memory_stats() or {}
        logger.info(
            extend_logging_data(
                base_logging_data,
                event="gpu_memory_stats",
                device=str(device),
                stats=memory_stats,
            )
        )
    except Exception as exc:
        logger.warning(
            extend_logging_data(
                base_logging_data,
                event="gpu_memory_stats_unavailable",
                error=str(exc),
            )
        )

# Setup "real" robot
real_robot = initialize_mujoco_env()
viewer = None
if not headless:
    viewer = mujoco.viewer.launch_passive(real_robot.model, real_robot.data)
real_robot.viewer = viewer
dt = real_robot.dt

obj_pos = DEFAULT_OBJECT_PROPS['pos']
true_mass = DEFAULT_OBJECT_PROPS['mass']

# Initiate Particle Filter
num_particles = int(os.getenv("SIMBAY_PARTICLES", "100"))
limits = ((0.0, 3.0))
if backend == "warp":
    try:
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
    env = FrankaMJXEnv(limits, num_particles, logging_data=base_logging_data)
    particle_filter = MJXParticleFilter(env, logging_data=base_logging_data)
else:
    env = FrankaMuJoCoEnv(limits, num_particles)
    particle_filter = ParticleFilter(env, logging_data=base_logging_data)
memory_profile = particle_filter.memory_profile()
env_memory_profile = env.memory_profile()
cpu_cores = os.cpu_count() or 1
execution_device = str(env_memory_profile.get("execution_device", backend))
metrics.set_particle_count(num_particles)
metrics.set_backend(backend, execution_device)
metrics.set_run_info(backend=backend, particles=num_particles, control_dt=dt)
if backend == "warp":
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
metrics.finish_stage(setup_stage)

# ==========================================
# 2. TRAJECTORY PLANNING
# ==========================================
planning_stage = metrics.start_stage("ik_planning")
target_quat = np.array([0.0, 1.0, 0.0, 0.0])

pre_grasp_pos = obj_pos + np.array([0.0, 0.0, 0.15])
pre_grasp_q7 = FrankaSmartSolver.solve(FRANKA_HOME_QPOS, np.concatenate([pre_grasp_pos, target_quat]))

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
            backend=backend,
            particles=particle_filter.N,
            rollout_lengths=warmed_rollout_lengths,
            phase4_step_warmup=1,
        )
    )
metrics.finish_stage(planning_stage)

# ==========================================
# 3. EXECUTION
# ==========================================

# Phase 1: Move ABOVE the object (No PF updates, just predict to stay synced)
approach_stage = metrics.start_stage("phase_1_approach")
logger.info(
    extend_logging_data(
        base_logging_data,
        event="phase_start",
        phase="approach",
        steps=len(traj1),
    )
)
robot_execute_stage = metrics.start_substage("phase_1_approach", "robot_execute")
for i, qpos in enumerate(traj1):
    real_robot.move_joints(qpos)
    if viewer is not None:
        viewer.sync()
    if (i + 1) % 100 == 0 or i == len(traj1) - 1:
        logger.info(
            extend_logging_data(
                base_logging_data,
                event="phase_progress",
                phase="approach",
                step=i + 1,
                total_steps=len(traj1),
            )
        )
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
pf_replay_stage = metrics.start_substage("phase_1_approach", "pf_replay")
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
if backend == "warp":
    env_memory_profile = env.memory_profile()
    metrics.update_warp_memory(
        stage="phase_1_approach",
        bytes_in_use=int(env_memory_profile["bytes_in_use"]),
        peak_bytes_in_use=int(env_memory_profile["peak_bytes_in_use"]),
        bytes_limit=int(env_memory_profile["bytes_limit"]),
        state_bytes_estimate=int(env_memory_profile.get("state_bytes_estimate", 0)),
    )
metrics.finish_stage(approach_stage)


# Phase 2: Descend vertically to the object (No PF updates)
descend_stage = metrics.start_stage("phase_2_descend")
logger.info(
    extend_logging_data(
        base_logging_data,
        event="phase_start",
        phase="descend",
        steps=len(traj2),
    )
)
robot_execute_stage = metrics.start_substage("phase_2_descend", "robot_execute")
for i, qpos in enumerate(traj2):
    real_robot.move_joints(qpos)
    if viewer is not None:
        viewer.sync()
    if (i + 1) % 100 == 0 or i == len(traj2) - 1:
        logger.info(
            extend_logging_data(
                base_logging_data,
                event="phase_progress",
                phase="descend",
                step=i + 1,
                total_steps=len(traj2),
            )
        )
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
pf_replay_stage = metrics.start_substage("phase_2_descend", "pf_replay")
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
if backend == "warp":
    env_memory_profile = env.memory_profile()
    metrics.update_warp_memory(
        stage="phase_2_descend",
        bytes_in_use=int(env_memory_profile["bytes_in_use"]),
        peak_bytes_in_use=int(env_memory_profile["peak_bytes_in_use"]),
        bytes_limit=int(env_memory_profile["bytes_limit"]),
        state_bytes_estimate=int(env_memory_profile.get("state_bytes_estimate", 0)),
    )
metrics.finish_stage(descend_stage)


# Phase 3: Close the Gripper (No PF updates)
grip_stage = metrics.start_stage("phase_3_grip")
logger.info(
    extend_logging_data(
        base_logging_data,
        event="phase_start",
        phase="close_gripper",
        steps=len(traj3),
    )
)
robot_execute_stage = metrics.start_substage("phase_3_grip", "robot_execute")
for i, qpos in enumerate(traj3):
    real_robot.move_joints(qpos)
    if viewer is not None:
        viewer.sync()
    if (i + 1) % 100 == 0 or i == len(traj3) - 1:
        logger.info(
            extend_logging_data(
                base_logging_data,
                event="phase_progress",
                phase="close_gripper",
                step=i + 1,
                total_steps=len(traj3),
            )
        )
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
pf_replay_stage = metrics.start_substage("phase_3_grip", "pf_replay")
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
if backend == "warp":
    env_memory_profile = env.memory_profile()
    metrics.update_warp_memory(
        stage="phase_3_grip",
        bytes_in_use=int(env_memory_profile["bytes_in_use"]),
        peak_bytes_in_use=int(env_memory_profile["peak_bytes_in_use"]),
        bytes_limit=int(env_memory_profile["bytes_limit"]),
        state_bytes_estimate=int(env_memory_profile.get("state_bytes_estimate", 0)),
    )
metrics.finish_stage(grip_stage)


# Phase 4: Lift straight up (OBJECT IS GRASPED - START TRACKING MASS)
lft_stage = metrics.start_stage("phase_4_lift")
logger.info(
    extend_logging_data(
        base_logging_data,
        event="phase_start",
        phase="lift_and_estimate",
    )
)

# <--- Initialize lists to hold the historical data for the graph --->
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
        "1" if backend in {"warp", "mujoco"} else "10",
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
latest_particles_snapshot = None
phase_4_bootstrap_applied = False

for step, qpos in enumerate(traj4):
    robot_execute_start = time.perf_counter()
    real_robot.move_joints(qpos)
    if viewer is not None:
        viewer.sync()
    phase_4_robot_execute_total += time.perf_counter() - robot_execute_start

    step_wall_start = time.perf_counter()
    step_cpu_start = time.process_time()

    # 1. Get the real measurement
    measurements = real_robot.get_sensor_reads()
    real_ft_reading = measurements
    
    # 2. INJECT NOISE to simulate physical hardware (e.g., 0.5N of sensor noise)
    noisy_ft_reading = real_ft_reading + np.random.normal(0, 0.5, size=3)

    # 3. Update beliefs based on the noisy real reading and resample
    if use_batched_backend:
        if backend == "warp" and not phase_4_bootstrap_applied:
            step_result = particle_filter.bootstrap_first_update(
                qpos,
                noisy_ft_reading,
                max_attempts=3,
            )
            phase_4_bootstrap_applied = True
        else:
            step_result = particle_filter.step(qpos, noisy_ft_reading)
    else:
        particle_filter.predict(qpos)
        particle_filter.update(noisy_ft_reading)
        particle_filter.resample()
        step_result = {
            "ess": particle_filter.effective_sample_size(),
            "resampled": False,
            "resample_count": phase_4_resample_count,
        }

    # <--- Save the state of the particles at this exact timestep --->
    particles_snapshot = None
    if backend == "warp":
        particles_snapshot = particle_filter.particles.copy()
    elif backend == "mujoco":
        particles_snapshot = particle_filter.particles.copy()
    elif mass_series.should_record(step, force=step == len(traj4) - 1):
        particles_snapshot = particle_filter.particles_host().copy()
    if particles_snapshot is not None:
        latest_particles_snapshot = particles_snapshot
        if mass_series.record(step, particles_snapshot, force=step == len(traj4) - 1):
            history_particle_steps.append(step)
            history_particles.append(particles_snapshot)
    current_estimate = float(particle_filter.estimate())
    history_estimates.append(current_estimate)

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
    if latest_particles_snapshot is not None:
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
    if backend == "warp":
        diagnostics = step_result.get("diagnostics", {})
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
    should_log_step = step in (0, len(traj4) - 1) or (step + 1) % 10 == 0
    if should_log_step:
        rss_bytes = get_process_memory_bytes()
        if backend == "warp":
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
                    event="particle_filter_step",
                    step=step,
                    particles=particle_filter.N,
                    wall_ms=step_wall_duration * 1000.0,
                    step_rate_hz=step_rate_hz,
                    cpu_ms=step_cpu_duration * 1000.0,
                    cpu_equivalent_cores=cpu_equivalent_cores_used,
                    cpu_percent_single_core=cpu_percent_single_core,
                    cpu_percent_total_machine=cpu_percent_total_machine,
                    ess=particle_filter.effective_sample_size(),
                    estimate=current_estimate,
                    rss_bytes=rss_bytes,
                    rss=format_bytes(rss_bytes),
                    backend="warp",
                    execution_platform=env_memory_profile["execution_platform"],
                    execution_device=env_memory_profile["execution_device"],
                )
            )
        else:
            logger.info(
                extend_logging_data(
                    base_logging_data,
                    event="particle_filter_step",
                    step=step,
                    particles=particle_filter.N,
                    wall_ms=step_wall_duration * 1000.0,
                    step_rate_hz=step_rate_hz,
                    cpu_ms=step_cpu_duration * 1000.0,
                    cpu_equivalent_cores=cpu_equivalent_cores_used,
                    cpu_percent_single_core=cpu_percent_single_core,
                    cpu_percent_total_machine=cpu_percent_total_machine,
                    ess=particle_filter.effective_sample_size(),
                    estimate=current_estimate,
                    rss_bytes=rss_bytes,
                    rss=format_bytes(rss_bytes),
                    mujoco_native_memory_total_bytes=env_memory_profile["native_bytes_total"],
                    mujoco_native_memory_total=format_bytes(
                        env_memory_profile["native_bytes_total"]
                    ),
                )
            )
metrics.set_substage_duration("phase_4_lift", "robot_execute", phase_4_robot_execute_total)
metrics.set_substage_duration("phase_4_lift", "pf_update", phase_4_pf_update_total)
_log_substage_duration(
    base_logging_data,
    "phase_4_lift",
    "robot_execute",
    phase_4_robot_execute_total,
    len(traj4),
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
metrics.finish_stage(lft_stage)

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
            execution_platform=env_memory_profile["execution_platform"],
            execution_device=env_memory_profile["execution_device"],
            default_jax_platform=env_memory_profile["default_jax_platform"],
            default_jax_device=env_memory_profile["default_jax_device"],
            device_fallback_applied=env_memory_profile["device_fallback_applied"],
        )
    )
elif backend == "warp":
    env_memory_profile = env.memory_profile()
    logger.info(
        extend_logging_data(
            base_logging_data,
            event="particle_filter_summary",
            steps=len(pf_wall_durations),
            avg_wall_ms=avg_wall_duration * 1000.0,
            avg_step_rate_hz=avg_step_rate_hz,
            avg_cpu_ms=avg_cpu_duration * 1000.0,
            avg_cpu_equivalent_cores=avg_cpu_equivalent_cores,
            final_estimate=float(particle_filter.estimate()),
            final_error_pct=abs(true_mass - particle_filter.estimate()) * 100,
            final_rss_bytes=get_process_memory_bytes(),
            final_rss=format_bytes(get_process_memory_bytes()),
            backend="warp",
            execution_platform=env_memory_profile["execution_platform"],
            execution_device=env_memory_profile["execution_device"],
        )
    )
else:
    logger.info(
        extend_logging_data(
            base_logging_data,
            event="particle_filter_summary",
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
        total_wall_s=time_to_prediction_seconds,
        final_mass_prediction_kg=final_prediction,
    )
)
logger.info(
    extend_logging_data(
        base_logging_data,
        event="final_mass_prediction",
        final_mass_prediction_kg=final_prediction,
    )
)
logger.info(
    extend_logging_data(
        base_logging_data,
        event="final_error",
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
logger.info(extend_logging_data(base_logging_data, event="plot_generation_start"))


plt.figure(figsize=(10, 6))

# Plot the 100 particles over time as highly transparent blue dots
for step_index, particle_masses in zip(history_particle_steps, history_particles):
    plt.scatter([step_index] * len(particle_masses), particle_masses, color='blue', alpha=0.05, s=15)

# Plot the Filter's official guess (Mean)
plt.plot(range(num_steps), history_estimates, color='red', linewidth=3, label='Filter Estimate (Mean)')

# Plot the True Mass
plt.axhline(y=true_mass, color='green', linestyle='--', linewidth=2, label=f'True Mass ({true_mass} kg)')

# Formatting
plt.title('Particle Filter: Mass Estimation Evolution', fontsize=14, fontweight='bold')
plt.xlabel('Simulation Step (Lifting Phase)', fontsize=12)
plt.ylabel('Estimated Mass (kg)', fontsize=12)
plt.ylim(env.min, env.max) # Lock the Y-axis to physical limits
plt.legend(loc='upper right')
plt.grid(True, linestyle=':', alpha=0.7)

output_dir = Path("temp")
output_dir.mkdir(exist_ok=True)
output_path = output_dir / "particle_filter_evolution.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
plt.close()
logger.info(
    extend_logging_data(
        base_logging_data,
        event="plot_saved",
        path=str(output_path),
    )
)
metrics.finish_stage(plot_stage)

if backend == "cpu":
    jax.clear_caches()
    gc.collect()
    logger.info(extend_logging_data(base_logging_data, event="jax_cleanup_complete"))

logger.info(
    extend_logging_data(
        base_logging_data,
        event="goodbye",
        shutdown_requested=shutdown_requested,
        signal=shutdown_signal_name or "none",
    )
)
shutdown_metrics(metrics)
