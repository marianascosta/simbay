import gc
import os
from pathlib import Path
import signal
import time

# Apply JAX/XLA settings before importing modules that may load JAX.
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"

import matplotlib.pyplot as plt
import jax
import mujoco
import mujoco.viewer
import numpy as np

from src.estimation import FrankaMJXEnv
from src.estimation import FrankaMuJoCoEnv
from src.estimation import MJXParticleFilter
from src.estimation import ParticleFilter
from src.planning import FrankaSmartSolver
from src.planning import plan_linear_trajectory
from src.utils import DEFAULT_OBJECT_PROPS
from src.utils import FRANKA_HOME_QPOS
from src.utils import initialize_mujoco_env
from src.utils.logging_utils import format_bytes
from src.utils.logging_utils import get_process_memory_bytes
from src.utils.logging_utils import setup_logging
from src.utils.metrics import create_metrics_from_env
from src.utils.metrics import shutdown_metrics


shutdown_requested = False
shutdown_signal_name: str | None = None
logger = None


def _handle_shutdown_signal(signum, _frame) -> None:
    global shutdown_requested
    global shutdown_signal_name

    shutdown_requested = True
    shutdown_signal_name = signal.Signals(signum).name
    message = (
        f"shutdown_requested signal={shutdown_signal_name} "
        "mode=graceful action=finish_current_run"
    )
    if logger is not None:
        logger.info(message)
    else:
        print(message)


def _log_substage_duration(
    phase: str,
    substage: str,
    duration_seconds: float,
    steps: int,
) -> None:
    logger.info(
        "phase_substage_duration phase=%s substage=%s steps=%d duration_ms=%.3f",
        phase,
        substage,
        steps,
        duration_seconds * 1000.0,
    )


def _env_enabled(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).lower() in {"1", "true", "yes", "on"}


def _create_particle_filter(use_mjx: bool, limits, num_particles: int):
    if use_mjx:
        env = FrankaMJXEnv(limits, num_particles)
        particle_filter = MJXParticleFilter(env)
    else:
        env = FrankaMuJoCoEnv(limits, num_particles)
        particle_filter = ParticleFilter(env)
    return env, particle_filter


def _parse_replay_chunk_size() -> int:
    raw_value = os.getenv("SIMBAY_REPLAY_CHUNK_SIZE", "0")
    chunk_size = int(raw_value)
    allowed_chunk_sizes = {0, 32, 64}
    if chunk_size not in allowed_chunk_sizes:
        raise ValueError(
            f"SIMBAY_REPLAY_CHUNK_SIZE must be one of {sorted(allowed_chunk_sizes)}, got {chunk_size}"
        )
    return chunk_size


def _replay_trajectory(particle_filter, phase: str, trajectory: np.ndarray, replay_chunk_size: int) -> None:
    if isinstance(particle_filter, MJXParticleFilter) and replay_chunk_size > 0:
        particle_filter.replay_chunked(trajectory, replay_chunk_size, phase=phase)
        return

    for qpos in trajectory:
        if isinstance(particle_filter, MJXParticleFilter):
            particle_filter.predict(qpos, phase=phase)
        else:
            particle_filter.predict(qpos)


def _run_reference_replay(
    particle_filter,
    trajectories: list[tuple[str, np.ndarray]],
    *,
    synchronize: bool,
) -> dict[str, dict[str, float | int]]:
    results: dict[str, dict[str, float | int]] = {}
    total_steps = 0
    total_wall_seconds = 0.0

    if hasattr(particle_filter, "reset_replay_profile"):
        particle_filter.reset_replay_profile()

    for phase, trajectory in trajectories:
        phase_start = time.perf_counter()
        for step_index, qpos in enumerate(trajectory):
            if isinstance(particle_filter, MJXParticleFilter):
                particle_filter.predict(qpos, synchronize=synchronize, phase=phase)
                particle_filter.log_replay_profile_summary(phase, step_index, len(trajectory))
            else:
                particle_filter.predict(qpos)
        phase_wall_seconds = time.perf_counter() - phase_start
        phase_steps = len(trajectory)
        total_steps += phase_steps
        total_wall_seconds += phase_wall_seconds
        results[phase] = {
            "steps": phase_steps,
            "wall_seconds": phase_wall_seconds,
            "ms_per_step": (phase_wall_seconds / phase_steps) * 1000.0 if phase_steps else 0.0,
        }

    results["total"] = {
        "steps": total_steps,
        "wall_seconds": total_wall_seconds,
        "ms_per_step": (total_wall_seconds / total_steps) * 1000.0 if total_steps else 0.0,
    }
    return results


def _log_replay_benchmark_summary(mode: str, summary: dict[str, dict[str, float | int]], snapshot) -> None:
    for phase in ("phase_1_approach", "phase_2_descend", "phase_3_grip", "total"):
        phase_summary = summary[phase]
        logger.info(
            "replay_reference_benchmark mode=%s phase=%s steps=%d wall_ms=%.3f ms_per_step=%.3f",
            mode,
            phase,
            int(phase_summary["steps"]),
            float(phase_summary["wall_seconds"]) * 1000.0,
            float(phase_summary["ms_per_step"]),
        )
    if snapshot is not None:
        logger.info(
            "replay_reference_profile mode=%s predict_calls=%d propagate_calls=%d batch_step_calls=%d "
            "rng_wall_ms=%.3f propagate_wall_ms=%.3f batch_step_wall_ms=%.3f block_wall_ms=%.3f "
            "mjx_bytes_in_use=%d mjx_peak_bytes_in_use=%d mjx_bytes_limit=%d",
            mode,
            int(snapshot["predict_call_count"]),
            int(snapshot["propagate_call_count"]),
            int(snapshot["batch_step_call_count"]),
            float(snapshot["rng_wall_seconds"]) * 1000.0,
            float(snapshot["propagate_wall_seconds"]) * 1000.0,
            float(snapshot["batch_step_wall_seconds"]) * 1000.0,
            float(snapshot["block_until_ready_seconds"]) * 1000.0,
            int(snapshot["mjx_bytes_in_use"]),
            int(snapshot["mjx_peak_bytes_in_use"]),
            int(snapshot["mjx_bytes_limit"]),
        )


signal.signal(signal.SIGINT, _handle_shutdown_signal)
signal.signal(signal.SIGTERM, _handle_shutdown_signal)

# ==========================================
# 1. SETUP
# ==========================================
run_wall_start = time.perf_counter()
metrics = create_metrics_from_env()
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
logger = setup_logging()
headless = os.getenv("SIMBAY_HEADLESS", "true").lower() in {"1", "true", "yes", "on"}
use_mjx = os.getenv("SIMBAY_USE_MJX", "true").lower() in {"1", "true", "yes", "on"}
export_particle_mass_metrics = os.getenv(
    "SIMBAY_EXPORT_PARTICLE_MASS_METRICS",
    "false",
).lower() in {"1", "true", "yes", "on"}
particle_mass_metrics_every_n_steps = max(
    1,
    int(os.getenv("SIMBAY_PARTICLE_MASS_METRICS_EVERY_N_STEPS", "10")),
)
benchmark_replay_only = _env_enabled("SIMBAY_REPLAY_BENCHMARK_ONLY")
benchmark_replay_sync = _env_enabled("SIMBAY_REPLAY_BENCHMARK_SYNC", "true")
replay_chunk_size = _parse_replay_chunk_size()

if use_mjx:
    try:
        device = jax.devices("gpu")[0]
        memory_stats = device.memory_stats() or {}
        print(f"GPU memory stats: {memory_stats}")
        logger.info("gpu_memory_stats device=%s stats=%s", device, memory_stats)
    except Exception as exc:
        logger.warning("gpu_memory_stats_unavailable error=%s", exc)

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
env, particle_filter = _create_particle_filter(use_mjx, limits, num_particles)
if export_particle_mass_metrics:
    if use_mjx:
        metrics.update_particle_mass_metrics(particle_filter.particles_host())
    else:
        metrics.update_particle_mass_metrics(particle_filter.particles.copy())
memory_profile = particle_filter.memory_profile()
env_memory_profile = env.memory_profile()
cpu_cores = os.cpu_count() or 1
metrics.set_particle_count(num_particles)

if use_mjx:
    metrics.set_backend("mjx", str(env_memory_profile["execution_device"]))
    metrics.update_mjx_memory(
        "setup",
        int(env_memory_profile["bytes_in_use"]),
        int(env_memory_profile["peak_bytes_in_use"]),
        int(env_memory_profile["bytes_limit"]),
    )
    logger.info(
        "simulation_setup dt=%.6f true_mass=%.4f particles=%d cpu_cores=%d "
        "headless=%s backend=mjx "
        "benchmark_replay_only=%s benchmark_replay_sync=%s "
        "replay_chunk_size=%d "
        "state_memory_total_bytes=%d state_memory_total=%s "
        "state_memory_per_particle_bytes=%.2f state_memory_per_particle=%s "
        "process_memory_per_particle_estimate_bytes=%.2f "
        "process_memory_per_particle_estimate=%s "
        "mjx_execution_platform=%s mjx_execution_device=%s "
        "jax_default_platform=%s jax_default_device=%s "
        "mjx_device_fallback_applied=%s "
        "mjx_bytes_in_use=%d mjx_peak_bytes_in_use=%d mjx_bytes_limit=%d",
        dt,
        true_mass,
        num_particles,
        cpu_cores,
        headless,
        benchmark_replay_only,
        benchmark_replay_sync,
        replay_chunk_size,
        memory_profile["state_bytes_total"],
        format_bytes(memory_profile["state_bytes_total"]),
        memory_profile["state_bytes_per_particle"],
        format_bytes(memory_profile["state_bytes_per_particle"]),
        memory_profile["process_memory_per_particle_estimate_bytes"],
        format_bytes(memory_profile["process_memory_per_particle_estimate_bytes"]),
        env_memory_profile["execution_platform"],
        env_memory_profile["execution_device"],
        env_memory_profile["default_jax_platform"],
        env_memory_profile["default_jax_device"],
        env_memory_profile["device_fallback_applied"],
        env_memory_profile["bytes_in_use"],
        env_memory_profile["peak_bytes_in_use"],
        env_memory_profile["bytes_limit"],
    )
else:
    logger.info(
        "simulation_setup dt=%.6f true_mass=%.4f particles=%d cpu_cores=%d "
        "headless=%s backend=mujoco "
        "benchmark_replay_only=%s benchmark_replay_sync=%s "
        "replay_chunk_size=%d "
        "state_memory_total_bytes=%d state_memory_total=%s "
        "state_memory_per_particle_bytes=%.2f state_memory_per_particle=%s "
        "process_memory_per_particle_estimate_bytes=%.2f "
        "process_memory_per_particle_estimate=%s "
        "mujoco_model_buffer_per_particle_bytes=%d mujoco_model_buffer_per_particle=%s "
        "mujoco_data_buffer_per_particle_bytes=%d mujoco_data_buffer_per_particle=%s "
        "mujoco_data_arena_per_particle_bytes=%d mujoco_data_arena_per_particle=%s "
        "mujoco_native_memory_per_particle_bytes=%d mujoco_native_memory_per_particle=%s "
        "mujoco_native_memory_total_bytes=%d mujoco_native_memory_total=%s",
        dt,
        true_mass,
        num_particles,
        cpu_cores,
        headless,
        benchmark_replay_only,
        benchmark_replay_sync,
        replay_chunk_size,
        memory_profile["state_bytes_total"],
        format_bytes(memory_profile["state_bytes_total"]),
        memory_profile["state_bytes_per_particle"],
        format_bytes(memory_profile["state_bytes_per_particle"]),
        memory_profile["process_memory_per_particle_estimate_bytes"],
        format_bytes(memory_profile["process_memory_per_particle_estimate_bytes"]),
        env_memory_profile["model_nbuffer_bytes_per_robot"],
        format_bytes(env_memory_profile["model_nbuffer_bytes_per_robot"]),
        env_memory_profile["data_nbuffer_bytes_per_robot"],
        format_bytes(env_memory_profile["data_nbuffer_bytes_per_robot"]),
        env_memory_profile["data_narena_bytes_per_robot"],
        format_bytes(env_memory_profile["data_narena_bytes_per_robot"]),
        env_memory_profile["native_bytes_per_robot"],
        format_bytes(env_memory_profile["native_bytes_per_robot"]),
        env_memory_profile["native_bytes_total"],
        format_bytes(env_memory_profile["native_bytes_total"]),
    )
    metrics.set_backend("mujoco", "cpu")
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
if use_mjx:
    replay_chunk_sizes = (replay_chunk_size,) if replay_chunk_size > 0 else ()
    particle_filter.warmup_runtime(replay_chunk_sizes=replay_chunk_sizes)
    logger.info("mjx_filter_warmup_complete particles=%d", particle_filter.N)
metrics.finish_stage(planning_stage)

if benchmark_replay_only:
    replay_trajectories = [
        ("phase_1_approach", traj1),
        ("phase_2_descend", traj2),
        ("phase_3_grip", traj3),
    ]

    if not use_mjx:
        logger.warning("replay_reference_benchmark unsupported backend=mujoco")
    else:
        cold_env, cold_filter = _create_particle_filter(use_mjx, limits, num_particles)
        cold_filter.warmup_runtime()
        cold_summary = _run_reference_replay(
            cold_filter,
            replay_trajectories,
            synchronize=benchmark_replay_sync,
        )
        _log_replay_benchmark_summary(
            "cold",
            cold_summary,
            cold_filter.replay_profile_snapshot(),
        )

        steady_env, steady_filter = _create_particle_filter(use_mjx, limits, num_particles)
        steady_filter.warmup_runtime()
        steady_summary = _run_reference_replay(
            steady_filter,
            replay_trajectories,
            synchronize=benchmark_replay_sync,
        )
        _log_replay_benchmark_summary(
            "steady",
            steady_summary,
            steady_filter.replay_profile_snapshot(),
        )

        compile_overhead_seconds = (
            float(cold_summary["total"]["wall_seconds"]) - float(steady_summary["total"]["wall_seconds"])
        )
        logger.info(
            "replay_reference_benchmark_compare cold_wall_ms=%.3f steady_wall_ms=%.3f "
            "compile_overhead_ms=%.3f sync=%s",
            float(cold_summary["total"]["wall_seconds"]) * 1000.0,
            float(steady_summary["total"]["wall_seconds"]) * 1000.0,
            compile_overhead_seconds * 1000.0,
            benchmark_replay_sync,
        )
        del cold_env, steady_env

    shutdown_metrics(metrics)
    raise SystemExit(0)

# ==========================================
# 3. EXECUTION
# ==========================================

# Phase 1: Move ABOVE the object (No PF updates, just predict to stay synced)
approach_stage = metrics.start_stage("phase_1_approach")
logger.info("phase_start name=approach steps=%d", len(traj1))
robot_execute_stage = metrics.start_substage("phase_1_approach", "robot_execute")
for i, qpos in enumerate(traj1):
    real_robot.move_joints(qpos)
    if viewer is not None:
        viewer.sync()
    if (i + 1) % 100 == 0 or i == len(traj1) - 1:
        logger.info("phase_progress name=approach step=%d/%d", i + 1, len(traj1))
robot_execute_duration = metrics.finish_substage(robot_execute_stage)
_log_substage_duration("phase_1_approach", "robot_execute", robot_execute_duration, len(traj1))
pf_replay_stage = metrics.start_substage("phase_1_approach", "pf_replay")
_replay_trajectory(particle_filter, "phase_1_approach", traj1, replay_chunk_size)
pf_replay_duration = metrics.finish_substage(pf_replay_stage)
_log_substage_duration("phase_1_approach", "pf_replay", pf_replay_duration, len(traj1))
if use_mjx:
    env_memory_profile = env.memory_profile()
    metrics.update_mjx_memory(
        "phase_1_approach",
        int(env_memory_profile["bytes_in_use"]),
        int(env_memory_profile["peak_bytes_in_use"]),
        int(env_memory_profile["bytes_limit"]),
    )
metrics.finish_stage(approach_stage)


# Phase 2: Descend vertically to the object (No PF updates)
descend_stage = metrics.start_stage("phase_2_descend")
logger.info("phase_start name=descend steps=%d", len(traj2))
robot_execute_stage = metrics.start_substage("phase_2_descend", "robot_execute")
for i, qpos in enumerate(traj2):
    real_robot.move_joints(qpos)
    if viewer is not None:
        viewer.sync()
    if (i + 1) % 100 == 0 or i == len(traj2) - 1:
        logger.info("phase_progress name=descend step=%d/%d", i + 1, len(traj2))
robot_execute_duration = metrics.finish_substage(robot_execute_stage)
_log_substage_duration("phase_2_descend", "robot_execute", robot_execute_duration, len(traj2))
pf_replay_stage = metrics.start_substage("phase_2_descend", "pf_replay")
_replay_trajectory(particle_filter, "phase_2_descend", traj2, replay_chunk_size)
pf_replay_duration = metrics.finish_substage(pf_replay_stage)
_log_substage_duration("phase_2_descend", "pf_replay", pf_replay_duration, len(traj2))
if use_mjx:
    env_memory_profile = env.memory_profile()
    metrics.update_mjx_memory(
        "phase_2_descend",
        int(env_memory_profile["bytes_in_use"]),
        int(env_memory_profile["peak_bytes_in_use"]),
        int(env_memory_profile["bytes_limit"]),
    )
metrics.finish_stage(descend_stage)


# Phase 3: Close the Gripper (No PF updates)
grip_stage = metrics.start_stage("phase_3_grip")
logger.info("phase_start name=close_gripper steps=%d", len(traj3))
robot_execute_stage = metrics.start_substage("phase_3_grip", "robot_execute")
for i, qpos in enumerate(traj3):
    real_robot.move_joints(qpos)
    if viewer is not None:
        viewer.sync()
    if (i + 1) % 100 == 0 or i == len(traj3) - 1:
        logger.info("phase_progress name=close_gripper step=%d/%d", i + 1, len(traj3))
robot_execute_duration = metrics.finish_substage(robot_execute_stage)
_log_substage_duration("phase_3_grip", "robot_execute", robot_execute_duration, len(traj3))
pf_replay_stage = metrics.start_substage("phase_3_grip", "pf_replay")
_replay_trajectory(particle_filter, "phase_3_grip", traj3, replay_chunk_size)
pf_replay_duration = metrics.finish_substage(pf_replay_stage)
_log_substage_duration("phase_3_grip", "pf_replay", pf_replay_duration, len(traj3))
if use_mjx:
    env_memory_profile = env.memory_profile()
    metrics.update_mjx_memory(
        "phase_3_grip",
        int(env_memory_profile["bytes_in_use"]),
        int(env_memory_profile["peak_bytes_in_use"]),
        int(env_memory_profile["bytes_limit"]),
    )
metrics.finish_stage(grip_stage)


# Phase 4: Lift straight up (OBJECT IS GRASPED - START TRACKING MASS)
lft_stage = metrics.start_stage("phase_4_lift")
logger.info("phase_start name=lift_and_estimate")

# <--- Initialize lists to hold the historical data for the graph --->
history_particles = []
history_estimates = []
pf_wall_durations = []
pf_cpu_durations = []
phase_4_robot_execute_total = 0.0
phase_4_pf_update_total = 0.0

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
    if use_mjx:
        particle_filter.step(qpos, noisy_ft_reading)
    else:
        particle_filter.predict(qpos)
        particle_filter.update(noisy_ft_reading)
        particle_filter.resample()

    # <--- Save the state of the particles at this exact timestep --->
    if use_mjx:
        current_particles = particle_filter.particles_host()
    else:
        current_particles = particle_filter.particles.copy()
    history_particles.append(current_particles.copy())
    if (
        use_mjx
        and export_particle_mass_metrics
        and (step % particle_mass_metrics_every_n_steps == 0)
    ):
        metrics.update_particle_mass_metrics(current_particles)
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
    )
    should_log_step = step in (0, len(traj4) - 1) or (step + 1) % 10 == 0
    if should_log_step:
        rss_bytes = get_process_memory_bytes()
        if use_mjx:
            env_memory_profile = env.memory_profile()
            metrics.update_mjx_memory(
                "phase_4_lift",
                int(env_memory_profile["bytes_in_use"]),
                int(env_memory_profile["peak_bytes_in_use"]),
                int(env_memory_profile["bytes_limit"]),
            )
            logger.info(
                "particle_filter_step step=%d particles=%d wall_ms=%.3f step_rate_hz=%.2f "
                "cpu_ms=%.3f cpu_equivalent_cores=%.3f cpu_percent_single_core=%.2f "
                "cpu_percent_total_machine=%.2f ess=%.2f estimate=%.6f "
                "rss_bytes=%d rss=%s backend=mjx "
                "mjx_execution_platform=%s mjx_execution_device=%s "
                "jax_default_platform=%s jax_default_device=%s "
                "mjx_device_fallback_applied=%s",
                step,
                particle_filter.N,
                step_wall_duration * 1000.0,
                step_rate_hz,
                step_cpu_duration * 1000.0,
                cpu_equivalent_cores_used,
                cpu_percent_single_core,
                cpu_percent_total_machine,
                particle_filter.effective_sample_size(),
                current_estimate,
                rss_bytes,
                format_bytes(rss_bytes),
                env_memory_profile["execution_platform"],
                env_memory_profile["execution_device"],
                env_memory_profile["default_jax_platform"],
                env_memory_profile["default_jax_device"],
                env_memory_profile["device_fallback_applied"],
            )
        else:
            logger.info(
                "particle_filter_step step=%d particles=%d wall_ms=%.3f step_rate_hz=%.2f "
                "cpu_ms=%.3f cpu_equivalent_cores=%.3f cpu_percent_single_core=%.2f "
                "cpu_percent_total_machine=%.2f ess=%.2f estimate=%.6f "
                "rss_bytes=%d rss=%s mujoco_native_memory_total_bytes=%d "
                "mujoco_native_memory_total=%s",
                step,
                particle_filter.N,
                step_wall_duration * 1000.0,
                step_rate_hz,
                step_cpu_duration * 1000.0,
                cpu_equivalent_cores_used,
                cpu_percent_single_core,
                cpu_percent_total_machine,
                particle_filter.effective_sample_size(),
                current_estimate,
                rss_bytes,
                format_bytes(rss_bytes),
                env_memory_profile["native_bytes_total"],
                format_bytes(env_memory_profile["native_bytes_total"]),
            )
metrics.set_substage_duration("phase_4_lift", "robot_execute", phase_4_robot_execute_total)
metrics.set_substage_duration("phase_4_lift", "pf_update", phase_4_pf_update_total)
_log_substage_duration("phase_4_lift", "robot_execute", phase_4_robot_execute_total, len(traj4))
_log_substage_duration("phase_4_lift", "pf_update", phase_4_pf_update_total, len(traj4))
metrics.finish_stage(lft_stage)

avg_wall_duration = sum(pf_wall_durations) / len(pf_wall_durations) if pf_wall_durations else 0.0
avg_cpu_duration = sum(pf_cpu_durations) / len(pf_cpu_durations) if pf_cpu_durations else 0.0
avg_step_rate_hz = 1.0 / avg_wall_duration if avg_wall_duration > 0 else 0.0
avg_cpu_equivalent_cores = (
    avg_cpu_duration / avg_wall_duration if avg_wall_duration > 0 else 0.0
)
if use_mjx:
    env_memory_profile = env.memory_profile()
    logger.info(
        "particle_filter_summary steps=%d avg_wall_ms=%.3f avg_step_rate_hz=%.2f "
        "avg_cpu_ms=%.3f avg_cpu_equivalent_cores=%.3f final_estimate=%.6f "
        "final_error_pct=%.2f final_rss_bytes=%d final_rss=%s "
        "backend=mjx mjx_execution_platform=%s mjx_execution_device=%s "
        "jax_default_platform=%s jax_default_device=%s "
        "mjx_device_fallback_applied=%s",
        len(pf_wall_durations),
        avg_wall_duration * 1000.0,
        avg_step_rate_hz,
        avg_cpu_duration * 1000.0,
        avg_cpu_equivalent_cores,
        float(particle_filter.estimate()),
        abs(true_mass - particle_filter.estimate()) * 100,
        get_process_memory_bytes(),
        format_bytes(get_process_memory_bytes()),
        env_memory_profile["execution_platform"],
        env_memory_profile["execution_device"],
        env_memory_profile["default_jax_platform"],
        env_memory_profile["default_jax_device"],
        env_memory_profile["device_fallback_applied"],
    )
else:
    logger.info(
        "particle_filter_summary steps=%d avg_wall_ms=%.3f avg_step_rate_hz=%.2f "
        "avg_cpu_ms=%.3f avg_cpu_equivalent_cores=%.3f final_estimate=%.6f "
        "final_error_pct=%.2f final_rss_bytes=%d final_rss=%s "
        "mujoco_native_memory_total_bytes=%d mujoco_native_memory_total=%s",
        len(pf_wall_durations),
        avg_wall_duration * 1000.0,
        avg_step_rate_hz,
        avg_cpu_duration * 1000.0,
        avg_cpu_equivalent_cores,
        float(particle_filter.estimate()),
        abs(true_mass - particle_filter.estimate()) * 100,
        get_process_memory_bytes(),
        format_bytes(get_process_memory_bytes()),
        env_memory_profile["native_bytes_total"],
        format_bytes(env_memory_profile["native_bytes_total"]),
    )

logger.info("sequence_complete awaiting_user_input=%s", not headless)
if not headless and not shutdown_requested:
    input()
elif not headless and shutdown_requested:
    logger.info("sequence_complete skipping_user_input signal=%s", shutdown_signal_name)
final_prediction = float(particle_filter.estimate())
time_to_prediction_seconds = time.perf_counter() - run_wall_start
metrics.set_prediction_ready(
    total_wall_seconds=time_to_prediction_seconds,
    final_error_pct=abs(true_mass - final_prediction) * 100,
)
logger.info(
    "prediction_ready total_wall_s=%.3f final_mass_prediction_kg=%.4f",
    time_to_prediction_seconds,
    final_prediction,
)
logger.info("final_mass_prediction_kg=%.4f", final_prediction)
logger.info("final_error_pct=%.2f", abs(true_mass - final_prediction) * 100)
# You should also print the real mass here to see if the filter got it right!


# ==========================================
# 4. GRAPH GENERATION
# ==========================================
plot_stage = metrics.start_stage("plot_generation")
logger.info("plot_generation_start")


plt.figure(figsize=(10, 6))

# Plot the 100 particles over time as highly transparent blue dots
num_steps = len(history_particles)
for t in range(num_steps):
    plt.scatter([t] * num_particles, history_particles[t], color='blue', alpha=0.05, s=15)

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
logger.info("plot_saved path=%s", output_path)
metrics.finish_stage(plot_stage)

if use_mjx:
    jax.clear_caches()
    gc.collect()
    logger.info("jax_cleanup_complete")

logger.info(
    "goodbye shutdown_requested=%s signal=%s",
    shutdown_requested,
    shutdown_signal_name or "none",
)
shutdown_metrics(metrics)
