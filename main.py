from pathlib import Path
import os
import time

import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np

from src.estimation import FrankaMJXEnv
from src.estimation import FrankaMuJoCoEnv
from src.estimation import ParticleFilter
from src.planning import FrankaSmartSolver
from src.planning import plan_linear_trajectory
from src.utils import DEFAULT_OBJECT_PROPS
from src.utils import FRANKA_HOME_QPOS
from src.utils import initialize_mujoco_env
from src.utils.logging_utils import format_bytes
from src.utils.logging_utils import get_process_memory_bytes
from src.utils.logging_utils import setup_logging

# ==========================================
# 1. SETUP
# ==========================================
logger = setup_logging()
headless = os.getenv("SIMBAY_HEADLESS", "").lower() in {"1", "true", "yes", "on"}
use_mjx = os.getenv("SIMBAY_USE_MJX", "").lower() in {"1", "true", "yes", "on"}

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
if use_mjx:
    env = FrankaMJXEnv(limits, num_particles)
else:
    env = FrankaMuJoCoEnv(limits, num_particles)
particle_filter = ParticleFilter(env)
memory_profile = particle_filter.memory_profile()
env_memory_profile = env.memory_profile()
cpu_cores = os.cpu_count() or 1

if use_mjx:
    logger.info(
        "simulation_setup dt=%.6f true_mass=%.4f particles=%d cpu_cores=%d "
        "headless=%s backend=mjx "
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

# ==========================================
# 2. TRAJECTORY PLANNING
# ==========================================
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

# ==========================================
# 3. EXECUTION
# ==========================================

# Phase 1: Move ABOVE the object (No PF updates, just predict to stay synced)
traj1 = plan_linear_trajectory(q_home, q_pre_grasp, max_velocity=1.0, dt=dt)
logger.info("phase_start name=approach steps=%d", len(traj1))
for i, qpos in enumerate(traj1):
    real_robot.move_joints(qpos)
    if viewer is not None:
        viewer.sync()
    particle_filter.predict(qpos)
    if (i + 1) % 100 == 0 or i == len(traj1) - 1:
        logger.info("phase_progress name=approach step=%d/%d", i + 1, len(traj1))


# Phase 2: Descend vertically to the object (No PF updates)
traj2 = plan_linear_trajectory(q_pre_grasp, q_grasp_open, max_velocity=0.5, dt=dt)
logger.info("phase_start name=descend steps=%d", len(traj2))
for i, qpos in enumerate(traj2):
    real_robot.move_joints(qpos)
    if viewer is not None:
        viewer.sync()
    particle_filter.predict(qpos)
    if (i + 1) % 100 == 0 or i == len(traj2) - 1:
        logger.info("phase_progress name=descend step=%d/%d", i + 1, len(traj2))


# Phase 3: Close the Gripper (No PF updates)
traj3 = plan_linear_trajectory(q_grasp_closed, q_grasp_closed, max_velocity=500, dt=dt, settle_time=0.5) # we close directly and only use the settle_time
logger.info("phase_start name=close_gripper steps=%d", len(traj3))
for i, qpos in enumerate(traj3):
    real_robot.move_joints(qpos)
    if viewer is not None:
        viewer.sync()
    particle_filter.predict(qpos)
    if (i + 1) % 100 == 0 or i == len(traj3) - 1:
        logger.info("phase_progress name=close_gripper step=%d/%d", i + 1, len(traj3))


# Phase 4: Lift straight up (OBJECT IS GRASPED - START TRACKING MASS)
logger.info("phase_start name=lift_and_estimate")
traj4 = plan_linear_trajectory(q_grasp_closed, q_lift_closed, max_velocity=0.5, dt=dt, settle_time=1.0) 

# <--- Initialize lists to hold the historical data for the graph --->
history_particles = []
history_estimates = []
pf_wall_durations = []
pf_cpu_durations = []

for step, qpos in enumerate(traj4):
    real_robot.move_joints(qpos)
    if viewer is not None:
        viewer.sync()

    step_wall_start = time.perf_counter()
    step_cpu_start = time.process_time()

    # 1. Step the particles forward
    particle_filter.predict(qpos)
    
    # 2. Get the real measurement
    measurements = real_robot.get_sensor_reads()
    real_ft_reading = measurements
    
    # 3. INJECT NOISE to simulate physical hardware (e.g., 0.5N of sensor noise)
    noisy_ft_reading = real_ft_reading + np.random.normal(0, 0.5, size=3)
    
    # 4. Update beliefs based on the noisy real reading and resample
    particle_filter.update(noisy_ft_reading)
    particle_filter.resample()

    # <--- Save the state of the particles at this exact timestep --->
    history_particles.append(particle_filter.particles.copy())
    history_estimates.append(particle_filter.estimate())

    step_wall_duration = time.perf_counter() - step_wall_start
    step_cpu_duration = time.process_time() - step_cpu_start
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
    should_log_step = step in (0, len(traj4) - 1) or (step + 1) % 10 == 0
    if should_log_step:
        rss_bytes = get_process_memory_bytes()
        if use_mjx:
            env_memory_profile = env.memory_profile()
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
                float(particle_filter.estimate()),
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
                float(particle_filter.estimate()),
                rss_bytes,
                format_bytes(rss_bytes),
                env_memory_profile["native_bytes_total"],
                format_bytes(env_memory_profile["native_bytes_total"]),
            )

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
if not headless:
    input()
logger.info("final_mass_prediction_kg=%.4f", float(particle_filter.estimate()))
logger.info("final_error_pct=%.2f", abs(true_mass - particle_filter.estimate()) * 100)
# You should also print the real mass here to see if the filter got it right!


# ==========================================
# 4. GRAPH GENERATION
# ==========================================
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
