import gc
import os
from pathlib import Path

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
from src.utils.constants import DEFAULT_OBJECT_PROPS
from src.utils.constants import FRANKA_HOME_QPOS
from src.utils.mujoco_utils import initialize_mujoco_env
from src.utils.settings import BACKEND
from src.utils.settings import HEADLESS
from src.utils.settings import NUM_PARTICLES
from src.utils.tracing import set_span_attributes
from src.utils.tracing import span as tracing_span
from src.utils.workflow import ExecutionContext
from src.utils.workflow import get_shutdown_signal_name
from src.utils.workflow import init_runtime
from src.utils.workflow import install_signal_handlers
from src.utils.workflow import is_shutdown_requested
from src.utils.workflow import log_setup_summary
from src.utils.workflow import log_stage_finished
from src.utils.workflow import log_stage_started
from src.utils.workflow import run_phase_1_approach
from src.utils.workflow import run_phase_2_descend
from src.utils.workflow import run_phase_3_grip
from src.utils.workflow import run_phase_4_lift
from src.utils.workflow import shutdown_runtime
from src.utils.workflow import stage_span_attrs
from src.utils.workflow import update_setup_metrics
from src.utils.workflow import update_warp_memory_metrics
from src.utils.workflow import RuntimeContext


def main(runtime: RuntimeContext) -> None:
    logger = runtime.logger
    metrics = runtime.metrics
    tracer = runtime.tracer
    run_id = runtime.run_id
    log_data = runtime.log_data

    setup_stage = metrics.start_stage("setup")
    log_stage_started(logger, "setup", **log_data)
    with tracing_span(tracer, "setup"):
        headless = HEADLESS
        backend = BACKEND

        use_batched_backend = backend == "mujoco-warp"
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
            _uniform_weight_metrics = None
            _update_and_optionally_resample = None
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
                "simbay.true_mass": float(true_mass),
            }
        )
        metrics.set_particle_count(num_particles)
        metrics.set_backend(backend, execution_device)
        metrics.set_run_info(backend=backend, particles=num_particles, control_dt=dt)
        update_setup_metrics(metrics, backend, env_memory_profile, memory_profile)
        if backend == "mujoco-warp":
            update_warp_memory_metrics(env, metrics, stage="setup")
        log_setup_summary(
            logger,
            backend,
            headless,
            **log_data,
        )
    setup_duration = metrics.finish_stage(setup_stage)
    log_stage_finished(logger, "setup", setup_duration, **log_data)

    ctx = ExecutionContext(
        runtime=runtime,
        backend=backend,
        num_particles=num_particles,
        dt=dt,
        execution_device=execution_device,
        cpu_cores=cpu_cores,
    )

    planning_stage = metrics.start_stage("ik_planning")
    log_stage_started(logger, "ik_planning", **log_data)
    with tracing_span(tracer, "ik_planning"):
        set_span_attributes(stage_span_attrs(ctx, "ik_planning"))
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
        if use_batched_backend:
            warmed_rollout_lengths = particle_filter.warmup_runtime([len(traj1), len(traj2), len(traj3)])
            logger.info({**log_data, "event": "backend_runtime_warmup_summary", "msg": f"Finished backend runtime warm-up for the {backend} backend.", "backend": backend, "rollout_lengths": warmed_rollout_lengths, "phase4_step_warmup": 1})
    planning_duration = metrics.finish_stage(planning_stage)
    log_stage_finished(logger, "ik_planning", planning_duration, **log_data)

    run_phase_1_approach(
        ctx,
        steps=len(traj1),
        trajectory=traj1,
        real_robot=real_robot,
        viewer=viewer,
        particle_filter=particle_filter,
        use_batched_backend=use_batched_backend,
        env=env,
    )
    run_phase_2_descend(
        ctx,
        steps=len(traj2),
        trajectory=traj2,
        real_robot=real_robot,
        viewer=viewer,
        particle_filter=particle_filter,
        use_batched_backend=use_batched_backend,
        env=env,
    )
    run_phase_3_grip(
        ctx,
        steps=len(traj3),
        trajectory=traj3,
        real_robot=real_robot,
        viewer=viewer,
        particle_filter=particle_filter,
        use_batched_backend=use_batched_backend,
        env=env,
    )
    lift_result = run_phase_4_lift(
        ctx,
        steps=len(traj4),
        trajectory=traj4,
        real_robot=real_robot,
        viewer=viewer,
        particle_filter=particle_filter,
        env=env,
        true_mass=true_mass,
        uniform_weight_metrics=_uniform_weight_metrics,
        update_and_optionally_resample=_update_and_optionally_resample,
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
        update_warp_memory_metrics(env, metrics, stage="phase_4_lift")
        logger.info({**log_data, "event": "particle_filter_completed", "msg": "Finished the particle filter run.", "backend": "mujoco-warp"})

    logger.info({**log_data, "event": "sequence_complete", "msg": "Finished the execution sequence.", "awaiting_user_input": not headless})
    if not headless and not is_shutdown_requested():
        input()
    elif not headless and is_shutdown_requested():
        logger.info({**log_data, "event": "sequence_complete_skipping_user_input", "msg": "Skipped waiting for user input because shutdown was requested.", "signal": get_shutdown_signal_name()})

    final_prediction = float(particle_filter.estimate())
    time_to_prediction_seconds = runtime.elapsed_seconds()
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
    logger.info({**log_data, "event": "goodbye", "msg": "Finished the run and shut down cleanly.", "shutdown_requested": is_shutdown_requested(), "signal": get_shutdown_signal_name() or "none"})


if __name__ == "__main__":
    runtime = init_runtime()
    install_signal_handlers(runtime.logger, runtime.log_data)
    try:
        main(runtime)
    finally:
        shutdown_runtime(runtime)
