from __future__ import annotations

import functools
import math
import signal
import time
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any

import numpy as np

from .logging_utils import logger as simbay_logger
from .logging_utils import setup_logging
from .metrics import init_metrics
from .metrics import shutdown_metrics
from .profiling import annotate
from .settings import RUN_ID
from .tracing import add_exemplar
from .tracing import force_flush_tracing
from .tracing import get_tracer
from .tracing import set_span_attributes
from .tracing import setup_tracing
from .tracing import shutdown_tracing
from .tracing import span as tracing_span


@dataclass
class RuntimeContext:
    run_id: str
    tracer: Any
    metrics: Any
    logger: Any
    log_data: dict[str, object]
    _exit_stack: ExitStack
    _started_at: float

    def elapsed_seconds(self) -> float:
        return time.perf_counter() - self._started_at

    def close(self) -> None:
        self._exit_stack.close()


@dataclass
class ExecutionContext:
    runtime: RuntimeContext
    backend: str
    num_particles: int
    dt: float
    execution_device: str
    cpu_cores: int

    @property
    def run_id(self) -> str:
        return self.runtime.run_id

    @property
    def tracer(self) -> Any:
        return self.runtime.tracer

    @property
    def metrics(self) -> Any:
        return self.runtime.metrics

    @property
    def logger(self) -> Any:
        return self.runtime.logger

    @property
    def log_data(self) -> dict[str, object]:
        return self.runtime.log_data


@dataclass
class LiftPhaseResult:
    history_estimates: list[float]
    pf_wall_durations: list[float]
    pf_cpu_durations: list[float]
    invalid_sensor_events: int
    invalid_state_events: int
    skipped_invalid_updates: int
    first_invalid_sensor_step: int
    first_invalid_state_step: int
    max_repaired_world_count: int


shutdown_requested = False
shutdown_signal_name: str | None = None


def init_runtime(run_id: str = RUN_ID) -> RuntimeContext:
    runtime_log_data = {"run_id": run_id}
    setup_tracing(run_id=run_id)
    runtime_tracer = get_tracer("simbay.main")
    runtime_metrics = init_metrics(run_id=run_id)
    runtime_logger = setup_logging(run_id=run_id)
    exit_stack = ExitStack()
    exit_stack.enter_context(tracing_span(runtime_tracer, "simbay.run"))
    exit_stack.callback(shutdown_tracing)
    exit_stack.callback(force_flush_tracing)
    exit_stack.callback(shutdown_metrics, runtime_metrics)
    set_span_attributes({"simbay.run_id": run_id})
    return RuntimeContext(
        run_id=run_id,
        tracer=runtime_tracer,
        metrics=runtime_metrics,
        logger=runtime_logger,
        log_data=runtime_log_data,
        _exit_stack=exit_stack,
        _started_at=time.perf_counter(),
    )


def shutdown_runtime(runtime: RuntimeContext | None) -> None:
    if runtime is not None:
        runtime.close()


def install_signal_handlers(logger: Any, log_data: dict[str, object]) -> None:
    def _handle_shutdown_signal(signum, _frame) -> None:
        global shutdown_requested
        global shutdown_signal_name

        shutdown_requested = True
        shutdown_signal_name = signal.Signals(signum).name
        logger.info(
            {
                **log_data,
                "event": "shutdown_requested",
                "msg": f"Received {shutdown_signal_name} and will finish the current run before shutting down.",
                "signal": shutdown_signal_name,
                "mode": "graceful",
                "action": "finish_current_run",
            }
        )

    signal.signal(signal.SIGINT, _handle_shutdown_signal)
    signal.signal(signal.SIGTERM, _handle_shutdown_signal)


def is_shutdown_requested() -> bool:
    return shutdown_requested


def get_shutdown_signal_name() -> str | None:
    return shutdown_signal_name


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
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


def stage_span_attrs(
    ctx: ExecutionContext,
    stage: str,
    *,
    steps: int | None = None,
) -> dict[str, str | int | float | bool]:
    attrs: dict[str, str | int | float | bool] = {
        "simbay.run_id": ctx.run_id,
        "simbay.stage": stage,
        "simbay.backend": ctx.backend,
        "simbay.particles": ctx.num_particles,
        "simbay.control_dt": ctx.dt,
        "simbay.execution_device": ctx.execution_device,
    }
    if steps is not None:
        attrs["simbay.steps"] = steps
    return attrs


def substage_span_attrs(
    ctx: ExecutionContext,
    stage: str,
    substage: str,
    *,
    steps: int,
    execution_mode: str,
    parallel_units: int,
) -> dict[str, str | int | float | bool]:
    attrs = stage_span_attrs(ctx, stage, steps=steps)
    attrs["simbay.substage"] = substage
    attrs["simbay.execution_mode"] = execution_mode
    attrs["simbay.parallel_units"] = parallel_units
    return attrs


def log_setup_summary(
    logger: Any,
    backend_name: str,
    headless: bool,
    **log_data: object,
) -> None:
    logger.info(
        {
            **log_data,
            "event": "simulation_setup",
            "msg": "Completed simulation setup.",
            "backend": backend_name,
            "headless": headless,
        }
    )


def log_stage_started(logger: Any, stage: str, *, steps: int | None = None, **log_data: object) -> None:
    payload = {
        **log_data,
        "event": "stage_started",
        "msg": f"Started {stage.replace('_', ' ')}.",
        "stage": stage,
    }
    logger.info(payload)


def log_stage_finished(logger: Any, stage: str, duration_seconds: float, **log_data: object) -> None:
    logger.info(
        {
            **log_data,
            "event": "stage_finished",
            "msg": f"Finished {stage.replace('_', ' ')}.",
            "stage": stage,
        }
    )


def log_substage_started(logger: Any, phase: str, substage: str, *, steps: int, **log_data: object) -> None:
    logger.info(
        {
            **log_data,
            "event": "substage_started",
            "msg": f"Started {substage.replace('_', ' ')} for {phase.replace('_', ' ')}.",
            "phase": phase,
            "substage": substage,
        }
    )


def log_substage_duration(
    logger: Any,
    phase: str,
    substage: str,
    duration_seconds: float,
    steps: int,
    **log_data: object,
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
    logger.info(
        {
            **log_data,
            "event": "substage_finished",
            "msg": (
                f"Finished {substage_labels.get(substage, substage.replace('_', ' '))} "
                f"for {phase_labels.get(phase, phase.replace('_', ' '))}."
            ),
            "phase": phase,
            "substage": substage,
        }
    )


def update_setup_metrics(
    metrics_obj: Any,
    backend_name: str,
    env_memory_profile: dict[str, Any],
    memory_profile: dict[str, Any],
) -> None:
    metrics_obj.set_memory_profile(
        state_bytes_total=int(memory_profile["state_bytes_total"]),
        state_bytes_per_particle=int(memory_profile["state_bytes_per_particle"]),
        process_memory_per_particle_estimate_bytes=int(memory_profile["process_memory_per_particle_estimate_bytes"]),
    )
    if backend_name == "mujoco-warp":
        metrics_obj.set_runtime_environment(
            execution_platform=str(env_memory_profile["execution_platform"]),
            execution_device=str(env_memory_profile["execution_device"]),
            default_jax_platform=str(env_memory_profile["default_jax_platform"]),
            default_jax_device=str(env_memory_profile["default_jax_device"]),
            device_fallback_applied=bool(env_memory_profile["device_fallback_applied"]),
        )
        return
    metrics_obj.set_mujoco_memory_profile(
        model_nbuffer_bytes_per_robot=int(env_memory_profile["model_nbuffer_bytes_per_robot"]),
        data_nbuffer_bytes_per_robot=int(env_memory_profile["data_nbuffer_bytes_per_robot"]),
        data_narena_bytes_per_robot=int(env_memory_profile["data_narena_bytes_per_robot"]),
        native_bytes_per_robot=int(env_memory_profile["native_bytes_per_robot"]),
        native_bytes_total=int(env_memory_profile["native_bytes_total"]),
    )


def update_warp_memory_metrics(env: Any, metrics_obj: Any, *, stage: str) -> None:
    env_memory_profile = env.memory_profile()
    metrics_obj.update_warp_memory(
        stage=stage,
        bytes_in_use=int(env_memory_profile["bytes_in_use"]),
        peak_bytes_in_use=int(env_memory_profile["peak_bytes_in_use"]),
        bytes_limit=int(env_memory_profile["bytes_limit"]),
        state_bytes_estimate=int(env_memory_profile.get("state_bytes_estimate", 0)),
    )


def tracked_stage(stage_name: str):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(ctx: ExecutionContext, *args, **kwargs):
            steps = kwargs.get("steps")
            stage_token = ctx.metrics.start_stage(stage_name)
            log_stage_started(ctx.logger, stage_name, steps=steps, **ctx.log_data)
            try:
                with tracing_span(ctx.tracer, stage_name):
                    set_span_attributes(stage_span_attrs(ctx, stage_name, steps=steps))
                    return func(ctx, *args, **kwargs)
            finally:
                duration = ctx.metrics.finish_stage(stage_token)
                log_stage_finished(ctx.logger, stage_name, duration, **ctx.log_data)

        return wrapper

    return decorator


def run_robot_motion_substage(
    ctx: ExecutionContext,
    phase: str,
    trajectory: list[np.ndarray] | np.ndarray,
    real_robot: Any,
    viewer: Any,
) -> float:
    with tracing_span(ctx.tracer, "robot_execute"):
        set_span_attributes(
            substage_span_attrs(
                ctx,
                phase,
                "robot_execute",
                steps=len(trajectory),
                execution_mode="sequential",
                parallel_units=1,
            )
        )
        stage_token = ctx.metrics.start_substage(phase, "robot_execute")
        log_substage_started(ctx.logger, phase, "robot_execute", steps=len(trajectory), **ctx.log_data)
        for _, qpos in enumerate(trajectory):
            real_robot.move_joints(qpos)
            if viewer is not None:
                viewer.sync()
        duration = ctx.metrics.finish_substage(stage_token)
        log_substage_duration(ctx.logger, phase, "robot_execute", duration, len(trajectory), **ctx.log_data)
        ctx.metrics.set_substage_workload(phase, "robot_execute", len(trajectory), 1, duration)
        return duration


def run_pf_replay_substage(
    ctx: ExecutionContext,
    phase: str,
    trajectory: list[np.ndarray] | np.ndarray,
    particle_filter: Any,
    use_batched_backend: bool,
) -> float:
    with tracing_span(ctx.tracer, "pf_replay"):
        set_span_attributes(
            substage_span_attrs(
                ctx,
                phase,
                "pf_replay",
                steps=len(trajectory),
                execution_mode="batched_parallel" if use_batched_backend else "sequential",
                parallel_units=particle_filter.N if use_batched_backend else 1,
            )
        )
        stage_token = ctx.metrics.start_substage(phase, "pf_replay")
        log_substage_started(ctx.logger, phase, "pf_replay", steps=len(trajectory), **ctx.log_data)
        if use_batched_backend:
            particle_filter.predict_trajectory(trajectory)
        else:
            for qpos in trajectory:
                particle_filter.predict(qpos)
        duration = ctx.metrics.finish_substage(stage_token)
        log_substage_duration(ctx.logger, phase, "pf_replay", duration, len(trajectory), **ctx.log_data)
        ctx.metrics.set_substage_workload(phase, "pf_replay", len(trajectory), particle_filter.N, duration)
        return duration


@tracked_stage("phase_1_approach")
def run_phase_1_approach(
    ctx: ExecutionContext,
    *,
    steps: int,
    trajectory: list[np.ndarray] | np.ndarray,
    real_robot: Any,
    viewer: Any,
    particle_filter: Any,
    use_batched_backend: bool,
    env: Any,
) -> None:
    run_robot_motion_substage(ctx, "phase_1_approach", trajectory, real_robot, viewer)
    run_pf_replay_substage(ctx, "phase_1_approach", trajectory, particle_filter, use_batched_backend)
    if ctx.backend == "mujoco-warp":
        update_warp_memory_metrics(env, ctx.metrics, stage="phase_1_approach")


@tracked_stage("phase_2_descend")
def run_phase_2_descend(
    ctx: ExecutionContext,
    *,
    steps: int,
    trajectory: list[np.ndarray] | np.ndarray,
    real_robot: Any,
    viewer: Any,
    particle_filter: Any,
    use_batched_backend: bool,
    env: Any,
) -> None:
    run_robot_motion_substage(ctx, "phase_2_descend", trajectory, real_robot, viewer)
    run_pf_replay_substage(ctx, "phase_2_descend", trajectory, particle_filter, use_batched_backend)
    if ctx.backend == "mujoco-warp":
        update_warp_memory_metrics(env, ctx.metrics, stage="phase_2_descend")


@tracked_stage("phase_3_grip")
def run_phase_3_grip(
    ctx: ExecutionContext,
    *,
    steps: int,
    trajectory: list[np.ndarray] | np.ndarray,
    real_robot: Any,
    viewer: Any,
    particle_filter: Any,
    use_batched_backend: bool,
    env: Any,
) -> None:
    run_robot_motion_substage(ctx, "phase_3_grip", trajectory, real_robot, viewer)
    run_pf_replay_substage(ctx, "phase_3_grip", trajectory, particle_filter, use_batched_backend)
    if ctx.backend == "mujoco-warp":
        update_warp_memory_metrics(env, ctx.metrics, stage="phase_3_grip")


@tracked_stage("phase_4_lift")
def run_phase_4_lift(
    ctx: ExecutionContext,
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
    history_estimates: list[float] = []
    pf_wall_durations: list[float] = []
    pf_cpu_durations: list[float] = []
    phase_4_robot_execute_total = 0.0
    phase_4_pf_update_total = 0.0
    phase_4_resample_count = 0
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
    latest_particles_snapshot = None

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
        uniform_weight_l1, uniform_weight_max_dev, collapsed_to_uniform = uniform_weight_metrics(
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

    for step, qpos in enumerate(trajectory):
        with tracing_span(ctx.tracer, "step"):
            set_span_attributes(
                {
                    "simbay.run_id": ctx.run_id,
                    "simbay.stage": "phase_4_lift",
                    "simbay.backend": ctx.backend,
                    "simbay.control_dt": ctx.dt,
                    "simbay.execution_device": ctx.execution_device,
                }
            )
            set_span_attributes(
                {
                    "simbay.step": step,
                    "simbay.execution_mode": ("batched_parallel" if ctx.backend == "mujoco-warp" else "sequential"),
                    "simbay.batched_particle_updates": (particle_filter.N if ctx.backend == "mujoco-warp" else 1),
                }
            )
            with tracing_span(ctx.tracer, "robot_execute"):
                set_span_attributes(
                    {
                        "simbay.run_id": ctx.run_id,
                        "simbay.stage": "phase_4_lift",
                        "simbay.substage": "robot_execute",
                        "simbay.backend": ctx.backend,
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
            with tracing_span(ctx.tracer, "pf_update"):
                previous_mass_estimate = float(particle_filter.estimate())
                set_span_attributes(
                    {
                        "simbay.run_id": ctx.run_id,
                        "simbay.stage": "phase_4_lift",
                        "simbay.substage": "pf_update",
                        "simbay.backend": ctx.backend,
                        "simbay.previous_mass_estimate_kg": previous_mass_estimate,
                        "simbay.execution_mode": ("batched_parallel" if ctx.backend == "mujoco-warp" else "sequential"),
                        "simbay.parallel_units": (particle_filter.N if ctx.backend == "mujoco-warp" else 1),
                    }
                )
                step_wall_start = time.perf_counter()
                step_cpu_start = time.process_time()
                measurements = real_robot.get_sensor_reads()
                real_ft_reading = measurements
                noisy_ft_reading = real_ft_reading + np.random.normal(0, 0.5, size=3)
                if ctx.backend == "mujoco-warp":
                    with annotate("phase4_particle_filter_step"):
                        if not phase_4_bootstrap_applied:
                            last_result = None
                            for attempt in range(1, 4):
                                step_result = _warp_step(qpos, noisy_ft_reading, step, attempt=attempt)
                                last_result = step_result
                                if not bool(step_result.get("skipped_invalid_update", False)):
                                    if attempt > 1:
                                        ctx.logger.info(
                                            {
                                                **ctx.log_data,
                                                "event": "warp_first_update_recovered",
                                                "msg": f"Recovered the first Warp update after {attempt} attempts.",
                                                "attempts": attempt,
                                                "step": particle_filter._step_index - 1,
                                            }
                                        )
                                    break
                            step_result = last_result if last_result is not None else _warp_step(
                                qpos, noisy_ft_reading, step
                            )
                            phase_4_bootstrap_applied = True
                        else:
                            step_result = _warp_step(qpos, noisy_ft_reading, step)
                else:
                    with annotate("phase4_particle_filter_step"):
                        particle_filter.predict(qpos)
                        particle_filter.update(noisy_ft_reading)
                        did_resample = particle_filter.effective_sample_size() < (particle_filter.N / 2)
                        if did_resample or ctx.backend != "cpu":
                            particle_filter.resample()
                        step_result = {
                            "ess": particle_filter.effective_sample_size(),
                            "resampled": did_resample,
                            "resample_count": getattr(
                                particle_filter,
                                "_resample_count",
                                phase_4_resample_count + int(did_resample),
                            ),
                            "uniform_weight_l1_distance": 0.0,
                            "uniform_weight_max_deviation": 0.0,
                            "collapsed_to_uniform": False,
                        }
                set_span_attributes({"simbay.new_mass_estimate_kg": float(particle_filter.estimate())})
        current_estimate = float(particle_filter.estimate())
        history_estimates.append(current_estimate)
        if hasattr(particle_filter, "particles"):
            latest_particles_snapshot = np.asarray(particle_filter.particles).copy()
        if time_to_first_estimate_seconds < 0.0:
            time_to_first_estimate_seconds = ctx.runtime.elapsed_seconds()
        abs_error_kg = abs(current_estimate - true_mass)
        rel_error_pct = (abs_error_kg / true_mass * 100.0) if true_mass != 0 else 0.0
        phase_4_abs_error_sum += abs_error_kg
        phase_4_squared_error_sum += abs_error_kg**2
        phase_4_mae_kg = phase_4_abs_error_sum / (step + 1)
        phase_4_rmse_kg = math.sqrt(phase_4_squared_error_sum / (step + 1))
        elapsed_since_run_start = ctx.runtime.elapsed_seconds()
        if convergence_time_to_10pct_seconds < 0.0 and rel_error_pct <= 10.0:
            convergence_time_to_10pct_seconds = elapsed_since_run_start
        if convergence_time_to_5pct_seconds < 0.0 and rel_error_pct <= 5.0:
            convergence_time_to_5pct_seconds = elapsed_since_run_start
        step_wall_duration = time.perf_counter() - step_wall_start
        step_cpu_duration = time.process_time() - step_cpu_start
        phase_4_pf_update_total += step_wall_duration
        pf_wall_durations.append(step_wall_duration)
        pf_cpu_durations.append(step_cpu_duration)
        cpu_equivalent_cores_used = step_cpu_duration / step_wall_duration if step_wall_duration > 0 else 0.0
        _ = add_exemplar(ctx.run_id, step)
        ctx.metrics.update_filter_state(
            ess=particle_filter.effective_sample_size(),
            estimate=current_estimate,
            wall_seconds=step_wall_duration,
            cpu_seconds=step_cpu_duration,
            cpu_equivalent_cores=cpu_equivalent_cores_used,
            particles=particle_filter.N,
        )
        phase_4_resample_count = int(step_result.get("resample_count", phase_4_resample_count))
        ctx.metrics.update_weight_health(
            uniform_weight_l1_distance=float(step_result.get("uniform_weight_l1_distance", 0.0)),
            uniform_weight_max_deviation=float(step_result.get("uniform_weight_max_deviation", 0.0)),
            collapsed_to_uniform=bool(step_result.get("collapsed_to_uniform", False)),
        )
        ctx.metrics.update_accuracy_metrics(
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
            ci50_low = weighted_quantile(particle_values, weights_snapshot, 0.25)
            ci50_high = weighted_quantile(particle_values, weights_snapshot, 0.75)
            ci90_low = weighted_quantile(particle_values, weights_snapshot, 0.05)
            ci90_high = weighted_quantile(particle_values, weights_snapshot, 0.95)
            safe_weights = np.clip(weights_snapshot, np.finfo(np.float64).tiny, 1.0)
            weight_entropy = float(-np.sum(safe_weights * np.log(safe_weights)))
            max_entropy = math.log(len(safe_weights)) if len(safe_weights) > 0 else 0.0
            weight_entropy_normalized = float(weight_entropy / max_entropy) if max_entropy > 0.0 else 0.0
            weight_perplexity = float(np.exp(weight_entropy))
            ctx.metrics.update_resample_state(
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
            ctx.metrics.update_uncertainty_metrics(
                credible_interval_50_width_kg=ci50_high - ci50_low,
                credible_interval_90_width_kg=ci90_high - ci90_low,
                credible_interval_50_contains_truth=ci50_low <= true_mass <= ci50_high,
                credible_interval_90_contains_truth=ci90_low <= true_mass <= ci90_high,
                weight_entropy=weight_entropy,
                weight_entropy_normalized=weight_entropy_normalized,
                weight_perplexity=weight_perplexity,
            )
        if ctx.backend == "mujoco-warp":
            diagnostics = step_result.get("diagnostics", {})
            phase_4_invalid_sensor_events = max(phase_4_invalid_sensor_events, int(diagnostics.get("invalid_sensor_events", 0.0)))
            phase_4_invalid_state_events = max(phase_4_invalid_state_events, int(diagnostics.get("invalid_state_events", 0.0)))
            phase_4_skipped_invalid_updates = max(
                phase_4_skipped_invalid_updates, int(step_result.get("skipped_invalid_updates", 0))
            )
            current_first_invalid_sensor_step = int(diagnostics.get("first_invalid_sensor_step", -1.0))
            current_first_invalid_state_step = int(diagnostics.get("first_invalid_state_step", -1.0))
            if phase_4_first_invalid_sensor_step < 0 and current_first_invalid_sensor_step >= 0:
                phase_4_first_invalid_sensor_step = current_first_invalid_sensor_step
            if phase_4_first_invalid_state_step < 0 and current_first_invalid_state_step >= 0:
                phase_4_first_invalid_state_step = current_first_invalid_state_step
            phase_4_max_repaired_world_count = max(
                phase_4_max_repaired_world_count, int(diagnostics.get("repaired_world_count", 0.0))
            )
            ctx.metrics.update_likelihood_health(
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
            ctx.metrics.update_invalid_state_counts(
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
            ctx.metrics.update_contact_health(
                contact_count_mean=float(diagnostics.get("contact_count_mean", 0.0)),
                contact_count_max=float(diagnostics.get("contact_count_max", 0.0)),
                active_contact_particle_ratio=float(diagnostics.get("active_contact_particle_ratio", 0.0)),
                contact_metric_available=bool(diagnostics.get("contact_metric_available", 0.0)),
                contact_force_mismatch=bool(diagnostics.get("contact_force_mismatch", 0.0)),
                valid_force_particle_ratio=float(diagnostics.get("valid_force_particle_ratio", 0.0)),
                sim_force_signal_particle_ratio=float(diagnostics.get("sim_force_signal_particle_ratio", 0.0)),
            )
        set_span_attributes(
            {
                "simbay.ess": float(particle_filter.effective_sample_size()),
                "simbay.resampled": bool(step_result.get("resampled", False)),
                "simbay.mass_estimate_kg": current_estimate,
                "simbay.step_wall_ms": step_wall_duration * 1000.0,
            }
        )
    if ctx.backend == "mujoco-warp":
        update_warp_memory_metrics(env, ctx.metrics, stage="phase_4_lift")
    ctx.metrics.set_substage_duration("phase_4_lift", "robot_execute", phase_4_robot_execute_total)
    ctx.metrics.set_substage_duration("phase_4_lift", "pf_update", phase_4_pf_update_total)
    log_substage_started(ctx.logger, "phase_4_lift", "robot_execute", steps=len(trajectory), **ctx.log_data)
    log_substage_duration(
        ctx.logger,
        "phase_4_lift",
        "robot_execute",
        phase_4_robot_execute_total,
        len(trajectory),
        **ctx.log_data,
    )
    log_substage_started(ctx.logger, "phase_4_lift", "pf_update", steps=len(trajectory), **ctx.log_data)
    log_substage_duration(
        ctx.logger,
        "phase_4_lift",
        "pf_update",
        phase_4_pf_update_total,
        len(trajectory),
        **ctx.log_data,
    )
    ctx.metrics.set_substage_workload("phase_4_lift", "robot_execute", len(trajectory), 1, phase_4_robot_execute_total)
    ctx.metrics.set_substage_workload(
        "phase_4_lift",
        "pf_update",
        len(trajectory),
        particle_filter.N,
        phase_4_pf_update_total,
    )
    force_flush_tracing()
    return LiftPhaseResult(
        history_estimates=history_estimates,
        pf_wall_durations=pf_wall_durations,
        pf_cpu_durations=pf_cpu_durations,
        invalid_sensor_events=phase_4_invalid_sensor_events,
        invalid_state_events=phase_4_invalid_state_events,
        skipped_invalid_updates=phase_4_skipped_invalid_updates,
        first_invalid_sensor_step=phase_4_first_invalid_sensor_step,
        first_invalid_state_step=phase_4_first_invalid_state_step,
        max_repaired_world_count=phase_4_max_repaired_world_count,
    )
