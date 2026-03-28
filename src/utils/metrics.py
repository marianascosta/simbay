import math
import inspect
import os
import subprocess
import threading
import time
from contextlib import suppress
from dataclasses import dataclass
from socketserver import ThreadingMixIn
from typing import Any
from typing import Iterable
from functools import wraps
from wsgiref.simple_server import WSGIServer
from wsgiref.simple_server import make_server

import numpy as np
from prometheus_client import CollectorRegistry
from prometheus_client import Gauge
from prometheus_client import make_wsgi_app

from .logging_utils import get_process_memory_bytes
from .settings import SYSTEM_METRICS_INTERVAL_SECONDS
from .tracing import add_exemplar
from .tracing import force_flush_tracing
from .tracing import set_span_attributes

DEFAULT_STAGE_NAMES: tuple[str, ...] = (
    "setup",
    "ik_planning",
    "phase_1_approach",
    "phase_2_descend",
    "phase_3_grip",
    "phase_4_lift",
    "plot_generation",
)

DEFAULT_SUBSTAGE_NAMES: tuple[tuple[str, str], ...] = (
    ("phase_1_approach", "robot_execute"),
    ("phase_1_approach", "pf_replay"),
    ("phase_2_descend", "robot_execute"),
    ("phase_2_descend", "pf_replay"),
    ("phase_3_grip", "robot_execute"),
    ("phase_3_grip", "pf_replay"),
    ("phase_4_lift", "robot_execute"),
    ("phase_4_lift", "pf_update"),
)

COMMON_LABELS = ("run_id",)
STAGE_LABELS = ("run_id", "stage")
SUBSTAGE_LABELS = ("run_id", "phase", "substage")


@dataclass(frozen=True)
class StageToken:
    stage: str
    started_at: float


@dataclass(frozen=True)
class SubstageToken:
    phase: str
    substage: str
    started_at: float


class _ThreadingWSGIServer(ThreadingMixIn, WSGIServer):
    daemon_threads = True

    def log_message(self, format: str, *args: Any) -> None:
        return None


class SimbayMetrics:
    def __init__(self, enabled: bool, port: int, run_id: str = "unknown") -> None:
        self.enabled = enabled
        self.port = port
        self.run_id = run_id
        self.registry = CollectorRegistry()
        self._host_cpu_count = max(os.cpu_count() or 1, 1)
        self._server = None
        self._server_thread: threading.Thread | None = None
        self._system_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._known_stages: set[str] = set()
        self._known_substages: set[tuple[str, str]] = set()

        self.backend_info = Gauge(
            "simbay_backend_info",
            "Resolved simulation backend and execution device.",
            (*COMMON_LABELS, "backend", "device"),
            registry=self.registry,
        )
        self.run_info = Gauge(
            "simbay_run_info",
            "Metadata for the current run.",
            (*COMMON_LABELS, "backend", "particles", "control_dt"),
            registry=self.registry,
        )
        self.particle_count = Gauge(
            "simbay_particle_count",
            "Configured particle count for the current run.",
            COMMON_LABELS,
            registry=self.registry,
        )
        self.runtime_environment_info = Gauge(
            "simbay_runtime_environment_info",
            "Resolved execution runtime environment for the current run.",
            (*COMMON_LABELS, "execution_platform", "execution_device", "default_jax_platform", "default_jax_device"),
            registry=self.registry,
        )
        self.device_fallback_applied = Gauge(
            "simbay_device_fallback_applied",
            "Whether runtime device fallback was applied during setup.",
            COMMON_LABELS,
            registry=self.registry,
        )
        self.state_memory_total_bytes = Gauge(
            "simbay_state_memory_total_bytes",
            "Estimated total bytes used by particle state storage.",
            COMMON_LABELS,
            registry=self.registry,
        )
        self.state_memory_per_particle_bytes = Gauge(
            "simbay_state_memory_per_particle_bytes",
            "Estimated bytes used by particle state storage per particle.",
            COMMON_LABELS,
            registry=self.registry,
        )
        self.process_memory_per_particle_estimate_bytes = Gauge(
            "simbay_process_memory_per_particle_estimate_bytes",
            "Estimated process memory usage attributable to each particle.",
            COMMON_LABELS,
            registry=self.registry,
        )
        self.mujoco_model_buffer_per_particle_bytes = Gauge(
            "simbay_mujoco_model_buffer_per_particle_bytes",
            "MuJoCo model buffer bytes per particle.",
            COMMON_LABELS,
            registry=self.registry,
        )
        self.mujoco_data_buffer_per_particle_bytes = Gauge(
            "simbay_mujoco_data_buffer_per_particle_bytes",
            "MuJoCo data buffer bytes per particle.",
            COMMON_LABELS,
            registry=self.registry,
        )
        self.mujoco_data_arena_per_particle_bytes = Gauge(
            "simbay_mujoco_data_arena_per_particle_bytes",
            "MuJoCo data arena bytes per particle.",
            COMMON_LABELS,
            registry=self.registry,
        )
        self.mujoco_native_memory_per_particle_bytes = Gauge(
            "simbay_mujoco_native_memory_per_particle_bytes",
            "Estimated native MuJoCo memory per particle.",
            COMMON_LABELS,
            registry=self.registry,
        )
        self.mujoco_native_memory_total_bytes = Gauge(
            "simbay_mujoco_native_memory_total_bytes",
            "Estimated total native MuJoCo memory across all particles.",
            COMMON_LABELS,
            registry=self.registry,
        )
        self.stage_active = Gauge(
            "simbay_stage_active",
            "Whether a simulation stage is currently active.",
            STAGE_LABELS,
            registry=self.registry,
        )
        self.stage_duration_seconds = Gauge(
            "simbay_stage_duration_seconds",
            "Wall-clock duration for each simulation stage.",
            STAGE_LABELS,
            registry=self.registry,
        )
        self.substage_active = Gauge(
            "simbay_substage_active",
            "Whether a simulation substage is currently active.",
            SUBSTAGE_LABELS,
            registry=self.registry,
        )
        self.substage_duration_seconds = Gauge(
            "simbay_substage_duration_seconds",
            "Wall-clock duration for each simulation substage.",
            SUBSTAGE_LABELS,
            registry=self.registry,
        )
        self.substage_steps = Gauge(
            "simbay_substage_steps",
            "Workload size in discrete control steps for each substage.",
            SUBSTAGE_LABELS,
            registry=self.registry,
        )
        self.substage_step_rate_hz = Gauge(
            "simbay_substage_step_rate_hz",
            "Discrete control steps processed per second for each substage.",
            SUBSTAGE_LABELS,
            registry=self.registry,
        )
        self.substage_ms_per_step = Gauge(
            "simbay_substage_ms_per_step",
            "Milliseconds spent per control step for each substage.",
            SUBSTAGE_LABELS,
            registry=self.registry,
        )
        self.substage_particle_steps_per_second = Gauge(
            "simbay_substage_particle_steps_per_second",
            "Particle-steps processed per second for each substage.",
            SUBSTAGE_LABELS,
            registry=self.registry,
        )
        self.substage_ms_per_particle_step = Gauge(
            "simbay_substage_ms_per_particle_step",
            "Milliseconds spent per particle-step for each substage.",
            SUBSTAGE_LABELS,
            registry=self.registry,
        )
        self.process_rss_bytes = Gauge(
            "simbay_process_rss_bytes",
            "Resident set size of the Simbay process.",
            STAGE_LABELS,
            registry=self.registry,
        )
        self.effective_sample_size = Gauge(
            "simbay_effective_sample_size",
            "Current effective sample size of the particle filter.",
            COMMON_LABELS,
            registry=self.registry,
        )
        self.mass_estimate_kg = Gauge(
            "simbay_mass_estimate_kg",
            "Current mass estimate in kilograms.",
            COMMON_LABELS,
            registry=self.registry,
        )
        self.phase4_step_wall_seconds = Gauge(
            "simbay_phase4_step_wall_seconds",
            "Wall-clock duration of the latest lift/update step.",
            COMMON_LABELS,
            registry=self.registry,
        )
        self.phase4_step_cpu_seconds = Gauge(
            "simbay_phase4_step_cpu_seconds",
            "CPU duration of the latest lift/update step.",
            COMMON_LABELS,
            registry=self.registry,
        )
        self.phase4_cpu_equivalent_cores = Gauge(
            "simbay_phase4_cpu_equivalent_cores",
            "Approximate CPU core usage during the latest lift/update step.",
            COMMON_LABELS,
            registry=self.registry,
        )
        self.phase4_step_rate_hz = Gauge(
            "simbay_phase4_step_rate_hz",
            "Lift/update control steps processed per second.",
            COMMON_LABELS,
            registry=self.registry,
        )
        self.phase4_particle_steps_per_second = Gauge(
            "simbay_phase4_particle_steps_per_second",
            "Lift/update particle-steps processed per second.",
            COMMON_LABELS,
            registry=self.registry,
        )
        self.phase4_ms_per_particle_step = Gauge(
            "simbay_phase4_ms_per_particle_step",
            "Milliseconds spent per particle-step during the latest lift/update step.",
            COMMON_LABELS,
            registry=self.registry,
        )
        self.mass_abs_error_kg = Gauge(
            "simbay_mass_abs_error_kg",
            "Absolute difference between the current mass estimate and the true mass.",
            COMMON_LABELS,
            registry=self.registry,
        )
        self.mass_rel_error_pct = Gauge(
            "simbay_mass_rel_error_pct",
            "Relative percent error of the current mass estimate.",
            COMMON_LABELS,
            registry=self.registry,
        )
        self.phase4_mae_kg = Gauge(
            "simbay_phase4_mae_kg",
            "Running mean absolute error across phase-4 updates.",
            COMMON_LABELS,
            registry=self.registry,
        )
        self.phase4_rmse_kg = Gauge(
            "simbay_phase4_rmse_kg",
            "Running root mean squared error across phase-4 updates.",
            COMMON_LABELS,
            registry=self.registry,
        )
        self.mass_error_within_1pct = Gauge(
            "simbay_mass_error_within_1pct",
            "Whether the latest mass estimate is within 1 percent of the true mass.",
            COMMON_LABELS,
            registry=self.registry,
        )
        self.mass_error_within_5pct = Gauge(
            "simbay_mass_error_within_5pct",
            "Whether the latest mass estimate is within 5 percent of the true mass.",
            COMMON_LABELS,
            registry=self.registry,
        )
        self.mass_error_within_10pct = Gauge(
            "simbay_mass_error_within_10pct",
            "Whether the latest mass estimate is within 10 percent of the true mass.",
            COMMON_LABELS,
            registry=self.registry,
        )
        self.convergence_time_to_5pct_seconds = Gauge(
            "simbay_convergence_time_to_5pct_seconds",
            "Time until the mass estimate first reached 5 percent relative error or better, or -1 if not reached.",
            COMMON_LABELS,
            registry=self.registry,
        )
        self.convergence_time_to_10pct_seconds = Gauge(
            "simbay_convergence_time_to_10pct_seconds",
            "Time until the mass estimate first reached 10 percent relative error or better, or -1 if not reached.",
            COMMON_LABELS,
            registry=self.registry,
        )
        self.time_to_first_estimate_seconds = Gauge(
            "simbay_time_to_first_estimate_seconds",
            "Time from run start until the first phase-4 estimate was produced.",
            COMMON_LABELS,
            registry=self.registry,
        )
        self.credible_interval_50_width_kg = Gauge(
            "simbay_credible_interval_50_width_kg",
            "Width of the central 50 percent credible interval over particle mass.",
            COMMON_LABELS,
            registry=self.registry,
        )
        self.credible_interval_90_width_kg = Gauge(
            "simbay_credible_interval_90_width_kg",
            "Width of the central 90 percent credible interval over particle mass.",
            COMMON_LABELS,
            registry=self.registry,
        )
        self.credible_interval_50_contains_truth = Gauge(
            "simbay_credible_interval_50_contains_truth",
            "Whether the true mass lies inside the central 50 percent credible interval.",
            COMMON_LABELS,
            registry=self.registry,
        )
        self.credible_interval_90_contains_truth = Gauge(
            "simbay_credible_interval_90_contains_truth",
            "Whether the true mass lies inside the central 90 percent credible interval.",
            COMMON_LABELS,
            registry=self.registry,
        )
        self.weight_entropy = Gauge(
            "simbay_weight_entropy",
            "Shannon entropy of the current particle-weight distribution.",
            COMMON_LABELS,
            registry=self.registry,
        )
        self.weight_entropy_normalized = Gauge(
            "simbay_weight_entropy_normalized",
            "Particle-weight entropy normalized by the maximum possible entropy.",
            COMMON_LABELS,
            registry=self.registry,
        )
        self.weight_perplexity = Gauge(
            "simbay_weight_perplexity",
            "Effective number of weight states implied by the current particle-weight distribution.",
            COMMON_LABELS,
            registry=self.registry,
        )
        self.resample_count = Gauge(
            "simbay_resample_count",
            "Total number of particle-filter resampling events in the current run.",
            COMMON_LABELS,
            registry=self.registry,
        )
        self.resample_rate = Gauge(
            "simbay_resample_rate",
            "Fraction of particle-filter update steps that triggered resampling.",
            COMMON_LABELS,
            registry=self.registry,
        )
        self.last_step_resampled = Gauge(
            "simbay_last_step_resampled",
            "Whether the latest particle-filter update step triggered resampling.",
            COMMON_LABELS,
            registry=self.registry,
        )
        self.particle_mass_min_kg = Gauge(
            "simbay_particle_mass_min_kg",
            "Minimum particle mass in kilograms for the latest particle cloud.",
            COMMON_LABELS,
            registry=self.registry,
        )
        self.particle_mass_max_kg = Gauge(
            "simbay_particle_mass_max_kg",
            "Maximum particle mass in kilograms for the latest particle cloud.",
            COMMON_LABELS,
            registry=self.registry,
        )
        self.particle_mass_mean_kg = Gauge(
            "simbay_particle_mass_mean_kg",
            "Mean particle mass in kilograms for the latest particle cloud.",
            COMMON_LABELS,
            registry=self.registry,
        )
        self.particle_mass_std_kg = Gauge(
            "simbay_particle_mass_std_kg",
            "Standard deviation of particle mass in kilograms for the latest particle cloud.",
            COMMON_LABELS,
            registry=self.registry,
        )
        self.particle_mass_p10_kg = Gauge(
            "simbay_particle_mass_p10_kg",
            "10th percentile of particle mass in kilograms for the latest particle cloud.",
            COMMON_LABELS,
            registry=self.registry,
        )
        self.particle_mass_p50_kg = Gauge(
            "simbay_particle_mass_p50_kg",
            "50th percentile of particle mass in kilograms for the latest particle cloud.",
            COMMON_LABELS,
            registry=self.registry,
        )
        self.particle_mass_p90_kg = Gauge(
            "simbay_particle_mass_p90_kg",
            "90th percentile of particle mass in kilograms for the latest particle cloud.",
            COMMON_LABELS,
            registry=self.registry,
        )
        self.sim_force_finite_ratio = Gauge(
            "simbay_sim_force_finite_ratio",
            "Fraction of simulated force samples that are finite.",
            COMMON_LABELS,
            registry=self.registry,
        )
        self.measurement_residual_finite_ratio = Gauge(
            "simbay_measurement_residual_finite_ratio",
            "Fraction of measurement residual samples that are finite.",
            COMMON_LABELS,
            registry=self.registry,
        )
        self.likelihood_finite_ratio = Gauge(
            "simbay_likelihood_finite_ratio",
            "Fraction of likelihood samples that are finite.",
            COMMON_LABELS,
            registry=self.registry,
        )
        self.sim_force_norm_mean = Gauge(
            "simbay_sim_force_norm_mean",
            "Mean norm of simulated force samples for the latest likelihood evaluation.",
            COMMON_LABELS,
            registry=self.registry,
        )
        self.measurement_residual_norm_mean = Gauge(
            "simbay_measurement_residual_norm_mean",
            "Mean norm of measurement residuals for the latest likelihood evaluation.",
            COMMON_LABELS,
            registry=self.registry,
        )
        self.likelihood_min = Gauge(
            "simbay_likelihood_min",
            "Minimum particle likelihood from the latest likelihood evaluation.",
            COMMON_LABELS,
            registry=self.registry,
        )
        self.likelihood_max = Gauge(
            "simbay_likelihood_max",
            "Maximum particle likelihood from the latest likelihood evaluation.",
            COMMON_LABELS,
            registry=self.registry,
        )
        self.likelihood_mean = Gauge(
            "simbay_likelihood_mean",
            "Mean particle likelihood from the latest likelihood evaluation.",
            COMMON_LABELS,
            registry=self.registry,
        )
        self.likelihood_std = Gauge(
            "simbay_likelihood_std",
            "Standard deviation of particle likelihoods from the latest likelihood evaluation.",
            COMMON_LABELS,
            registry=self.registry,
        )
        self.invalid_sensor_events_total = Gauge(
            "simbay_invalid_sensor_events_total",
            "Cumulative count of likelihood evaluations that produced non-finite sensor-derived values.",
            COMMON_LABELS,
            registry=self.registry,
        )
        self.invalid_state_events_total = Gauge(
            "simbay_invalid_state_events_total",
            "Cumulative count of Warp state snapshots with non-finite entries.",
            COMMON_LABELS,
            registry=self.registry,
        )
        self.skipped_invalid_updates_total = Gauge(
            "simbay_skipped_invalid_updates_total",
            "Cumulative count of Warp filter updates skipped because the likelihood batch or backend state was invalid.",
            COMMON_LABELS,
            registry=self.registry,
        )
        self.skipped_invalid_update = Gauge(
            "simbay_skipped_invalid_update",
            "Whether the latest Warp filter update was skipped because the likelihood batch or backend state was invalid.",
            COMMON_LABELS,
            registry=self.registry,
        )
        self.first_update_bootstrap_attempts = Gauge(
            "simbay_first_update_bootstrap_attempts",
            "Number of attempts used to obtain a valid first Warp phase-4 measurement update.",
            COMMON_LABELS,
            registry=self.registry,
        )
        self.first_invalid_sensor_step = Gauge(
            "simbay_first_invalid_sensor_step",
            "First Warp filter step index that produced a non-finite sensor-derived value, or -1 if none.",
            COMMON_LABELS,
            registry=self.registry,
        )
        self.first_invalid_state_step = Gauge(
            "simbay_first_invalid_state_step",
            "First Warp filter step index that produced a non-finite backend state value, or -1 if none.",
            COMMON_LABELS,
            registry=self.registry,
        )
        self.sim_force_nonfinite_count = Gauge(
            "simbay_sim_force_nonfinite_count",
            "Count of non-finite simulated force entries from the latest likelihood evaluation.",
            COMMON_LABELS,
            registry=self.registry,
        )
        self.measurement_residual_nonfinite_count = Gauge(
            "simbay_measurement_residual_nonfinite_count",
            "Count of non-finite measurement residual entries from the latest likelihood evaluation.",
            COMMON_LABELS,
            registry=self.registry,
        )
        self.likelihood_nonfinite_count = Gauge(
            "simbay_likelihood_nonfinite_count",
            "Count of non-finite particle likelihoods from the latest likelihood evaluation.",
            COMMON_LABELS,
            registry=self.registry,
        )
        self.qpos_nonfinite_count = Gauge(
            "simbay_qpos_nonfinite_count",
            "Count of non-finite Warp qpos entries in the latest sampled state snapshot.",
            COMMON_LABELS,
            registry=self.registry,
        )
        self.qvel_nonfinite_count = Gauge(
            "simbay_qvel_nonfinite_count",
            "Count of non-finite Warp qvel entries in the latest sampled state snapshot.",
            COMMON_LABELS,
            registry=self.registry,
        )
        self.sensordata_nonfinite_count = Gauge(
            "simbay_sensordata_nonfinite_count",
            "Count of non-finite Warp sensordata entries in the latest sampled state snapshot.",
            COMMON_LABELS,
            registry=self.registry,
        )
        self.ctrl_nonfinite_count = Gauge(
            "simbay_ctrl_nonfinite_count",
            "Count of non-finite Warp ctrl entries in the latest sampled state snapshot.",
            COMMON_LABELS,
            registry=self.registry,
        )
        self.contact_count_mean = Gauge(
            "simbay_contact_count_mean",
            "Mean contact count across Warp particles for the latest likelihood evaluation.",
            COMMON_LABELS,
            registry=self.registry,
        )
        self.contact_count_max = Gauge(
            "simbay_contact_count_max",
            "Maximum contact count across Warp particles for the latest likelihood evaluation.",
            COMMON_LABELS,
            registry=self.registry,
        )
        self.active_contact_particle_ratio = Gauge(
            "simbay_active_contact_particle_ratio",
            "Fraction of Warp particles with at least one active contact.",
            COMMON_LABELS,
            registry=self.registry,
        )
        self.contact_metric_available = Gauge(
            "simbay_contact_metric_available",
            "Whether Warp exposed a contact-count array for the latest likelihood evaluation.",
            COMMON_LABELS,
            registry=self.registry,
        )
        self.contact_force_mismatch = Gauge(
            "simbay_contact_force_mismatch",
            "Whether the latest likelihood evaluation had non-zero force signal but zero reported contacts.",
            COMMON_LABELS,
            registry=self.registry,
        )
        self.valid_force_particle_ratio = Gauge(
            "simbay_valid_force_particle_ratio",
            "Fraction of particles with finite force vectors in the latest likelihood evaluation.",
            COMMON_LABELS,
            registry=self.registry,
        )
        self.force_signal_particle_ratio = Gauge(
            "simbay_force_signal_particle_ratio",
            "Fraction of particles with non-trivial simulated force norm in the latest likelihood evaluation.",
            COMMON_LABELS,
            registry=self.registry,
        )
        self.uniform_weight_l1_distance = Gauge(
            "simbay_uniform_weight_l1_distance",
            "L1 distance between current particle weights and a perfectly uniform distribution.",
            COMMON_LABELS,
            registry=self.registry,
        )
        self.uniform_weight_max_deviation = Gauge(
            "simbay_uniform_weight_max_deviation",
            "Maximum per-particle absolute deviation from uniform weights.",
            COMMON_LABELS,
            registry=self.registry,
        )
        self.likelihood_collapsed_to_uniform = Gauge(
            "simbay_likelihood_collapsed_to_uniform",
            "Whether the latest particle-weight update remained effectively uniform.",
            COMMON_LABELS,
            registry=self.registry,
        )
        self.warp_bytes_in_use = Gauge(
            "simbay_warp_bytes_in_use",
            "Observed execution-device bytes used when Warp reports free/total memory, otherwise a tracked Warp state-byte estimate.",
            STAGE_LABELS,
            registry=self.registry,
        )
        self.warp_peak_bytes_in_use = Gauge(
            "simbay_warp_peak_bytes_in_use",
            "Peak observed execution-device bytes used during the run, or peak tracked Warp state-byte estimate when device memory usage is unavailable.",
            STAGE_LABELS,
            registry=self.registry,
        )
        self.warp_bytes_limit = Gauge(
            "simbay_warp_bytes_limit",
            "Warp execution-device total memory in bytes when available.",
            STAGE_LABELS,
            registry=self.registry,
        )
        self.warp_state_bytes_estimate = Gauge(
            "simbay_warp_state_bytes_estimate",
            "Tracked lower-bound byte estimate for Warp model/data arrays and recovery snapshots.",
            STAGE_LABELS,
            registry=self.registry,
        )
        self.prediction_ready_seconds = Gauge(
            "simbay_prediction_ready_seconds",
            "Total wall-clock time until a final prediction is available.",
            COMMON_LABELS,
            registry=self.registry,
        )
        self.final_error_pct = Gauge(
            "simbay_final_error_pct",
            "Final percent error of the predicted mass.",
            COMMON_LABELS,
            registry=self.registry,
        )
        self.process_memory_rss_bytes = Gauge(
            "simbay_process_memory_rss_bytes",
            "Resident set size of the Simbay process.",
            COMMON_LABELS,
            registry=self.registry,
        )
        self.host_memory_used_bytes = Gauge(
            "simbay_host_memory_used_bytes",
            "Host memory currently in use for this benchmark run.",
            COMMON_LABELS,
            registry=self.registry,
        )
        self.host_cpu_utilization_pct = Gauge(
            "simbay_host_cpu_utilization_pct",
            "Host CPU utilisation percentage for this benchmark run.",
            COMMON_LABELS,
            registry=self.registry,
        )
        self.process_cpu_utilization_pct = Gauge(
            "simbay_process_cpu_utilization_pct",
            "CPU utilisation percentage of the Simbay process, expressed relative to one full core.",
            COMMON_LABELS,
            registry=self.registry,
        )
        self.process_cpu_machine_pct = Gauge(
            "simbay_process_cpu_machine_pct",
            "CPU utilisation percentage of the Simbay process, expressed relative to total machine CPU capacity.",
            COMMON_LABELS,
            registry=self.registry,
        )
        self.gpu_utilization_pct = Gauge(
            "simbay_gpu_utilization_pct",
            "GPU utilisation percentage for this benchmark run.",
            COMMON_LABELS,
            registry=self.registry,
        )
        self.gpu_fb_used_bytes = Gauge(
            "simbay_gpu_fb_used_bytes",
            "GPU frame-buffer memory currently in use for this benchmark run.",
            COMMON_LABELS,
            registry=self.registry,
        )
        self.gpu_fb_utilization_pct = Gauge(
            "simbay_gpu_fb_utilization_pct",
            "GPU frame-buffer memory utilisation percentage for this benchmark run.",
            COMMON_LABELS,
            registry=self.registry,
        )
        self.gpu_power_watts = Gauge(
            "simbay_gpu_power_watts",
            "GPU power draw in watts for this benchmark run.",
            COMMON_LABELS,
            registry=self.registry,
        )
        self.gpu_temp_celsius = Gauge(
            "simbay_gpu_temp_celsius",
            "GPU temperature in degrees Celsius for this benchmark run.",
            COMMON_LABELS,
            registry=self.registry,
        )
        self.gpu_sm_clock_hz = Gauge(
            "simbay_gpu_sm_clock_hz",
            "GPU SM clock frequency in hertz for this benchmark run.",
            COMMON_LABELS,
            registry=self.registry,
        )
        self.gpu_mem_clock_hz = Gauge(
            "simbay_gpu_mem_clock_hz",
            "GPU memory clock frequency in hertz for this benchmark run.",
            COMMON_LABELS,
            registry=self.registry,
        )

    def _common(self) -> tuple[str]:
        return (self.run_id,)

    def _stage(self, stage: str) -> tuple[str, str]:
        return (self.run_id, stage)

    def _substage(self, phase: str, substage: str) -> tuple[str, str, str]:
        return (self.run_id, phase, substage)

    def start(self) -> None:
        if not self.enabled or self._server is not None:
            return
        try:
            app = make_wsgi_app(self.registry)
            self._server = make_server("0.0.0.0", self.port, app, server_class=_ThreadingWSGIServer)
        except OSError:
            self._server = None
            return
        self._stop_event.clear()
        self._server_thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._server_thread.start()
        self._sample_system_metrics(previous_cpu_totals=None, previous_process_totals=None)
        self._system_thread = threading.Thread(target=self._run_system_sampler, daemon=True)
        self._system_thread.start()

    def initialize_defaults(self) -> "SimbayMetrics":
        self.register_stages(DEFAULT_STAGE_NAMES)
        self.register_substages(DEFAULT_SUBSTAGE_NAMES)
        return self

    def start_runtime(self) -> "SimbayMetrics":
        self.start()
        return self

    def stop(self) -> None:
        self._stop_event.set()
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._server_thread is not None:
            self._server_thread.join()
        if self._system_thread is not None:
            self._system_thread.join()
        self._server = None
        self._server_thread = None
        self._system_thread = None

    def register_stages(self, stages: Iterable[str]) -> None:
        for stage in stages:
            self._known_stages.add(stage)
            self.stage_active.labels(*self._stage(stage)).set(0.0)
            self.stage_duration_seconds.labels(*self._stage(stage)).set(0.0)

    def register_substages(self, substages: Iterable[tuple[str, str]]) -> None:
        for phase, substage in substages:
            self._known_substages.add((phase, substage))
            self.substage_active.labels(*self._substage(phase, substage)).set(0.0)
            self.substage_duration_seconds.labels(*self._substage(phase, substage)).set(0.0)

    def set_backend(self, backend: str, device: str) -> None:
        self.backend_info.labels(*self._common(), backend, device).set(1.0)

    def set_run_info(self, backend: str, particles: int, control_dt: float) -> None:
        self.run_info.labels(*self._common(), backend, str(particles), f"{control_dt:.6f}").set(1.0)

    def set_particle_count(self, particles: int) -> None:
        self.particle_count.labels(*self._common()).set(particles)

    def set_memory_profile(
        self,
        *,
        state_bytes_total: int,
        state_bytes_per_particle: int,
        process_memory_per_particle_estimate_bytes: int,
    ) -> None:
        self.state_memory_total_bytes.labels(*self._common()).set(state_bytes_total)
        self.state_memory_per_particle_bytes.labels(*self._common()).set(state_bytes_per_particle)
        self.process_memory_per_particle_estimate_bytes.labels(*self._common()).set(process_memory_per_particle_estimate_bytes)

    def set_mujoco_memory_profile(
        self,
        *,
        model_nbuffer_bytes_per_robot: int,
        data_nbuffer_bytes_per_robot: int,
        data_narena_bytes_per_robot: int,
        native_bytes_per_robot: int,
        native_bytes_total: int,
    ) -> None:
        self.mujoco_model_buffer_per_particle_bytes.labels(*self._common()).set(model_nbuffer_bytes_per_robot)
        self.mujoco_data_buffer_per_particle_bytes.labels(*self._common()).set(data_nbuffer_bytes_per_robot)
        self.mujoco_data_arena_per_particle_bytes.labels(*self._common()).set(data_narena_bytes_per_robot)
        self.mujoco_native_memory_per_particle_bytes.labels(*self._common()).set(native_bytes_per_robot)
        self.mujoco_native_memory_total_bytes.labels(*self._common()).set(native_bytes_total)

    def set_runtime_environment(
        self,
        *,
        execution_platform: str,
        execution_device: str,
        default_jax_platform: str,
        default_jax_device: str,
        device_fallback_applied: bool,
    ) -> None:
        self.runtime_environment_info.labels(
            *self._common(),
            execution_platform,
            execution_device,
            default_jax_platform,
            default_jax_device,
        ).set(1.0)
        self.device_fallback_applied.labels(*self._common()).set(1.0 if device_fallback_applied else 0.0)

    def start_stage(self, stage: str) -> StageToken:
        if stage not in self._known_stages:
            self.register_stages([stage])
        for known_stage in self._known_stages:
            self.stage_active.labels(*self._stage(known_stage)).set(1.0 if known_stage == stage else 0.0)
        self.update_process_rss(stage)
        return StageToken(stage=stage, started_at=time.perf_counter())

    def finish_stage(self, token: StageToken) -> float:
        duration = time.perf_counter() - token.started_at
        self.stage_active.labels(*self._stage(token.stage)).set(0.0)
        self.stage_duration_seconds.labels(*self._stage(token.stage)).set(duration)
        self.update_process_rss(token.stage)
        return duration

    def start_substage(self, phase: str, substage: str) -> SubstageToken:
        if (phase, substage) not in self._known_substages:
            self.register_substages([(phase, substage)])
        self.substage_active.labels(*self._substage(phase, substage)).set(1.0)
        return SubstageToken(phase=phase, substage=substage, started_at=time.perf_counter())

    def finish_substage(self, token: SubstageToken) -> float:
        duration = time.perf_counter() - token.started_at
        self.set_substage_duration(token.phase, token.substage, duration)
        return duration

    def set_substage_duration(self, phase: str, substage: str, duration: float) -> None:
        if (phase, substage) not in self._known_substages:
            self.register_substages([(phase, substage)])
        self.substage_active.labels(*self._substage(phase, substage)).set(0.0)
        self.substage_duration_seconds.labels(*self._substage(phase, substage)).set(duration)

    def set_substage_workload(self, phase: str, substage: str, steps: int, particles: int, duration_seconds: float) -> None:
        particle_steps = float(steps * max(particles, 0))
        step_rate = float(steps / duration_seconds) if duration_seconds > 0 else 0.0
        particle_step_rate = particle_steps / duration_seconds if duration_seconds > 0 else 0.0
        ms_per_step = (duration_seconds * 1000.0 / steps) if steps > 0 else 0.0
        ms_per_particle_step = duration_seconds * 1000.0 / particle_steps if particle_steps > 0 else 0.0
        labels = self._substage(phase, substage)
        self.substage_steps.labels(*labels).set(steps)
        self.substage_step_rate_hz.labels(*labels).set(step_rate)
        self.substage_ms_per_step.labels(*labels).set(ms_per_step)
        self.substage_particle_steps_per_second.labels(*labels).set(particle_step_rate)
        self.substage_ms_per_particle_step.labels(*labels).set(ms_per_particle_step)

    def update_process_rss(self, stage: str) -> None:
        self.process_rss_bytes.labels(*self._stage(stage)).set(get_process_memory_bytes())

    def update_filter_state(self, ess: float, estimate: float, wall_seconds: float, cpu_seconds: float, cpu_equivalent_cores: float, particles: int) -> None:
        self.effective_sample_size.labels(*self._common()).set(ess)
        self.mass_estimate_kg.labels(*self._common()).set(estimate)
        self.phase4_step_wall_seconds.labels(*self._common()).set(wall_seconds)
        self.phase4_step_cpu_seconds.labels(*self._common()).set(cpu_seconds)
        self.phase4_cpu_equivalent_cores.labels(*self._common()).set(cpu_equivalent_cores)
        self.phase4_step_rate_hz.labels(*self._common()).set(1.0 / wall_seconds if wall_seconds > 0 else 0.0)
        particle_steps = float(max(particles, 0))
        self.phase4_particle_steps_per_second.labels(*self._common()).set(particle_steps / wall_seconds if wall_seconds > 0 else 0.0)
        self.phase4_ms_per_particle_step.labels(*self._common()).set((wall_seconds * 1000.0 / particle_steps) if particle_steps > 0 else 0.0)

    def update_accuracy_metrics(self, *, mass_abs_error_kg: float, mass_rel_error_pct: float, phase4_mae_kg: float, phase4_rmse_kg: float, mass_error_within_1pct: bool, mass_error_within_5pct: bool, mass_error_within_10pct: bool, convergence_time_to_5pct_seconds: float, convergence_time_to_10pct_seconds: float, time_to_first_estimate_seconds: float) -> None:
        self.mass_abs_error_kg.labels(*self._common()).set(mass_abs_error_kg)
        self.mass_rel_error_pct.labels(*self._common()).set(mass_rel_error_pct)
        self.phase4_mae_kg.labels(*self._common()).set(phase4_mae_kg)
        self.phase4_rmse_kg.labels(*self._common()).set(phase4_rmse_kg)
        self.mass_error_within_1pct.labels(*self._common()).set(1.0 if mass_error_within_1pct else 0.0)
        self.mass_error_within_5pct.labels(*self._common()).set(1.0 if mass_error_within_5pct else 0.0)
        self.mass_error_within_10pct.labels(*self._common()).set(1.0 if mass_error_within_10pct else 0.0)
        self.convergence_time_to_5pct_seconds.labels(*self._common()).set(convergence_time_to_5pct_seconds)
        self.convergence_time_to_10pct_seconds.labels(*self._common()).set(convergence_time_to_10pct_seconds)
        self.time_to_first_estimate_seconds.labels(*self._common()).set(time_to_first_estimate_seconds)

    def update_uncertainty_metrics(self, *, credible_interval_50_width_kg: float, credible_interval_90_width_kg: float, credible_interval_50_contains_truth: bool, credible_interval_90_contains_truth: bool, weight_entropy: float, weight_entropy_normalized: float, weight_perplexity: float) -> None:
        self.credible_interval_50_width_kg.labels(*self._common()).set(credible_interval_50_width_kg)
        self.credible_interval_90_width_kg.labels(*self._common()).set(credible_interval_90_width_kg)
        self.credible_interval_50_contains_truth.labels(*self._common()).set(1.0 if credible_interval_50_contains_truth else 0.0)
        self.credible_interval_90_contains_truth.labels(*self._common()).set(1.0 if credible_interval_90_contains_truth else 0.0)
        self.weight_entropy.labels(*self._common()).set(weight_entropy)
        self.weight_entropy_normalized.labels(*self._common()).set(weight_entropy_normalized)
        self.weight_perplexity.labels(*self._common()).set(weight_perplexity)

    def update_resample_state(self, *, steps: int, resample_count: int, resampled: bool, particle_min: float, particle_max: float, particle_mean: float, particle_std: float, particle_p10: float, particle_p50: float, particle_p90: float) -> None:
        self.resample_count.labels(*self._common()).set(resample_count)
        self.resample_rate.labels(*self._common()).set((resample_count / steps) if steps > 0 else 0.0)
        self.last_step_resampled.labels(*self._common()).set(1.0 if resampled else 0.0)
        self.particle_mass_min_kg.labels(*self._common()).set(particle_min)
        self.particle_mass_max_kg.labels(*self._common()).set(particle_max)
        self.particle_mass_mean_kg.labels(*self._common()).set(particle_mean)
        self.particle_mass_std_kg.labels(*self._common()).set(particle_std)
        self.particle_mass_p10_kg.labels(*self._common()).set(particle_p10)
        self.particle_mass_p50_kg.labels(*self._common()).set(particle_p50)
        self.particle_mass_p90_kg.labels(*self._common()).set(particle_p90)

    def update_likelihood_health(self, *, sim_force_finite_ratio: float, diff_finite_ratio: float, likelihood_finite_ratio: float, sim_force_norm_mean: float, diff_norm_mean: float, likelihood_min: float, likelihood_max: float, likelihood_mean: float, likelihood_std: float) -> None:
        self.sim_force_finite_ratio.labels(*self._common()).set(sim_force_finite_ratio)
        self.measurement_residual_finite_ratio.labels(*self._common()).set(diff_finite_ratio)
        self.likelihood_finite_ratio.labels(*self._common()).set(likelihood_finite_ratio)
        self.sim_force_norm_mean.labels(*self._common()).set(sim_force_norm_mean)
        self.measurement_residual_norm_mean.labels(*self._common()).set(diff_norm_mean)
        self.likelihood_min.labels(*self._common()).set(likelihood_min)
        self.likelihood_max.labels(*self._common()).set(likelihood_max)
        self.likelihood_mean.labels(*self._common()).set(likelihood_mean)
        self.likelihood_std.labels(*self._common()).set(likelihood_std)

    def update_invalid_state_counts(self, *, invalid_sensor_events: int, invalid_state_events: int, skipped_invalid_updates: int, skipped_invalid_update: bool, bootstrap_attempts: int, first_invalid_sensor_step: int, first_invalid_state_step: int, sim_force_nonfinite_count: int, diff_nonfinite_count: int, likelihood_nonfinite_count: int, qpos_nonfinite_count: int, qvel_nonfinite_count: int, sensordata_nonfinite_count: int, ctrl_nonfinite_count: int) -> None:
        self.invalid_sensor_events_total.labels(*self._common()).set(invalid_sensor_events)
        self.invalid_state_events_total.labels(*self._common()).set(invalid_state_events)
        self.skipped_invalid_updates_total.labels(*self._common()).set(skipped_invalid_updates)
        self.skipped_invalid_update.labels(*self._common()).set(1.0 if skipped_invalid_update else 0.0)
        self.first_update_bootstrap_attempts.labels(*self._common()).set(bootstrap_attempts)
        self.first_invalid_sensor_step.labels(*self._common()).set(first_invalid_sensor_step)
        self.first_invalid_state_step.labels(*self._common()).set(first_invalid_state_step)
        self.sim_force_nonfinite_count.labels(*self._common()).set(sim_force_nonfinite_count)
        self.measurement_residual_nonfinite_count.labels(*self._common()).set(diff_nonfinite_count)
        self.likelihood_nonfinite_count.labels(*self._common()).set(likelihood_nonfinite_count)
        self.qpos_nonfinite_count.labels(*self._common()).set(qpos_nonfinite_count)
        self.qvel_nonfinite_count.labels(*self._common()).set(qvel_nonfinite_count)
        self.sensordata_nonfinite_count.labels(*self._common()).set(sensordata_nonfinite_count)
        self.ctrl_nonfinite_count.labels(*self._common()).set(ctrl_nonfinite_count)

    def update_contact_health(self, *, contact_count_mean: float, contact_count_max: float, active_contact_particle_ratio: float, contact_metric_available: bool, contact_force_mismatch: bool, valid_force_particle_ratio: float, sim_force_signal_particle_ratio: float) -> None:
        self.contact_count_mean.labels(*self._common()).set(contact_count_mean)
        self.contact_count_max.labels(*self._common()).set(contact_count_max)
        self.active_contact_particle_ratio.labels(*self._common()).set(active_contact_particle_ratio)
        self.contact_metric_available.labels(*self._common()).set(1.0 if contact_metric_available else 0.0)
        self.contact_force_mismatch.labels(*self._common()).set(1.0 if contact_force_mismatch else 0.0)
        self.valid_force_particle_ratio.labels(*self._common()).set(valid_force_particle_ratio)
        self.force_signal_particle_ratio.labels(*self._common()).set(sim_force_signal_particle_ratio)

    def update_weight_health(self, *, uniform_weight_l1_distance: float, uniform_weight_max_deviation: float, collapsed_to_uniform: bool) -> None:
        self.uniform_weight_l1_distance.labels(*self._common()).set(uniform_weight_l1_distance)
        self.uniform_weight_max_deviation.labels(*self._common()).set(uniform_weight_max_deviation)
        self.likelihood_collapsed_to_uniform.labels(*self._common()).set(1.0 if collapsed_to_uniform else 0.0)

    def update_warp_memory(self, *, stage: str, bytes_in_use: int, peak_bytes_in_use: int, bytes_limit: int, state_bytes_estimate: int) -> None:
        labels = self._stage(stage)
        self.warp_bytes_in_use.labels(*labels).set(bytes_in_use)
        self.warp_peak_bytes_in_use.labels(*labels).set(peak_bytes_in_use)
        self.warp_bytes_limit.labels(*labels).set(bytes_limit)
        self.warp_state_bytes_estimate.labels(*labels).set(state_bytes_estimate)

    def set_prediction_ready(self, total_wall_seconds: float, final_error_pct: float) -> None:
        self.prediction_ready_seconds.labels(*self._common()).set(total_wall_seconds)
        self.final_error_pct.labels(*self._common()).set(final_error_pct)

    def _run_system_sampler(self) -> None:
        interval = SYSTEM_METRICS_INTERVAL_SECONDS
        previous_cpu_totals = self._read_host_cpu_totals()
        previous_process_totals = self._read_process_cpu_totals()
        while not self._stop_event.wait(interval):
            previous_cpu_totals, previous_process_totals = self._sample_system_metrics(previous_cpu_totals, previous_process_totals)

    def _sample_system_metrics(self, previous_cpu_totals: tuple[float, float] | None, previous_process_totals: tuple[float, float] | None) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
        self.process_memory_rss_bytes.labels(*self._common()).set(get_process_memory_bytes())
        self.host_memory_used_bytes.labels(*self._common()).set(self._read_host_memory_used_bytes())
        current_cpu_totals = self._read_host_cpu_totals()
        if previous_cpu_totals is not None and current_cpu_totals is not None:
            previous_total, previous_idle = previous_cpu_totals
            current_total, current_idle = current_cpu_totals
            total_delta = current_total - previous_total
            idle_delta = current_idle - previous_idle
            cpu_percent = 0.0
            if total_delta > 0:
                cpu_percent = max(0.0, min(100.0, 100.0 * (1.0 - (idle_delta / total_delta))))
            self.host_cpu_utilization_pct.labels(*self._common()).set(cpu_percent)
        current_process_totals = self._read_process_cpu_totals()
        if previous_process_totals is not None and current_process_totals is not None:
            previous_wall, previous_cpu = previous_process_totals
            current_wall, current_cpu = current_process_totals
            wall_delta = current_wall - previous_wall
            cpu_delta = current_cpu - previous_cpu
            process_cpu_percent = 0.0
            if wall_delta > 0:
                process_cpu_percent = max(0.0, 100.0 * (cpu_delta / wall_delta))
            self.process_cpu_utilization_pct.labels(*self._common()).set(process_cpu_percent)
            self.process_cpu_machine_pct.labels(*self._common()).set(process_cpu_percent / float(self._host_cpu_count))
        gpu_metrics = self._read_gpu_metrics()
        if gpu_metrics is not None:
            self.gpu_utilization_pct.labels(*self._common()).set(gpu_metrics["utilization_pct"])
            self.gpu_fb_used_bytes.labels(*self._common()).set(gpu_metrics["fb_used_bytes"])
            self.gpu_fb_utilization_pct.labels(*self._common()).set(gpu_metrics["fb_utilization_pct"])
            self.gpu_power_watts.labels(*self._common()).set(gpu_metrics["power_watts"])
            self.gpu_temp_celsius.labels(*self._common()).set(gpu_metrics["temp_celsius"])
            self.gpu_sm_clock_hz.labels(*self._common()).set(gpu_metrics["sm_clock_hz"])
            self.gpu_mem_clock_hz.labels(*self._common()).set(gpu_metrics["mem_clock_hz"])
        return current_cpu_totals, current_process_totals

    @staticmethod
    def _read_host_cpu_totals() -> tuple[float, float] | None:
        try:
            with open("/proc/stat", "r", encoding="utf-8") as handle:
                first_line = handle.readline().strip()
        except OSError:
            return None
        parts = first_line.split()
        if len(parts) < 5 or parts[0] != "cpu":
            return None
        values = [float(value) for value in parts[1:]]
        total = sum(values)
        idle = values[3] + (values[4] if len(values) > 4 else 0.0)
        return total, idle

    @staticmethod
    def _read_host_memory_used_bytes() -> float:
        values: dict[str, float] = {}
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as handle:
                for line in handle:
                    key, raw_value = line.split(":", 1)
                    values[key] = float(raw_value.strip().split()[0]) * 1024.0
        except OSError:
            return 0.0
        total = values.get("MemTotal", 0.0)
        available = values.get("MemAvailable", 0.0)
        return max(total - available, 0.0)

    @staticmethod
    def _read_process_cpu_totals() -> tuple[float, float]:
        return time.perf_counter(), time.process_time()

    @staticmethod
    def _read_gpu_metrics() -> dict[str, float] | None:
        command = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,clocks.sm,clocks.mem",
            "--format=csv,noheader,nounits",
        ]
        try:
            completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=2.0)
        except (OSError, subprocess.SubprocessError):
            return None
        lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        if not lines:
            return None
        fields = [field.strip() for field in lines[0].split(",")]
        if len(fields) != 7:
            return None
        try:
            utilization_pct = float(fields[0])
            fb_used_bytes = float(fields[1]) * 1024.0 * 1024.0
            fb_total_bytes = float(fields[2]) * 1024.0 * 1024.0
            temp_celsius = float(fields[3])
            power_watts = float(fields[4])
            sm_clock_hz = float(fields[5]) * 1_000_000.0
            mem_clock_hz = float(fields[6]) * 1_000_000.0
        except ValueError:
            return None
        return {
            "utilization_pct": utilization_pct,
            "fb_used_bytes": fb_used_bytes,
            "fb_utilization_pct": (100.0 * fb_used_bytes / fb_total_bytes) if fb_total_bytes > 0 else 0.0,
            "temp_celsius": temp_celsius,
            "power_watts": power_watts,
            "sm_clock_hz": sm_clock_hz,
            "mem_clock_hz": mem_clock_hz,
        }


@dataclass(frozen=True)
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


def init_stage_state(stage_name: str) -> dict[str, Any] | None:
    if stage_name != "phase_4_lift":
        return None
    return {
        "history_estimates": [],
        "pf_wall_durations": [],
        "pf_cpu_durations": [],
        "robot_execute_total": 0.0,
        "pf_update_total": 0.0,
        "resample_count": 0,
        "bootstrap_applied": False,
        "invalid_sensor_events": 0,
        "invalid_state_events": 0,
        "skipped_invalid_updates": 0,
        "first_invalid_sensor_step": -1,
        "first_invalid_state_step": -1,
        "max_repaired_world_count": 0,
        "abs_error_sum": 0.0,
        "squared_error_sum": 0.0,
        "time_to_first_estimate_seconds": -1.0,
        "convergence_time_to_5pct_seconds": -1.0,
        "convergence_time_to_10pct_seconds": -1.0,
        "latest_particles_snapshot": None,
    }


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


def observed_stage(stage: str, *, env_arg: str | None = None):
    def decorator(func):
        signature = inspect.signature(func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            bound = signature.bind_partial(*args, **kwargs)
            metrics_obj = bound.arguments.get("metrics")
            logger = bound.arguments.get("logger")
            log_data = bound.arguments.get("log_data")
            backend = bound.arguments.get("backend")
            env = bound.arguments.get(env_arg) if env_arg is not None else None
            ctx = bound.arguments.get("ctx")

            if ctx is not None:
                metrics_obj = ctx["metrics"]
                logger = ctx["logger"]
                log_data = ctx["log_data"]
                backend = ctx.get("backend")
                if env_arg is not None and env is None:
                    env = bound.arguments.get(env_arg)

            if metrics_obj is None or logger is None or log_data is None:
                raise ValueError(f"observed_stage requires metrics, logger, and log_data for stage {stage}")

            stage_token = metrics_obj.start_stage(stage)
            stage_label = stage.replace("_", " ")
            logger.info(
                {
                    **log_data,
                    "event": "stage_started",
                    "msg": f"Started {stage_label}.",
                    "stage": stage,
                }
            )
            try:
                return func(*args, **kwargs)
            finally:
                if backend == "mujoco-warp" and env is not None:
                    update_warp_memory_metrics(env, metrics_obj, stage=stage)
                metrics_obj.finish_stage(stage_token)
                logger.info(
                    {
                        **log_data,
                        "event": "stage_finished",
                        "msg": f"Finished {stage_label}.",
                        "stage": stage,
                    }
                )

        return wrapper

    return decorator


def apply_setup_observability(
    *,
    run_id: str,
    metrics: SimbayMetrics,
    backend_name: str,
    num_particles: int,
    dt: float,
    execution_device: str,
    true_mass: float,
    env: Any,
    env_memory_profile: dict[str, Any],
    memory_profile: dict[str, Any],
) -> None:
    set_span_attributes(
        {
            "simbay.run_id": run_id,
            "simbay.stage": "setup",
            "simbay.backend": backend_name,
            "simbay.particles": num_particles,
            "simbay.control_dt": dt,
            "simbay.execution_device": execution_device,
            "simbay.true_mass": float(true_mass),
        }
    )
    metrics.set_particle_count(num_particles)
    metrics.set_backend(backend_name, execution_device)
    metrics.set_run_info(backend=backend_name, particles=num_particles, control_dt=dt)
    update_setup_metrics(metrics, backend_name, env_memory_profile, memory_profile)
    if backend_name == "mujoco-warp":
        update_warp_memory_metrics(env, metrics, stage="setup")


def update_phase_4_state(
    state: dict[str, Any],
    *,
    particle_filter: Any,
    step: int,
    true_mass: float,
    started_at: float,
    step_result: dict[str, Any],
    step_wall_duration: float,
    step_cpu_duration: float,
) -> dict[str, float]:
    current_estimate = float(particle_filter.estimate())
    state["history_estimates"].append(current_estimate)
    if hasattr(particle_filter, "particles"):
        state["latest_particles_snapshot"] = np.asarray(particle_filter.particles).copy()
    if state["time_to_first_estimate_seconds"] < 0.0:
        state["time_to_first_estimate_seconds"] = time.perf_counter() - started_at
    abs_error_kg = abs(current_estimate - true_mass)
    rel_error_pct = (abs_error_kg / true_mass * 100.0) if true_mass != 0 else 0.0
    state["abs_error_sum"] += abs_error_kg
    state["squared_error_sum"] += abs_error_kg**2
    phase_4_mae_kg = state["abs_error_sum"] / (step + 1)
    phase_4_rmse_kg = math.sqrt(state["squared_error_sum"] / (step + 1))
    elapsed_since_run_start = time.perf_counter() - started_at
    if state["convergence_time_to_10pct_seconds"] < 0.0 and rel_error_pct <= 10.0:
        state["convergence_time_to_10pct_seconds"] = elapsed_since_run_start
    if state["convergence_time_to_5pct_seconds"] < 0.0 and rel_error_pct <= 5.0:
        state["convergence_time_to_5pct_seconds"] = elapsed_since_run_start
    state["pf_update_total"] += step_wall_duration
    state["pf_wall_durations"].append(step_wall_duration)
    state["pf_cpu_durations"].append(step_cpu_duration)
    state["resample_count"] = int(step_result.get("resample_count", state["resample_count"]))
    return {
        "current_estimate": current_estimate,
        "abs_error_kg": abs_error_kg,
        "rel_error_pct": rel_error_pct,
        "phase_4_mae_kg": phase_4_mae_kg,
        "phase_4_rmse_kg": phase_4_rmse_kg,
    }


def update_phase_4_metrics(
    ctx: dict[str, Any],
    state: dict[str, Any],
    *,
    particle_filter: Any,
    step: int,
    true_mass: float,
    step_result: dict[str, Any],
    step_wall_duration: float,
    step_cpu_duration: float,
) -> None:
    computed = update_phase_4_state(
        state,
        particle_filter=particle_filter,
        step=step,
        true_mass=true_mass,
        started_at=ctx["started_at"],
        step_result=step_result,
        step_wall_duration=step_wall_duration,
        step_cpu_duration=step_cpu_duration,
    )
    current_estimate = computed["current_estimate"]
    abs_error_kg = computed["abs_error_kg"]
    rel_error_pct = computed["rel_error_pct"]
    cpu_equivalent_cores_used = step_cpu_duration / step_wall_duration if step_wall_duration > 0 else 0.0
    add_exemplar(ctx["run_id"], step)
    ctx["metrics"].update_filter_state(
        ess=particle_filter.effective_sample_size(),
        estimate=current_estimate,
        wall_seconds=step_wall_duration,
        cpu_seconds=step_cpu_duration,
        cpu_equivalent_cores=cpu_equivalent_cores_used,
        particles=particle_filter.N,
    )
    ctx["metrics"].update_weight_health(
        uniform_weight_l1_distance=float(step_result.get("uniform_weight_l1_distance", 0.0)),
        uniform_weight_max_deviation=float(step_result.get("uniform_weight_max_deviation", 0.0)),
        collapsed_to_uniform=bool(step_result.get("collapsed_to_uniform", False)),
    )
    ctx["metrics"].update_accuracy_metrics(
        mass_abs_error_kg=abs_error_kg,
        mass_rel_error_pct=rel_error_pct,
        phase4_mae_kg=computed["phase_4_mae_kg"],
        phase4_rmse_kg=computed["phase_4_rmse_kg"],
        mass_error_within_1pct=rel_error_pct <= 1.0,
        mass_error_within_5pct=rel_error_pct <= 5.0,
        mass_error_within_10pct=rel_error_pct <= 10.0,
        convergence_time_to_5pct_seconds=state["convergence_time_to_5pct_seconds"],
        convergence_time_to_10pct_seconds=state["convergence_time_to_10pct_seconds"],
        time_to_first_estimate_seconds=state["time_to_first_estimate_seconds"],
    )
    latest_particles_snapshot = state["latest_particles_snapshot"]
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
        ctx["metrics"].update_resample_state(
            steps=step + 1,
            resample_count=state["resample_count"],
            resampled=bool(step_result.get("resampled", False)),
            particle_min=float(np.min(latest_particles_snapshot)),
            particle_max=float(np.max(latest_particles_snapshot)),
            particle_mean=float(np.mean(latest_particles_snapshot)),
            particle_std=float(np.std(latest_particles_snapshot)),
            particle_p10=float(np.percentile(latest_particles_snapshot, 10)),
            particle_p50=float(np.percentile(latest_particles_snapshot, 50)),
            particle_p90=float(np.percentile(latest_particles_snapshot, 90)),
        )
        ctx["metrics"].update_uncertainty_metrics(
            credible_interval_50_width_kg=ci50_high - ci50_low,
            credible_interval_90_width_kg=ci90_high - ci90_low,
            credible_interval_50_contains_truth=ci50_low <= true_mass <= ci50_high,
            credible_interval_90_contains_truth=ci90_low <= true_mass <= ci90_high,
            weight_entropy=weight_entropy,
            weight_entropy_normalized=weight_entropy_normalized,
            weight_perplexity=weight_perplexity,
        )
    if ctx["backend"] == "mujoco-warp":
        diagnostics = step_result.get("diagnostics", {})
        state["invalid_sensor_events"] = max(state["invalid_sensor_events"], int(diagnostics.get("invalid_sensor_events", 0.0)))
        state["invalid_state_events"] = max(state["invalid_state_events"], int(diagnostics.get("invalid_state_events", 0.0)))
        state["skipped_invalid_updates"] = max(state["skipped_invalid_updates"], int(step_result.get("skipped_invalid_updates", 0)))
        current_first_invalid_sensor_step = int(diagnostics.get("first_invalid_sensor_step", -1.0))
        current_first_invalid_state_step = int(diagnostics.get("first_invalid_state_step", -1.0))
        if state["first_invalid_sensor_step"] < 0 and current_first_invalid_sensor_step >= 0:
            state["first_invalid_sensor_step"] = current_first_invalid_sensor_step
        if state["first_invalid_state_step"] < 0 and current_first_invalid_state_step >= 0:
            state["first_invalid_state_step"] = current_first_invalid_state_step
        state["max_repaired_world_count"] = max(state["max_repaired_world_count"], int(diagnostics.get("repaired_world_count", 0.0)))
        ctx["metrics"].update_likelihood_health(
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
        ctx["metrics"].update_invalid_state_counts(
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
        ctx["metrics"].update_contact_health(
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


def phase_4_step_observability(
    ctx: dict[str, Any],
    stage_state: dict[str, Any],
    *,
    particle_filter: Any,
    step: int,
    true_mass: float,
    step_result: dict[str, Any],
    step_wall_duration: float,
    step_cpu_duration: float,
) -> None:
    previous_mass_estimate = (
        float(stage_state["history_estimates"][-1]) if stage_state["history_estimates"] else float(particle_filter.estimate())
    )
    set_span_attributes(
        {
            **ctx["span_attrs"],
            "simbay.stage": "phase_4_lift",
            "simbay.substage": "robot_execute",
            "simbay.substage_execution_strategy": "single_robot_control_loop",
            "simbay.particles_updated_at_the_same_time": 1,
        }
    )
    set_span_attributes({"simbay.phase_step_index": step})
    set_span_attributes(
        {
            **ctx["span_attrs"],
            "simbay.stage": "phase_4_lift",
            "simbay.substage": "pf_update",
            "simbay.previous_mass_estimate_kg": previous_mass_estimate,
            "simbay.particle_update_strategy": (
                "single_control_step_across_all_particles_in_batch"
                if ctx["backend"] == "mujoco-warp"
                else "single_control_step_one_particle_update_at_a_time"
            ),
            "simbay.particles_updated_at_the_same_time": (particle_filter.N if ctx["backend"] == "mujoco-warp" else 1),
        }
    )
    set_span_attributes({"simbay.new_mass_estimate_kg": float(particle_filter.estimate())})
    recovered_attempts = int(step_result.get("recovered_first_update_attempts", 0))
    if recovered_attempts > 1:
        ctx["logger"].info(
            {
                **ctx["log_data"],
                "event": "warp_first_update_recovered",
                "msg": f"Recovered the first Warp update after {recovered_attempts} attempts.",
                "attempts": recovered_attempts,
                "step": particle_filter._step_index - 1,
            }
        )
    update_phase_4_metrics(
        ctx,
        stage_state,
        particle_filter=particle_filter,
        step=step,
        true_mass=true_mass,
        step_result=step_result,
        step_wall_duration=step_wall_duration,
        step_cpu_duration=step_cpu_duration,
    )


def finalize_phase_4_metrics(
    ctx: dict[str, Any],
    state: dict[str, Any],
    *,
    trajectory: list[np.ndarray] | np.ndarray,
    particle_filter: Any,
) -> LiftPhaseResult:
    phase = "phase_4_lift"
    ctx["metrics"].set_substage_duration(phase, "robot_execute", state["robot_execute_total"])
    ctx["metrics"].set_substage_duration(phase, "pf_update", state["pf_update_total"])
    substage = "robot_execute"
    ctx["logger"].info(
        {
            **ctx["log_data"],
            "event": "substage_started",
            "msg": "Started robot execute for phase 4 lift.",
            "phase": phase,
            "substage": substage,
        }
    )
    ctx["logger"].info(
        {
            **ctx["log_data"],
            "event": "substage_finished",
            "msg": "Finished robot motion for phase 4 (lift).",
            "phase": phase,
            "substage": substage,
        }
    )
    substage = "pf_update"
    ctx["logger"].info(
        {
            **ctx["log_data"],
            "event": "substage_started",
            "msg": "Started pf update for phase 4 lift.",
            "phase": phase,
            "substage": substage,
        }
    )
    ctx["logger"].info(
        {
            **ctx["log_data"],
            "event": "substage_finished",
            "msg": "Finished particle filter update for phase 4 (lift).",
            "phase": phase,
            "substage": substage,
        }
    )
    ctx["metrics"].set_substage_workload(phase, "robot_execute", len(trajectory), 1, state["robot_execute_total"])
    ctx["metrics"].set_substage_workload(phase, "pf_update", len(trajectory), particle_filter.N, state["pf_update_total"])
    force_flush_tracing()
    return LiftPhaseResult(
        history_estimates=state["history_estimates"],
        pf_wall_durations=state["pf_wall_durations"],
        pf_cpu_durations=state["pf_cpu_durations"],
        invalid_sensor_events=state["invalid_sensor_events"],
        invalid_state_events=state["invalid_state_events"],
        skipped_invalid_updates=state["skipped_invalid_updates"],
        first_invalid_sensor_step=state["first_invalid_sensor_step"],
        first_invalid_state_step=state["first_invalid_state_step"],
        max_repaired_world_count=state["max_repaired_world_count"],
    )


def init_metrics(run_id: str = "unknown") -> SimbayMetrics:
    return SimbayMetrics(enabled=True, port=8000, run_id=run_id).initialize_defaults().start_runtime()


def shutdown_metrics(metrics: SimbayMetrics) -> None:
    with suppress(Exception):
        metrics.stop()
