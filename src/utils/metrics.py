import os
import subprocess
import threading
import time
from contextlib import suppress
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
from typing import Iterable

from .logging_utils import get_process_memory_bytes


@dataclass(frozen=True)
class StageToken:
    stage: str
    started_at: float


@dataclass(frozen=True)
class SubstageToken:
    phase: str
    substage: str
    started_at: float


class _MetricsStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._values: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}
        self._help: dict[str, str] = {}
        self._types: dict[str, str] = {}
        self._common_labels: dict[str, str] = {}

    def set_common_labels(self, labels: dict[str, str]) -> None:
        with self._lock:
            self._common_labels = dict(labels)

    def set_gauge(
        self,
        name: str,
        value: float,
        help_text: str,
        labels: dict[str, str] | None = None,
    ) -> None:
        merged_labels = dict(self._common_labels)
        if labels:
            merged_labels.update(labels)
        label_items = tuple(sorted(merged_labels.items()))
        with self._lock:
            self._help[name] = help_text
            self._types[name] = "gauge"
            self._values[(name, label_items)] = float(value)

    def render(self) -> bytes:
        lines: list[str] = []
        with self._lock:
            metric_names = sorted(self._help)
            for name in metric_names:
                lines.append(f"# HELP {name} {self._help[name]}")
                lines.append(f"# TYPE {name} {self._types[name]}")
                samples = [
                    (label_items, value)
                    for (metric_name, label_items), value in self._values.items()
                    if metric_name == name
                ]
                for label_items, value in sorted(samples):
                    lines.append(self._format_sample(name, label_items, value))
        lines.append("")
        return "\n".join(lines).encode("utf-8")

    @staticmethod
    def _format_sample(
        name: str,
        label_items: tuple[tuple[str, str], ...],
        value: float,
    ) -> str:
        if not label_items:
            return f"{name} {value}"
        def escape_label(raw_value: str) -> str:
            escaped = raw_value.replace("\\", "\\\\")
            return escaped.replace('"', '\\"')

        labels = ",".join(
            f'{key}="{escape_label(label_value)}"'
            for key, label_value in label_items
        )
        return f"{name}{{{labels}}} {value}"


class _ThreadedHTTPServer(ThreadingHTTPServer):
    daemon_threads = True


class SimbayMetrics:
    def __init__(self, enabled: bool, port: int, run_id: str = "unknown") -> None:
        self.enabled = enabled
        self.port = port
        self.run_id = run_id
        self._store = _MetricsStore()
        self._store.set_common_labels({"run_id": self.run_id})
        self._server: _ThreadedHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._system_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._host_cpu_count = max(os.cpu_count() or 1, 1)
        self._known_stages: set[str] = set()
        self._known_substages: set[tuple[str, str]] = set()

    def start(self) -> None:
        if not self.enabled or self._server is not None:
            return

        store = self._store

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                if self.path != "/metrics":
                    self.send_response(404)
                    self.end_headers()
                    return
                payload = store.render()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; version=0.0.4")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, format: str, *args) -> None:  # noqa: A003
                return

        try:
            self._server = _ThreadedHTTPServer(("0.0.0.0", self.port), Handler)
        except OSError:
            self._server = None
            return
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        self._stop_event.clear()
        self._sample_system_metrics(previous_cpu_totals=None, previous_process_totals=None)
        self._system_thread = threading.Thread(target=self._run_system_sampler, daemon=True)
        self._system_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join()
        if self._system_thread is not None:
            self._system_thread.join()
        self._server = None
        self._thread = None
        self._system_thread = None

    def register_stages(self, stages: Iterable[str]) -> None:
        for stage in stages:
            self._known_stages.add(stage)
            self._store.set_gauge(
                "simbay_stage_active",
                0.0,
                "Whether a simulation stage is currently active.",
                {"stage": stage},
            )
            self._store.set_gauge(
                "simbay_stage_duration_seconds",
                0.0,
                "Wall-clock duration for each simulation stage.",
                {"stage": stage},
            )

    def register_substages(self, substages: Iterable[tuple[str, str]]) -> None:
        for phase, substage in substages:
            key = (phase, substage)
            self._known_substages.add(key)
            labels = {"phase": phase, "substage": substage}
            self._store.set_gauge(
                "simbay_substage_active",
                0.0,
                "Whether a simulation substage is currently active.",
                labels,
            )
            self._store.set_gauge(
                "simbay_substage_duration_seconds",
                0.0,
                "Wall-clock duration for each simulation substage.",
                labels,
            )

    def set_backend(self, backend: str, device: str) -> None:
        self._store.set_gauge(
            "simbay_backend_info",
            1.0,
            "Resolved simulation backend and execution device.",
            {"backend": backend, "device": device},
        )

    def set_run_info(self, backend: str, particles: int, control_dt: float) -> None:
        self._store.set_gauge(
            "simbay_run_info",
            1.0,
            "Metadata for the current run.",
            {
                "backend": backend,
                "particles": str(particles),
                "control_dt": f"{control_dt:.6f}",
            },
        )

    def set_particle_count(self, particles: int) -> None:
        self._store.set_gauge(
            "simbay_particle_count",
            particles,
            "Configured particle count for the current run.",
        )

    def start_stage(self, stage: str) -> StageToken:
        if stage not in self._known_stages:
            self.register_stages([stage])
        for known_stage in self._known_stages:
            self._store.set_gauge(
                "simbay_stage_active",
                1.0 if known_stage == stage else 0.0,
                "Whether a simulation stage is currently active.",
                {"stage": known_stage},
            )
        self.update_process_rss(stage)
        return StageToken(stage=stage, started_at=time.perf_counter())

    def finish_stage(self, token: StageToken) -> None:
        duration = time.perf_counter() - token.started_at
        self._store.set_gauge(
            "simbay_stage_active",
            0.0,
            "Whether a simulation stage is currently active.",
            {"stage": token.stage},
        )
        self._store.set_gauge(
            "simbay_stage_duration_seconds",
            duration,
            "Wall-clock duration for each simulation stage.",
            {"stage": token.stage},
        )
        self.update_process_rss(token.stage)
        return duration

    def start_substage(self, phase: str, substage: str) -> SubstageToken:
        if (phase, substage) not in self._known_substages:
            self.register_substages([(phase, substage)])
        self._store.set_gauge(
            "simbay_substage_active",
            1.0,
            "Whether a simulation substage is currently active.",
            {"phase": phase, "substage": substage},
        )
        return SubstageToken(phase=phase, substage=substage, started_at=time.perf_counter())

    def finish_substage(self, token: SubstageToken) -> float:
        duration = time.perf_counter() - token.started_at
        self.set_substage_duration(token.phase, token.substage, duration)
        return duration

    def set_substage_duration(self, phase: str, substage: str, duration: float) -> None:
        if (phase, substage) not in self._known_substages:
            self.register_substages([(phase, substage)])
        labels = {"phase": phase, "substage": substage}
        self._store.set_gauge(
            "simbay_substage_active",
            0.0,
            "Whether a simulation substage is currently active.",
            labels,
        )
        self._store.set_gauge(
            "simbay_substage_duration_seconds",
            duration,
            "Wall-clock duration for each simulation substage.",
            labels,
        )

    def set_substage_workload(
        self,
        phase: str,
        substage: str,
        steps: int,
        particles: int,
        duration_seconds: float,
    ) -> None:
        labels = {"phase": phase, "substage": substage}
        particle_steps = float(steps * max(particles, 0))
        step_rate = float(steps / duration_seconds) if duration_seconds > 0 else 0.0
        particle_step_rate = particle_steps / duration_seconds if duration_seconds > 0 else 0.0
        ms_per_step = (duration_seconds * 1000.0 / steps) if steps > 0 else 0.0
        ms_per_particle_step = (
            duration_seconds * 1000.0 / particle_steps if particle_steps > 0 else 0.0
        )
        self._store.set_gauge(
            "simbay_substage_steps",
            steps,
            "Workload size in discrete control steps for each substage.",
            labels,
        )
        self._store.set_gauge(
            "simbay_substage_step_rate_hz",
            step_rate,
            "Discrete control steps processed per second for each substage.",
            labels,
        )
        self._store.set_gauge(
            "simbay_substage_ms_per_step",
            ms_per_step,
            "Milliseconds spent per control step for each substage.",
            labels,
        )
        self._store.set_gauge(
            "simbay_substage_particle_steps_per_second",
            particle_step_rate,
            "Particle-steps processed per second for each substage.",
            labels,
        )
        self._store.set_gauge(
            "simbay_substage_ms_per_particle_step",
            ms_per_particle_step,
            "Milliseconds spent per particle-step for each substage.",
            labels,
        )

    def update_process_rss(self, stage: str) -> None:
        self._store.set_gauge(
            "simbay_process_rss_bytes",
            get_process_memory_bytes(),
            "Resident set size of the Simbay process.",
            {"stage": stage},
        )

    def update_filter_state(
        self,
        ess: float,
        estimate: float,
        wall_seconds: float,
        cpu_seconds: float,
        cpu_equivalent_cores: float,
        particles: int,
    ) -> None:
        self._store.set_gauge(
            "simbay_effective_sample_size",
            ess,
            "Current effective sample size of the particle filter.",
        )
        self._store.set_gauge(
            "simbay_mass_estimate_kg",
            estimate,
            "Current mass estimate in kilograms.",
        )
        self._store.set_gauge(
            "simbay_phase4_step_wall_seconds",
            wall_seconds,
            "Wall-clock duration of the latest lift/update step.",
        )
        self._store.set_gauge(
            "simbay_phase4_step_cpu_seconds",
            cpu_seconds,
            "CPU duration of the latest lift/update step.",
        )
        self._store.set_gauge(
            "simbay_phase4_cpu_equivalent_cores",
            cpu_equivalent_cores,
            "Approximate CPU core usage during the latest lift/update step.",
        )
        self._store.set_gauge(
            "simbay_phase4_step_rate_hz",
            1.0 / wall_seconds if wall_seconds > 0 else 0.0,
            "Lift/update control steps processed per second.",
        )
        particle_steps = float(max(particles, 0))
        self._store.set_gauge(
            "simbay_phase4_particle_steps_per_second",
            particle_steps / wall_seconds if wall_seconds > 0 else 0.0,
            "Lift/update particle-steps processed per second.",
        )
        self._store.set_gauge(
            "simbay_phase4_ms_per_particle_step",
            (wall_seconds * 1000.0 / particle_steps) if particle_steps > 0 else 0.0,
            "Milliseconds spent per particle-step during the latest lift/update step.",
        )

    def update_accuracy_metrics(
        self,
        *,
        mass_abs_error_kg: float,
        mass_rel_error_pct: float,
        phase4_mae_kg: float,
        phase4_rmse_kg: float,
        mass_error_within_1pct: bool,
        mass_error_within_5pct: bool,
        mass_error_within_10pct: bool,
        convergence_time_to_5pct_seconds: float,
        convergence_time_to_10pct_seconds: float,
        time_to_first_estimate_seconds: float,
    ) -> None:
        self._store.set_gauge(
            "simbay_mass_abs_error_kg",
            mass_abs_error_kg,
            "Absolute difference between the current mass estimate and the true mass.",
        )
        self._store.set_gauge(
            "simbay_mass_rel_error_pct",
            mass_rel_error_pct,
            "Relative percent error of the current mass estimate.",
        )
        self._store.set_gauge(
            "simbay_phase4_mae_kg",
            phase4_mae_kg,
            "Running mean absolute error across phase-4 updates.",
        )
        self._store.set_gauge(
            "simbay_phase4_rmse_kg",
            phase4_rmse_kg,
            "Running root mean squared error across phase-4 updates.",
        )
        self._store.set_gauge(
            "simbay_mass_error_within_1pct",
            1.0 if mass_error_within_1pct else 0.0,
            "Whether the latest mass estimate is within 1 percent of the true mass.",
        )
        self._store.set_gauge(
            "simbay_mass_error_within_5pct",
            1.0 if mass_error_within_5pct else 0.0,
            "Whether the latest mass estimate is within 5 percent of the true mass.",
        )
        self._store.set_gauge(
            "simbay_mass_error_within_10pct",
            1.0 if mass_error_within_10pct else 0.0,
            "Whether the latest mass estimate is within 10 percent of the true mass.",
        )
        self._store.set_gauge(
            "simbay_convergence_time_to_5pct_seconds",
            convergence_time_to_5pct_seconds,
            "Time until the mass estimate first reached 5 percent relative error or better, or -1 if not reached.",
        )
        self._store.set_gauge(
            "simbay_convergence_time_to_10pct_seconds",
            convergence_time_to_10pct_seconds,
            "Time until the mass estimate first reached 10 percent relative error or better, or -1 if not reached.",
        )
        self._store.set_gauge(
            "simbay_time_to_first_estimate_seconds",
            time_to_first_estimate_seconds,
            "Time from run start until the first phase-4 estimate was produced.",
        )

    def update_uncertainty_metrics(
        self,
        *,
        credible_interval_50_width_kg: float,
        credible_interval_90_width_kg: float,
        credible_interval_50_contains_truth: bool,
        credible_interval_90_contains_truth: bool,
        weight_entropy: float,
        weight_entropy_normalized: float,
        weight_perplexity: float,
    ) -> None:
        self._store.set_gauge(
            "simbay_credible_interval_50_width_kg",
            credible_interval_50_width_kg,
            "Width of the central 50 percent credible interval over particle mass.",
        )
        self._store.set_gauge(
            "simbay_credible_interval_90_width_kg",
            credible_interval_90_width_kg,
            "Width of the central 90 percent credible interval over particle mass.",
        )
        self._store.set_gauge(
            "simbay_credible_interval_50_contains_truth",
            1.0 if credible_interval_50_contains_truth else 0.0,
            "Whether the true mass lies inside the central 50 percent credible interval.",
        )
        self._store.set_gauge(
            "simbay_credible_interval_90_contains_truth",
            1.0 if credible_interval_90_contains_truth else 0.0,
            "Whether the true mass lies inside the central 90 percent credible interval.",
        )
        self._store.set_gauge(
            "simbay_weight_entropy",
            weight_entropy,
            "Shannon entropy of the current particle-weight distribution.",
        )
        self._store.set_gauge(
            "simbay_weight_entropy_normalized",
            weight_entropy_normalized,
            "Particle-weight entropy normalized by the maximum possible entropy.",
        )
        self._store.set_gauge(
            "simbay_weight_perplexity",
            weight_perplexity,
            "Effective number of weight states implied by the current particle-weight distribution.",
        )

    def update_resample_state(
        self,
        *,
        steps: int,
        resample_count: int,
        resampled: bool,
        particle_min: float,
        particle_max: float,
        particle_mean: float,
        particle_std: float,
        particle_p10: float,
        particle_p50: float,
        particle_p90: float,
    ) -> None:
        self._store.set_gauge(
            "simbay_resample_count",
            resample_count,
            "Total number of particle-filter resampling events in the current run.",
        )
        self._store.set_gauge(
            "simbay_resample_rate",
            (resample_count / steps) if steps > 0 else 0.0,
            "Fraction of particle-filter update steps that triggered resampling.",
        )
        self._store.set_gauge(
            "simbay_last_step_resampled",
            1.0 if resampled else 0.0,
            "Whether the latest particle-filter update step triggered resampling.",
        )
        self._store.set_gauge(
            "simbay_particle_mass_min_kg",
            particle_min,
            "Minimum particle mass in kilograms for the latest particle cloud.",
        )
        self._store.set_gauge(
            "simbay_particle_mass_max_kg",
            particle_max,
            "Maximum particle mass in kilograms for the latest particle cloud.",
        )
        self._store.set_gauge(
            "simbay_particle_mass_mean_kg",
            particle_mean,
            "Mean particle mass in kilograms for the latest particle cloud.",
        )
        self._store.set_gauge(
            "simbay_particle_mass_std_kg",
            particle_std,
            "Standard deviation of particle mass in kilograms for the latest particle cloud.",
        )
        self._store.set_gauge(
            "simbay_particle_mass_p10_kg",
            particle_p10,
            "10th percentile of particle mass in kilograms for the latest particle cloud.",
        )
        self._store.set_gauge(
            "simbay_particle_mass_p50_kg",
            particle_p50,
            "50th percentile of particle mass in kilograms for the latest particle cloud.",
        )
        self._store.set_gauge(
            "simbay_particle_mass_p90_kg",
            particle_p90,
            "90th percentile of particle mass in kilograms for the latest particle cloud.",
        )

    def update_likelihood_health(
        self,
        *,
        sim_force_finite_ratio: float,
        diff_finite_ratio: float,
        likelihood_finite_ratio: float,
        sim_force_norm_mean: float,
        diff_norm_mean: float,
        likelihood_min: float,
        likelihood_max: float,
        likelihood_mean: float,
        likelihood_std: float,
    ) -> None:
        self._store.set_gauge(
            "simbay_sim_force_finite_ratio",
            sim_force_finite_ratio,
            "Fraction of simulated force samples that are finite.",
        )
        self._store.set_gauge(
            "simbay_measurement_residual_finite_ratio",
            diff_finite_ratio,
            "Fraction of measurement residual samples that are finite.",
        )
        self._store.set_gauge(
            "simbay_likelihood_finite_ratio",
            likelihood_finite_ratio,
            "Fraction of likelihood samples that are finite.",
        )
        self._store.set_gauge(
            "simbay_sim_force_norm_mean",
            sim_force_norm_mean,
            "Mean norm of simulated force samples for the latest likelihood evaluation.",
        )
        self._store.set_gauge(
            "simbay_measurement_residual_norm_mean",
            diff_norm_mean,
            "Mean norm of measurement residuals for the latest likelihood evaluation.",
        )
        self._store.set_gauge(
            "simbay_likelihood_min",
            likelihood_min,
            "Minimum particle likelihood from the latest likelihood evaluation.",
        )
        self._store.set_gauge(
            "simbay_likelihood_max",
            likelihood_max,
            "Maximum particle likelihood from the latest likelihood evaluation.",
        )
        self._store.set_gauge(
            "simbay_likelihood_mean",
            likelihood_mean,
            "Mean particle likelihood from the latest likelihood evaluation.",
        )
        self._store.set_gauge(
            "simbay_likelihood_std",
            likelihood_std,
            "Standard deviation of particle likelihoods from the latest likelihood evaluation.",
        )

    def update_invalid_state_counts(
        self,
        *,
        invalid_sensor_events: int,
        invalid_state_events: int,
        skipped_invalid_updates: int,
        skipped_invalid_update: bool,
        bootstrap_attempts: int,
        first_invalid_sensor_step: int,
        first_invalid_state_step: int,
        sim_force_nonfinite_count: int,
        diff_nonfinite_count: int,
        likelihood_nonfinite_count: int,
        qpos_nonfinite_count: int,
        qvel_nonfinite_count: int,
        sensordata_nonfinite_count: int,
        ctrl_nonfinite_count: int,
    ) -> None:
        self._store.set_gauge(
            "simbay_invalid_sensor_events_total",
            invalid_sensor_events,
            "Cumulative count of likelihood evaluations that produced non-finite sensor-derived values.",
        )
        self._store.set_gauge(
            "simbay_invalid_state_events_total",
            invalid_state_events,
            "Cumulative count of Warp state snapshots with non-finite entries.",
        )
        self._store.set_gauge(
            "simbay_skipped_invalid_updates_total",
            skipped_invalid_updates,
            "Cumulative count of Warp filter updates skipped because the likelihood batch or backend state was invalid.",
        )
        self._store.set_gauge(
            "simbay_skipped_invalid_update",
            1.0 if skipped_invalid_update else 0.0,
            "Whether the latest Warp filter update was skipped because the likelihood batch or backend state was invalid.",
        )
        self._store.set_gauge(
            "simbay_first_update_bootstrap_attempts",
            bootstrap_attempts,
            "Number of attempts used to obtain a valid first Warp phase-4 measurement update.",
        )
        self._store.set_gauge(
            "simbay_first_invalid_sensor_step",
            first_invalid_sensor_step,
            "First Warp filter step index that produced a non-finite sensor-derived value, or -1 if none.",
        )
        self._store.set_gauge(
            "simbay_first_invalid_state_step",
            first_invalid_state_step,
            "First Warp filter step index that produced a non-finite backend state value, or -1 if none.",
        )
        self._store.set_gauge(
            "simbay_sim_force_nonfinite_count",
            sim_force_nonfinite_count,
            "Count of non-finite simulated force entries from the latest likelihood evaluation.",
        )
        self._store.set_gauge(
            "simbay_measurement_residual_nonfinite_count",
            diff_nonfinite_count,
            "Count of non-finite measurement residual entries from the latest likelihood evaluation.",
        )
        self._store.set_gauge(
            "simbay_likelihood_nonfinite_count",
            likelihood_nonfinite_count,
            "Count of non-finite particle likelihoods from the latest likelihood evaluation.",
        )
        self._store.set_gauge(
            "simbay_qpos_nonfinite_count",
            qpos_nonfinite_count,
            "Count of non-finite Warp qpos entries in the latest sampled state snapshot.",
        )
        self._store.set_gauge(
            "simbay_qvel_nonfinite_count",
            qvel_nonfinite_count,
            "Count of non-finite Warp qvel entries in the latest sampled state snapshot.",
        )
        self._store.set_gauge(
            "simbay_sensordata_nonfinite_count",
            sensordata_nonfinite_count,
            "Count of non-finite Warp sensordata entries in the latest sampled state snapshot.",
        )
        self._store.set_gauge(
            "simbay_ctrl_nonfinite_count",
            ctrl_nonfinite_count,
            "Count of non-finite Warp ctrl entries in the latest sampled state snapshot.",
        )

    def update_contact_health(
        self,
        *,
        contact_count_mean: float,
        contact_count_max: float,
        active_contact_particle_ratio: float,
        contact_metric_available: bool,
        contact_force_mismatch: bool,
        valid_force_particle_ratio: float,
        sim_force_signal_particle_ratio: float,
    ) -> None:
        self._store.set_gauge(
            "simbay_contact_count_mean",
            contact_count_mean,
            "Mean contact count across Warp particles for the latest likelihood evaluation.",
        )
        self._store.set_gauge(
            "simbay_contact_count_max",
            contact_count_max,
            "Maximum contact count across Warp particles for the latest likelihood evaluation.",
        )
        self._store.set_gauge(
            "simbay_active_contact_particle_ratio",
            active_contact_particle_ratio,
            "Fraction of Warp particles with at least one active contact.",
        )
        self._store.set_gauge(
            "simbay_contact_metric_available",
            1.0 if contact_metric_available else 0.0,
            "Whether Warp exposed a contact-count array for the latest likelihood evaluation.",
        )
        self._store.set_gauge(
            "simbay_contact_force_mismatch",
            1.0 if contact_force_mismatch else 0.0,
            "Whether the latest likelihood evaluation had non-zero force signal but zero reported contacts.",
        )
        self._store.set_gauge(
            "simbay_valid_force_particle_ratio",
            valid_force_particle_ratio,
            "Fraction of particles with finite force vectors in the latest likelihood evaluation.",
        )
        self._store.set_gauge(
            "simbay_force_signal_particle_ratio",
            sim_force_signal_particle_ratio,
            "Fraction of particles with non-trivial simulated force norm in the latest likelihood evaluation.",
        )

    def update_weight_health(
        self,
        *,
        uniform_weight_l1_distance: float,
        uniform_weight_max_deviation: float,
        collapsed_to_uniform: bool,
    ) -> None:
        self._store.set_gauge(
            "simbay_uniform_weight_l1_distance",
            uniform_weight_l1_distance,
            "L1 distance between current particle weights and a perfectly uniform distribution.",
        )
        self._store.set_gauge(
            "simbay_uniform_weight_max_deviation",
            uniform_weight_max_deviation,
            "Maximum per-particle absolute deviation from uniform weights.",
        )
        self._store.set_gauge(
            "simbay_likelihood_collapsed_to_uniform",
            1.0 if collapsed_to_uniform else 0.0,
            "Whether the latest particle-weight update remained effectively uniform.",
        )

    def update_warp_memory(
        self,
        *,
        stage: str,
        bytes_in_use: int,
        peak_bytes_in_use: int,
        bytes_limit: int,
        state_bytes_estimate: int,
    ) -> None:
        labels = {"stage": stage}
        self._store.set_gauge(
            "simbay_warp_bytes_in_use",
            bytes_in_use,
            "Observed execution-device bytes used when Warp reports free/total memory, otherwise a tracked Warp state-byte estimate.",
            labels,
        )
        self._store.set_gauge(
            "simbay_warp_peak_bytes_in_use",
            peak_bytes_in_use,
            "Peak observed execution-device bytes used during the run, or peak tracked Warp state-byte estimate when device memory usage is unavailable.",
            labels,
        )
        self._store.set_gauge(
            "simbay_warp_bytes_limit",
            bytes_limit,
            "Warp execution-device total memory in bytes when available.",
            labels,
        )
        self._store.set_gauge(
            "simbay_warp_state_bytes_estimate",
            state_bytes_estimate,
            "Tracked lower-bound byte estimate for Warp model/data arrays and recovery snapshots.",
            labels,
        )

    def set_prediction_ready(self, total_wall_seconds: float, final_error_pct: float) -> None:
        self._store.set_gauge(
            "simbay_prediction_ready_seconds",
            total_wall_seconds,
            "Total wall-clock time until a final prediction is available.",
        )
        self._store.set_gauge(
            "simbay_final_error_pct",
            final_error_pct,
            "Final percent error of the predicted mass.",
        )

    def _run_system_sampler(self) -> None:
        interval = float(os.getenv("SIMBAY_SYSTEM_METRICS_INTERVAL_SECONDS", "1.0"))
        previous_cpu_totals = self._read_host_cpu_totals()
        previous_process_totals = self._read_process_cpu_totals()
        while not self._stop_event.wait(interval):
            previous_cpu_totals, previous_process_totals = self._sample_system_metrics(
                previous_cpu_totals,
                previous_process_totals,
            )

    def _sample_system_metrics(
        self,
        previous_cpu_totals: tuple[float, float] | None,
        previous_process_totals: tuple[float, float] | None,
    ) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
        self._store.set_gauge(
            "simbay_process_memory_rss_bytes",
            get_process_memory_bytes(),
            "Resident set size of the Simbay process.",
        )
        self._store.set_gauge(
            "simbay_host_memory_used_bytes",
            self._read_host_memory_used_bytes(),
            "Host memory currently in use for this benchmark run.",
        )
        current_cpu_totals = self._read_host_cpu_totals()
        if previous_cpu_totals is not None and current_cpu_totals is not None:
            previous_total, previous_idle = previous_cpu_totals
            current_total, current_idle = current_cpu_totals
            total_delta = current_total - previous_total
            idle_delta = current_idle - previous_idle
            cpu_percent = 0.0
            if total_delta > 0:
                cpu_percent = max(0.0, min(100.0, 100.0 * (1.0 - (idle_delta / total_delta))))
            self._store.set_gauge(
                "simbay_host_cpu_utilization_pct",
                cpu_percent,
                "Host CPU utilisation percentage for this benchmark run.",
            )
        current_process_totals = self._read_process_cpu_totals()
        if previous_process_totals is not None and current_process_totals is not None:
            previous_wall, previous_cpu = previous_process_totals
            current_wall, current_cpu = current_process_totals
            wall_delta = current_wall - previous_wall
            cpu_delta = current_cpu - previous_cpu
            process_cpu_percent = 0.0
            if wall_delta > 0:
                process_cpu_percent = max(0.0, 100.0 * (cpu_delta / wall_delta))
            self._store.set_gauge(
                "simbay_process_cpu_utilization_pct",
                process_cpu_percent,
                "CPU utilisation percentage of the Simbay process, expressed relative to one full core.",
            )
            self._store.set_gauge(
                "simbay_process_cpu_machine_pct",
                process_cpu_percent / float(self._host_cpu_count),
                "CPU utilisation percentage of the Simbay process, expressed relative to total machine CPU capacity.",
            )
        gpu_metrics = self._read_gpu_metrics()
        if gpu_metrics is not None:
            self._store.set_gauge(
                "simbay_gpu_utilization_pct",
                gpu_metrics["utilization_pct"],
                "GPU utilisation percentage for this benchmark run.",
            )
            self._store.set_gauge(
                "simbay_gpu_fb_used_bytes",
                gpu_metrics["fb_used_bytes"],
                "GPU frame-buffer memory currently in use for this benchmark run.",
            )
            self._store.set_gauge(
                "simbay_gpu_fb_utilization_pct",
                gpu_metrics["fb_utilization_pct"],
                "GPU frame-buffer memory utilisation percentage for this benchmark run.",
            )
            self._store.set_gauge(
                "simbay_gpu_power_watts",
                gpu_metrics["power_watts"],
                "GPU power draw in watts for this benchmark run.",
            )
            self._store.set_gauge(
                "simbay_gpu_temp_celsius",
                gpu_metrics["temp_celsius"],
                "GPU temperature in degrees Celsius for this benchmark run.",
            )
            self._store.set_gauge(
                "simbay_gpu_sm_clock_hz",
                gpu_metrics["sm_clock_hz"],
                "GPU SM clock frequency in hertz for this benchmark run.",
            )
            self._store.set_gauge(
                "simbay_gpu_mem_clock_hz",
                gpu_metrics["mem_clock_hz"],
                "GPU memory clock frequency in hertz for this benchmark run.",
            )
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
            completed = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=2.0,
            )
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


def create_metrics_from_env(run_id: str = "unknown") -> SimbayMetrics:
    enabled = os.getenv("SIMBAY_METRICS_ENABLED", "").lower() in {"1", "true", "yes", "on"}
    port = int(os.getenv("SIMBAY_METRICS_PORT", "8000"))
    return SimbayMetrics(enabled=enabled, port=port, run_id=run_id)


def shutdown_metrics(metrics: SimbayMetrics) -> None:
    with suppress(Exception):
        metrics.stop()
