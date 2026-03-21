import os
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

    def set_gauge(
        self,
        name: str,
        value: float,
        help_text: str,
        labels: dict[str, str] | None = None,
    ) -> None:
        label_items = tuple(sorted((labels or {}).items()))
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
    def __init__(self, enabled: bool, port: int) -> None:
        self.enabled = enabled
        self.port = port
        self._store = _MetricsStore()
        self._server: _ThreadedHTTPServer | None = None
        self._thread: threading.Thread | None = None
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

    def stop(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join()
        self._server = None
        self._thread = None

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

    def update_process_rss(self, stage: str) -> None:
        self._store.set_gauge(
            "simbay_process_rss_bytes",
            get_process_memory_bytes(),
            "Resident set size of the Simbay process.",
            {"stage": stage},
        )

    def update_mjx_memory(
        self,
        stage: str,
        bytes_in_use: int,
        peak_bytes_in_use: int,
        bytes_limit: int,
    ) -> None:
        self._store.set_gauge(
            "simbay_mjx_bytes_in_use",
            bytes_in_use,
            "Active MJX allocator bytes on the execution device.",
            {"stage": stage},
        )
        self._store.set_gauge(
            "simbay_mjx_peak_bytes_in_use",
            peak_bytes_in_use,
            "Peak MJX allocator bytes observed on the execution device.",
            {"stage": stage},
        )
        self._store.set_gauge(
            "simbay_mjx_bytes_limit",
            bytes_limit,
            "MJX allocator byte limit on the execution device.",
            {"stage": stage},
        )

    def update_filter_state(
        self,
        ess: float,
        estimate: float,
        wall_seconds: float,
        cpu_seconds: float,
        cpu_equivalent_cores: float,
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

    def update_replay_state(
        self,
        phase: str,
        wall_seconds: float,
        sync_seconds: float,
        step_rate_hz: float,
        cpu_equivalent_cores: float,
        mjx_bytes_in_use: int,
    ) -> None:
        labels = {"phase": phase}
        self._store.set_gauge(
            "simbay_replay_step_wall_seconds",
            wall_seconds,
            "Wall-clock duration of the latest replay predict step.",
            labels,
        )
        self._store.set_gauge(
            "simbay_replay_step_sync_seconds",
            sync_seconds,
            "Synchronization drain time for the latest replay predict step.",
            labels,
        )
        self._store.set_gauge(
            "simbay_replay_step_rate_hz",
            step_rate_hz,
            "Replay predict step rate derived from wall-clock duration.",
            labels,
        )
        self._store.set_gauge(
            "simbay_replay_cpu_equivalent_cores",
            cpu_equivalent_cores,
            "Approximate CPU core usage during the latest replay predict step.",
            labels,
        )
        self._store.set_gauge(
            "simbay_replay_mjx_bytes_in_use",
            mjx_bytes_in_use,
            "Active MJX allocator bytes during the latest replay predict step.",
            labels,
        )

    def update_particle_mass_metrics(self, particles: Iterable[float]) -> None:
        for idx, mass in enumerate(particles):
            self._store.set_gauge(
                "simbay_particle_mass_kg",
                float(mass),
                "Latest mass value for each particle.",
                {"particle": str(idx)},
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


def create_metrics_from_env() -> SimbayMetrics:
    enabled = os.getenv("SIMBAY_METRICS_ENABLED", "").lower() in {"1", "true", "yes", "on"}
    port = int(os.getenv("SIMBAY_METRICS_PORT", "8000"))
    return SimbayMetrics(enabled=enabled, port=port)


def shutdown_metrics(metrics: SimbayMetrics) -> None:
    with suppress(Exception):
        metrics.stop()
