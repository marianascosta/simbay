#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt

DEFAULT_PARTICLE_COUNTS = (1, 10, 50, 100, 250, 500)
SYSTEM_METRIC_NAMES = (
    "simbay_host_cpu_utilization_pct",
    "simbay_process_cpu_utilization_pct",
    "simbay_process_cpu_machine_pct",
    "simbay_process_memory_rss_bytes",
    "simbay_host_memory_used_bytes",
    "simbay_gpu_utilization_pct",
    "simbay_gpu_fb_used_bytes",
    "simbay_gpu_fb_utilization_pct",
    "simbay_gpu_power_watts",
    "simbay_gpu_temp_celsius",
    "simbay_gpu_sm_clock_hz",
    "simbay_gpu_mem_clock_hz",
)
STAGE_ORDER = (
    "setup",
    "ik_planning",
    "phase_1_approach",
    "phase_2_descend",
    "phase_3_grip",
    "phase_4_lift",
    "plot_generation",
)


@dataclass(frozen=True)
class MetricSample:
    timestamp: float
    active_stage: str | None
    metrics: dict[tuple[str, tuple[tuple[str, str], ...]], float]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Run a particle-count benchmark sweep using Simbay's existing logs and " "Prometheus metrics."),
    )
    parser.add_argument(
        "--backend",
        default=os.getenv("SIMBAY_BACKEND", "mujoco-warp"),
        help="Backend to benchmark. Defaults to SIMBAY_BACKEND or mujoco-warp.",
    )
    parser.add_argument(
        "--counts",
        default=",".join(str(value) for value in DEFAULT_PARTICLE_COUNTS),
        help="Comma-separated particle counts. Defaults to 1,10,50,100,250,500.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for sweep artifacts. Defaults to temp/benchmark_runs/<timestamp>.",
    )
    parser.add_argument(
        "--sample-interval",
        type=float,
        default=0.5,
        help="Metrics scrape interval in seconds. Defaults to 0.5.",
    )
    parser.add_argument(
        "--metrics-port-base",
        type=int,
        default=9100,
        help="Base Prometheus port. Each run increments from this value.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to launch main.py. Defaults to the current interpreter.",
    )
    return parser.parse_args()


def _metric_key(
    name: str,
    labels: dict[str, str] | None = None,
) -> tuple[str, tuple[tuple[str, str], ...]]:
    return name, tuple(sorted((labels or {}).items()))


def _parse_prometheus_metrics(payload: str) -> dict[tuple[str, tuple[tuple[str, str], ...]], float]:
    parsed: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}
    for raw_line in payload.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        metric_part, value_part = line.rsplit(" ", 1)
        value = float(value_part)
        if "{" not in metric_part:
            parsed[(metric_part, tuple())] = value
            continue
        name, raw_labels = metric_part.split("{", 1)
        raw_labels = raw_labels[:-1]
        labels: list[tuple[str, str]] = []
        if raw_labels:
            for item in raw_labels.split(","):
                label_name, raw_value = item.split("=", 1)
                labels.append((label_name, raw_value.strip().strip('"')))
        parsed[(name, tuple(sorted(labels)))] = value
    return parsed


def _scrape_metrics(port: int) -> dict[tuple[str, tuple[tuple[str, str], ...]], float] | None:
    url = f"http://127.0.0.1:{port}/metrics"
    try:
        with urllib.request.urlopen(url, timeout=1.0) as response:
            payload = response.read().decode("utf-8")
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return None
    return _parse_prometheus_metrics(payload)


def _find_active_stage(
    metrics: dict[tuple[str, tuple[tuple[str, str], ...]], float],
) -> str | None:
    active_stages: list[str] = []
    for stage in STAGE_ORDER:
        value = metrics.get(_metric_key("simbay_stage_active", {"stage": stage}), 0.0)
        if value >= 0.5:
            active_stages.append(stage)
    return active_stages[0] if active_stages else None


def _collect_metric_series(
    samples: list[MetricSample],
    metric_name: str,
    *,
    stage: str | None = None,
) -> list[float]:
    series: list[float] = []
    for sample in samples:
        if stage is not None and sample.active_stage != stage:
            continue
        value = sample.metrics.get(_metric_key(metric_name))
        if value is not None:
            series.append(float(value))
    return series


def _series_stats(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {
            "samples": 0,
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "first": 0.0,
            "last": 0.0,
        }
    return {
        "samples": len(values),
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
        "first": values[0],
        "last": values[-1],
    }


def _load_json_lines(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            records.append(payload)
    return records


def _find_first_event(
    records: list[dict[str, object]],
    event_name: str,
) -> dict[str, object] | None:
    for record in records:
        if record.get("event") == event_name:
            return record
    return None


def _find_last_event(
    records: list[dict[str, object]],
    event_name: str,
) -> dict[str, object] | None:
    for record in reversed(records):
        if record.get("event") == event_name:
            return record
    return None


def _build_stage_duration_map(records: list[dict[str, object]]) -> dict[str, float]:
    durations: dict[str, float] = {}
    for record in records:
        if record.get("event") != "stage_finished":
            continue
        stage = str(record.get("stage", "unknown"))
        durations[stage] = float(record.get("duration_ms", 0.0)) / 1000.0
    return durations


def _build_substage_duration_map(records: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    durations: dict[str, dict[str, float]] = defaultdict(dict)
    for record in records:
        if record.get("event") != "substage_finished":
            continue
        phase = str(record.get("phase", "unknown"))
        substage = str(record.get("substage", "unknown"))
        durations[phase][substage] = float(record.get("duration_ms", 0.0)) / 1000.0
    return {phase: dict(values) for phase, values in durations.items()}


def _extract_metric_snapshot(
    samples: list[MetricSample],
) -> dict[tuple[str, tuple[tuple[str, str], ...]], float]:
    if not samples:
        return {}
    return dict(samples[-1].metrics)


def _snapshot_value(
    snapshot: dict[tuple[str, tuple[tuple[str, str], ...]], float],
    name: str,
    labels: dict[str, str] | None = None,
) -> float:
    return float(snapshot.get(_metric_key(name, labels), 0.0))


def _summarize_run(
    *,
    run_id: str,
    particles: int,
    backend: str,
    log_path: Path,
    samples: list[MetricSample],
) -> dict[str, object]:
    records = _load_json_lines(log_path)
    setup_event = _find_first_event(records, "simulation_setup") or {}
    run_metadata = _find_first_event(records, "run_metadata") or {}
    pf_summary = _find_last_event(records, "particle_filter_summary") or {}
    prediction_ready = _find_last_event(records, "prediction_ready") or {}
    final_error = _find_last_event(records, "final_error") or {}
    stage_durations = _build_stage_duration_map(records)
    substage_durations = _build_substage_duration_map(records)
    snapshot = _extract_metric_snapshot(samples)

    stage_metrics: dict[str, dict[str, object]] = {}
    for stage in STAGE_ORDER:
        stage_metrics[stage] = {
            "duration_seconds": stage_durations.get(stage, 0.0),
            "host_cpu_utilization_pct": _series_stats(
                _collect_metric_series(samples, "simbay_host_cpu_utilization_pct", stage=stage)
            ),
            "process_cpu_utilization_pct": _series_stats(
                _collect_metric_series(samples, "simbay_process_cpu_utilization_pct", stage=stage)
            ),
            "process_cpu_machine_pct": _series_stats(
                _collect_metric_series(samples, "simbay_process_cpu_machine_pct", stage=stage)
            ),
            "process_memory_rss_bytes": _series_stats(
                _collect_metric_series(samples, "simbay_process_memory_rss_bytes", stage=stage)
            ),
            "host_memory_used_bytes": _series_stats(
                _collect_metric_series(samples, "simbay_host_memory_used_bytes", stage=stage)
            ),
            "gpu_utilization_pct": _series_stats(
                _collect_metric_series(samples, "simbay_gpu_utilization_pct", stage=stage)
            ),
            "gpu_fb_used_bytes": _series_stats(
                _collect_metric_series(samples, "simbay_gpu_fb_used_bytes", stage=stage)
            ),
            "gpu_power_watts": _series_stats(_collect_metric_series(samples, "simbay_gpu_power_watts", stage=stage)),
            "gpu_temp_celsius": _series_stats(_collect_metric_series(samples, "simbay_gpu_temp_celsius", stage=stage)),
            "warp_bytes_in_use": _snapshot_value(
                snapshot,
                "simbay_warp_bytes_in_use",
                {"stage": stage},
            ),
            "warp_peak_bytes_in_use": _snapshot_value(
                snapshot,
                "simbay_warp_peak_bytes_in_use",
                {"stage": stage},
            ),
            "warp_bytes_limit": _snapshot_value(
                snapshot,
                "simbay_warp_bytes_limit",
                {"stage": stage},
            ),
            "warp_state_bytes_estimate": _snapshot_value(
                snapshot,
                "simbay_warp_state_bytes_estimate",
                {"stage": stage},
            ),
        }

    substage_metrics: dict[str, dict[str, dict[str, float]]] = {}
    for phase in ("phase_1_approach", "phase_2_descend", "phase_3_grip", "phase_4_lift"):
        phase_substages: dict[str, dict[str, float]] = {}
        for substage in ("robot_execute", "pf_replay", "pf_update"):
            duration_seconds = substage_durations.get(phase, {}).get(substage, 0.0)
            phase_substages[substage] = {
                "duration_seconds": duration_seconds,
                "steps": _snapshot_value(
                    snapshot,
                    "simbay_substage_steps",
                    {"phase": phase, "substage": substage},
                ),
                "step_rate_hz": _snapshot_value(
                    snapshot,
                    "simbay_substage_step_rate_hz",
                    {"phase": phase, "substage": substage},
                ),
                "ms_per_step": _snapshot_value(
                    snapshot,
                    "simbay_substage_ms_per_step",
                    {"phase": phase, "substage": substage},
                ),
                "particle_steps_per_second": _snapshot_value(
                    snapshot,
                    "simbay_substage_particle_steps_per_second",
                    {"phase": phase, "substage": substage},
                ),
                "ms_per_particle_step": _snapshot_value(
                    snapshot,
                    "simbay_substage_ms_per_particle_step",
                    {"phase": phase, "substage": substage},
                ),
            }
        substage_metrics[phase] = phase_substages

    system_summary = {name: _series_stats(_collect_metric_series(samples, name)) for name in SYSTEM_METRIC_NAMES}

    launch_stage = stage_metrics["setup"]
    phase4_pf_update = substage_metrics["phase_4_lift"]["pf_update"]

    return {
        "run_id": run_id,
        "backend": backend,
        "particles": particles,
        "log_path": str(log_path),
        "sample_count": len(samples),
        "sample_interval_seconds": (float(samples[1].timestamp - samples[0].timestamp) if len(samples) > 1 else 0.0),
        "run_metadata": run_metadata,
        "hardware": {
            "gpu_device": setup_event.get("execution_device", run_metadata.get("device", "unknown")),
            "execution_platform": setup_event.get("execution_platform", "unknown"),
            "cpu_hardware": platform.processor() or "unknown",
            "cpu_machine": platform.machine() or "unknown",
            "host_platform": platform.platform() or "unknown",
        },
        "setup": {
            "launch_particles_seconds": float(launch_stage["duration_seconds"]),
            "execution_device": setup_event.get("execution_device", run_metadata.get("device", "unknown")),
            "execution_platform": setup_event.get("execution_platform", "unknown"),
            "state_memory_total_bytes": float(setup_event.get("state_memory_total_bytes", 0.0)),
            "state_memory_per_particle_bytes": float(setup_event.get("state_memory_per_particle_bytes", 0.0)),
            "process_memory_per_particle_estimate_bytes": float(
                setup_event.get("process_memory_per_particle_estimate_bytes", 0.0)
            ),
            "setup_stage_metrics": launch_stage,
        },
        "stage_metrics": stage_metrics,
        "substage_metrics": substage_metrics,
        "phase4": {
            "steps": int(float(pf_summary.get("steps", 0.0))),
            "avg_wall_ms": float(pf_summary.get("avg_wall_ms", 0.0)),
            "avg_cpu_ms": float(pf_summary.get("avg_cpu_ms", 0.0)),
            "avg_step_rate_hz": float(pf_summary.get("avg_step_rate_hz", 0.0)),
            "avg_cpu_equivalent_cores": float(pf_summary.get("avg_cpu_equivalent_cores", 0.0)),
            "final_estimate_kg": float(pf_summary.get("final_estimate", 0.0)),
            "final_error_pct": float(pf_summary.get("final_error_pct", 0.0)),
            "final_rss_bytes": float(pf_summary.get("final_rss_bytes", 0.0)),
            "pf_update_particle_steps_per_second": float(phase4_pf_update["particle_steps_per_second"]),
            "pf_update_ms_per_particle_step": float(phase4_pf_update["ms_per_particle_step"]),
            "ess_final": _snapshot_value(snapshot, "simbay_effective_sample_size"),
            "resample_count": _snapshot_value(snapshot, "simbay_resample_count"),
            "resample_rate": _snapshot_value(snapshot, "simbay_resample_rate"),
            "credible_interval_50_width_kg": _snapshot_value(snapshot, "simbay_credible_interval_50_width_kg"),
            "credible_interval_90_width_kg": _snapshot_value(snapshot, "simbay_credible_interval_90_width_kg"),
            "weight_entropy": _snapshot_value(snapshot, "simbay_weight_entropy"),
            "weight_entropy_normalized": _snapshot_value(snapshot, "simbay_weight_entropy_normalized"),
            "weight_perplexity": _snapshot_value(snapshot, "simbay_weight_perplexity"),
        },
        "accuracy": {
            "prediction_ready_seconds": float(prediction_ready.get("total_wall_s", 0.0)),
            "final_error_pct": float(final_error.get("final_error_pct", 0.0)),
            "convergence_time_to_5pct_seconds": _snapshot_value(snapshot, "simbay_convergence_time_to_5pct_seconds"),
            "convergence_time_to_10pct_seconds": _snapshot_value(snapshot, "simbay_convergence_time_to_10pct_seconds"),
            "time_to_first_estimate_seconds": _snapshot_value(snapshot, "simbay_time_to_first_estimate_seconds"),
            "phase4_mae_kg": _snapshot_value(snapshot, "simbay_phase4_mae_kg"),
            "phase4_rmse_kg": _snapshot_value(snapshot, "simbay_phase4_rmse_kg"),
        },
        "warp_health": {
            "invalid_sensor_events_total": _snapshot_value(snapshot, "simbay_invalid_sensor_events_total"),
            "invalid_state_events_total": _snapshot_value(snapshot, "simbay_invalid_state_events_total"),
            "skipped_invalid_updates_total": _snapshot_value(snapshot, "simbay_skipped_invalid_updates_total"),
            "first_invalid_sensor_step": _snapshot_value(snapshot, "simbay_first_invalid_sensor_step"),
            "first_invalid_state_step": _snapshot_value(snapshot, "simbay_first_invalid_state_step"),
            "contact_count_mean": _snapshot_value(snapshot, "simbay_contact_count_mean"),
            "contact_count_max": _snapshot_value(snapshot, "simbay_contact_count_max"),
            "active_contact_particle_ratio": _snapshot_value(snapshot, "simbay_active_contact_particle_ratio"),
            "force_signal_particle_ratio": _snapshot_value(snapshot, "simbay_force_signal_particle_ratio"),
            "likelihood_finite_ratio": _snapshot_value(snapshot, "simbay_likelihood_finite_ratio"),
            "likelihood_collapsed_to_uniform": _snapshot_value(snapshot, "simbay_likelihood_collapsed_to_uniform"),
        },
        "system_summary": system_summary,
    }


def _write_samples(path: Path, samples: list[MetricSample]) -> None:
    serialized = [
        {
            "timestamp": sample.timestamp,
            "active_stage": sample.active_stage,
            "metrics": {
                f"{name}{json.dumps(dict(labels), sort_keys=True)}": value
                for (name, labels), value in sorted(sample.metrics.items())
            },
        }
        for sample in samples
    ]
    path.write_text(json.dumps(serialized, indent=2, sort_keys=True), encoding="utf-8")


def _run_single_benchmark(
    *,
    particles: int,
    backend: str,
    output_dir: Path,
    metrics_port: int,
    sample_interval: float,
    python_executable: str,
) -> dict[str, object]:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"sweep-{backend}-n{particles}-{timestamp}"
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "stdout.jsonl"
    sample_path = run_dir / "metrics_samples.json"

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["SIMBAY_BACKEND"] = backend
    env["SIMBAY_PARTICLES"] = str(particles)
    env["SIMBAY_HEADLESS"] = env.get("SIMBAY_HEADLESS", "1")
    env["SIMBAY_MASS_TIMESERIES_ENABLED"] = "0"
    env["SIMBAY_SKIP_MAIN_PLOT"] = "1"
    env["SIMBAY_RUN_ID"] = run_id
    env["SIMBAY_METRICS_ENABLED"] = "1"
    env["SIMBAY_METRICS_PORT"] = str(metrics_port)
    env["SIMBAY_SYSTEM_METRICS_INTERVAL_SECONDS"] = str(sample_interval)

    samples: list[MetricSample] = []
    stop_event = threading.Event()
    process: subprocess.Popen[str] | None = None

    def sampler() -> None:
        while not stop_event.is_set():
            scraped = _scrape_metrics(metrics_port)
            if scraped is not None:
                samples.append(
                    MetricSample(
                        timestamp=time.time(),
                        active_stage=_find_active_stage(scraped),
                        metrics=scraped,
                    )
                )
            stop_event.wait(sample_interval)

    sampler_thread = threading.Thread(target=sampler, daemon=True)

    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            [python_executable, "-u", "main.py"],
            cwd=Path(__file__).resolve().parent.parent,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        sampler_thread.start()
        return_code = process.wait()

    stop_event.set()
    sampler_thread.join()

    final_scrape = _scrape_metrics(metrics_port)
    if final_scrape is not None:
        samples.append(
            MetricSample(
                timestamp=time.time(),
                active_stage=_find_active_stage(final_scrape),
                metrics=final_scrape,
            )
        )

    _write_samples(sample_path, samples)

    if return_code != 0:
        tail_lines = log_path.read_text(encoding="utf-8").splitlines()[-40:]
        tail_text = "\n".join(tail_lines) if tail_lines else "(no log output captured)"
        raise RuntimeError(
            "Benchmark run failed.\n"
            f"run_id={run_id}\n"
            f"log_path={log_path}\n"
            f"exit_code={return_code}\n"
            "Last log lines:\n"
            f"{tail_text}"
        )

    summary = _summarize_run(
        run_id=run_id,
        particles=particles,
        backend=backend,
        log_path=log_path,
        samples=samples,
    )
    summary["run_dir"] = str(run_dir)
    summary_path = run_dir / "benchmark_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _plot_run_timeseries(run_dir=run_dir, summary=summary, samples=samples)
    return summary


def _metric_series(samples: list[MetricSample], name: str) -> list[float]:
    values: list[float] = []
    for sample in samples:
        values.append(float(sample.metrics.get(_metric_key(name), float("nan"))))
    return values


def _plot_run_timeseries(
    *,
    run_dir: Path,
    summary: dict[str, object],
    samples: list[MetricSample],
) -> Path:
    if not samples:
        return run_dir / "run_timeseries.png"

    t0 = samples[0].timestamp
    times = [sample.timestamp - t0 for sample in samples]

    rss_gib = [value / (1024.0**3) for value in _metric_series(samples, "simbay_process_memory_rss_bytes")]
    gpu_fb_gib = [value / (1024.0**3) for value in _metric_series(samples, "simbay_gpu_fb_used_bytes")]
    gpu_util_pct = _metric_series(samples, "simbay_gpu_utilization_pct")
    phase4_particle_steps = _metric_series(samples, "simbay_phase4_particle_steps_per_second")
    mass_estimate = _metric_series(samples, "simbay_mass_estimate_kg")
    mass_rel_error_pct = _metric_series(samples, "simbay_mass_rel_error_pct")

    launch_seconds = float(summary["setup"]["launch_particles_seconds"])
    particles = int(summary["particles"])
    launch_rate = (particles / launch_seconds) if launch_seconds > 0 else 0.0
    hardware = summary.get("hardware", {})
    gpu_device = str(hardware.get("gpu_device", summary["setup"].get("execution_device", "unknown")))
    cpu_hardware = str(hardware.get("cpu_hardware", "unknown"))
    cpu_machine = str(hardware.get("cpu_machine", "unknown"))
    execution_platform = str(hardware.get("execution_platform", summary["setup"].get("execution_platform", "unknown")))
    setup_stage = [1.0 if (sample.active_stage or "") == "setup" else 0.0 for sample in samples]
    launch_rate_series = [launch_rate if is_setup >= 0.5 else float("nan") for is_setup in setup_stage]

    figure, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True)
    plots = [
        (axes[0, 0], launch_rate_series, "Particle Launch Rate During Setup", "particles/s"),
        (axes[0, 1], gpu_fb_gib, "VRAM Usage (GPU FB Used)", "GiB"),
        (axes[1, 0], phase4_particle_steps, "Phase-4 Particle Steps/s", "steps/s"),
        (axes[1, 1], mass_estimate, "Mass Estimate", "kg"),
        (axes[2, 0], mass_rel_error_pct, "Mass Relative Error", "%"),
        (axes[2, 1], gpu_util_pct, "GPU Utilization", "%"),
    ]
    for axis, values, title, ylabel in plots:
        axis.plot(times, values, linewidth=1.8)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(True, linestyle=":", alpha=0.6)
        axis.set_xlabel("Time (s)")

    gpu_fb_valid = [value for value in gpu_fb_gib if value == value]
    gpu_util_valid = [value for value in gpu_util_pct if value == value]
    rss_valid = [value for value in rss_gib if value == value]
    axes[0, 1].text(
        0.98,
        0.95,
        (
            (
                f"Peak VRAM: {max(gpu_fb_valid):.2f} GiB\n"
                f"Peak GPU util: {max(gpu_util_valid):.1f}%\n"
                f"Peak RSS: {max(rss_valid):.2f} GiB"
            )
            if (gpu_fb_valid and gpu_util_valid and rss_valid)
            else "GPU metrics unavailable"
        ),
        transform=axes[0, 1].transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.75),
    )
    axes[0, 0].text(
        0.98,
        0.95,
        f"particles={particles}\nsetup={launch_seconds:.3f}s\nrate={launch_rate:.2f} particles/s",
        transform=axes[0, 0].transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.75),
    )

    figure.suptitle(
        (f"Run Time-Series: backend={summary['backend']} " f"particles={summary['particles']}"),
        fontsize=14,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.965,
        (
            f"Particles: {particles} | GPU: {gpu_device} | "
            f"CPU: {cpu_hardware} ({cpu_machine}) | Platform: {execution_platform}"
        ),
        ha="center",
        va="top",
        fontsize=10,
    )
    figure.tight_layout()
    plot_path = run_dir / "run_timeseries.png"
    figure.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return plot_path


def _write_markdown_report(
    output_dir: Path,
    summaries: list[dict[str, object]],
    timestamp: str,
) -> Path:
    report_path = output_dir / f"report-{timestamp}.md"
    lines = [
        "# Simbay Particle Sweep",
        "",
        "| Particles | Backend | Device | Setup s | Phase4 PF particle-steps/s | Final error % | Prediction ready s | Peak RSS GiB | Peak GPU GiB |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in summaries:
        lines.append(
            "| {particles} | {backend} | {device} | {setup:.3f} | {throughput:.2f} | "
            "{error:.3f} | {ready:.3f} | {rss:.3f} | {gpu:.3f} |".format(
                particles=int(item["particles"]),
                backend=str(item["backend"]),
                device=str(item["setup"]["execution_device"]),
                setup=float(item["setup"]["launch_particles_seconds"]),
                throughput=float(item["phase4"]["pf_update_particle_steps_per_second"]),
                error=float(item["accuracy"]["final_error_pct"]),
                ready=float(item["accuracy"]["prediction_ready_seconds"]),
                rss=float(item["system_summary"]["simbay_process_memory_rss_bytes"]["max"]) / (1024.0**3),
                gpu=float(item["system_summary"]["simbay_gpu_fb_used_bytes"]["max"]) / (1024.0**3),
            )
        )

    lines.extend(
        [
            "",
            "## Included Metrics",
            "",
            "- Setup / launch cost from the existing `setup` stage timing.",
            "- Per-stage CPU, GPU, process RSS, host memory, and Warp memory metrics aggregated from the existing `/metrics` exporter.",
            "- Phase-4 particle-filter throughput from the existing substage workload metrics.",
            "- Accuracy, convergence, uncertainty, resampling, and Warp health from the existing particle-filter metrics.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> int:
    args = _parse_args()
    if args.backend != "mujoco-warp":
        raise SystemExit(
            "This benchmark plotting workflow is configured for mujoco-warp only. "
            "Please run with --backend mujoco-warp."
        )
    particle_counts = [int(value.strip()) for value in args.counts.split(",") if value.strip()]
    if not particle_counts:
        raise SystemExit("No particle counts were provided.")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else Path("temp") / "benchmark_runs" / f"particle-sweep-{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, object]] = []
    for index, particles in enumerate(particle_counts):
        metrics_port = args.metrics_port_base + index
        print(
            f"Running backend={args.backend} particles={particles} metrics_port={metrics_port}",
            flush=True,
        )
        summary = _run_single_benchmark(
            particles=particles,
            backend=args.backend,
            output_dir=output_dir,
            metrics_port=metrics_port,
            sample_interval=args.sample_interval,
            python_executable=args.python,
        )
        summaries.append(summary)
        print(
            "  setup={setup:.3f}s phase4_pf={throughput:.2f} particle-steps/s "
            "final_error={error:.3f}%".format(
                setup=float(summary["setup"]["launch_particles_seconds"]),
                throughput=float(summary["phase4"]["pf_update_particle_steps_per_second"]),
                error=float(summary["accuracy"]["final_error_pct"]),
            ),
            flush=True,
        )

    combined_summary = {"runs": summaries}
    combined_summary_path = output_dir / f"particle_sweep_summary-{timestamp}.json"
    combined_summary_path.write_text(
        json.dumps(combined_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    report_path = _write_markdown_report(output_dir, summaries, timestamp)

    print(f"Saved sweep summary to {combined_summary_path}")
    for item in summaries:
        run_dir = Path(str(item["run_dir"]))
        print(f"Saved run timeseries plot to {run_dir / 'run_timeseries.png'}")
    print(f"Saved markdown report to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
