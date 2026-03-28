#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class MetricSample:
    timestamp: float
    stage: str | None
    metrics: dict[str, float]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show GPU frame-buffer usage for a recorded benchmark run.")
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Run directory produced by the benchmark sweep (contains metrics_samples.json and benchmark_summary.json).",
    )
    parser.add_argument(
        "--output",
        default="vram_profile.png",
        help="Destination PNG path.",
    )
    parser.add_argument(
        "--title",
        default="GPU Frame-buffer Usage",
        help="Optional title for the plot.",
    )
    return parser.parse_args()


def _load_summary(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"Missing benchmark summary at {path}") from exc


def _split_key(key: str) -> tuple[str, dict[str, str]]:
    if "{" not in key:
        return key, {}
    name, raw = key.split("{", 1)
    labels = json.loads("{" + raw)
    return name, labels


def _load_samples(path: Path) -> list[MetricSample]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"Missing metrics_samples.json at {path}") from exc

    samples: list[MetricSample] = []
    for entry in raw:
        metrics = {}
        for key, value in entry.get("metrics", {}).items():
            name, labels = _split_key(key)
            label_text = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
            display_key = f"{name}{{{label_text}}}" if label_text else name
            metrics[display_key] = float(value)
        samples.append(
            MetricSample(
                timestamp=float(entry["timestamp"]),
                stage=entry.get("active_stage"),
                metrics=metrics,
            )
        )
    return samples


def _stage_segments(samples: list[MetricSample], times: list[float]) -> list[tuple[float, float, str]]:
    if not samples:
        return []
    segments: list[tuple[float, float, str]] = []
    current_stage = samples[0].stage or "unknown"
    start_time = times[0]
    for sample, t in zip(samples[1:], times[1:]):
        stage = sample.stage or "unknown"
        if stage != current_stage:
            segments.append((start_time, t, current_stage))
            current_stage = stage
            start_time = t
    segments.append((start_time, times[-1], current_stage))
    return segments


def main() -> int:
    args = _parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise SystemExit(f"Run directory does not exist: {run_dir}")

    summary_path = run_dir / "benchmark_summary.json"
    summary = _load_summary(summary_path)
    metrics_path = run_dir / "metrics_samples.json"
    samples = _load_samples(metrics_path)
    if not samples:
        raise SystemExit("No metrics samples recorded for this run.")

    start_time = samples[0].timestamp
    relative_times = [sample.timestamp - start_time for sample in samples]
    values = []
    for sample in samples:
        value = sample.metrics.get("simbay_gpu_fb_used_bytes")
        # fallback to unlabeled metric
        if value is None:
            value = sample.metrics.get("simbay_gpu_fb_used_bytes{}")
        values.append(value if value is not None else float("nan"))

    segments = _stage_segments(samples, relative_times)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(relative_times, values, marker="o", markersize=3, linewidth=1.5, color="tab:purple")
    for start, end, stage in segments:
        ax.axvspan(start, end, alpha=0.06, label=stage, facecolor="tab:gray")
    ax.set_title(args.title, fontsize=16, fontweight="bold")
    ax.set_xlabel("Wall-clock since metrics start (s)", fontsize=12)
    ax.set_ylabel("GPU FB used (bytes)", fontsize=12)
    ax.grid(True, linestyle=":", alpha=0.6)

    ax.text(
        0.98,
        0.95,
        f"Device: {summary['setup'].get('execution_device','unknown')}\n"
        f"Platform: {summary['setup'].get('execution_platform','unknown')}\n"
        f"Backend: {summary.get('backend','unknown')} ({summary.get('run_metadata',{}).get('backend','')})\n"
        f"Particles: {summary.get('particles',0)}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
    )

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), loc="upper left", fontsize=9)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved VRAM profile to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
