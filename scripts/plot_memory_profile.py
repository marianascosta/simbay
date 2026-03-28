#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from plot_utils import (
    ensure_run_dir,
    load_samples,
    metric_max_across_labels,
    metric_series,
    relative_times,
    stage_segments,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot memory usage for one benchmark run.")
    parser.add_argument("--run-dir", required=True, help="Benchmark run directory.")
    parser.add_argument("--output", default="memory_profile.png", help="Destination PNG path.")
    parser.add_argument("--title", default="Memory Profile", help="Plot title.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    run_dir = Path(args.run_dir)
    ensure_run_dir(run_dir)

    samples = load_samples(run_dir / "metrics_samples.json")
    if not samples:
        raise SystemExit("No metrics samples recorded for this run.")

    times = relative_times(samples)
    rss_bytes = metric_series(samples, "simbay_process_memory_rss_bytes")
    host_bytes = metric_series(samples, "simbay_host_memory_used_bytes")
    warp_used_bytes = [
        float("nan") if value is None else float(value)
        for value in (metric_max_across_labels(sample, "simbay_warp_bytes_in_use") for sample in samples)
    ]
    warp_peak_bytes = [
        float("nan") if value is None else float(value)
        for value in (metric_max_across_labels(sample, "simbay_warp_peak_bytes_in_use") for sample in samples)
    ]

    gib = 1024.0**3
    rss_gib = [value / gib for value in rss_bytes]
    host_gib = [value / gib for value in host_bytes]
    warp_used_gib = [value / gib for value in warp_used_bytes]
    warp_peak_gib = [value / gib for value in warp_peak_bytes]

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(times, rss_gib, linewidth=1.8, color="tab:blue", label="Process RSS")
    axes[0].plot(times, host_gib, linewidth=1.5, color="tab:gray", label="Host memory used")
    axes[0].set_ylabel("GiB")
    axes[0].set_title(args.title, fontsize=15, fontweight="bold")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(loc="upper left")

    axes[1].plot(times, warp_used_gib, linewidth=1.8, color="tab:green", label="Warp bytes in use")
    axes[1].plot(times, warp_peak_gib, linewidth=1.2, color="tab:orange", label="Warp peak bytes")
    axes[1].set_xlabel("Wall-clock since metrics start (s)")
    axes[1].set_ylabel("GiB")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(loc="upper left")

    for start, end, _stage in stage_segments(samples, times):
        for axis in axes:
            axis.axvspan(start, end, alpha=0.03, facecolor="black")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved memory profile to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
