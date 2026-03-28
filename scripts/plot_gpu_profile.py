#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from plot_utils import ensure_run_dir, load_samples, metric_series, relative_times


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot GPU metrics for one benchmark run.")
    parser.add_argument("--run-dir", required=True, help="Benchmark run directory.")
    parser.add_argument("--output", default="gpu_profile.png", help="Destination PNG path.")
    parser.add_argument("--title", default="GPU Profile", help="Plot title.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    run_dir = Path(args.run_dir)
    ensure_run_dir(run_dir)

    samples = load_samples(run_dir / "metrics_samples.json")
    if not samples:
        raise SystemExit("No metrics samples recorded for this run.")

    times = relative_times(samples)
    gpu_util = metric_series(samples, "simbay_gpu_utilization_pct")
    gpu_fb_gib = [value / (1024.0**3) for value in metric_series(samples, "simbay_gpu_fb_used_bytes")]
    gpu_fb_pct = metric_series(samples, "simbay_gpu_fb_utilization_pct")
    gpu_power = metric_series(samples, "simbay_gpu_power_watts")
    gpu_temp = metric_series(samples, "simbay_gpu_temp_celsius")

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    axes[0, 0].plot(times, gpu_util, color="tab:blue", linewidth=1.8)
    axes[0, 0].set_title("GPU Utilization (%)")
    axes[0, 0].set_ylabel("Percent")
    axes[0, 0].grid(True, linestyle=":", alpha=0.6)

    axes[0, 1].plot(times, gpu_fb_gib, color="tab:purple", linewidth=1.8)
    axes[0, 1].set_title("GPU Frame Buffer Used")
    axes[0, 1].set_ylabel("GiB")
    axes[0, 1].grid(True, linestyle=":", alpha=0.6)

    axes[1, 0].plot(times, gpu_fb_pct, color="tab:orange", linewidth=1.8)
    axes[1, 0].set_title("GPU Frame Buffer Utilization (%)")
    axes[1, 0].set_xlabel("Wall-clock since metrics start (s)")
    axes[1, 0].set_ylabel("Percent")
    axes[1, 0].grid(True, linestyle=":", alpha=0.6)

    axes[1, 1].plot(times, gpu_power, color="tab:red", linewidth=1.6, label="Power (W)")
    ax_temp = axes[1, 1].twinx()
    ax_temp.plot(times, gpu_temp, color="tab:green", linewidth=1.6, label="Temp (C)")
    axes[1, 1].set_title("GPU Power and Temperature")
    axes[1, 1].set_xlabel("Wall-clock since metrics start (s)")
    axes[1, 1].set_ylabel("Watts")
    ax_temp.set_ylabel("Celsius")
    axes[1, 1].grid(True, linestyle=":", alpha=0.6)
    power_lines, power_labels = axes[1, 1].get_legend_handles_labels()
    temp_lines, temp_labels = ax_temp.get_legend_handles_labels()
    axes[1, 1].legend(power_lines + temp_lines, power_labels + temp_labels, loc="upper left")

    fig.suptitle(args.title, fontsize=15, fontweight="bold")
    fig.tight_layout()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved GPU profile to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
