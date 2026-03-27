#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from plot_utils import ensure_run_dir, load_samples, metric_series, relative_times


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot sensor-derived health metrics for one benchmark run.")
    parser.add_argument("--run-dir", required=True, help="Benchmark run directory.")
    parser.add_argument("--output", default="sensor_readings.png", help="Destination PNG path.")
    parser.add_argument("--title", default="Sensor Readings Health", help="Plot title.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    run_dir = Path(args.run_dir)
    ensure_run_dir(run_dir)

    samples = load_samples(run_dir / "metrics_samples.json")
    if not samples:
        raise SystemExit("No metrics samples recorded for this run.")

    times = relative_times(samples)
    sim_force_norm_mean = metric_series(samples, "simbay_sim_force_norm_mean")
    residual_norm_mean = metric_series(samples, "simbay_measurement_residual_norm_mean")
    sim_force_finite = metric_series(samples, "simbay_sim_force_finite_ratio")
    residual_finite = metric_series(samples, "simbay_measurement_residual_finite_ratio")
    likelihood_finite = metric_series(samples, "simbay_likelihood_finite_ratio")
    invalid_sensor_total = metric_series(samples, "simbay_invalid_sensor_events_total")
    sensordata_nonfinite = metric_series(samples, "simbay_sensordata_nonfinite_count")

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    axes[0, 0].plot(times, sim_force_norm_mean, color="tab:blue", linewidth=1.7)
    axes[0, 0].set_title("Sim Force Norm Mean")
    axes[0, 0].set_ylabel("Norm")
    axes[0, 0].grid(True, linestyle=":", alpha=0.6)

    axes[0, 1].plot(times, residual_norm_mean, color="tab:orange", linewidth=1.7)
    axes[0, 1].set_title("Measurement Residual Norm Mean")
    axes[0, 1].set_ylabel("Norm")
    axes[0, 1].grid(True, linestyle=":", alpha=0.6)

    axes[1, 0].plot(times, sim_force_finite, color="tab:green", linewidth=1.5, label="Sim force finite")
    axes[1, 0].plot(
        times,
        residual_finite,
        color="tab:purple",
        linewidth=1.5,
        label="Residual finite",
    )
    axes[1, 0].plot(
        times,
        likelihood_finite,
        color="tab:red",
        linewidth=1.5,
        label="Likelihood finite",
    )
    axes[1, 0].set_title("Finite Ratios")
    axes[1, 0].set_xlabel("Wall-clock since metrics start (s)")
    axes[1, 0].set_ylabel("Ratio [0,1]")
    axes[1, 0].grid(True, linestyle=":", alpha=0.6)
    axes[1, 0].legend(loc="lower left")

    axes[1, 1].plot(times, invalid_sensor_total, color="tab:brown", linewidth=1.7, label="Invalid sensor events")
    axes[1, 1].plot(
        times,
        sensordata_nonfinite,
        color="tab:gray",
        linewidth=1.3,
        label="Non-finite sensordata count",
    )
    axes[1, 1].set_title("Sensor Invalidity Counters")
    axes[1, 1].set_xlabel("Wall-clock since metrics start (s)")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].grid(True, linestyle=":", alpha=0.6)
    axes[1, 1].legend(loc="upper left")

    fig.suptitle(args.title, fontsize=15, fontweight="bold")
    fig.tight_layout()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved sensor readings plot to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
