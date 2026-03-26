#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from plot_utils import ensure_run_dir, load_samples, metric_series, relative_times


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot particle mass estimation metrics for one benchmark run."
    )
    parser.add_argument("--run-dir", required=True, help="Benchmark run directory.")
    parser.add_argument(
        "--output",
        default="particle_mass_estimation.png",
        help="Destination PNG path.",
    )
    parser.add_argument("--title", default="Particle Mass Estimation", help="Plot title.")
    parser.add_argument(
        "--true-mass",
        type=float,
        default=None,
        help="Optional true mass in kilograms.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    run_dir = Path(args.run_dir)
    ensure_run_dir(run_dir)

    samples = load_samples(run_dir / "metrics_samples.json")
    if not samples:
        raise SystemExit("No metrics samples recorded for this run.")

    times = relative_times(samples)
    estimate = metric_series(samples, "simbay_mass_estimate_kg")
    particle_p10 = metric_series(samples, "simbay_particle_mass_p10_kg")
    particle_p50 = metric_series(samples, "simbay_particle_mass_p50_kg")
    particle_p90 = metric_series(samples, "simbay_particle_mass_p90_kg")
    rel_error = metric_series(samples, "simbay_mass_rel_error_pct")

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(times, estimate, color="tab:red", linewidth=2.0, label="Mass estimate")
    axes[0].plot(times, particle_p50, color="tab:blue", linewidth=1.4, label="Particle p50")
    axes[0].fill_between(
        times,
        particle_p10,
        particle_p90,
        color="tab:blue",
        alpha=0.2,
        label="Particle p10-p90",
    )
    if args.true_mass is not None:
        axes[0].axhline(
            args.true_mass,
            color="tab:green",
            linestyle="--",
            linewidth=1.5,
            label="True mass",
        )
    axes[0].set_ylabel("Mass (kg)")
    axes[0].set_title(args.title, fontsize=15, fontweight="bold")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(loc="upper right")

    axes[1].plot(times, rel_error, color="tab:orange", linewidth=1.8)
    axes[1].set_xlabel("Wall-clock since metrics start (s)")
    axes[1].set_ylabel("Relative error (%)")
    axes[1].set_title("Mass Relative Error")
    axes[1].grid(True, linestyle=":", alpha=0.6)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved particle mass estimation plot to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
