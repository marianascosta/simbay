#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from plot_utils import ensure_run_dir, load_samples, metric_series, relative_times


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot step-rate metrics for one benchmark run.")
    parser.add_argument("--run-dir", required=True, help="Benchmark run directory.")
    parser.add_argument("--output", default="step_rate_profile.png", help="Destination PNG path.")
    parser.add_argument("--title", default="Step Rate Profile", help="Plot title.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    run_dir = Path(args.run_dir)
    ensure_run_dir(run_dir)

    samples = load_samples(run_dir / "metrics_samples.json")
    if not samples:
        raise SystemExit("No metrics samples recorded for this run.")

    times = relative_times(samples)
    phase4_step_rate = metric_series(samples, "simbay_phase4_step_rate_hz")
    phase4_particle_step_rate = metric_series(samples, "simbay_phase4_particle_steps_per_second")
    phase4_ms_per_particle_step = metric_series(samples, "simbay_phase4_ms_per_particle_step")

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(times, phase4_step_rate, color="tab:blue", linewidth=1.9, label="Step rate (Hz)")
    axes[0].plot(
        times,
        phase4_particle_step_rate,
        color="tab:orange",
        linewidth=1.7,
        label="Particle-steps/s",
    )
    axes[0].set_ylabel("Rate")
    axes[0].set_title(args.title, fontsize=15, fontweight="bold")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(loc="upper right")

    axes[1].plot(
        times,
        phase4_ms_per_particle_step,
        color="tab:red",
        linewidth=1.7,
    )
    axes[1].set_xlabel("Wall-clock since metrics start (s)")
    axes[1].set_ylabel("ms / particle-step")
    axes[1].set_title("Phase-4 Latency per Particle-step")
    axes[1].grid(True, linestyle=":", alpha=0.6)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved step-rate profile to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
