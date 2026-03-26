#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a detailed mass-estimate timeline from the recorded particle snapshots."
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Directory containing the mass-timeseries .npz artifacts (e.g. temp/mass_timeseries/<run_id>).",
    )
    parser.add_argument(
        "--output",
        default="mass_timeseries.png",
        help="Path for the generated PNG (defaults to mass_timeseries.png).",
    )
    parser.add_argument(
        "--title",
        default="Particle Mass Estimates Over Time",
        help="Optional plot title.",
    )
    parser.add_argument(
        "--true-mass",
        type=float,
        default=None,
        help="Optional true mass to draw as a horizontal reference line.",
    )
    return parser.parse_args()


def _load_snapshots(directory: Path) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
]:
    files = sorted(directory.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files under {directory}")

    all_steps: list[np.ndarray] = []
    all_mass_snapshots: list[np.ndarray] = []
    for file_path in files:
        data = np.load(str(file_path))
        steps = data["steps"]
        masses = data["masses"]
        if masses.ndim != 2:
            raise ValueError(f"Unexpected masses shape in {file_path}: {masses.shape}")
        if steps.shape[0] != masses.shape[0]:
            raise ValueError(
                f"Steps length ({steps.shape[0]}) does not match masses rows ({masses.shape[0]}) in {file_path}"
            )
        all_steps.append(steps)
        all_mass_snapshots.append(masses)

    steps_array = np.concatenate(all_steps, axis=0)
    snapshots = np.concatenate(all_mass_snapshots, axis=0)
    per_snapshot_mean = snapshots.mean(axis=1)
    per_snapshot_p10 = np.percentile(snapshots, 10, axis=1)
    per_snapshot_p90 = np.percentile(snapshots, 90, axis=1)

    particles_per_snapshot = snapshots.shape[1]
    footprint = np.repeat(steps_array, particles_per_snapshot)
    flattened_masses = snapshots.reshape(-1)

    return (
        steps_array,
        flattened_masses,
        footprint,
        per_snapshot_mean,
        per_snapshot_p10,
        per_snapshot_p90,
        particles_per_snapshot,
    )


def main() -> int:
    args = _parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise SystemExit(f"Run directory does not exist: {run_dir}")

    (
        steps,
        flattened_masses,
        expanded_steps,
        mean_series,
        p10_series,
        p90_series,
        particle_count,
    ) = _load_snapshots(run_dir)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(
        expanded_steps,
        flattened_masses,
        s=10,
        alpha=0.04,
        color="tab:blue",
        label="Particle samples",
    )
    ax.plot(steps, mean_series, color="tab:red", linewidth=2.2, label="Mean estimate")
    ax.fill_between(steps, p10_series, p90_series, color="tab:red", alpha=0.15, label="10-90th percentile")
    if args.true_mass is not None:
        ax.axhline(args.true_mass, color="tab:green", linestyle="--", linewidth=1.6, label="True mass")
    ax.set_title(args.title, fontsize=16, fontweight="bold")
    ax.set_xlabel("Phase 4 step", fontsize=12)
    ax.set_ylabel("Estimated particle mass (kg)", fontsize=12)
    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.7)
    ax.legend(loc="upper right", fontsize=10)
    ax.text(
        0.98,
        0.02,
        f"Particles per snapshot: {particle_count}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
    )
    fig.tight_layout()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved mass-timeseries plot to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
