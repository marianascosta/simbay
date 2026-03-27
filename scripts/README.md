# Scripts

Small CLI utilities for benchmark sweeps and PNG plots.

## Requirements

- Run from repository root.
- Use the same Python environment as the project (`poetry run ...` or your `.venv`).
- Plot scripts expect a benchmark run directory containing:
  - `metrics_samples.json`
  - `benchmark_summary.json` (optional for some plots, but recommended)

## 1) Run a benchmark sweep

```bash
python scripts/benchmark_particle_sweep.py --backend mujoco-warp --counts 500
```

This creates run folders under `temp/benchmark_runs/<timestamp>/...`.
Each run folder also gets a time-series overview PNG:
`run_timeseries.png`.

## 2) Run plot scripts for one run

Replace `<run_dir>` with one concrete run folder, for example:
`temp/benchmark_runs/20260326-210000/sweep-mujoco-warp-n100-20260326-210102`

### Memory

```bash
python scripts/plot_memory_profile.py --run-dir <run_dir> --output temp/memory_profile.png
```

### GPU

```bash
python scripts/plot_gpu_profile.py --run-dir <run_dir> --output temp/gpu_profile.png
```

### Particle Mass Estimation

```bash
python scripts/plot_particle_mass_estimation.py --run-dir <run_dir> --output temp/particle_mass_estimation.png
```

Optional true mass reference line:

```bash
python scripts/plot_particle_mass_estimation.py --run-dir <run_dir> --true-mass 2.5 --output temp/particle_mass_estimation.png
```

### Sensor Readings / Health

```bash
python scripts/plot_sensor_readings.py --run-dir <run_dir> --output temp/sensor_readings.png
```

### Steps per Second / Throughput

```bash
python scripts/plot_step_rate_profile.py --run-dir <run_dir> --output temp/step_rate_profile.png
```

## Existing plot scripts

### VRAM profile

```bash
python scripts/plot_vram_profile.py --run-dir <run_dir> --output temp/vram_profile.png
```

## Quick full example

```bash
python scripts/benchmark_particle_sweep.py --backend mujoco-warp --counts 100
python scripts/plot_memory_profile.py --run-dir <run_dir> --output temp/memory_profile.png
python scripts/plot_gpu_profile.py --run-dir <run_dir> --output temp/gpu_profile.png
python scripts/plot_particle_mass_estimation.py --run-dir <run_dir> --output temp/particle_mass_estimation.png
python scripts/plot_sensor_readings.py --run-dir <run_dir> --output temp/sensor_readings.png
python scripts/plot_step_rate_profile.py --run-dir <run_dir> --output temp/step_rate_profile.png
```

If you use Poetry:

```bash
poetry run python scripts/plot_memory_profile.py --run-dir <run_dir> --output temp/memory_profile.png
```

## Time-series first workflow (single context, e.g. 500 particles)

```bash
python scripts/benchmark_particle_sweep.py --backend mujoco-warp --counts 500
```

Then use the generated run folder:

```bash
python scripts/plot_memory_profile.py --run-dir <run_dir> --output <run_dir>/memory_profile.png
python scripts/plot_gpu_profile.py --run-dir <run_dir> --output <run_dir>/gpu_profile.png
python scripts/plot_particle_mass_estimation.py --run-dir <run_dir> --output <run_dir>/particle_mass_estimation.png
python scripts/plot_sensor_readings.py --run-dir <run_dir> --output <run_dir>/sensor_readings.png
python scripts/plot_step_rate_profile.py --run-dir <run_dir> --output <run_dir>/step_rate_profile.png
```
