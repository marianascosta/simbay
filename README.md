# Simbay

MuJoCo-based Franka simulation, planning, and particle-filter mass estimation.

## Setup

This repository uses Poetry with Python 3.10.

```sh
poetry env use python3.10
poetry install
```

Run commands inside the environment with `poetry run ...`.

## Project Layout

- `main.py`: demo entrypoint for the grasp, lift, and estimation flow
- `src/robots/`: robot interfaces
- `src/kinematics/`: IK and solver logic
- `src/planning/`: trajectory planning
- `src/estimation/`: particle-filter environment and estimator
- `models/`: active MuJoCo XML entrypoints (`scene.xml`, `fr3v2_nohand.xml`)
- `assets/franka_fr3_v2/assets/`: FR3 mesh assets used by the XML models

## Running on macOS

For MuJoCo viewer support on macOS, use `mjpython` instead of plain `python`. If Matplotlib raises the `FigureManager` main-thread error, set `MPLBACKEND=Agg`:

```sh
make run-macos
```

`make run-macos` resolves the repository-local `.venv` directly and runs that
environment's `mjpython` wrapper through its matching Python interpreter. This
avoids stale shebang paths after moving or copying the repository, and it does
not depend on `poetry` or `pyenv` resolving the active environment correctly.

## Observability

Start the CPU-friendly local monitoring stack with the Makefile helper:

```sh
make docker-run
```

That command opens the usual app metrics view plus Prometheus, Grafana, and
`node-exporter`. Enable the NVIDIA-only parts (the `cuda` build, `dcgm-exporter`,
and the reserved GPU device) by setting `SIMBAY_ENABLE_GPU=1` before running the
target:

```sh
SIMBAY_ENABLE_GPU=1 make docker-run
```

## Profiling

In-process Nsight Python profiling can be enabled for container runs with:

```sh
SIMBAY_ENABLE_NSIGHT=1 docker compose up --build simbay
```

Or via:

```sh
make docker-simbay-profile
```

The image installs `nsight-python`, and the Warp phase-4 hot path is annotated
in code. Profiling stays disabled by default and is activated only when
`SIMBAY_ENABLE_NSIGHT=1`.
