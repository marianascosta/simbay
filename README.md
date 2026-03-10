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
- `assets/franka_fr3_v2/`: MuJoCo XML, meshes, and robot assets

## Running on macOS

For MuJoCo viewer support on macOS, use `mjpython` instead of plain `python`. If Matplotlib raises the `FigureManager` main-thread error, set `MPLBACKEND=Agg`:

```sh
make run-macos
```

`make run-macos` resolves the repository-local `.venv` directly and runs that
environment's `mjpython` wrapper through its matching Python interpreter. This
avoids stale shebang paths after moving or copying the repository, and it does
not depend on `poetry` or `pyenv` resolving the active environment correctly.
