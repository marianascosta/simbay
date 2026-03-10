# Repository Guidelines

## Project Structure & Module Organization
`main.py` is the current local entrypoint for the MuJoCo demo and estimation flow. Core code lives in `src/` and is split by concern: `src/robots/` for robot interfaces, `src/kinematics/` for IK and engine abstractions, `src/planning/` for trajectory generation, `src/estimation/` for particle-filter logic, and `src/utils/` for shared constants and MuJoCo helpers. Static robot models, meshes, and scene XML files live in `assets/franka_fr3_v2/`. Keep generated backups and local experiment outputs out of git.

## Build, Test, and Development Commands
Use a local virtual environment before running the project.

```sh
python -m venv .venv
source .venv/bin/activate
python main.py
```

`python main.py` launches the MuJoCo viewer and runs the grasp, lift, and mass-estimation sequence. For a quick syntax check before committing, run:

```sh
python -m compileall main.py src
```

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indentation. Use `snake_case` for functions, variables, and module names; use `PascalCase` for classes such as `ParticleFilter` and `FrankaSmartSolver`; use `UPPER_CASE` for constants like `FRANKA_HOME_QPOS`. Keep modules focused on one responsibility and prefer explicit imports from `src.<module>`.

## Testing Guidelines
There is no automated test suite in the repository yet. When adding tests, place them under a top-level `tests/` directory and mirror the source layout, for example `tests/planning/test_trajectory.py`. Prefer `pytest` style tests and cover trajectory generation, solver behavior, and estimator updates without requiring the interactive viewer when possible.

## Commit & Pull Request Guidelines
The current history uses short, imperative commit messages such as `initial commit`. Continue with concise messages like `Add gripper force noise model` or `Refactor MuJoCo robot wrapper`. Keep each commit scoped to one logical change. Pull requests should describe the behavior change, list validation steps, and include screenshots or short recordings when viewer behavior or plotted output changes.

## Configuration & Assets
Do not commit virtual environments, caches, logs, compiled MuJoCo artifacts, or backup XML files. Large asset updates should note source and licensing in the PR when relevant.
