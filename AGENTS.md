# Repository Guidelines

## Project Structure & Module Organization
`main.py` is the main demo entrypoint for simulation, planning, and mass estimation. Core code lives under `src/`: `robots/` for hardware and simulator interfaces, `kinematics/` for IK engines, `planning/` for trajectory generation, `estimation/` for particle-filter backends, and `utils/` for settings, metrics, logging, and profiling. MuJoCo models and meshes live in `assets/franka_fr3_v2/`. Container and observability files are in `docker-compose*.yml` and `monitoring/`. Analysis and plotting helpers live in `scripts/`.

## Build, Test, and Development Commands
Install dependencies with `poetry install` or `make install-dev`. Run the default app with `poetry run python main.py`.

- `make lint`: check formatting with Black.
- `make format`: reformat the repository with Black (`-l 120`).
- `make local-mujoco`: run the headless MuJoCo backend locally.
- `make local-mujoco-warp`: run the headless MuJoCo-Warp backend locally.
- `make docker-mujoco`: start the CPU Docker service.
- `make docker-mujoco-warp`: start the GPU-enabled Docker service.

Use `SIMBAY_ENABLE_GPU=1 make docker-run` when you need the full monitoring stack with GPU services.

## Coding Style & Naming Conventions
Follow Python conventions already used in `src/`: 4-space indentation, `snake_case` for functions and modules, `PascalCase` for classes, and concise docstrings or comments only where the control flow is non-obvious. Format with Black before submitting; the repository currently uses Black as the only enforced formatter/linter, with a 120-character line length.

## Testing Guidelines
There is no dedicated `tests/` package yet. Until automated tests are added, validate changes by running the relevant entrypoint or backend target and confirming the affected flow works end to end. Put any future tests under `tests/` and name them `test_*.py` so they are easy to discover once pytest is introduced.

## Commit & Pull Request Guidelines
Recent commits use short, imperative subjects such as `Improve docker workflows and Makefile targets` and `Restore Makefile runtime helpers`. Keep commit messages focused on one change. Pull requests should describe the impacted backend or subsystem, list the commands used for validation, and include screenshots or plots when UI, monitoring, or estimation outputs change.

## Configuration Tips
Use Poetry-managed environments and prefer `poetry run ...` for local commands. Backend selection and runtime behavior are controlled through environment variables such as `SIMBAY_BACKEND`, `SIMBAY_HEADLESS`, and `SIMBAY_ENABLE_GPU`; keep new configuration names consistent with that pattern.
