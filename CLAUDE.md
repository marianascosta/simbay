# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Simbay is a MuJoCo-based Franka FR3 robot simulation that estimates object mass via a particle filter during a grasp-and-lift task. It has two execution backends: sequential MuJoCo (CPU, reference implementation) and batched MJX (JAX, GPU-capable).

## Commands

```bash
make install          # Install dependencies via Poetry
make run              # Run simulation
make run-macos        # Run with mjpython (required for MuJoCo viewer on macOS)
make shell            # Enter Poetry shell
make format           # Auto-format with Black (120 char line width)
make lint             # Check formatting
make check            # Compile-check main.py and src/ (python -m compileall)
make setup-precommit  # Install pre-commit hooks
make docker-build     # Build Docker image
make docker-run       # Run in container (headless, 10 GB memory limit)
```

## Environment Variables

- `SIMBAY_USE_MJX` — Enable JAX/MJX batched backend (default: sequential MuJoCo)
- `SIMBAY_PARTICLES` — Number of particles (default: 100)
- `SIMBAY_HEADLESS` — Disable viewer for headless runs
- `SIMBAY_LOG_LEVEL` — Logging level (default: INFO)

## Architecture

**Two parallel execution paths diverge at `main.py`:**

- **Sequential path:** `ParticleFilter` → `FrankaMuJoCoEnv` → N independent `MujocoRobot` instances. Each particle owns a full MuJoCo model (~83 MiB each). This is the correctness reference.
- **MJX path:** `MJXParticleFilter` (in `mjx_filter.py`) → `FrankaMJXEnv` → `MJXBatch`. Stacks N model copies into JAX arrays for vmapped stepping. Keeps particles device-resident.

**Key modules:**
- `src/estimation/particle_filter.py` — Generic SMC filter (reference implementation, must always work)
- `src/estimation/mjx_filter.py` — JAX-native filter with compiled predict-only rollouts
- `src/estimation/mjx_batch.py` — Low-level MJX batched stepping wrapper
- `src/estimation/mujoco_particle_filter.py` / `mjx_particle_filter.py` — Backend-specific particle environments
- `src/kinematics/` — IK solvers; `engines.py` has the numerical stepping algorithms
- `src/planning/` — Trajectory generation via cached IK (`FrankaSmartSolver`)
- `src/robots/` — Robot wrappers; `real_robot.py` is a hardware stub
- `src/utils/constants.py` — `DEFAULT_OBJECT_PROPS`, `FRANKA_HOME_QPOS`

**Runtime flow:** Setup → IK trajectory planning (approach, descend, grip, lift) → Execute phases 1-3 (predict only) → Phase 4 lift (predict → measure → update → resample loop) → Plot results to `temp/particle_filter_evolution.png`.

## Prime Directives

1. **Simplicity over cleverness.** Choose implementations with fewer moving parts.
2. **Never break the sequential path.** The original `ParticleFilter` + `FrankaMuJoCoEnv` must remain runnable — it is the correctness reference and local debug fallback.
3. **One concern per file.** Don't combine actor, orchestration, and filter logic in one module.
4. **No premature abstractions.** No base classes, registries, or plugin systems unless explicitly required.
5. **Preserve existing interfaces.** `main.py` public behavior must not change. New parameters must be additive and opt-in.

## Code Style

- Black formatter, 120 character line width
- Always use JAX/MJX for new physics work — no MuJoCo CPU fallback paths
- Robot assets are in `assets/franka_fr3_v2/` (`scene.xml` for full scene, `fr3v2.xml` for IK)
- Logs go to `logs/simbay.log` (rotating, 5 MB × 5 backups)

## Memory Considerations

Each MuJoCo particle holds ~83 MiB (model ~72 MB + arena ~14 MB). At 100 particles that's ~8 GiB. The MJX path reduces this via batched JAX arrays but still duplicates models. Docker is capped at 10 GB.
