# Decisions

## 2026-03-14

### Change
Added a separate JAX-backed [src/estimation/mjx_filter.py](/Users/marianacosta/Documents/fcul/simbay/simbay/src/estimation/mjx_filter.py) and extended the MJX backend to support device-resident particle propagation plus compiled predict-only rollouts. Updated [main.py](/Users/marianacosta/Documents/fcul/simbay/simbay/main.py) and the MJX notebooks to use that fast path only when the MJX backend is selected.

### Reason
The previous MJX path still stored particle weights in NumPy, converted particles and likelihoods back to host arrays every step, and drove all phases from Python one timestep at a time. That made GPU execution synchronization-heavy and erased most of the benefit of batched MJX stepping.

### Tradeoffs
This adds a second particle-filter implementation instead of folding JAX behavior into the existing reference [src/estimation/particle_filter.py](/Users/marianacosta/Documents/fcul/simbay/simbay/src/estimation/particle_filter.py). The codebase is slightly larger, but the sequential reference path stays unchanged and the GPU-oriented path can make device-specific assumptions without contaminating the fallback implementation.

### Future Considerations
If the MJX path becomes the default execution mode, consider lifting the shared filter interface into a small protocol so both filter implementations can be consumed without backend checks. If lift-phase performance is still dominated by host orchestration, move the measurement update loop itself into a higher-level `jax.lax.scan` path once the observation source is device-friendly.

## 2026-03-14

### Change
Introduced an `MJXBatch` wrapper in [src/estimation/mjx_batch.py](/Users/marianacosta/Documents/fcul/simbay/simbay/src/estimation/mjx_batch.py) and rewired [src/estimation/mjx_particle_filter.py](/Users/marianacosta/Documents/fcul/simbay/simbay/src/estimation/mjx_particle_filter.py) to use it for batched `mjx` stepping and resampling.

### Reason
The previous `FrankaMJXEnv` mixed particle-environment logic with low-level batched `mjx` model/data orchestration. Splitting the backend wrapper out keeps the sequential MuJoCo path untouched, reduces divergence between the two environments, and makes the `mjx` path follow a clearer "wrap once, step many worlds" structure inspired by the tutorial reference.

### Tradeoffs
This adds one new module and an extra handoff between the environment and the backend batch object. It does not implement `mujoco_warp`; the project still uses `mujoco-mjx`, so the notebook was treated as a batching pattern reference rather than a literal dependency target.

### Future Considerations
If the project later adopts `mujoco_warp`, add it as a separate opt-in backend instead of folding it into the `mjx` wrapper. If more particle state parameters become dynamic beyond body mass, extend `MJXBatch` in place rather than pushing backend-specific updates back into the environment.

## 2026-03-14

### Change
Adjusted MJX logging to report the actual execution device selected by [src/estimation/mjx_batch.py](/Users/marianacosta/Documents/fcul/simbay/simbay/src/estimation/mjx_batch.py) instead of JAX's default device, and updated [main.py](/Users/marianacosta/Documents/fcul/simbay/simbay/main.py) to distinguish MJX execution metadata from JAX runtime defaults.

### Reason
On Apple Silicon, JAX initializes a Metal default device even when MJX must fall back to CPU. The previous logs incorrectly reported `METAL` as the simulation backend, which made performance/debugging output misleading.

### Tradeoffs
The logs now contain a few more MJX-specific fields. This is slightly noisier, but it removes ambiguity between "JAX is installed with Metal support" and "MJX is actually executing on Metal."

### Future Considerations
If the project adds more MJX backends or explicit device selection flags, keep the logs centered on the resolved execution device and retain the separate default-JAX fields only as debugging context.
