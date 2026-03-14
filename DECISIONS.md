# Decisions

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
