# Decisions

## 2026-03-18

### Change
Reworked local observability to run as a Compose-managed stack with checked-in Prometheus and Grafana provisioning via [docker-compose.yml](/Users/marianacosta/Documents/fcul/simbay/simbay/docker-compose.yml), [monitoring/prometheus/prometheus.compose.yml](/Users/marianacosta/Documents/fcul/simbay/simbay/monitoring/prometheus/prometheus.compose.yml), and [monitoring/grafana/provisioning/dashboards/simbay.yml](/Users/marianacosta/Documents/fcul/simbay/simbay/monitoring/grafana/provisioning/dashboards/simbay.yml). Updated [docs/observability.md](/Users/marianacosta/Documents/fcul/simbay/simbay/docs/observability.md) to use a one-command `docker compose` workflow.

### Reason
The previous host-local setup required manual service startup, OS-specific package installation, and manual Grafana configuration. A Compose-first workflow is simpler to run, easier to reproduce across machines, and better aligned with the project's existing Docker-based local development path.

### Tradeoffs
This makes the local monitoring path more container-centric and introduces an optional GPU-specific Compose profile for `dcgm-exporter`. Host-native installs remain possible, but the documented happy path now depends on Docker and Compose rather than directly installed binaries.

### Future Considerations
If the Compose stack becomes the default team workflow, consider adding healthchecks and a small CPU-only override file for machines without GPU container support instead of expanding the base file with more conditionals.

## 2026-03-16

### Change
Added a minimal Prometheus-compatible observability layer with [src/utils/metrics.py](/Users/marianacosta/Documents/fcul/simbay/simbay/src/utils/metrics.py), stage-aware instrumentation in [main.py](/Users/marianacosta/Documents/fcul/simbay/simbay/main.py), and host-local Prometheus/Grafana configuration under [monitoring/](/Users/marianacosta/Documents/fcul/simbay/simbay/monitoring/).

### Reason
The project needs stable resource visibility by stage without relying on parsing log lines. A small in-process metrics surface lets Prometheus and Grafana correlate simulation phases with host and GPU exporters while keeping `main.py` flat and preserving the existing sequential execution path.

### Tradeoffs
This adds a small amount of app-level instrumentation and a lightweight custom metrics endpoint instead of pulling in a larger Python metrics dependency. The result is intentionally narrow: the app exports simulation context and phase-4 filter state, while CPU and GPU hardware telemetry stay outside the process in standard host exporters.

### Future Considerations
If the observability stack becomes a core workflow, add a provisioned Grafana dashboard next rather than expanding the app metrics set aggressively. If logs also need to become queryable in Grafana, add Loki as a separate step instead of folding log shipping into the metrics helper.

## 2026-03-17

### Change
Added a reusable local-development Grafana dashboard at [monitoring/grafana/dashboards/simbay-overview.json](/Users/marianacosta/Documents/fcul/simbay/simbay/monitoring/grafana/dashboards/simbay-overview.json) and aligned the local Prometheus datasource reference in [monitoring/grafana/provisioning/datasources/prometheus.yml](/Users/marianacosta/Documents/fcul/simbay/simbay/monitoring/grafana/provisioning/datasources/prometheus.yml) to `localhost:9090`.

### Reason
The metrics endpoint is more useful when developers can immediately correlate prediction quality, phase timing, RSS, MJX allocator memory, and host/GPU resource pressure in a single dashboard. A checked-in dashboard JSON keeps the local observability workflow repeatable without forcing a Docker-specific deployment path.

### Tradeoffs
The dashboard includes panels for `node-exporter` and `dcgm-exporter`, so those graphs will be empty until the corresponding local exporters are running. The dashboard is imported manually rather than auto-provisioned because local Grafana installations do not share a stable filesystem layout.

### Future Considerations
If the dashboard stabilizes as the team default, add one or two focused variants for GPU-heavy benchmarking and CPU-reference debugging instead of expanding a single dashboard indefinitely.

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
