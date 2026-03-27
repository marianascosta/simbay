# Metrics TODO

- [x] Add `run_id` to every log line.
- [x] Add `run_id` as a common label on exported Prometheus metrics.
- [x] Export workload-normalized throughput metrics for replay/update hot paths.
- [x] Export likelihood health metrics: finite ratio, min/max/mean/std.
- [x] Export invalid sensor / non-finite state counters.
- [x] Export resample rate and particle health metrics.
- [x] Export Warp GPU memory metrics for allocator tracking.
- [x] Export contact-count and valid-force-particle coverage metrics.
- [x] Implement batched particle-mass time-series collection without per-step device-host sync.
- [x] Export low-cardinality particle-mass summaries while writing full batched snapshots to artifacts.
