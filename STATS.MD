## CPU 
NUM_PARTICLES = 2
```
Summary
Final estimate: 2.4728 kg
Final error: 1.09%
Lift PF average wall time: 29.986 ms
Lift PF average step rate: 33.35 Hz
Lift PF average CPU time: 42.307 ms
CPU cores available: 8
RSS: 2.55 GiB

MJX Profiling Report
- Particles: 3
- Backend requested: mjx
- XLA_FLAGS: --xla_backend_extra_options=xla_cpu_disable_new_fusion_emitters=true
- JAX default backend: METAL
- JAX local device count: 1
- JAX device 1: platform=METAL kind=Metal
- Setup wall time: 10.418 s
- Setup CPU time: 18.655 s
- Planning wall time: 0.007 s
- Planning CPU time: 0.007 s
- approach steps: 1036
- approach total wall time: 19.400 s
- approach total CPU time: 45.344 s
- approach step avg wall: 18.722 ms
- approach step p95 wall: 16.701 ms
- approach predict avg wall: 18.639 ms
- approach predict p95 wall: 16.572 ms
- approach predict avg CPU: 43.662 ms
- approach predict rate: 53.65 Hz
- descend steps: 778
- descend total wall time: 9.868 s
- descend total CPU time: 23.624 s
- descend step avg wall: 12.680 ms
- descend step p95 wall: 15.143 ms
- descend predict avg wall: 12.603 ms
- descend predict p95 wall: 15.062 ms
- descend predict avg CPU: 30.268 ms
- descend predict rate: 79.35 Hz
- close_gripper steps: 250
- close_gripper total wall time: 6.240 s
- close_gripper total CPU time: 13.989 s
- close_gripper step avg wall: 24.956 ms
- close_gripper step p95 wall: 33.747 ms
- close_gripper predict avg wall: 24.770 ms
- close_gripper predict p95 wall: 33.463 ms
- close_gripper predict avg CPU: 55.519 ms
- close_gripper predict rate: 40.37 Hz
- lift_and_estimate steps: 1335
- lift_and_estimate total wall time: 40.221 s
- lift_and_estimate total CPU time: 56.678 s
- lift full-step avg wall: 30.116 ms
- lift full-step median wall: 26.012 ms
- lift full-step p95 wall: 34.955 ms
- lift filter avg wall: 29.986 ms
- lift filter median wall: 25.890 ms
- lift filter p95 wall: 34.730 ms
- lift filter min/max wall: 22.165/1611.653 ms
- lift filter avg CPU: 42.307 ms
- lift filter p95 CPU: 49.439 ms
- lift filter rate: 33.35 Hz
- True mass: 2.5000 kg
- Final estimate: 2.4728 kg
- Final absolute error: 0.0272 kg
- Final error: 1.09%
- Pre-resample ESS min/max/avg: 1.00/3.00/2.39
- Post-resample ESS avg: 2.43
- Resample trigger count: 30
- Particle state bytes total: 48.00 B
- Particle state bytes per particle: 16.00 B
- Process memory estimate per particle: 0.00 B
- Final RSS: 2.55 GiB
- Snapshot cell_start: rss=2.55 GiB
- Snapshot post_setup: rss=2.55 GiB platform=cpu device=cpu fallback=True bytes_in_use=0.00 B peak_bytes_in_use=0.00 B bytes_limit=0.00 B
- Snapshot after_approach: rss=2.55 GiB platform=cpu device=cpu fallback=True bytes_in_use=0.00 B peak_bytes_in_use=0.00 B bytes_limit=0.00 B
- Snapshot after_descend: rss=2.55 GiB platform=cpu device=cpu fallback=True bytes_in_use=0.00 B peak_bytes_in_use=0.00 B bytes_limit=0.00 B
- Snapshot after_close_gripper: rss=2.55 GiB platform=cpu device=cpu fallback=True bytes_in_use=0.00 B peak_bytes_in_use=0.00 B bytes_limit=0.00 B
- Snapshot lift_step_250: rss=2.55 GiB platform=cpu device=cpu fallback=True bytes_in_use=0.00 B peak_bytes_in_use=0.00 B bytes_limit=0.00 B
- Snapshot lift_step_500: rss=2.55 GiB platform=cpu device=cpu fallback=True bytes_in_use=0.00 B peak_bytes_in_use=0.00 B bytes_limit=0.00 B
- Snapshot lift_step_750: rss=2.55 GiB platform=cpu device=cpu fallback=True bytes_in_use=0.00 B peak_bytes_in_use=0.00 B bytes_limit=0.00 B
- Snapshot lift_step_1000: rss=2.55 GiB platform=cpu device=cpu fallback=True bytes_in_use=0.00 B peak_bytes_in_use=0.00 B bytes_limit=0.00 B
- Snapshot lift_step_1250: rss=2.55 GiB platform=cpu device=cpu fallback=True bytes_in_use=0.00 B peak_bytes_in_use=0.00 B bytes_limit=0.00 B
- Snapshot lift_step_1335: rss=2.55 GiB platform=cpu device=cpu fallback=True bytes_in_use=0.00 B peak_bytes_in_use=0.00 B bytes_limit=0.00 B
- Snapshot post_run: rss=2.55 GiB platform=cpu device=cpu fallback=True bytes_in_use=0.00 B peak_bytes_in_use=0.00 B bytes_limit=0.00 B
- Simulation dt: 0.002000 s
- CPU cores available: 8
- Notebook cell wall time: 86.159 s
- Notebook cell CPU time: 158.311 s
```