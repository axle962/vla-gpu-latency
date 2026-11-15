# Excluded runs

Kept for the record, outside the globs the model scripts use.

- `smolvlacompile_cam3_steps10_chunk50/NVIDIA_GeForce_RTX_2080_Ti_cuda2_20261115-142823`:
  its timing phase overlapped an Nsight Systems capture (workloads/nsys_smolvla.py) on the same
  CPU NUMA node of cuda2 (2026-11-15 ~14:25 to 14:28). Eager loop 520 ms and compiled eager
  launch 148 ms vs 401 to 413 ms and 115 to 119 ms in the two clean runs (14:26 and the rerun
  at 15:36); GPU-side times unchanged.
- `pi0_libero_fp16_steps10_chunk50/NVIDIA_GeForce_RTX_2080_Ti_cuda2_20261115-164057`: started
  16:37:26 and overlapped a LIBERO rendering test on the same GPU (cuda2 GPU 0) and CPU node,
  ~16:38 to 16:40. The clean repeat of this config on cuda2 is the 17:01 run.
