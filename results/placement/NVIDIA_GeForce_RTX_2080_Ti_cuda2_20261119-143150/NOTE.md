# cuda2 run of 2026-11-19 (both GPUs): what is used

GPU 0 of cuda2 was stuck at a reduced clock during this run (idles in P0 at 1350 MHz since
about 17 Nov; graph loop 54.1 vs 39.6 ms on GPU 1, prefix about 7% slower). Every placement
except `gpu-gpu` had its GPU half on GPU 0, so they are superseded by the clean rerun on GPU 1,
`NVIDIA_GeForce_RTX_2080_Ti_cuda2_20261122-183039`.

Still used: the `gpu-gpu` row (prefix on GPU 0, loop on the healthy GPU 1; it needs both GPUs)
and its prediction from per-device baselines,
`model/placement_report.py <this run> --device-baseline cuda:1=../NVIDIA_GeForce_RTX_2080_Ti_cuda2_20261119-153552`
(best step 120.8 predicted vs 120.6 ms measured). Its prefix is about 5 ms slower than on a
healthy GPU.
