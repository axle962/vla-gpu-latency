# Hosting placement report: smolvla

Host `cuda2.cims.nyu.edu` (Intel(R) Xeon(R) CPU E5-2660 v3 @ 2.60GHz, 20 torch threads, 20 CPUs in the affinity mask), GPUs: NVIDIA GeForce RTX 2080 Ti, NVIDIA GeForce RTX 2080 Ti, torch 2.10.0+cu128. 3 cameras, 10 flow steps, chunk 50. Precision per device: {'cuda:0': 'torch.float16', 'cuda:1': 'torch.float16', 'cpu': 'torch.float32'}.

## Measured (median ms)

| placement | prefix on | denoise on | prefix | handoff | denoise_eager | denoise_graph | e2e_eager | e2e_hybrid | eager vs gpu | best vs gpu best | max diff vs first |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gpu | cuda:0 | cuda:0 | 79.29 |  | 401.21 | 54.09 | 481.91 | 134.10 | 1.00x | 1.00x | 0.0e+00 |
| gpu-gpu | cuda:0 | cuda:1 | 79.33 | 1.54 | 413.16 | 39.75 | 495.98 | 120.57 | 1.03x | 0.90x | 0.0e+00 |
| gpu-cpu | cuda:0 | cpu | 79.29 | 2.74 | 651.09 |  | 737.49 |  | 1.53x | 5.50x | 6.0e-04 |
| cpu-gpu | cpu | cuda:0 | 1696.31 | 3.20 | 400.83 | 53.99 | 2040.44 | 1750.72 | 4.23x | 13.06x | 3.9e-01 |
| cpu | cpu | cpu | 1685.41 |  | 662.77 |  | 2388.53 |  | 4.96x | 17.81x | 3.9e-01 |

`best` is e2e_hybrid where the denoise device is a GPU (CUDA graph), else e2e_eager.

Drift check: `gpu` e2e_eager 481.9 ms first, 497.7 ms after all other placements (+3.3%). **More than 3%: later placements ran under different conditions; treat cross-placement comparisons with care.**

## Copy calibration

| src | dst | per-tensor t0 (us) | bandwidth (GB/s) |
|---|---|---|---|
| cpu | cuda:0 | 17.5 | 9.26 |
| cuda:0 | cpu | 18.1 | 2.05 |
| cuda:0 | cuda:1 | 34.4 | 5.85 |

## Split placements predicted from colocated parts

| placement | KV MB | tensors | handoff pred | handoff meas | err | eager pred | eager meas | eager err | hybrid pred | hybrid meas | hybrid err |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gpu-gpu | 4.9 | 33 | 1.98 | 1.54 | +28.9% | 483.7 | 496.0 | -2.5% | 120.8 | 120.6 | +0.2% |
| gpu-cpu | 4.9 | 33 | 3.00 | 2.74 | +9.5% | 745.1 | 737.5 | +1.0% |  |  |  |
| cpu-gpu | 9.9 | 33 | 1.64 | 3.20 | -48.6% | 2088.3 | 2040.4 | +2.3% | 1741.1 | 1750.7 | -0.5% |

Colocated baselines: cuda:1 from `results/placement/NVIDIA_GeForce_RTX_2080_Ti_cuda2_20261119-153552`; others from this run.

## Additivity inside each placement

| placement | sum of parts | e2e_eager | gap |
|---|---|---|---|
| gpu | 480.5 | 481.9 | -0.3% |
| gpu-gpu | 494.0 | 496.0 | -0.4% |
| gpu-cpu | 733.1 | 737.5 | -0.6% |
| cpu-gpu | 2100.3 | 2040.4 | +2.9% |
| cpu | 2348.2 | 2388.5 | -1.7% |

A negative gap means the end-to-end step costs more than its parts timed alone (for example cold caches after the handoff, or the two halves competing for host threads).
