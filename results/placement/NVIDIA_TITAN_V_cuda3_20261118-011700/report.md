# Hosting placement report: smolvla

Host `cuda3.cims.nyu.edu` (Intel(R) Xeon(R) Gold 5118 CPU @ 2.30GHz, 24 torch threads, 24 CPUs in the affinity mask), GPUs: NVIDIA TITAN V, NVIDIA TITAN V, torch 2.10.0+cu128. 3 cameras, 10 flow steps, chunk 50. Precision per device: {'cuda:0': 'torch.float16', 'cuda:1': 'torch.float16', 'cpu': 'torch.float32'}.

## Measured (median ms)

| placement | prefix on | denoise on | prefix | handoff | denoise_eager | denoise_graph | e2e_eager | e2e_hybrid | eager vs gpu | best vs gpu best | max diff vs first |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gpu | cuda:0 | cuda:0 | 59.17 |  | 280.37 | 66.53 | 340.63 | 126.15 | 1.00x | 1.00x | 0.0e+00 |
| gpu-gpu | cuda:0 | cuda:1 | 68.01 | 1.36 | 373.27 |  | 442.96 |  | 1.30x | 3.51x | 0.0e+00 |
| gpu-cpu | cuda:0 | cpu | 67.66 | 3.55 | 927.56 |  | 1056.29 |  | 3.10x | 8.37x | 7.3e-04 |
| cpu-gpu | cpu | cuda:0 | 2024.76 | 4.29 | 349.02 | 72.94 | 2431.05 | 2111.84 | 7.14x | 16.74x | 3.8e-01 |
| cpu | cpu | cpu | 1996.78 |  | 944.41 |  | 2950.39 |  | 8.66x | 23.39x | 3.9e-01 |

`best` is e2e_hybrid where the denoise device is a GPU (CUDA graph), else e2e_eager.

No drift check in this run (older script). Runs made before the fix of 2026-11-19 profiled each placement right after timing it, which slowed every later placement; compare only within a placement.

## Copy calibration

| src | dst | per-tensor t0 (us) | bandwidth (GB/s) |
|---|---|---|---|
| cpu | cuda:0 | 36.3 | 4.62 |
| cuda:0 | cpu | 34.1 | 1.76 |
| cuda:0 | cuda:1 | 65.3 | 5.95 |

## Split placements predicted from colocated parts

| placement | KV MB | tensors | handoff pred | handoff meas | err | eager pred | eager meas | eager err | hybrid pred | hybrid meas | hybrid err |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gpu-gpu | 4.9 | 33 | 2.98 | 1.36 | +120.0% | 342.5 | 443.0 | -22.7% |  |  |  |
| gpu-cpu | 4.9 | 33 | 3.93 | 3.55 | +10.9% | 1007.5 | 1056.3 | -4.6% |  |  |  |
| cpu-gpu | 9.9 | 33 | 3.33 | 4.29 | -22.2% | 2280.5 | 2431.1 | -6.2% | 2066.6 | 2111.8 | -2.1% |

Colocated baselines: all from this run (a second GPU is assumed as fast as cuda:0).

## Additivity inside each placement

| placement | sum of parts | e2e_eager | gap |
|---|---|---|---|
| gpu | 339.5 | 340.6 | -0.3% |
| gpu-gpu | 442.6 | 443.0 | -0.1% |
| gpu-cpu | 998.8 | 1056.3 | -5.4% |
| cpu-gpu | 2378.1 | 2431.1 | -2.2% |
| cpu | 2941.2 | 2950.4 | -0.3% |

A negative gap means the end-to-end step costs more than its parts timed alone (for example cold caches after the handoff, or the two halves competing for host threads).
