# Hosting placement report: smolvla

Host `b-20-3` (Intel(R) Xeon(R) CPU @ 2.20GHz, 12 torch threads, 24 CPUs in the affinity mask), GPUs: NVIDIA A100-SXM4-40GB, NVIDIA A100-SXM4-40GB, torch 2.10.0+cu128. 3 cameras, 10 flow steps, chunk 50. Precision per device: {'cuda:0': 'torch.float16', 'cuda:1': 'torch.float16', 'cpu': 'torch.float32'}.

## Measured (median ms)

| placement | prefix on | denoise on | prefix | handoff | denoise_eager | denoise_graph | e2e_eager | e2e_hybrid | eager vs gpu | best vs gpu best | max diff vs first |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gpu | cuda:0 | cuda:0 | 58.39 |  | 323.59 | 35.73 | 380.54 | 94.93 | 1.00x | 1.00x | 0.0e+00 |
| gpu-gpu | cuda:0 | cuda:1 | 58.26 | 1.05 | 334.01 | 35.86 | 392.83 | 96.06 | 1.03x | 1.01x | 0.0e+00 |
| gpu-cpu | cuda:0 | cpu | 58.11 | 2.56 | 584.71 |  | 645.28 |  | 1.70x | 6.80x | 5.8e-04 |
| cpu-gpu | cpu | cuda:0 | 850.63 | 4.22 | 319.03 | 35.71 | 1196.70 | 915.83 | 3.14x | 9.65x | 3.9e-01 |
| cpu | cpu | cpu | 859.07 |  | 589.09 |  | 1453.96 |  | 3.82x | 15.32x | 3.9e-01 |

`best` is e2e_hybrid where the denoise device is a GPU (CUDA graph), else e2e_eager.

Drift check: `gpu` e2e_eager 380.5 ms first, 380.6 ms after all other placements (+0.0%). Within 3%: placements are comparable.

## Copy calibration

| src | dst | per-tensor t0 (us) | bandwidth (GB/s) |
|---|---|---|---|
| cpu | cuda:0 | 20.1 | 4.67 |
| cuda:0 | cpu | 20.8 | 2.82 |
| cuda:0 | cuda:1 | 27.3 | 267.26 |

## Split placements predicted from colocated parts

| placement | KV MB | tensors | handoff pred | handoff meas | err | eager pred | eager meas | eager err | hybrid pred | hybrid meas | hybrid err |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gpu-gpu | 4.9 | 33 | 0.92 | 1.05 | -12.2% | 382.9 | 392.8 | -2.5% | 95.0 | 96.1 | -1.1% |
| gpu-cpu | 4.9 | 33 | 2.44 | 2.56 | -4.7% | 649.9 | 645.3 | +0.7% |  |  |  |
| cpu-gpu | 9.9 | 33 | 2.78 | 4.22 | -34.2% | 1185.4 | 1196.7 | -0.9% | 897.6 | 915.8 | -2.0% |

Colocated baselines: all from this run (a second GPU is assumed as fast as cuda:0).

## Additivity inside each placement

| placement | sum of parts | e2e_eager | gap |
|---|---|---|---|
| gpu | 382.0 | 380.5 | +0.4% |
| gpu-gpu | 393.3 | 392.8 | +0.1% |
| gpu-cpu | 645.4 | 645.3 | +0.0% |
| cpu-gpu | 1173.9 | 1196.7 | -1.9% |
| cpu | 1448.2 | 1454.0 | -0.4% |

A negative gap means the end-to-end step costs more than its parts timed alone (for example cold caches after the handoff, or the two halves competing for host threads).
