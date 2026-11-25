# Hosting placement report: smolvla

Host `cuda2.cims.nyu.edu` (Intel(R) Xeon(R) CPU E5-2660 v3 @ 2.60GHz, 20 torch threads, 20 CPUs in the affinity mask), GPUs: NVIDIA GeForce RTX 2080 Ti, torch 2.10.0+cu128. 3 cameras, 10 flow steps, chunk 50. Precision per device: {'cuda:0': 'torch.float16', 'cpu': 'torch.float32'}.

## Measured (median ms)

| placement | prefix on | denoise on | prefix | handoff | denoise_eager | denoise_graph | e2e_eager | e2e_hybrid | eager vs gpu | best vs gpu best | max diff vs first |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gpu | cuda:0 | cuda:0 | 73.46 |  | 389.19 | 39.84 | 468.99 | 114.02 | 1.00x | 1.00x | 0.0e+00 |
| gpu-cpu | cuda:0 | cpu | 73.51 | 2.84 | 649.62 |  | 734.54 |  | 1.57x | 6.44x | 6.7e-04 |
| cpu-gpu | cpu | cuda:0 | 1659.17 | 3.17 | 381.39 | 39.87 | 2011.07 | 1719.84 | 4.29x | 15.08x | 3.9e-01 |
| cpu | cpu | cpu | 1667.36 |  | 657.06 |  | 2348.34 |  | 5.01x | 20.60x | 3.9e-01 |

`best` is e2e_hybrid where the denoise device is a GPU (CUDA graph), else e2e_eager.

Drift check: `gpu` e2e_eager 469.0 ms first, 472.9 ms after all other placements (+0.8%). Within 3%: placements are comparable.

## Copy calibration

| src | dst | per-tensor t0 (us) | bandwidth (GB/s) |
|---|---|---|---|
| cpu | cuda:0 | 17.3 | 9.30 |
| cuda:0 | cpu | 17.8 | 2.05 |

## Split placements predicted from colocated parts

| placement | KV MB | tensors | handoff pred | handoff meas | err | eager pred | eager meas | eager err | hybrid pred | hybrid meas | hybrid err |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gpu-cpu | 4.9 | 33 | 2.99 | 2.84 | +5.2% | 733.5 | 734.5 | -0.1% |  |  |  |
| cpu-gpu | 9.9 | 33 | 1.63 | 3.17 | -48.4% | 2058.2 | 2011.1 | +2.3% | 1708.8 | 1719.8 | -0.6% |

Colocated baselines: all from this run (a second GPU is assumed as fast as cuda:0).

## Additivity inside each placement

| placement | sum of parts | e2e_eager | gap |
|---|---|---|---|
| gpu | 462.6 | 469.0 | -1.4% |
| gpu-cpu | 726.0 | 734.5 | -1.2% |
| cpu-gpu | 2043.7 | 2011.1 | +1.6% |
| cpu | 2324.4 | 2348.3 | -1.0% |

A negative gap means the end-to-end step costs more than its parts timed alone (for example cold caches after the handoff, or the two halves competing for host threads).
