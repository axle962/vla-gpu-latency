# Hosting placement report: smolvla

Host `cuda2.cims.nyu.edu` (Intel(R) Xeon(R) CPU E5-2660 v3 @ 2.60GHz, 20 torch threads, 20 CPUs in the affinity mask), GPUs: NVIDIA GeForce RTX 2080 Ti, torch 2.10.0+cu128. 3 cameras, 10 flow steps, chunk 50. Precision per device: {'cuda:0': 'torch.float16'}.

## Measured (median ms)

| placement | prefix on | denoise on | prefix | denoise_eager | denoise_graph | e2e_eager | e2e_hybrid | eager vs gpu | best vs gpu best | max diff vs first |
|---|---|---|---|---|---|---|---|---|---|---|
| gpu | cuda:0 | cuda:0 | 74.04 | 402.46 | 39.57 | 482.04 | 114.40 | 1.00x | 1.00x | 0.0e+00 |

`best` is e2e_hybrid where the denoise device is a GPU (CUDA graph), else e2e_eager.

Drift check: `gpu` e2e_eager 482.0 ms first, 480.4 ms after all other placements (-0.3%). Within 3%: placements are comparable.

## Additivity inside each placement

| placement | sum of parts | e2e_eager | gap |
|---|---|---|---|
| gpu | 476.5 | 482.0 | -1.2% |

A negative gap means the end-to-end step costs more than its parts timed alone (for example cold caches after the handoff, or the two halves competing for host threads).
