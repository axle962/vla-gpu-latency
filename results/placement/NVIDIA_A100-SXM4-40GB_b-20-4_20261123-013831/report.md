# Hosting placement report: smolvla

Host `b-20-4` (Intel(R) Xeon(R) CPU @ 2.20GHz, 12 torch threads, 24 CPUs in the affinity mask), GPUs: NVIDIA A100-SXM4-40GB, NVIDIA A100-SXM4-40GB, torch 2.10.0+cu128. 3 cameras, 10 flow steps, chunk 50. Precision per device: {'cuda:0': 'torch.float16', 'cuda:1': 'torch.float16'}.

## Measured (median ms)

| placement | prefix on | denoise on | prefix | handoff | denoise_eager | denoise_graph | e2e_eager | e2e_hybrid | eager vs gpu | best vs gpu best | max diff vs first |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gpu | cuda:0 | cuda:0 | 58.08 |  | 323.18 | 36.00 | 383.41 | 94.87 | 1.00x | 1.00x | 0.0e+00 |
| gpu-gpu | cuda:0 | cuda:1 | 58.82 | 1.06 | 333.64 | 35.62 | 396.54 | 95.20 | 1.03x | 1.00x | 0.0e+00 |

`best` is e2e_hybrid where the denoise device is a GPU (CUDA graph), else e2e_eager.

Drift check: `gpu` e2e_eager 383.4 ms first, 383.6 ms after all other placements (+0.0%). Within 3%: placements are comparable.

## Copy calibration

| src | dst | per-tensor t0 (us) | bandwidth (GB/s) |
|---|---|---|---|
| cuda:0 | cuda:1 | 27.6 | 266.06 |

## Split placements predicted from colocated parts

| placement | KV MB | tensors | handoff pred | handoff meas | err | eager pred | eager meas | eager err | hybrid pred | hybrid meas | hybrid err |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gpu-gpu | 4.9 | 33 | 0.93 | 1.06 | -12.0% | 382.2 | 396.5 | -3.6% | 95.0 | 95.2 | -0.2% |

Colocated baselines: all from this run (a second GPU is assumed as fast as cuda:0).

## Pipelined requests

| placement | sequential (ms/request) | pipelined (ms/request) | throughput gain | bound max(prefix, handoff+loop) | measured vs bound | max diff vs sequential |
|---|---|---|---|---|---|---|
| gpu | 94.9 | 61.2 | 1.55x | 58.1 | +5.4% | 0.0e+00 |
| gpu-gpu | 95.2 | 61.6 | 1.54x | 58.8 | +4.8% | 0.0e+00 |

The prefix of request t+1 runs while the graph loop of request t replays (same GPU: its own stream; two GPUs: the second GPU). The bound assumes the stages do not slow each other; on one GPU they share the SMs, so the measured interval sits above it by the contention. Requests alternate between two observations and every output is compared with the sequential one (0 means no race on the shared KV buffers).

## Additivity inside each placement

| placement | sum of parts | e2e_eager | gap |
|---|---|---|---|
| gpu | 381.3 | 383.4 | -0.6% |
| gpu-gpu | 393.5 | 396.5 | -0.8% |

A negative gap means the end-to-end step costs more than its parts timed alone (for example cold caches after the handoff, or the two halves competing for host threads).
