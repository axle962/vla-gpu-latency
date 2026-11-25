# Hosting placement report: smolvla

Host `b-19-31` (Intel(R) Xeon(R) CPU @ 2.20GHz, 6 torch threads, 12 CPUs in the affinity mask), GPUs: NVIDIA A100-SXM4-40GB, torch 2.10.0+cu128. 3 cameras, 10 flow steps, chunk 50. Precision per device: {'cuda:0': 'torch.float16'}.

## Measured (median ms)

| placement | prefix on | denoise on | prefix | denoise_eager | denoise_graph | e2e_eager | e2e_hybrid | eager vs gpu | best vs gpu best | max diff vs first |
|---|---|---|---|---|---|---|---|---|---|---|
| gpu | cuda:0 | cuda:0 | 58.03 | 319.88 | 35.62 | 381.61 | 93.73 | 1.00x | 1.00x | 0.0e+00 |

`best` is e2e_hybrid where the denoise device is a GPU (CUDA graph), else e2e_eager.

Drift check: `gpu` e2e_eager 381.6 ms first, 378.0 ms after all other placements (-1.0%). Within 3%: placements are comparable.

## Pipelined requests

| placement | sequential (ms/request) | pipelined (ms/request) | throughput gain | bound max(prefix, handoff+loop) | measured vs bound | max diff vs sequential |
|---|---|---|---|---|---|---|
| gpu | 93.7 | 60.4 | 1.55x | 58.0 | +4.1% | 7.6e+04 |

The prefix of request t+1 runs while the graph loop of request t replays (same GPU: its own stream; two GPUs: the second GPU). The bound assumes the stages do not slow each other; on one GPU they share the SMs, so the measured interval sits above it by the contention. Requests alternate between two observations and every output is compared with the sequential one (0 means no race on the shared KV buffers).

## Additivity inside each placement

| placement | sum of parts | e2e_eager | gap |
|---|---|---|---|
| gpu | 377.9 | 381.6 | -1.0% |

A negative gap means the end-to-end step costs more than its parts timed alone (for example cold caches after the handoff, or the two halves competing for host threads).
