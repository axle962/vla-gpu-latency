# Hosting placement study (pending direction 4): summary

Owner: Archana. Branch `profiling-hosting`. Method and how to run: `docs/placement.md`.

Question: what does a SmolVLA step cost when its two halves (prefix = vision tower + VLM that
fills the KV cache, and the action-expert loop of 10 flow steps) run on different devices,
and can that cost be predicted from single-device measurements?

## Runs

| Run | Machine | Status |
|---|---|---|
| `NVIDIA_A100-SXM4-40GB_b-20-3_20261123-003524` | NYU HPC Cloud Bursting, 2x A100 40 GB (P2P, 267 GB/s), cloud Xeon 2.2 GHz, 12 torch threads | **Clean, all five placements**, drift check 0.0% (job 40230, about 1.3 GPU-h) |
| `NVIDIA_A100-SXM4-40GB_b-20-4_20261123-013831` | NYU HPC, 2x A100 (job 40298) | **Pipelining**, 1 GPU and 2 GPUs, outputs checked (drift 0.0%) |
| `NVIDIA_A100-SXM4-40GB_b-20-3_20261123-010931`, `pipecheck_20261123-012524` | | First pipelining run (timings only) and the diagnostic that found the bug below |
| `NVIDIA_GeForce_RTX_2080_Ti_cuda2_20261122-183039` | cuda2, GPU 1 only (RTX 2080 Ti), Xeon E5-2660 v3, 20 threads (NUMA node 1) | **Headline.** 1 GPU, GPU+CPU, CPU+GPU, CPU only; drift check +0.8% |
| `NVIDIA_GeForce_RTX_2080_Ti_cuda2_20261119-143150` | cuda2, both GPUs; GPU 0 stuck at a reduced clock | **Only the GPU-GPU row** (see its `NOTE.md`); the other rows are superseded |
| `NVIDIA_GeForce_RTX_2080_Ti_cuda2_20261119-153552` | cuda2, GPU 1 only | Colocated baseline used to predict the GPU-GPU split |
| `NVIDIA_TITAN_V_cuda3_20261118-011700` | cuda3, 2x TITAN V, Xeon Gold 5118, 24 threads (node 0) | **Partly usable** (first script version, see its `NOTE.md`): the `gpu` row, the handoffs and the CPU halves only |
| `..._cuda2_20261119-143039`, `..._cuda3_20261118-011511` | | Failed at model load (shared token permission), logs only |

`results/placement/cuda2_gpu{0,1}_clocks.txt` and the `gpus.csv` of the 22 Nov run record GPU 0
idling in P0 at 1350 MHz while GPU 1 drops to P8.

## Results, cuda2 (RTX 2080 Ti, GPU 1), median ms

| Placement | Prefix | Handoff | Loop eager | Loop graph | Step eager | Step best |
|---|---|---|---|---|---|---|
| 1 GPU | 73.5 | | 389.2 | 39.8 | 469.0 | **114.0** |
| prefix GPU 0\*, loop GPU 1 (19 Nov) | 79.3\* | 1.54 | 413.2 | 39.8 | 496.0 | 120.6 |
| prefix GPU, loop CPU | 73.5 | 2.84 | 649.6 (CPU) | | 734.5 | 734.5 |
| prefix CPU, loop GPU | 1,659 (CPU) | 3.17 | 381.4 | 39.9 | 2,011 | 1,720 |
| CPU only | 1,667 | | 657.1 | | 2,348 | 2,348 |

"Best" is eager prefix + CUDA-graph loop when the loop is on a GPU, else eager. \* GPU 0 was
stuck at a reduced clock; its prefix is about 5 ms slower than on GPU 1.
Figure: `paper/figures/placement_cuda2.{pdf,png}` (`model/placement_figure.py RUN --extra
OLD_RUN=gpu-gpu=LABEL`).

## Results, 2x A100 40 GB (NYU HPC), median ms

| Placement | Prefix | Handoff | Loop eager | Loop graph | Step eager | Step best |
|---|---|---|---|---|---|---|
| 1 GPU | 58.4 | | 323.6 | 35.7 | 380.5 | **94.9** |
| prefix GPU 0, loop GPU 1 | 58.3 | 1.05 | 334.0 | 35.9 | 392.8 | 96.1 |
| prefix GPU, loop CPU | 58.1 | 2.56 | 584.7 (CPU) | | 645.3 | 645.3 |
| prefix CPU, loop GPU | 850.6 (CPU) | 4.22 | 319.0 | 35.7 | 1,197 | 916 |
| CPU only | 859.1 | | 589.1 | | 1,454 | 1,454 |

The 1-GPU row matches the project's A100 traces on other nodes (eager 377 to 383, prefix 58,
graph loop 35.6 to 35.8, hybrid 94 ms). Figure: `paper/figures/placement_a100.{pdf,png}`.

TITAN V (cuda3), usable numbers: 1 GPU prefix 59.2, loop eager 280.4, loop graph 66.5, step
eager 340.6, best 126.2 ms; handoff GPU->GPU 1.36, GPU->CPU 3.55, CPU->GPU 4.29 ms; CPU
prefix about 2.0 s, CPU loop 928 to 944 ms.

## Pipelining (2x A100)

The prefix of request t+1 runs while the CUDA-graph loop of request t replays: on a second
stream of the same GPU, or on the second GPU. Requests alternate between two observations;
every pipelined output equals the sequential output for its observation (max |diff| 0), and
the observations really differ (KV 3.5, actions 0.73).

| Setup | One at a time (ms/request) | Pipelined (ms/request) | Throughput gain | Bound max(prefix, handoff + loop) | Measured vs bound |
|---|---|---|---|---|---|
| 1 GPU, two streams | 94.9 | **61.2** | 1.55x | 58.1 | +5.4% |
| 2 GPUs | 95.2 | **61.6** | 1.54x | 58.8 | +4.8% |

## Predicted from single-device parts

`T(A, B) = T_prefix(A alone) + n_tensors * t0 + KV_bytes / BW(A -> B) + T_loop(B alone)`,
with t0 and BW from the copy calibration and each device's own baseline
(`model/placement_report.py RUN [--device-baseline cuda:1=<GPU 1 run>]`).

| Split | Handoff pred / meas | Eager step pred / meas | Best step pred / meas |
|---|---|---|---|
| GPU 0 -> GPU 1 (19 Nov) | 1.98 / 1.54 ms (+29%) | 483.7 / 496.0 (-2.5%) | 120.8 / 120.6 (**+0.2%**) |
| GPU -> CPU | 2.99 / 2.84 ms (+5%) | 733.5 / 734.5 (**-0.1%**) | same as eager |
| CPU -> GPU | 1.63 / 3.17 ms (-48%) | 2,058 / 2,011 (+2.3%) | 1,709 / 1,720 (-0.6%) |
| **A100:** GPU 0 -> GPU 1 | 0.92 / 1.05 ms (-12%) | 382.9 / 392.8 (-2.5%) | 95.0 / 96.1 (-1.1%) |
| **A100:** GPU -> CPU | 2.44 / 2.56 ms (-5%) | 649.9 / 645.3 (+0.7%) | same as eager |
| **A100:** CPU -> GPU | 2.78 / 4.22 ms (-34%) | 1,185 / 1,197 (-0.9%) | 898 / 916 (-2.0%) |

## Findings

1. **The KV handoff is cheap.** 4.9 MB in 33 tensors: 1.0 to 1.5 ms GPU to GPU, 2.6 to 3.6 ms
   GPU to CPU, 3.2 to 4.3 ms CPU to GPU (FP32 KV, cast on the GPU), 1 to 5% of the best
   single-GPU step (95 ms on the A100, 114 ms on the 2080 Ti). Splitting across two A100s
   costs 1.2% (96.1 vs 94.9 ms). It does not make one inference faster, because the halves
   run one after the other. Its value is capacity, which direction 6 (multi-policy hosting,
   `docs/multipolicy.md`) measures directly: with MPS, one GPU already hosts several policies.
2. **The step is additive and predictable.** Parts timed alone sum to the end-to-end step
   within 1.9% in every placement on both clean machines, and the six split placements are
   predicted within 0.1 to 2.5% from single-device runs. The handoff term itself is off by 5
   to 48%, but it is only 1 to 4 ms: it misses the per-tensor overlap on GPU-GPU and the
   FP32 to FP16 cast on CPU-GPU.
3. **A CPU is not a useful host for either half.** Prefix on the CPU: 15x (A100 cloud host)
   to 23x (cuda2) to 34x (cuda3) slower than on the GPU. Action loop on the CPU (FP32, 12 to
   24 threads): 1.7 to 1.8x the dispatch-bound eager GPU loop on cuda2 and the A100 host, 3.3x
   on cuda3, and 14 to 16x the graph loop everywhere. The eager GPU loop is so host bound that
   a CPU gets within 1.8x of it; CUDA graphs remove that. CPU only is 15 to 21x slower than
   the best single-GPU step.
4. **Graph mode exposed a faulty GPU that eager mode hid.** On cuda2, GPU 0 ran the graph loop
   in 54.1 ms and GPU 1 in 39.6 to 39.8 ms (the project's 12 Nov 2080 Ti traces: 40.1 ms),
   while the eager steps were the same (482 ms) because eager is host bound. GPU 0 is not slow
   by design: on 12 and 13 Nov it idled in P8 and held 1,950 MHz under load with a 10-step C++
   graph chain of 38.9 ms, equal to GPU 1. From 17 Nov on it idles stuck in P0 at 1,350 MHz,
   55 C and 64 W with no clock event reason (GPU 1: P8, 300 MHz, reason "idle"); still so on
   22 Nov. 1,950 / 1,350 = 1.44 is close to the 1.37 loop ratio. The headline numbers above are
   therefore from GPU 1 only, and `scripts/run_placement.sh` now refuses a GPU that idles in P0
   and logs every GPU's power state and clock in `gpus.csv`. Results on `main` up to the
   17 Nov traces are unaffected (15 Nov cuda2 graph loops 37 to 40 ms).
5. **The link barely matters; the tensor count does.** cuda2 has no P2P (copies go through
   host memory, 5.9 GB/s), the A100 pair has P2P at 267 GB/s, 45x faster. Yet the handoff
   only drops from 1.54 to 1.05 ms: 33 tensors at about 27 us of fixed cost each (0.9 ms)
   dominate the 4.9 MB transfer (0.02 ms at 267 GB/s). Packing the KV cache into one buffer
   would matter more than a faster link at this size.
6. **Numerics (to check).** Moving the loop to the CPU changes the actions by at most 7e-4.
   Moving the prefix to the CPU (FP32 instead of FP16) changes them by up to 0.39, the same in
   all three runs. So the FP16 prefix, not the loop, is where FP16 and FP32 SmolVLA diverge.
   This may relate to the FP16 vs stock LIBERO gap (50 vs 58% success), untested.

7. **Pipelining gives 1.55x throughput, and a second GPU adds nothing to it.** One request
   at a time, the best step is 95 ms; with the next prefix overlapping the current graph
   loop, a request completes every 61 ms on one GPU (second stream) and 62 ms on two GPUs.
   The prefix is host bound (58 ms of Python and dispatch with the GPU mostly idle), so the
   graph loop (36 ms of GPU work) fits into its idle time on the same GPU. The two-stage bound
   max(prefix, handoff + loop) predicts both within 5.4%. Per-request latency stays about
   95 ms; what grows is throughput. With Shrikar's MPS results (direction 6): one GPU, used
   well, already gives the capacity that splitting a policy across GPUs might have promised.

8. **A bug in this study, found by the pipelining check.** `capture_graph` did not keep the
   graph's static noise input alive, so after the first later allocations every replay
   computed on reused memory. Timings are unaffected (same kernels on the same shapes; the
   fixed rerun matches the earlier one within 1%), and right after capture every graph had
   matched eager exactly, but the graph-mode outputs of the placement runs before
   2026-11-23 01:38 UTC were not valid. Fixed (e3ed026, 02e736c); every run now re-checks the
   graph path against eager after all timing. Aditya's tracers keep their static inputs alive
   and are not affected.

## Open items

- Optional: cuda3 rerun with the fixed script (a third machine; the A100 run already gives a
  second clean one).
- Report cuda2 GPU 0 to helpdesk@cims.nyu.edu; after a reset, rerun the GPU-GPU placement on
  cuda2 (`scripts/run_placement.sh`, both GPUs) for a fully clean split.
- Optional: CPU thread sweep (`scripts/run_placement.sh threads`).
