# Hosting placement study (pending direction 4): summary

Owner: Archana. Branch `profiling-hosting`. Method and how to run: `docs/placement.md`.

Question: what does a SmolVLA step cost when its two halves (prefix = vision tower + VLM that
fills the KV cache, and the action-expert loop of 10 flow steps) run on different devices,
and can that cost be predicted from single-device measurements?

## Runs

| Run | Machine | Status |
|---|---|---|
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

TITAN V (cuda3), usable numbers: 1 GPU prefix 59.2, loop eager 280.4, loop graph 66.5, step
eager 340.6, best 126.2 ms; handoff GPU->GPU 1.36, GPU->CPU 3.55, CPU->GPU 4.29 ms; CPU
prefix about 2.0 s, CPU loop 928 to 944 ms.

## Predicted from single-device parts (cuda2)

`T(A, B) = T_prefix(A alone) + n_tensors * t0 + KV_bytes / BW(A -> B) + T_loop(B alone)`,
with t0 and BW from the copy calibration and each device's own baseline
(`model/placement_report.py RUN [--device-baseline cuda:1=<GPU 1 run>]`).

| Split | Handoff pred / meas | Eager step pred / meas | Best step pred / meas |
|---|---|---|---|
| GPU 0 -> GPU 1 (19 Nov) | 1.98 / 1.54 ms (+29%) | 483.7 / 496.0 (-2.5%) | 120.8 / 120.6 (**+0.2%**) |
| GPU -> CPU | 2.99 / 2.84 ms (+5%) | 733.5 / 734.5 (**-0.1%**) | same as eager |
| CPU -> GPU | 1.63 / 3.17 ms (-48%) | 2,058 / 2,011 (+2.3%) | 1,709 / 1,720 (-0.6%) |

## Findings

1. **The KV handoff is cheap.** 4.9 MB in 33 tensors: 1.4 to 1.5 ms GPU to GPU, 2.8 to 3.6 ms
   GPU to CPU, 3.2 to 4.3 ms CPU to GPU (FP32 KV, cast on the GPU). That is 1 to 3% of the best
   single-GPU step (114 ms). Splitting across two GPUs therefore costs only the handoff; it
   does not make one inference faster, because the halves run one after the other. Its value
   is capacity, which direction 6 (multi-policy hosting, `docs/multipolicy.md`) measures
   directly: with MPS, one GPU already hosts several policies.
2. **The step is additive and predictable.** Parts timed alone sum to the end-to-end step
   within 1.6% in every placement of the clean run, and split placements are predicted within
   0.1 to 2.5% from single-device runs. The handoff term itself is off by 5 to 48%, but it is
   only 1.5 to 3 ms: it overestimates GPU-GPU (per-tensor setup overlaps the transfer) and
   underestimates CPU-GPU (the model has no term for the FP32 to FP16 cast).
3. **A CPU is not a useful host for either half.** Prefix on the CPU: 23x (2080 Ti host) to
   34x (TITAN V host) slower than on the GPU. Action loop on the CPU (FP32, 20 to 24 threads):
   1.7x (cuda2) to 3.3x (cuda3) the dispatch-bound eager GPU loop, and 14 to 16x the graph
   loop. The eager GPU loop is so host bound that a CPU gets within 1.7x of it on cuda2; CUDA
   graphs remove that. CPU only is 21x slower than the best single-GPU step.
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
5. **No P2P on cuda2** (`nvidia-smi topo -p2p r`: chipset not supported), so GPU 0 to GPU 1
   copies go through host memory; cuda3 reports P2P OK although its GPUs sit on different
   sockets. Both handoffs are about 1.5 ms, so it does not matter at this KV size.
6. **Numerics (to check).** Moving the loop to the CPU changes the actions by at most 7e-4.
   Moving the prefix to the CPU (FP32 instead of FP16) changes them by up to 0.39, the same in
   all three runs. So the FP16 prefix, not the loop, is where FP16 and FP32 SmolVLA diverge.
   This may relate to the FP16 vs stock LIBERO gap (50 vs 58% success), untested.

## Open items

- cuda3 rerun with the fixed script (needs both TITAN Vs idle, about 20 min) for a second
  clean machine.
- Report cuda2 GPU 0 to helpdesk@cims.nyu.edu; after a reset, rerun the GPU-GPU placement on
  cuda2 (`scripts/run_placement.sh`, both GPUs) for a fully clean split.
- Optional: CPU thread sweep (`scripts/run_placement.sh threads`).
