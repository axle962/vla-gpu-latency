# Hosting placement study (pending direction 4): summary

Owner: Archana. Branch `profiling-hosting`. Method and how to run: `docs/placement.md`.

Question: what does a SmolVLA step cost when its two halves (prefix = vision tower + VLM that
fills the KV cache, and the action-expert loop of 10 flow steps) run on different devices,
and can that cost be predicted from single-device measurements?

## Runs

| Run | Machine | Status |
|---|---|---|
| `NVIDIA_GeForce_RTX_2080_Ti_cuda2_20261119-143150` | cuda2, 2x RTX 2080 Ti, Xeon E5-2660 v3, 20 threads (NUMA node 1) | **Clean.** All 5 placements, graphs on both GPUs, drift check +3.3% |
| `NVIDIA_GeForce_RTX_2080_Ti_cuda2_20261119-153552` | cuda2, GPU 1 only | **Clean.** Colocated baseline on GPU 1, drift -0.3% |
| `NVIDIA_TITAN_V_cuda3_20261118-011700` | cuda3, 2x TITAN V, Xeon Gold 5118, 24 threads (node 0) | **Partly usable** (first script version, see its `NOTE.md`): the `gpu` row, the handoffs and the CPU halves only |
| `..._cuda2_20261119-143039`, `..._cuda3_20261118-011511` | | Failed at model load (shared token permission), logs only |

`results/placement/cuda2_gpu{0,1}_clocks.txt`: idle clock state of the two cuda2 GPUs.

## Results, cuda2 (RTX 2080 Ti), median ms

| Placement | Prefix | Handoff | Loop eager | Loop graph | Step eager | Step best |
|---|---|---|---|---|---|---|
| 1 GPU (GPU 1) | 74.0 | | 402.5 | 39.6 | 482.0 | **114.4** |
| 1 GPU (GPU 0) | 79.3 | | 401.2 | 54.1 | 481.9 | 134.1 |
| prefix GPU 0, loop GPU 1 | 79.3 | 1.54 | 413.2 | 39.8 | 496.0 | 120.6 |
| prefix GPU, loop CPU | 79.3 | 2.74 | 651.1 (CPU) | | 737.5 | 737.5 |
| prefix CPU, loop GPU | 1,696 (CPU) | 3.20 | 400.8 | 54.0 | 2,040 | 1,751 |
| CPU only | 1,685 | | 662.8 | | 2,389 | 2,389 |

"Best" is eager prefix + CUDA-graph loop when the loop is on a GPU, else eager.
Figure: `paper/figures/placement_cuda2.{pdf,png}` (`model/placement_figure.py`).

TITAN V (cuda3), usable numbers: 1 GPU prefix 59.2, loop eager 280.4, loop graph 66.5, step
eager 340.6, best 126.2 ms; handoff GPU->GPU 1.36, GPU->CPU 3.55, CPU->GPU 4.29 ms; CPU
prefix about 2.0 s, CPU loop 928 to 944 ms.

## Predicted from single-device parts (cuda2)

`T(A, B) = T_prefix(A alone) + n_tensors * t0 + KV_bytes / BW(A -> B) + T_loop(B alone)`,
with t0 and BW from the copy calibration and each GPU's own baseline
(`model/placement_report.py RUN --device-baseline cuda:1=<GPU 1 run>`).

| Split | Handoff pred / meas | Eager step pred / meas | Best step pred / meas |
|---|---|---|---|
| GPU 0 -> GPU 1 | 1.98 / 1.54 ms (+29%) | 483.7 / 496.0 (-2.5%) | 120.8 / 120.6 (**+0.2%**) |
| GPU -> CPU | 3.00 / 2.74 ms (+10%) | 745.1 / 737.5 (+1.0%) | same as eager |
| CPU -> GPU | 1.64 / 3.20 ms (-49%) | 2,088 / 2,040 (+2.3%) | 1,741 / 1,751 (-0.5%) |

## Findings

1. **The KV handoff is cheap.** 4.9 MB in 33 tensors: 1.4 to 1.5 ms GPU to GPU, 2.7 to 3.6 ms
   GPU to CPU, 3.2 to 4.3 ms CPU to GPU (FP32 KV, cast on the GPU). That is about 1% of the
   best single-GPU step. Splitting across two GPUs therefore costs only the handoff; it does
   not make one inference faster, because the halves run one after the other. Its value is
   capacity (two policies, or the next prefix overlapping the current loop), which is
   direction 6 (multi-policy hosting).
2. **The step is additive and predictable.** Parts timed alone sum to the end-to-end step
   within 3% in every placement, and split placements are predicted within 0.2 to 2.5% from
   single-device runs. The handoff term itself is off by 10 to 49%, but it is only 1.5 to 3 ms.
   It overestimates GPU-GPU (per-tensor setup overlaps with the transfer) and underestimates
   CPU-GPU (the model has no term for the FP32 to FP16 cast).
3. **A CPU is not a useful host for either half.** Prefix on the CPU: 21x (2080 Ti host) to
   34x (TITAN V host) slower than the GPU. Action loop on the CPU (FP32, 20 to 24 threads):
   1.6x (cuda2) to 3.3x (cuda3) slower than the dispatch-bound eager GPU loop, and 14 to 16x
   slower than the graph loop. The eager GPU loop is so host bound that a CPU gets within 1.6x
   of it on cuda2; CUDA graphs remove that.
4. **Two GPUs of the same model differ by 37% in graph mode.** On cuda2, GPU 0 runs the graph
   loop in 54.1 ms and GPU 1 in 39.6 ms (the project's earlier 2080 Ti traces: 40.1 ms). Eager
   timings are identical (482 ms) because eager is host bound, so only graph mode exposes it.
   At idle, GPU 0 sits in P0 at 1,350 MHz while GPU 1 drops to P8; the clock under load was not
   sampled. Earlier in this project the TITAN V gap was traced to its SM clock (1,335 vs
   1,950 MHz); 1,950 / 1,350 = 1.44 is close to the 1.37 seen here, so a lower sustained clock
   on GPU 0 is the likely cause (to confirm with `scripts/clock_probe.sh` on GPU 0). Any
   comparison of graph-mode latency across cards needs a per-card baseline.
5. **No P2P on cuda2** (`nvidia-smi topo -p2p r`: chipset not supported), so GPU 0 to GPU 1
   copies go through host memory; cuda3 reports P2P OK although its GPUs sit on different
   sockets. Both handoffs are about 1.5 ms, so it does not matter at this KV size.
6. **Numerics (to check).** Moving the loop to the CPU changes the actions by at most 6e-4.
   Moving the prefix to the CPU (FP32 instead of FP16) changes them by up to 0.39, the same on
   both machines. So the FP16 prefix, not the loop, is where FP16 and FP32 SmolVLA diverge.
   This may relate to the FP16 vs stock LIBERO gap (50 vs 58% success), untested.

## Open items

- cuda3 rerun with the fixed script (needs both TITAN Vs idle, about 20 min) for a second
  clean machine.
- Sample GPU 0's SM clock under load on cuda2 (`scripts/clock_probe.sh`).
- Optional: CPU thread sweep (`scripts/run_placement.sh threads`).
