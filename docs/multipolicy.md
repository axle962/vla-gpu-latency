# Multi-policy hosting on one GPU (pending direction 6)

Question: when several robot policies share one GPU, what does each policy's step cost, and how
much total inference does the GPU deliver, in eager mode and with CUDA graphs? The project's
main finding says an eager VLA step is host bound: the GPU is busy about 15% of the step
(SmolVLA on the A100: 58 of 377 ms). So an eager policy should leave room for others, while a
graphed policy is a GPU chain (35 ms of back-to-back kernels) that competes for the GPU.
Hosting several policies is where the eager/graph trade-off turns into a capacity question:
graphs cut one policy's latency 4x, but do N graphed policies still fit?

## What is measured

`workloads/multipolicy.py` hosts N policy copies on one GPU in four ways and times every
copy closed loop (a new step as soon as the previous one returned) for a fixed window, all
copies at once:

| hosting | how | what it isolates |
|---|---|---|
| `procs` | one process per policy, one CUDA context each | driver time-slicing between contexts |
| `procs-mps` | the same under an MPS control daemon | kernels of different processes running at the same time |
| `threads` | one process, one Python thread and one CUDA stream per policy | one context, shared GIL, streams |
| `batch` | one process, one policy at batch N | the batched reference (identical policies only) |

Policies: SmolVLA (FP16, 3 cameras, 10 flow steps, chunk 50: dispatch bound), ACT (FP32,
480x640: mixed) and Diffusion Policy (FP32, 10 DDPM steps: GPU bound), the same code paths as
`trace_smolvla.py`, `trace_act.py` and `trace_diffusion.py`. Every copy checks that its split
loop and its CUDA graph reproduce the stock step. Execution modes per policy: `eager` (stock
step), `graph` (eager prefix + graph loop, the deployment the traces call hybrid; the whole step
as one graph for ACT) and `loop` (the graph loop alone, no host work).

Per step: start time and wall time (`latency.csv`). Per config: median and p95 per copy over the
middle 80% of the window, slowdown against the policy alone, and total steps per second
(`summary.csv`, `report.md` from `model/multipolicy_report.py`). Solo runs are also profiled
for a few steps to get the GPU busy time per step (`torch.profiler`, parsed by
`workloads/trace_parse.py`, the profiler's 0.95 us per kernel removed).

## Prediction from the policies alone

Each copy is a client of a closed queueing network with two stations, host and GPU: a step
spends Z = T_solo - D on the host and D (GPU busy per step) on the GPU. For `procs` the host is
a delay station (each process has its own cores) and the GPU a single FIFO server (contexts
time-slice; kernels of different contexts do not overlap). For `threads` the host is a single
server too (the GIL). Solved with the Bard-Schweitzer approximate MVA, one client per class,
which gives each policy's hosted latency and throughput with nothing measured under contention.
The report prints the model's error per config next to the no-interference bound (T_solo).

Expected: eager `procs` copies keep their solo latency until N x D approaches T_solo (SmolVLA
on the A100: about 6 copies), then the GPU saturates and latency grows linearly; `graph`
copies serialize from N = 2 (each is a 35 ms GPU chain), so the graph speedup per policy
shrinks as N grows, the same way the batch sweep showed it shrinking with batch size. Under
MPS the small loop kernels (achieved occupancy ~13%) can co-run, so graphed copies should beat
the serialized model; the gap is the measured overlap. `threads` eager should collapse onto the
GIL (every copy pays every other copy's Python time), `threads` graph should not (a replay is
one C++ call).

## How to run (CIMS cudaN, after `scripts/run_cuda2.sh setup` and `more`)

```bash
module load cuda-12.4
export WORK=/scratch/$USER
scripts/run_multipolicy.sh smoke     # toy policy on the GPU, ~1 min, checks MPS and streams
scripts/run_multipolicy.sh quick     # SmolVLA + ACT, N = 1 2 4, every hosting, ~30 min
scripts/run_multipolicy.sh full      # + Diffusion, N = 8, loop mode, mixed sets, ~80 min
git add results/multipolicy && git commit -m "multipolicy runs on <host>" && git push
```

The script refuses to start while another process is on the GPU (CIMS GPUs are shared) and
pins everything to the GPU's NUMA node. MPS uses a private pipe directory under `$WORK/mps`
and is stopped at exit; if the daemon cannot start the `procs-mps` configs are skipped. On a
Slurm cluster: `sbatch -p <partition> --export=ALL,WORK=<dir> scripts/cluster/multipolicy.sbatch quick`.

Memory: N SmolVLA copies need about N x 1.5 GB (FP16 weights plus a context), so the 11 GB
2080 Ti holds the quick matrix (N <= 4) and the 12 GB TITAN V and RTX 4070 too; N = 8 needs a
larger card (A100, H100).

## Results: RTX PRO 6000 Blackwell (Empire AI, 2026-11-18, quick matrix)

`results/multipolicy/NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition_alphagpu54_20261118-000622/`
(report.md, summary.csv, predictions.csv; Slurm job 203123, 1 GPU of a shared node, 16 cores
of a Xeon 6960P, torch 2.10.0+cu128, 15 s per mode, all 43 policy copies bit-exact).

Alone: SmolVLA eager 181 ms, with graphs 61 ms (GPU busy 28 ms per step, 16% of the eager
step); ACT eager 5.0 ms, graph 4.2 ms (GPU busy 4.9 ms, 99%). Median step per copy (ms) and
total steps per second over all copies:

| hosting | SmolVLA eager | SmolVLA graph | ACT eager | ACT graph |
|---|---|---|---|---|
| alone | 181 (5.6/s) | 61 (16.3/s) | 5.0 (200/s) | 4.2 (238/s) |
| procs x2 | 181 (11.1/s) | 100 (20.0/s) | 10.1 (198/s) | 8.6 (233/s) |
| procs x4 | 268 (15.0/s) | 191 (20.7/s) | 20.3 (197/s) | 17.1 (233/s) |
| procs-mps x2 | 177 (11.4/s) | 71 (28.1/s) | 5.8 (345/s) | 5.4 (369/s) |
| procs-mps x4 | 169 (23.8/s) | 86 (46.5/s) | 8.3 (482/s) | 8.0 (502/s) |
| threads x2 | 385 (5.0/s) | 87 (23.3/s) | 10.7 (176/s) | 5.2 (385/s) |
| threads x4 | 1493 (2.7/s) | 166 (24.8/s) | 39.9 (94/s) | 7.8 (516/s) |
| batch 4 | 192 (21.0/s) | 80 (50.0/s) | | |

Mixed SmolVLA + ACT, one copy each (SmolVLA / ACT ms): procs eager 207 / 7.8, graph 131 / 6.7;
procs-mps eager 161 / 5.3, graph 79 / 4.6.

What it says:

1. Eager, host-bound policies co-host almost for free. Two eager SmolVLA processes keep the
   solo latency (181 ms) at twice the throughput; four reach 268 ms without MPS and 169 ms
   (4.3x the throughput) with it. The GPU was idle 84% of the time, and other processes can
   use that time.
2. Graphed policies compete for the GPU, and plain context time-slicing is worse than
   serialization: four graphed SmolVLA copies take 191 ms each (3.1x) and deliver only 1.3x
   the solo throughput, while the serialized model (4 x 28 ms of GPU time plus the host part)
   predicts 120 ms. Under MPS the same four run at 86 ms (1.4x) and 46.5 steps/s, 2.8x the
   solo throughput and close to batch 4 (50 steps/s): the loop's tiny kernels (median 1.4 us)
   leave most SMs free, so kernels of different processes overlap once MPS allows it.
3. ACT is GPU bound by time (99% busy) but not by SMs: without MPS N copies are exactly
   serial (2.03x, 4.08x; model error -2 to -3%), with MPS four copies run at 8.3 ms for 2.4x
   the throughput. "GPU bound" at batch 1 still means a mostly empty GPU.
4. Threads in one process are the wrong way to host eager policies: the GIL serializes the
   dispatch and the collapse is superlinear (four eager SmolVLA threads: 8.3x, four ACT: 8x).
   A fully graphed policy (ACT) in threads matches MPS, because a replay is one C++ call; the
   SmolVLA hybrid still pays its eager prefix under the GIL (166 ms vs 86 ms with MPS).
5. Mixed hosting without MPS hurts the graphed policy most: ACT's steady GPU demand doubles
   SmolVLA's graphed step (61 to 131 ms) and SmolVLA's eager dispatch slows ACT 1.6x. MPS
   removes most of it (79 ms and 4.6 ms).
6. The hosting model predicts the non-MPS process case for eager policies within 9% MAPE and
   underpredicts graphed copies (24%), because time-slicing between contexts costs more than
   serializing their kernels. Under MPS the measurements sit well below the serialized model,
   as the model is an upper bound there; a prediction for MPS needs an SM-occupancy term.

For a deployment: with CUDA graphs, hosting several policies on one GPU needs MPS (or batching
when the policies are identical); without it the graph speedup is lost from the second policy
on. Eager hosting keeps per-policy latency but tops out at a quarter of the throughput of
graphs plus MPS.

Known limitation: SmolVLA and ACT as two threads of one process (`threads smolvla,act`) hit a
CUDA illegal memory access in the graph phase on this run (the same-policy threads configs
ran); not debugged, the process and MPS hostings cover the mixed case.

## Results: full matrix on the same GPU (2026-11-18, job 204101, 85 min)

`results/multipolicy/NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition_alphagpu54_20261118-111750/`:
47 configs (SmolVLA, ACT, Diffusion Policy at 10 DDPM steps; N = 1, 2, 4, 8; every hosting;
modes eager, graph, loop; 10 s per mode), 43 ran, 4 failed (the in-process threads mixes that
include SmolVLA, the limitation above). Alone: Diffusion eager 137 ms, graph 132 ms, GPU busy
129 ms (95%); SmolVLA loop 32 ms. Median step per copy (ms) and total steps/s:

| hosting | SmolVLA eager | SmolVLA graph | SmolVLA loop | ACT graph | Diffusion graph |
|---|---|---|---|---|---|
| alone | 174 (5.9/s) | 60 (16.8/s) | 32 (30.9/s) | 4.1 (242/s) | 132 (7.5/s) |
| procs x8 | 512 (15.9/s) | 375 (21.3/s) | starved | 33.7 (239/s) | 1133 (7.0/s) |
| procs-mps x8 | 155 (51.6/s) | 111 (72.1/s) | 57 (141/s) | 14.1 (567/s) | 218 (37.0/s) |
| threads x8 | starved | starved | 54 (147/s) | 13.8 (578/s) | 218 (36.6/s) |
| batch 8 | 202 (39.0/s) | 99 (81.0/s) | 39 (205/s) | | |

Three policies at once, one copy each (SmolVLA / ACT / Diffusion, graph mode): procs 198 / 11.1
/ 374 ms (3.3x / 2.7x / 2.8x alone); procs-mps 79 / 4.7 / 184 ms (1.3x / 1.1x / 1.4x).

What the full matrix adds:

1. "GPU bound" at batch 1 is bound by time, not by SMs. Diffusion Policy keeps the GPU busy
   95% of its step, and without MPS N copies are exactly serial (8 copies: 8.5x, model error
   -8 to -11%). Under MPS 8 copies run at 218 ms each, 4.9x the throughput of one: the 10-step
   UNet at batch 1 launches small grids, so most of the 188 SMs are idle inside a "busy" step.
   The same holds for ACT (8 copies with MPS: 2.3x throughput) and for SmolVLA's graph loop
   (8 copies: 4.6x).
2. Eight eager SmolVLA processes under MPS still run at the solo latency (155 ms, 8.8x the
   throughput); without MPS they reach 512 ms. Eight graphed copies under MPS run at 111 ms
   and 72 steps/s, within 11% of batch 8 (81 steps/s) with no change to the policy code.
3. Context time-slicing without MPS costs more than serialization, and the cost grows with the
   number of kernels: Diffusion (2.8k kernels per step) runs 8 to 11% above the serialized
   model, SmolVLA's graph loop (12k kernels) 25 to 43% above it. Eight processes replaying the
   loop back to back starved each other outright (one step per process in a 10 s window), and
   eight eager threads of one process starved on the GIL (a step took 40 s): both are marked
   `steps/copy` < 5 in the report and left out of the prediction tables.
4. Streams in one process overlap as well as MPS for fully graphed policies (ACT and Diffusion
   in threads match procs-mps at every N), so for policies that are one graph replay, threads
   are a cheap way to host; for anything with eager Python per step, processes plus MPS.
5. The serialized model: procs eager 16% MAPE, graph 21%, loop 17% over 20 cases each (the
   no-interference bound is at 51 to 64%). Under MPS and for graphed threads the measurements
   sit far below it (an SM-sharing term is the natural next step for the model).

## Results: CIMS RTX 2080 Ti (cuda2), TITAN V (cuda3), RTX 4070 (cuda5), 2026-11-23, quick matrix

`results/multipolicy/NVIDIA_GeForce_RTX_2080_Ti_cuda2_20261123-133707/` (Xeon E5-2660 v3
2.6 GHz, 20 cores pinned, driver 595), `NVIDIA_TITAN_V_cuda3_20261123-130336/` (Xeon Gold 5118
2.3 GHz, 24 cores pinned, driver 580) and `NVIDIA_GeForce_RTX_4070_cuda5_20261123-130736/`
(Xeon E5-2650 v2 2.6 GHz, 8 cores pinned, driver 595); all torch 2.10.0+cu128, 15 s per mode,
43 policy copies bit-exact on each. Run directly on the hosts (no Slurm) in the gaps between
other users' jobs; cuda2 and cuda3 had their card to themselves, the cuda5 run shared the card
with an idle 154 MB process of another user (0% utilization). The threads SmolVLA+ACT mix
failed on all three as before.

Alone (eager / graph ms, GPU share of the graph step): 2080 Ti SmolVLA 485 / 135 (62%), ACT
12.3 / 6.8 (95%); TITAN V SmolVLA 337 / 126 (75%), ACT 8.8 / 6.0 (100%); RTX 4070 SmolVLA
607 / 118 (34%), ACT 15.7 / 3.8 (87%). The hosts set the eager step (the 4070's 8-core
E5-2650 v2 is the slowest, the TITAN V's Gold 5118 the fastest) while the graph step is within
15% across the three cards; the TITAN V is GPU bound once graphed (95 ms of GPU work in a
126 ms step).

Median step per copy (ms) and total steps/s, SmolVLA graph mode:

| hosting | 2080 Ti | TITAN V | RTX 4070 |
|---|---|---|---|
| alone | 135 (7.4/s) | 126 (8.0/s) | 118 (8.5/s) |
| procs x2 | 241 (8.3/s) | 238 (8.3/s) | 154 (12.8/s) |
| procs x4 | 440 (9.3/s) | 461 (8.7/s) | 249 (16.0/s) |
| procs-mps x2 | 162 (12.4/s) | 159 (12.6/s) | 129 (15.5/s) |
| procs-mps x4 | 231 (17.2/s) | 231 (17.3/s) | 163 (24.7/s) |
| threads x4 | 246 (16.1/s) | 227 (17.4/s) | 660 (6.1/s) |
| batch 4 | 260 (15.7/s) | 214 (18.7/s) | 183 (21.7/s) |

ACT graph mode:

| hosting | 2080 Ti | TITAN V | RTX 4070 |
|---|---|---|---|
| alone | 6.8 (147/s) | 6.0 (167/s) | 3.8 (264/s) |
| procs x4 | 30.2 (132/s) | 26.7 (150/s) | 15.6 (254/s) |
| procs-mps x4 | 19.7 (203/s) | 17.1 (234/s) | 13.5 (294/s) |
| threads x4 | 19.5 (205/s) | 16.8 (239/s) | 13.2 (302/s) |

Eager SmolVLA, four copies (procs / MPS ms): 2080 Ti 562 / 495, TITAN V 476 / 366, RTX 4070
674 / 667 (1.02x to 1.41x alone). Mixed SmolVLA + ACT (graph, ms): 2080 Ti procs 251 / 13.3,
MPS 211 / 8.5; TITAN V procs 269 / 12.0, MPS 203 / 7.3; RTX 4070 procs 173 / 4.5, MPS
167 / 4.6.

What the CIMS GPUs add:

1. The hosting picture holds on four generations (Turing, Volta, Ada, Blackwell): eager copies
   share a GPU almost for free, graphed copies serialize without MPS (4 copies: 3.3x on the
   2080 Ti, 3.7x on the TITAN V, 2.1x on the 4070, 3.1x on the PRO 6000) and recover most of
   it with MPS (1.7x, 1.8x, 1.4x, 1.4x). Batching identical copies is the ceiling MPS
   approaches, and on the 2080 Ti and the 4070 MPS even beats batch 4 (17.2 vs 15.7 and 24.7
   vs 21.7 steps/s), because the batched step still pays the host-bound eager prefix.
2. How much MPS helps depends on how full the GPU already is. Four MPS copies of graphed
   SmolVLA give 2.2x the solo throughput on the TITAN V (solo GPU share 75%), 2.3x on the
   2080 Ti (62%), 2.8x on the PRO 6000 (44%) and 2.9x on the 4070 (34%). ACT is 87 to 100%
   busy on every card and MPS gives it 1.1 to 2.1x only.
3. The host CPU sets the eager and threads results. On the 4070's 8-core Xeon E5-2650 v2
   eager SmolVLA is 607 ms and four threads take 4.4 s per step (7.3x); four ACT threads
   8.5x. The 2080 Ti host (20 cores) collapses to 3.1x and 3.8x, the TITAN V host (24 cores)
   to 4.7x and 6.3x.
4. The serialized model is at its best on these cards: procs eager 5.5% (2080 Ti), 6.7%
   (TITAN V) and 3.2% (4070) MAPE, procs graph 15.4%, 12.5% and 16.0%, against 57%, 59% and
   42% for the no-interference bound in graph mode. Under MPS it stays within 8% on the 4070
   (eager 4.8%, graph 8.1%), where the overlap is small, and overpredicts by 14 to 37% on the
   2080 Ti and the TITAN V, where kernels of different processes overlap more.

Not run on CIMS: N = 8 SmolVLA needs ~12 GB and does not fit the 11 to 12 GB cards; Diffusion
Policy and loop mode (full matrix) are on the PRO 6000 only.
