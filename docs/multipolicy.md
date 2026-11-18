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

## Results

To be filled from `results/multipolicy/<gpu>_<host>_<stamp>/report.md` after the runs.
