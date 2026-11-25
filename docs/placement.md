# Hosting placement profile (pending direction 4)

Question: what does a VLA step cost when its two halves are hosted on different devices?
"GPU-GPU hosting" is read here as the prefix (vision tower + VLM, fills the KV cache) on one
GPU and the action expert's denoising loop on a second GPU; "GPU-CPU hosting" as one half on
a GPU and the other on the host CPU. If the team meant something else (for example a policy
server on one machine and the robot client on another), the timing harness still applies, only
the handoff changes.

## What is measured

`workloads/placement.py` builds one copy of SmolVLA per device and times, for each placement:

| placement | prefix on | denoise on | what it shows |
|---|---|---|---|
| `gpu` | cuda:0 | cuda:0 | colocated baseline, same numbers as `trace_smolvla.py` |
| `gpu-gpu` | cuda:0 | cuda:1 | cost of the KV handoff between two GPUs |
| `gpu-cpu` | cuda:0 | cpu | action expert without any kernel launches |
| `cpu-gpu` | cpu | cuda:0 | vision + VLM on the host, expert on the GPU |
| `cpu` | cpu | cpu | CPU-only baseline |

Modes: `prefix`, `handoff`, `denoise_eager`, `denoise_graph` (GPU denoise only), `e2e_eager`,
`e2e_hybrid`. It also runs a copy calibration (4 KB to 256 MB per device pair) so the handoff
can be predicted from the KV cache size. `--profile` saves a torch.profiler trace of two
end-to-end steps per placement (git-ignored, open in Perfetto).

`model/placement_report.py <run dir>` writes `report.md`: measured table, copy calibration
fit, and each split placement predicted from parts that never ran split,
`T(A, B) = T_prefix(A) + n_tensors * t0 + KV_bytes / BW(A -> B) + T_denoise(B)`, with
percentage error. This is the same predicted-vs-measured format as the rest of the project.

## How to run (CIMS, after `scripts/run_cuda2.sh setup`)

```bash
module load cuda-12.4
export WORK=/scratch/$USER
scripts/run_placement.sh smoke     # toy model, checks the env and both GPUs, ~1 min
scripts/run_placement.sh           # all placements, ~10 min
scripts/run_placement.sh threads   # plus CPU thread sweep 1 / 4 / 8 / all, ~30 min
git add results/placement && git commit -m "placement runs on <host>" && git push
```

Run it on cuda2 (2x RTX 2080 Ti) and cuda3 (2x TITAN V) for GPU-GPU; cuda5 has one GPU, so
only the GPU-CPU and CPU placements run there (its 2013 host CPU is the slow extreme). Both
GPUs on cuda2 share NUMA node 1 and its cores, so run nothing else at the same time.

## What to expect (estimates from existing traces, to be replaced by measurements)

- The SmolVLA KV cache is small: 4.9 MB in 32 tensors (`kv_cache_bytes` in the trace
  `meta.json`, 241 prefix tokens). At a few GB/s the copy itself is ~1 to 2 ms, so the handoff
  should be dominated by per-tensor fixed costs (33 copies and syncs), not bandwidth. Expected:
  GPU-GPU end to end within a few ms of colocated (2080 Ti hybrid 116 ms).
- Splitting does not shorten a single step, because the halves run one after the other. Its
  value is capacity (two policies, or a pipelined prefix for the next observation), which
  connects to direction 6 (multi-policy hosting).
- GPU-CPU is the interesting case for the paper's thesis. The eager GPU loop is dispatch bound
  (2080 Ti: 407 ms, of which ~40 ms is GPU work). A CPU-hosted expert pays no launches at
  all; whether it lands near the eager GPU loop or far above it depends on the host's FP32
  GEMM throughput and thread count. The graph loop (40 ms) should beat it clearly.
- Precision differs by device (FP16 on GPU, FP32 on CPU), so compare outputs with the
  `max diff vs first` column and treat CPU timings as FP32 timings.

## Not covered yet

- Other models: pi0 / pi0.5 have the same two-half structure (`workloads/trace_pi.py`) and a
  much larger KV cache; adding a `build_pi` next to `build_smolvla` is the natural extension.
- GPU pairs of different models (for example prefix on a fast GPU, expert on a slow one):
  no CIMS host has two different GPUs.
- Overlapped execution (prefix for observation t+1 while the expert finishes chunk t).
