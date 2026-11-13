# Launch-Bound and Cache-Aware: Predicting VLA Inference Latency and CUDA-Graph Speedup Across GPU Generations

Course project, NYU GPU Architecture and Programming (Fall 2026, Prof. Mohamed Zahran).
Author: Aditya Maddur Guruprakash (am16489).
Track: **Project 2, Predicting the Performance of a Kernel, flavor B**: "Knowing the performance of a kernel on a GPU, can we predict its performance on another GPU with different specifications?"

This file is the project brief for Claude Code. Read it fully before writing code.

---

## 0. Where things are (updated 2026-11-10)

- Repo: `/scratch/am16489/vla-gpu-latency`, GitHub `axle962/vla-gpu-latency` (push via the
  `github-me` SSH alias, which authenticates as axle962).
- Slurm account: **`csci_ga_3033_025-2026fa`**, cap **300 GPU-hours** (`gres/gpu=18000`
  GPU-minutes). Never use the wildfire account `csci_ga_2271-2026fa` here. Estimate GPU-hours
  before submitting, tally with the `sacct` line in `scripts/cluster/site.example.sh`.
- `source scripts/cluster/site.sh` in each new cluster shell (copy from `site.example.sh`).
- CUDA 12.4 toolchain on the cluster login node: conda env at `/scratch/am16489/.tools/cuda124`
  (nvcc 12.4.131, same version as cuda2). The system only has CUDA 11.1; there is no
  `cuda-12.4` module. Containers with newer CUDA are in `/share/apps/images/*.sif`.
- Build: `make -C bench` (sm_75, for cuda2) or
  `make -C bench ARCHS="75 70 80 89" BUILD=build-cluster` (fat binary; PTX for sm_89 lets
  Blackwell JIT it, since CUDA 12.4 cannot emit sm_120).
- Calibration run: `scripts/run_calib.sh [build_dir]`; on the cluster
  `sbatch -p <partition> scripts/cluster/calib.sbatch`. CSVs land in `results/calib/`.
- Profiler check on cuda2: `scripts/check_profilers.sh` (with `TMPDIR` private, see 0.1).

---

## 0.0 If this session runs on CIMS cuda2 (or another CIMS cudaN machine)

Check `hostname` first. On a CIMS `cudaN` host the cluster rules below (Slurm, `sbatch`, the
`csci_ga_3033_025-2026fa` account, GPU-hour budget, "never run GPU work on the login node",
`/scratch/am16489/...` paths, conda nvcc) **do not apply**: there is no Slurm, GPU work runs
directly on the machine (after checking with `nvidia-smi` that the GPU is idle), the toolchain
comes from `module load cuda-12.4`, and home has a 4 GB quota. The task there is
**`docs/cuda2_runbook.md`**: follow it step by step via `scripts/run_cuda2.sh <stage>`, then
commit and push (or tar) the results. Never run GPU work on the `access` gateway.

---

## 0.1 Resume here (updated 2026-11-12 ~22:30)

State:
- **Calibration done on A100** (`results/calib/NVIDIA_A100*`, two nodes; launch numbers agree
  within 2%). The first run's bandwidth data was deleted (old sweep starved small working sets;
  see its NOTE.txt). Findings are in section 4.2 (eager per-kernel penalty ~2 us persists for long
  kernels) and below: A100 BW_L2 ~2.65 TB/s, BW_DRAM 1.48 TB/s (95% of spec), L2 knee ~20 to 27 MB
  (half of the nominal 40 MB, consistent with A100's two L2 partitions).
- **L4 (`g2-standard-12`) is unusable right now.** Jobs 28144, 28473 and 28488 never started
  the script: each node boot failed, Slurm requeued automatically (about 8 billed minutes per
  attempt, 28488 alone made 9 attempts), and the job ended `CANCELLED by 0`. About 2.1 of the
  2.46 GPU-h used so far went to these failed boots. 20 L4 nodes are drained. The sbatch files
  set `--no-requeue`, but that does **not** stop requeues after a failed node boot (sweep job
  28998 was still requeued twice on A100, ~6 billed minutes each, before running on b-19-49).
  A100 boots also fail sometimes; L4 boots have never succeeded. Do not resubmit to L4; ask NYU
  HPC about g2-standard-12 boot failures first.
- **Cluster quirks found 2026-11-11.** The login node has ~1.8 GB of RAM and no swap: importing
  torch or parsing a 100 MB trace JSON there stalls for minutes. Parse traces on the CPU
  partition `n2c48m24` (allowed for our account, no GPU, costs no GPU-hours):
  `sbatch -p n2c48m24 scripts/cluster/reparse.sbatch traces/<model>/<run>/`. One CPU node
  (b-10-119) hung without writing a log; `--exclude` it if it happens again. Slurm job IDs are
  reused from 2024, so `sacct -j <id>` can show an unrelated old job; filter with `-A` and `-S`.
- **SmolVLA traced on A100** (`traces/smolvla_cam3_steps10_chunk50/NVIDIA_A100*_b-19-37*`, job 28487, FP16, 3
  cameras 256x256 resized to 512, 10 flow steps, chunk 50, batch 1):
  eager full step 377 ms; eager prefix 58 ms; eager denoise loop 316 ms; **CUDA-graph denoise
  loop 35.8 ms (8.8x)**; hybrid (eager prefix + graph loop) 94 ms (4.0x end to end).
  13,190 kernels per step, median kernel 3.3 us, 78% under 5 us, GPU busy 57.7 ms (15% of the
  eager step). Per step also 529 memcpys and 36 stream syncs. The host is a 2.2 GHz cloud Xeon,
  so eager cost is ~27 us per kernel of CPU (Python + aten dispatch), far above the 2.5 us
  `cudaLaunchKernel` cost from calibration. Model implication: `L_launch` for real PyTorch
  workloads is framework dispatch cost per op, not the bare driver launch cost; calibrate it from
  the eager trace (or a PyTorch microbenchmark), and record host CPU. Graph replay and the split
  loop both match the stock output bit for bit.
  - Why only the denoising loop is graphed: the SigLIP vision embedding calls
    `torch.arange(mask.sum())` (data-dependent host sync). The loop's constant uploads
    (`torch.tensor(list, device=cuda)`) are memoized during capture (`CudaConstMemo`).
- **OpenVLA-7B traced on A100** (`traces/openvla7b_fp16/NVIDIA_A100*`, job 28964, FP16, SDPA,
  one 224x224 image, 7 action tokens, `predict_action`): eager 312 ms per action; prefill
  (vision + 283-token Llama prefill + first token) 69 ms; **40 ms per decode token**. 11,519
  kernels per step, GPU busy 127 ms (41%). Decode GEMVs take ~10.5 ms of GPU per token, close
  to streaming 13 GB of weights at 1.48 TB/s (8.8 ms), so decode is ~4x launch/dispatch bound.
  **Graph mode (job 29163):** static KV cache + the 6 greedy decode steps captured as one CUDA
  graph (`StaticPath` in `trace_openvla.py`); tokens match `generate` exactly. Decode 238.6 ms
  eager (static cache) vs 85.2 ms graph (**2.8x**, 14.2 ms per token, 62% of the 8.8 ms
  weight-streaming bound); end to end 311 ms stock vs 155 ms (eager prefill + graph decode,
  **2.0x**). Smaller than SmolVLA's 4x because OpenVLA kernels are larger (decode GEMVs ~70 us).
  Needed two transformers 4.40 workarounds (documented in the module docstring): patch
  `_ignore_causal_mask_sdpa` (static cache + 1-token query loses the causal mask; 2D mask check
  is a host sync), and delete, not None, the per-layer `past_key_value` after use.
- **Parser fix (2026-11-11):** kernels are now linked to ops by launch timestamps
  (innermost enclosing cpu_op on the launching thread), because torch 2.2 fills
  "External id" differently and mislabeled OpenVLA ops. Driver-API launches are counted too
  (SmolVLA: 13,526 kernels per step instead of 13,190).
- **SmolVLA sweep on A100 done** (job 28998, 7 configs, all bit-exact;
  `results/traces_summary.csv` from `python3 model/summarize_traces.py`):
  - Reproducible: the baseline on two nodes differs by 1.4% (eager 377 vs 383 ms), graph
    denoise 35.8 vs 35.6 ms.
  - Each flow step is ~1,166 kernels: eager ~32 ms per step (~27.5 us per kernel of CPU
    dispatch), graph ~3.5 ms per step (~3.0 us per kernel = kernel time + node cost).
    Graph speedup of the loop is a constant ~9x for 5, 10 and 20 steps; end-to-end hybrid
    speedup grows with steps (2.9x at 5, 4.0x at 10, 5.4x at 20) because the eager prefix is
    fixed (~59 ms, ~1,870 kernels).
  - Chunk 25 to 100 (4x action tokens) moves graph denoise only 35.0 to 37.2 ms: even under
    graphs the kernels are latency bound, not work bound. Eager denoise is unchanged.
  - Cameras 1 to 3: prefix 42 to 59 ms (~8.4 ms per camera); denoise unchanged.
  - Model implication: T_eager ~ T_prefix + N_steps * K_step * c_dispatch, and
    T_graph_loop ~ N_steps * sum_k (t_k + delta_node); both linear in N_steps, so the loop's
    graph speedup is set by c_dispatch / (mean t_k + delta_node), independent of N_steps.
- **PyTorch dispatch calibration on A100 host** (`results/calib/NVIDIA_A100*_b-19-28*/
  torch_dispatch_*.csv`, `workloads/torch_dispatch.py`; chains of 400 ops, eager vs CPU issue
  vs graph, both torch builds):
  - CPU cost per op is op-specific: elementwise add 6.8 us (torch 2.2) / 7.3 us (2.10);
    with output allocation 9.4 us; `F.linear` 17 to 22 us (cuBLAS dispatch); SDPA 31 us; the
    6-kernel RMSNorm 95 (2.2) / 105 (2.10) us. torch 2.10 is ~5 to 10% slower to dispatch.
  - Graph cost per tiny kernel ~1.6 us (kernel + ~0.8 us node, matches launch_latency).
  - `eager ~ max(issue, graph)` per op holds: tiny ops are CPU bound (eager/graph 3 to 7x);
    the 1x4096x11008 decode GEMV (70 us GPU) and the 283x4096x4096 prefill GEMM are GPU bound
    (1.0x). Crossover for adds is ~4M elements (~7 us of GPU work).
  - The VLA traces average ~27 us per kernel eager, above these per-op costs: the gap is
    Python module/HF-code overhead between aten ops. Model plan: c_dispatch per aten op type
    from this table, times op counts from the trace, plus a per-step Python residual fitted on
    one config and validated on the others.
- **Launch-aware model v1, same GPU (A100), `model/launch_model.py`** ->
  `results/predictions_same_gpu.csv`. Two-timeline replay of the traced step (per-op PyTorch
  dispatch from the calibration table, GPU-side g_eager = 2.14 us, host copies synchronous) and
  graph = L_graph + sum(t_k + delta_node), delta_node = 0.28 us (C++ spin chains). Two fitted
  parameters, both on one baseline run and held fixed: p (Python time per kernel beyond
  single-op dispatch: 17.3 us SmolVLA, 16.7 us OpenVLA, nearly equal across models and torch
  versions) and b (profiler bias per traced kernel: 0.98 us torch 2.10/cu128, 0.66 us torch
  2.2/cu121). Held-out MAPE: SmolVLA (8 runs) eager 1.8%, eager denoise 3.4%, prefix 4.1%,
  graph loop 0.7%, hybrid 2.8%; OpenVLA (repeat runs only) eager 0.7%, graph decode 0.3%,
  prefill -14% (more Python per kernel in HF prefill/generate setup), hybrid -8%. Baselines:
  GPU-time only 24 to 86%, bare driver launch 52 to 78%, dispatch table without p 51 to 62%.
  - **Profiler bias finding (job 30203, graph replays profiled with `profile_graph`):** with
    CUPTI tracing on, the SmolVLA graph replay takes 48.3 ms instead of 35.7 ms, and short
    kernels measure the same eager (idle gaps) and in-graph (back to back), 2.99 vs 3.01 us.
    So idle gaps do not slow kernels; the profiler adds ~1 us per kernel. Under torch 2.2/cu121
    the overhead shows up as in-graph gaps (17 ms per OpenVLA decode) instead of longer
    durations. Traced durations must be corrected by b before summing; b is a property of the
    source machine's profiler and is fitted there, so this stays valid for cross-GPU use.
  - Cluster note: two trace jobs (29191, 29303) failed their node boot and were then cancelled
    by a system sweep at the top of the hour (`CANCELLED by 0` at hh:00). Failed boots cost 0
    s. Watch for `REQUEUED` and resubmit instead of waiting.
- **cuda2 (RTX 2080 Ti, source GPU) done 2026-11-12** (`docs/cuda2_runbook.md`, run by Claude
  Code on cuda2; host Xeon E5-2660 v3 2.6 GHz, driver 595.104.02, torch 2.10.0+cu128):
  - Calibration `results/calib/NVIDIA_GeForce_RTX_2080_Ti_cuda2_20261112-170614/`: DRAM
    ~582 GB/s (94% of 616), L2 ~2.1 TB/s, L2 knee between 5.66 and 6.73 MB (matches 5.5 MB);
    PyTorch dispatch table (add 7.6 us eager, 1.6 us as a graph node).
  - **ncu works on cuda2**: `/tmp/nsight-compute-lock` belongs to another user, so ncu fails
    with `InterprocessLockFailed` unless `TMPDIR` is private (`run_cuda2.sh` sets
    `TMPDIR=$WORK/tmp`). Counters are allowed (no `ERR_NVGPUCTRPERM`); no helpdesk email needed.
  - SmolVLA sweep, same 7 configs as the A100 plus a baseline repeat, all bit-exact
    (`traces/*/NVIDIA_GeForce_RTX_2080_Ti_*`, `results/traces_summary.csv`). Baseline: eager
    491 ms (repeat 497, +1.2%), prefix 75 ms, eager denoise 407 ms, graph denoise 40.1 ms
    (10.2x), hybrid 116 ms (4.2x). Versus A100: eager 1.3x slower (CPU-bound, older host),
    but graph denoise only 6 to 19% slower despite 2.5x less DRAM bandwidth: the kernels are
    latency bound.
  - **Eager timing on cuda2 has a host slow mode.** The 1-camera config's eager step was
    bimodal in its first run (459 vs 588 ms across reps) and stuck at ~596 ms (denoise 522 ms)
    in an unpinned rerun at 21:02, while the 3-camera baseline rerun right after was normal
    (484 ms). Pinned with `numactl --cpunodebind=N --membind=N` the 1-camera config gives
    457 ms (node 1, the GPUs' node) and 461 ms (node 0), so it is OS scheduling on the shared
    40-thread host, not NUMA distance and not the workload; graph times are unaffected (37 ms).
    Both unpinned 1-camera runs are kept in `traces/`; prefer runs with a small eager std, and
    pin future eager timings on cuda2 (`numactl --cpunodebind=1 --membind=1`). The pinned test
    outputs are only in `/scratch/am16489/tmp/numa_test` on cuda2.
  - **Standalone CUDA C++ benchmarks (course requirement) build and run on cuda2**
    (`results/bench/NVIDIA_GeForce_RTX_2080_Ti_cuda2_20261112-175909/`, `scripts/run_bench.sh`):
    our kernels (`bench/kernels/kernels.cuh`: WMMA FP16 GEMM with split-K, FP32 SIMT GEMM,
    tiled online-softmax attention, softmax, LayerNorm/RMSNorm, RoPE, SiLU/GELU, elementwise,
    scan) all pass correctness checks against cuBLAS or a CPU reference. At the expert shapes
    (M = 50) our GEMM is 1.2 to 1.5x cuBLAS; our attention is ~9x slower than PyTorch's
    memory-efficient kernel for the vision tower (1.37 ms vs ~0.15 ms).
  - **C++ replay of the traced step** (`bench/chains/chain_replay.cu`, lists from
    `profile/extract_chain.py`): the 10-step denoise loop with cuBLAS GEMMs takes 39.4 ms as a
    CUDA graph vs 40.1 ms for the real PyTorch graph (-1.6%). In C++, eager is 44.7 ms vs
    PyTorch eager 407 ms: ~89% of SmolVLA's eager denoise time is Python/framework dispatch,
    and the C++ eager/graph ratio is only 1.14x (0.98x with our kernels) vs 10.2x in PyTorch.
    Full step: 75 ms (cuBLAS) / 146 ms (our kernels, slow attention) vs 64 ms traced GPU time.
    Per-launch times of one flow step: 3.07 ms (cuBLAS) / 3.79 ms (ours) vs 3.45 ms traced
    (the trace includes ~1 us profiler bias per kernel).
  - **ncu per-kernel features** (`results/ncu/NVIDIA_GeForce_RTX_2080_Ti_cuda2_20261112-190706/`,
    `scripts/run_ncu.sh`, 16 metrics in 3 passes, application replay, boost clocks): the C++
    flow step with our kernels (L2 warm and flushed), with cuBLAS, and the real SmolVLA step
    (prefix + first flow step = 3,008 kernels; the other 9 flow steps repeat the same kernels).
    Findings: per flow step the GEMMs read 264 MB from DRAM (= the 250 MB of weights: weights
    stream from DRAM every step on a 5.5 MB L2) while elementwise kernels hit L2 (99.6%).
    Flushing L2 before every kernel costs +18% (4.7 to 5.6 ms) and raises DRAM traffic from
    408 to 692 MB: the value of inter-kernel L2 residency on this GPU. Real step: 70% of
    kernels under 5 us, median achieved occupancy 13.5%. ncu cost: SASS-instrumented FLOP
    counters (`smsp__sass_thread_inst_*`) and kernel replay made a 1,300-kernel run take 35+
    min; without them ~0.23 s per kernel. FLOPs come from the shapes instead. ncu does not
    support Maxwell (cuda4).
- **Other CIMS GPUs (2026-11-12, user approved; `scripts/run_cims_host.sh`, same torch build):**
  - cuda3, TITAN V (Volta): calibration (DRAM 650 GB/s, L2 ~1.8 TB/s, knee 4 to 8 MB),
    dispatch table, bench (all correctness checks pass), full 7-config trace sweep + baseline
    repeat, all bit-exact. **Baseline: eager 343 ms (fastest of the three GPUs: its Xeon Gold
    5118 host dispatches faster), graph denoise 61.5 ms (slowest), graph speedup 4.6x vs
    10.2x on the 2080 Ti and 9.0x on the A100.** The GPU ordering inverts between eager and
    graph mode: the graph speedup is set by host dispatch cost vs GPU kernel time, not by the
    workload alone (section 3, hypothesis 3). C++ replay, 10-step loop as a graph: 52.8 ms
    (cuBLAS) vs 39.4 ms on the 2080 Ti, although the TITAN V has more DRAM bandwidth.
  - cuda4, GTX TITAN X (Maxwell): C++ calibration only (DRAM 264 GB/s, L2 ~650 GB/s, knee
    ~4 MB). No WMMA, no ncu.
  - cuda5, RTX 4070: pipeline started, but the GPU was held by another user's job (11.2 of
    12 GB, 0% utilization) for hours; see `logs/cuda5-pipeline.log`. Rerun when free:
    `docs/cuda2_runbook.md` section 6.
- **Workloads.** SmolVLA (`lerobot/smolvla_base`, 450M, fits the 2080 Ti) and OpenVLA-7B
  (`openvla/openvla-7b`, A100 only, since L4 is down). pi0 deferred (gated PaliGemma; needs the
  user's HF token). GR00T N1.5-3B optional.
  - Code: `workloads/trace_common.py` (timing + profiling), `workloads/trace_parse.py`
    (pandas-only trace parser), `workloads/trace_smolvla.py`,
    `scripts/cluster/trace_smolvla.sbatch` (run with `HF_HUB_OFFLINE=1`; weights are in
    `$HF_HOME=/scratch/am16489/.cache/huggingface`, including the SmolVLM2-500M backbone).
  - Envs (git-ignored): `envs/openvla` (torch 2.2.0 cu121, transformers 4.40.1),
    `envs/smolvla` (`lerobot[smolvla]`, torch 2.10.0 cu128, transformers 4.57.6). Recreate with
    `UV_CACHE_DIR=/scratch/am16489/.cache/uv uv venv -p 3.10 envs/smolvla && VIRTUAL_ENV=$PWD/envs/smolvla uv pip install "lerobot[smolvla]" pandas`.
- **Next:**
  1. Cross-GPU (flavor B): the 2080 Ti traces are in (`git pull`); extend `launch_model.py` to take
     a source run and a target GPU: dispatch table and b, p from the source host; kernel times
     scaled to the target (per-kernel roofline transfer with the L2 term, section 4.2);
     g_eager, delta_node from the target's C++ calibration. Validate on the A100 sweep.
  2. Run `scripts/run_bench.sh bench/build-cluster` on the A100 (C++ kernels and replay on the
     target GPU: validation of per-kernel transfer on code we control); build with
     `make -C bench ARCHS="75 70 80 89" BUILD=build-cluster` (needs cuBLAS from the conda
     CUDA 12.4 env). ~2 GPU-minutes.
  3. cuda2 is done (section above). Use the ncu feature tables as per-kernel source features
     (DRAM/L2 bytes, occupancy; FLOPs from shapes) for the transfer model. TITAN V (cuda3) is a
     second validation target with full traces; the RTX 4070 (cuda5) is pending.
- GPU-hours used on `csci_ga_3033_025-2026fa`: 3.23 of 300 (2026-11-12 22:30).
- Pending for the user (CIMS): see "Needs the user" in section 1 and the cuda2 commands in
  section 6. The user asked for a full "what to run on cuda machines and elsewhere" list once
  the cluster runs are finished.

### 0.2 Memory note to carry over

The earlier session kept this as a Claude Code memory under the wildfire project, which a
session started in this folder does not see. To add it here, save the block below as
`~/.claude/projects/-scratch-am16489-vla-gpu-latency/memory/gpu-budget.md` and add the index
line to `MEMORY.md` in the same folder (or ask Claude to "save the section 0.2 memory note").

```markdown
---
name: gpu-budget
description: This project's Slurm account is csci_ga_3033_025-2026fa with a 300 GPU-hour cap; estimate and tally GPU jobs against it
metadata:
  type: project
---

The GPU Architecture course project (cross-GPU VLA latency + CUDA-graph speedup prediction) lives at `/scratch/am16489/vla-gpu-latency`, GitHub `axle962/vla-gpu-latency` (pushed via the `github-me` SSH alias). Its Slurm account is `csci_ga_3033_025-2026fa`, with its own 300 GPU-hour cap (`gres/gpu=18000` GPU-minutes; stated by the user 2026-11-10). The user's wildfire project (`/scratch/am16489/wildfire-spread-simulation`) has a separate account, `csci_ga_2271-2026fa`, with its own 300 GPU-hour cap.

**Why:** two courses, two allocations; submitting under the wrong account bills the wrong budget.

**How to apply:** never submit this project's jobs under `csci_ga_2271-2026fa` (or the reverse). Estimate GPU-hours before submitting, smoke-test before sweeps, and request 1 GPU unless needed. Tally only this account, with `-D` so requeued attempts after failed node boots (frequent here) are counted:
`sacct -D -S 2026-09-01 -X -A csci_ga_3033_025-2026fa -n -P --format=ElapsedRaw,AllocTRES | awk -F'|' '$2~/gres\/gpu/{s+=$1} END{print s/3600" GPU-h"}'`
State the cost when proposing multi-run experiments.
```

Index line for `MEMORY.md`:

```markdown
- [GPU budget](gpu-budget.md) — Slurm account csci_ga_3033_025-2026fa, 300 GPU-h cap; estimate and tally GPU jobs
```

---

## 1. Status and open items (check these first)

| Item | Status | Action |
|---|---|---|
| Professor approval | Email drafted Sep 30 (after the "before Sep 30 lecture" deadline for modified projects). **Approval status unknown here.** | Confirm approval and any scope changes he asked for. Record them in section 9. |
| Team size | Unknown (handout allows 1 to 4; bigger team, more output expected) | Fill in. |
| Report and presentation deadlines | Unknown | Fill in; the milestone plan in section 8 is relative. |
| Nsight Compute permissions on cuda2 | **Works (2026-11-12)** with a private `TMPDIR` (the `/tmp` lock file belongs to another user). No helpdesk email needed. | None. |
| Target GPU access | **Updated 2026-11-10.** The Slurm cluster reachable from `/scratch` (login node `b-9-198`) lists T4, V100, A100, L4 and RTX 6000-class (`g4-standard-48`) partitions, but a submit filter lets account `csci_ga_3033_025-2026fa` use only **`c12m85-a100-1` (1x A100) and `g2-standard-12` (1x L4)** (checked with `sbatch --test-only`). H200 and L40S (Torch spec sheet) are not on this cluster; H100, B200, PRO 6000 are external. | Calibrate A100 and L4. Ask the course staff / HPC whether the account can get T4, V100 or `g4-standard-48` (PRO 6000 is the key L2 GPU), and how to reach Torch's H200/L40S. |
| VLA-Perf prior work | **Read 2026-11-10** (section 3.4). Pure spec-sheet roofline; explicitly does not model launch overhead; no L2 term; no graph prediction; validated only on RTX 4090. | Verify Table 1 and Section 4 numbers against the PDF before quoting them. |

---

## 2. Proposal (as sent for approval)

> We profile the kernels of vision-language-action (VLA) robot policies on a source GPU (RTX 2080 Ti, CIMS) and build a cross-GPU model that predicts their end-to-end inference latency, and the speedup from CUDA-graph capture, on target GPUs with different specifications (L40S, H100, H200, B200). The model makes kernel launch overhead and L2 working-set fit first-class terms, because batch-1 robotics inference is dominated by many short, small kernels, and spec-ratio and roofline models mispredict exactly this regime.

Short slide title: *Predicting Robot-Policy GPU Latency Across Architectures*.

---

## 3. Novelty (what the paper must defend)

1. **New regime.** Classic GPU performance models (Hong and Kim MWP/CWP, GPUPerf, cross-benchmarking under frequency scaling) target throughput kernels that saturate the GPU. Batch-1 VLA inference is the opposite: thousands of short kernels where launch overhead and cache residency dominate.
2. **Explains observed anomalies that spec ratios cannot.** Public 2026 benchmarks (Yotta Labs) show:
   - H200 running GR00T v1.7 (297 ms) and OpenVLA (273 ms) *slower* than H100 (180 ms, 173 ms), roughly at RTX 4090 level.
   - RTX PRO 6000 (128 MB L2) beating H200 (2.7x the bandwidth) on Lingbot-VLA at default chunk size, reversing at larger working sets.
   - OpenVLA plateauing near 173 ms on Hopper/Ada but dropping to 73 ms on B300: a per-inference floor beyond raw compute.
   A model with explicit launch and L2 terms should predict these inversions. That is the headline test.
3. **Predicts the optimization, not only the baseline.** CUDA graphs roughly halve VLA latency in recent real-time work (pi0 on RTX 4090: about 100 ms to 40 to 50 ms, roughly 2x), but nobody predicts *when* graphs pay off. Hypothesis: faster GPUs shrink kernel time but not CPU launch cost, so graph benefit grows with GPU generation.
4. **Differentiation from VLA-Perf** (read 2026-11-10). W. Jiang, J. Clemons, K. Sankaralingam, C. Kozyrakis (NVIDIA Research), "How Fast Can I Run My VLA? Demystifying VLA Inference Performance with VLA-Perf", arXiv:2602.18397, Feb 2026, code at github.com/NVlabs/vla-perf.
   - Method: "an analytical, roofline-based performance model", per operator
     `T_o = max(FLOPs_o / FLOP/s_h, Bytes_o / MemBW_h)`, summed per component plus network
     transfer terms. Hardware is described by spec-sheet numbers only, and it "assumes that the
     maximum theoretical compute capability and memory bandwidth are attainable for every operator".
   - It says explicitly that "kernel launch latencies, operating system interference, and
     runtime library overhead ... are not explicitly modeled". No L2/cache term; CUDA graphs
     appear only as related work.
   - Validation: a single platform (pi0 on RTX 4090 vs a Triton implementation), real/roofline
     "fidelity" of 73 to 83 percent; the authors consider "exceeding 80% sufficient". Models are
     pi0 variants, BF16 (FP16 fallback), batch 1. GPUs studied: Jetson Thor, RTX 4090, A100, H100, B100.
   - **Our angles:** (a) it predicts an optimistic upper bound and cannot produce the H200 < H100
     or PRO 6000 > H200 inversions, because both need launch and cache terms; (b) we predict from a
     measured source-GPU profile (flavor B), not peak specs; (c) we predict CUDA-graph speedup,
     which a roofline model gives as 1.0 by construction; (d) we validate on several real target
     GPUs with percentage error per (workload, GPU). Use VLA-Perf's roofline as baseline 2 directly.

---

## 4. Approach

### 4.1 Pipeline
1. **Workload extraction.** Trace kernel sequences, shapes and launch counts of small open VLAs (pi0-class action expert, OpenVLA-scale policy, optionally GR00T) with Nsight Systems on a GPU where the full model fits. Export per-step kernel lists.
2. **Standalone CUDA C++ benchmarks.** Rebuild the traced workloads as standalone programs: GEMMs at the traced shapes, attention, RMSNorm/LayerNorm, RoPE, activations, and full denoising/action-chunk chains. Each runs in two modes: eager launches and CUDA-graph capture/replay. **These must compile and run on CIMS (`cuda2`)**; this is a hard course requirement.
3. **Source profiling (2080 Ti).** Per kernel, collect from Nsight Compute: duration, DRAM bytes read/written, L2 bytes and hit rate, achieved occupancy, FLOPs/instruction mix, registers and shared memory, grid/block dims.
4. **Model.** Predict each kernel's time on the target, add launch terms, sum over the trace (4.2).
5. **Validation.** Measure real latency and graph speedup on targets. Use both the extracted benchmarks and full models where they fit.
6. **Report** predicted vs measured with percentage error for every (workload, target GPU) pair, which the handout requires, against baselines (4.3).

### 4.2 Starting model formulation (hypothesis, refine with data)
Per kernel k on target GPU G:

- Compute time: `C_k(G) = FLOPs_k / (peak_G * eff_k)`, where `eff_k` is the efficiency measured on the source GPU, transferred with an occupancy/SM-count correction for small grids. A batch-1 grid may not fill 132+ SMs.
- Memory time: `M_k(G) = DRAMbytes_k(G) / BW_DRAM(G) + L2bytes_k / BW_L2(G)`, where `DRAMbytes_k(G)` depends on whether the working set (weights + activations touched across consecutive kernels) fits in `L2(G)`. This is the cache-aware term that should explain the PRO 6000 vs H200 inversion.
- Kernel time: `t_k(G) = max(C_k, M_k) + fixed_k(G)`.
- **Eager step time**: `T_eager(G) ≈ Σ_k max(t_k(G), L_launch(G))`, plus host gaps. When kernels are shorter than CPU launch latency, the GPU starves and launch cost dominates. This is the "launch-bound" term.
- **Graph step time**: `T_graph(G) ≈ L_graph(G) + Σ_k (t_k(G) + δ_node(G))`.
- **Predicted graph speedup** = `T_eager / T_graph`.
- Optional: a small regression on residuals (gradient boosting or linear on counter features), reported as an ablation against the purely analytical model.

**Design decision to record in the paper:** whether `L_launch(G)`, `BW_L2(G)` and `δ_node(G)` come from spec sheets only, or from a few tiny calibration microbenchmarks on the target (empty kernel, bandwidth sweep). Flavor B is strictest if the target is described by specs only. Calibration microbenchmarks can be an ablation ("specs-only vs specs + 3 microbenchmarks"). The microbenchmarks exist in `bench/calib/` (section 10).

**First calibration evidence (A100-SXM4-40GB, 2026-11-10, `results/calib/NVIDIA_A100*`)
contradicts the `max(t_k, L_launch)` form.** Eager back-to-back empty kernels are spaced
2.53 us apart (equal to the 2.53 us CPU cost of one `cudaLaunchKernel`); graph nodes are
0.79 us apart. But for a chain of 100 dependent busy-wait kernels of duration D (grid of 1 block), eager
takes about `D + 2.1 us` per kernel and graph about `D + 0.1 us` for every
D from 2 to 100 us. The roughly 2 us eager penalty does not disappear when the kernel is
longer than the CPU launch cost, so it is a GPU-side per-kernel cost (dependency resolution
and kernel start for stream launches), not CPU starvation. Working hypothesis:
`T_eager ≈ Σ_k (t_k + g_eager(G))` with `g_eager ≈ 2 us` plus a CPU-starvation term only when
`t_k + g_eager < L_launch`, and `T_graph ≈ L_graph + Σ_k (t_k + δ_node(G))`. Check on L4 and
cuda2, and with full-grid kernels, before committing to it.

Note: `L_launch` is partly a property of the **host CPU and driver**, not the GPU. Record the
host CPU model with every calibration run, and say in the paper that cross-GPU launch-cost
transfer assumes a comparable host (or measure it).

### 4.3 Baselines to beat
1. Spec-ratio scaling: time scaled by the ratio of peak FLOPs or bandwidth.
2. Pure roofline per kernel, no launch or L2 terms (this is the VLA-Perf model).
3. The model without the L2 term, and without the launch term (ablations).
4. VLA-Perf itself (code is public at NVlabs/vla-perf), run on our workloads and GPUs.

### 4.4 Metrics
- MAPE and per-case percentage error for end-to-end step latency, per target GPU and per workload.
- Error on predicted CUDA-graph speedup.
- Whether the model correctly predicts the *ordering* of GPUs per workload, especially the documented inversions (rank correlation).
- Sweeps: action chunk size, number of denoising/flow steps, number of camera views, precision (FP16 everywhere; BF16/FP8 only on GPUs that support them).

---

## 5. Hardware

**Verify every number with `bench/build/device_info` on each machine; do not trust this table blindly.**

| GPU | Where | Arch / CC | SMs | L2 | DRAM BW | Memory | Notes |
|---|---|---|---|---|---|---|---|
| RTX 2080 Ti (x2) | CIMS `cuda2` (verified) | Turing, sm_75 | 68 | 5.5 MB | ~616 GB/s | 11 GB GDDR6 | Source GPU. FP16 tensor cores; **no BF16, no FP8; FlashAttention-2 unsupported (needs Ampere+)**. Full 7B VLAs do not fit in FP16. |
| TITAN V (x2) | CIMS `cuda3` (found 2026-11-12) | Volta, sm_70 | 80 | 4.5 MB | ~653 GB/s | 12 GB HBM2 | Tensor cores; similar bandwidth to the source, different architecture. |
| RTX 4070 | CIMS `cuda5` (found 2026-11-12) | Ada, sm_89 | 46 | 36 MB | ~504 GB/s | 12 GB GDDR6X | **Similar DRAM bandwidth to the 2080 Ti but 6.5x the L2: isolates the cache term on CIMS.** GPU shared with other users. |
| GTX TITAN X (x2) | CIMS `cuda4` (found 2026-11-12) | Maxwell, sm_52 | 24 | 3 MB | ~336 GB/s | 12 GB | No tensor cores, no WMMA: C++ calibration only. |
| T4 | cluster `n1s8-t4-1` (not allowed for our account) | Turing, sm_75 | 40 | 4 MB | ~320 GB/s | 16 GB | Same arch as source, different specs: cleanest flavor-B test. |
| V100 | cluster `n1s4-v100-1` etc. (not allowed for our account) | Volta, sm_70 | 80 | 6 MB | ~900 GB/s | 16/32 GB | |
| A100 | cluster `c12m85-a100-1` (allowed) | Ampere, sm_80 | 108 | 40 MB | ~1.6 to 2.0 TB/s | 40/80 GB | |
| L4 | cluster `g2-standard-12` (allowed) | Ada, sm_89 | 58 | 48 MB | ~300 GB/s | 24 GB | Huge L2 relative to bandwidth. |
| RTX PRO 6000 | cluster `g4-standard-48` (to confirm; not allowed for our account) | Blackwell, sm_120 | TBD | 128 MB | ~1.8 TB/s | 96 GB | Key GPU for the L2 inversion. Runs our sm_89 PTX via JIT. |
| L40S | NYU Torch (68 GPUs) | Ada, sm_89 | 142 | 96 MB | ~864 GB/s | 48 GB | Not on this cluster. |
| H200 | NYU Torch (29 GPUs) | Hopper, sm_90 | 132 | 50 MB | ~4.8 TB/s | 141 GB | Not on this cluster. |
| H100 | External / unconfirmed | Hopper, sm_90 | 132 (SXM) | 50 MB | ~3.35 TB/s (SXM) | 80 GB | |
| B200 | External / unconfirmed | Blackwell | TBD | TBD | TBD | TBD | |

Cluster rows are from public spec sheets and must be replaced by `device_info.csv` values.
The CIMS machines were checked on 2026-11-12 (`docs/cuda2_runbook.md` section 6): cuda3 (TITAN V), cuda4 (TITAN X) and cuda5 (RTX 4070) are free extra targets that also satisfy the "runs on CIMS" rule; cuda1 is unreachable. Their rows above are spec-sheet values until `device_info.csv` and the bandwidth sweep from those hosts replace them.

---

## 6. Environment and workflow

### CIMS access
```bash
ssh am16489@access.cims.nyu.edu
ssh cuda2.cims.nyu.edu
module load cuda-12.4          # nvcc is NOT on PATH by default; redo after every login
nvidia-smi                      # check the GPUs are idle before timing anything
git clone git@github.com:axle962/vla-gpu-latency.git && cd vla-gpu-latency
make -C bench && scripts/check_profilers.sh && scripts/run_calib.sh
```
Home quota is 4 GB: keep large traces and `.ncu-rep`/`.nsys-rep` files out of home, or compress them and pull them back.

### Profiler sanity check (do this first)
`scripts/check_profilers.sh` runs `ncu` with three counters and an `nsys` trace on the
launch benchmark. Generate reports on CIMS/cluster, then analyze `.ncu-rep`/`.nsys-rep` in the
Nsight GUI (the handout explicitly allows this).

### Where Claude Code runs
On the Slurm cluster login node, in `/scratch/am16489/vla-gpu-latency`. It compiles there
with the conda nvcc 12.4, submits GPU work with `sbatch` (never runs GPU work on the login
node), and the user runs the CIMS steps (cuda2 is not reachable from here).

### Timing rules learned in Lab 1 (apply everywhere)
- **CUDA 12+ lazy module loading:** the first kernel launch includes loading the GPU code. In Lab 1 this inflated GPU times by several times (best speedup 12.5x with default loading vs 47.5x with `CUDA_MODULE_LOADING=EAGER`). Always warm up, or set `CUDA_MODULE_LOADING=EAGER`, and say which in the paper. The harness records it in every CSV row.
- Use `cudaEvent` timing, not `clock()` (which measures CPU time).
- Always `cudaDeviceSynchronize()` before stopping a host timer; launches are asynchronous.
- Repeat runs (at least 5, ideally 20+ for short kernels), report mean and spread, and use idle GPUs.
- Compile with an explicit `-arch` (e.g. `-arch=sm_75` on cuda2) to avoid JIT compilation on first run.

---

## 7. Course requirements (from the handout)

- **Must run on the CIMS CUDA cluster.** "It works on my Mac" is not accepted. Other GPUs are for validation only.
- **Report predicted vs real performance and analyze percentage error** (explicit Project 2 requirement).
- Justify the choice of method (analytical, ML or hybrid).
- **Final report is a scientific paper** with sections: Abstract, Introduction, Survey, Problem Definition and Challenges, Proposed Idea, Experimental Setup, Experiments and Analysis, Conclusions.
- End-of-semester group presentation.
- Suggested benchmarks: Parboil, Rodinia. Useful as sanity checks of the predictor on classic kernels before the VLA workloads.
- Avoid proprietary dependencies where possible. cuBLAS/cuDNN ship with the toolkit; if you use them for GEMMs, also write your own tiled GEMM so the predictor is tested on code you control.

---

## 8. Milestone plan (relative; set dates once deadlines are known)

1. **Setup and literature (week 1).** Confirm approval. Verify ncu/nsys on cuda2. deviceQuery every GPU. Read VLA-Perf (done), Hong and Kim, GPUPerf, Wang/Liu/Chu, the two handout DOIs, the real-time VLA paper, and Reflex/FlashDrive. Draft the Survey section.
2. **Workload extraction (week 2).** nsys traces of 2 to 3 VLAs on a GPU where they fit (A100 80 GB or PRO 6000 here). Export kernel lists and shapes. Choose what to rebuild.
3. **Benchmarks (weeks 2 to 3).** Standalone CUDA C++ kernel chains with eager and graph modes; a timing harness emitting CSV (done: `bench/common/harness.cuh`). Calibration microbenchmarks (done: launch latency eager vs graph, spin-chain crossover, bandwidth/L2 sweep).
4. **Source profiling (week 3).** ncu metric collection on cuda2; scripts producing a per-kernel feature table.
5. **Model and baselines (weeks 3 to 4).** Implement 4.2 and 4.3 in Python; predict for every target.
6. **Validation runs (week 4).** Same benchmarks and full models on every target GPU. Compute errors; test the documented inversions.
7. **Paper and presentation (week 5).** Error tables, ablations, GPU-ordering results, lessons learned.

---

## 9. Decisions log (append as you go)

- 2026-09-30: Chose Project 2 flavor B with VLA focus (merged ideas: cross-GPU VLA latency prediction + "when do CUDA graphs pay off"). Approval email sent late; status: ____.
- 2026-11-10: Repo created at `/scratch/am16489/vla-gpu-latency`. Slurm account `csci_ga_3033_025-2026fa`, 300 GPU-h cap.
- 2026-11-10: Read VLA-Perf (section 3.4). It is a spec-sheet roofline with no launch, cache or graph terms; our contribution is complementary, and its roofline is our baseline 2.
- 2026-11-10: The Slurm cluster has T4, V100, A100, L4 and (probably) RTX PRO 6000 partitions, but our account may only submit to A100 (`c12m85-a100-1`) and L4 (`g2-standard-12`). Requesting T4 (same arch as the source GPU, isolates spec from architecture differences) and PRO 6000 (L2 inversion) access is worthwhile.
- 2026-11-11: Workload traces use torch.profiler (CUPTI) because nsys is not on the cluster.
  SmolVLA graph mode captures only the denoising loop (vision encoder has a host sync).
- 2026-11-11: Cluster sbatch files use `--no-requeue` after failed L4 boots billed ~2 GPU-h.
- 2026-11-10: Calibration microbenchmarks write one CSV schema:
  `gpu,bench,kernel,config,mode,timer,n_runs,mean_us,std_us,min_us,median_us,module_loading`
  (`config` is `;`-separated `key=value`; `timer` is `event` or `host`).
- 2026-11-12: cuda2 runs by Claude Code on cuda2 (section 0.1). WORK=/scratch/am16489 on cuda2
  (349 GB free; home quota 4 GB). The CIMS GPUs are shared: another user's jobs appeared on
  both GPUs mid-run, so `run_bench.sh` logs other GPU processes (`busy_warnings.txt`) and a
  contaminated bench run was deleted and rerun on the idle GPU 1 (same model as GPU 0).
- 2026-11-12: Own kernels in `bench/kernels/kernels.cuh`; cuBLAS is the reference. The C++
  replay keeps one weight buffer per GEMM row (a step streams its weights from DRAM, as in the
  model) and zero-filled data (timing does not depend on values). In replay graph mode,
  HtoD/DtoH copies are skipped (the captured SmolVLA loop has none); in eager mode HtoD comes
  from pageable memory and every DtoH is followed by a stream sync, as in PyTorch.
- 2026-11-12: ncu settings: kernel replay, `--clock-control none` (boost clocks like the timed
  runs), `--cache-control none` (warm, real sequence) plus one `all` (flushed) run as the cold
  bound on DRAM bytes.
- ____

---

## 10. Repo layout

```
.
├── CLAUDE.md                # this file
├── bench/                   # CUDA C++ (must build on cuda2)
│   ├── common/harness.cuh   # timing harness, CSV output, CUDA-graph capture helper
│   ├── calib/               # device_info, launch_latency, bandwidth (L2 sweep)
│   ├── kernels/             # kernels.cuh (our GEMM, attention, norm, rope, ...), kernels_bench.cu
│   ├── chains/              # chain_replay.cu + smolvla_{flow,full}.csv (from profile/extract_chain.py)
│   └── Makefile             # ARCHS ?= 75
├── traces/                  # nsys exports, kernel lists (small files only)
├── profile/                 # extract_chain.py, ncu_smolvla.py, ncu_features.py
├── results/bench/           # kernels_bench + chain_replay CSVs per GPU run (scripts/run_bench.sh)
├── results/ncu/             # per-kernel ncu feature tables per GPU run (scripts/run_ncu.sh)
├── model/                   # Python predictor, baselines, ablations
├── results/calib/           # calibration CSVs per GPU run
├── validate/                # measured results per GPU (CSV), error analysis
├── scripts/                 # run_calib.sh, run_bench.sh, run_ncu.sh, run_cuda2.sh, check_profilers.sh, cluster/
└── paper/                   # LaTeX, figures
```

---

## 11. References gathered so far

Handout-suggested (Track 2):
- ACM, https://doi.org/10.1145/3639403
- IEEE BDCloud 2018, https://doi.org/10.1109/BDCloud.2018.00139
- Q. Wang, C. Liu, X. Chu, "GPGPU Performance Estimation for Frequency Scaling Using Cross-Benchmarking"
- S. Hong, H. Kim, "An Analytical Model for a GPU Architecture with Memory-level and Thread-level Parallelism Awareness", ISCA 2009, https://doi.org/10.1145/1555754.1555775
- J. Sim, A. Dasgupta, H. Kim, R. Vuduc, "A Performance Analysis Framework for Identifying Potential Benefits in GPGPU Applications" (GPUPerf), PPoPP 2012

Found during scoping:
- VLA-Perf: W. Jiang, J. Clemons, K. Sankaralingam, C. Kozyrakis, arXiv:2602.18397 (2026), https://arxiv.org/abs/2602.18397, code https://github.com/NVlabs/vla-perf
- VLA inference GPU benchmarks (Yotta Labs, Aug 2026): https://www.yottalabs.ai/post/vla-inference-gpu-benchmarks
- Running VLAs at Real-time Speed (review): https://www.themoonlight.io/review/running-vlas-at-real-time-speed (Ma et al. 2025, cited by VLA-Perf for the RTX 4090 Triton pi0 numbers and CUDA-graph gains)
- Reflex: Fast and Predictive VLA Models for Reaction-Critical Manipulation: https://arxiv.org/pdf/2608.14379
- FlashDrive: Flash VLA Inference for Autonomous Driving: https://arxiv.org/pdf/2608.12932
- NYU Torch spec sheet: https://services.rt.nyu.edu/docs/hpc/spec_sheet

---

## 12. Instructions for Claude Code

- Writing style for any prose (paper, README, comments): **no em dashes**; use commas, periods or parentheses.
- All CUDA code must build with `nvcc` from CUDA 12.4 on sm_75 without BF16/FP8/FlashAttention dependencies. Guard newer-architecture paths with `#if __CUDA_ARCH__ >= ...`.
- Every benchmark binary prints machine-readable CSV in the schema of section 9 (gpu name, kernel id, config, mode eager/graph, n_runs, mean_us, std_us, ...) so results from all GPUs merge cleanly. Use `bench/common/harness.cuh`.
- Never trust a single timing. Warm up, use cudaEvents, repeat, and record `CUDA_MODULE_LOADING`.
- Keep the analytical model readable and separate from any ML residual model so ablations are trivial.
- Before claiming a result, check it against the documented anomalies in section 3.
- On the Slurm cluster: GPU work only through `sbatch` with account `csci_ga_3033_025-2026fa`; state the GPU-hour estimate before submitting multi-job sweeps. On CIMS cudaN hosts there is no Slurm: run directly on an idle GPU (section 0.0, `docs/cuda2_runbook.md`).
