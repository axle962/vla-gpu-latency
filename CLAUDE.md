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
- Profiler check on cuda2: `scripts/check_profilers.sh`.

---

## 0.1 Resume here (paused 2026-11-10 ~23:35)

State when the last session paused:
- **Calibration done on A100** (`results/calib/NVIDIA_A100*`, two nodes; launch numbers agree
  within 2%). The first run's bandwidth data was deleted (old sweep starved small working sets;
  see its NOTE.txt). Findings are in section 4.2 (eager per-kernel penalty ~2 us persists for long
  kernels) and below: A100 BW_L2 ~2.65 TB/s, BW_DRAM 1.48 TB/s (95% of spec), L2 knee ~20 to 27 MB
  (half of the nominal 40 MB, consistent with A100's two L2 partitions).
- **L4 calibration has no results yet.** Job 28144 failed node boot several times, was requeued,
  and was finally `CANCELLED by 0` (the system) without producing output. Resubmit:
  `source scripts/cluster/site.sh && sbatch -p g2-standard-12 scripts/cluster/calib.sbatch`,
  then commit `results/calib/NVIDIA_L4*` and run `uv run python model/calib.py results/calib/*/`.
- **Workload extraction started.** Chosen VLAs (all ungated on HF): SmolVLA
  (`lerobot/smolvla_base`, 450M, pi0-style flow-matching expert, fits on the 2080 Ti so it is the
  one full VLA measurable on the source GPU) and OpenVLA-7B (`openvla/openvla-7b`, A100/L4 only).
  pi0 is deferred: it needs gated `google/paligemma-3b-pt-224` (user must accept the license and
  provide an HF token). GR00T N1.5-3B is ungated but heavy; optional.
  - `workloads/trace_common.py` (written, linted, **not yet run**): eager/graph step latency +
    torch.profiler kernel list (grid, block, regs, smem, aten op and input shapes) per step.
  - Envs (git-ignored): `envs/openvla` (torch 2.2.0 cu121, transformers 4.40.1; installed OK).
    `envs/smolvla` (`lerobot[smolvla]`, torch 2.10.0 cu128; installed OK). Recreate with
    `UV_CACHE_DIR=/scratch/am16489/.cache/uv uv venv -p 3.10 envs/smolvla && VIRTUAL_ENV=$PWD/envs/smolvla uv pip install "lerobot[smolvla]" pandas`.
    Set `HF_HOME=/scratch/am16489/.cache/huggingface`. No model weights downloaded yet.
    Note the two envs use different torch/CUDA builds (2.2.0 cu121 vs 2.10.0 cu128); record
    this in each trace's meta.json (trace_common.py already does).
  - Next: write `workloads/trace_smolvla.py` and `workloads/trace_openvla.py` (dummy 224x224
    images + instruction, FP16 everywhere so the 2080 Ti can run the same thing, batch 1), plus
    an sbatch that runs them on A100 (~15 min each, ~0.5 GPU-h total), then try CUDA-graph
    capture of the SmolVLA denoising loop for measured graph speedup.
- GPU-hours used so far on `csci_ga_3033_025-2026fa`: well under 0.5.
- Pending for the user (CIMS): see "Needs the user" in section 1 and the cuda2 commands in
  section 6. The user asked for a full "what to run on cuda machines and elsewhere" list once
  the cluster runs are finished.

---

## 1. Status and open items (check these first)

| Item | Status | Action |
|---|---|---|
| Professor approval | Email drafted Sep 30 (after the "before Sep 30 lecture" deadline for modified projects). **Approval status unknown here.** | Confirm approval and any scope changes he asked for. Record them in section 9. |
| Team size | Unknown (handout allows 1 to 4; bigger team, more output expected) | Fill in. |
| Report and presentation deadlines | Unknown | Fill in; the milestone plan in section 8 is relative. |
| Nsight Compute permissions on cuda2 | Untested | Run `scripts/check_profilers.sh` on cuda2. If it fails with `ERR_NVGPUCTRPERM`, email CIMS helpdesk immediately; the whole project depends on counters. |
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
Also check `nvidia-smi` on the other CIMS machines (`cuda1` to `cuda5`). If any has a different GPU, it is a free extra source or target that also satisfies the "runs on CIMS" rule.

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
- 2026-11-10: Calibration microbenchmarks write one CSV schema:
  `gpu,bench,kernel,config,mode,timer,n_runs,mean_us,std_us,min_us,median_us,module_loading`
  (`config` is `;`-separated `key=value`; `timer` is `event` or `host`).
- ____

---

## 10. Repo layout

```
.
├── CLAUDE.md                # this file
├── bench/                   # CUDA C++ (must build on cuda2)
│   ├── common/harness.cuh   # timing harness, CSV output, CUDA-graph capture helper
│   ├── calib/               # device_info, launch_latency, bandwidth (L2 sweep)
│   ├── kernels/             # GEMM, attention, norm, rope, activation kernels (todo)
│   ├── chains/              # VLA step chains rebuilt from traces, eager + graph (todo)
│   └── Makefile             # ARCHS ?= 75
├── traces/                  # nsys exports, kernel lists (small files only)
├── profile/                 # ncu metric scripts, per-GPU feature tables (CSV)
├── model/                   # Python predictor, baselines, ablations
├── results/calib/           # calibration CSVs per GPU run
├── validate/                # measured results per GPU (CSV), error analysis
├── scripts/                 # run_calib.sh, check_profilers.sh, cluster/ (sbatch, site.sh)
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
- GPU work only through `sbatch` with account `csci_ga_3033_025-2026fa`; state the GPU-hour estimate before submitting multi-job sweeps.
