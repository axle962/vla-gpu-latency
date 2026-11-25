Launch-Bound and Cache-Aware: Predicting VLA Inference Latency and CUDA-Graph Speedup
Across GPU Generations
NYU GPU Architecture and Programming, Fall 2026, Project 2 (flavor B)
Team: Aditya Maddur Guruprakash, Archana Jayakumar, Shrikar Madhu

Report: paper/main.pdf (LaTeX source paper/main.tex plus paper/*_section.tex).

Everything below runs on the CIMS CUDA machines (tested on cuda2 = 2x RTX 2080 Ti, cuda3 =
TITAN V, cuda5 = RTX 4070). Only CUDA 12.4 (module cuda-12.4) and Python packages installed
with pip/uv into a directory you own are used; nothing needs root.


1. WHAT IS IN HERE
------------------
bench/       CUDA C++ (our own code, builds with nvcc 12.4 for sm_75 / sm_70 / sm_89)
  calib/       device_info, launch_latency (eager vs CUDA graph), bandwidth (L2/DRAM sweep)
  kernels/     our kernels (WMMA FP16 GEMM, FP32 GEMM, attention, softmax, LayerNorm/RMSNorm,
               RoPE, SiLU/GELU, scan) + kernels_bench (correctness vs cuBLAS / CPU, timing)
  chains/      chain_replay: the traced SmolVLA step rebuilt in C++, eager vs CUDA graph
workloads/   Python tracers (PyTorch profiler) for SmolVLA, pi0/pi0.5, pi0-FAST, OpenVLA, ACT,
             Diffusion Policy; dispatch/probe microbenchmarks; LIBERO/RoboTwin delay harness;
             placement.py (split-device hosting), multipolicy.py (several policies on one GPU),
             eps_probe.py (graph-worthiness calibration)
model/       the predictors and report scripts (pure Python, CPU only)
  cross_gpu.py         cross-GPU model (final variant slot_drv) -> results/predictions_cross_gpu.csv
  report_cross_gpu.py  MAPE tables, GPU ranking, inversion -> results/report_cross_gpu*.md
  figures.py           paper/figures/pred_vs_meas, mape_variants
  launch_model.py      same-GPU model
  graph_worthiness.py  graph speedup predicted from eager mode only
  placement_report.py  split-device prediction; multipolicy_report.py queueing model
  ablation.py, batch_predict.py, compile_predict.py, whatif.py, vlaperf_baseline.py,
  libero_analysis.py, robotwin_analysis.py, rodinia_predict.py, host_experiment.py
profile/     ncu feature extraction, chain extraction from traces
scripts/     run_*.sh drivers (see below); scripts/cluster/ = Slurm jobs used for the A100
traces/      per-run kernel lists (kernels.csv.gz) and meta.json of every traced run
results/     every measured CSV and every report the paper quotes
docs/        cuda2_runbook.md (step-by-step CIMS runbook), placement.md, multipolicy.md,
             graph_worthiness.md


2. SETUP ON CIMS (once)
-----------------------
  ssh <netid>@access.cims.nyu.edu
  ssh cuda2.cims.nyu.edu                 # or cuda3 / cuda5
  module load cuda-12.4                  # every login: puts nvcc, ncu, nsys on PATH
  nvidia-smi                             # pick an idle GPU; the machines are shared
  export WORK=/scratch/$USER             # home quota is only 4 GB; envs and weights go here
  cd <this directory>

Python environment (~6 GB, plus ~2 GB of SmolVLA weights from Hugging Face):
  scripts/run_cuda2.sh setup             # installs uv, Python 3.10, lerobot[smolvla], torch 2.10
Optional, for pi0 / pi0.5 / pi0-FAST (needs a Hugging Face login with the PaliGemma license):
  bash logs/setup_pi_env.sh


3. COMPILE AND RUN THE CUDA C++ CODE
------------------------------------
  make -C bench                          # sm_75 (cuda2); on cuda3 use ARCHS=70, cuda5 ARCHS=89
  scripts/run_calib.sh                   # ~2 min: device info, launch latency, bandwidth sweep
                                         #   -> results/calib/<gpu>_<host>_<time>/*.csv
  scripts/run_bench.sh                   # ~6 min: our kernels (correctness checks print ok/
                                         #   FAIL) + C++ replay of the SmolVLA step, eager vs graph
                                         #   -> results/bench/<gpu>_<host>_<time>/
  scripts/l2_sweep.sh                    # L2 working-set sweep on the C++ replay (~2 min)
Pick the GPU with CUDA_VISIBLE_DEVICES=<n>. On cuda2, GPU 0 was stuck at a reduced clock in
Nov 2026; use GPU 1. Every binary prints CSV:
  gpu,bench,kernel,config,mode,timer,n_runs,mean_us,std_us,min_us,median_us,module_loading


4. RERUN THE MODELS AND REPORTS (CPU only, from the committed results)
---------------------------------------------------------------------
  PY=$WORK/envs/smolvla/bin/python
  $PY model/cross_gpu.py                                 # cross-GPU predictions
  $PY model/report_cross_gpu.py > results/report_cross_gpu.md
  $PY model/report_cross_gpu.py --workload act > results/report_cross_gpu_act.md
                                   # also: diffusion, diffusionbench, pi0, pi05, pi0fast
  $PY model/figures.py                                   # paper figures
  $PY model/graph_worthiness.py                          # graph-worthiness rule
  $PY model/placement_report.py results/placement/<run>/
  $PY model/multipolicy_report.py results/multipolicy/<run>/
All of section 3 and 4 plus a smoke test of every workload in one command:
  scripts/integration_check.sh cpp models smoke          # ~50 min; summary in logs/integration-*/


5. RERUN THE GPU MEASUREMENTS (optional, long)
----------------------------------------------
Stages of scripts/run_cuda2.sh (any CIMS host; details in docs/cuda2_runbook.md):
  dispatch   PyTorch per-op dispatch table (~1 min)      probe   host/cold-cache probe (~2 min)
  trace      SmolVLA, 7 configurations (~25 min)         ncu     Nsight Compute counters (~20 min)
  more       Diffusion Policy + ACT (~45 min)            diffbench  autotuned cuDNN (~25 min)
  extras     batch sweep, torch.compile, energy (~1.5 h) pi      pi0 / pi0.5 / pi0-FAST traces
  rodinia    Rodinia sanity check
Teammates' studies:
  scripts/run_placement.sh [smoke]       split-device hosting (2 GPUs on cuda2/cuda3; ~10 min)
  scripts/run_multipolicy.sh smoke|quick|full    several policies on one GPU (needs MPS; 30-80 min)
  scripts/run_eps_probe.sh               graph-worthiness calibration (~1 min)
Task success (LIBERO, headless EGL): EXEC_HORIZON=10 scripts/run_libero.sh (hours per policy).
The A100 runs, RoboTwin and the RTX PRO 6000 runs used Slurm clusters (scripts/cluster/,
Empire AI); their results are in results/ and are reproduced from there by section 4.

Notes: Nsight Compute on cuda2 needs a private TMPDIR (the run scripts set TMPDIR=$WORK/tmp).
Timings need an idle GPU; the scripts warn when another process is on it.
