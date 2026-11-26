Launch-Bound and Cache-Aware: Predicting VLA Inference Latency and CUDA-Graph Speedup
Across GPU Generations
NYU GPU Architecture and Programming, Fall 2026, Project 2 (flavor B)
Team: Aditya Maddur Guruprakash, Archana Jayakumar, Shrikar Madhu

The report is paper/main.pdf (LaTeX source in paper/main.tex and paper/*_section.tex).
Code and data are also at https://github.com/axle962/vla-gpu-latency.


0. THE SHORT VERSION
--------------------
The project asks: if you profile a robot policy (a VLA model) on one GPU, can you predict how
fast it runs on a different GPU, and how much CUDA graphs will speed it up there? At batch 1
these models launch thousands of tiny kernels, so the answer depends mostly on launch and
framework overhead, not on FLOPs or bandwidth. We measured four GPUs (RTX 2080 Ti, TITAN V,
RTX 4070 on CIMS, plus an A100 on the NYU cluster), wrote CUDA C++ benchmarks that rebuild
the traced model step, and built a Python model that predicts one GPU from another.

There are two halves to the code:
  - CUDA C++ (bench/). Needs nothing but the cuda-12.4 module. Builds in under a minute.
  - Python (workloads/, model/). Traces the real PyTorch models and runs the predictor.
    Needs a Python env that you install into your own directory (no root needed).

If you only have 15 minutes, do sections 2 and 3. That compiles and runs all of our CUDA code
and checks our kernels against cuBLAS. Section 4 reruns the predictor on the data we already
collected, and section 6 is for redoing the GPU measurements from scratch (hours).


1. WHAT'S IN THE FOLDERS
------------------------
bench/       Our CUDA C++ code. Builds with nvcc 12.4 for sm_75 (2080 Ti), sm_70 (TITAN V)
             and sm_89 (RTX 4070).
  calib/       device_info, launch_latency (eager launches vs a CUDA graph), bandwidth
               (sweeps the working set from L2-sized to DRAM-sized)
  kernels/     our own kernels: WMMA FP16 GEMM, FP32 GEMM, attention, softmax,
               LayerNorm/RMSNorm, RoPE, SiLU/GELU, scan. kernels_bench checks each one
               against cuBLAS or a CPU reference and times it.
  chains/      chain_replay: one traced SmolVLA step rebuilt as a C++ kernel sequence, run
               eagerly and as a CUDA graph. The kernel lists are the CSVs in this folder.
workloads/   Python tracers (PyTorch profiler) for SmolVLA, pi0/pi0.5, pi0-FAST, OpenVLA, ACT
             and Diffusion Policy; small PyTorch microbenchmarks (dispatch cost per op);
             the LIBERO and RoboTwin delay experiments; placement.py (split a policy across
             devices), multipolicy.py (several policies on one GPU), eps_probe.py.
model/       The predictors and the scripts that make the paper's tables. Pure Python, CPU.
  cross_gpu.py         the cross-GPU model (final variant is slot_drv)
  report_cross_gpu.py  error tables, GPU ranking, the eager/graph ordering flip
  figures.py           paper figures
  launch_model.py      same-GPU model
  graph_worthiness.py  predicts the CUDA-graph speedup from an eager run alone
  placement_report.py, multipolicy_report.py   the two teammate studies
  everything else      ablations, batch sweep, torch.compile, what-if, LIBERO/RoboTwin
                       analysis, Rodinia, host CPU experiment, VLA-Perf baseline
profile/     Nsight Compute feature extraction; turns a trace into a chain_replay kernel list
scripts/     The shell drivers used below. scripts/cluster/ has the Slurm jobs we used for
             the A100 (not needed on CIMS).
traces/      Per-run kernel lists (kernels.csv.gz) and meta.json for every traced run. The
             models read these, so section 4 works without a GPU.
results/     Every measured CSV and every report the paper quotes.
docs/        cuda2_runbook.md (the full step-by-step CIMS runbook), placement.md,
             multipolicy.md, graph_worthiness.md.


2. SETUP ON A CIMS MACHINE
--------------------------
Log in and pick a machine. cuda2 has 2x RTX 2080 Ti, cuda3 2x TITAN V, cuda5 one RTX 4070.

  ssh <netid>@access.cims.nyu.edu
  ssh cuda2.cims.nyu.edu
  module load cuda-12.4        # do this every time you log in, it puts nvcc on your PATH
  nvidia-smi                   # the machines are shared, so look for a GPU nobody is using
  cd <wherever you unzipped this>

Home has a 4 GB quota, which is too small for the Python env and model weights. Everything
big goes under $WORK instead. Use /scratch/$USER (create it if it's not there):

  mkdir -p /scratch/$USER
  export WORK=/scratch/$USER   # every script below reads this

That's all the C++ part needs. You can go straight to section 3.


3. COMPILE AND RUN THE CUDA C++ CODE
------------------------------------
  make -C bench                # builds for sm_75 (cuda2)
  make -C bench ARCHS=70       # on cuda3 (TITAN V)
  make -C bench ARCHS=89       # on cuda5 (RTX 4070)

This puts five binaries in bench/build/. Then:

  export CUDA_VISIBLE_DEVICES=1    # pick an idle GPU; on cuda2 use 1 (see below)
  scripts/run_calib.sh             # ~2 min: device info, launch latency, bandwidth sweep
  scripts/run_bench.sh             # ~6 min: our kernels + the C++ replay of the SmolVLA step
  scripts/l2_sweep.sh bench/build   # ~2 min: how much it helps when the weights fit in L2

Output goes to results/calib/<gpu>_<host>_<time>/ and results/bench/<gpu>_<host>_<time>/.
To check it worked, look at kernels_check.txt in the bench folder: every kernel prints "ok"
(24 lines on the 2080 Ti) and nothing says FAIL. chains.csv has the replay times; the 10-step
loop as a CUDA graph takes about 39 ms on the 2080 Ti.

Every binary prints CSV with the same columns, so runs from different GPUs can be merged:
  gpu,bench,kernel,config,mode,timer,n_runs,mean_us,std_us,min_us,median_us,module_loading

On cuda2, GPU 0 was stuck at a low clock in Nov 2026, so its timings are off. Use GPU 1.
Timings only mean something on an idle GPU; run_bench.sh writes busy_warnings.txt if another
process showed up on the GPU while it ran.


4. PYTHON ENVIRONMENT
---------------------
Needed for sections 5 and 6. It takes about 6 GB plus 2 GB of SmolVLA weights, all under
$WORK, and needs internet the first time.

  scripts/run_cuda2.sh setup

This installs uv (a fast pip) into ~/.local/bin, makes a Python 3.10 env in
$WORK/envs/smolvla, installs lerobot and PyTorch, downloads the SmolVLA weights from Hugging
Face, and finishes with a quick FP16 matmul on the GPU to show torch works.

The setup script does not pin versions, and newer lerobot releases moved modules around.
Right after setup, pin the versions we ran with and add the packages the report scripts need:

  export PATH=$HOME/.local/bin:$PATH
  VIRTUAL_ENV=$WORK/envs/smolvla uv pip install "lerobot[smolvla]==0.4.4" \
      "torch==2.10.0" "torchvision==0.25.0" "transformers==4.57.6" \
      pandas matplotlib tabulate scipy nvidia-ml-py

To check: $WORK/envs/smolvla/bin/python -c "import torch, lerobot; print(torch.__version__)"
should print 2.10.0+cu128.

Optional, only for pi0 / pi0.5 / pi0-FAST: these use Google's PaliGemma, which is gated. You
need a Hugging Face account that has accepted the PaliGemma license, then:

  HF_HOME=$WORK/.cache/huggingface $WORK/envs/smolvla/bin/huggingface-cli login
  bash logs/setup_pi_env.sh        # makes a second env, $WORK/envs/pi

setup_pi_env.sh has its paths fixed to /scratch/$USER, so it only works if WORK is exactly
/scratch/$USER.


5. RERUN THE MODEL AND THE PAPER'S TABLES (CPU only, no GPU)
------------------------------------------------------------
These read the traces and results that are already in the folder, so you can do this
without measuring anything.

  PY=$WORK/envs/smolvla/bin/python
  $PY model/cross_gpu.py                       # predictions -> results/predictions_cross_gpu.csv
  $PY model/report_cross_gpu.py > results/report_cross_gpu.md
  $PY model/report_cross_gpu.py --workload act > results/report_cross_gpu_act.md
        # other workloads: diffusion, diffusionbench, pi0, pi05, pi0fast
  $PY model/figures.py                         # paper/figures/pred_vs_meas, mape_variants
  $PY model/graph_worthiness.py                # graph speedup from eager mode
  $PY model/placement_report.py results/placement/<run>/
  $PY model/multipolicy_report.py results/multipolicy/<run>/

report_cross_gpu.md is the main result: prediction error (MAPE) for every source/target GPU
pair, and whether the model gets the GPU ranking right. If you got a git clone instead of the
zip, "git diff results/" afterwards shows whether you reproduced our numbers exactly.

Two scripts need code from outside this repo:
  - model/vlaperf_baseline.py needs the VLA-Perf code (github.com/NVlabs/vla-perf, we used
    commit 1e2b9a7) on the Python path. Its output is already in results/vlaperf_baseline.csv.
  - Rodinia (section 6) needs github.com/yuhc/gpu-rodinia cloned somewhere. Point
    RODINIA=<path> at it, or put it at ~/fall2026/GPU/ext/gpu-rodinia.


6. REDO THE GPU MEASUREMENTS (optional, takes hours)
----------------------------------------------------
Everything goes through scripts/run_cuda2.sh <stage>. Despite the name it works on any CIMS
host: it picks the idlest GPU itself (or set CUDA_VISIBLE_DEVICES) and names its build and
result folders after the host. Run the stages in this order:

  preflight   shows the GPU, driver, free disk, and which GPU it will use
  calib       same as section 3, plus a quick check that ncu and nsys work (~2 min)
  dispatch    PyTorch per-op dispatch cost table (~1 min)
  probe       host Python speed and cold-cache small kernels (~2 min)
  trace       SmolVLA in 7 configurations, eager vs CUDA graph (~25 min)
  bench       same as run_bench.sh above (~6 min)
  l2          same as l2_sweep.sh above (~2 min)
  ncu         Nsight Compute counters per kernel (~20 min)
  more        Diffusion Policy and ACT, downloads ~1.3 GB of weights (~45 min)
  diffbench   Diffusion Policy with cuDNN autotuning on (~25 min)
  extras      batch sweep, torch.compile, GPU energy (~1.5 h)
  pi          pi0 / pi0.5 / pi0-FAST traces (needs envs/pi from section 4)
  rodinia     Rodinia kernels under nsys (needs the Rodinia clone)
  results     lists the new result folders

Each one writes into traces/ or results/ with the GPU and host in the folder name, so new runs
never overwrite ours. docs/cuda2_runbook.md walks through every stage in more detail, with
what the output should look like.

Teammates' studies:
  scripts/run_placement.sh smoke     1 min toy run, no weights needed
  scripts/run_placement.sh           SmolVLA split across devices (~10 min; the GPU+GPU case
                                     needs two idle GPUs, so cuda2 or cuda3)
  scripts/run_multipolicy.sh smoke|quick|full
                                     several policies on one GPU, with and without MPS
                                     (30 to 80 min; skips the MPS cases if
                                     nvidia-cuda-mps-control isn't there)
  scripts/run_eps_probe.sh           calibration for the graph-worthiness rule (~1 min)
run_placement.sh refuses to start if someone else is on the GPU (FORCE=1 overrides).
run_eps_probe.sh refuses if the GPU is still in P0 from the previous run; wait about 30 s
until nvidia-smi shows P8 and run it again.

Checking everything at once:
  scripts/integration_check.sh cpp models smoke      # ~50 min
This builds and runs the C++ code, reruns every model script, and does a short run of every
workload. It runs offline, so run the "more" stage first (it downloads the ACT and Diffusion
weights the smoke test needs). The pi0 smoke test is skipped if envs/pi doesn't exist. One
PASS/FAIL line per step lands in logs/integration-<host>-<time>/SUMMARY.txt.

LIBERO (task success vs inference delay). This is simulation and takes hours per policy.
Install the simulator into each env you want to use, then run with a job list per GPU:

  VIRTUAL_ENV=$WORK/envs/smolvla uv pip install hf-libero
  EXEC_HORIZON=10 scripts/run_libero.sh "1=smolvla:fp16:0,2,4,6,8,10,15"

The job list reads gpu=policy:precision:delays, and the delays are in 50 ms control steps.
Policies are smolvla, pi0 and pi05 (the pi ones need envs/pi with hf-libero in it). It renders
headless with EGL, so no display is needed. The first run creates ~/.libero and downloads the
LIBERO assets into ~/.cache/libero. Results go to results/libero/.
scripts/libero_benchmark.sh reruns the official lerobot-eval benchmark (see its header).

What can't be rerun on CIMS: the A100 runs and RoboTwin used the NYU Slurm cluster
(scripts/cluster/), and the RTX PRO 6000 runs used Empire AI. Their results are in results/
and traces/, and section 5 rebuilds the paper's numbers from them.

Nsight Compute on cuda2 needs a private TMPDIR because another user owns the lock file in
/tmp. run_cuda2.sh sets TMPDIR=$WORK/tmp for you.
