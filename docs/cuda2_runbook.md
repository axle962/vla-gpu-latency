# cuda2 runbook: source-GPU runs on the RTX 2080 Ti

Written for a Claude Code session started on CIMS `cuda2` inside this repo, and for the user
doing it by hand. Everything here runs directly on cuda2: there is **no Slurm** and **no GPU-hour
budget** on CIMS (the `sbatch`/`csci_ga_3033_025-2026fa` rules in CLAUDE.md apply only to the
Greene-style cluster). The cluster session cannot reach cuda2, so these results come back
through git (or a tarball).

Goal: give the predictor its source GPU. The A100 traces, sweep and calibration already exist
(CLAUDE.md section 0.1); this runbook produces the same data on the 2080 Ti so that
`model/launch_model.py` can predict A100 latency from 2080 Ti measurements (flavor B).

All stages are wrapped by `scripts/run_cuda2.sh <stage>`. Read that script before running it.

## 0. Before starting (user)

```bash
ssh am16489@access.cims.nyu.edu     # access is a login gateway: never run GPU work there
ssh cuda2.cims.nyu.edu
module load cuda-12.4               # every login; nvcc, ncu, nsys are not on PATH otherwise
git clone https://github.com/axle962/vla-gpu-latency.git ~/vla-gpu-latency   # first time
cd ~/vla-gpu-latency && git pull
```

Start Claude Code inside `tmux` (`tmux new -s vla`, reattach with `tmux attach -t vla`) so a
dropped SSH connection does not kill a 25-minute trace run. Then ask Claude to "execute
docs/cuda2_runbook.md".

## 1. Rules for the session on cuda2

- Confirm the host first: `hostname` must be `cuda2` (or another CIMS `cudaN`, in which case say
  so and record it). Stop if it is `access` or any login node.
- `module load cuda-12.4` is a shell function. In a fresh non-interactive shell run
  `source /etc/profile.d/modules.sh 2>/dev/null; module load cuda-12.4` in the same command as
  anything needing nvcc/ncu/nsys, or check `command -v nvcc` first.
- **Home quota is 4 GB.** The Python env, uv/pip caches, Hugging Face cache and model weights
  must live under `$WORK` (step 2), never under `~`. Do not `pip install` into home.
- **The GPUs are shared.** Before every timing stage run `nvidia-smi`. If the chosen GPU shows
  utilization above ~5% or another user's process, wait and re-check (or pick the other GPU by
  exporting `CUDA_VISIBLE_DEVICES`); do not time on a busy GPU. Record which GPU index was used.
- Never edit measured CSVs by hand. If a stage fails, report the error output; do not "fix"
  results.
- Do not change code in `workloads/` or `model/` to make a run pass without telling the user:
  the A100 data was produced by this exact code, and the cross-GPU comparison depends on it.
  If a change is unavoidable (e.g. a torch API difference), make it minimal, guard it so the
  A100 path is unchanged, and describe it in the commit message.
- Writing style for anything committed: no em dashes.

## 2. Choose WORK (needs >= 15 GB free, not home)

```bash
df -h /scratch /tmp /var/tmp "$HOME" 2>/dev/null; quota -s 2>/dev/null
export WORK=/scratch/am16489        # adjust to a writable location with space
mkdir -p "$WORK"
```

`WORK` must be exported in every shell that runs a stage. If `/scratch` does not exist on
cuda2, use another large local directory and tell the user (prefer one not wiped on reboot;
`/tmp` is a last resort).

## 3. Stages

Run in order. Expected output and time are given for each; stop and report if a stage fails.

### 3.1 preflight (seconds)
```bash
scripts/run_cuda2.sh preflight
```
Expect two `NVIDIA GeForce RTX 2080 Ti` rows, the driver version, free space for `$WORK`, the
nvcc version line (12.4) and the host CPU model. Note the driver version and CPU model in the
final report.

### 3.2 setup (10 to 15 min, ~8 GB under $WORK)
```bash
scripts/run_cuda2.sh setup
```
Installs `uv` (to `~/.local/bin`, small), a Python 3.10 env at `$WORK/envs/smolvla` with
`lerobot[smolvla]` (same torch build as the A100 runs: 2.10.0 cu128), and downloads
`lerobot/smolvla_base` plus the SmolVLM2-500M backbone without its ONNX files.

Must end with `available True`, `NVIDIA GeForce RTX 2080 Ti sm (7, 5)` and `fp16 matmul ok`.
If `available False`: the driver is too old for cu128. Run the printed fallback
(`uv pip install torch --index-url https://download.pytorch.org/whl/cu126`), re-check, and
record the torch build actually used (it lands in every meta.json too).

### 3.3 calib (~2 min)
```bash
scripts/run_cuda2.sh calib
```
Builds `bench/` for sm_75 (`make -C bench`), runs `scripts/check_profilers.sh`, then
`scripts/run_calib.sh`, which writes `results/calib/NVIDIA_GeForce_RTX_2080_Ti_cuda2_<stamp>/`.

- ncu needs a private `TMPDIR` on cuda2 (`run_cuda2.sh` sets `TMPDIR=$WORK/tmp`): the lock
  file `/tmp/nsight-compute-lock` belongs to another user, and without it ncu fails with
  `InterprocessLockFailed`. That is not a permission problem. Checked 2026-11-12: with the
  private TMPDIR, counters work for users on cuda2 (no `ERR_NVGPUCTRPERM`). Only a real
  `ERR_NVGPUCTRPERM` needs a CIMS helpdesk email; the other stages do not need ncu.
- Expect `bandwidth_summary.txt` to show DRAM bandwidth near 616 GB/s spec (roughly 500 to
  600 GB/s) and an L2 knee around the 5.5 MB L2. If the GPU was busy, run_calib prints a
  warning; rerun the stage when idle.

### 3.4 dispatch (~1 min)
```bash
scripts/run_cuda2.sh dispatch
```
Writes `torch_dispatch_smolvla_env.csv` (46 lines incl. header) and `host_cpu.txt` into the
calibration directory from 3.3. This is the per-op PyTorch CPU cost table for this host, the
launch term of the model.

### 3.5 trace (~25 to 40 min)
```bash
scripts/run_cuda2.sh trace
```
Runs `workloads/trace_smolvla.py` for the 7 configs of the A100 sweep (cameras 1/2/3, flow
steps 5/10/20, chunk 25/50/100), FP16, batch 1. Per config expect:
- `split-vs-stock max |diff| = 0.000e+00` and `graph-vs-eager denoise max |diff| = 0.000e+00`
  (small nonzero values like 1e-3 are acceptable on a different GPU; report them),
- a `graph profile: {...}` line,
- `wrote traces/smolvla_cam*_steps*_chunk*/NVIDIA_GeForce_RTX_2080_Ti_cuda2_<stamp>`.

Logs are in `logs/cuda2-*.log`. If a config fails, report the traceback from its log.
Out-of-memory is not expected (450M params in FP16 on 11 GB).

Optional, only if time allows and the GPU stays idle: `scripts/run_cuda2.sh trace-baseline`
reruns the baseline config once (~4 min) to measure run-to-run variance on this GPU, as was
done on the A100.

### 3.6 bench (~1 min): standalone CUDA C++ benchmarks (course requirement)
```bash
scripts/run_cuda2.sh bench
```
Builds and runs `scripts/run_bench.sh`, writing `results/bench/NVIDIA_GeForce_RTX_2080_Ti_cuda2_<stamp>/`:
- `kernels.csv`: our kernels (`bench/kernels/kernels.cuh`: WMMA FP16 GEMM with split-K, FP32
  GEMM, attention, softmax, LayerNorm/RMSNorm, RoPE, activations, elementwise) and cuBLAS at
  the traced SmolVLA shapes, eager vs graph. `kernels_check.txt` must end with
  `all correctness checks passed` (every kernel is checked against cuBLAS or a CPU reference).
- `chains.csv`: `bench/chains/chain_replay` replays the traced SmolVLA step as C++ (one flow
  step, the 10-step loop, the full step; our kernels or cuBLAS GEMMs), eager vs graph.
- `perop_flow_{own,cublas}.csv`: every launch of one flow step timed alone.
The kernel lists `bench/chains/smolvla_{flow,full}.csv` come from
`profile/extract_chain.py <trace run> --region flow|full`.
The GPUs are shared: if `busy_warnings.txt` appears, another process ran during a stage;
delete that output directory and rerun when `nvidia-smi` shows no other process.

### 3.7 ncu (~20 min): per-kernel counters (milestone 4)
```bash
scripts/run_cuda2.sh ncu
```
Runs `scripts/run_ncu.sh`: Nsight Compute on the C++ flow-step replay (our kernels with warm
and with flushed L2, and with cuBLAS) and on one eager step of the real SmolVLA
(`profile/ncu_smolvla.py`). Per-kernel feature tables (duration, DRAM and L2 bytes, L2 hit
rate, occupancy, FLOPs, tensor instructions, launch config) go to
`results/ncu/NVIDIA_GeForce_RTX_2080_Ti_cuda2_<stamp>/*.csv.gz`; raw ncu CSVs stay in
`$WORK/ncu_raw`. Needs an idle GPU too (DRAM counters are GPU-wide).

### 3.8 results
```bash
scripts/run_cuda2.sh results
```
Lists the new directories. Expected: one `results/calib/NVIDIA_GeForce_RTX_2080_Ti_*`
directory and 7 (or 8) `traces/smolvla_*/NVIDIA_GeForce_RTX_2080_Ti_*` directories, each with
`meta.json`, `latency.csv`, `ops.csv`, `launch_api_calls.csv`, `kernels.csv.gz` and
`kernels_graph.csv.gz`, plus `results/bench/*2080*` and `results/ncu/*2080*`. Raw
`trace*.json.gz` files are git-ignored by design.

## 4. Send the results back

```bash
git add results/calib traces results/bench results/ncu
git -c user.name="Aditya Maddur Guruprakash" -c user.email="am16489@nyu.edu" \
    commit -m "cuda2 RTX 2080 Ti: calibration, dispatch table, SmolVLA traces"
git push            # needs GitHub credentials on cuda2; see below
```

On cuda2 the clone now pushes over SSH (`git@github.com:axle962/vla-gpu-latency.git`, key
`~/.ssh/id_ed25519` registered on GitHub, 2026-11-12), so `git push` needs no input. For a
fresh HTTPS clone, `git push` asks for a username and a personal access token, which only works
from the user's own terminal (`unset SSH_ASKPASS GIT_ASKPASS` first); Claude must not ask for
or store the token. If pushing is not
possible, make a tarball instead and tell the user to copy it to
`/scratch/am16489/vla-gpu-latency` on the cluster:

```bash
tar czf ~/cuda2_results.tgz results/calib/*2080* traces/smolvla_*/*2080* results/bench/*2080* \
    results/ncu/*2080*
```

## 5. Final report to the user

Summarize in a few lines: host, GPU index used, driver, torch build, host CPU; profiler check
result; DRAM bandwidth and L2 knee; for the baseline config the eager step, eager prefix,
eager denoise, graph denoise and hybrid medians (from its meta.json) and the graph speedup;
anything that failed or was rerun; and whether results were pushed or tarred.

## 6. Other CIMS GPUs

Checked 2026-11-12 (`nvidia-smi` over ssh from cuda2; passwordless once
`~/.ssh/id_ed25519.pub` is in `~/.ssh/authorized_keys`, which the shared home makes valid on
every CIMS host):

| Host | GPU | Notes |
|---|---|---|
| cuda1 | unreachable ("No route to host") | |
| cuda2 | 2x RTX 2080 Ti (Turing sm_75) | source GPU |
| cuda3 | 2x TITAN V (Volta sm_70, tensor cores, HBM2) | /scratch has ~35 GB free |
| cuda4 | 2x GTX TITAN X (Maxwell sm_52, no tensor cores) | C++ calibration only (our WMMA GEMM needs sm_70+) |
| cuda5 | 1x RTX 4070 (Ada sm_89, large L2) | GPU often shared with other users |

All hosts share the home directory, so they see this repo; `scripts/run_cuda2.sh` therefore
uses a per-host build dir (`bench/build-<host>`, CUDA arch from `nvidia-smi`), per-host logs
and per-host result directories. torch 2.10.0+cu128 includes sm_70, so the TITAN V uses the
same torch build as cuda2 and the A100 (setup switches to cu126 only if the arch is missing).

`scripts/run_cims_host.sh full|calib-only` runs the stages unattended and waits before each
GPU stage until a GPU has no other process. Start it detached from cuda2 (the user approved
runs on cuda3, cuda4 and cuda5 on 2026-11-12):

```bash
ssh -n cuda3.cims.nyu.edu 'cd ~/fall2026/GPU/vla-gpu-latency && WORK=/scratch/am16489 \
  setsid nohup scripts/run_cims_host.sh full > logs/cuda3-pipeline.log 2>&1 < /dev/null & disown'
```
(the ssh command may not return by itself; the detached pipeline keeps running, and
`logs/<host>-pipeline.log` is visible from every host). The `/tmp/nsight-compute-lock`
clash exists on cuda3 and cuda5 too; `run_cuda2.sh` sets a private `TMPDIR` everywhere.

## 7. Round 2 (2026-11-13): probes for the cross-GPU model

`model/cross_gpu.py` needs two more target-side measurements on each CIMS GPU that has
traces: host Python speed and cold-cache small PyTorch kernels (`workloads/torch_probe.py`,
stage `probe`, ~1 min, pinned to the GPU's NUMA node when `numactl` exists). It writes
`torch_probe_smolvla_env.csv` into the host's latest `results/calib/` dir. The A100 run is
done (job 30624).

```bash
# on cuda2 (module load cuda-12.4 not needed for this stage)
cd ~/fall2026/GPU/vla-gpu-latency && git pull
WORK=/scratch/am16489 scripts/run_cuda2.sh probe
# cuda3, from cuda2 (waits for an idle GPU):
ssh -n cuda3.cims.nyu.edu 'cd ~/fall2026/GPU/vla-gpu-latency && WORK=/scratch/am16489 \
  setsid nohup scripts/run_cims_host.sh probe-only > logs/cuda3-probe.log 2>&1 < /dev/null & disown'
# cuda5 (RTX 4070), when its GPU is free: the full pipeline, which now includes probe
ssh -n cuda5.cims.nyu.edu 'cd ~/fall2026/GPU/vla-gpu-latency && WORK=/scratch/am16489 \
  setsid nohup scripts/run_cims_host.sh full > logs/cuda5-pipeline.log 2>&1 < /dev/null & disown'
```
Then commit `results/calib/*/torch_probe_smolvla_env.csv` (and the cuda5 results) and push.
On the cluster: rerun `model/cross_gpu.py`, `model/report_cross_gpu.py`, `model/figures.py`.
For cuda5, add an `RTX4070` entry to `GPUS` and `TRACE_KEY` in `model/cross_gpu.py`.
