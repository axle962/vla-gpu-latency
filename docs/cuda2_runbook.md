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

- `check_profilers.sh` failing with `ERR_NVGPUCTRPERM` (or any ncu permission error) is
  important: Nsight Compute counters are blocked for users. Report it prominently; the user
  must email the CIMS helpdesk for GPU performance-counter access. The remaining stages do
  not need ncu, so continue.
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

### 3.6 results
```bash
scripts/run_cuda2.sh results
```
Lists the new directories. Expected: one `results/calib/NVIDIA_GeForce_RTX_2080_Ti_*`
directory and 7 (or 8) `traces/smolvla_*/NVIDIA_GeForce_RTX_2080_Ti_*` directories, each with
`meta.json`, `latency.csv`, `ops.csv`, `launch_api_calls.csv`, `kernels.csv.gz` and
`kernels_graph.csv.gz`. Raw `trace*.json.gz` files are git-ignored by design.

## 4. Send the results back

```bash
git add results/calib traces
git -c user.name="Aditya Maddur Guruprakash" -c user.email="am16489@nyu.edu" \
    commit -m "cuda2 RTX 2080 Ti: calibration, dispatch table, SmolVLA traces"
git push            # needs GitHub credentials on cuda2; see below
```

The clone is HTTPS, so `git push` asks for a username and a GitHub personal access token
(the user provides these; Claude must not ask for or store the token). If pushing is not
possible, make a tarball instead and tell the user to copy it to
`/scratch/am16489/vla-gpu-latency` on the cluster:

```bash
tar czf ~/cuda2_results.tgz results/calib/*2080* traces/smolvla_*/*2080*
```

## 5. Final report to the user

Summarize in a few lines: host, GPU index used, driver, torch build, host CPU; profiler check
result; DRAM bandwidth and L2 knee; for the baseline config the eager step, eager prefix,
eager denoise, graph denoise and hybrid medians (from its meta.json) and the graph speedup;
anything that failed or was rerun; and whether results were pushed or tarred.

## 6. Other CIMS GPUs (optional)

If `cuda1`, `cuda3`, `cuda4` or `cuda5` have a different GPU model (check with `nvidia-smi`
over ssh), the same stages 3.1 to 3.5 there give a free extra GPU that also satisfies the
course's "runs on CIMS" rule. Report the models found; do not run there without asking the
user.
