#!/usr/bin/env bash
# GPU runs on a CIMS cudaN host, in order (written for cuda2, the RTX 2080 Ti source GPU; works
# on any cudaN: build dir, CUDA arch, logs and result dirs are per host, because all CIMS hosts
# share the home directory and therefore this repo). Run each stage once:
#
#   ssh am16489@access.cims.nyu.edu, then ssh cudaN.cims.nyu.edu
#   module load cuda-12.4                      # every login; puts nvcc, ncu, nsys on PATH
#   cd <repo> && git pull
#   export WORK=<dir with >= 15 GB free, NOT your 4 GB home>   # check: df -h /scratch /tmp
#   scripts/run_cuda2.sh preflight   # GPU, driver, disk; picks the idlest GPU
#   scripts/run_cuda2.sh setup       # uv + Python 3.10 env (~6 GB) + SmolVLA weights (~2 GB);
#                                    # switches to the cu126 torch wheel if cu128 lacks the GPU arch
#   scripts/run_cuda2.sh calib       # C++ microbenchmarks + ncu/nsys check (~2 min)
#   scripts/run_cuda2.sh dispatch    # PyTorch per-op dispatch table (~1 min)
#   scripts/run_cuda2.sh trace       # SmolVLA, same 7 configs as the A100 sweep (~25 min)
#   scripts/run_cuda2.sh trace-baseline   # optional: repeat the baseline (variance)
#   scripts/run_cuda2.sh bench       # our CUDA kernels + C++ replay of the traced step (~6 min)
#   scripts/run_cuda2.sh ncu         # Nsight Compute per-kernel counters (~20 min)
#   scripts/run_cuda2.sh results     # list what to commit
# Full instructions (also for a Claude Code session on cuda2): docs/cuda2_runbook.md
#
# Then: git add results/calib traces && git commit -m "cuda2 2080 Ti runs" && git push
# (or tar the new dirs and scp them back). Raw trace*.json.gz files are git-ignored.
# Re-run a stage if the GPU was busy (the scripts warn); timings need an idle GPU.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${WORK:?set WORK to a directory with >= 15 GB free (your home quota is only 4 GB)}"
export UV_CACHE_DIR="$WORK/.cache/uv" UV_PYTHON_INSTALL_DIR="$WORK/.uv-python"
export HF_HOME="$WORK/.cache/huggingface" PATH="$HOME/.local/bin:$PATH"
# ncu's lock file /tmp/nsight-compute-lock belongs to another user on cuda2: use a private TMPDIR.
export TMPDIR="$WORK/tmp"
mkdir -p "$TMPDIR"
ENV="$WORK/envs/smolvla"
PY="$ENV/bin/python"
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  # Idlest GPU by utilization, then memory used.
  CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used \
    --format=csv,noheader,nounits | sort -t, -k2n -k3n | head -1 | cut -d, -f1)
  export CUDA_VISIBLE_DEVICES
fi

HOST=$(hostname -s)
DEV=${CUDA_VISIBLE_DEVICES%%,*}
CC=$(nvidia-smi -i "$DEV" --query-gpu=compute_cap --format=csv,noheader | tr -d '. ')
ARCHS=${ARCHS:-$CC}                 # e.g. 75 on cuda2, 70 on cuda3, 89 on cuda5
BUILD_NAME=${BUILD_NAME:-build-$HOST}
BUILD="bench/$BUILD_NAME"
build() { make -C bench ARCHS="$ARCHS" BUILD="$BUILD_NAME"; }

latest_calib() { ls -dt results/calib/*_"$HOST"_* 2>/dev/null | head -1; }

case "${1:-}" in
preflight)
  nvidia-smi --query-gpu=index,name,driver_version,utilization.gpu,memory.used,memory.total \
    --format=csv
  echo "using GPU $CUDA_VISIBLE_DEVICES (compute capability $CC, build $BUILD)"
  df -h "$WORK" "$HOME" | sed 's/^/  /'
  command -v nvcc >/dev/null && nvcc --version | tail -1 || echo "nvcc missing: module load cuda-12.4"
  grep -m1 "model name" /proc/cpuinfo
  ;;
setup)
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  [[ -x "$PY" ]] || uv venv -p 3.10 "$ENV"
  VIRTUAL_ENV="$ENV" uv pip install "lerobot[smolvla]" pandas
  "$PY" - <<'PY'
from huggingface_hub import snapshot_download
# Skip the SmolVLM repo's ONNX exports (~5 GB), which lerobot does not use.
pat = ["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja", "*.py"]
print(snapshot_download("lerobot/smolvla_base", allow_patterns=pat))
print(snapshot_download("HuggingFaceTB/SmolVLM2-500M-Video-Instruct", allow_patterns=pat))
PY
  # The cu128 wheels of recent torch drop older archs (e.g. Volta sm_70); cu126 still has them.
  if ! "$PY" -c "import torch, sys; c = 'sm_%d%d' % torch.cuda.get_device_capability(0); \
      sys.exit(0 if torch.cuda.is_available() and c in torch.cuda.get_arch_list() else 1)"; then
    tv=$("$PY" -c "import torch; print(torch.__version__.split('+')[0])")
    vv=$("$PY" -c "import torchvision; print(torchvision.__version__.split('+')[0])")
    echo "GPU arch not in this torch build: installing torch $tv / torchvision $vv for cu126"
    VIRTUAL_ENV="$ENV" uv pip install --reinstall "torch==$tv" "torchvision==$vv" \
      --index-url https://download.pytorch.org/whl/cu126
  fi
  "$PY" - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print(torch.cuda.get_device_name(0), "sm", torch.cuda.get_device_capability(0),
      "archs", torch.cuda.get_arch_list())
x = torch.randn(256, 256, device="cuda", dtype=torch.float16)
print("fp16 matmul ok:", float((x @ x).float().abs().mean()))
PY
  ;;
calib)
  command -v nvcc >/dev/null || { echo "run: module load cuda-12.4"; exit 1; }
  build
  BUILD="$BUILD" scripts/check_profilers.sh || echo "PROFILER CHECK FAILED: send the output above to Claude"
  scripts/run_calib.sh "$BUILD"
  ;;
dispatch)
  out=$(latest_calib)
  [[ -n "$out" ]] || { echo "run the calib stage first"; exit 1; }
  grep -m1 "model name" /proc/cpuinfo > "$out/host_cpu.txt"
  CUDA_MODULE_LOADING=EAGER "$PY" workloads/torch_dispatch.py > "$out/torch_dispatch_smolvla_env.csv"
  echo "wrote $out/torch_dispatch_smolvla_env.csv ($(wc -l < "$out/torch_dispatch_smolvla_env.csv") lines)"
  ;;
trace)
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_MODULE_LOADING=EAGER
  mkdir -p logs
  for c in "--cameras 3 --num-steps 10 --chunk-size 50" \
           "--cameras 1 --num-steps 10 --chunk-size 50" \
           "--cameras 2 --num-steps 10 --chunk-size 50" \
           "--cameras 3 --num-steps 5 --chunk-size 50" \
           "--cameras 3 --num-steps 20 --chunk-size 50" \
           "--cameras 3 --num-steps 10 --chunk-size 25" \
           "--cameras 3 --num-steps 10 --chunk-size 100"; do
    log="logs/$HOST-$(echo "$c" | tr -d ' -').log"
    echo "=== $c ($(date +%T)) -> $log"
    # shellcheck disable=SC2086
    "$PY" workloads/trace_smolvla.py $c > "$log" 2>&1 \
      && grep -E "max \|diff\||graph profile|wrote" "$log" \
      || echo "FAILED: $c (see $log)"
  done
  ;;
trace-baseline)
  # One more baseline run (3 cameras, 10 steps, chunk 50) for run-to-run variance.
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_MODULE_LOADING=EAGER
  mkdir -p logs
  log="logs/$HOST-baseline-repeat-$(date +%H%M%S).log"
  "$PY" workloads/trace_smolvla.py > "$log" 2>&1 \
    && grep -E "max \|diff\||graph profile|wrote" "$log" || echo "FAILED (see $log)"
  ;;
bench)
  command -v nvcc >/dev/null || { echo "run: module load cuda-12.4"; exit 1; }
  build
  scripts/run_bench.sh "$BUILD"
  ;;
ncu)
  command -v ncu >/dev/null || { echo "run: module load cuda-12.4"; exit 1; }
  build
  PY="$PY" scripts/run_ncu.sh "$BUILD" all
  ;;
results)
  ls -d results/calib/*_"$HOST"_* traces/smolvla_*/*_"$HOST"_* results/bench/*_"$HOST"_* \
    results/ncu/*_"$HOST"_* 2>/dev/null
  git status --short results traces
  ;;
*)
  sed -n '2,24p' "$0"
  exit 1
  ;;
esac
