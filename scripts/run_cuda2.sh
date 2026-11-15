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
#   scripts/run_cuda2.sh probe       # host Python speed + cold-cache small kernels (~2 min)
#   scripts/run_cuda2.sh trace       # SmolVLA, same 7 configs as the A100 sweep (~25 min)
#   scripts/run_cuda2.sh trace-baseline   # optional: repeat the baseline (variance)
#   scripts/run_cuda2.sh bench       # our CUDA kernels + C++ replay of the traced step (~6 min)
#   scripts/run_cuda2.sh l2          # L2 working-set sweep on the C++ replay (~2 min)
#   scripts/run_cuda2.sh ncu         # Nsight Compute per-kernel counters (~20 min)
#   scripts/run_cuda2.sh more        # extra workloads: Diffusion Policy + ACT sweep, conv
#                                    # dispatch rows (~45 min; downloads ~1.3 GB of weights)
#   scripts/run_cuda2.sh diffbench   # Diffusion Policy sweep with cudnn.benchmark=True (~25 min)
#   scripts/run_cuda2.sh extras      # SmolVLA batch sweep, torch.compile, GPU energy (~1.5 h)
#   scripts/run_cuda2.sh rodinia     # Rodinia per-kernel times under nsys (scripts/rodinia.sh)
#   scripts/run_cuda2.sh pi          # pi0, pi0.5, pi0-FAST traces (envs/pi, logs/setup_pi_env.sh;
#                                    # needs a Hugging Face login with the PaliGemma license)
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
  # Use the cu128 wheel (same build as every other GPU) unless it has no binary for this GPU.
  # A binary for sm_XY runs on sm_XZ with Z >= Y (e.g. cu128 has sm_86, which runs on the
  # RTX 4070's sm_89), so test compatibility by major version, not by exact name.
  if ! "$PY" -c "import torch, sys; M, m = torch.cuda.get_device_capability(0); \
      arch = [a[3:] for a in torch.cuda.get_arch_list() if a.startswith('sm_')]; \
      ok = any(int(a[:-1]) == M and int(a[-1]) <= m for a in arch if a.isdigit()); \
      sys.exit(0 if torch.cuda.is_available() and ok else 1)"; then
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
probe)
  # Target-side probes for model/cross_gpu.py (workloads/torch_probe.py). Pinned to the GPUs'
  # NUMA node when numactl exists (cuda2 has an OS-scheduling slow mode on unpinned runs).
  out=$(latest_calib)
  [[ -n "$out" ]] || { echo "run the calib stage first"; exit 1; }
  grep -m1 "model name" /proc/cpuinfo > "$out/host_cpu.txt"
  pin=()
  if command -v numactl >/dev/null; then
    node=$(cat /sys/bus/pci/devices/0000:$(nvidia-smi -i "$DEV" --query-gpu=pci.bus_id \
      --format=csv,noheader | cut -d: -f2-3 | tr 'A-F' 'a-f')/numa_node 2>/dev/null || echo -1)
    [[ $node -ge 0 ]] && pin=(numactl --cpunodebind="$node" --membind="$node")
  fi
  echo "probe on GPU $DEV ${pin[*]:-(unpinned)}"
  CUDA_MODULE_LOADING=EAGER "${pin[@]}" "$PY" workloads/torch_probe.py > "$out/torch_probe_smolvla_env.csv"
  echo "wrote $out/torch_probe_smolvla_env.csv ($(wc -l < "$out/torch_probe_smolvla_env.csv") lines)"
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
more|diffbench)
  # Extra workloads (workloads/trace_diffusion.py, trace_act.py; same configs as
  # scripts/cluster/trace_more.sbatch) plus the conv rows of the dispatch table. Pinned to the
  # GPU's NUMA node when numactl exists.
  out=$(latest_calib)
  [[ -n "$out" ]] || { echo "run the calib stage first"; exit 1; }
  export TORCH_HOME="$WORK/.cache/torch" CUDA_MODULE_LOADING=EAGER
  ck="$TORCH_HOME/hub/checkpoints/resnet18-f37072fd.pth"
  if [[ ! -s $ck ]]; then
    mkdir -p "$(dirname "$ck")"
    curl -sSL -o "$ck" https://download.pytorch.org/models/resnet18-f37072fd.pth
  fi
  "$PY" -c 'from huggingface_hub import snapshot_download as d
for r in ("lerobot/diffusion_pusht", "lerobot/act_aloha_sim_transfer_cube_human"): print(d(r))'
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  pin=()
  if command -v numactl >/dev/null; then
    node=$(cat /sys/bus/pci/devices/0000:$(nvidia-smi -i "$DEV" --query-gpu=pci.bus_id \
      --format=csv,noheader | cut -d: -f2-3 | tr 'A-F' 'a-f')/numa_node 2>/dev/null || echo -1)
    [[ $node -ge 0 ]] && pin=(numactl --cpunodebind="$node" --membind="$node")
  fi
  echo "extra workloads on GPU $DEV ${pin[*]:-(unpinned)}"
  if [[ $1 == diffbench ]]; then
    runs=("trace_diffusion.py --cudnn-benchmark --num-steps 100"
          "trace_diffusion.py --cudnn-benchmark --num-steps 10"
          "trace_diffusion.py --cudnn-benchmark --num-steps 25"
          "trace_diffusion.py --cudnn-benchmark --num-steps 50"
          "trace_diffusion.py --cudnn-benchmark --num-steps 100")
  else
    "${pin[@]}" "$PY" workloads/torch_dispatch.py --set conv > "$out/torch_dispatch_conv_smolvla_env.csv" \
      && echo "wrote $out/torch_dispatch_conv_smolvla_env.csv" || echo "FAILED: conv dispatch"
    runs=("trace_diffusion.py --num-steps 100" "trace_diffusion.py --num-steps 10"
          "trace_diffusion.py --num-steps 25" "trace_diffusion.py --num-steps 50"
          "trace_diffusion.py --num-steps 100" "trace_act.py --height 480 --width 640"
          "trace_act.py --height 240 --width 320" "trace_act.py --height 480 --width 640")
  fi
  mkdir -p logs
  for a in "${runs[@]}"; do
    log="logs/$HOST-$(echo "$a" | tr -d ' -' | sed 's/\.py//')-$(date +%H%M%S).log"
    echo "=== $a ($(date +%T)) -> $log"
    # shellcheck disable=SC2086
    "${pin[@]}" "$PY" workloads/$a > "$log" 2>&1 \
      && grep -E "max \|diff\||graph profile|wrote" "$log" || echo "FAILED: $a (see $log)"
  done
  ;;
extras)
  # SmolVLA batch sweep (traces/smolvlabatch_*), torch.compile of the denoising loop
  # (traces/smolvlacompile_*, workloads/trace_smolvla_compile.py) and GPU energy per inference
  # (results/energy/, workloads/energy_smolvla.py). Pinned to the GPU's NUMA node.
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_MODULE_LOADING=EAGER
  export TORCHINDUCTOR_CACHE_DIR="$WORK/.cache/inductor" TRITON_CACHE_DIR="$WORK/.cache/triton"
  pin=()
  if command -v numactl >/dev/null; then
    node=$(cat /sys/bus/pci/devices/0000:$(nvidia-smi -i "$DEV" --query-gpu=pci.bus_id \
      --format=csv,noheader | cut -d: -f2-3 | tr 'A-F' 'a-f')/numa_node 2>/dev/null || echo -1)
    [[ $node -ge 0 ]] && pin=(numactl --cpunodebind="$node" --membind="$node")
  fi
  echo "extras on GPU $DEV ${pin[*]:-(unpinned)}"
  mkdir -p logs results/energy
  for a in "trace_smolvla.py --batch 2" "trace_smolvla.py --batch 4" "trace_smolvla.py --batch 8" \
           "trace_smolvla.py --batch 16" "trace_smolvla_compile.py" "trace_smolvla_compile.py" \
           "trace_smolvla_compile.py --batch 8"; do
    log="logs/$HOST-$(echo "$a" | tr -d ' -' | sed 's/\.py//')-$(date +%H%M%S).log"
    echo "=== $a ($(date +%T)) -> $log"
    # shellcheck disable=SC2086
    "${pin[@]}" "$PY" workloads/$a > "$log" 2>&1 \
      && grep -E "max \|diff\||compile\+warmup|graph profile|wrote" "$log" || echo "FAILED: $a (see $log)"
  done
  gname=$(nvidia-smi -i "$DEV" --query-gpu=name --format=csv,noheader | tr ' ' _)
  for b in 1 8; do
    e="results/energy/${gname}_${HOST}_b${b}_$(date +%Y%m%d-%H%M%S).csv"
    echo "=== energy batch $b ($(date +%T)) -> $e"
    "${pin[@]}" "$PY" workloads/energy_smolvla.py --batch $b > "$e" 2> "logs/$HOST-energy-b$b.log" \
      && column -s, -t < "$e" || echo "FAILED: energy batch $b (see logs/$HOST-energy-b$b.log)"
  done
  ;;
rodinia)
  WORK="$WORK" scripts/rodinia.sh
  ;;
pi)
  # pi0 / pi0.5 (workloads/trace_pi.py) and pi0-FAST (workloads/trace_pi0fast.py), LIBERO
  # checkpoints, FP16 on every GPU (Turing/Volta have no BF16) plus BF16 where supported (the
  # numeric reference). On hosts with < 40 GB free in $WORK each model is fetched before its runs
  # and removed after. Pinned to the GPU's NUMA node.
  PYPI="$WORK/envs/pi/bin/python"
  [[ -x $PYPI ]] || { echo "build envs/pi first: logs/setup_pi_env.sh"; exit 1; }
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_MODULE_LOADING=EAGER
  pin=()
  if command -v numactl >/dev/null; then
    node=$(cat /sys/bus/pci/devices/0000:$(nvidia-smi -i "$DEV" --query-gpu=pci.bus_id \
      --format=csv,noheader | cut -d: -f2-3 | tr 'A-F' 'a-f')/numa_node 2>/dev/null || echo -1)
    [[ $node -ge 0 ]] && pin=(numactl --cpunodebind="$node" --membind="$node")
  fi
  precs="fp16"
  [[ $CC -ge 80 ]] && precs="fp16 bf16"
  free_gb=$(df -BG --output=avail "$WORK" | tail -1 | tr -dc 0-9)
  echo "pi models on GPU $DEV ${pin[*]:-(unpinned)}, precisions: $precs, $free_gb GB free"
  mkdir -p logs
  run_model() {  # repo, then trace commands
    local repo=$1; shift
    HF_HUB_OFFLINE=0 "$PYPI" -c "from huggingface_hub import snapshot_download as d; d('$repo')" \
      > /dev/null 2>&1 || { echo "FAILED: download $repo"; return; }
    local a log
    for a in "$@"; do
      log="logs/$HOST-$(echo "$a" | tr -d ' -' | sed 's/\.py//')-$(date +%H%M%S).log"
      echo "=== $a ($(date +%T)) -> $log"
      # shellcheck disable=SC2086
      "${pin[@]}" "$PYPI" workloads/$a > "$log" 2>&1 \
        && grep -E "max \|diff\||tokens|graph profile|wrote" "$log" || echo "FAILED: $a (see $log)"
    done
    if [[ $free_gb -lt 40 ]]; then
      rm -rf "$HF_HOME/hub/models--${repo//\//--}" && echo "removed $repo from the cache"
    fi
  }
  for p in $precs; do
    run_model lerobot/pi0_libero_finetuned_v044 "trace_pi.py --policy pi0 --precision $p"
    run_model lerobot/pi05_libero_finetuned_v044 "trace_pi.py --policy pi05 --precision $p"
    run_model lerobot/pi0fast-libero-v044 "trace_pi0fast.py --precision $p"
  done
  run_model lerobot/pi0_libero_finetuned_v044 "trace_pi.py --policy pi0 --precision fp16"
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
l2)
  command -v nvcc >/dev/null || { echo "run: module load cuda-12.4"; exit 1; }
  build
  scripts/l2_sweep.sh "$BUILD"
  ;;
ncu)
  command -v ncu >/dev/null || { echo "run: module load cuda-12.4"; exit 1; }
  build
  PY="$PY" scripts/run_ncu.sh "$BUILD" all
  ;;
results)
  ls -d results/calib/*_"$HOST"_* traces/*/*_"$HOST"_* results/bench/*_"$HOST"_* \
    results/ncu/*_"$HOST"_* 2>/dev/null
  git status --short results traces
  ;;
*)
  sed -n '2,31p' "$0"
  exit 1
  ;;
esac
