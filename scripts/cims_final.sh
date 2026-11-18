#!/usr/bin/env bash
# Final CIMS jobs (CLAUDE.md "Run list 2026-11-17"), run unattended on a cudaN host:
#   trace   SmolVLA RoboTwin checkpoint (lerobot/smolvla_robotwin) at the baseline shapes,
#           pinned to the GPU's NUMA node: same architecture as smolvla_base, so its timings
#           should match the base traces on the same GPU.
#   repro   cuda2 only: the course reproducibility check from a fresh GitHub clone
#           (module load cuda-12.4, make -C bench, scripts/run_calib.sh, scripts/run_bench.sh).
#           The clone lives in $WORK/repro; its result directories are copied back here.
# Each mode waits until a GPU has had no compute process for 60 s.
#   WORK=/scratch/$USER scripts/cims_final.sh trace [repro]
set -uo pipefail
cd "$(dirname "$0")/.."
: "${WORK:?set WORK}"
REPO=$PWD
HOST=$(hostname -s)
source /etc/profile.d/modules.sh 2>/dev/null
module load cuda-12.4 2>/dev/null

idle_gpu() {  # wait for a GPU that stays idle for 60 s, print its index
  local g
  while :; do
    for g in $(nvidia-smi --query-gpu=index --format=csv,noheader); do
      if [[ $(nvidia-smi -i "$g" --query-compute-apps=pid --format=csv,noheader | grep -c .) -eq 0 ]]; then
        sleep 60
        if [[ $(nvidia-smi -i "$g" --query-compute-apps=pid --format=csv,noheader | grep -c .) -eq 0 ]]; then
          echo "$g"; return
        fi
      fi
    done
    sleep 60
  done
}
pin_for() {  # numactl prefix for GPU $1
  local node
  node=$(cat /sys/bus/pci/devices/0000:$(nvidia-smi -i "$1" --query-gpu=pci.bus_id \
    --format=csv,noheader | cut -d: -f2-3 | tr 'A-F' 'a-f')/numa_node 2>/dev/null || echo -1)
  command -v numactl > /dev/null && [[ $node -ge 0 ]] && echo "numactl --cpunodebind=$node --membind=$node"
}

for mode in "$@"; do
  gpu=$(idle_gpu)
  pin=$(pin_for "$gpu")
  echo "[$(date +%T)] $HOST: $mode on GPU $gpu ${pin:-(unpinned)}"
  case $mode in
  trace)
    PY="$WORK/envs/smolvla/bin/python"
    HF_HOME="$WORK/.cache/huggingface" "$PY" -c \
      "from huggingface_hub import snapshot_download as d; d('lerobot/smolvla_robotwin')" > /dev/null 2>&1
    log="logs/$HOST-trace-smolvla-robotwin-$(date +%H%M%S).log"
    HF_HOME="$WORK/.cache/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      CUDA_MODULE_LOADING=EAGER CUDA_VISIBLE_DEVICES=$gpu $pin \
      "$PY" workloads/trace_smolvla.py --model lerobot/smolvla_robotwin > "$log" 2>&1 \
      && grep -E "max \|diff\||graph profile|wrote" "$log" || echo "FAILED: trace (see $log)"
    ;;
  repro)
    rm -rf "$WORK/repro" && mkdir -p "$WORK/repro" && cd "$WORK/repro" \
      && git clone -q git@github.com:axle962/vla-gpu-latency.git && cd vla-gpu-latency \
      || { echo "FAILED: clone"; cd "$REPO"; continue; }
    echo "fresh clone at $(git log --oneline -1)"
    nvcc --version | tail -1
    { make -C bench > "$REPO/logs/$HOST-repro-make.log" 2>&1 && echo "make -C bench: ok"; } \
      || { echo "FAILED: make (see logs/$HOST-repro-make.log)"; cd "$REPO"; continue; }
    CUDA_VISIBLE_DEVICES=$gpu $pin scripts/run_calib.sh > "$REPO/logs/$HOST-repro-calib.log" 2>&1 \
      && echo "run_calib.sh: ok" || echo "FAILED: run_calib.sh"
    CUDA_VISIBLE_DEVICES=$gpu $pin scripts/run_bench.sh > "$REPO/logs/$HOST-repro-bench.log" 2>&1 \
      && echo "run_bench.sh: ok" || echo "FAILED: run_bench.sh"
    grep -ciE "fail|mismatch" "$REPO/logs/$HOST-repro-bench.log" | sed 's/^/bench lines with fail|mismatch: /'
    for d in results/calib/*_"$HOST"_* results/bench/*_"$HOST"_*; do
      [[ -e "$REPO/$d" ]] || { cp -r "$d" "$REPO/$d" && echo "copied $d"; }
    done
    cd "$REPO"
    ;;
  *) echo "unknown mode $mode" ;;
  esac
done
echo "[$(date +%T)] $HOST: cims_final done"
