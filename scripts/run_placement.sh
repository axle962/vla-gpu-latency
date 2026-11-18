#!/usr/bin/env bash
# Hosting placement profile on a CIMS cudaN host (workloads/placement.py): SmolVLA with its two
# halves (prefix, denoising loop) on one GPU, split across two GPUs, split GPU/CPU, and on the
# CPU alone. Needs the envs/smolvla env and weights from `scripts/run_cuda2.sh setup`.
#
#   WORK=/scratch/$USER scripts/run_placement.sh            # all placements, default threads
#   WORK=/scratch/$USER scripts/run_placement.sh threads    # + CPU thread sweep (1 4 8 all)
#   WORK=/scratch/$USER scripts/run_placement.sh smoke      # toy model, 1 minute, no weights
#
# GPU-GPU needs two idle GPUs (cuda2: 2x RTX 2080 Ti, cuda3: 2x TITAN V). With one GPU (cuda5)
# that placement is skipped. The script refuses to start while another process is on a GPU it
# would use (CIMS GPUs are shared; set FORCE=1 to run anyway, the timings are then suspect).
# Everything is pinned to the first GPU's NUMA node (cuda2 has a slow mode when unpinned).
# Time: ~10 min for the default run (the CPU halves dominate), ~30 min with the thread sweep.
# Results: results/placement/<gpu>_<host>_<time>[_t<threads>]/ with report.md in each.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${WORK:?set WORK to the directory that holds envs/smolvla (not your 4 GB home)}"
MODE=${1:-run}
PY="$WORK/envs/smolvla/bin/python"
export HF_HOME="$WORK/.cache/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_MODULE_LOADING=EAGER
# A shared $HF_HOME (symlinked to a teammate's cache) holds their private token file, which
# huggingface_hub tries to read even offline; point it at a file of our own instead.
if [[ -e $HF_HOME/token && ! -r $HF_HOME/token ]]; then
  export HF_TOKEN_PATH="$WORK/.hf_token_unused"
fi
[[ -x "$PY" ]] || { echo "no $PY: run scripts/run_cuda2.sh setup first"; exit 1; }

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index --format=csv,noheader | head -2 | paste -sd,)
  export CUDA_VISIBLE_DEVICES
fi
for g in ${CUDA_VISIBLE_DEVICES//,/ }; do
  n=$(nvidia-smi -i "$g" --query-compute-apps=pid --format=csv,noheader | grep -c . || true)
  if [[ $n -gt 0 && -z "${FORCE:-}" ]]; then
    echo "GPU $g has $n other process(es); wait for it to be idle or set FORCE=1"
    nvidia-smi -i "$g" --query-compute-apps=pid,used_memory --format=csv
    exit 1
  fi
done

DEV=${CUDA_VISIBLE_DEVICES%%,*}
pin=()
if command -v numactl >/dev/null; then
  node=$(cat /sys/bus/pci/devices/0000:$(nvidia-smi -i "$DEV" --query-gpu=pci.bus_id \
    --format=csv,noheader | cut -d: -f2-3 | tr 'A-F' 'a-f')/numa_node 2>/dev/null || echo -1)
  [[ $node -ge 0 ]] && pin=(numactl --cpunodebind="$node" --membind="$node")
fi
HOST=$(hostname -s)
GPU=$(nvidia-smi -i "$DEV" --query-gpu=name --format=csv,noheader | tr ' ' '_')
STAMP=$(date +%Y%m%d-%H%M%S)
BASE="results/placement/${GPU}_${HOST}_${STAMP}"
echo "GPUs $CUDA_VISIBLE_DEVICES on $HOST ${pin[*]:-(unpinned)}"

one() {  # out_dir, then extra args
  local out=$1; shift
  mkdir -p "$out"
  nvidia-smi --query-gpu=index,name,pci.bus_id,utilization.gpu,memory.used --format=csv \
    > "$out/gpus.csv"
  # Peer-to-peer capability between the two GPUs (decides whether GPU->GPU copies bounce
  # through host memory); not available on every driver.
  nvidia-smi topo -p2p r > "$out/p2p.txt" 2>&1 || true
  "${pin[@]}" "$PY" workloads/placement.py --out "$out" "$@" 2>&1 | tee "$out/run.log"
  "$PY" model/placement_report.py "$out" > /dev/null
  echo "report: $out/report.md"
}

case "$MODE" in
smoke)
  one "${TMPDIR:-/tmp}/placement_smoke_$STAMP" --model toy --reps 5 --reps-cpu 3
  ;;
run)
  one "$BASE" --profile
  ;;
threads)
  one "$BASE" --profile
  # CPU halves depend on the thread count. gpu-gpu is not repeated; gpu stays as the
  # baseline the report needs.
  all=$("${pin[@]}" nproc)
  for t in 1 4 8 "$all"; do
    one "${BASE}_t$t" --placements gpu,gpu-cpu,cpu-gpu,cpu --cpu-threads "$t" --no-graph
  done
  ;;
*)
  echo "usage: WORK=... $0 [run|threads|smoke]"; exit 1
  ;;
esac
