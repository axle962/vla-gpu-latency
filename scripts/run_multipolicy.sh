#!/usr/bin/env bash
# Multi-policy hosting on one GPU (workloads/multipolicy.py, pending direction 6), on a CIMS
# cudaN host or inside a Slurm job. Needs the envs/smolvla env and weights from
# `scripts/run_cuda2.sh setup` (plus `more` for the ACT and Diffusion weights).
#
#   WORK=/scratch/$USER scripts/run_multipolicy.sh smoke   # toy policy, GPU, ~1 min, no weights
#   WORK=/scratch/$USER scripts/run_multipolicy.sh quick   # SmolVLA + ACT, N = 1 2 4, ~30 min
#   WORK=/scratch/$USER scripts/run_multipolicy.sh full    # + Diffusion, N = 8, loop mode, ~80 min
#   DUR=10 MODES=eager scripts/run_multipolicy.sh custom "procs smolvla,smolvla" "threads smolvla,act"
#
# Hosting modes: procs (one process per policy), procs-mps (same, under an MPS daemon with a
# private pipe directory under $WORK/mps; skipped if nvidia-cuda-mps-control is missing or the
# daemon does not start), threads (one process, a thread and a stream per policy), batch (one
# policy at batch N). The script refuses to start while another process is on the GPU (CIMS
# GPUs are shared; FORCE=1 overrides, timings are then suspect) and pins everything to the
# GPU's NUMA node when numactl exists.
# Results: results/multipolicy/<gpu>_<host>_<stamp>/ with report.md (model/multipolicy_report.py).
set -euo pipefail
cd "$(dirname "$0")/.."
: "${WORK:?set WORK to the directory that holds envs/smolvla (not a 4 GB home)}"
MODE=${1:-quick}
PY=${PY:-$WORK/envs/smolvla/bin/python}
DUR=${DUR:-15}
MODES=${MODES:-eager,graph}
export HF_HOME="${HF_HOME:-$WORK/.cache/huggingface}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCH_HOME="${TORCH_HOME:-$WORK/.cache/torch}" CUDA_MODULE_LOADING=EAGER
[[ -x "$PY" ]] || { echo "no $PY: run scripts/run_cuda2.sh setup first"; exit 1; }

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used \
    --format=csv,noheader,nounits | sort -t, -k2n -k3n | head -1 | cut -d, -f1)
  export CUDA_VISIBLE_DEVICES
fi
DEV=${CUDA_VISIBLE_DEVICES%%,*}
n=$(nvidia-smi -i "$DEV" --query-compute-apps=pid --format=csv,noheader | grep -c . || true)
if [[ $n -gt 0 && -z "${FORCE:-}" ]]; then
  echo "GPU $DEV has $n other process(es); wait for it to be idle or set FORCE=1"
  nvidia-smi -i "$DEV" --query-compute-apps=pid,used_memory --format=csv
  exit 1
fi
pin=()
if command -v numactl >/dev/null; then
  node=$(cat /sys/bus/pci/devices/0000:$(nvidia-smi -i "$DEV" --query-gpu=pci.bus_id \
    --format=csv,noheader | cut -d: -f2-3 | tr 'A-F' 'a-f')/numa_node 2>/dev/null || echo -1)
  [[ $node -ge 0 ]] && pin=(numactl --cpunodebind="$node" --membind="$node")
  # Inside a Slurm cgroup the allocated CPUs may not be on the GPU's node: then do not pin.
  if [[ ${#pin[@]} -gt 0 ]] && ! "${pin[@]}" true 2>/dev/null; then pin=(); fi
fi
HOST=$(hostname -s)
GPU=$(nvidia-smi -i "$DEV" --query-gpu=name --format=csv,noheader | tr ' ' '_')
STAMP=$(date +%Y%m%d-%H%M%S)
OUT=${OUT:-results/multipolicy/${GPU}_${HOST}_${STAMP}}
mkdir -p "$OUT" logs
echo "GPU $DEV ($GPU) on $HOST ${pin[*]:-(unpinned)} -> $OUT, $DUR s per mode, modes $MODES"
nvidia-smi --query-gpu=index,name,pci.bus_id,utilization.gpu,memory.used --format=csv > "$OUT/gpus.csv"

MPS_OK=0
mps_start() {
  command -v nvidia-cuda-mps-control >/dev/null || { echo "no nvidia-cuda-mps-control: skipping procs-mps"; return 1; }
  export CUDA_MPS_PIPE_DIRECTORY="$WORK/mps/pipe-$HOST-$DEV" CUDA_MPS_LOG_DIRECTORY="$WORK/mps/log-$HOST-$DEV"
  rm -rf "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
  mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
  if nvidia-cuda-mps-control -d; then
    sleep 1
    echo "MPS daemon started (pipe $CUDA_MPS_PIPE_DIRECTORY)"
    MPS_OK=1
  else
    echo "MPS daemon failed to start: skipping procs-mps"; unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY; return 1
  fi
}
mps_stop() {
  [[ $MPS_OK -eq 1 ]] || return 0
  echo quit | nvidia-cuda-mps-control || true
  sleep 1
  unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY
  MPS_OK=0
  echo "MPS daemon stopped"
}
trap mps_stop EXIT

cfg() {  # hosting policies [extra args]
  local hosting=$1 pols=$2; shift 2
  local extra=()
  [[ $pols != *,* ]] && extra+=(--profile)
  if [[ $hosting == procs-mps ]]; then
    [[ $MPS_OK -eq 1 ]] || mps_start || return 0
  else
    mps_stop
  fi
  "${pin[@]}" "$PY" workloads/multipolicy.py host --out "$OUT" --hosting "$hosting" \
    --policies "$pols" --modes "$MODES" --duration "$DUR" "${extra[@]}" "$@" \
    2>&1 | tee -a "$OUT/run.log" || echo "FAILED: $hosting $pols" | tee -a "$OUT/run.log"
}
rep() { for i in $(seq 1 "$2"); do printf "%s," "$1"; done | sed 's/,$//'; }

case "$MODE" in
smoke)
  DUR=3
  cfg procs toy --warm-s 1
  cfg procs toy,toy --warm-s 1
  cfg procs-mps toy,toy --warm-s 1
  cfg threads toy,toy --warm-s 1
  cfg batch toy,toy --warm-s 1
  ;;
quick)
  for p in smolvla act; do
    cfg procs "$p"
    for n in 2 4; do cfg procs "$(rep $p $n)"; done
    for n in 2 4; do cfg procs-mps "$(rep $p $n)"; done
    for n in 2 4; do cfg threads "$(rep $p $n)"; done
  done
  cfg batch "$(rep smolvla 4)"
  cfg procs smolvla,act
  cfg procs-mps smolvla,act
  cfg threads smolvla,act
  ;;
full)
  MODES=${MODES_FULL:-eager,graph,loop}
  for p in smolvla act diffusion; do
    cfg procs "$p"
    for n in 2 4 8; do cfg procs "$(rep $p $n)"; done
    for n in 2 4 8; do cfg procs-mps "$(rep $p $n)"; done
    for n in 2 4 8; do cfg threads "$(rep $p $n)"; done
  done
  for n in 4 8; do cfg batch "$(rep smolvla $n)"; done
  for mix in smolvla,act smolvla,diffusion act,diffusion smolvla,act,diffusion \
             smolvla,smolvla,act,act; do
    cfg procs "$mix"
    cfg procs-mps "$mix"
    cfg threads "$mix"
  done
  ;;
custom)
  shift
  for spec in "$@"; do
    # shellcheck disable=SC2086
    cfg $spec
  done
  ;;
*)
  echo "usage: WORK=... $0 [smoke|quick|full|custom \"hosting policies\" ...]"; exit 1
  ;;
esac
mps_stop
"$PY" model/multipolicy_report.py "$OUT" > /dev/null && echo "report: $OUT/report.md"
