#!/usr/bin/env bash
# Run the scripts/run_cuda2.sh stages on this CIMS cudaN host, unattended, waiting before each
# GPU stage until the chosen GPU has no other process (CIMS GPUs are shared with other users).
#   WORK=/scratch/$USER scripts/run_cims_host.sh [full|calib-only]
#     full        preflight setup calib dispatch probe bench trace trace-baseline
#     probe-only  preflight probe (needs an earlier calib run on this host)
#     l2-only     preflight l2 (L2 working-set sweep; builds the C++ benchmarks)
#     calib-only  preflight calib (C++ microbenchmarks only, e.g. for GPUs without tensor cores)
# Start it detached so it survives the SSH session, for example from cuda2:
#   ssh cuda3.cims.nyu.edu 'cd ~/fall2026/GPU/vla-gpu-latency && WORK=/scratch/$USER \
#     setsid nohup scripts/run_cims_host.sh full > logs/cuda3-pipeline.log 2>&1 < /dev/null &'
# Progress: logs/<host>-pipeline.log (the home directory, and so this log, is shared).
set -uo pipefail
cd "$(dirname "$0")/.."
: "${WORK:?set WORK to a local directory with >= 20 GB free}"
MODE=${1:-full}
source /etc/profile.d/modules.sh 2>/dev/null
module load cuda-12.4 2>/dev/null
mkdir -p "$WORK" logs
MAX_WAIT_S=${MAX_WAIT_S:-14400}

busy() {  # number of compute processes on GPU $1
  nvidia-smi -i "$1" --query-compute-apps=pid --format=csv,noheader | grep -c .
}
pick_gpu() {  # an idle GPU index, or nothing
  local i
  for i in $(nvidia-smi --query-gpu=index --format=csv,noheader); do
    if [[ $(busy "$i") -eq 0 ]]; then echo "$i"; return; fi
  done
}
wait_idle() {  # sets CUDA_VISIBLE_DEVICES to a GPU that has been idle for 60 s
  local waited=0 g
  while :; do
    g=$(pick_gpu)
    if [[ -n $g ]]; then
      sleep 60
      if [[ $(busy "$g") -eq 0 ]]; then
        export CUDA_VISIBLE_DEVICES=$g
        echo "[$(date +%T)] GPU $g idle, starting $1"
        return 0
      fi
    else
      sleep 60
    fi
    waited=$((waited + 60))
    if (( waited % 600 == 0 )); then
      echo "[$(date +%T)] waiting for an idle GPU before $1 (${waited}s)"
      nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
    fi
    if (( waited >= MAX_WAIT_S )); then
      echo "[$(date +%T)] gave up waiting for an idle GPU before $1"
      return 1
    fi
  done
}
stage() {  # name [gpu]
  local s=$1
  if [[ ${2:-} == gpu ]]; then wait_idle "$s" || exit 1; fi
  echo "[$(date +%T)] === stage $s on $(hostname -s) (GPU ${CUDA_VISIBLE_DEVICES:-?})"
  if ! scripts/run_cuda2.sh "$s"; then
    echo "[$(date +%T)] stage $s FAILED"
    exit 1
  fi
}

echo "[$(date +%T)] $(hostname -s): mode $MODE, WORK=$WORK"
export CUDA_VISIBLE_DEVICES=$(pick_gpu)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
stage preflight
if [[ $MODE == calib-only ]]; then
  stage calib gpu
elif [[ $MODE == probe-only ]]; then
  stage probe gpu
elif [[ $MODE == l2-only ]]; then
  stage l2 gpu
else
  stage setup
  stage calib gpu
  stage dispatch gpu
  stage probe gpu
  stage bench gpu
  stage l2 gpu
  stage trace gpu
  stage trace-baseline gpu
fi
echo "[$(date +%T)] all stages done on $(hostname -s)"
scripts/run_cuda2.sh results
