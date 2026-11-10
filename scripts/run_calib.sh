#!/usr/bin/env bash
# Run every calibration microbenchmark on the current GPU and write CSVs to
# results/calib/<gpu>_<host>_<timestamp>/. Works on CIMS cuda2 and inside a Slurm job.
#   scripts/run_calib.sh [build_dir]   (default: bench/build)
set -euo pipefail
cd "$(dirname "$0")/.."
BUILD="${1:-bench/build}"
GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"

name=$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 | head -1 | tr ' ' '_' | tr -cd '[:alnum:]_-')
out="results/calib/${name}_$(hostname -s)_$(date +%Y%m%d-%H%M%S)"
mkdir -p "$out"
echo "writing to $out (CUDA_VISIBLE_DEVICES=$GPU_ID)"

{
  nvidia-smi
  nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,clocks.max.mem,power.limit,utilization.gpu,memory.used --format=csv
} > "$out/nvidia_smi.txt" 2>&1
if [[ $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 0 | head -1) -gt 5 ]]; then
  echo "WARNING: GPU is busy; timings will be noisy" | tee -a "$out/nvidia_smi.txt"
fi

"$BUILD/device_info" > "$out/device_info.csv"
# Launch overhead under both module-loading modes (the lazy/eager difference is itself a
# finding about first-launch cost). Warm-up runs make steady-state rows comparable.
CUDA_MODULE_LOADING=EAGER "$BUILD/launch_latency" > "$out/launch_eager_loading.csv"
CUDA_MODULE_LOADING=LAZY "$BUILD/launch_latency" > "$out/launch_lazy_loading.csv"
CUDA_MODULE_LOADING=EAGER "$BUILD/bandwidth" > "$out/bandwidth.csv" 2> "$out/bandwidth_summary.txt"
cat "$out/bandwidth_summary.txt"
echo "done: $out"
