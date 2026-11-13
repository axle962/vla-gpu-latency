#!/usr/bin/env bash
# L2 working-set experiment on the C++ replay of the SmolVLA denoising loop
# (bench/chains/chain_replay.cu --weight-pool-mb): the 10-step loop runs eager and as a CUDA
# graph while its GEMM weights share a pool of N MB, from far below to far above this GPU's
# L2. If inter-kernel L2 residency matters, loop time drops once the pool fits in L2.
#   scripts/l2_sweep.sh <build_dir>     (needs an idle GPU; ~2 min)
# Writes results/bench/<gpu>_<host>_<stamp>/l2_sweep.csv (+ device and clock info).
set -euo pipefail
cd "$(dirname "$0")/.."
BUILD=${1:?build dir, e.g. bench/build-cluster or bench/build-$(hostname -s)}
export CUDA_MODULE_LOADING=EAGER
dev=${CUDA_VISIBLE_DEVICES:-0}
dev=${dev%%,*}
gpu=$(nvidia-smi -i "$dev" --query-gpu=name --format=csv,noheader | tr ' ' '_')
out="results/bench/${gpu}_$(hostname -s)_$(date +%Y%m%d-%H%M%S)"
mkdir -p "$out"
echo "writing to $out"
nvidia-smi > "$out/nvidia_smi.txt"
n=$(nvidia-smi -i "$dev" --query-compute-apps=pid --format=csv,noheader | grep -c . || true)
[[ $n -eq 0 ]] || echo "WARNING: $n other process(es) on GPU $dev" | tee "$out/busy_warnings.txt"
"$BUILD/device_info" > "$out/device_info.csv" 2>/dev/null || true
first=1
for g in cublas own; do
  for mb in 0 1 2 4 8 16 24 32 48 64 96 128 256; do
    "$BUILD/chain_replay" --list=bench/chains/smolvla_flow.csv --steps=10 --gemm=$g \
      --reps=20 --weight-pool-mb=$mb 2>> "$out/l2_sweep_log.txt" \
      | if [[ $first == 1 ]]; then cat; else tail -n +2; fi >> "$out/l2_sweep.csv"
    first=0
  done
done
echo "done: $out"
awk -F, 'NR > 1 && $6 == "event" {
  pool = "full"; if (match($4, /weight_pool_mb=[0-9]+/)) pool = substr($4, RSTART + 15, RLENGTH - 15);
  split($4, a, ";"); printf "%-11s %-6s pool %-5s MB  %8.2f ms\n", a[1], $5, pool, $11 / 1000 }' \
  "$out/l2_sweep.csv"
