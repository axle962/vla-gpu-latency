#!/usr/bin/env bash
# Standalone CUDA C++ benchmarks (bench/kernels, bench/chains) on one GPU:
#   kernels_bench  our kernels + cuBLAS at the traced SmolVLA shapes, eager vs graph
#   chain_replay   the traced SmolVLA flow step (1 and 10 steps) and full step, rebuilt in
#                  C++ (our kernels, or cuBLAS GEMMs), eager vs graph, plus per-launch times
# Writes results/bench/<gpu>_<host>_<stamp>/. Usage: scripts/run_bench.sh [build_dir]
# Needs an idle GPU (pick one with CUDA_VISIBLE_DEVICES). ~6 min on the RTX 2080 Ti.
set -euo pipefail
cd "$(dirname "$0")/.."
BUILD=${1:-bench/build}
for b in kernels_bench chain_replay; do
  [[ -x $BUILD/$b ]] || { echo "missing $BUILD/$b: run make -C bench"; exit 1; }
done
export CUDA_MODULE_LOADING=EAGER
dev=${CUDA_VISIBLE_DEVICES:-0}
dev=${dev%%,*}
gpu=$(nvidia-smi -i "$dev" --query-gpu=name --format=csv,noheader | tr ' ' '_')
out="results/bench/${gpu}_$(hostname -s)_$(date +%Y%m%d-%H%M%S)"
mkdir -p "$out"
echo "writing to $out (CUDA_VISIBLE_DEVICES=$dev)"
nvidia-smi > "$out/nvidia_smi.txt"
grep -m1 "model name" /proc/cpuinfo > "$out/host_cpu.txt" || true
# CIMS GPUs are shared: record any other process present when a stage starts.
busy_check() {
  local n
  n=$(nvidia-smi -i "$dev" --query-compute-apps=pid --format=csv,noheader | wc -l)
  if [[ $n -gt 0 ]]; then
    echo "WARNING: $n other process(es) on GPU $dev during: $1; rerun when idle" \
      | tee -a "$out/busy_warnings.txt"
  fi
}
busy_check kernels_bench
"$BUILD/kernels_bench" > "$out/kernels.csv" 2> "$out/kernels_check.txt"
tail -1 "$out/kernels_check.txt"

first=1
for g in own cublas; do
  for spec in "smolvla_flow.csv --steps=1" "smolvla_flow.csv --steps=10" \
              "smolvla_full.csv --steps=1 --reps=10"; do
    set -- $spec
    busy_check "chain $g $spec"
    "$BUILD/chain_replay" --list="bench/chains/$1" "${@:2}" --gemm=$g 2>> "$out/chains_log.txt" \
      | if [[ $first == 1 ]]; then cat; else tail -n +2; fi >> "$out/chains.csv"
    first=0
  done
done
for g in own cublas; do
  busy_check "per-op $g"
  "$BUILD/chain_replay" --list=bench/chains/smolvla_flow.csv --gemm=$g --reps=3 \
    --per-op="$out/perop_flow_$g.csv" > /dev/null 2>> "$out/chains_log.txt"
done
# Sustained SM clock under this workload (latency-bound kernels scale with it).
scripts/clock_probe.sh "$BUILD" "$out" > /dev/null
cat "$out/sm_clock_summary.txt"
echo "done: $out"
cut -d, -f3,4,5,6,11 "$out/chains.csv" | column -t -s,
