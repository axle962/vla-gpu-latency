#!/usr/bin/env bash
# First thing to run on CIMS cuda2: do Nsight Compute counters and Nsight Systems work?
# The whole project depends on ncu counters. If ncu prints ERR_NVGPUCTRPERM, email the
# CIMS helpdesk and ask for GPU performance-counter access for your user on cuda2.
#   module load cuda-12.4 && make -C bench && scripts/check_profilers.sh
set -uo pipefail
cd "$(dirname "$0")/.."
BIN=${BUILD:-bench/build}/launch_latency
OUT="${TMPDIR:-/tmp}/vla_profcheck_$USER"
mkdir -p "$OUT"

echo "== nvcc / ncu / nsys versions"
nvcc --version | tail -1
ncu --version | tail -1 || echo "ncu not found"
nsys --version || echo "nsys not found"

echo "== ncu counter access"
if ncu --metrics gpu__time_duration.sum,dram__bytes_read.sum,lts__t_sector_hit_rate.pct \
    --launch-count 3 "$BIN" --reps=1 --warmup=0 > "$OUT/ncu.txt" 2>&1; then
  grep -E "gpu__time_duration|dram__bytes|lts__t_sector" "$OUT/ncu.txt" | head -6
  echo "ncu OK"
else
  grep -E "ERR_NVGPUCTRPERM|ERROR|==ERROR==" "$OUT/ncu.txt" | head -5
  echo "ncu FAILED (full log: $OUT/ncu.txt)"
fi

echo "== nsys trace"
if nsys profile -o "$OUT/nsys_test" --force-overwrite true "$BIN" --reps=2 > "$OUT/nsys.txt" 2>&1; then
  ls -la "$OUT"/nsys_test.* && echo "nsys OK"
else
  tail -5 "$OUT/nsys.txt"; echo "nsys FAILED"
fi
