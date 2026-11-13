#!/usr/bin/env bash
# Sustained SM clock under the VLA workload: samples nvidia-smi every 250 ms while the C++ replay
# of the SmolVLA flow step (10 steps, cuBLAS GEMMs) runs eager and as a CUDA graph. Spec boost
# clocks are not what the GPU runs at: on 2026-11-12 the TITAN V (cuda3) held 1335 MHz (max
# 1912) while the 2080 Ti (cuda2) held ~1965 MHz (max 2160). Latency-bound small kernels scale
# with this clock, so it is a per-GPU feature for model/cross_gpu.py.
#   scripts/clock_probe.sh <build_dir> [out_dir]   (default out_dir: this host's latest calib dir)
# Writes sm_clock_under_load.csv (raw samples) and sm_clock_summary.txt. Needs an idle GPU.
set -euo pipefail
cd "$(dirname "$0")/.."
BUILD=${1:?build dir, e.g. bench/build-$(hostname -s)}
out=${2:-$(ls -dt results/calib/*_"$(hostname -s)"_* 2>/dev/null | head -1)}
[[ -n $out ]] || { echo "no calib dir for $(hostname -s); run the calib stage first"; exit 1; }
dev=${CUDA_VISIBLE_DEVICES:-0}
dev=${dev%%,*}
n=$(nvidia-smi -i "$dev" --query-compute-apps=pid --format=csv,noheader | grep -c . || true)
[[ $n -eq 0 ]] || echo "WARNING: $n other process(es) on GPU $dev"
q=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_throttle_reasons.active
nvidia-smi -i "$dev" --query-gpu=$q --format=csv -lms 250 > "$out/sm_clock_under_load.csv" &
mon=$!
trap 'kill $mon 2>/dev/null || true' EXIT
CUDA_VISIBLE_DEVICES=$dev CUDA_MODULE_LOADING=EAGER "$BUILD/chain_replay" \
  --list=bench/chains/smolvla_flow.csv --steps=10 --gemm=cublas --reps=60 > "$out/clock_probe_chain.csv"
kill $mon
wait $mon 2>/dev/null || true
{
  echo "host $(hostname -s), GPU $dev ($(nvidia-smi -i "$dev" --query-gpu=name --format=csv,noheader))"
  echo "spec: $(nvidia-smi -i "$dev" --query-gpu=clocks.max.sm,clocks.applications.graphics --format=csv,noheader) (max SM, application clock)"
  # Samples above 50% of the busiest clock are "under load" (drops the idle samples at the ends).
  tail -n +2 "$out/sm_clock_under_load.csv" | awk -F', ' '{gsub(/ MHz/,"",$2); print $2}' | sort -n \
    | awk '{a[NR]=$1} END{mx=a[NR]; for(i=1;i<=NR;i++) if(a[i]>0.5*mx) b[++m]=a[i];
            printf "SM clock under load: median %d MHz, min %d, max %d (%d of %d samples)\n",
            b[int((m+1)/2)], b[1], b[m], m, NR}'
  awk -F, 'NR > 1 && $6 == "event" {printf "chain 10 flow steps, cuBLAS, %s: %.2f ms\n", $5, $11/1000}' \
    "$out/clock_probe_chain.csv"
} | tee "$out/sm_clock_summary.txt"
