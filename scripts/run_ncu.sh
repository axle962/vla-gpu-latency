#!/usr/bin/env bash
# Nsight Compute per-kernel counters (milestone 4: source profiling) on one GPU:
#   chain_flow_own_{warm,cold}  the C++ replay of one SmolVLA flow step, our kernels, with L2
#                               left as the previous kernel left it (warm) or flushed (cold)
#   chain_flow_cublas           same with cuBLAS GEMMs (warm)
#   smolvla_step                one eager step of the real model (3 cameras, 10 steps, chunk 50),
#                               first NCU_MODEL_KERNELS kernels (default 3008 = the prefix plus
#                               the first flow step; the other 9 flow steps repeat the same kernels)
# Each becomes a per-kernel feature table (profile/ncu_features.py) in
# results/ncu/<gpu>_<host>_<stamp>/; the raw ncu CSVs stay in $WORK/ncu_raw (large).
#
# Settings: --replay-mode application (the whole program reruns once per counter pass; kernel
# replay saved and restored all device memory per kernel and pass, ~1.6 s per kernel with the
# replay's 350 MB of buffers, i.e. hours for one SmolVLA step), --clock-control none (boost
# clocks, as in the timed runs; ncu's default locks base clocks), --cache-control none for the
# "warm" runs so L2 hit rates reflect the real kernel sequence; the "cold" run
# (--cache-control all, L2 flushed before every kernel) bounds DRAM traffic from above.
# Application replay needs the same kernel sequence on every run, which holds here.
#
# Usage: WORK=... [PY=...] scripts/run_ncu.sh [build_dir] [chain|model|all]   (timing: see the log)
# On CIMS cuda2 set TMPDIR to a private dir: /tmp/nsight-compute-lock belongs to another user.
set -euo pipefail
cd "$(dirname "$0")/.."
BUILD=${1:-bench/build}
WHAT=${2:-all}
: "${WORK:?set WORK (raw ncu output goes there)}"
PY=${PY:-$WORK/envs/smolvla/bin/python}
command -v ncu >/dev/null || { echo "ncu not on PATH (module load cuda-12.4)"; exit 1; }
export CUDA_MODULE_LOADING=EAGER
export TMPDIR=${TMPDIR:-$WORK/tmp}
mkdir -p "$TMPDIR"
METRICS=$("$PY" -c 'import sys; sys.path.insert(0, "profile"); import ncu_features; print(ncu_features.METRICS)')
dev=${CUDA_VISIBLE_DEVICES:-0}
dev=${dev%%,*}
gpu=$(nvidia-smi -i "$dev" --query-gpu=name --format=csv,noheader | tr ' ' '_')
tag="${gpu}_$(hostname -s)_$(date +%Y%m%d-%H%M%S)"
out="results/ncu/$tag"
raw="$WORK/ncu_raw/$tag"
mkdir -p "$out" "$raw"
nvidia-smi > "$out/nvidia_smi.txt"
ncu --version | tail -1 > "$out/ncu_version.txt"
NCU=(ncu --profile-from-start off --replay-mode application --clock-control none
     --metrics "$METRICS" --csv --page raw --print-units base)

run() {  # name, cache-control, command...
  local name=$1 cc=$2
  shift 2
  echo "=== $name ($(date +%T))"
  local n
  n=$(nvidia-smi -i "$dev" --query-compute-apps=pid --format=csv,noheader | wc -l)
  [[ $n -eq 0 ]] || echo "WARNING: $n other process(es) on GPU $dev before $name; counters" \
    "such as DRAM bytes are GPU-wide" | tee -a "$out/busy_warnings.txt"
  "${NCU[@]}" --cache-control "$cc" --log-file "$raw/$name.csv" "$@" > "$raw/$name.out" 2>&1 \
    || { echo "ncu failed for $name, see $raw/$name.out"; tail -5 "$raw/$name.out"; return 1; }
  local chain=()
  [[ $name == chain_flow_own* ]] && chain=(--chain bench/chains/smolvla_flow.csv)
  "$PY" profile/ncu_features.py "$raw/$name.csv" -o "$out/$name.csv.gz" "${chain[@]}"
}

if [[ $WHAT == chain || $WHAT == all ]]; then
  L=bench/chains/smolvla_flow.csv
  run chain_flow_own_warm none "$BUILD/chain_replay" --list=$L --gemm=own --profile-once
  run chain_flow_own_cold all "$BUILD/chain_replay" --list=$L --gemm=own --profile-once
  run chain_flow_cublas none "$BUILD/chain_replay" --list=$L --gemm=cublas --profile-once
fi
if [[ $WHAT == model || $WHAT == all ]]; then
  export HF_HOME=${HF_HOME:-$WORK/.cache/huggingface} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  NCU+=(--launch-count "${NCU_MODEL_KERNELS:-3008}")
  run smolvla_step none "$PY" profile/ncu_smolvla.py
fi
echo "done: $out"
