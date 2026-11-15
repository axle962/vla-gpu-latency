#!/usr/bin/env bash
# Host CPU as the experimental variable, on one GPU (cuda2): does the cross-GPU model's host
# scaling predict a slower host? Everything runs pinned to one core C of the GPU's NUMA node,
# under four conditions:
#   alone     nothing else on C or its hyperthread sibling S
#   smt_alu   an integer spin loop on S (the core's other hardware thread: a mildly slower core)
#   smt_mem   a memory-streaming loop (64 MB copies) on S (also contends for L1/L2/L3 and DRAM)
#   share     the spin loop on C itself (the OS time-slices the core: about half a core)
# Per condition it measures the target-side inputs of model/cross_gpu.py that depend on the
# host (C++ launch_latency, PyTorch dispatch table, Python/cold probe) and the SmolVLA and ACT
# baselines (ground truth). GPU-side calibration (bandwidth, device_info, C++ bench) is the
# GPU's own and is copied from its latest calib run. OMP_NUM_THREADS=1 in every condition.
# Outputs (names avoid the patterns model/cross_gpu.py uses for the normal 2080 Ti runs):
#   results/calib/HOSTEXP-<cond>-<gpu>-<host>-<stamp>/
#   traces/<config>/HOSTEXP-<cond>-<gpu>-<host>-<stamp>/
# Analysis: model/host_experiment.py.
#   WORK=/scratch/$USER CUDA_VISIBLE_DEVICES=<idle gpu> scripts/host_contention.sh [conditions]
set -euo pipefail
cd "$(dirname "$0")/.."
: "${WORK:?set WORK}"
DEV=${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to one idle GPU}
HOST=$(hostname -s)
PY="$WORK/envs/smolvla/bin/python"
BUILD=bench/build-$HOST
export HF_HOME="$WORK/.cache/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCH_HOME="$WORK/.cache/torch" CUDA_MODULE_LOADING=EAGER OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
node=$(cat /sys/bus/pci/devices/0000:$(nvidia-smi -i "$DEV" --query-gpu=pci.bus_id \
  --format=csv,noheader | cut -d: -f2-3 | tr 'A-F' 'a-f')/numa_node)
# Core C: the third physical core of the node (the first cores often carry OS work); S: its
# hyperthread sibling. cpulist may be "1,3,5" or "0-9,20-29".
cpus=$(tr ',' '\n' < /sys/devices/system/node/node$node/cpulist | \
  awk -F- '{ if (NF == 2) for (i = $1; i <= $2; i++) print i; else print $1 }')
C=$(for c in $cpus; do
      sib=$(cut -d, -f1 < /sys/devices/system/cpu/cpu$c/topology/thread_siblings_list)
      [[ $sib == "$c" ]] && echo "$c"
    done | sed -n 3p)
S=$(tr ',' '\n' < /sys/devices/system/cpu/cpu$C/topology/thread_siblings_list | grep -vx "$C" | head -1)
echo "GPU $DEV (NUMA node $node), core $C, sibling $S"
gname=$(nvidia-smi -i "$DEV" --query-gpu=name --format=csv,noheader | tr ' ' _)
src_calib=$(ls -dt results/calib/${gname}_${HOST}_* | head -1)

hogs="$WORK/tmp/hogs"
mkdir -p "$hogs"
cat > "$hogs/alu.c" <<'EOF'
int main(void) { volatile unsigned long x = 0; for (;;) x += 3; }
EOF
cat > "$hogs/mem.c" <<'EOF'
#include <stdlib.h>
#include <string.h>
int main(void) {
  size_t n = 64u << 20; char *a = malloc(n), *b = malloc(n);
  memset(a, 1, n); memset(b, 2, n);
  for (;;) { memcpy(b, a, n); memcpy(a, b, n); }
}
EOF
gcc -O2 -o "$hogs/alu" "$hogs/alu.c"
gcc -O2 -o "$hogs/mem" "$hogs/mem.c"

run_cond() {
  local cond=$1 hog="" cpu=""
  case $cond in
    alone) ;;
    smt_alu) hog=alu cpu=$S ;;
    smt_mem) hog=mem cpu=$S ;;
    share) hog=alu cpu=$C ;;
    *) echo "unknown condition $cond"; return 1 ;;
  esac
  local hp=""
  if [[ -n $hog ]]; then
    taskset -c "$cpu" "$hogs/$hog" & hp=$!
    sleep 2
  fi
  local stamp tag out pin
  stamp=$(date +%Y%m%d-%H%M%S)
  # Dashes only: model/cross_gpu.py finds 2080 Ti traces by "RTX_2080_Ti" and run_cuda2.sh finds
  # calib dirs by "*_<host>_*"; neither may match these runs.
  tag="HOSTEXP-${cond}-${gname//_/-}-${HOST}-${stamp}"
  out="results/calib/$tag"
  mkdir -p "$out"
  cp "$src_calib"/{device_info.csv,bandwidth.csv,bandwidth_summary.txt} "$out/"
  { grep -m1 "model name" /proc/cpuinfo; echo "condition=$cond core=$C sibling=$S hog=${hog:-none} hog_cpu=${cpu:-none}"; } > "$out/host_cpu.txt"
  pin=(numactl --physcpubind="$C" --membind="$node")
  echo "[$(date +%T)] === $cond -> $out"
  "${pin[@]}" "$BUILD/launch_latency" > "$out/launch_eager_loading.csv"
  "${pin[@]}" "$PY" workloads/torch_dispatch.py > "$out/torch_dispatch_smolvla_env.csv"
  "${pin[@]}" "$PY" workloads/torch_probe.py > "$out/torch_probe_smolvla_env.csv"
  "${pin[@]}" "$PY" workloads/trace_smolvla.py \
    --out "traces/smolvla_cam3_steps10_chunk50/$tag" > "logs/$HOST-hostexp-$cond-smolvla.log" 2>&1 \
    || echo "FAILED: smolvla $cond"
  "${pin[@]}" "$PY" workloads/trace_smolvla.py --cameras 1 \
    --out "traces/smolvla_cam1_steps10_chunk50/$tag" > "logs/$HOST-hostexp-$cond-smolvla-cam1.log" 2>&1 \
    || echo "FAILED: smolvla cam1 $cond"
  "${pin[@]}" "$PY" workloads/trace_act.py \
    --out "traces/act_res480x640_chunk100/$tag" > "logs/$HOST-hostexp-$cond-act.log" 2>&1 \
    || echo "FAILED: act $cond"
  grep -h '"eager_ms_median"\|"hybrid_ms_median"\|"graph_ms_median"' \
    traces/*/"$tag"/meta.json | tr -s ' ' || true
  [[ -n $hp ]] && kill "$hp"
  wait 2>/dev/null || true
}

for cond in ${@:-alone smt_alu smt_mem share}; do
  run_cond "$cond"
done
echo "[$(date +%T)] host contention done"
