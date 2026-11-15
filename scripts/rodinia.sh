#!/usr/bin/env bash
# Rodinia sanity check for the cross-GPU predictor (course handout suggests Rodinia/Parboil):
# classic throughput kernels, in contrast to the batch-1 VLA regime. Builds nine Rodinia CUDA
# benchmarks that generate their own inputs (the suite's data/ files are not needed) with
# nvcc -O3 for this GPU, runs each once to warm up and once under nsys (CUDA trace), and writes
# per-kernel summaries to results/rodinia/<gpu>_<host>_<stamp>/<bench>.csv
# (nsys cuda_gpu_kern_sum: total, instances, avg, median, min, max per kernel name).
#   RODINIA=/path/to/gpu-rodinia WORK=/scratch/$USER scripts/rodinia.sh
# Source: https://github.com/yuhc/gpu-rodinia (commit 9c10d3e), cloned outside the repo.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${WORK:?set WORK}"
R="${RODINIA:-$HOME/fall2026/GPU/ext/gpu-rodinia}/cuda"
[[ -d $R ]] || { echo "clone https://github.com/yuhc/gpu-rodinia to $R/.."; exit 1; }
command -v nsys >/dev/null || { source /etc/profile.d/modules.sh; module load cuda-12.4; }
DEV=${CUDA_VISIBLE_DEVICES:-0}
CC=$(nvidia-smi -i "$DEV" --query-gpu=compute_cap --format=csv,noheader | tr -d '. ')
HOST=$(hostname -s)
B="$WORK/rodinia-build-$HOST-sm$CC"
mkdir -p "$B"
A=(-O3 "-arch=sm_$CC" -w)
build() {  # name sources...
  local n=$1; shift
  [[ -x $B/$n ]] || (cd "$R" && nvcc "${A[@]}" "$@" -o "$B/$n" -lm) > "$B/$n.log" 2>&1 \
    || { echo "BUILD FAILED: $n (see $B/$n.log)"; return 1; }
}
build backprop backprop/backprop_cuda.cu backprop/facetrain.c backprop/imagenet.c backprop/backprop.c
build gaussian gaussian/gaussian.cu
build lavaMD lavaMD/main.c lavaMD/kernel/kernel_gpu_cuda_wrapper.cu lavaMD/util/num/num.c \
  lavaMD/util/timer/timer.c lavaMD/util/device/device.cu
build lud lud/cuda/lud.cu lud/cuda/lud_kernel.cu lud/common/common.c -Ilud/common
build nw nw/needle.cu
build particlefilter_float particlefilter/ex_particle_CUDA_float_seq.cu -use_fast_math
build pathfinder pathfinder/pathfinder.cu
build srad_v2 srad/srad_v2/srad.cu
build streamcluster streamcluster/streamcluster_cuda_cpu.cpp streamcluster/streamcluster_cuda.cu \
  streamcluster/streamcluster_header.cu

gname=$(nvidia-smi -i "$DEV" --query-gpu=name --format=csv,noheader | tr ' ' _)
out="results/rodinia/${gname}_${HOST}_$(date +%Y%m%d-%H%M%S)"
mkdir -p "$out"
nvidia-smi -i "$DEV" --query-gpu=name,driver_version,clocks.max.sm --format=csv > "$out/gpu.txt"
grep -m1 "model name" /proc/cpuinfo >> "$out/gpu.txt"
node=$(cat /sys/bus/pci/devices/0000:$(nvidia-smi -i "$DEV" --query-gpu=pci.bus_id \
  --format=csv,noheader | cut -d: -f2-3 | tr 'A-F' 'a-f')/numa_node 2>/dev/null || echo -1)
pin=()
command -v numactl >/dev/null && [[ $node -ge 0 ]] && pin=(numactl --cpunodebind="$node" --membind="$node")
echo "rodinia on GPU $DEV ${pin[*]:-(unpinned)}" | tee -a "$out/gpu.txt"
run_dir="$WORK/tmp/rodinia-run-$HOST"
mkdir -p "$run_dir"
declare -A ARGS=(
  [backprop]="1048560"  # default 2097152 exceeds the 65535 grid.y limit
  [gaussian]="-s 1024"
  [lavaMD]="-boxes1d 10"
  [lud]="-s 2048"
  [nw]="4096 10"
  [particlefilter_float]="-x 128 -y 128 -z 10 -np 100000"
  [pathfinder]="100000 100 20"
  [srad_v2]="2048 2048 0 127 0 127 0.5 20"
  [streamcluster]="10 20 256 65536 65536 1000 none $run_dir/sc_out.txt 1"
)
for n in backprop gaussian lavaMD lud nw particlefilter_float pathfinder srad_v2 streamcluster; do
  [[ -x $B/$n ]] || continue
  echo "=== $n ${ARGS[$n]}"
  # shellcheck disable=SC2086
  (cd "$run_dir" && CUDA_MODULE_LOADING=EAGER "${pin[@]}" "$B/$n" ${ARGS[$n]} > /dev/null 2>&1) || echo "warm-up failed: $n"
  # shellcheck disable=SC2086
  (cd "$run_dir" && CUDA_MODULE_LOADING=EAGER "${pin[@]}" nsys profile --trace=cuda --force-overwrite=true \
    -o "$run_dir/$n" "$B/$n" ${ARGS[$n]} > "$run_dir/$n.stdout" 2>&1) || { echo "nsys failed: $n"; continue; }
  nsys stats --force-export=true --report cuda_gpu_kern_sum --format csv -o "$run_dir/$n" \
    "$run_dir/$n.nsys-rep" > "$run_dir/$n.stats.log" 2>&1 || { echo "nsys stats failed: $n"; continue; }
  mv "$run_dir/${n}_cuda_gpu_kern_sum.csv" "$out/$n.csv" && head -4 "$out/$n.csv" | cut -c1-150
done
echo "wrote $out"
