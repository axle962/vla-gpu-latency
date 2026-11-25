#!/usr/bin/env bash
# eps probe for the graph-worthiness rule (workloads/eps_probe.py): per-kernel graph time minus
# CUPTI-traced eager duration on chains of small ops, no VLA. About 1 minute per GPU.
# Writes results/calib/<gpu>_<host>_<time>/eps_probe_smolvla_env.csv, which
# model/graph_worthiness.py uses for its "eps from probe" variant.
#
# CIMS (cuda2 GPU 1, cuda3, cuda5), on an idle GPU:
#   CUDA_VISIBLE_DEVICES=1 WORK=/scratch/$USER scripts/run_eps_probe.sh
# NYU HPC A100 (about 0.05 GPU-hours), from the repo on /scratch:
#   unset SLURM_EXPORT_ENV SLURM_GET_USER_ENV
#   sbatch -A csci_ga_3033_025-2026fa -p c12m85-a100-1 --gres=gpu:1 --time=00:15:00 \
#     --export=ALL --output=logs/eps-probe-%j.out --wrap "scripts/run_eps_probe.sh"
set -euo pipefail
cd "$(dirname "$0")/.."
WORK=${WORK:-/scratch/$USER}
PY=${PY:-$WORK/envs/smolvla/bin/python}
[[ -x "$PY" ]] || { echo "no $PY: set PY or WORK to where envs/smolvla lives"; exit 1; }
export CUDA_MODULE_LOADING=EAGER
DEV=${CUDA_VISIBLE_DEVICES:-0}
DEV=${DEV%%,*}
if [[ -z ${SLURM_JOB_ID:-} && -z ${FORCE:-} ]]; then
  n=$(nvidia-smi -i "$DEV" --query-compute-apps=pid --format=csv,noheader | grep -c . || true)
  [[ $n -eq 0 ]] || { echo "GPU $DEV has $n other process(es); wait or set FORCE=1"; exit 1; }
  read -r ps clk < <(nvidia-smi -i "$DEV" --query-gpu=pstate,clocks.sm --format=csv,noheader \
    | tr -d ',' | awk '{print $1, $2}')
  name=$(nvidia-smi -i "$DEV" --query-gpu=name --format=csv,noheader)
  if [[ $ps == P0 && $name =~ GeForce|TITAN ]]; then
    echo "GPU $DEV is idle but in P0 at $clk MHz: possibly stuck at a fixed clock (cuda2 GPU 0)."
    echo "Use another GPU or set FORCE=1."
    exit 1
  fi
fi
pin=()
if command -v numactl >/dev/null; then
  bus=$(nvidia-smi -i "$DEV" --query-gpu=pci.bus_id --format=csv,noheader | cut -d: -f2-3 \
    | tr 'A-F' 'a-f')
  node=$(cat "/sys/bus/pci/devices/0000:$bus/numa_node" 2>/dev/null || echo -1)
  [[ $node -ge 0 ]] && pin=(numactl --cpunodebind="$node" --membind="$node")
fi
GPU=$(nvidia-smi -i "$DEV" --query-gpu=name --format=csv,noheader | tr ' ' '_')
OUT="results/calib/${GPU}_$(hostname -s)_$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT"
grep -m1 "model name" /proc/cpuinfo > "$OUT/host_cpu.txt"
nvidia-smi -i "$DEV" --query-gpu=name,pstate,clocks.sm,clocks.max.sm,driver_version \
  --format=csv > "$OUT/nvidia_smi.txt"
echo "eps probe on GPU $DEV ($GPU) ${pin[*]:-(unpinned)} -> $OUT"
"${pin[@]}" "$PY" workloads/eps_probe.py > "$OUT/eps_probe_smolvla_env.csv"
column -s, -t < <(cut -d, -f3,5,11 "$OUT/eps_probe_smolvla_env.csv")
