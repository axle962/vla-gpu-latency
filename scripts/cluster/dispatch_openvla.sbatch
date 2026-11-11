#!/usr/bin/env bash
# One A100 job (one node boot): PyTorch per-op dispatch benchmark under both torch builds used
# by the workloads, then the OpenVLA trace with static-cache and CUDA-graph decode modes.
#   source scripts/cluster/site.sh
#   sbatch -p c12m85-a100-1 scripts/cluster/dispatch_openvla.sbatch
# Cost: ~2 min dispatch + ~5 min OpenVLA + boot, about 0.25 GPU-h.
#SBATCH --job-name=dispatch-openvla
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --no-requeue
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
echo "job $SLURM_JOB_ID on $(hostname) partition ${SLURM_JOB_PARTITION:-?}"
nvidia-smi --query-gpu=name,driver_version,clocks.max.sm --format=csv
export HF_HOME=/scratch/$USER/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_MODULE_LOADING=EAGER
name=$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 | head -1 | tr ' ' '_' | tr -cd '[:alnum:]_-')
out="results/calib/${name}_$(hostname -s)_$(date +%Y%m%d-%H%M%S)"
mkdir -p "$out"
grep -m1 "model name" /proc/cpuinfo > "$out/host_cpu.txt"
for env in smolvla openvla; do
  echo "=== torch_dispatch in envs/$env ($(date +%T))"
  envs/$env/bin/python workloads/torch_dispatch.py > "$out/torch_dispatch_${env}_env.csv" \
    && wc -l "$out/torch_dispatch_${env}_env.csv" || echo "FAILED: torch_dispatch $env"
done
echo "=== openvla ($(date +%T))"
envs/openvla/bin/python workloads/trace_openvla.py 2>&1 | grep -v "Warning\|warn(" | tail -60
echo "done ($(date +%T))"
