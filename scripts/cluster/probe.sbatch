#!/usr/bin/env bash
# Target-side probes for the cross-GPU model (workloads/torch_probe.py): host Python speed and
# cold-cache small PyTorch kernels in CUDA graphs. sbatch -p c12m85-a100-1 scripts/cluster/probe.sbatch
# Cost: ~2 min of one GPU (< 0.1 GPU-h including boot).
#SBATCH --job-name=probe
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --no-requeue
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
echo "job $SLURM_JOB_ID on $(hostname)"
gpu=$(nvidia-smi -i 0 --query-gpu=name --format=csv,noheader | tr ' ' '_')
out="results/calib/${gpu}_$(hostname -s)_$(date +%Y%m%d-%H%M%S)"
mkdir -p "$out"
grep -m1 "model name" /proc/cpuinfo > "$out/host_cpu.txt" || true
nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm --format=csv > "$out/nvidia_smi.txt"
CUDA_MODULE_LOADING=EAGER envs/smolvla/bin/python workloads/torch_probe.py > "$out/torch_probe_smolvla_env.csv"
cat "$out/torch_probe_smolvla_env.csv"
