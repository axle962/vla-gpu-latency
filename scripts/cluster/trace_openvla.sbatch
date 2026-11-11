#!/usr/bin/env bash
# Trace OpenVLA-7B inference (eager, prefill vs decode) on one cluster GPU.
#   source scripts/cluster/site.sh
#   sbatch -p c12m85-a100-1 scripts/cluster/trace_openvla.sbatch [extra trace_openvla.py args]
# Weights must be in $HF_HOME already (scripts/cluster/download_openvla.sbatch); runs offline.
# Cost: 15 GB load ~3 min, 20+3 steps of ~0.3 s, profile; about 0.25 GPU-h including boot.
#SBATCH --job-name=trace-openvla
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --no-requeue
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
echo "job $SLURM_JOB_ID on $(hostname) partition ${SLURM_JOB_PARTITION:-?}"
nvidia-smi --query-gpu=name,driver_version,clocks.max.sm --format=csv
export HF_HOME=/scratch/$USER/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_MODULE_LOADING=EAGER
envs/openvla/bin/python workloads/trace_openvla.py "$@"
