#!/usr/bin/env bash
# Trace SmolVLA inference (eager, per-phase, CUDA-graph denoising loop) on one cluster GPU.
#   source scripts/cluster/site.sh
#   sbatch -p c12m85-a100-1 scripts/cluster/trace_smolvla.sbatch [extra trace_smolvla.py args]
# Weights must be in $HF_HOME already (download on the login node); the job runs offline.
# Cost: model load ~2 min plus ~1 min of timing; under 0.5 GPU-h per job including boot.
#SBATCH --job-name=trace-smolvla
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --no-requeue
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
echo "job $SLURM_JOB_ID on $(hostname) partition ${SLURM_JOB_PARTITION:-?}"
nvidia-smi --query-gpu=name,driver_version,clocks.max.sm --format=csv
export HF_HOME=/scratch/$USER/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_MODULE_LOADING=EAGER
envs/smolvla/bin/python workloads/trace_smolvla.py "$@"
