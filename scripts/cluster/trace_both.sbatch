#!/usr/bin/env bash
# SmolVLA baseline + OpenVLA-7B traces (eager, phases, CUDA graph, profiled graph replays) in
# one job (one node boot).
#   source scripts/cluster/site.sh && sbatch -p c12m85-a100-1 scripts/cluster/trace_both.sbatch
# Cost: ~3 min SmolVLA + ~5 min OpenVLA + boot, about 0.2 GPU-h.
#SBATCH --job-name=trace-both
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
echo "=== smolvla ($(date +%T))"
envs/smolvla/bin/python workloads/trace_smolvla.py 2>&1 | grep -E "diff|graph profile|wrote|Error|error"
echo "=== openvla ($(date +%T))"
envs/openvla/bin/python workloads/trace_openvla.py 2>&1 | grep -E "tokens|graph profile|wrote|Error|error"
echo "done ($(date +%T))"
