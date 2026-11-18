#!/usr/bin/env bash
# Multi-policy hosting matrix on one Slurm GPU (scripts/run_multipolicy.sh). Site-neutral: pass
# the partition, QOS and account on the command line, and WORK (the directory that holds
# envs/smolvla and .cache/huggingface) in the environment, e.g. on Empire AI:
#   sbatch -p alpha --qos=test --export=ALL,WORK=$HOME/vla scripts/cluster/multipolicy.sbatch quick
# on the NYU cluster (source scripts/cluster/site.sh first):
#   sbatch -p c12m85-a100-1 --export=ALL,WORK=/scratch/$USER scripts/cluster/multipolicy.sbatch quick
# A GPU type can be requested with --gres=gpu:<type>:1 (sinfo -o %G lists the names).
#SBATCH --job-name=multipolicy
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --no-requeue
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
echo "job ${SLURM_JOB_ID:-?} on $(hostname) partition ${SLURM_JOB_PARTITION:-?}, cpus ${SLURM_CPUS_PER_TASK:-?}"
nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,memory.total --format=csv
grep -m1 "model name" /proc/cpuinfo
export WORK=${WORK:-/scratch/$USER}
exec scripts/run_multipolicy.sh "$@"
