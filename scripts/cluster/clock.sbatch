#!/usr/bin/env bash
# Sustained SM clock under the SmolVLA C++ chain (scripts/clock_probe.sh) on a cluster GPU.
#   sbatch -p c12m85-a100-1 scripts/cluster/clock.sbatch [out_dir]
# Cost: ~1 min of one GPU (< 0.05 GPU-h including boot).
#SBATCH --job-name=clock
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --no-requeue
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
echo "job $SLURM_JOB_ID on $(hostname)"
export LD_LIBRARY_PATH=/scratch/$USER/.tools/cuda124/lib:${LD_LIBRARY_PATH:-}
scripts/clock_probe.sh bench/build-cluster "${1:?out dir}"
