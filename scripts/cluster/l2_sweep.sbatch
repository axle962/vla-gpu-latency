#!/usr/bin/env bash
# L2 working-set sweep (scripts/l2_sweep.sh) on a cluster GPU:
#   sbatch -p c12m85-a100-1 scripts/cluster/l2_sweep.sbatch
# Build first: make -C bench ARCHS="75 70 80 89" BUILD=build-cluster
# Cost: ~2 to 3 min of one GPU (< 0.1 GPU-h including boot).
#SBATCH --job-name=l2sweep
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --no-requeue
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
echo "job $SLURM_JOB_ID on $(hostname)"
export LD_LIBRARY_PATH=/scratch/$USER/.tools/cuda124/lib:${LD_LIBRARY_PATH:-}
scripts/l2_sweep.sh bench/build-cluster
