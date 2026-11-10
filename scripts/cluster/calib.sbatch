#!/usr/bin/env bash
# Calibration microbenchmarks on one cluster GPU. Pick the GPU with the partition:
#   source scripts/cluster/site.sh
#   sbatch -p g2-standard-12 scripts/cluster/calib.sbatch   # L4 (Ada)
#   sbatch -p c12m85-a100-1  scripts/cluster/calib.sbatch   # A100
#   (T4, V100 and g4-standard-48 partitions reject our account as of 2026-11-10)
# Build first on the login node: make -C bench ARCHS="75 70 80 89" BUILD=build-cluster
# Cost: a few minutes of one GPU (well under 0.2 GPU-h per job including boot).
#SBATCH --job-name=calib
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
echo "job $SLURM_JOB_ID on $(hostname) partition ${SLURM_JOB_PARTITION:-?}"
scripts/run_calib.sh bench/build-cluster
