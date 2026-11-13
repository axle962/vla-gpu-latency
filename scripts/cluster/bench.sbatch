#!/usr/bin/env bash
# Standalone CUDA C++ benchmarks (our kernels, cuBLAS, C++ replay of the SmolVLA step) on one
# cluster GPU: sbatch -p c12m85-a100-1 scripts/cluster/bench.sbatch
# Build first on the login node (needs libcublas-dev and cuda-profiler-api in the conda env):
#   make -C bench ARCHS="75 70 80 89" BUILD=build-cluster
# Cost: ~6 min on the RTX 2080 Ti, so about 0.1 to 0.15 GPU-h per job including boot.
#SBATCH --job-name=bench
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --no-requeue
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
echo "job $SLURM_JOB_ID on $(hostname) partition ${SLURM_JOB_PARTITION:-?}"
export LD_LIBRARY_PATH=/scratch/$USER/.tools/cuda124/lib:${LD_LIBRARY_PATH:-}
scripts/run_bench.sh bench/build-cluster
