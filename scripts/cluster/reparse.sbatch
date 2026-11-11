#!/usr/bin/env bash
# Re-parse saved torch.profiler traces on a CPU node (the login node has ~2 GB of RAM, too
# little for a 100+ MB trace). No GPU, so this costs no GPU-hours.
#   source scripts/cluster/site.sh
#   sbatch -p n2c48m24 scripts/cluster/reparse.sbatch traces/smolvla_*/*/
#SBATCH --job-name=reparse
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
envs/smolvla/bin/python workloads/trace_parse.py "$@"
