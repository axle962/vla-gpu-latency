# Cluster settings. Copy to scripts/cluster/site.sh (git-ignored) and `source` it in every
# login shell before sbatch. sbatch reads SBATCH_* variables as option defaults.

# GPU course allocation: 300 GPU-hours (gres/gpu=18000 GPU-minutes). Do NOT use the
# wildfire account (csci_ga_2271-2026fa) for this project.
export SBATCH_ACCOUNT="csci_ga_3033_025-2026fa"

# CUDA 12.4 toolchain (same version as CIMS cuda2), installed with conda into scratch:
#   conda create -p /scratch/$USER/.tools/cuda124 -c nvidia/label/cuda-12.4.1 \
#       cuda-nvcc cuda-cudart-dev cuda-cudart-static
export PATH="/scratch/$USER/.tools/cuda124/bin:$PATH"

# GPU-hour tally for this account (-D includes requeued attempts after failed boots):
#   sacct -D -S 2026-09-01 -X -A $SBATCH_ACCOUNT -n -P --format=ElapsedRaw,AllocTRES \
#     | awk -F'|' '$2~/gres\/gpu/{s+=$1} END{print s/3600" GPU-h"}'
