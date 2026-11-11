#!/usr/bin/env bash
# Download openvla/openvla-7b (safetensors, configs, remote code; ~15 GB) into $HF_HOME on a
# CPU node. No GPU, costs no GPU-hours.
#   source scripts/cluster/site.sh && sbatch -p n2c48m24 scripts/cluster/download_openvla.sbatch
#SBATCH --job-name=dl-openvla
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --no-requeue
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
export HF_HOME=/scratch/$USER/.cache/huggingface HF_HUB_ENABLE_HF_TRANSFER=0
envs/openvla/bin/python - <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download(
    "openvla/openvla-7b",
    allow_patterns=["*.json", "*.py", "*.safetensors", "*.model", "*.txt"],
)
print(p)
import os
for f in sorted(os.listdir(p)):
    print(f, os.path.getsize(os.path.join(p, f)))
PY
