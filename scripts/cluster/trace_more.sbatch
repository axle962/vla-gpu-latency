#!/usr/bin/env bash
# Trace the extra workloads (Diffusion Policy, ACT) on one cluster GPU.
#   source scripts/cluster/site.sh
#   sbatch -p c12m85-a100-1 scripts/cluster/trace_more.sbatch smoke   # ~5 min: 1 config each, few reps
#   sbatch -p c12m85-a100-1 scripts/cluster/trace_more.sbatch sweep   # ~20 min: full config sweep
# Weights must be in $HF_HOME (lerobot/diffusion_pusht, lerobot/act_aloha_sim_transfer_cube_human)
# and the torchvision ResNet18 file in $TORCH_HOME/hub/checkpoints (download on the login node).
#SBATCH --job-name=trace-more
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --no-requeue
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
echo "job ${SLURM_JOB_ID:-?} on $(hostname) partition ${SLURM_JOB_PARTITION:-?}"
nvidia-smi --query-gpu=name,driver_version,clocks.max.sm --format=csv
export HF_HOME=/scratch/$USER/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCH_HOME=/scratch/$USER/.cache/torch CUDA_MODULE_LOADING=EAGER
PY=envs/smolvla/bin/python
mode=${1:-sweep}
if [ "$mode" = smoke ]; then
  o=/scratch/$USER/tmp/smoke_more_${SLURM_JOB_ID:-x}
  $PY workloads/trace_diffusion.py --num-steps 10 --warmup 2 --reps 3 --prof-steps 1 --out $o/diffusion
  $PY workloads/trace_act.py --warmup 2 --reps 3 --prof-steps 1 --out $o/act
  $PY workloads/torch_dispatch.py --set conv > $o/torch_dispatch_conv.csv && cat $o/torch_dispatch_conv.csv
  exit 0
fi
# conv dispatch rows for the model (same host as the traces)
c=results/calib/$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' _)_$(hostname -s)_$(date +%Y%m%d-%H%M%S)
mkdir -p "$c" && grep -m1 "model name" /proc/cpuinfo > "$c/host_cpu.txt"
$PY workloads/torch_dispatch.py --set conv > "$c/torch_dispatch_conv_smolvla_env.csv" || echo "FAILED: conv dispatch"
for s in 100 10 25 50 100; do $PY workloads/trace_diffusion.py --num-steps $s || echo "FAILED: diffusion $s"; done
for hw in "480 640" "240 320" "480 640"; do
  set -- $hw; $PY workloads/trace_act.py --height $1 --width $2 || echo "FAILED: act $hw"
done
