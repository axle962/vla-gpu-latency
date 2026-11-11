#!/usr/bin/env bash
# SmolVLA sweep on one GPU, one factor at a time around the baseline (3 cameras, 10 flow steps,
# chunk 50), plus a baseline repeat for run-to-run variance. One job, so one node boot.
#   source scripts/cluster/site.sh
#   sbatch -p c12m85-a100-1 scripts/cluster/sweep_smolvla.sbatch
# Cost: 7 runs x ~2.5 min (model load, timing, profiling, trace parse) ~ 20 min, about
# 0.35 GPU-h including boot. A failed config is logged and the sweep continues.
#SBATCH --job-name=sweep-smolvla
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --no-requeue
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
echo "job $SLURM_JOB_ID on $(hostname) partition ${SLURM_JOB_PARTITION:-?}"
nvidia-smi --query-gpu=name,driver_version,clocks.max.sm --format=csv
export HF_HOME=/scratch/$USER/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_MODULE_LOADING=EAGER

configs=(
  "--cameras 3 --num-steps 10 --chunk-size 50"
  "--cameras 1 --num-steps 10 --chunk-size 50"
  "--cameras 2 --num-steps 10 --chunk-size 50"
  "--cameras 3 --num-steps 5 --chunk-size 50"
  "--cameras 3 --num-steps 20 --chunk-size 50"
  "--cameras 3 --num-steps 10 --chunk-size 25"
  "--cameras 3 --num-steps 10 --chunk-size 100"
)
fail=0
for c in "${configs[@]}"; do
  echo "=== $c ($(date +%T))"
  # shellcheck disable=SC2086
  envs/smolvla/bin/python workloads/trace_smolvla.py $c > "logs/sweep-${SLURM_JOB_ID}-$(echo $c | tr -d ' -').log" 2>&1 \
    && grep -E "max \|diff\||wrote" "logs/sweep-${SLURM_JOB_ID}-$(echo $c | tr -d ' -').log" \
    || { echo "FAILED: $c"; fail=$((fail + 1)); }
done
echo "done ($(date +%T)), $fail failed"
