#!/usr/bin/env bash
# RoboTwin 2.0 evaluation with lerobot-eval in envs/robotwin (built by robotwin_build.sbatch).
# The simulator runs in-process (no env server): RoboTwin is on PYTHONPATH and the working
# directory is the RoboTwin root, where its assets are read from.
#   source scripts/cluster/site.sh
#   sbatch -p c12m85-a100-1 scripts/cluster/robotwin_eval.sbatch [policy] [tasks] [episodes] [tag]
# Defaults: lerobot/smolvla_robotwin, beat_block_hammer, 5 episodes, tag "smoke".
# EVAL_BATCH=N runs N simulator copies in parallel (async envs, one GPU).
# Outputs in results/robotwin/<tag>_<job>/ (eval_info.json, a few episode videos).
#SBATCH --job-name=robotwin-eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --no-requeue
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
REPO=$PWD
POLICY=${1:-lerobot/smolvla_robotwin}
TASKS=${2:-beat_block_hammer}
EPISODES=${3:-5}
TAG=${4:-smoke}
OUT=$REPO/results/robotwin/${TAG}_${SLURM_JOB_ID:-x}
# lerobot/smolvla_robotwin was trained on 480x640 frames named camera1..3 (train_config.json:
# cam_high, cam_left_wrist, cam_right_wrist); the simulator names them head/left/right_camera.
H=${ROBOTWIN_H:-480} W=${ROBOTWIN_W:-640}
RENAME=${ROBOTWIN_RENAME:-'{"observation.images.head_camera": "observation.images.camera1", "observation.images.left_camera": "observation.images.camera2", "observation.images.right_camera": "observation.images.camera3"}'}
echo "job ${SLURM_JOB_ID:-?} on $(hostname): $POLICY on $TASKS, $EPISODES episodes -> $OUT"
nvidia-smi --query-gpu=name,driver_version,clocks.max.sm --format=csv
grep -m1 "model name" /proc/cpuinfo

export HF_HOME=/scratch/$USER/.cache/huggingface
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json
export CUDA_HOME=/scratch/$USER/.tools/cuda128 PATH="$REPO/envs/robotwin/bin:/scratch/$USER/.tools/cuda128/bin:$PATH"
RT=/scratch/$USER/ext/RoboTwin
export PYTHONPATH=$RT ROBOTWIN_ROOT=$RT SIM_WORKDIR=$RT
mkdir -p "$(dirname "$OUT")"
cd "$RT"
start=$(date +%s)
# ROBOTWIN_DELAY=d (control steps) runs lerobot-eval through workloads/robotwin_delay_eval.py,
# which wraps the policy with the simulated-delay model (also for d = 0, for paired seeds).
if [ -n "${ROBOTWIN_DELAY:-}" ]; then EVAL=(python "$REPO/workloads/robotwin_delay_eval.py"); else EVAL=(lerobot-eval); fi
"${EVAL[@]}" \
  --policy.path="$POLICY" \
  --policy.device=cuda \
  --env.type=robotwin \
  --env.task="$TASKS" \
  --eval.batch_size="${EVAL_BATCH:-1}" \
  --eval.n_episodes="$EPISODES" \
  --env.observation_height="$H" --env.observation_width="$W" \
  --rename_map="$RENAME" \
  --output_dir="$OUT"
echo "wall time $(( $(date +%s) - start )) s"
ls -la "$OUT"
