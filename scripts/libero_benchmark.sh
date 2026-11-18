#!/usr/bin/env bash
# Reproduce the published LIBERO protocol with the official lerobot-eval (huggingface.co/docs/
# lerobot/libero, "Reproducing published results"): --policy.n_action_steps=10 (OpenPI's
# replanning interval), 10 episodes per task, batch size 1, one task at a time, stock precision.
# Each argument is policy_repo[:extra flag;extra flag]. Results (eval_info.json, no videos) are
# copied to results/libero_benchmark/<suite>/<repo name>_<host>/.
#   WORK=/scratch/$USER CUDA_VISIBLE_DEVICES=0 scripts/libero_benchmark.sh lerobot/pi05_libero_finetuned
set -uo pipefail
cd "$(dirname "$0")/.."
: "${WORK:?set WORK}"
SUITE=${SUITE:-libero_10}
EPISODES=${EPISODES:-10}
NTASKS=${NTASKS:-10}
HOST=$(hostname -s)
DEV=${CUDA_VISIBLE_DEVICES:-0}
export HF_HOME="$WORK/.cache/huggingface" MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$DEV CUDA_MODULE_LOADING=EAGER
node=$(cat /sys/bus/pci/devices/0000:$(nvidia-smi -i "$DEV" --query-gpu=pci.bus_id \
  --format=csv,noheader | cut -d: -f2-3 | tr 'A-F' 'a-f')/numa_node 2>/dev/null || echo -1)
pin=()
command -v numactl > /dev/null && [[ $node -ge 0 ]] && pin=(numactl --cpunodebind="$node" --membind="$node")
mkdir -p logs results/libero_benchmark/"$SUITE"
for spec in "$@"; do
  repo=${spec%%:*}
  extra=()
  [[ $spec == *:* ]] && IFS=';' read -ra extra <<< "${spec#*:}"
  name=${repo##*/}
  py="$WORK/envs/pi/bin"
  [[ $name == smolvla* ]] && py="$WORK/envs/smolvla/bin"
  "$py/python" -c "from huggingface_hub import snapshot_download as d; d('$repo')" > /dev/null 2>&1 \
    || { echo "FAILED: download $repo"; continue; }
  dst="results/libero_benchmark/$SUITE/${name}_$HOST"
  mkdir -p "$dst"
  # One task per lerobot-eval call: building all tasks of a suite at once creates one EGL
  # rendering context per task on the GPU (~6 GB for LIBERO-10), which with a 7.5 GB policy
  # overflows 12 GB GPUs. Same protocol otherwise; per-task results are merged below.
  for tid in $(seq 0 $((NTASKS - 1))); do
    out="$WORK/tmp/libero_benchmark/$SUITE/${name}_$HOST/task$tid"
    rm -rf "$out"
    echo "[$(date +%T)] $repo task $tid on GPU $DEV ${pin[*]:-} ${extra[*]:-}"
    (cd "$WORK/tmp" && HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "${pin[@]}" "$py/lerobot-eval" \
       --output_dir="$out" --env.type=libero --env.task="$SUITE" "--env.task_ids=[$tid]" \
       --eval.batch_size=1 --eval.n_episodes="$EPISODES" --policy.path="$repo" \
       --policy.n_action_steps=10 --env.max_parallel_tasks=1 --policy.device=cuda "${extra[@]}") \
       >> "logs/$HOST-libero-benchmark-$name.log" 2>&1 \
      || echo "[$(date +%T)] FAILED: $repo task $tid (see logs/$HOST-libero-benchmark-$name.log)"
    f=$(find "$out" -name "eval_info.json" | head -1)
    [[ -n $f ]] && cp "$f" "$dst/eval_info_task$tid.json"
  done
  python3 - "$dst" "$name" <<'PY'
import json, glob, sys
k = n = 0
for f in sorted(glob.glob(sys.argv[1] + "/eval_info_task*.json")):
    for t in json.load(open(f)).get("per_task", []):
        s = t["metrics"]["successes"]; k += sum(s); n += len(s)
print(f"{sys.argv[2]}: {k}/{n} = {100 * k / max(n, 1):.1f}% success")
PY
done
echo "[$(date +%T)] benchmark runs done"
