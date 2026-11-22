#!/usr/bin/env bash
# LIBERO latency-vs-success runs (workloads/libero_eval.py) on this CIMS host. Each argument is
# one GPU's job list, run in order; the GPUs run in parallel. A job is
#   policy:precision:delays    e.g. pi0:fp16:0,2,5
# Waits until no timing stage (run_cims_host.sh) is running on this host, so success runs never
# overlap latency measurements. Results append to results/libero/<policy>_<prec>_<host>_gpu<i>.csv.
#   WORK=/scratch/$USER scripts/run_libero.sh "0=smolvla:fp16:0,2,5" "1=smolvla:fp16:10,15,20"
set -uo pipefail
cd "$(dirname "$0")/.."
: "${WORK:?set WORK}"
SUITE=${SUITE:-libero_10}
TASKS=${TASKS:-0-9}
EPISODES=${EPISODES:-10}
EXEC_HORIZON=${EXEC_HORIZON:-0}   # 0 = the policy's n_action_steps; 10 = the published protocol
HTAG=""
[[ $EXEC_HORIZON -gt 0 ]] && HTAG="_h$EXEC_HORIZON"
HOST=$(hostname -s)
export HF_HOME="$WORK/.cache/huggingface" CUDA_MODULE_LOADING=EAGER MUJOCO_GL=egl
while pgrep -u "$USER" -f "[r]un_cims_host.sh|[c]ims_final.sh" > /dev/null; do sleep 60; done
echo "[$(date +%T)] $HOST: starting LIBERO jobs"
mkdir -p results/libero logs

gpu_jobs() {  # gpu, job list
  local gpu=$1 jobs=$2 job pol prec delays py node pin
  node=$(cat /sys/bus/pci/devices/0000:$(nvidia-smi -i "$gpu" --query-gpu=pci.bus_id \
    --format=csv,noheader | cut -d: -f2-3 | tr 'A-F' 'a-f')/numa_node 2>/dev/null || echo -1)
  pin=()
  command -v numactl > /dev/null && [[ $node -ge 0 ]] && pin=(numactl --cpunodebind="$node" --membind="$node")
  for job in ${jobs//;/ }; do
    IFS=: read -r pol prec delays <<< "$job"
    py="$WORK/envs/pi/bin/python"
    [[ $pol == smolvla ]] && py="$WORK/envs/smolvla/bin/python"
    local repo
    repo=$("$py" -c "import sys; sys.path.insert(0, 'workloads'); import libero_eval as L; print(L.REPOS['$pol'])")
    "$py" -c "from huggingface_hub import snapshot_download as d; d('$repo')" > /dev/null 2>&1
    echo "[$(date +%T)] GPU $gpu: $pol $prec delays $delays"
    CUDA_VISIBLE_DEVICES=$gpu MUJOCO_EGL_DEVICE_ID=$gpu HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      "${pin[@]}" "$py" workloads/libero_eval.py --policy "$pol" --precision "$prec" \
      --suite "$SUITE" --tasks "$TASKS" --episodes "$EPISODES" --delays "$delays" \
      --exec-horizon "$EXEC_HORIZON" \
      --out "results/libero/${pol}_${prec}${HTAG}_${HOST}_gpu${gpu}.csv" \
      >> "logs/$HOST-libero-$pol-$prec$HTAG-gpu$gpu.log" 2>&1 \
      || echo "[$(date +%T)] FAILED: $pol $prec $delays (see logs/$HOST-libero-$pol-$prec-gpu$gpu.log)"
  done
  echo "[$(date +%T)] GPU $gpu done"
}

for spec in "$@"; do
  gpu_jobs "${spec%%=*}" "${spec#*=}" &
done
wait
echo "[$(date +%T)] $HOST: all LIBERO jobs done"
