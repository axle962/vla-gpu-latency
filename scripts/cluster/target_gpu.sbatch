#!/usr/bin/env bash
# Everything model/cross_gpu.py needs from a new target GPU, in one job (one node allocation):
#   calib    C++ microbenchmarks (launch latency, bandwidth/L2 sweep, device info)
#   dispatch PyTorch per-op dispatch tables (main + conv) and host/cold-cache probes, written
#            into the same results/calib/<gpu>_<host>_<time>/ directory
#   bench    our CUDA kernels, the C++ replay of the SmolVLA step, sustained SM clock
#   trace    SmolVLA sweep (7 configs, as on every other GPU) + a baseline repeat
#   pi       pi0 and pi0.5 traces, FP16 and BF16 (needs envs/pi and the weights in $HF_HOME)
#   place    hosting placements on this GPU and its host CPU (workloads/placement.py)
#   pipe     pipelined requests on one GPU and across two (needs --gres=gpu:2): a toy-model
#            check of the stream ordering first, then SmolVLA (placements gpu and gpu-gpu)
# Stages are the arguments (default: calib dispatch bench trace). Example on NYU Torch:
#   sbatch -A <account> -p <partition> --gres=gpu:1 scripts/cluster/target_gpu.sbatch
#   sbatch -A <account> -p <partition> --gres=gpu:1 scripts/cluster/target_gpu.sbatch pi place
# Build first on the login node (sm_80/89/90 for A100, L40S, H100/H200; CUDA >= 12.0):
#   make -C bench ARCHS="80 89 90" BUILD=build-hpc
# Cost estimate per GPU: calib 5 min, dispatch 5 min, bench 8 min, trace 35 min, pi 30 min,
# place 25 min (CPU halves), pipe 10 min on 2 GPUs; default stages about 1 GPU-hour.
#SBATCH --job-name=target-gpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --no-requeue
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
STAGES=("$@")
[[ ${#STAGES[@]} -gt 0 ]] || STAGES=(calib dispatch bench trace)
BUILD=${BUILD:-bench/build-hpc}
WORK=${WORK:-/scratch/$USER}
PY=${PY:-$WORK/envs/smolvla/bin/python}
PYPI=${PYPI:-$WORK/envs/pi/bin/python}
export HF_HOME=${HF_HOME:-$WORK/.cache/huggingface} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_MODULE_LOADING=EAGER
mkdir -p logs
echo "job ${SLURM_JOB_ID:-?} on $(hostname) partition ${SLURM_JOB_PARTITION:-?}, stages: ${STAGES[*]}"
grep -m1 "model name" /proc/cpuinfo
# Record the GPU state before anything runs (a GPU idling in P0 may be stuck at a fixed clock).
nvidia-smi --query-gpu=index,name,pci.bus_id,pstate,clocks.sm,clocks.max.sm,temperature.gpu,power.draw,memory.used,driver_version --format=csv
cc=$(nvidia-smi -i 0 --query-gpu=compute_cap --format=csv,noheader | tr -d '. ')
fail=0
run() {  # label, command...
  local label=$1; shift
  echo "=== $label ($(date +%T))"
  "$@" || { echo "FAILED: $label"; fail=$((fail + 1)); }
}
latest_calib() { ls -dt results/calib/*_"$(hostname -s)"_* 2>/dev/null | head -1; }

for s in "${STAGES[@]}"; do
  case $s in
  calib)
    run calib scripts/run_calib.sh "$BUILD"
    ;;
  dispatch)
    out=$(latest_calib)
    [[ -n $out ]] || { echo "FAILED: dispatch needs a calib dir from this host"; fail=$((fail + 1)); continue; }
    grep -m1 "model name" /proc/cpuinfo > "$out/host_cpu.txt"
    run dispatch-main sh -c "$PY workloads/torch_dispatch.py > $out/torch_dispatch_smolvla_env.csv"
    run dispatch-conv sh -c "$PY workloads/torch_dispatch.py --set conv > $out/torch_dispatch_conv_smolvla_env.csv"
    run probe sh -c "$PY workloads/torch_probe.py > $out/torch_probe_smolvla_env.csv"
    ;;
  bench)
    run bench scripts/run_bench.sh "$BUILD"
    ;;
  trace)
    for c in "--cameras 3 --num-steps 10 --chunk-size 50" \
             "--cameras 1 --num-steps 10 --chunk-size 50" \
             "--cameras 2 --num-steps 10 --chunk-size 50" \
             "--cameras 3 --num-steps 5 --chunk-size 50" \
             "--cameras 3 --num-steps 20 --chunk-size 50" \
             "--cameras 3 --num-steps 10 --chunk-size 25" \
             "--cameras 3 --num-steps 10 --chunk-size 100" \
             "--cameras 3 --num-steps 10 --chunk-size 50"; do
      log="logs/target-${SLURM_JOB_ID:-x}-$(echo "$c" | tr -d ' -')-$(date +%H%M%S).log"
      # shellcheck disable=SC2086
      run "smolvla $c" sh -c "$PY workloads/trace_smolvla.py $c > $log 2>&1 && grep -E 'max \|diff\||wrote' $log"
    done
    ;;
  pi)
    [[ -x $PYPI ]] || { echo "FAILED: no $PYPI (build envs/pi first)"; fail=$((fail + 1)); continue; }
    precs="fp16"; [[ $cc -ge 80 ]] && precs="fp16 bf16"
    for p in $precs; do
      for pol in pi0 pi05; do
        log="logs/target-${SLURM_JOB_ID:-x}-$pol-$p.log"
        run "$pol $p" sh -c "$PYPI workloads/trace_pi.py --policy $pol --precision $p > $log 2>&1 && grep -E 'max \|diff\||wrote' $log"
      done
    done
    ;;
  place)
    # One GPU here, so gpu-gpu is skipped. The CPU halves use this job's cores. Slurm gives the
    # job its GPU exclusively, so the runner's shared-GPU checks are skipped (FORCE=1).
    run placement sh -c "WORK=$WORK FORCE=1 scripts/run_placement.sh run"
    ;;
  pipecheck)
    # Short diagnostic: does the graph loop follow a new KV cache? (1 GPU, no profiling)
    run pipecheck sh -c "$PY workloads/placement.py --placements gpu --pipeline --reps 5 --pipe-windows 2 --out results/placement/pipecheck_$(date +%Y%m%d-%H%M%S)"
    ;;
  pipe)
    run pipe-smoke sh -c "WORK=$WORK FORCE=1 scripts/run_placement.sh smoke --placements gpu,gpu-gpu --pipeline"
    run pipe sh -c "WORK=$WORK FORCE=1 scripts/run_placement.sh run --placements gpu,gpu-gpu --pipeline"
    ;;
  *)
    echo "unknown stage $s"; fail=$((fail + 1))
    ;;
  esac
done
nvidia-smi --query-gpu=index,pstate,clocks.sm,temperature.gpu,power.draw --format=csv
echo "done ($(date +%T)), $fail failed"
