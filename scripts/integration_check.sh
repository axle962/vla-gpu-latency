#!/usr/bin/env bash
# Integration check of the combined work (Aditya + Archana + Shrikar) on a CIMS cudaN host,
# from a fresh clone of the `integration` branch. Three stages, run in this order:
#
#   cpp     build the CUDA C++ benchmarks, run calibration and the C++ bench (correctness
#           checks of our kernels vs cuBLAS / CPU reference). ~10 min, idle GPU.
#   models  rerun every analysis/model script on the committed results (no GPU), then show
#           which tracked outputs changed. Identical outputs = the combined code reproduces
#           the committed numbers. ~10 min.
#   smoke   one short run of every workload runner (SmolVLA, ACT, Diffusion, pi0 if envs/pi
#           exists, placement, multi-policy, eps probe) on an idle GPU; each must finish and
#           print graph-vs-eager max |diff|. Outputs go to $WORK/integration_smoke, not traces/.
#           ~30 min.
#
# Usage (cuda2):
#   module load cuda-12.4
#   export WORK=/scratch/$USER          # holds envs/smolvla, envs/pi, .cache/huggingface
#   git clone -b integration git@github.com:axle962/vla-gpu-latency.git $WORK/integration
#   cd $WORK/integration && scripts/integration_check.sh cpp models smoke
# Summary: logs/integration-<host>-<stamp>/SUMMARY.txt (one PASS/FAIL line per step).
set -uo pipefail
cd "$(dirname "$0")/.."
: "${WORK:?set WORK (directory holding envs/smolvla, not your 4 GB home)}"
HOST=$(hostname -s)
STAMP=$(date +%Y%m%d-%H%M%S)
LOG="logs/integration-$HOST-$STAMP"
mkdir -p "$LOG"
SUM="$LOG/SUMMARY.txt"
PY="$WORK/envs/smolvla/bin/python"
PYPI="$WORK/envs/pi/bin/python"
SMOKE="$WORK/integration_smoke/$STAMP"
export HF_HOME="$WORK/.cache/huggingface" TMPDIR="$WORK/tmp" CUDA_MODULE_LOADING=EAGER
mkdir -p "$TMPDIR"
source /etc/profile.d/modules.sh 2>/dev/null
module load cuda-12.4 2>/dev/null

echo "integration check on $HOST at $(git log --oneline -1)" | tee "$SUM"

step() {  # step <name> <command...>: run, log, record PASS/FAIL
  local name=$1; shift
  local f="$LOG/$name.log"
  echo "[$(date +%T)] $name" >&2
  if "$@" > "$f" 2>&1; then echo "PASS  $name" | tee -a "$SUM"
  else echo "FAIL  $name (exit $?, see $f)" | tee -a "$SUM"; fi
}

idle_gpu() {  # idle GPU, skipping consumer GPUs stuck in P0 (cuda2 GPU 0)
  local g n ps
  for g in $(nvidia-smi --query-gpu=index --format=csv,noheader); do
    n=$(nvidia-smi -i "$g" --query-compute-apps=pid --format=csv,noheader | grep -c .)
    ps=$(nvidia-smi -i "$g" --query-gpu=pstate --format=csv,noheader)
    [[ $n -eq 0 && $ps != P0 ]] && { echo "$g"; return; }
  done
  return 1
}
pin_for() {
  local node
  node=$(cat /sys/bus/pci/devices/0000:$(nvidia-smi -i "$1" --query-gpu=pci.bus_id \
    --format=csv,noheader | cut -d: -f2-3 | tr 'A-F' 'a-f')/numa_node 2>/dev/null || echo -1)
  command -v numactl > /dev/null && [[ $node -ge 0 ]] && echo "numactl --cpunodebind=$node --membind=$node"
}
need_gpu() {
  GPU=$(idle_gpu) || { echo "no idle GPU (other processes or P0); retry later" | tee -a "$SUM"; exit 1; }
  export CUDA_VISIBLE_DEVICES=$GPU
  PIN=$(pin_for "$GPU")
  echo "GPU $GPU ($(nvidia-smi -i "$GPU" --query-gpu=name --format=csv,noheader)) ${PIN:-unpinned}" | tee -a "$SUM"
}

for stage in "$@"; do
  echo "== $stage" | tee -a "$SUM"
  case $stage in
  cpp)
    need_gpu
    CC=$(nvidia-smi -i "$GPU" --query-gpu=compute_cap --format=csv,noheader | tr -d '. ')
    B="build-integration-$HOST"
    step make make -C bench ARCHS="$CC" BUILD="$B"
    step calib $PIN scripts/run_calib.sh "bench/$B"
    step bench $PIN scripts/run_bench.sh "bench/$B"
    echo "      bench lines with fail|mismatch: $(grep -ciE 'fail|mismatch' "$LOG/bench.log")" | tee -a "$SUM"
    ;;
  models)
    [[ -x $PY ]] || { echo "no $PY" | tee -a "$SUM"; exit 1; }
    step summarize_traces sh -c "$PY model/summarize_traces.py > results/traces_summary.csv"
    step launch_model     $PY model/launch_model.py
    step cross_gpu        $PY model/cross_gpu.py
    step report_cross_gpu sh -c "$PY model/report_cross_gpu.py > results/report_cross_gpu.md"
    for w in act diffusion diffusionbench pi0 pi05 pi0fast; do
      step report_$w sh -c "$PY model/report_cross_gpu.py --workload $w > results/report_cross_gpu_$w.md"
    done
    step figures          $PY model/figures.py
    step ablation         sh -c "$PY model/ablation.py > results/ablation_calibration.md"
    step batch_predict    $PY model/batch_predict.py
    step compile_predict  $PY model/compile_predict.py
    step host_experiment  $PY model/host_experiment.py
    step rodinia_predict  $PY model/rodinia_predict.py
    step whatif           sh -c "$PY model/whatif.py > results/whatif.md"
    step libero_analysis  sh -c "$PY model/libero_analysis.py > results/libero_analysis.md"
    step robotwin_analysis sh -c "$PY model/robotwin_analysis.py > results/robotwin_analysis.md"
    # teammates' analyses
    step graph_worthiness $PY model/graph_worthiness.py
    for d in results/placement/*/; do
      [[ -f $d/latency.csv ]] && step "placement_report_$(basename "$d")" $PY model/placement_report.py "$d"
    done
    for d in results/multipolicy/*/; do
      [[ -f $d/latency.csv ]] && step "multipolicy_report_$(basename "$d")" $PY model/multipolicy_report.py "$d"
    done
    # Which tracked text outputs changed? (PDF/PNG differ by embedded timestamps: ignored.)
    git diff --stat -- . ':(exclude)*.pdf' ':(exclude)*.png' > "$LOG/models_diff_stat.txt"
    git diff -- . ':(exclude)*.pdf' ':(exclude)*.png' > "$LOG/models_diff.patch"
    n=$(git diff --name-only -- . ':(exclude)*.pdf' ':(exclude)*.png' | wc -l)
    echo "      tracked text outputs changed by the rerun: $n (see $LOG/models_diff_stat.txt)" | tee -a "$SUM"
    ;;
  smoke)
    need_gpu
    mkdir -p "$SMOKE"
    off="HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1"
    q="--warmup 2 --reps 3 --prof-steps 1"
    step smoke_smolvla   env $off $PIN $PY workloads/trace_smolvla.py $q --out "$SMOKE/smolvla"
    step smoke_act       env $off $PIN $PY workloads/trace_act.py $q --out "$SMOKE/act"
    step smoke_diffusion env $off $PIN $PY workloads/trace_diffusion.py --num-steps 10 $q --out "$SMOKE/diffusion"
    if [[ -x $PYPI ]]; then
      step smoke_pi0 env $off $PIN $PYPI workloads/trace_pi.py --policy pi0 $q --out "$SMOKE/pi0"
    else echo "SKIP  smoke_pi0 (no $PYPI)" | tee -a "$SUM"; fi
    step smoke_placement   scripts/run_placement.sh smoke
    step smoke_multipolicy scripts/run_multipolicy.sh smoke
    step smoke_eps_probe   env PY="$PY" scripts/run_eps_probe.sh
    grep -hE "max \|diff\|" "$LOG"/smoke_*.log | sed 's/^/      /' | tee -a "$SUM"
    ;;
  *) echo "unknown stage $stage (cpp, models, smoke)" | tee -a "$SUM" ;;
  esac
done
echo "[$(date +%T)] done; summary in $SUM" | tee -a "$SUM"
