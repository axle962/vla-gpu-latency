#!/usr/bin/env bash
# Feasibility check for RoboTwin 2.0 on a cluster GPU node (no installs, ~2 min):
#   source scripts/cluster/site.sh
#   sbatch -p c12m85-a100-1 scripts/cluster/robotwin_check.sbatch
# RoboTwin's simulator (SAPIEN) renders with Vulkan, so the node needs NVIDIA's Vulkan ICD; the
# fallback route is an Apptainer/Singularity container with --nv. Prints what is available;
# read logs/robotwin-check-<job>.out. Cost: ~0.05 GPU-h (plus boot).
#SBATCH --job-name=robotwin-check
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --no-requeue
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
echo "job ${SLURM_JOB_ID:-?} on $(hostname) partition ${SLURM_JOB_PARTITION:-?}"
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv
grep -m1 "model name" /proc/cpuinfo; free -g | head -2; df -h /scratch/"$USER" | tail -1

echo "== Vulkan ICD files (need an NVIDIA one, e.g. nvidia_icd.json)"
ls -l /usr/share/vulkan/icd.d/ /etc/vulkan/icd.d/ 2>&1
echo "== NVIDIA graphics libraries from the driver"
ls /usr/lib64/libGLX_nvidia.so* /usr/lib/x86_64-linux-gnu/libGLX_nvidia.so* \
   /usr/lib64/libEGL_nvidia.so* /usr/lib/x86_64-linux-gnu/libEGL_nvidia.so* 2>&1 | head
ldconfig -p | grep -E "libvulkan.so|libEGL_nvidia|libGLX_nvidia" || echo "none in ldconfig"
echo "== vulkaninfo"
if command -v vulkaninfo > /dev/null; then
  timeout 60 vulkaninfo --summary 2>&1 | grep -E "deviceName|driverName|apiVersion|ERROR" | head
else
  echo "vulkaninfo not installed (check the ICD files above instead)"
fi
echo "== container runtimes"
for c in apptainer singularity podman docker; do printf "%-12s %s\n" "$c" "$(command -v $c || echo none)"; done
echo "== CUDA modules / conda"
(source /etc/profile.d/modules.sh 2>/dev/null; module avail 2>&1 | grep -ioE "cuda[-/][0-9.]+" | sort -u | tr '\n' ' '); echo
ls /scratch/"$USER"/.tools 2>/dev/null
echo "== EGL headless (MuJoCo-style) probe"
python3 - <<'PY' 2>&1 | tail -3
import ctypes
for lib in ("libEGL_nvidia.so.0", "libvulkan.so.1"):
    try:
        ctypes.CDLL(lib); print(lib, "loads")
    except OSError as e:
        print(lib, "missing:", e)
PY
echo "done"
