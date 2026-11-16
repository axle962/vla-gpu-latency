#!/usr/bin/env bash
# Build the RoboTwin 2.0 environment natively (no Docker on the cluster), following the
# robotwin_native stage of lerobot's docker/Dockerfile.sim (lerobot main, 2026-11-16):
# CUDA 12.8 nvcc, torch 2.7.1, pinned RoboTwin commit + assets, CuRobo v0.7.8 (sm_80 only),
# pytorch3d, RoboTwin's mplib/SAPIEN patches. Unlike the Dockerfile, the policy side
# (lerobot[smolvla]==0.7.0, which accepts torch>=2.7) goes into the same venv, so lerobot-eval
# runs the simulator in-process and no env server is needed.
#   source scripts/cluster/site.sh
#   sbatch -p n2c48m24 scripts/cluster/robotwin_build.sbatch     # CPU partition: no GPU-hours
# Each stage writes a marker in envs/.robotwin_stages, so a rerun resumes after a failure.
#SBATCH --job-name=robotwin-build
#SBATCH --cpus-per-task=24
#SBATCH --mem=22G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --no-requeue
#SBATCH --exclude=b-10-119
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
REPO=$PWD
echo "job ${SLURM_JOB_ID:-?} on $(hostname) $(nproc) cpus"
curl -sfI https://pypi.org > /dev/null && echo "internet ok" || { echo "NO INTERNET on this node"; exit 1; }

export UV_CACHE_DIR=/scratch/$USER/.cache/uv HF_HOME=/scratch/$USER/.cache/huggingface
export MAX_JOBS=${MAX_JOBS:-6}
CUDA=/scratch/$USER/.tools/cuda128
ENV=$REPO/envs/robotwin
RT=/scratch/$USER/ext/RoboTwin
ROBOTWIN_SHA=0aeea2d669c0f8516f4d5785f0aa33ba812c14b4
ST=$REPO/envs/.robotwin_stages; mkdir -p "$ST"
done_() { [ -f "$ST/$1" ]; }
mark() { touch "$ST/$1"; echo "== stage $1 done $(date +%T)"; }

if [ ! -x "$CUDA/bin/nvcc" ]; then
  /share/apps/pyenv/py3.9/bin/conda create -y -q -p "$CUDA" -c nvidia/label/cuda-12.8.1 \
    cuda-nvcc cuda-cudart-dev cuda-cudart-static
fi
export CUDA_HOME=$CUDA PATH="$CUDA/bin:$PATH"
nvcc --version | tail -2

if [ ! -x "$ENV/bin/python" ]; then
  uv venv -p 3.12 "$ENV"
fi
export VIRTUAL_ENV=$ENV PATH="$ENV/bin:$PATH"

if ! done_ base; then
  uv pip install "numpy==2.2.6" "gymnasium==1.4.0" pyyaml termcolor
  uv pip install "torch==2.7.1" "torchvision==0.22.1" "setuptools<82" av scipy ninja
  uv pip install "sapien==3.0.0b1" "mplib==0.2.1" "transforms3d==0.4.2" "trimesh==4.4.3" \
    "opencv-python-headless==4.10.0.84" "open3d==0.19.0" "imageio==2.34.2" termcolor zarr \
    pydantic h5py
  mark base
fi

if ! done_ robotwin_src; then
  [ -d "$RT/.git" ] || git clone https://github.com/RoboTwin-Platform/RoboTwin.git "$RT"
  git -C "$RT" checkout "$ROBOTWIN_SHA"
  mark robotwin_src
fi

if ! done_ pytorch3d; then
  uv pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable"
  mark pytorch3d
fi

if ! done_ curobo; then
  [ -d "$RT/envs/curobo" ] || git clone --branch v0.7.8 --depth 1 https://github.com/NVlabs/curobo.git "$RT/envs/curobo"
  (cd "$RT/envs/curobo" && TORCH_CUDA_ARCH_LIST="8.0" uv pip install -e . "warp-lang==1.12.1" --no-build-isolation)
  mark curobo
fi

if ! done_ patches; then
  python - << 'EOF'
import pathlib, re, site
for d in site.getsitepackages():
    p = pathlib.Path(d) / "mplib" / "planner.py"
    if p.exists():
        p.write_text(re.sub(r"\bor collide\b", "", p.read_text(), count=1))
        print(f"mplib patch applied: {p}")
    p = pathlib.Path(d) / "sapien" / "wrapper" / "urdf_loader.py"
    if p.exists():
        src = p.read_text().replace(
            "with open(srdf_path) as f:", 'with open(srdf_path, encoding="utf-8") as f:'
        ).replace('"srdf"', '".srdf"')
        p.write_text(src)
        print(f"sapien patch applied: {p}")
EOF
  mark patches
fi

if ! done_ assets; then
  uv pip install "huggingface-hub>=1.31,<3"
  ROBOTWIN_ROOT=$RT python - << 'EOF'
import os, pathlib, zipfile
from huggingface_hub import hf_hub_download
assets_dir = pathlib.Path(os.environ["ROBOTWIN_ROOT"]) / "assets"
assets_dir.mkdir(parents=True, exist_ok=True)
for fname in ("embodiments.zip", "objects.zip"):  # ~220 MB + ~3.7 GB; backgrounds (~11 GB) skipped
    local = hf_hub_download(repo_id="TianxingChen/RoboTwin2.0", repo_type="dataset",
                            revision="3dc3b798668feb99ac61cc9086d84cbcc3d79186",
                            filename=fname, local_dir=str(assets_dir))
    with zipfile.ZipFile(local, "r") as z:
        z.extractall(str(assets_dir))
    pathlib.Path(local).unlink()
    print("extracted", fname)
EOF
  (cd "$RT" && python script/update_embodiment_config_path.py)
  mark assets
fi

if ! done_ lerobot; then
  # Policy side in the same venv; keep RoboTwin's torch 2.7.1 (lerobot 0.7.0 allows >=2.7).
  printf 'torch==2.7.1\ntorchvision==0.22.1\nnumpy==2.2.6\n' > "$ENV/overrides.txt"
  uv pip install --override "$ENV/overrides.txt" "lerobot[smolvla]==0.7.0" "eclipse-zenoh==1.10.1" msgpack
  mark lerobot
fi

echo "== import check (no GPU on this node: CuRobo's planner is not constructed)"
cd "$RT"
PYTHONPATH=$RT python -c "import sapien.core, mplib, open3d, cv2, pytorch3d, trimesh, warp; warp.torch; \
from curobo.types.math import Pose; import lerobot, torch; \
from lerobot.envs.configs import EnvConfig; print('imports ok, lerobot', lerobot.__version__, 'torch', torch.__version__)"
du -sh "$ENV" "$RT" 2> /dev/null
echo "build done"
