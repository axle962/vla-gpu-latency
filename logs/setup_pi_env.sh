#!/usr/bin/env bash
# Build envs/pi (lerobot 0.4.4 + transformers fix/lerobot_openpi, same torch as envs/smolvla)
# and fetch the tokenizers once online. Models: all three on hosts with room, none on cuda3
# (24 GB free; fetched one at a time by the trace stage).
set -euo pipefail
W=/scratch/$USER
export UV_CACHE_DIR=$W/.cache/uv UV_PYTHON_INSTALL_DIR=$W/.uv-python HF_HOME=$W/.cache/huggingface
cd $W/envs
[ -x pi/bin/python ] || ~/.local/bin/uv venv -q -p 3.10 pi
tv=$($W/envs/smolvla/bin/python -c "import torch;print(torch.__version__.split('+')[0])")
vv=$($W/envs/smolvla/bin/python -c "import torchvision;print(torchvision.__version__.split('+')[0])")
VIRTUAL_ENV=$W/envs/pi ~/.local/bin/uv pip install -q "lerobot==0.4.4" "torch==$tv" "torchvision==$vv" \
  "transformers @ git+https://github.com/huggingface/transformers.git@fix/lerobot_openpi" \
  pandas nvidia-ml-py tabulate scipy --index-strategy unsafe-best-match
$W/envs/pi/bin/python - <<'PY'
from huggingface_hub import snapshot_download as d
from transformers import AutoTokenizer, AutoProcessor
for r in ("physical-intelligence/fast", "jadechoghari/fast-libero-tokenizer-mean-std"):
    d(r); AutoProcessor.from_pretrained(r, trust_remote_code=True)
AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224", trust_remote_code=True, add_eos_token=True, add_bos_token=False)
import torch, transformers; print("env ok", torch.__version__, transformers.__version__)
PY
