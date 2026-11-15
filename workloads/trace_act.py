"""Trace ACT (Action Chunking Transformer, lerobot/act_aloha_sim_transfer_cube_human, ~52M,
FP32) inference at batch 1.

A third workload with a different call mix: one forward pass, no iterative loop. ResNet18 on a
480x640 camera image (cuDNN convolutions, BatchNorm folded into eval kernels), a sinusoidal 2D
position embedding, a 4-layer transformer encoder over 1 + 1 + 300 tokens (nn.MultiheadAttention,
seq-first) and a 1-layer decoder over the 100-step action chunk. The Python side is unusual:
the 15x20 feature map is split into 300 per-token views with list() and re-stacked, so ACT
issues many Python-level tensor operations per kernel.

Timed modes (batch 1, same weights and inputs; predict_action_chunk without the
normalization processors):
  eager           stock ACT.forward in eval (latent = zeros, no VAE encoder)
  graph           the whole forward captured as one CUDA graph and replayed (nothing in it
                  needs the host, so the full step is captured; there is no eager prefix)

Capture detail: ACT builds its latent as torch.zeros(..., CPU).to(cuda), a pageable host to
device copy that is illegal during stream capture. While warming up and capturing, the module's
torch.zeros is served from a cached CUDA tensor (ZerosOnGpu), so the copy disappears from the
graph; the eager timing runs the stock code with the copy.

Usage (inside envs/smolvla, on a GPU node; the ImageNet ResNet18 weights torchvision fetches at
construction must be in $TORCH_HOME, see scripts/cluster/trace_act.sbatch):
  HF_HOME=/scratch/$USER/.cache/huggingface HF_HUB_OFFLINE=1 python workloads/trace_act.py
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trace_common import default_out, profile_and_save, profile_graph  # noqa: E402

import lerobot.policies.act.modeling_act as act_mod  # noqa: E402
from lerobot.policies.act.modeling_act import ACTPolicy  # noqa: E402
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE  # noqa: E402

MODEL_ID = "lerobot/act_aloha_sim_transfer_cube_human"


class ZerosOnGpu:
    """Stand-in for the torch module inside modeling_act: zeros(size, dtype) without a device
    returns a cached CUDA tensor, everything else is torch."""

    def __init__(self) -> None:
        self.cache: dict = {}

    def __getattr__(self, name):
        return getattr(torch, name)

    def zeros(self, *size, dtype=None, device=None, **kw):
        if device is None and not kw:
            key = (repr(size), dtype)
            if key not in self.cache:
                self.cache[key] = torch.zeros(*size, dtype=dtype, device="cuda")
            return self.cache[key]
        return torch.zeros(*size, dtype=dtype, device=device, **kw)

    @contextlib.contextmanager
    def active(self):
        act_mod.torch = self
        try:
            yield
        finally:
            act_mod.torch = torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--height", type=int, default=480, help="camera image height (stock 480)")
    ap.add_argument("--width", type=int, default=640, help="camera image width (stock 640)")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--prof-steps", type=int, default=3)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    policy = ACTPolicy.from_pretrained(MODEL_ID).cuda().eval()
    cfg = policy.config
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(0)
    img_keys = list(cfg.image_features)
    batch = {OBS_STATE: torch.rand(1, cfg.robot_state_feature.shape[0], device=dev,
                                   generator=gen)}
    batch[OBS_IMAGES] = [torch.rand(1, 3, args.height, args.width, device=dev, generator=gen)
                         for _ in img_keys]

    @torch.no_grad()
    def step():
        return policy.model(batch)[0]

    ref = step()
    torch.cuda.synchronize()

    memo = ZerosOnGpu()
    g = torch.cuda.CUDAGraph()
    graph_ok, graph_err = True, ""
    try:
        with memo.active():
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    step()
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            with torch.cuda.graph(g):
                static_out = step()
        g.replay()
        torch.cuda.synchronize()
        graph_diff = (static_out - ref).abs().max().item()
        print(f"graph-vs-eager max |diff| = {graph_diff:.3e}")
    except Exception as e:  # report and keep the eager data
        graph_ok, graph_err, graph_diff = False, f"{type(e).__name__}: {e}", float("nan")
        print("graph capture failed:", graph_err)

    tokens = 2 + (args.height // 32) * (args.width // 32)
    meta = {
        "hf_model": MODEL_ID,
        "workload": "act",
        "dtype": "fp32",
        "batch": 1,
        "cameras": len(img_keys),
        "image_input_hw": [args.height, args.width],
        "encoder_tokens": tokens,
        "num_steps": 1,
        "chunk_size": cfg.chunk_size,
        "params": sum(p.numel() for p in policy.parameters()),
        "graph_ok": graph_ok,
        "graph_error": graph_err,
        "graph_vs_eager_max_abs_diff": graph_diff,
        "graph_scope": "full step",
        "tf32": False,
    }
    out = args.out or default_out(f"act_res{args.height}x{args.width}_chunk{cfg.chunk_size}")
    profile_and_save(out, "act", step, meta, warmup=args.warmup, reps=args.reps,
                     prof_steps=args.prof_steps, graph_step_fn=g.replay if graph_ok else None)
    if graph_ok:
        profile_graph(out, g.replay, n=args.prof_steps)
    print("wrote", out)


if __name__ == "__main__":
    main()
