"""Trace Diffusion Policy (lerobot/diffusion_pusht, ~263M, FP32) inference at batch 1.

A second workload with a different call mix from SmolVLA: a ResNet18 image encoder (cuDNN
convolutions, GroupNorm, spatial softmax) runs once per step on n_obs_steps = 2 frames, then a
1D-convolution UNet (Conv1d + GroupNorm + Mish + FiLM, down dims 512/1024/2048 over a
16-step horizon) runs once per DDPM denoising step, followed by the DDPM scheduler update
(scalar CPU math, a few elementwise kernels).

One "step" is one action-chunk inference (DiffusionModel.generate_actions without the
observation queue and the normalization processors):
  eager           _prepare_global_conditioning + conditional_sample (stock lerobot code)
  eager_prefix    image/state conditioning only (ResNet18 encoder)
  eager_denoise   the DDPM loop only, from a fixed conditioning vector
  graph_denoise   the same loop captured as one CUDA graph and replayed
  hybrid          eager prefix + copy into the graph's static input + graph replay

Variance noise: DDPMScheduler.step draws fresh Gaussian noise every step (randn_tensor). To make
eager, split and graph outputs comparable bit for bit, the noise is pre-sampled once (fixed seed)
and served in order by a patched randn_tensor in every mode, so all modes run the same kernels
(the per-step randn kernel is replaced by a buffer read). Same idea as CudaConstMemo for SmolVLA.

Precision is the checkpoint's own FP32 (use_amp false); TF32 is disabled so every GPU runs the
same FP32 math. That makes the GEMM/conv kernels SIMT FP32, unlike SmolVLA's FP16 tensor cores.

Usage (inside envs/smolvla, on a GPU node):
  HF_HOME=/scratch/$USER/.cache/huggingface HF_HUB_OFFLINE=1 \\
    python workloads/trace_diffusion.py --num-steps 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trace_common import default_out, profile_and_save, profile_graph  # noqa: E402

import diffusers.schedulers.scheduling_ddpm as ddpm_mod  # noqa: E402
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy  # noqa: E402
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE  # noqa: E402

MODEL_ID = "lerobot/diffusion_pusht"


class FixedNoise:
    """Replaces diffusers' randn_tensor in the DDPM scheduler with pre-sampled noise, served
    in call order; reset() before each denoising loop."""

    def __init__(self, n: int, shape: tuple, device: torch.device, seed: int = 1) -> None:
        g = torch.Generator(device=device).manual_seed(seed)
        self.bufs = [torch.randn(shape, device=device, generator=g) for _ in range(n)]
        self.i = 0
        ddpm_mod.randn_tensor = self._randn

    def reset(self) -> None:
        self.i = 0

    def _randn(self, shape, generator=None, device=None, dtype=None, **kw):
        b = self.bufs[self.i]
        self.i += 1
        assert tuple(shape) == tuple(b.shape) and (dtype is None or dtype == b.dtype)
        return b


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-steps", type=int, default=100, help="DDPM inference steps (stock 100)")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--prof-steps", type=int, default=3)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    policy = DiffusionPolicy.from_pretrained(MODEL_ID).cuda().eval()
    dm = policy.diffusion
    cfg = policy.config
    dm.num_inference_steps = args.num_steps
    dev = torch.device("cuda")

    gen = torch.Generator(device=dev).manual_seed(0)
    n_obs = cfg.n_obs_steps
    img_key = list(cfg.image_features)[0]
    c, h, w = cfg.image_features[img_key].shape
    n_cam = len(cfg.image_features)
    batch = {
        OBS_STATE: torch.rand(1, n_obs, cfg.robot_state_feature.shape[0], device=dev,
                              generator=gen),
        OBS_IMAGES: torch.rand(1, n_obs, n_cam, c, h, w, device=dev, generator=gen),
    }
    act_dim = cfg.action_feature.shape[0]
    noise = torch.randn(1, cfg.horizon, act_dim, device=dev, generator=gen)
    fixed = FixedNoise(args.num_steps, (1, cfg.horizon, act_dim), dev)

    @torch.no_grad()
    def prefix():
        return dm._prepare_global_conditioning(batch)

    @torch.no_grad()
    def denoise(gc, x0):
        fixed.reset()
        return dm.conditional_sample(1, global_cond=gc, noise=x0)

    @torch.no_grad()
    def full_step():
        fixed.reset()
        return dm.generate_actions(batch, noise=noise)

    gc0 = prefix()
    torch.cuda.synchronize()
    ref = full_step()
    split = denoise(gc0, noise)
    s0, e0 = n_obs - 1, n_obs - 1 + cfg.n_action_steps
    max_diff = (ref - split[:, s0:e0]).abs().max().item()
    print(f"split-vs-stock max |diff| = {max_diff:.3e}")

    static_gc = gc0.clone()
    static_x = noise.clone()
    g = torch.cuda.CUDAGraph()
    graph_ok, graph_err = True, ""
    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                denoise(static_gc, static_x)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        with torch.cuda.graph(g):
            static_out = denoise(static_gc, static_x)
        g.replay()
        torch.cuda.synchronize()
        graph_diff = (static_out - split).abs().max().item()
        print(f"graph-vs-eager denoise max |diff| = {graph_diff:.3e}")
    except Exception as e:  # report and keep the eager data
        graph_ok, graph_err, graph_diff = False, f"{type(e).__name__}: {e}", float("nan")
        print("graph capture failed:", graph_err)

    def hybrid():
        static_gc.copy_(prefix())
        g.replay()

    extra = {"eager_prefix": prefix, "eager_denoise": lambda: denoise(gc0, noise)}
    if graph_ok:
        extra["graph_denoise"] = g.replay
        extra["hybrid"] = hybrid

    meta = {
        "hf_model": MODEL_ID,
        "workload": "diffusion",
        "dtype": "fp32",
        "batch": 1,
        "cameras": n_cam,
        "n_obs_steps": n_obs,
        "image_input_hw": [h, w],
        "crop_shape": list(cfg.crop_shape or []),
        "num_steps": args.num_steps,
        "chunk_size": cfg.horizon,
        "n_action_steps": cfg.n_action_steps,
        "params": sum(p.numel() for p in policy.parameters()),
        "split_vs_stock_max_abs_diff": max_diff,
        "graph_ok": graph_ok,
        "graph_error": graph_err,
        "graph_vs_eager_max_abs_diff": graph_diff,
        "graph_scope": "denoising loop only (prefix eager)",
        "variance_noise": "pre-sampled (FixedNoise)",
        "tf32": False,
    }
    out = args.out or default_out(f"diffusion_steps{args.num_steps}_h{cfg.horizon}")
    profile_and_save(out, "diffusion", full_step, meta, warmup=args.warmup, reps=args.reps,
                     prof_steps=args.prof_steps, extra_modes=extra)
    if graph_ok:
        profile_graph(out, g.replay, n=args.prof_steps)
    print("wrote", out)


if __name__ == "__main__":
    main()
