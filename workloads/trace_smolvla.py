"""Trace SmolVLA (lerobot/smolvla_base, ~450M) inference at batch 1.

One "step" is one action-chunk inference: image preprocessing, the VLM prefix pass that fills
the KV cache (SigLIP + 16 SmolVLM layers), then num_steps flow-matching denoising steps through
the action expert. Tokenization happens once on the CPU and is not timed.

Timed modes (all batch 1, same weights and inputs):
  eager           stock lerobot step (prepare_images + prepare_state + sample_actions)
  eager_prefix    prefix pass only (images -> KV cache)
  eager_denoise   denoising loop only, from a fixed KV cache
  graph_denoise   the same denoising loop captured as one CUDA graph and replayed
  hybrid          eager prefix + copy KV into the graph's static cache + graph replay
                  (what a deployment would run)

Why the full step is not captured: the SigLIP/Idefics3 vision embedding computes
torch.arange(mask.sum()), a data-dependent host sync, and several torch.tensor(list,
device=cuda) constant uploads sit inside the step. The denoising loop only has the constant
uploads, which are memoized on the GPU during warmup and capture (they are constants, so the
values are unchanged).

Precision: --dtype fp16 casts the VLM and the action expert to FP16 (runs on the RTX 2080 Ti,
which has no BF16). The small projection heads (state_proj, action_in/out_proj,
action_time_mlp) stay FP32 as in stock lerobot. --dtype stock keeps lerobot's default
(VLM BF16, expert FP32); use it only on Ampere and newer.

Usage (inside envs/smolvla, on a GPU node):
  HF_HOME=/scratch/$USER/.cache/huggingface HF_HUB_OFFLINE=1 \\
    python workloads/trace_smolvla.py --cameras 3 --num-steps 10
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trace_common import default_out, profile_and_save  # noqa: E402

from lerobot.policies.smolvla.modeling_smolvla import (  # noqa: E402
    SmolVLAPolicy,
    make_att_2d_masks,
)
from lerobot.utils.constants import (  # noqa: E402
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)

MODEL_ID = "lerobot/smolvla_base"
INSTRUCTION = "Pick up the red cube and place it in the bowl."


class CudaConstMemo:
    """While active, torch.tensor(python scalar/list, device=cuda) returns a cached GPU tensor
    instead of uploading from pageable host memory (illegal during stream capture). Only
    literal constants are cached, keyed by value, dtype and device."""

    def __init__(self) -> None:
        self.cache: dict[tuple, torch.Tensor] = {}
        self._orig = torch.tensor

    def _tensor(self, data, *args, dtype=None, device=None, **kw):
        if (
            not args
            and not kw
            and device is not None
            and torch.device(device).type == "cuda"
            and isinstance(data, (bool, int, float, list, tuple))
        ):
            key = (repr(data), dtype, str(device))
            t = self.cache.get(key)
            if t is None:
                t = self._orig(data, dtype=dtype, device=device)
                self.cache[key] = t
            return t
        return self._orig(data, *args, dtype=dtype, device=device, **kw)

    @contextlib.contextmanager
    def active(self):
        torch.tensor = self._tensor
        try:
            yield
        finally:
            torch.tensor = self._orig


def build(args):
    policy = SmolVLAPolicy.from_pretrained(MODEL_ID)
    cfg = policy.config
    cfg.num_steps = args.num_steps
    cfg.chunk_size = args.chunk_size
    cfg.n_action_steps = args.chunk_size
    policy = policy.cuda().eval()
    if args.dtype == "fp16":
        policy.model.vlm_with_expert.to(torch.float16)
    # Special image tokens are plain CPU tensors moved with .to(device) every step; keep them
    # resident (the .to() becomes a no-op) so the prefix has no extra H2D copies in any mode.
    m = policy.model
    m.global_image_start_token = m.global_image_start_token.cuda()
    m.image_end_token = m.image_end_token.cuda()

    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(0)
    img_keys = list(cfg.image_features)[: args.cameras]
    batch = {}
    for k in img_keys:
        c, h, w = cfg.image_features[k].shape
        batch[k] = torch.rand(1, c, h, w, device=dev, generator=gen)
    state_dim = cfg.robot_state_feature.shape[0]
    batch[OBS_STATE] = torch.rand(1, state_dim, device=dev, generator=gen)
    tok = m.vlm_with_expert.processor.tokenizer
    enc = tok(
        INSTRUCTION + "\n",
        padding="max_length",
        max_length=cfg.tokenizer_max_length,
        truncation=True,
        return_tensors="pt",
    )
    batch[OBS_LANGUAGE_TOKENS] = enc["input_ids"].to(dev)
    batch[OBS_LANGUAGE_ATTENTION_MASK] = enc["attention_mask"].to(dev).bool()
    noise = torch.randn(
        1, cfg.chunk_size, cfg.max_action_dim, device=dev, generator=gen, dtype=torch.float32
    )
    return policy, batch, noise, img_keys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cameras", type=int, default=3, help="1 to 3 camera views")
    ap.add_argument("--num-steps", type=int, default=10, help="flow-matching steps")
    ap.add_argument("--chunk-size", type=int, default=50, help="action chunk length")
    ap.add_argument("--dtype", choices=["fp16", "stock"], default="fp16")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--prof-steps", type=int, default=3)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    policy, batch, noise, img_keys = build(args)
    m = policy.model
    cfg = policy.config

    @torch.no_grad()
    def prefix():
        images, img_masks = policy.prepare_images(batch)
        state = policy.prepare_state(batch)
        embs, pad, att = m.embed_prefix(
            images, img_masks, batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK],
            state=state,
        )
        att2d = make_att_2d_masks(pad, att)
        pos = torch.cumsum(pad, dim=1) - 1
        _, kv = m.vlm_with_expert.forward(
            attention_mask=att2d, position_ids=pos, past_key_values=None,
            inputs_embeds=[embs, None], use_cache=cfg.use_cache, fill_kv_cache=True,
        )
        return pad, kv

    @torch.no_grad()
    def full_step():
        images, img_masks = policy.prepare_images(batch)
        state = policy.prepare_state(batch)
        return m.sample_actions(
            images, img_masks, batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK],
            state, noise=noise,
        )

    dt = -1.0 / cfg.num_steps
    times = [
        torch.full((1,), 1.0 + s * dt, dtype=torch.float32, device="cuda")
        for s in range(cfg.num_steps)
    ]

    @torch.no_grad()
    def denoise(pad, kv, x0):
        # Same math as VLAFlowMatching.sample_actions (no RTC), timesteps precomputed.
        x_t = x0
        for t in times:
            v_t = m.denoise_step(x_t=x_t, prefix_pad_masks=pad, past_key_values=kv, timestep=t)
            x_t = x_t + dt * v_t
        return x_t

    pad0, kv0 = prefix()
    torch.cuda.synchronize()

    # Correctness: our split loop must reproduce the stock step.
    ref = full_step()
    split = denoise(pad0, kv0, noise)
    max_diff = (ref - split).abs().max().item()
    print(f"split-vs-stock max |diff| = {max_diff:.3e}")

    # Capture the denoising loop with static inputs.
    memo = CudaConstMemo()
    static_x = noise.clone()
    static_pad = pad0.clone()
    static_kv = {
        i: {k: v.clone() for k, v in layer.items()} for i, layer in kv0.items()
    }
    graph_ok, graph_err = True, ""
    g = torch.cuda.CUDAGraph()
    try:
        with memo.active():
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    denoise(static_pad, static_kv, static_x)
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            with torch.cuda.graph(g):
                static_out = denoise(static_pad, static_kv, static_x)
        g.replay()
        torch.cuda.synchronize()
        graph_diff = (static_out - split).abs().max().item()
        print(f"graph-vs-eager denoise max |diff| = {graph_diff:.3e}")
    except Exception as e:  # report and keep the eager data
        graph_ok, graph_err, graph_diff = False, f"{type(e).__name__}: {e}", float("nan")
        print("graph capture failed:", graph_err)

    def hybrid():
        pad, kv = prefix()
        static_pad.copy_(pad)
        for i, layer in kv.items():
            for k, v in layer.items():
                static_kv[i][k].copy_(v)
        g.replay()

    extra = {
        "eager_prefix": prefix,
        "eager_denoise": lambda: denoise(pad0, kv0, noise),
    }
    if graph_ok:
        extra["graph_denoise"] = g.replay
        extra["hybrid"] = hybrid

    n_kv = sum(v.numel() * v.element_size() for layer in kv0.values() for v in layer.values())
    meta = {
        "hf_model": MODEL_ID,
        "dtype": args.dtype,
        "batch": 1,
        "cameras": len(img_keys),
        "image_keys": img_keys,
        "image_input_hw": list(cfg.image_features[img_keys[0]].shape[1:]),
        "image_resized_to": list(cfg.resize_imgs_with_padding or []),
        "num_steps": cfg.num_steps,
        "chunk_size": cfg.chunk_size,
        "prefix_len": int(pad0.shape[1]),
        "kv_cache_bytes": int(n_kv),
        "params": sum(p.numel() for p in policy.parameters()),
        "split_vs_stock_max_abs_diff": max_diff,
        "graph_ok": graph_ok,
        "graph_error": graph_err,
        "graph_vs_eager_max_abs_diff": graph_diff,
        "graph_scope": "denoising loop only (prefix eager)",
        "tf32": False,
    }
    out = args.out or default_out(
        f"smolvla_cam{len(img_keys)}_steps{cfg.num_steps}_chunk{cfg.chunk_size}"
    )
    profile_and_save(
        out, "smolvla", full_step, meta,
        warmup=args.warmup, reps=args.reps, prof_steps=args.prof_steps, extra_modes=extra,
    )
    print("wrote", out)


if __name__ == "__main__":
    main()
