"""Trace pi0 / pi0.5 (lerobot ports of Physical Intelligence's openpi models) at batch 1.

Same structure and timed modes as workloads/trace_smolvla.py: a PaliGemma prefix (SigLIP on
every camera, Gemma-2B over image + language tokens) fills a KV cache, then num_steps
flow-matching steps run the 300M Gemma action expert. pi0 feeds the robot state to the expert;
pi0.5 writes a discretized state into the language prompt and conditions the expert on time
through adaptive RMSNorm.
  eager           stock step (image preprocessing + model.sample_actions)
  eager_prefix    images -> KV cache
  eager_denoise   the denoising loop from a fixed KV cache
  graph_denoise   that loop captured as one CUDA graph
  hybrid          eager prefix + copy into the graph's static KV cache + graph replay

Checkpoints: the LIBERO fine-tunes in lerobot 0.4.4 format (lerobot/pi0_libero_finetuned_v044,
lerobot/pi05_libero_finetuned_v044): two 256x256 cameras (resized with padding to 224), an
empty third camera slot (SigLIP still runs on it, masked out), 8-dim state, chunk 50.
Inputs go through the policy's own preprocessor (normalization, PaliGemma tokenizer, which is
gated on the Hub: accept its license and log in). Tokenization is not timed.

Precision: the checkpoints are BF16. --precision fp16 casts the PaliGemma + expert stack to FP16
(for Turing/Volta, which have no BF16), keeping lerobot's FP32 exceptions (norms, patch
embedding); --precision bf16 keeps the checkpoint dtype (Ampere and newer). compile_model is
forced off (the checkpoints enable torch.compile).

Needs envs/pi (lerobot 0.4.4 + transformers from the fix/lerobot_openpi branch, which lerobot's
pi models check for). Usage:
  HF_HOME=... python workloads/trace_pi.py --policy pi0 --precision fp16
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trace_common import default_out, profile_and_save, profile_graph  # noqa: E402

from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS  # noqa: E402

REPOS = {"pi0": "lerobot/pi0_libero_finetuned_v044", "pi05": "lerobot/pi05_libero_finetuned_v044"}
TASK = "put the black bowl on the plate"
KEEP_FP32 = ("vision_tower.vision_model.embeddings.patch_embedding",
             "vision_tower.vision_model.embeddings.position_embedding",
             "input_layernorm", "post_attention_layernorm", "model.norm")


class CudaConstMemo:
    """torch.tensor(python constant, device=cuda) returns a cached GPU tensor while active
    (uploads from pageable memory are illegal during capture; the values are constants)."""

    def __init__(self) -> None:
        self.cache: dict = {}
        self._orig = torch.tensor

    def _tensor(self, data, *args, dtype=None, device=None, **kw):
        if (not args and not kw and device is not None and torch.device(device).type == "cuda"
                and isinstance(data, (bool, int, float, list, tuple))):
            key = (repr(data), dtype, str(device))
            if key not in self.cache:
                self.cache[key] = self._orig(data, dtype=dtype, device=device)
            return self.cache[key]
        return self._orig(data, *args, dtype=dtype, device=device, **kw)

    @contextlib.contextmanager
    def active(self):
        torch.tensor = self._tensor
        try:
            yield
        finally:
            torch.tensor = self._orig


def kv_tensors(kv) -> list:
    """Every tensor of a KV cache (HF Cache object, legacy tuples or dicts), in a fixed order."""
    if hasattr(kv, "layers"):
        return [t for layer in kv.layers for t in (layer.keys, layer.values)]
    if hasattr(kv, "key_cache"):
        return list(kv.key_cache) + list(kv.value_cache)
    if isinstance(kv, dict):
        return [t for layer in kv.values() for t in layer.values()]
    return [t for layer in kv for t in layer]


def _cast_pre_hook(dtype):
    """Forward pre-hook: cast the block's floating-point input activations to dtype."""
    def hook(mod, args, kwargs):
        args = tuple(a.to(dtype) if torch.is_tensor(a) and a.is_floating_point() and i == 0
                     else a for i, a in enumerate(args))
        for k in ("hidden_states", "inputs_embeds"):  # positional arg 0 or these kwargs
            v = kwargs.get(k)
            if torch.is_tensor(v) and v.is_floating_point():
                kwargs[k] = v.to(dtype)
        return args, kwargs
    return hook


def add_fp16_casts(pw, expert: bool = True) -> None:
    """The fix/lerobot_openpi transformers casts activations to BF16 only when the weights are
    BF16 (siglip SiglipVisionTransformer.forward before the encoder, gemma GemmaModel.forward
    before the layers). With FP16 weights the FP32 activations from the FP32 patch embedding and
    norms would reach FP16 linears; these hooks apply the same cast for FP16 at the same places."""
    vt = pw.paligemma.vision_tower.vision_model
    vt.encoder.register_forward_pre_hook(_cast_pre_hook(torch.float16), with_kwargs=True)
    # At the model input, before GemmaModel.forward derives the rotary tables from it. lerobot
    # calls these models' .forward() directly, which bypasses module hooks, so wrap the method.
    lms = [pw.paligemma.language_model] + ([pw.gemma_expert.model] if expert else [])
    for lm in lms:
        orig, hook = lm.forward, _cast_pre_hook(torch.float16)

        def fwd(*args, _orig=orig, _hook=hook, _mod=lm, **kwargs):
            args, kwargs = _hook(_mod, args, kwargs)
            return _orig(*args, **kwargs)
        lm.forward = fwd


class PeakActivation:
    """Max |output| of a module while active (FP16 overflow margin; FP16 max is 65504)."""

    def __init__(self, module) -> None:
        self.peak, self.finite = 0.0, True
        self._h = module.register_forward_hook(self._hook)

    def _hook(self, mod, inp, out):
        x = out[0] if isinstance(out, tuple) else out
        if torch.is_tensor(x):
            self.peak = max(self.peak, x.float().abs().max().item())
            self.finite &= bool(torch.isfinite(x).all())

    def remove(self) -> None:
        self._h.remove()


def build(args):
    if args.policy == "pi0":
        from lerobot.policies.pi0.modeling_pi0 import PI0Policy as Policy
    else:
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy as Policy
    repo = REPOS[args.policy]
    cfg = PreTrainedConfig.from_pretrained(repo)
    cfg.compile_model = False
    cfg.num_inference_steps = args.num_steps
    cfg.device = "cuda"
    policy = Policy.from_pretrained(repo, config=cfg).cuda().eval()
    if args.precision == "fp16":
        pw = policy.model.paligemma_with_expert
        pw.to(torch.float16)
        for name, p in pw.named_parameters():
            if any(k in name for k in KEEP_FP32):
                p.data = p.data.to(torch.float32)
        add_fp16_casts(pw)
    pre, _ = make_pre_post_processors(cfg, pretrained_path=repo)
    gen = torch.Generator().manual_seed(0)
    obs = {}
    for k, ft in cfg.image_features.items():
        if "empty_camera" in k:
            continue  # absent in LIBERO: the policy pads it with a masked -1 image
        obs[k] = torch.rand(1, *ft.shape, generator=gen)
    obs["observation.state"] = torch.rand(1, cfg.robot_state_feature.shape[0], generator=gen)
    obs["task"] = [TASK]
    batch = pre(obs)
    noise = torch.randn(1, cfg.chunk_size, cfg.max_action_dim, generator=gen).cuda()
    return policy, batch, noise


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=list(REPOS), default="pi0")
    ap.add_argument("--precision", choices=["fp16", "bf16"], default="fp16")
    ap.add_argument("--num-steps", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--prof-steps", type=int, default=3)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    policy, batch, noise = build(args)
    m, cfg = policy.model, policy.config
    is_pi0 = args.policy == "pi0"
    tok, msk = batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK]
    state = policy.prepare_state(batch) if is_pi0 else None

    from lerobot.policies.pi0.modeling_pi0 import make_att_2d_masks

    @torch.no_grad()
    def prefix():
        images, img_masks = policy._preprocess_images(batch)
        embs, pad, att = m.embed_prefix(images, img_masks, tok, msk)
        att4 = m._prepare_attention_masks_4d(make_att_2d_masks(pad, att))
        m.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
        _, kv = m.paligemma_with_expert.forward(
            attention_mask=att4, position_ids=torch.cumsum(pad, dim=1) - 1,
            past_key_values=None, inputs_embeds=[embs, None], use_cache=True)
        return pad, kv

    @torch.no_grad()
    def full_step():
        images, img_masks = policy._preprocess_images(batch)
        if is_pi0:
            return m.sample_actions(images, img_masks, tok, msk, state, noise=noise)
        return m.sample_actions(images, img_masks, tok, msk, noise=noise)

    dt = -1.0 / cfg.num_inference_steps
    times = [torch.full((1,), 1.0 + s * dt, dtype=torch.float32, device="cuda")
             for s in range(cfg.num_inference_steps)]

    @torch.no_grad()
    def denoise(pad, kv, x0):
        x_t = x0
        for t in times:
            if is_pi0:
                v = m.denoise_step(state=state, prefix_pad_masks=pad, past_key_values=kv,
                                   x_t=x_t, timestep=t)
            else:
                v = m.denoise_step(prefix_pad_masks=pad, past_key_values=kv, x_t=x_t, timestep=t)
            x_t = x_t + dt * v
        return x_t

    peak = PeakActivation(m.paligemma_with_expert.paligemma.language_model.layers[-1])
    pad0, kv0 = prefix()
    n_kv = len(kv_tensors(kv0))
    ref = full_step()
    peak.remove()
    split = denoise(pad0, kv0, noise)
    max_diff = (ref - split).abs().max().item()
    finite = bool(torch.isfinite(ref).all())
    print(f"split-vs-stock max |diff| = {max_diff:.3e}, finite output: {finite}, "
          f"|actions| max {ref.abs().max().item():.3f}")

    memo = CudaConstMemo()
    static_x, static_pad, static_kv = noise.clone(), pad0.clone(), copy.deepcopy(kv0)
    static_t = kv_tensors(static_kv)
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
            assert len(kv_tensors(static_kv)) == n_kv, "the denoise loop changed the KV cache"
            with torch.cuda.graph(g):
                static_out = denoise(static_pad, static_kv, static_x)
        g.replay()
        torch.cuda.synchronize()
        graph_diff = (static_out - split).abs().max().item()
        print(f"graph-vs-eager denoise max |diff| = {graph_diff:.3e}")
    except Exception as e:  # keep the eager data
        graph_ok, graph_err, graph_diff = False, f"{type(e).__name__}: {e}", float("nan")
        print("graph capture failed:", graph_err)

    def hybrid():
        pad, kv = prefix()
        static_pad.copy_(pad)
        for d, s_ in zip(static_t, kv_tensors(kv)):
            d.copy_(s_)
        g.replay()

    extra = {"eager_prefix": prefix, "eager_denoise": lambda: denoise(pad0, kv0, noise)}
    if graph_ok:
        extra["graph_denoise"] = g.replay
        extra["hybrid"] = hybrid

    meta = {
        "hf_model": REPOS[args.policy],
        "workload": args.policy,
        "dtype": args.precision,
        "batch": 1,
        "cameras": len([k for k in cfg.image_features if "empty_camera" not in k]),
        "camera_slots": len(cfg.image_features),
        "num_steps": cfg.num_inference_steps,
        "chunk_size": cfg.chunk_size,
        "prefix_len": int(pad0.shape[1]),
        "tokenizer_max_length": cfg.tokenizer_max_length,
        "params": sum(p.numel() for p in policy.parameters()),
        "split_vs_stock_max_abs_diff": max_diff,
        "output_finite": finite,
        "last_layer_peak_abs": peak.peak,
        "last_layer_finite": peak.finite,
        "graph_ok": graph_ok,
        "graph_error": graph_err,
        "graph_vs_eager_max_abs_diff": graph_diff,
        "graph_scope": "denoising loop only (prefix eager)",
        "tf32": False,
    }
    out = args.out or default_out(
        f"{args.policy}_libero_{args.precision}_steps{cfg.num_inference_steps}_chunk{cfg.chunk_size}")
    profile_and_save(out, args.policy, full_step, meta, warmup=args.warmup, reps=args.reps,
                     prof_steps=args.prof_steps, extra_modes=extra)
    if graph_ok:
        profile_graph(out, g.replay, n=args.prof_steps)
    print("wrote", out)


if __name__ == "__main__":
    main()
