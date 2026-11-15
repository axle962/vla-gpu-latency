"""Trace pi0-FAST (lerobot port; autoregressive FAST action tokens) at batch 1.

pi0-FAST encodes the cameras with SigLIP, prefills Gemma-2B over [images, prompt, BOS] (the
prompt carries the task and the discretized state), then greedily decodes FAST action tokens one
at a time through Gemma-2B and its 257k-word LM head. lerobot's decoder always runs
max_decoding_steps (256) steps, with no early stop at end of sequence; the CPU-side FAST
detokenization (inverse DCT) is not timed. This is the OpenVLA regime (decode = one token per
full-model pass) in a model that fits the 11-12 GB CIMS GPUs.

Timed modes (same names as workloads/trace_openvla.py):
  eager            stock step: image preprocessing + model.sample_actions_fast_kv_cache
  eager_prefill    images -> prefill (DynamicCache) -> first token
  static_decode    our static-cache decoder, eager launches (the remaining tokens)
  graph_decode     the same decoder captured as one CUDA graph
  static_hybrid    static-cache prefill (eager) + graph_decode replay
Static path: a StaticCache of length prefix + max_decoding_steps, an explicit cache_position
and a preallocated 4D attention mask (prefix pads masked; decode slot t unmasked at step t), so
every decode step has fixed shapes and no host syncs. Its tokens must equal the stock tokens.

Precision and environment as workloads/trace_pi.py (envs/pi; --precision fp16 for
Turing/Volta, with the same activation casts). Usage:
  HF_HOME=... python workloads/trace_pi0fast.py --precision fp16 [--decode-steps 256]
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trace_common import default_out, profile_and_save, profile_graph  # noqa: E402
from trace_pi import KEEP_FP32, PeakActivation, add_fp16_casts  # noqa: E402

from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.policies.pi0_fast.modeling_pi0_fast import PI0FastPolicy  # noqa: E402
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS  # noqa: E402
from transformers.cache_utils import StaticCache  # noqa: E402

REPO = "lerobot/pi0fast-libero-v044"
TASK = "put the black bowl on the plate"


def build(args):
    cfg = PreTrainedConfig.from_pretrained(REPO)
    cfg.compile_model = False
    cfg.max_decoding_steps = args.decode_steps
    cfg.temperature = 0.0
    cfg.device = "cuda"
    policy = PI0FastPolicy.from_pretrained(REPO, config=cfg).cuda().eval()
    if args.precision == "fp16":
        pw = policy.model.paligemma_with_expert
        pw.to(torch.float16)
        for name, p in pw.named_parameters():
            if any(k in name for k in KEEP_FP32):
                p.data = p.data.to(torch.float32)
        add_fp16_casts(pw, expert=False)
    pre, _ = make_pre_post_processors(cfg, pretrained_path=REPO)
    gen = torch.Generator().manual_seed(0)
    obs = {k: torch.rand(1, *ft.shape, generator=gen) for k, ft in cfg.image_features.items()
           if "empty_camera" not in k}
    # LIBERO's state is 8-dim; the checkpoint config declares the padded 32, but its
    # normalization statistics (and the pad-to-32 step in the preprocessor) expect 8.
    obs["observation.state"] = torch.rand(1, 8, generator=gen) * 2 - 1
    obs["task"] = [TASK]
    return policy, pre(obs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", choices=["fp16", "bf16"], default="fp16")
    ap.add_argument("--decode-steps", type=int, default=256, help="lerobot default 256")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--prof-steps", type=int, default=1)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    policy, batch = build(args)
    m, cfg = policy.model, policy.config
    pw = m.paligemma_with_expert
    lm, lm_head = pw.paligemma.language_model, pw.paligemma.lm_head
    lm.config._attn_implementation = "eager"
    tok, msk = batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK]
    D = args.decode_steps
    bos = torch.full((1, 1), m._paligemma_tokenizer.bos_token_id, dtype=torch.long, device="cuda")
    tokens_in = torch.cat([tok, bos], dim=1)
    masks_in = torch.cat([msk, torch.ones((1, 1), dtype=torch.bool, device="cuda")], dim=1)

    @torch.no_grad()
    def embed():
        images, img_masks = policy._preprocess_images(batch)
        embs, pad, att, _, _ = m.embed_prefix_fast(images, img_masks, tokens_in, masks_in)
        return embs, pad, att

    @torch.no_grad()
    def full_step():
        images, img_masks = policy._preprocess_images(batch)
        return m.sample_actions_fast_kv_cache(images, img_masks, tok, msk,
                                              max_decoding_steps=D, temperature=0.0)

    @torch.no_grad()
    def eager_prefill():
        embs, pad, att = embed()
        att4 = m._prepare_attention_masks_4d(att, dtype=embs.dtype)
        (out, _), kv = pw.forward(attention_mask=att4, position_ids=torch.cumsum(pad, 1) - 1,
                                  past_key_values=None, inputs_embeds=[embs, None],
                                  use_cache=True, adarms_cond=[None, None])
        return torch.argmax(lm_head(out[:, -1:, :])[:, -1], dim=-1, keepdim=True)

    # --- static path ---
    embs0, pad0, att0 = embed()
    P = embs0.shape[1]
    L = P + D
    wdtype = lm.layers[0].self_attn.q_proj.weight.dtype
    cache = StaticCache(config=lm.config, max_batch_size=1, max_cache_len=L, device="cuda",
                        dtype=wdtype)
    mval = torch.finfo(wdtype).min

    def mask_rows(att2d_bool, n_rows):  # (1, n_rows, P) bool -> (1, 1, n_rows, L) additive
        out = torch.full((1, 1, n_rows, L), mval, dtype=wdtype, device="cuda")
        out[:, :, :, :P] = m._prepare_attention_masks_4d(att2d_bool, dtype=wdtype)[:, :, -n_rows:]
        return out

    pre_cpos = torch.arange(P, device="cuda")
    dmask = torch.full((1, 1, 1, L), mval, dtype=wdtype, device="cuda")
    dmask_init = dmask.clone()
    base = torch.zeros((), dtype=torch.long, device="cuda")      # valid prefix tokens
    first_tok = torch.zeros((1, 1), dtype=torch.long, device="cuda")
    gen = torch.zeros((1, D), dtype=torch.long, device="cuda")
    cpos = [torch.tensor([P + t - 1], device="cuda") for t in range(1, D)]
    offs = [torch.tensor(t - 1, device="cuda") for t in range(1, D)]

    @torch.no_grad()
    def static_prefill():
        embs, pad, att = embed()
        out = lm.forward(inputs_embeds=embs, attention_mask=mask_rows(att, P),
                         position_ids=torch.cumsum(pad, 1) - 1, past_key_values=cache,
                         use_cache=True, cache_position=pre_cpos)
        t0 = torch.argmax(lm_head(out.last_hidden_state[:, -1:, :])[:, -1], dim=-1, keepdim=True)
        first_tok.copy_(t0)
        base.copy_(pad.sum())
        # decode rows see the valid prefix (pads masked) and nothing beyond it yet
        dmask_init.copy_(mask_rows(pad[:, None, :], 1))
        dmask.copy_(dmask_init)

    @torch.no_grad()
    def static_decode():
        nxt = first_tok
        gen[:, 0:1].copy_(nxt)
        for i, t in enumerate(range(1, D)):
            e = pw.embed_language_tokens(nxt)
            e = e * math.sqrt(e.shape[-1])
            if wdtype == torch.bfloat16:
                e = e.to(torch.bfloat16)
            dmask[..., P + t - 1] = 0.0
            out = lm.forward(inputs_embeds=e, attention_mask=dmask,
                             position_ids=(base + offs[i]).view(1, 1), past_key_values=cache,
                             use_cache=True, cache_position=cpos[i])
            nxt = torch.argmax(lm_head(out.last_hidden_state[:, -1:, :])[:, -1], dim=-1,
                               keepdim=True)
            gen[:, t:t + 1].copy_(nxt)
        return gen

    peak = PeakActivation(lm.layers[-1])
    ref = full_step()
    peak.remove()
    static_prefill()
    st = static_decode().clone()
    n_same = int((st == ref).sum())
    diff_pos = (st != ref)[0].nonzero().flatten().tolist()
    print(f"static vs stock tokens: {n_same}/{D} equal; first 12 stock {ref[0, :12].tolist()}; "
          f"differ at {diff_pos[:10]}: stock {ref[0, diff_pos[:5]].tolist()} "
          f"static {st[0, diff_pos[:5]].tolist()}")

    graph_ok, graph_err, n_graph_same = True, "", -1
    g = torch.cuda.CUDAGraph()
    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                static_prefill()
                static_decode()
        torch.cuda.current_stream().wait_stream(s)
        static_prefill()
        torch.cuda.synchronize()
        with torch.cuda.graph(g):
            static_decode()
        static_prefill()
        g.replay()
        torch.cuda.synchronize()
        n_graph_same = int((gen == ref).sum())
        print(f"graph vs stock tokens: {n_graph_same}/{D} equal")
    except Exception as e:  # keep eager data
        graph_ok, graph_err = False, f"{type(e).__name__}: {e}"
        print("graph capture failed:", graph_err)

    def decode_eager():
        dmask.copy_(dmask_init)
        static_decode()

    def graph_decode():
        dmask.copy_(dmask_init)
        g.replay()

    def hybrid():
        static_prefill()
        g.replay()

    extra = {"eager_prefill": eager_prefill, "static_decode": decode_eager}
    if graph_ok:
        extra["graph_decode"] = graph_decode
        extra["static_hybrid"] = hybrid

    meta = {
        "hf_model": REPO,
        "workload": "pi0fast",
        "dtype": args.precision,
        "batch": 1,
        "cameras": len([k for k in cfg.image_features if "empty_camera" not in k]),
        "camera_slots": len(cfg.image_features),
        "prefix_len": int(P),
        "valid_prefix_len": int(pad0.sum()),
        "decode_steps": D,
        "action_tokens": D,
        "chunk_size": cfg.chunk_size,
        "params": sum(p.numel() for p in policy.parameters()),
        "stock_tokens_head": ref[0, :16].tolist(),
        "last_layer_peak_abs": peak.peak,
        "last_layer_finite": peak.finite,
        "static_vs_stock_tokens_equal": n_same,
        "static_vs_stock_first_diff": diff_pos[0] if diff_pos else -1,
        "graph_vs_stock_tokens_equal": n_graph_same,
        "graph_ok": graph_ok,
        "graph_error": graph_err,
        "graph_scope": f"decode loop ({D - 1} tokens; prefill eager)",
        "tf32": False,
    }
    out = args.out or default_out(f"pi0fast_libero_{args.precision}_decode{D}")
    profile_and_save(out, "pi0fast", full_step, meta, warmup=args.warmup, reps=args.reps,
                     prof_steps=args.prof_steps, extra_modes=extra)
    if graph_ok:
        profile_graph(out, graph_decode, n=args.prof_steps)
    print("wrote", out)


if __name__ == "__main__":
    main()
