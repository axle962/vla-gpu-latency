"""Trace OpenVLA-7B (openvla/openvla-7b) inference at batch 1, eager mode.

One "step" is one action prediction: the fused DINOv2 + SigLIP vision backbone on one 224x224
image, the projector, a Llama-2-7B prefill over the image patches and the prompt, then 7
greedy decode steps (one token per action dimension) and the CPU de-tokenization to a
7-DoF action (vla.predict_action, as in the model card).

Timed modes:
  eager           stock vla.predict_action (prefill + 7 decode tokens + CPU decode)
  eager_prefill   vla.generate with max_new_tokens=1 (vision + prefill + first token)
The per-token decode cost is (eager - eager_prefill) / 6; meta.json records it.

Static-cache modes (same math as generate's greedy decode, without HF generate):
  static_prefill  vision + projector + Llama prefill into a transformers StaticCache, eager
  static_decode   the 6 remaining greedy decode steps (forward + argmax), eager
  graph_decode    the same 6 decode steps captured as one CUDA graph and replayed
  static_hybrid   static_prefill + graph_decode (a deployable graph-mode inference)
HF generate grows a dynamic KV cache, so it cannot be captured. The static path builds the
multimodal embeddings as Prismatic does (patch embeddings spliced after <BOS>) and writes K/V
with index_put. transformers 4.40 drops the causal mask for 1-token queries when no
attention_mask is given (fine for a dynamic cache, wrong for a static one whose future slots
are zeros), and checks a 2D mask with torch.all() (a host sync), so the static path patches
AttentionMaskConverter._ignore_causal_mask_sdpa to return False: Llama then builds the
cache_position causal mask on the GPU. meta.json records whether the 7 action tokens match
generate's.

Precision: --dtype fp16 (default, matches what the other workloads use) or bf16 (the model
card's dtype). The 7B model needs ~15 GB in 16-bit, so it does not fit the 2080 Ti.

Usage (inside envs/openvla, on a GPU node):
  HF_HOME=/scratch/$USER/.cache/huggingface HF_HUB_OFFLINE=1 \\
    python workloads/trace_openvla.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trace_common import default_out, profile_and_save  # noqa: E402

import contextlib  # noqa: E402

from transformers import AutoModelForVision2Seq, AutoProcessor, StaticCache  # noqa: E402
from transformers.modeling_attn_mask_utils import AttentionMaskConverter  # noqa: E402

MODEL_ID = "openvla/openvla-7b"
INSTRUCTION = "pick up the red cube and place it in the bowl"
UNNORM_KEY = "bridge_orig"


def load(dtype: torch.dtype):
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    attn = "sdpa"
    try:
        vla = AutoModelForVision2Seq.from_pretrained(
            MODEL_ID, attn_implementation=attn, torch_dtype=dtype,
            low_cpu_mem_usage=True, trust_remote_code=True,
        )
    except (ValueError, ImportError) as e:
        print(f"sdpa unavailable ({e}); falling back to eager attention")
        attn = "eager"
        vla = AutoModelForVision2Seq.from_pretrained(
            MODEL_ID, attn_implementation=attn, torch_dtype=dtype,
            low_cpu_mem_usage=True, trust_remote_code=True,
        )
    return processor, vla.to("cuda").eval(), attn


class StaticPath:
    """Greedy OpenVLA inference with a static KV cache, so decode can be graph-captured."""

    def __init__(self, vla, pixel_values: torch.Tensor, ids: torch.Tensor, n_action: int):
        self.vla, self.lm = vla, vla.language_model
        self.ids, self.pixel = ids, pixel_values.clone()
        self.n_patches = None
        with torch.no_grad():
            self.n_patches = vla.projector(vla.vision_backbone(self.pixel)).shape[1]
        self.seq_len = ids.shape[1] + self.n_patches
        self.n_action = n_action
        self.lm._setup_cache(StaticCache, 1, max_cache_len=self.seq_len + n_action + 1)
        self.caches = [layer.self_attn.past_key_value for layer in self.lm.model.layers]
        self._detach()
        dev = ids.device
        self.pos_prefill = torch.arange(self.seq_len, device=dev)
        self.pos_decode = [
            torch.tensor([self.seq_len + i], device=dev) for i in range(n_action - 1)
        ]
        self.tok = torch.zeros(1, 1, dtype=torch.long, device=dev)
        self.out = torch.zeros(1, n_action, dtype=torch.long, device=dev)
        self.graph: torch.cuda.CUDAGraph | None = None

    def _detach(self):
        # Delete (not None) the per-layer cache attribute: Llama detects a static cache with
        # hasattr(), and attention prefers the attribute over the cache generate passes in, so
        # transformers' own _reset_cache (which sets None) would break stock generate.
        for layer in self.lm.model.layers:
            if hasattr(layer.self_attn, "past_key_value"):
                del layer.self_attn.past_key_value

    @contextlib.contextmanager
    def active(self):
        orig = AttentionMaskConverter._ignore_causal_mask_sdpa
        AttentionMaskConverter._ignore_causal_mask_sdpa = staticmethod(lambda *a, **k: False)
        for layer, c in zip(self.lm.model.layers, self.caches):
            layer.self_attn.past_key_value = c
        try:
            yield
        finally:
            self._detach()
            AttentionMaskConverter._ignore_causal_mask_sdpa = orig

    @torch.no_grad()
    def _prefill(self):
        v = self.vla
        proj = v.projector(v.vision_backbone(self.pixel))
        emb = v.get_input_embeddings()(self.ids)
        mm = torch.cat([emb[:, :1], proj, emb[:, 1:]], dim=1)
        logits = self.lm(inputs_embeds=mm, cache_position=self.pos_prefill, use_cache=False).logits
        self.tok.copy_(logits[:, -1].argmax(-1, keepdim=True))

    @torch.no_grad()
    def _decode(self):
        tok, toks = self.tok, [self.tok]
        for pos in self.pos_decode:
            logits = self.lm(input_ids=tok, cache_position=pos, use_cache=False).logits
            tok = logits[:, -1].argmax(-1, keepdim=True)
            toks.append(tok)
        self.out.copy_(torch.cat(toks, dim=1))

    def prefill(self):
        with self.active():
            self._prefill()

    def decode(self):
        with self.active():
            self._decode()

    def capture(self):
        with self.active():
            self._prefill()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    self._decode()
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self._decode()

    def hybrid(self):
        self.prefill()
        self.graph.replay()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--prof-steps", type=int, default=3)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--no-graph", action="store_true", help="skip the static/graph modes")
    args = ap.parse_args()
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16

    torch.backends.cuda.matmul.allow_tf32 = False
    processor, vla, attn = load(dtype)
    rng = np.random.default_rng(0)
    image = Image.fromarray(rng.integers(0, 256, (224, 224, 3), dtype=np.uint8))
    prompt = f"In: What action should the robot take to {INSTRUCTION}?\nOut:"
    inputs = processor(prompt, image).to("cuda", dtype=dtype)

    @torch.no_grad()
    def step():
        return vla.predict_action(**inputs, unnorm_key=UNNORM_KEY, do_sample=False)

    # predict_action appends the empty token 29871 before generating (and passes the original
    # attention_mask through unchanged); do exactly the same so the prefill matches the full
    # step's prompt.
    pre_ids = inputs["input_ids"]
    if not torch.all(pre_ids[:, -1] == 29871):
        pre_ids = torch.cat(
            [pre_ids, torch.tensor([[29871]], dtype=pre_ids.dtype, device=pre_ids.device)],
            dim=1,
        )
    pre_inputs = {**inputs, "input_ids": pre_ids}

    @torch.no_grad()
    def prefill():
        return vla.generate(**pre_inputs, max_new_tokens=1, do_sample=False)

    action = step()
    print("action:", np.round(np.asarray(action), 4).tolist())
    n_action = int(vla.get_action_dim(UNNORM_KEY))
    with torch.no_grad():
        ref_tokens = vla.generate(**pre_inputs, max_new_tokens=n_action, do_sample=False)[
            0, -n_action:
        ].tolist()

    extra = {"eager_prefill": prefill}
    graph_meta: dict = {"graph_ok": False}
    if not args.no_graph:
        try:
            sp = StaticPath(vla, inputs["pixel_values"], pre_ids, n_action)
            sp.prefill()
            sp.decode()
            static_tokens = sp.out[0].tolist()
            sp.capture()
            sp.out.zero_()
            sp.prefill()
            sp.graph.replay()
            torch.cuda.synchronize()
            graph_tokens = sp.out[0].tolist()
            print("tokens generate/static/graph:", ref_tokens, static_tokens, graph_tokens)
            graph_meta = {
                "graph_ok": True,
                "graph_scope": "6 greedy decode steps (prefill eager, static KV cache)",
                "ref_tokens": ref_tokens,
                "static_tokens": static_tokens,
                "graph_tokens": graph_tokens,
                "static_matches_generate": static_tokens == ref_tokens,
                "graph_matches_static": graph_tokens == static_tokens,
                "static_cache_len": sp.seq_len + n_action + 1,
            }
            extra.update({
                "static_prefill": sp.prefill,
                "static_decode": sp.decode,
                "graph_decode": sp.graph.replay,
                "static_hybrid": sp.hybrid,
            })
        except Exception as e:  # keep the eager data
            graph_meta = {"graph_ok": False, "graph_error": f"{type(e).__name__}: {e}"}
            print("static/graph path failed:", graph_meta["graph_error"])

    meta = {
        "hf_model": MODEL_ID,
        "dtype": args.dtype,
        "attn_implementation": attn,
        "batch": 1,
        "cameras": 1,
        "image_hw": [224, 224],
        "prompt_tokens": int(pre_ids.shape[1]),
        "action_tokens": n_action,
        "unnorm_key": UNNORM_KEY,
        "params": sum(p.numel() for p in vla.parameters()),
        "action_sample": np.asarray(action).tolist(),
        "tf32": False,
        "ref_tokens": ref_tokens,
        **graph_meta,
    }
    out = Path(args.out or default_out(f"openvla7b_{args.dtype}"))
    profile_and_save(
        out, "openvla-7b", step, meta,
        warmup=args.warmup, reps=args.reps, prof_steps=args.prof_steps,
        extra_modes=extra,
    )
    # Derived per-token decode cost.
    import json

    m = json.loads((out / "meta.json").read_text())
    m["decode_ms_per_token"] = (m["eager_ms_median"] - m["eager_prefill_ms_median"]) / (
        n_action - 1
    )
    if m.get("graph_ok"):
        m["graph_decode_speedup"] = m["static_decode_ms_median"] / m["graph_decode_ms_median"]
        m["static_hybrid_speedup"] = m["eager_ms_median"] / m["static_hybrid_ms_median"]
    (out / "meta.json").write_text(json.dumps(m, indent=2))
    print("decode_ms_per_token", m["decode_ms_per_token"], "wrote", out)


if __name__ == "__main__":
    main()
