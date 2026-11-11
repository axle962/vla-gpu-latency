"""Trace OpenVLA-7B (openvla/openvla-7b) inference at batch 1, eager mode.

One "step" is one action prediction: the fused DINOv2 + SigLIP vision backbone on one 224x224
image, the projector, a Llama-2-7B prefill over the image patches and the prompt, then 7
greedy decode steps (one token per action dimension) and the CPU de-tokenization to a
7-DoF action (vla.predict_action, as in the model card).

Timed modes:
  eager           stock vla.predict_action (prefill + 7 decode tokens + CPU decode)
  eager_prefill   vla.generate with max_new_tokens=1 (vision + prefill + first token)
The per-token decode cost is (eager - eager_prefill) / 6; meta.json records it.

CUDA graphs are not attempted yet: HF generate in transformers 4.40 grows a dynamic KV cache,
so capture would need a static-cache rewrite. The eager trace already gives the kernel list.

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

from transformers import AutoModelForVision2Seq, AutoProcessor  # noqa: E402

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--prof-steps", type=int, default=3)
    ap.add_argument("--out", type=str, default=None)
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
    }
    out = Path(args.out or default_out(f"openvla7b_{args.dtype}"))
    profile_and_save(
        out, "openvla-7b", step, meta,
        warmup=args.warmup, reps=args.reps, prof_steps=args.prof_steps,
        extra_modes={"eager_prefill": prefill},
    )
    # Derived per-token decode cost.
    import json

    m = json.loads((out / "meta.json").read_text())
    m["decode_ms_per_token"] = (m["eager_ms_median"] - m["eager_prefill_ms_median"]) / (
        n_action - 1
    )
    (out / "meta.json").write_text(json.dumps(m, indent=2))
    print("decode_ms_per_token", m["decode_ms_per_token"], "wrote", out)


if __name__ == "__main__":
    main()
