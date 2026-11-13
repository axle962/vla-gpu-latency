"""Baseline 4 (CLAUDE.md section 4.3): VLA-Perf itself (NVlabs/vla-perf, arXiv:2602.18397), run
on our workloads and GPUs, compared with our measured latencies.

VLA-Perf does not read traces: it is an analytical roofline over operator FLOPs and bytes from a
model architecture description, with spec-sheet peak FLOP/s and memory bandwidth. So we give it
(a) SmolVLA component configs matching lerobot/smolvla_base exactly as we traced it (SigLIP-B
vision tower at 512 px = 1024 patches per camera, 16 SmolLM2 layers, 720-wide action expert,
chunk 50, flow steps as configured), using the same pipeline as its pi0 model: vision prefill
(batch = cameras) + VLM prefill over the prefix + steps x action-expert parallel decode; and
(b) its own OpenVLA model (DINOv2 + SigLIP vision, Llama-2-7B prefill, decode of the action
tokens), with our prefix length (283). Batch 1, one GPU, FP16, its default settings otherwise.
Hardware: its A100_40GB entry; RTX 2080 Ti and TITAN V added from spec sheets in its own
convention (dense FP16 tensor FLOP/s with FP32 accumulate, like its RTX_4090 entry).

VLA-Perf is not a source-to-target predictor (it uses no measurement at all), so it gives one
prediction per (GPU, config), compared here with every measured quantity it can be mapped to:
  eager / hybrid / graph step  <- vision + VLM + action expert (the only total it predicts)
  eager_prefix                 <- vision + VLM
  graph_loop                   <- action expert (steps x one denoising pass)
A roofline predicts the same time for eager and graph execution, so its predicted CUDA-graph
speedup is 1.0 by construction.

Setup (once):  git clone https://github.com/NVlabs/vla-perf /scratch/$USER/ext/vla-perf
  uv venv -p 3.10 envs/vlaperf && VIRTUAL_ENV=$PWD/envs/vlaperf uv pip install \\
      -r /scratch/$USER/ext/vla-perf/genz/requirements.txt ipython
Usage: envs/vlaperf/bin/python model/vlaperf_baseline.py > results/vlaperf_baseline.csv
"""

from __future__ import annotations

import contextlib
import copy
import csv
import io
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VLAPERF = Path(os.environ.get("VLAPERF_DIR", f"/scratch/{os.environ.get('USER')}/ext/vla-perf"))
sys.path[:0] = [str(VLAPERF / "vla-perf"), str(VLAPERF / "genz")]

import GenZ  # noqa: E402
from GenZ.Models.default_models import ModelConfig  # noqa: E402
from Systems.system_configs import system_configs  # noqa: E402
from perf_utils import (collect_decode_perf, collect_parallel_decode_perf,  # noqa: E402
                        collect_prefill_perf)

assert str(VLAPERF) in GenZ.__file__, f"wrong GenZ imported: {GenZ.__file__}"

# Spec sheets, VLA-Perf convention (TFLOP/s dense FP16 tensor with FP32 accumulate, GB/s, GB).
system_configs.setdefault("RTX_2080_Ti", {
    "Flops": {"fp32": 13.4, "fp16": 53.8}, "Memory_size": 11, "Memory_BW": 616, "ICN": 0,
    "real_values": True})
system_configs.setdefault("TITAN_V", {
    "Flops": {"fp32": 14.9, "fp16": 110}, "Memory_size": 12, "Memory_BW": 653, "ICN": 0,
    "real_values": True})
SYSTEM = {"A100": "A100_40GB", "2080Ti": "RTX_2080_Ti", "TITANV": "TITAN_V"}
TRACE_KEY = {"A100": "A100", "2080Ti": "RTX_2080_Ti", "TITANV": "TITAN_V"}

# lerobot/smolvla_base (SmolVLM2-500M backbone truncated to 16 layers, expert width x0.75)
SMOLVLA_VISION = ModelConfig(
    model="vla/smolvla-siglip-b16-512-vision", vocab_size=0, max_model_len=1024,
    hidden_size=768, intermediate_size=3072, num_ffi=1, num_encoder_layers=12,
    num_decoder_layers=0, num_attention_heads=12, num_key_value_heads=12,
    hidden_act="gelu_pytorch_tanh")
SMOLVLA_VLM = ModelConfig(
    model="vla/smolvla-vlm", vocab_size=0, max_model_len=1024**3, hidden_size=960,
    intermediate_size=2560, num_ffi=2, num_encoder_layers=0, num_decoder_layers=16,
    num_attention_heads=15, num_key_value_heads=5, head_dim=64, hidden_act="silu")
SMOLVLA_EXPERT = ModelConfig(
    model="vla/smolvla-action-expert", vocab_size=0, max_model_len=1024**3, hidden_size=720,
    intermediate_size=2048, num_ffi=2, num_encoder_layers=0, num_decoder_layers=16,
    num_attention_heads=15, num_key_value_heads=5, head_dim=64, hidden_act="silu")
SMOLVLA_VISION_TOKENS = 1024  # (512 / 16)^2 patches per camera


def one(rows: list[dict]) -> float:
    """time_ms of the batch-1 result (single device, so tp = pp = 1)."""
    r = [x for x in rows if x["batch_size"] == 1]
    if not r:
        raise RuntimeError("VLA-Perf returned no batch-1 result")
    return float(min(x["time_ms"] for x in r))


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def smolvla(system: str, cams: int, prefix: int, steps: int, chunk: int) -> dict:
    vis = one(quiet(collect_prefill_perf, SMOLVLA_VISION, system, 1, SMOLVLA_VISION_TOKENS,
                    bits="fp16", max_batch_size=1, batch_size_multiplier=cams))
    vlm = one(quiet(collect_prefill_perf, SMOLVLA_VLM, system, 1, prefix, bits="fp16",
                    max_batch_size=1))
    exp = one(quiet(collect_parallel_decode_perf, SMOLVLA_EXPERT, system, 1, prefix, chunk,
                    self_attention=True, bits="fp16", max_batch_size=1)) * steps
    return {"vision": vis, "vlm": vlm, "expert": exp}


def openvla(system: str, prefix: int, tokens: int) -> dict:
    vis = sum(one(quiet(collect_prefill_perf, m, system, 1, 256, bits="fp16", max_batch_size=1))
              for m in ("dinov2-large-patch14-vision", "siglip2-so400m-patch14-384-vision"))
    pre = one(quiet(collect_prefill_perf, "openvla-7b-llm", system, 1, prefix, bits="fp16",
                    max_batch_size=1))
    # GenZ returns the latency of ONE decode step; VLA-Perf's openvla_perf.py multiplies it by
    # the number of action tokens, and so do we.
    dec = one(quiet(collect_decode_perf, "openvla-7b-llm", system, 1, prefix, tokens,
                    bits="fp16", max_batch_size=1)) * tokens
    return {"vision": vis, "prefill": pre, "decode": dec}


def measured_runs(model: str, gpu_key: str) -> dict:
    """model config -> meta of the run with the smallest eager std (as in cross_gpu.py)."""
    pat = "smolvla_*" if model == "smolvla" else "openvla*"
    best: dict = {}
    for p in sorted((ROOT / "traces").glob(f"{pat}/*{TRACE_KEY[gpu_key]}*/meta.json")):
        m = json.loads(p.read_text())
        k = (m.get("cameras"), m.get("num_steps"), m.get("chunk_size"))
        def rank(x):  # prefer runs with graph timings, then the smallest eager std
            return (not (x.get("graph_denoise_ms_median") or x.get("graph_decode_ms_median")),
                    x.get("eager_ms_std", 1e9))
        if k not in best or rank(m) < rank(best[k]):
            best[k] = m
    return best


def main() -> None:
    out = csv.writer(sys.stdout)
    out.writerow(["model", "gpu", "config", "quantity", "measured_ms", "vlaperf_ms",
                  "error_pct", "vlaperf_components"])

    def row(model, gpu, cfg, q, meas, pred, comp):
        if meas:
            out.writerow([model, gpu, cfg, q, f"{meas:.3f}", f"{pred:.3f}",
                          f"{100 * (pred - meas) / meas:.1f}", comp])

    for gk, system in SYSTEM.items():
        for (cams, steps, chunk), m in sorted(measured_runs("smolvla", gk).items()):
            c = smolvla(system, cams, m["prefix_len"], steps, chunk)
            total = sum(c.values())
            comp = ";".join(f"{k}={v:.3f}" for k, v in c.items())
            cfg = f"cam{cams}_steps{steps}_chunk{chunk}"
            row("smolvla", gk, cfg, "eager", m["eager_ms_median"], total, comp)
            row("smolvla", gk, cfg, "hybrid", m.get("hybrid_ms_median"), total, comp)
            row("smolvla", gk, cfg, "eager_prefix", m["eager_prefix_ms_median"],
                c["vision"] + c["vlm"], comp)
            row("smolvla", gk, cfg, "graph_loop", m.get("graph_denoise_ms_median"),
                c["expert"], comp)
            if m.get("graph_denoise_ms_median"):
                row("smolvla", gk, cfg, "speedup_loop_x",
                    m["eager_denoise_ms_median"] / m["graph_denoise_ms_median"], 1.0, comp)
        if gk == "A100":
            for m in measured_runs("openvla-7b", gk).values():
                n_tok = m.get("action_tokens", 7)
                c = openvla(system, m.get("prefix_len", 283), n_tok)
                comp = ";".join(f"{k}={v:.3f}" for k, v in c.items())
                total = sum(c.values())
                row("openvla-7b", gk, "default", "eager", m["eager_ms_median"], total, comp)
                row("openvla-7b", gk, "default", "hybrid", m.get("static_hybrid_ms_median"),
                    total, comp)
                row("openvla-7b", gk, "default", "eager_prefix",
                    m.get("eager_prefill_ms_median"), c["vision"] + c["prefill"], comp)
                # graph decode covers the n_tok - 1 tokens after the first (prefill emits one)
                row("openvla-7b", gk, "default", "graph_loop", m.get("graph_decode_ms_median"),
                    c["decode"] * (n_tok - 1) / n_tok, comp)


if __name__ == "__main__":
    main()
