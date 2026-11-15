"""SmolVLA denoising loop under torch.compile, next to eager and our manual CUDA graph.

The usual alternative to hand-written graph capture is torch.compile:
  compile_fused   mode="default": Inductor fuses elementwise/norm ops into Triton kernels, but
                  every kernel is still launched from the host (fewer, larger launches)
  compile_graph   mode="reduce-overhead": the same fused kernels replayed as CUDA graphs
Both compile only the denoising loop (the prefix has a data-dependent host sync, see
trace_smolvla.py), from the same fixed KV cache as eager_denoise and graph_denoise.

The profiled step ("eager" in meta.json, kernels.csv.gz) is compile_fused, so kernels.csv.gz is
the fused kernel list launched one at a time; kernels_graph.csv.gz is compile_graph replays.
Fusion reorders floating-point work, so outputs are compared with a tolerance, not bit for bit.

Usage (inside envs/smolvla):
  HF_HOME=... HF_HUB_OFFLINE=1 python workloads/trace_smolvla_compile.py --num-steps 10
Writes traces/smolvlacompile_cam<C>_steps<N>_chunk<K>/<gpu>_<host>_<stamp>/.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trace_common import default_out, profile_and_save, profile_graph  # noqa: E402
from trace_smolvla import CudaConstMemo, build  # noqa: E402

from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks  # noqa: E402
from lerobot.utils.constants import (  # noqa: E402
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cameras", type=int, default=3)
    ap.add_argument("--num-steps", type=int, default=10)
    ap.add_argument("--chunk-size", type=int, default=50)
    ap.add_argument("--dtype", choices=["fp16", "stock"], default="fp16")
    ap.add_argument("--batch", type=int, default=1)
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

    dt = -1.0 / cfg.num_steps
    times = [torch.full((args.batch,), 1.0 + s * dt, dtype=torch.float32, device="cuda")
             for s in range(cfg.num_steps)]

    def denoise(pad, kv, x0):
        x_t = x0
        for t in times:
            v_t = m.denoise_step(x_t=x_t, prefix_pad_masks=pad, past_key_values=kv, timestep=t)
            x_t = x_t + dt * v_t
        return x_t

    pad0, kv0 = prefix()
    torch.cuda.synchronize()
    with torch.no_grad():
        ref = denoise(pad0, kv0, noise)

    # Manual CUDA graph of the eager loop (same as trace_smolvla.py), for the comparison.
    memo = CudaConstMemo()
    static_x = noise.clone()
    g = torch.cuda.CUDAGraph()
    with torch.no_grad(), memo.active():
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                denoise(pad0, kv0, static_x)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        with torch.cuda.graph(g):
            g_out = denoise(pad0, kv0, static_x)
    g.replay()
    torch.cuda.synchronize()
    manual_diff = (g_out - ref).abs().max().item()

    compile_s, diffs = {}, {}
    fns = {}
    for name, mode in (("compile_fused", "default"), ("compile_graph", "reduce-overhead")):
        torch._dynamo.reset()
        fn = torch.compile(lambda x: denoise(pad0, kv0, x), mode=mode, fullgraph=False)
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(3):  # compile, then record (reduce-overhead records on call 2 to 3)
                out = fn(noise)
        torch.cuda.synchronize()
        compile_s[name] = time.perf_counter() - t0
        diffs[name] = (out.float() - ref.float()).abs().max().item()
        print(f"{name}: compile+warmup {compile_s[name]:.1f} s, max |diff| vs eager "
              f"{diffs[name]:.3e}")
        fns[name] = fn

    def run(fn):
        def f():
            with torch.no_grad():
                return fn(noise)
        return f

    extra = {
        "eager_denoise": lambda: torch.no_grad()(denoise)(pad0, kv0, noise),
        "graph_denoise": g.replay,
        "compile_graph": run(fns["compile_graph"]),
    }
    meta = {
        "hf_model": "lerobot/smolvla_base",
        "workload": "smolvla_compile",
        "dtype": args.dtype,
        "batch": args.batch,
        "cameras": len(img_keys),
        "num_steps": cfg.num_steps,
        "chunk_size": cfg.chunk_size,
        "profiled_mode": "compile_fused (denoising loop only)",
        "compile_seconds": compile_s,
        "compile_fused_max_abs_diff": diffs["compile_fused"],
        "compile_graph_max_abs_diff": diffs["compile_graph"],
        "graph_vs_eager_max_abs_diff": manual_diff,
        "ref_max_abs": ref.abs().max().item(),
        "triton_version": getattr(__import__("triton"), "__version__", ""),
        "tf32": False,
    }
    name = f"smolvlacompile_cam{len(img_keys)}_steps{cfg.num_steps}_chunk{cfg.chunk_size}"
    if args.batch > 1:
        name += f"_b{args.batch}"
    out = args.out or default_out(name)
    profile_and_save(out, "smolvla_compile", run(fns["compile_fused"]), meta,
                     warmup=args.warmup, reps=args.reps, prof_steps=args.prof_steps,
                     extra_modes=extra)
    profile_graph(out, run(fns["compile_graph"]), n=args.prof_steps)
    print("wrote", out)


if __name__ == "__main__":
    main()
