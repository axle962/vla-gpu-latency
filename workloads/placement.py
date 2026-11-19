"""Hosting placement profile: where the two halves of a VLA step run.

A flow-matching VLA step has two halves joined by one handoff:
  prefix    images + language + state -> KV cache (vision tower + VLM), run once
  denoise   num_steps passes of the action expert over that KV cache
This script hosts each half on its own device and times every part, so the cost of splitting
a policy across two GPUs (GPU-GPU) or across a GPU and the host CPU (GPU-CPU) can be compared
with the colocated baseline and with a prediction built from parts.

Placements (prefix device, denoise device), --placements takes a comma list:
  gpu       cuda:0, cuda:0   colocated baseline (the configuration every other trace uses)
  gpu-gpu   cuda:0, cuda:1   split across two GPUs, KV cache crosses GPU to GPU
  gpu-cpu   cuda:0, cpu      prefix on the GPU, action expert on the host CPU
  cpu-gpu   cpu,    cuda:0   prefix on the host CPU, action expert on the GPU
  cpu       cpu,    cpu      CPU-only baseline
Placements whose devices are missing are skipped with a message.

Timed modes per placement (wall clock, every CUDA device involved is synchronized):
  prefix          prefix on device A
  handoff         KV cache + pad mask, device A -> device B (and dtype cast on B if needed)
  denoise_eager   denoising loop on device B from a KV cache already on B
  denoise_graph   the same loop as one CUDA graph (only when B is a GPU)
  e2e_eager       prefix + handoff + eager loop
  e2e_hybrid      prefix + handoff into the graph's static cache + graph replay (B is a GPU)

Also measured: a copy calibration for every device pair used (one tensor of 4 KB to 256 MB,
giving a per-tensor fixed cost and a bandwidth), so the handoff can be predicted from the KV
cache size instead of measured. model/placement_report.py turns the outputs into
predicted-vs-measured tables.

Precision: FP16 on GPUs (as in trace_smolvla.py), FP32 on the CPU (FP16 CPU kernels are slow
or missing). Outputs are compared with the first placement run; expect ~1e-3 differences
between FP16 and FP32 placements and ~0 between same-precision placements.

Outputs in <out>/ (default results/placement/<gpu>_<host>_<time>/):
  latency.csv      placement,mode,step,wall_ms
  summary.csv      placement,dev_prefix,dev_denoise,mode,n,median_ms,mean_ms,std_ms,min_ms
  copy_calib.csv   src,dst,bytes,n,median_ms
  meta.json        devices, host CPU, threads, versions, KV bytes and tensor count, diffs
  trace_<placement>.json.gz   with --profile: torch.profiler trace of 2 e2e_eager steps

Usage (inside envs/smolvla, both GPUs idle; pin to the GPUs' NUMA node as in run_cuda2.sh):
  CUDA_VISIBLE_DEVICES=0,1 HF_HUB_OFFLINE=1 python workloads/placement.py
  python workloads/placement.py --model toy --placements cpu     # smoke test, no weights
"""

from __future__ import annotations

import argparse
import contextlib
import gzip
import json
import os
import platform
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

PLACEMENTS = {
    "gpu": ("cuda:0", "cuda:0"),
    "gpu-gpu": ("cuda:0", "cuda:1"),
    "gpu-cpu": ("cuda:0", "cpu"),
    "cpu-gpu": ("cpu", "cuda:0"),
    "cpu": ("cpu", "cpu"),
}
INSTRUCTION = "Pick up the red cube and place it in the bowl."


@dataclass
class Half:
    """One copy of the policy on one device. prefix() returns (pad, kv); denoise(pad, kv, x)
    returns the action chunk. kv is {layer: {name: tensor}}."""

    dev: torch.device
    dtype: torch.dtype
    prefix: Callable[[], tuple]
    denoise: Callable[[torch.Tensor, dict, torch.Tensor], torch.Tensor]
    noise: torch.Tensor
    capture_ctx: Callable[[], contextlib.AbstractContextManager]
    params: int


def available(dev: str) -> bool:
    d = torch.device(dev)
    if d.type == "cpu":
        return True
    return torch.cuda.is_available() and d.index < torch.cuda.device_count()


def sync(*devs: torch.device) -> None:
    for d in {d for d in devs if d.type == "cuda"}:
        torch.cuda.synchronize(d)


def timed(fn: Callable[[], object], devs: tuple, warmup: int, reps: int) -> list[float]:
    for _ in range(warmup):
        fn()
    out = []
    for _ in range(reps):
        sync(*devs)
        t0 = time.perf_counter()
        fn()
        sync(*devs)
        out.append((time.perf_counter() - t0) * 1e3)
    return out


def kv_bytes(pad: torch.Tensor, kv: dict) -> tuple[int, int]:
    ts = [pad] + [v for layer in kv.values() for v in layer.values()]
    return sum(t.numel() * t.element_size() for t in ts), len(ts)


def move(pad: torch.Tensor, kv: dict, dev: torch.device, dtype: torch.dtype):
    """Device copy first (the bytes on the wire are the source precision), then the cast."""
    kv2 = {
        i: {k: v.to(dev).to(dtype) for k, v in layer.items()} for i, layer in kv.items()
    }
    return pad.to(dev), kv2


def copy_into(pad, kv, static_pad, static_kv) -> None:
    static_pad.copy_(pad)
    for i, layer in kv.items():
        for k, v in layer.items():
            static_kv[i][k].copy_(v)


# ---------------------------------------------------------------------------------------------
# Models


def build_smolvla(dev: torch.device, args) -> Half:
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, make_att_2d_masks
    from lerobot.utils.constants import OBS_STATE
    from trace_smolvla import MODEL_ID, CudaConstMemo

    policy = SmolVLAPolicy.from_pretrained(MODEL_ID)
    cfg = policy.config
    cfg.num_steps = args.num_steps
    cfg.chunk_size = args.chunk_size
    cfg.n_action_steps = args.chunk_size
    policy = policy.to(dev).eval()
    dtype = torch.float16 if dev.type == "cuda" else torch.float32
    policy.model.vlm_with_expert.to(dtype)  # stock is BF16 VLM: FP16 on GPU, FP32 on CPU
    m = policy.model
    m.global_image_start_token = m.global_image_start_token.to(dev)
    m.image_end_token = m.image_end_token.to(dev)

    # Inputs are drawn on the CPU so every device sees the same observation and noise.
    gen = torch.Generator().manual_seed(0)
    img_keys = list(cfg.image_features)[: args.cameras]
    batch = {}
    for k in img_keys:
        c, h, w = cfg.image_features[k].shape
        batch[k] = torch.rand(1, c, h, w, generator=gen).to(dev)
    state_dim = cfg.robot_state_feature.shape[0]
    batch[OBS_STATE] = torch.rand(1, state_dim, generator=gen).to(dev)
    enc = m.vlm_with_expert.processor.tokenizer(
        INSTRUCTION + "\n",
        padding="max_length",
        max_length=cfg.tokenizer_max_length,
        truncation=True,
        return_tensors="pt",
    )
    lang, lang_mask = enc["input_ids"].to(dev), enc["attention_mask"].to(dev).bool()
    noise = torch.randn(1, cfg.chunk_size, cfg.max_action_dim, generator=gen).to(dev)

    @torch.no_grad()
    def prefix():
        images, img_masks = policy.prepare_images(batch)
        state = policy.prepare_state(batch)
        embs, pad, att = m.embed_prefix(images, img_masks, lang, lang_mask, state=state)
        att2d = make_att_2d_masks(pad, att)
        pos = torch.cumsum(pad, dim=1) - 1
        _, kv = m.vlm_with_expert.forward(
            attention_mask=att2d, position_ids=pos, past_key_values=None,
            inputs_embeds=[embs, None], use_cache=cfg.use_cache, fill_kv_cache=True,
        )
        return pad, kv

    dt = -1.0 / cfg.num_steps
    times = [
        torch.full((1,), 1.0 + s * dt, dtype=torch.float32, device=dev)
        for s in range(cfg.num_steps)
    ]

    @torch.no_grad()
    def denoise(pad, kv, x0):
        x_t = x0
        for t in times:
            v_t = m.denoise_step(x_t=x_t, prefix_pad_masks=pad, past_key_values=kv, timestep=t)
            x_t = x_t + dt * v_t
        return x_t

    memo = CudaConstMemo()
    return Half(
        dev, dtype, prefix, denoise, noise, memo.active,
        sum(p.numel() for p in policy.parameters()),
    )


def build_toy(dev: torch.device, args) -> Half:
    """Small random transformer with the same two-half structure (prefix fills a per-layer KV
    cache, an expert cross-attends to it for num_steps). For smoke tests without weights; its
    timings mean nothing."""
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(0)
    dtype = torch.float16 if dev.type == "cuda" else torch.float32
    d, layers, heads, plen, adim = 192, 4, 4, 96, 32

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))
            self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
            self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)

        def split(self, x):
            return x.view(x.shape[0], x.shape[1], heads, -1).transpose(1, 2)

        def forward(self, x, kv=None):
            h = self.n1(x)
            k, v = self.k(h), self.v(h)
            mine = {"key_states": k, "value_states": v}
            if kv is not None:
                k = torch.cat([kv["key_states"].to(k.dtype), k], dim=1)
                v = torch.cat([kv["value_states"].to(v.dtype), v], dim=1)
            a = F.scaled_dot_product_attention(self.split(self.q(h)), self.split(k), self.split(v))
            x = x + self.o(a.transpose(1, 2).reshape(x.shape))
            return x + self.mlp(self.n2(x)), mine

    vlm = nn.ModuleList(Layer() for _ in range(layers)).to(dev, dtype).eval()
    expert = nn.ModuleList(Layer() for _ in range(layers)).to(dev, dtype).eval()
    a_in = nn.Linear(adim, d).to(dev, dtype).eval()
    a_out = nn.Linear(d, adim).to(dev, dtype).eval()
    gen = torch.Generator().manual_seed(0)
    obs = torch.randn(1, plen, d, generator=gen).to(dev, dtype)
    noise = torch.randn(1, args.chunk_size, adim, generator=gen).to(dev)

    @torch.no_grad()
    def prefix():
        x, kv = obs, {}
        for i, layer in enumerate(vlm):
            x, kv[i] = layer(x)
        return torch.ones(1, plen, dtype=torch.bool, device=dev), kv

    dt = -1.0 / args.num_steps

    @torch.no_grad()
    def denoise(pad, kv, x0):
        x_t = x0
        for _ in range(args.num_steps):
            h = a_in(x_t.to(dtype))
            for i, layer in enumerate(expert):
                h, _ = layer(h, kv[i])
            x_t = x_t + dt * a_out(h).float()
        return x_t

    n = sum(p.numel() for mod in (vlm, expert, a_in, a_out) for p in mod.parameters())
    return Half(dev, dtype, prefix, denoise, noise, contextlib.nullcontext, n)


BUILDERS = {"smolvla": build_smolvla, "toy": build_toy}


# ---------------------------------------------------------------------------------------------
# Measurement


def capture_graph(half: Half, pad, kv):
    """Capture half.denoise on half.dev with static inputs. Returns (graph, static_pad,
    static_kv, static_out) or raises."""
    static_x = half.noise.clone()
    static_pad = pad.clone()
    static_kv = {i: {k: v.clone() for k, v in layer.items()} for i, layer in kv.items()}
    with torch.cuda.device(half.dev), half.capture_ctx():
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream(device=half.dev)
        s.wait_stream(torch.cuda.current_stream(half.dev))
        with torch.cuda.stream(s):
            for _ in range(3):
                half.denoise(static_pad, static_kv, static_x)
        torch.cuda.current_stream(half.dev).wait_stream(s)
        torch.cuda.synchronize(half.dev)
        # torch.cuda.graph's default capture stream is one per process, created on the device
        # that captured first (cuda:0); capturing cuda:1 work on it invalidates the capture.
        cs = torch.cuda.Stream(device=half.dev)
        with torch.cuda.graph(g, stream=cs):
            static_out = half.denoise(static_pad, static_kv, static_x)
    return g, static_pad, static_kv, static_out


def copy_calibration(pairs: set[tuple[str, str]], reps: int) -> list[dict]:
    rows = []
    for src, dst in sorted(pairs):
        s, d = torch.device(src), torch.device(dst)
        for nbytes in (4 << 10, 1 << 20, 4 << 20, 16 << 20, 64 << 20, 256 << 20):
            x = torch.zeros(nbytes // 2, dtype=torch.float16, device=s)
            t = timed(lambda x=x, d=d: x.to(d), (s, d), 3, reps)
            rows.append({
                "src": src, "dst": dst, "bytes": nbytes, "n": reps,
                "median_ms": statistics.median(t),
            })
            del x
        # Per-copy cost when 32 small copies are queued with one sync at the end, as in the
        # KV handoff (the single-copy rows above include a full sync per copy).
        xs = [torch.zeros(2048, dtype=torch.float16, device=s) for _ in range(32)]
        t = timed(lambda xs=xs, d=d: [x.to(d) for x in xs], (s, d), 3, reps)
        rows.append({
            "src": src, "dst": dst, "bytes": 4 << 10, "n": reps,
            "median_ms": statistics.median(t) / 32, "batch": 32,
        })
    for r in rows:
        r.setdefault("batch", 1)
    return rows


def run_placement(name: str, a: Half, b: Half, args, lat: list[dict]) -> dict:
    devs = (a.dev, b.dev)
    same = a.dev == b.dev
    reps = args.reps_cpu if "cpu" in (a.dev.type, b.dev.type) else args.reps
    warm = min(args.warmup, reps)
    info: dict = {"dev_prefix": str(a.dev), "dev_denoise": str(b.dev), "reps": reps}

    pad_a, kv_a = a.prefix()
    info["kv_bytes"], info["kv_tensors"] = kv_bytes(pad_a, kv_a)

    def handoff():
        return (pad_a, kv_a) if same else move(pad_a, kv_a, b.dev, b.dtype)

    pad_b, kv_b = handoff()
    out = b.denoise(pad_b, kv_b, b.noise)
    sync(*devs)
    info["output"] = out.float().cpu()

    def e2e_eager():
        pad, kv = a.prefix()
        if not same:
            pad, kv = move(pad, kv, b.dev, b.dtype)
        return b.denoise(pad, kv, b.noise)

    modes: dict[str, Callable[[], object]] = {
        "prefix": a.prefix,
        "denoise_eager": lambda: b.denoise(pad_b, kv_b, b.noise),
        "e2e_eager": e2e_eager,
    }
    if not same:
        modes["handoff"] = handoff

    info["graph_ok"], info["graph_error"] = False, ""
    if b.dev.type == "cuda" and not args.no_graph:
        try:
            g, s_pad, s_kv, s_out = capture_graph(b, pad_b, kv_b)
            g.replay()
            sync(b.dev)
            diff = (s_out.float().cpu() - info["output"]).abs().max().item()
            info["graph_vs_eager_max_abs_diff"] = diff
            info["graph_ok"] = True

            def e2e_hybrid():
                pad, kv = a.prefix()
                copy_into(pad, kv, s_pad, s_kv)  # cross-device copy_ casts on the way
                g.replay()

            modes["denoise_graph"] = g.replay
            modes["e2e_hybrid"] = e2e_hybrid
        except Exception as e:  # keep the eager data
            info["graph_error"] = f"{type(e).__name__}: {e}"
            print(f"[{name}] graph capture failed: {info['graph_error']}")

    for mode, fn in modes.items():
        t = timed(fn, devs, warm, reps)
        lat += [
            {"placement": name, "mode": mode, "step": i, "wall_ms": v} for i, v in enumerate(t)
        ]
        print(f"[{name}] {mode:14s} median {statistics.median(t):9.3f} ms  (n={reps})", flush=True)

    info["e2e_eager_fn"] = e2e_eager
    return info


def profile_placement(name: str, fn: Callable[[], object], devs: tuple, out: Path) -> None:
    """torch.profiler trace of 2 end-to-end steps. Runs only after every placement has been
    timed: CUPTI stays active after a profiler session and slows later timings (first cuda3
    run: prefix 59 -> 68 ms, eager loop 280 -> 349 ms, graph loop 66.5 -> 72.9 ms)."""
    from torch.profiler import ProfilerActivity, profile, record_function

    acts = [ProfilerActivity.CPU]
    if any(d.type == "cuda" for d in devs):
        acts.append(ProfilerActivity.CUDA)
    p = out / f"trace_{name}.json"
    with profile(activities=acts, record_shapes=True) as prof:
        for i in range(2):
            with record_function(f"vla_step_{i}"):
                fn()
            sync(*devs)
    prof.export_chrome_trace(str(p))
    with open(p, "rb") as f, gzip.open(str(p) + ".gz", "wb") as z:
        z.write(f.read())
    p.unlink()


def cpu_model() -> str:
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", choices=sorted(BUILDERS), default="smolvla")
    ap.add_argument("--placements", default="gpu,gpu-gpu,gpu-cpu,cpu-gpu,cpu")
    ap.add_argument("--cameras", type=int, default=3)
    ap.add_argument("--num-steps", type=int, default=10)
    ap.add_argument("--chunk-size", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--reps", type=int, default=30, help="reps for all-GPU placements")
    ap.add_argument("--reps-cpu", type=int, default=5, help="reps when a half runs on the CPU")
    ap.add_argument("--cpu-threads", type=int, default=0, help="torch threads, 0 = default")
    ap.add_argument("--no-graph", action="store_true")
    ap.add_argument("--profile", action="store_true", help="save a profiler trace per placement")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.cpu_threads:
        torch.set_num_threads(args.cpu_threads)
    torch.backends.cuda.matmul.allow_tf32 = False

    wanted = [p.strip() for p in args.placements.split(",") if p.strip()]
    unknown = [p for p in wanted if p not in PLACEMENTS]
    if unknown:
        ap.error(f"unknown placements {unknown}; choose from {list(PLACEMENTS)}")
    todo = []
    for p in wanted:
        missing = [d for d in PLACEMENTS[p] if not available(d)]
        if missing:
            print(f"skip {p}: no device {missing} (visible GPUs: {torch.cuda.device_count()})")
        else:
            todo.append(p)
    if not todo:
        sys.exit("no placement can run on this machine")

    gpu = torch.cuda.get_device_name(0).replace(" ", "_") if torch.cuda.is_available() else "cpu"
    host = platform.node().split(".")[0]
    if args.out is None:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        args.out = str(Path("results/placement") / f"{gpu}_{host}_{stamp}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    halves: dict[str, Half] = {}
    for d in dict.fromkeys(d for p in todo for d in PLACEMENTS[p]):
        print(f"building {args.model} on {d}", flush=True)
        halves[d] = BUILDERS[args.model](torch.device(d), args)

    lat: list[dict] = []
    infos, ref = {}, None
    for p in todo:
        a, b = (halves[d] for d in PLACEMENTS[p])
        info = run_placement(p, a, b, args, lat)
        o = info.pop("output")
        ref = o if ref is None else ref
        info["max_abs_diff_vs_first"] = (o - ref).abs().max().item()
        infos[p] = info

    # Drift check: time the first placement's end-to-end step again after all the others. A
    # large change means something left over from later placements (CPU threads, memory,
    # clocks) affected the timings, and the comparison across placements is not clean.
    first = todo[0]
    devs0 = tuple(torch.device(d) for d in PLACEMENTS[first])
    reps0 = args.reps_cpu if any(d.type == "cpu" for d in devs0) else args.reps
    t = timed(infos[first]["e2e_eager_fn"], devs0, min(args.warmup, reps0), reps0)
    lat += [
        {"placement": first, "mode": "e2e_eager_repeat", "step": i, "wall_ms": v}
        for i, v in enumerate(t)
    ]
    print(f"[{first}] e2e_eager_repeat median {statistics.median(t):9.3f} ms (drift check)")

    pairs = {PLACEMENTS[p] for p in todo if PLACEMENTS[p][0] != PLACEMENTS[p][1]}
    calib = copy_calibration(pairs, 10)

    if args.profile:
        for p in todo:
            devs = tuple(torch.device(d) for d in PLACEMENTS[p])
            profile_placement(p, infos[p]["e2e_eager_fn"], devs, out)
    for info in infos.values():
        info.pop("e2e_eager_fn")

    import pandas as pd

    df = pd.DataFrame(lat)
    df.to_csv(out / "latency.csv", index=False)
    summ = (
        df.groupby(["placement", "mode"], sort=False).wall_ms
        .agg(n="size", median_ms="median", mean_ms="mean", std_ms="std", min_ms="min")
        .reset_index()
    )
    summ.insert(1, "dev_prefix", summ.placement.map(lambda p: PLACEMENTS[p][0]))
    summ.insert(2, "dev_denoise", summ.placement.map(lambda p: PLACEMENTS[p][1]))
    summ.to_csv(out / "summary.csv", index=False)
    pd.DataFrame(calib, columns=["src", "dst", "bytes", "n", "median_ms", "batch"]).to_csv(
        out / "copy_calib.csv", index=False
    )
    meta = {
        "model": args.model,
        "host": platform.node(),
        "cpu": cpu_model(),
        "cpu_count": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "affinity_cpus": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_module_loading": os.environ.get("CUDA_MODULE_LOADING", "LAZY(default)"),
        "params": next(iter(halves.values())).params,
        "dtype_by_device": {d: str(h.dtype) for d, h in halves.items()},
        "cameras": args.cameras,
        "num_steps": args.num_steps,
        "chunk_size": args.chunk_size,
        "placements": infos,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(summ.to_string(index=False))
    print("wrote", out)


if __name__ == "__main__":
    main()
