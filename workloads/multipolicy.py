"""Multi-policy hosting on one GPU (pending direction 6).

Question: when several robot policies share one GPU, what happens to each policy's step latency
and to the GPU's total throughput, in eager mode and with CUDA graphs? The project's thesis says
an eager VLA step is host bound (the GPU is busy ~15% of the step), so several eager policies
should fit on one GPU almost for free until the GPU or the host cores saturate, while graphed
policies are GPU chains that compete for the GPU. This script measures that, and
model/multipolicy_report.py predicts the co-hosted latencies from the policies measured alone.

Hosting modes (--hosting, how the N policies share the GPU):
  procs      one process per policy, one CUDA context each (the driver time-slices contexts)
  procs-mps  the same under an MPS control daemon started by scripts/run_multipolicy.sh, so
             kernels of different processes can run at the same time
  threads    one process, one Python thread and one CUDA stream per policy (one context, shared
             GIL, kernels on different streams can overlap)
  batch      one process, one policy that serves N observations per step (identical policies
             only: the batched reference, as in trace_smolvla.py --batch)

Execution modes per policy (--modes):
  eager   the stock step
  graph   the deployment with CUDA graphs: eager prefix + graph loop for SmolVLA and Diffusion
          (called hybrid in the traces), the whole step as one graph for ACT
  loop    the graph loop alone (no host work besides the launch): the GPU-only part

Protocol: the host process starts the workers, each worker loads its policy copies and checks
that the split loop and the graph reproduce the stock step, then for each execution mode all
workers signal ready, the host writes a common start time (warm_s ahead, so warmup already runs
under contention), and every copy runs closed loop (a new step as soon as the previous one
returned, the way a robot that acts as fast as it can would) until start + duration. Every step
is logged with its start time and wall time; the report keeps the middle 80% of the window.
With --profile (solo runs) each mode is also profiled for a few steps to get the GPU busy time
per step, the service demand the prediction needs.

Outputs (one host call = one config, under <out>/<config>/, plus rows appended to
<out>/latency.csv and an entry in <out>/configs.json):
  worker_<i>.json     policies, correctness diffs, profile (solo), per-step rows
  worker_<i>.log      stdout of that worker
  <out>/latency.csv   config,hosting,n_policies,batch,worker,copy,policy,mode,step,t_s,wall_ms
  <out>/meta.json     GPU, driver, host CPU, affinity, torch, MPS state

Usage (inside envs/smolvla, one idle GPU; scripts/run_multipolicy.sh runs the whole matrix):
  python workloads/multipolicy.py host --out results/multipolicy/<run> \\
      --hosting procs --policies smolvla,smolvla --modes eager,graph --duration 15
  python workloads/multipolicy.py host --out /tmp/mp --hosting threads --policies toy,toy --cpu
"""

from __future__ import annotations

import argparse
import contextlib
import gzip
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

HOSTINGS = ("procs", "procs-mps", "threads", "batch")
MODES = ("eager", "graph", "loop")
PROFILER_BIAS_US = 0.95  # per traced kernel, measured on A100 / 2080 Ti / TITAN V (CLAUDE.md 0.1)


# ---------------------------------------------------------------------------------------------
# One policy copy


@dataclass
class Copy:
    """One policy instance on its own CUDA stream. modes maps an execution mode to a callable
    that runs one step; activate() is called by the thread that will run the copy."""

    name: str
    stream: object
    modes: dict[str, Callable[[], object]]
    info: dict
    activate: Callable[[], None] = field(default=lambda: None)


def stream_ctx(stream):
    return torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext()


def sync(stream) -> None:
    if stream is not None:
        stream.synchronize()


def capture(stream, fn: Callable[[], object], warm: int = 3):
    """Warm fn on stream, then capture it as a CUDA graph on the same stream."""
    for _ in range(warm):
        fn()
    stream.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=stream):
        out = fn()
    return g, out


# ---------------------------------------------------------------------------------------------
# Builders


def build_smolvla(stream, batch: int, args, cuda: bool) -> Copy:
    import trace_smolvla as ts
    from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
    from lerobot.utils.constants import (
        OBS_LANGUAGE_ATTENTION_MASK,
        OBS_LANGUAGE_TOKENS,
        OBS_STATE,
    )

    ns = argparse.Namespace(
        model=ts.MODEL_ID, cameras=args.cameras, num_steps=args.num_steps,
        chunk_size=args.chunk_size, dtype="fp16", batch=batch,
    )
    with stream_ctx(stream):
        policy, obs, noise, img_keys = ts.build(ns)
    m, cfg = policy.model, policy.config

    @torch.no_grad()
    def prefix():
        images, img_masks = policy.prepare_images(obs)
        state = policy.prepare_state(obs)
        embs, pad, att = m.embed_prefix(
            images, img_masks, obs[OBS_LANGUAGE_TOKENS], obs[OBS_LANGUAGE_ATTENTION_MASK],
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
        images, img_masks = policy.prepare_images(obs)
        state = policy.prepare_state(obs)
        return m.sample_actions(
            images, img_masks, obs[OBS_LANGUAGE_TOKENS], obs[OBS_LANGUAGE_ATTENTION_MASK],
            state, noise=noise,
        )

    dt = -1.0 / cfg.num_steps
    times = [
        torch.full((batch,), 1.0 + s * dt, dtype=torch.float32, device="cuda")
        for s in range(cfg.num_steps)
    ]

    @torch.no_grad()
    def denoise(pad, kv, x0):
        x_t = x0
        for t in times:
            v_t = m.denoise_step(x_t=x_t, prefix_pad_masks=pad, past_key_values=kv, timestep=t)
            x_t = x_t + dt * v_t
        return x_t

    with stream_ctx(stream):
        pad0, kv0 = prefix()
        ref = full_step()
        split = denoise(pad0, kv0, noise)
        sync(stream)
        split_diff = (ref - split).abs().max().item()
        static_x = noise.clone()
        static_pad = pad0.clone()
        static_kv = {i: {k: v.clone() for k, v in layer.items()} for i, layer in kv0.items()}
        with ts.CudaConstMemo().active():
            g, static_out = capture(stream, lambda: denoise(static_pad, static_kv, static_x))
        g.replay()
        sync(stream)
        graph_diff = (static_out - split).abs().max().item()

    def hybrid():
        pad, kv = prefix()
        static_pad.copy_(pad)
        for i, layer in kv.items():
            for k, v in layer.items():
                static_kv[i][k].copy_(v)
        g.replay()

    info = {
        "hf_model": ts.MODEL_ID, "dtype": "fp16", "batch": batch, "cameras": len(img_keys),
        "num_steps": cfg.num_steps, "chunk_size": cfg.chunk_size,
        "params": sum(p.numel() for p in policy.parameters()),
        "kv_cache_bytes": sum(v.numel() * v.element_size() for la in kv0.values()
                              for v in la.values()),
        "split_vs_stock_max_abs_diff": split_diff, "graph_vs_eager_max_abs_diff": graph_diff,
        "graph_scope": "denoising loop (prefix eager)",
    }
    return Copy("smolvla", stream, {"eager": full_step, "graph": hybrid, "loop": g.replay}, info)


def build_act(stream, batch: int, args, cuda: bool) -> Copy:
    import trace_act as ta
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.utils.constants import OBS_IMAGES, OBS_STATE

    torch.backends.cudnn.allow_tf32 = False
    with stream_ctx(stream):
        policy = ACTPolicy.from_pretrained(ta.MODEL_ID).cuda().eval()
        cfg = policy.config
        dev = torch.device("cuda")
        gen = torch.Generator(device=dev).manual_seed(0)
        img_keys = list(cfg.image_features)
        obs = {OBS_STATE: torch.rand(batch, cfg.robot_state_feature.shape[0], device=dev,
                                     generator=gen)}
        obs[OBS_IMAGES] = [torch.rand(batch, 3, args.act_height, args.act_width, device=dev,
                                      generator=gen) for _ in img_keys]

        @torch.no_grad()
        def step():
            return policy.model(obs)[0]

        ref = step()
        sync(stream)
        with ta.ZerosOnGpu().active():
            g, static_out = capture(stream, step)
        g.replay()
        sync(stream)
        graph_diff = (static_out - ref).abs().max().item()

    info = {
        "hf_model": ta.MODEL_ID, "dtype": "fp32", "batch": batch, "cameras": len(img_keys),
        "image_input_hw": [args.act_height, args.act_width], "chunk_size": cfg.chunk_size,
        "params": sum(p.numel() for p in policy.parameters()),
        "graph_vs_eager_max_abs_diff": graph_diff, "graph_scope": "full step",
    }
    return Copy("act", stream, {"eager": step, "graph": g.replay, "loop": g.replay}, info)


_NOISE_BANKS: dict[int, "NoiseBank"] = {}


class NoiseBank:
    """Pre-sampled DDPM variance noise for one Diffusion Policy copy, served in call order
    (same idea as trace_diffusion.FixedNoise). Several copies in one process (threads
    hosting) each have their own bank; the patched randn_tensor picks the bank of the calling
    thread, so copies never share a counter."""

    def __init__(self, n: int, shape: tuple, device, seed: int = 1) -> None:
        g = torch.Generator(device=device).manual_seed(seed)
        self.bufs = [torch.randn(shape, device=device, generator=g) for _ in range(n)]
        self.i = 0

    def reset(self) -> None:
        self.i = 0

    def activate(self) -> None:
        _NOISE_BANKS[threading.get_ident()] = self

    def take(self, shape):
        b = self.bufs[self.i]
        self.i += 1
        assert tuple(shape) == tuple(b.shape)
        return b


def _randn_dispatch(shape, generator=None, device=None, dtype=None, **kw):
    return _NOISE_BANKS[threading.get_ident()].take(shape)


def build_diffusion(stream, batch: int, args, cuda: bool) -> Copy:
    import diffusers.schedulers.scheduling_ddpm as ddpm_mod
    import trace_diffusion as td
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.utils.constants import OBS_IMAGES, OBS_STATE

    ddpm_mod.randn_tensor = _randn_dispatch
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    with stream_ctx(stream):
        policy = DiffusionPolicy.from_pretrained(td.MODEL_ID).cuda().eval()
        dm, cfg = policy.diffusion, policy.config
        dm.num_inference_steps = args.diffusion_steps
        dev = torch.device("cuda")
        gen = torch.Generator(device=dev).manual_seed(0)
        n_obs = cfg.n_obs_steps
        img_key = list(cfg.image_features)[0]
        c, h, w = cfg.image_features[img_key].shape
        n_cam = len(cfg.image_features)
        obs = {
            OBS_STATE: torch.rand(batch, n_obs, cfg.robot_state_feature.shape[0], device=dev,
                                  generator=gen),
            OBS_IMAGES: torch.rand(batch, n_obs, n_cam, c, h, w, device=dev, generator=gen),
        }
        act_dim = cfg.action_feature.shape[0]
        noise = torch.randn(batch, cfg.horizon, act_dim, device=dev, generator=gen)
        bank = NoiseBank(args.diffusion_steps, (batch, cfg.horizon, act_dim), dev)
        bank.activate()

        @torch.no_grad()
        def prefix():
            return dm._prepare_global_conditioning(obs)

        @torch.no_grad()
        def denoise(gc, x0):
            bank.reset()
            return dm.conditional_sample(batch, global_cond=gc, noise=x0)

        @torch.no_grad()
        def full_step():
            bank.reset()
            return dm.generate_actions(obs, noise=noise)

        gc0 = prefix()
        ref = full_step()
        split = denoise(gc0, noise)
        sync(stream)
        s0, e0 = n_obs - 1, n_obs - 1 + cfg.n_action_steps
        split_diff = (ref - split[:, s0:e0]).abs().max().item()
        static_gc, static_x = gc0.clone(), noise.clone()
        g, static_out = capture(stream, lambda: denoise(static_gc, static_x))
        g.replay()
        sync(stream)
        graph_diff = (static_out - split).abs().max().item()

    def hybrid():
        static_gc.copy_(prefix())
        g.replay()

    info = {
        "hf_model": td.MODEL_ID, "dtype": "fp32", "batch": batch, "cameras": n_cam,
        "num_steps": args.diffusion_steps, "chunk_size": cfg.horizon,
        "params": sum(p.numel() for p in policy.parameters()),
        "split_vs_stock_max_abs_diff": split_diff, "graph_vs_eager_max_abs_diff": graph_diff,
        "graph_scope": "denoising loop (prefix eager)",
    }
    return Copy("diffusion", stream, {"eager": full_step, "graph": hybrid, "loop": g.replay},
                info, activate=bank.activate)


def build_toy(stream, batch: int, args, cuda: bool) -> Copy:
    """32 tiny Linear+ReLU layers: a dispatch-bound stand-in for smoke tests (CPU or GPU).
    Its timings mean nothing."""
    dev = "cuda" if cuda else "cpu"
    torch.manual_seed(0)
    with stream_ctx(stream):
        layers = [torch.nn.Linear(256, 256).to(dev).eval() for _ in range(32)]
        x = torch.randn(batch, 256, device=dev)

        @torch.no_grad()
        def step():
            h = x
            for la in layers:
                h = torch.relu(la(h))
            return h

        ref = step()
        if cuda:
            g, static_out = capture(stream, step)
            g.replay()
            sync(stream)
            graph_diff = (static_out - ref).abs().max().item()
            modes = {"eager": step, "graph": g.replay, "loop": g.replay}
        else:
            graph_diff = 0.0
            modes = {"eager": step, "graph": step, "loop": step}
    info = {"hf_model": "toy", "dtype": "fp32", "batch": batch,
            "params": sum(p.numel() for la in layers for p in la.parameters()),
            "graph_vs_eager_max_abs_diff": graph_diff, "graph_scope": "full step"}
    return Copy("toy", stream, modes, info)


BUILDERS = {"smolvla": build_smolvla, "act": build_act, "diffusion": build_diffusion,
            "toy": build_toy}


# ---------------------------------------------------------------------------------------------
# Worker: one process, one or more copies


def closed_loop(copy: Copy, mode: str, start: float, end: float) -> list[tuple[float, float]]:
    """Run copy's mode back to back: warm until start, then log (t since start, wall ms) per
    step until end. The stream is synchronized after every step (the robot needs the actions)."""
    fn = copy.modes[mode]
    copy.activate()
    rows: list[tuple[float, float]] = []
    with stream_ctx(copy.stream):
        while time.time() < start:
            fn()
            sync(copy.stream)
        while True:
            tw = time.time()
            t0 = time.perf_counter()
            fn()
            sync(copy.stream)
            rows.append((tw - start, (time.perf_counter() - t0) * 1e3))
            if time.time() >= end:
                break
    return rows


def run_phase(copies: list[Copy], mode: str, start: float, duration: float) -> list[list]:
    end = start + duration
    if len(copies) == 1:
        return [closed_loop(copies[0], mode, start, end)]
    out: list = [None] * len(copies)
    errors: list[str] = []
    bar = threading.Barrier(len(copies))

    def run(i: int) -> None:
        try:
            bar.wait()
            out[i] = closed_loop(copies[i], mode, start, end)
        except Exception as e:  # reported by the worker, not swallowed by the thread
            errors.append(f"copy {i} ({copies[i].name}): {type(e).__name__}: {e}")

    ts = [threading.Thread(target=run, args=(i,), name=f"copy{i}") for i in range(len(copies))]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    if errors:
        raise RuntimeError(f"{mode} phase failed in {len(errors)} thread(s):\n" + "\n".join(errors))
    return out


def profile_mode(copy: Copy, mode: str, n: int = 3) -> dict:
    """GPU busy time per step of one mode (torch.profiler, parsed by trace_parse). Raw busy
    includes ~PROFILER_BIAS_US per kernel of profiler bias; the corrected value removes it."""
    from torch.profiler import ProfilerActivity, profile, record_function

    from trace_parse import summarize_trace

    fn = copy.modes[mode]
    copy.activate()
    with stream_ctx(copy.stream):
        for _ in range(2):
            fn()
        sync(copy.stream)
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for i in range(n):
                with record_function(f"vla_step_{i}"):
                    fn()
                sync(copy.stream)
    tmp = Path(tempfile.mkdtemp(prefix="multipolicy_prof_"))
    prof.export_chrome_trace(str(tmp / "trace.json"))
    with open(tmp / "trace.json", "rb") as f, gzip.open(tmp / "trace.json.gz", "wb") as z:
        z.write(f.read())
    (tmp / "trace.json").unlink()
    s = summarize_trace(tmp)
    shutil.rmtree(tmp, ignore_errors=True)
    busy = s["gpu_busy_ms_per_step"]
    return {
        "kernels_per_step": s["kernels_per_step"],
        "memcpy_per_step": s["memcpy_per_step"],
        "gpu_busy_ms_per_step": busy,
        "gpu_busy_ms_per_step_corrected": busy - s["kernels_per_step"] * PROFILER_BIAS_US / 1e3,
        "gpu_span_ms_per_step": s["gpu_span_ms_per_step"],
        "kernel_dur_us_median": s["kernel_dur_us_median"],
    }


def wait_for(path: Path, timeout: float, poll: float = 0.02) -> str:
    t0 = time.time()
    while not path.exists():
        if time.time() - t0 > timeout:
            raise TimeoutError(f"waited {timeout:.0f}s for {path}")
        time.sleep(poll)
    return path.read_text()


def worker(args) -> None:
    run = Path(args.run_dir)
    wid = args.worker_id
    cuda = torch.cuda.is_available() and not args.cpu
    if not cuda and not args.cpu:
        sys.exit("no CUDA device (use --cpu for the toy smoke test)")
    torch.backends.cuda.matmul.allow_tf32 = False
    names = args.policies.split(",")
    copies: list[Copy] = []
    for i, name in enumerate(names):
        t0 = time.time()
        stream = torch.cuda.Stream() if cuda else None
        copies.append(BUILDERS[name](stream, args.batch, args, cuda))
        print(f"[w{wid}] copy {i} {name} ready in {time.time() - t0:.1f}s: "
              f"{json.dumps({k: v for k, v in copies[-1].info.items() if 'diff' in k})}",
              flush=True)
    modes = [m for m in args.modes.split(",") if m]
    rows: list[dict] = []
    for mode in modes:
        (run / f"ready_{mode}_{wid}").write_text(str(time.time()))
        start = float(wait_for(run / f"go_{mode}", args.load_timeout))
        per_copy = run_phase(copies, mode, start, args.duration)
        for ci, (c, r) in enumerate(zip(copies, per_copy)):
            rows += [{"worker": wid, "copy": ci, "policy": c.name, "mode": mode, "step": k,
                      "t_s": round(t, 4), "wall_ms": round(ms, 4)} for k, (t, ms) in enumerate(r)]
            med = statistics.median(v for _, v in r) if r else float("nan")
            print(f"[w{wid}] {mode:5s} copy {ci} {c.name:9s} {len(r):5d} steps, median "
                  f"{med:8.2f} ms", flush=True)
        (run / f"done_{mode}_{wid}").write_text(str(time.time()))
    prof = {}
    if args.profile and cuda:
        for mode in modes:
            try:
                prof[mode] = profile_mode(copies[0], mode)
                print(f"[w{wid}] profile {mode}: {json.dumps(prof[mode])}", flush=True)
            except Exception as e:  # keep the timings
                prof[mode] = {"error": f"{type(e).__name__}: {e}"}
                print(f"[w{wid}] profile {mode} failed: {prof[mode]['error']}", flush=True)
    import pandas as pd

    pd.DataFrame(rows).to_csv(run / f"latency_{wid}.csv", index=False)
    (run / f"worker_{wid}.json").write_text(json.dumps({
        "worker": wid, "pid": os.getpid(), "policies": names, "batch": args.batch,
        "copies": [c.info for c in copies], "profile": prof,
        "torch_threads": torch.get_num_threads(),
        "affinity_cpus": (len(os.sched_getaffinity(0))
                          if hasattr(os, "sched_getaffinity") else None),
    }, indent=2))
    (run / f"exit_{wid}").write_text("ok")


# ---------------------------------------------------------------------------------------------
# Host: one config = N policies under one hosting mode


def cpu_model() -> str:
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def nvsmi(query: str) -> str:
    try:
        return subprocess.run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=20).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def mps_state() -> str:
    if "CUDA_MPS_PIPE_DIRECTORY" not in os.environ:
        return "no pipe dir in env"
    try:
        r = subprocess.run(["nvidia-cuda-mps-control"], input="get_server_list\n",
                           capture_output=True, text=True, timeout=10)
        return "daemon up, servers: " + (r.stdout.strip() or "(none yet)")
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"daemon not reachable: {e}"


def config_name(hosting: str, policies: list[str], batch: int) -> str:
    parts, i = [], 0
    while i < len(policies):
        j = i
        while j < len(policies) and policies[j] == policies[i]:
            j += 1
        parts.append(policies[i] + (f"x{j - i}" if j - i > 1 else ""))
        i = j
    return f"{hosting}_{'+'.join(parts)}" + (f"_b{batch}" if batch > 1 else "")


def tail(path: Path, n: int = 15) -> str:
    try:
        return "\n".join(path.read_text().splitlines()[-n:])
    except OSError:
        return ""


def host(args) -> None:
    policies = [p for p in args.policies.split(",") if p]
    for p in policies:
        if p not in BUILDERS:
            sys.exit(f"unknown policy {p}; choose from {sorted(BUILDERS)}")
    if args.hosting == "batch":
        if len(set(policies)) != 1:
            sys.exit("batch hosting needs identical policies")
        batch, groups = len(policies), [policies[:1]]
    elif args.hosting == "threads":
        batch, groups = 1, [policies]
    else:
        batch, groups = 1, [[p] for p in policies]
    name = args.name or config_name(args.hosting, policies, batch)
    out = Path(args.out)
    run = out / name
    if run.exists():
        shutil.rmtree(run)
    run.mkdir(parents=True)
    meta_p = out / "meta.json"
    if not meta_p.exists():
        meta = {
            "host": platform.node(), "cpu": cpu_model(), "cpu_count": os.cpu_count(),
            "affinity_cpus": (len(os.sched_getaffinity(0))
                              if hasattr(os, "sched_getaffinity") else None),
            "gpu": nvsmi("name"), "driver": nvsmi("driver_version"),
            "gpu_memory_total": nvsmi("memory.total"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
            "cuda_module_loading": os.environ.get("CUDA_MODULE_LOADING", "LAZY(default)"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
            "profiler_bias_us": PROFILER_BIAS_US,
            "policy_args": {"cameras": args.cameras, "num_steps": args.num_steps,
                            "chunk_size": args.chunk_size, "diffusion_steps": args.diffusion_steps,
                            "act_hw": [args.act_height, args.act_width]},
        }
        meta_p.write_text(json.dumps(meta, indent=2))
    modes = [m for m in args.modes.split(",") if m]
    me = Path(__file__).resolve()
    procs: list[subprocess.Popen] = []
    print(f"=== {name}: {len(policies)} policies, {args.hosting}, modes {modes}, "
          f"{args.duration:.0f}s per mode", flush=True)
    for wid, group in enumerate(groups):
        cmd = [sys.executable, str(me), "worker", "--run-dir", str(run), "--worker-id", str(wid),
               "--policies", ",".join(group), "--modes", ",".join(modes), "--batch", str(batch),
               "--duration", str(args.duration), "--load-timeout", str(args.load_timeout),
               "--cameras", str(args.cameras), "--num-steps", str(args.num_steps),
               "--chunk-size", str(args.chunk_size), "--diffusion-steps",
               str(args.diffusion_steps), "--act-height", str(args.act_height),
               "--act-width", str(args.act_width)]
        if args.cpu:
            cmd.append("--cpu")
        if args.profile and len(policies) == 1:
            cmd.append("--profile")
        log = open(run / f"worker_{wid}.log", "w")
        procs.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT))

    def alive_or_fail(stage: str) -> None:
        for wid, p in enumerate(procs):
            if p.poll() is not None and not (run / f"exit_{wid}").exists():
                for q in procs:
                    if q.poll() is None:
                        q.kill()
                raise RuntimeError(f"worker {wid} died during {stage} (rc {p.returncode}):\n"
                                   + tail(run / f"worker_{wid}.log"))

    def wait_all(prefix: str, timeout: float, stage: str) -> None:
        t0 = time.time()
        while True:
            if all((run / f"{prefix}_{wid}").exists() for wid in range(len(procs))):
                return
            alive_or_fail(stage)
            if time.time() - t0 > timeout:
                for q in procs:
                    q.kill()
                raise TimeoutError(f"{stage}: waited {timeout:.0f}s for {prefix}_* in {run}")
            time.sleep(0.05)

    t_load = time.time()
    for mode in modes:
        wait_all(f"ready_{mode}", args.load_timeout, f"load/ready {mode}")
        if mode == modes[0]:
            print(f"    loaded in {time.time() - t_load:.0f}s", flush=True)
        start = time.time() + args.warm_s
        (run / f"go_{mode}").write_text(repr(start))
        wait_all(f"done_{mode}", args.warm_s + args.duration + 300, f"phase {mode}")
        print(f"    {mode} done", flush=True)
    for p in procs:
        p.wait(timeout=1800)
    alive_or_fail("exit")

    import pandas as pd

    frames = []
    infos = {}
    for wid in range(len(procs)):
        d = pd.read_csv(run / f"latency_{wid}.csv")
        frames.append(d)
        infos[wid] = json.loads((run / f"worker_{wid}.json").read_text())
    lat = pd.concat(frames, ignore_index=True)
    lat.insert(0, "config", name)
    lat.insert(1, "hosting", args.hosting)
    lat.insert(2, "n_policies", len(policies))
    lat.insert(3, "batch", batch)
    lat_p = out / "latency.csv"
    lat.to_csv(lat_p, mode="a", header=not lat_p.exists(), index=False)
    entry = {
        "config": name, "hosting": args.hosting, "policies": policies, "batch": batch,
        "modes": modes, "duration_s": args.duration, "warm_s": args.warm_s,
        "mps": mps_state() if args.hosting == "procs-mps" else "",
        "gpu_memory_used_after": nvsmi("memory.used"),
        "workers": infos, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    cfgs_p = out / "configs.json"
    cfgs = json.loads(cfgs_p.read_text()) if cfgs_p.exists() else []
    cfgs = [c for c in cfgs if c["config"] != name] + [entry]
    cfgs_p.write_text(json.dumps(cfgs, indent=2))
    for (mode, wid, ci, pol), g in lat.groupby(["mode", "worker", "copy", "policy"]):
        mid = g[(g.t_s >= 0.1 * args.duration) & (g.t_s <= 0.9 * args.duration)].wall_ms
        print(f"    {mode:5s} w{wid} c{ci} {pol:9s} median {mid.median():8.2f} ms  "
              f"p95 {mid.quantile(0.95):8.2f}  n {len(mid)}", flush=True)
    for wid in range(len(procs)):
        for f in run.glob(f"latency_{wid}.csv"):
            f.unlink()
    for f in list(run.glob("ready_*")) + list(run.glob("go_*")) + list(run.glob("done_*")) \
            + list(run.glob("exit_*")):
        f.unlink()


# ---------------------------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--modes", default="eager,graph", help=f"comma list of {MODES}")
    common.add_argument("--duration", type=float, default=15.0, help="seconds per mode")
    common.add_argument("--load-timeout", type=float, default=1500.0)
    common.add_argument("--cpu", action="store_true", help="toy policy on the CPU (smoke test)")
    common.add_argument("--cameras", type=int, default=3)
    common.add_argument("--num-steps", type=int, default=10, help="SmolVLA flow steps")
    common.add_argument("--chunk-size", type=int, default=50)
    common.add_argument("--diffusion-steps", type=int, default=10, help="DDPM steps (stock 100)")
    common.add_argument("--act-height", type=int, default=480)
    common.add_argument("--act-width", type=int, default=640)
    h = sub.add_parser("host", parents=[common], help="run one config")
    h.add_argument("--out", required=True, help="run directory (results/multipolicy/<run>)")
    h.add_argument("--hosting", choices=HOSTINGS, default="procs")
    h.add_argument("--policies", required=True, help="comma list, e.g. smolvla,smolvla,act")
    h.add_argument("--name", default=None, help="config name (default from hosting/policies)")
    h.add_argument("--warm-s", type=float, default=5.0, help="warmup under contention")
    h.add_argument("--profile", action="store_true",
                   help="profile each mode when there is a single policy (GPU busy per step)")
    w = sub.add_parser("worker", parents=[common])
    w.add_argument("--run-dir", required=True)
    w.add_argument("--worker-id", type=int, required=True)
    w.add_argument("--policies", required=True)
    w.add_argument("--batch", type=int, default=1)
    w.add_argument("--profile", action="store_true")
    args = ap.parse_args()
    for m in args.modes.split(","):
        if m and m not in MODES:
            ap.error(f"unknown mode {m}; choose from {MODES}")
    if args.cmd == "host":
        host(args)
    else:
        worker(args)


if __name__ == "__main__":
    main()
