"""Shared tracing for VLA workloads: step latency (eager and, where possible, CUDA graph) and a
per-kernel list from torch.profiler (CUPTI), linked back to the PyTorch op and input shapes.

Outputs in <out_dir>/:
  meta.json          model, GPU, versions, dtype, latency stats
  latency.csv        one row per timed step: mode,step,wall_ms
  kernels.csv        one row per GPU kernel/memcpy/memset in the profiled steps
  ops.csv            aggregated per (op, input shapes): count, kernels, gpu_us
  trace.json.gz      raw chrome trace (open in chrome://tracing or Perfetto)

Why torch.profiler and not nsys: nsys is not installed on the cluster, and torch.profiler reads
the same CUPTI activity records (kernel name, duration, grid, block, registers, shared memory)
and also links each kernel to the aten op and input shapes that launched it, which nsys cannot.
"""

from __future__ import annotations

import gzip
import json
import os
import platform
import statistics
import time
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import torch
from torch.profiler import ProfilerActivity, profile, record_function

GPU_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}


def _sync_time(fn: Callable[[], object]) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3


def time_steps(fn: Callable[[], object], warmup: int, reps: int) -> list[float]:
    for _ in range(warmup):
        fn()
    return [_sync_time(fn) for _ in range(reps)]


def _parse_trace(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    with open(path) as f:
        events = json.load(f)["traceEvents"]
    steps = sorted(
        (
            e
            for e in events
            if e.get("ph") == "X" and str(e.get("name", "")).startswith("vla_step_")
        ),
        key=lambda e: e["ts"],
    )
    # Map External id -> innermost aten op (name, shapes). Kernels carry the External id of the
    # op that launched them.
    ops: dict[int, dict] = {}
    for e in events:
        if e.get("cat") == "cpu_op":
            ext = e.get("args", {}).get("External id")
            if ext is not None:
                ops[ext] = e
    rows = []
    for e in events:
        if e.get("cat") not in GPU_CATS or e.get("ph") != "X":
            continue
        a = e.get("args", {})
        op = ops.get(a.get("External id"), {})
        rows.append(
            {
                "cat": e["cat"],
                "name": e["name"],
                "ts_us": float(e["ts"]),
                "dur_us": float(e["dur"]),
                "correlation": a.get("correlation"),
                "stream": a.get("stream"),
                "grid": "x".join(map(str, a.get("grid", []))),
                "block": "x".join(map(str, a.get("block", []))),
                "regs_per_thread": a.get("registers per thread"),
                "smem_bytes": a.get("shared memory"),
                "blocks_per_sm": a.get("blocks per SM"),
                "warps_per_sm": a.get("warps per SM"),
                "est_occupancy_pct": a.get("est. achieved occupancy %"),
                "op": op.get("name", ""),
                "op_input_dims": json.dumps(op.get("args", {}).get("Input Dims", [])),
                "op_input_types": json.dumps(op.get("args", {}).get("Input type", [])),
            }
        )
    k = pd.DataFrame(rows).sort_values("ts_us").reset_index(drop=True)
    # Assign GPU events to steps through the CPU runtime call that launched them (same
    # correlation id): GPU work lags the CPU, so GPU timestamps cannot be used directly.
    launches = sorted(
        (
            e
            for e in events
            if e.get("cat") == "cuda_runtime"
            and e.get("ph") == "X"
            and e.get("args", {}).get("correlation") is not None
        ),
        key=lambda e: e["ts"],
    )
    corr_step: dict[int, int] = {}
    starts = [s["ts"] for s in steps] + [float("inf")]
    j = 0
    for e in launches:
        while j < len(steps) and e["ts"] >= starts[j + 1]:
            j += 1
        if steps and starts[0] <= e["ts"]:
            corr_step[e["args"]["correlation"]] = j
    k["step"] = k.correlation.map(corr_step)
    k = k[k.step.notna()].copy()
    k["step"] = k.step.astype(int)
    k["seq"] = k.groupby("step").cumcount()
    gap = k.groupby("step").ts_us.diff() - k.groupby("step").dur_us.shift()
    k["gap_before_us"] = gap
    # CPU runtime calls per step (cudaLaunchKernel, cudaGraphLaunch, cudaMemcpyAsync, ...).
    cpu_launch = defaultdict(int)
    for e in launches:
        s = corr_step.get(e["args"]["correlation"])
        if s is not None:
            cpu_launch[(s, e["name"])] += 1
    launch_df = pd.DataFrame(
        [{"step": s, "api": n, "count": c} for (s, n), c in sorted(cpu_launch.items())]
    )
    return k, launch_df


def profile_and_save(
    out_dir: str | Path,
    model_name: str,
    step_fn: Callable[[], object],
    meta: dict,
    warmup: int = 5,
    reps: int = 20,
    prof_steps: int = 3,
    graph_step_fn: Callable[[], object] | None = None,
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    lat_rows = []
    eager = time_steps(step_fn, warmup, reps)
    lat_rows += [{"mode": "eager", "step": i, "wall_ms": t} for i, t in enumerate(eager)]
    if graph_step_fn is not None:
        g = time_steps(graph_step_fn, warmup, reps)
        lat_rows += [{"mode": "graph", "step": i, "wall_ms": t} for i, t in enumerate(g)]
    lat = pd.DataFrame(lat_rows)
    lat.to_csv(out / "latency.csv", index=False)

    trace = out / "trace.json"
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True
    ) as prof:
        for i in range(prof_steps):
            with record_function(f"vla_step_{i}"):
                step_fn()
            torch.cuda.synchronize()
    prof.export_chrome_trace(str(trace))
    kernels, launches = _parse_trace(trace)
    with open(trace, "rb") as f, gzip.open(str(trace) + ".gz", "wb") as g:
        g.write(f.read())
    trace.unlink()
    kernels.to_csv(out / "kernels.csv", index=False)
    launches.to_csv(out / "launch_api_calls.csv", index=False)
    kk = kernels[kernels.cat == "kernel"]
    ops = (
        kk.groupby(["op", "op_input_dims"])
        .agg(kernels=("name", "size"), gpu_us=("dur_us", "sum"))
        .reset_index()
        .sort_values("gpu_us", ascending=False)
    )
    ops["kernels"] //= prof_steps
    ops["gpu_us"] /= prof_steps
    ops.to_csv(out / "ops.csv", index=False)

    per_step = kk.groupby("step").agg(n=("name", "size"), busy_us=("dur_us", "sum"))
    dev = torch.cuda.get_device_properties(0)
    summary = {
        "model": model_name,
        "gpu": torch.cuda.get_device_name(0),
        "sm_count": dev.multi_processor_count,
        "host": platform.node(),
        "cpu": platform.processor() or platform.machine(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_module_loading": os.environ.get("CUDA_MODULE_LOADING", "LAZY(default)"),
        "eager_ms_median": statistics.median(eager),
        "eager_ms_mean": statistics.fmean(eager),
        "eager_ms_std": statistics.stdev(eager) if len(eager) > 1 else 0.0,
        "kernels_per_step": float(per_step.n.median()),
        "gpu_busy_ms_per_step": float(per_step.busy_us.median()) / 1e3,
        "kernel_dur_us_median": float(kk.dur_us.median()),
        "kernel_dur_us_p90": float(kk.dur_us.quantile(0.9)),
        "frac_kernels_under_5us": float((kk.dur_us < 5).mean()),
        **meta,
    }
    if graph_step_fn is not None:
        gm = lat[lat["mode"] == "graph"].wall_ms
        summary["graph_ms_median"] = float(gm.median())
        summary["graph_speedup"] = summary["eager_ms_median"] / summary["graph_ms_median"]
    summary["gpu_busy_frac"] = summary["gpu_busy_ms_per_step"] / summary["eager_ms_median"]
    (out / "meta.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def default_out(model: str) -> Path:
    name = torch.cuda.get_device_name(0).replace(" ", "_")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return Path("traces") / model / f"{name}_{platform.node().split('.')[0]}_{stamp}"
