"""Shared tracing for VLA workloads: step latency (eager and, where possible, CUDA graph) and a
per-kernel list from torch.profiler (CUPTI), linked back to the PyTorch op and input shapes.

Outputs in <out_dir>/:
  meta.json          model, GPU, versions, dtype, latency stats
  latency.csv        one row per timed step: mode,step,wall_ms
  kernels.csv.gz     one row per GPU kernel/memcpy/memset in the profiled steps
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
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import torch
from torch.profiler import ProfilerActivity, profile, record_function
from trace_parse import summarize_trace

def _sync_time(fn: Callable[[], object]) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3


def _cpu_model() -> str:
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def time_steps(fn: Callable[[], object], warmup: int, reps: int) -> list[float]:
    for _ in range(warmup):
        fn()
    return [_sync_time(fn) for _ in range(reps)]


def profile_and_save(
    out_dir: str | Path,
    model_name: str,
    step_fn: Callable[[], object],
    meta: dict,
    warmup: int = 5,
    reps: int = 20,
    prof_steps: int = 3,
    graph_step_fn: Callable[[], object] | None = None,
    extra_modes: dict[str, Callable[[], object]] | None = None,
) -> None:
    """Time step_fn ("eager"), graph_step_fn ("graph") and each extra mode, then profile
    prof_steps calls of step_fn. Extra modes add <name>_ms_median/_std to meta.json."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    lat_rows = []
    eager = time_steps(step_fn, warmup, reps)
    lat_rows += [{"mode": "eager", "step": i, "wall_ms": t} for i, t in enumerate(eager)]
    if graph_step_fn is not None:
        g = time_steps(graph_step_fn, warmup, reps)
        lat_rows += [{"mode": "graph", "step": i, "wall_ms": t} for i, t in enumerate(g)]
    extra_stats = {}
    for name, fn in (extra_modes or {}).items():
        t = time_steps(fn, warmup, reps)
        lat_rows += [{"mode": name, "step": i, "wall_ms": v} for i, v in enumerate(t)]
        extra_stats[f"{name}_ms_median"] = statistics.median(t)
        extra_stats[f"{name}_ms_std"] = statistics.stdev(t) if len(t) > 1 else 0.0
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
    with open(trace, "rb") as f, gzip.open(str(trace) + ".gz", "wb") as g:
        g.write(f.read())
    trace.unlink()
    trace_stats = summarize_trace(out)

    dev = torch.cuda.get_device_properties(0)
    summary = {
        "model": model_name,
        "gpu": torch.cuda.get_device_name(0),
        "sm_count": dev.multi_processor_count,
        "host": platform.node(),
        "cpu": _cpu_model(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_module_loading": os.environ.get("CUDA_MODULE_LOADING", "LAZY(default)"),
        "eager_ms_median": statistics.median(eager),
        "eager_ms_mean": statistics.fmean(eager),
        "eager_ms_std": statistics.stdev(eager) if len(eager) > 1 else 0.0,
        **trace_stats,
        **extra_stats,
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
