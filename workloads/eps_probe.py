"""Microbenchmark for the graph-worthiness rule's per-kernel term eps, without any VLA workload.

The rule (model/graph_worthiness.py) predicts a region's graph time as T_graph ~= D + K * eps,
where D is the summed kernel time of the region as traced by torch.profiler (CUPTI) in eager
mode and K its kernel count. So eps is, per kernel,

  eps = (time per kernel when replayed in a CUDA graph, no profiler)
        - (duration of the same kernel as CUPTI traces it in eager mode)

which is the graph node cost minus the profiler's per-kernel bias. This script measures exactly
that on chains of small ops typical of batch-1 VLA inference (elementwise add, add with
allocation, the SmolVLA expert linear, a 6-kernel RMSNorm, softmax, layer norm, SiLU):

  graph     chain of N_OPS ops captured once, replayed, timed with CUDA events (per kernel)
  profiled  the same chain run eagerly under torch.profiler (CPU + CUDA activities, as in
            workloads/trace_common.py); kernel durations from the exported Chrome trace
  eps       graph - profiled, per kernel

Prints CSV in the calibration schema (CLAUDE.md section 9). Save it as
results/calib/<gpu run>/eps_probe_<env>_env.csv; graph_worthiness.py picks it up for the
"micro" variant (eps from microbenchmarks only). Takes about a minute per GPU.

  CUDA_MODULE_LOADING=EAGER python workloads/eps_probe.py > eps_probe_smolvla_env.csv
"""

from __future__ import annotations

import gzip
import json
import os
import statistics
import sys
import tempfile
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile

DEV = "cuda"
DT = torch.float16
N_OPS = 400
REPS = 20
PROF_REPS = 3


def rmsnorm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    xf = x.to(torch.float32)
    var = xf.pow(2).mean(-1, keepdim=True)
    xf = xf * torch.rsqrt(var + 1e-6)
    return w * xf.to(x.dtype)


def cases() -> list[tuple[str, str, Callable[[], object]]]:
    """(kernel id, config, op). Shapes from the SmolVLA action expert at batch 1."""
    a = torch.randn(1 << 14, device=DEV, dtype=DT)
    b = torch.randn(1 << 14, device=DEV, dtype=DT)
    c = torch.empty_like(a)
    x = torch.randn(50, 720, device=DEV, dtype=DT)
    w = torch.randn(2048, 720, device=DEV, dtype=DT)
    nw = torch.randn(720, device=DEV, dtype=DT)
    s = torch.randn(15, 50, 400, device=DEV, dtype=DT)
    return [
        ("add_n", "numel=16384", lambda: torch.add(a, b, out=c)),
        ("add_alloc", "numel=16384", lambda: a + b),
        ("linear", "m=50;k=720;n=2048", lambda: F.linear(x, w)),
        ("rmsnorm6", "m=50;d=720", lambda: rmsnorm(x, nw)),
        ("softmax", "shape=15x50x400", lambda: torch.softmax(s, -1)),
        ("layer_norm", "m=50;d=720", lambda: F.layer_norm(x, (720,), nw)),
        ("silu", "m=50;d=720", lambda: F.silu(x)),
    ]


def kernels_in(op: Callable[[], object]) -> tuple[int, list[float]]:
    """Kernel count and CUPTI-traced durations (us) of one eager chain of N_OPS ops."""
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(N_OPS):
            op()
        torch.cuda.synchronize()
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "t.json")
        prof.export_chrome_trace(path)
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt") as f:
            ev = json.load(f)["traceEvents"]
    durs = [float(e["dur"]) for e in ev if e.get("cat") == "kernel" and "dur" in e]
    return len(durs), durs


def main() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    gpu = torch.cuda.get_device_name(0)
    loading = os.environ.get("CUDA_MODULE_LOADING", "default")
    cfg_tail = f";torch={torch.__version__};cuda={torch.version.cuda};n_ops={N_OPS}"
    print("gpu,bench,kernel,config,mode,timer,n_runs,mean_us,std_us,min_us,median_us,"
          "module_loading")

    def row(kid, cfg, mode, timer, xs):
        print(",".join([gpu, "eps_probe", kid, cfg + cfg_tail, mode, timer, str(len(xs)),
                        f"{statistics.mean(xs):.4f}",
                        f"{statistics.stdev(xs) if len(xs) > 1 else 0.0:.4f}",
                        f"{min(xs):.4f}", f"{statistics.median(xs):.4f}", loading]))
        sys.stdout.flush()

    for kid, cfg, op in cases():
        for _ in range(3):  # warm up (module loading, cuBLAS handles, allocator)
            op()
        torch.cuda.synchronize()
        n_k, _ = kernels_in(op)  # first profiled pass warms CUPTI
        prof_us = []
        for _ in range(PROF_REPS):
            n_k, durs = kernels_in(op)
            prof_us.append(sum(durs) / n_k)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                op()
        torch.cuda.current_stream().wait_stream(stream)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=stream):
            for _ in range(N_OPS):
                op()
        g.replay()
        torch.cuda.synchronize()
        graph_us = []
        for _ in range(REPS):
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            g.replay()
            e1.record()
            torch.cuda.synchronize()
            graph_us.append(e0.elapsed_time(e1) * 1e3 / n_k)
        c = f"{cfg};kernels_per_op={n_k // N_OPS}"
        row(kid, c, "graph", "event", graph_us)
        row(kid, c, "profiled", "cupti", prof_us)
        g_med = statistics.median(graph_us)
        row(kid, c, "eps", "derived", [g_med - p for p in prof_us])
        del g


if __name__ == "__main__":
    main()
