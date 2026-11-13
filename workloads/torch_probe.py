"""Target-side probes for the cross-GPU model (model/cross_gpu.py), complementing
workloads/torch_dispatch.py. Two parts, both a few seconds:

1. Host Python speed (no GPU work). The eager VLA step spends ~15 to 25 us per kernel in
   Python/HF module code beyond aten dispatch (the fitted residual p). It scales with the
   host's Python speed, which the aten dispatch table does not capture (cuda3/cuda2: dispatch
   ratio 0.70, true p ratio 0.61). Measured per call, host timer:
     py_loop        a pure-Python loop body with attribute, dict and method work
     module_call    nn.Module.__call__ of a module whose forward does nothing
     module_tree    forward through a 4-deep tree of nested nn.Modules (HF style), CPU tensor

2. Small PyTorch kernels on COLD data, replayed in a CUDA graph (GPU time + node cost per
   kernel). In the real denoise loop the GEMMs stream ~250 MB of weights per step, so the
   activations a small kernel reads are often no longer in L2 and its time includes DRAM
   latency. torch_dispatch.py's chains reuse one tiny tensor (L1/L2 hot), which misses this
   (TITAN V vs 2080 Ti: 1.2x hot vs ~1.5x in the real loop). Here each op in the chain uses a
   different tensor from a pool of POOL_MB, far larger than any L2 we have (<= 50 MB).
     add_cold, cast_cold, rmsnorm6_cold, linear_cold (each linear has its own weight)
   plus the same ops hot (one tensor) for the hot/cold ratio.

CSV schema as in CLAUDE.md section 9; bench = torch_probe.
"""

from __future__ import annotations

import os
import statistics
import sys
import time
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = "cuda"
N_OPS = 400
REPS = 20
POOL_MB = 512


def stats(xs: list[float]) -> list[str]:
    return [f"{statistics.fmean(xs):.4f}", f"{statistics.stdev(xs):.4f}", f"{min(xs):.4f}",
            f"{statistics.median(xs):.4f}"]


# ---------------------------------------------------------------- host Python probes

class _Obj:
    def __init__(self) -> None:
        self.a, self.d = 1, {"k": 2}

    def m(self, x: int) -> int:
        return x + self.a + self.d["k"]


class Empty(nn.Module):
    def forward(self, x):
        return x


class Tree(nn.Module):
    def __init__(self, depth: int) -> None:
        super().__init__()
        self.kids = nn.ModuleList([Tree(depth - 1)] if depth > 1 else [Empty()])
        self.scale = 1.0

    def forward(self, x):
        for k in self.kids:
            x = k(x)
        return x


def host_probes() -> list[tuple[str, str, Callable[[int], None], int]]:
    o = _Obj()

    def py_loop(n: int) -> None:
        s = 0
        for i in range(n):
            s = o.m(i) + len(o.d)

    e, t = Empty(), Tree(4)
    x = torch.zeros(4)

    def module_call(n: int) -> None:
        for _ in range(n):
            e(x)

    def module_tree(n: int) -> None:
        for _ in range(n):
            t(x)

    return [("py_loop", "iters=20000", py_loop, 20000),
            ("module_call", "iters=5000", module_call, 5000),
            ("module_tree", "depth=4;iters=2000", module_tree, 2000)]


# ---------------------------------------------------------------- cold GPU probes

def rmsnorm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    xf = x.to(torch.float32)
    var = xf.pow(2).mean(-1, keepdim=True)
    xf = xf * torch.rsqrt(var + 1e-6)
    return w * xf.to(x.dtype)


def gpu_cases() -> list[tuple[str, str, list[Callable[[], object]], int]]:
    """(kernel id, config, list of N_OPS callables, kernels per call). Cold: every callable
    touches its own tensors, spread over a pool of POOL_MB."""
    out = []
    f16 = torch.float16

    def pool(shape, dtype, n):
        per = torch.empty(shape, dtype=dtype).numel() * torch.empty(0, dtype=dtype).element_size()
        n_alloc = max(n, POOL_MB * 2**20 // per)
        big = torch.randn((n_alloc, *shape), device=DEV).to(dtype)
        idx = torch.randperm(n_alloc)[:n].tolist()
        return [big[i] for i in idx], big

    keep = []
    for kid, shape, dt in (("add", (50, 720), f16), ("add", (241, 960), f16),
                           ("add_f32", (50, 720), torch.float32)):
        a, ba = pool(shape, dt, N_OPS)
        c = torch.empty(shape, device=DEV, dtype=dt)
        keep += [ba]
        tag = f"shape={shape[0]}x{shape[1]};dtype={str(dt).split('.')[-1]}"
        out.append((kid + "_cold", tag, [lambda a=a[i], c=c: torch.add(a, a, out=c)
                                         for i in range(N_OPS)], 1))
        out.append((kid + "_hot", tag, [lambda a=a[0], c=c: torch.add(a, a, out=c)] * N_OPS, 1))
    a, ba = pool((50, 720), f16, N_OPS)
    keep += [ba]
    out.append(("cast_cold", "shape=50x720;f16->f32",
                [lambda a=a[i]: a.to(torch.float32) for i in range(N_OPS)], 1))
    out.append(("cast_hot", "shape=50x720;f16->f32",
                [lambda a=a[0]: a.to(torch.float32)] * N_OPS, 1))
    w = torch.ones(720, device=DEV, dtype=f16)
    out.append(("rmsnorm6_cold", "shape=50x720",
                [lambda a=a[i]: rmsnorm(a, w) for i in range(N_OPS // 6)], 6))
    out.append(("rmsnorm6_hot", "shape=50x720",
                [lambda a=a[0]: rmsnorm(a, w)] * (N_OPS // 6), 6))
    for m, k, n in ((50, 720, 2048), (50, 960, 720)):
        n_lin = 100
        ws, bw = pool((n, k), f16, n_lin)
        x = torch.randn(m, k, device=DEV, dtype=f16)
        keep += [bw]
        tag = f"m={m};k={k};n={n}"
        out.append(("linear_cold", tag, [lambda w=ws[i], x=x: F.linear(x, w)
                                         for i in range(n_lin)], 1))
        out.append(("linear_hot", tag, [lambda w=ws[0], x=x: F.linear(x, w)] * n_lin, 1))
    gpu_cases.keep = keep  # keep pools alive
    return out


def main() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    gpu = torch.cuda.get_device_name(0)
    loading = os.environ.get("CUDA_MODULE_LOADING", "LAZY(default)")
    tv = f"torch={torch.__version__};cuda={torch.version.cuda}"
    w = sys.stdout.write
    w("gpu,bench,kernel,config,mode,timer,n_runs,mean_us,std_us,min_us,median_us,"
      "module_loading\n")
    for kid, cfg, fn, n in host_probes():
        fn(n // 10)
        xs = []
        for _ in range(REPS):
            t0 = time.perf_counter()
            fn(n)
            xs.append((time.perf_counter() - t0) / n * 1e6)
        w(",".join([gpu, "torch_probe", kid, f"{cfg};{tv}", "host", "host", str(REPS),
                    *stats(xs), loading]) + "\n")
    sys.stdout.flush()
    for kid, cfg, ops, kpc in gpu_cases():

        def chain(ops=ops):
            for op in ops:
                op()

        for _ in range(3):
            chain()
        torch.cuda.synchronize()
        eager = []
        for _ in range(REPS):
            t0 = time.perf_counter()
            chain()
            torch.cuda.synchronize()
            eager.append((time.perf_counter() - t0) / len(ops) * 1e6)
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            chain()
        torch.cuda.current_stream().wait_stream(s)
        with torch.cuda.graph(g):
            chain()
        for _ in range(3):
            g.replay()
        torch.cuda.synchronize()
        st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        graph = []
        for _ in range(REPS):
            st.record()
            g.replay()
            en.record()
            en.synchronize()
            graph.append(st.elapsed_time(en) * 1e3 / len(ops))
        full = f"{cfg};kernels_per_op={kpc};n_ops={len(ops)};pool_mb={POOL_MB};{tv}"
        for mode, timer, xs in (("eager", "host", eager), ("graph", "event", graph)):
            w(",".join([gpu, "torch_probe", kid, full, mode, timer, str(REPS), *stats(xs),
                        loading]) + "\n")
        sys.stdout.flush()
        del g


if __name__ == "__main__":
    main()
