"""Per-op PyTorch dispatch cost vs GPU time, eager vs CUDA graph (calibration for the launch
term of the model).

The VLA traces show ~27 us of CPU per kernel in eager mode on the cluster hosts, about 10x the
bare cudaLaunchKernel cost (bench/calib/launch_latency). This benchmark measures that framework
cost directly, without a VLA: a chain of N identical ops on one stream, timed

  eager        host wall time per op, synchronized at the end of the chain
  issue        host time per op to enqueue the chain (before the final sync): CPU cost only
  graph        the same chain captured once and replayed: GPU time + per-node cost

so per op, eager ~ max(issue, graph) when CPU and GPU overlap, and the crossover size where
the GPU becomes the bottleneck is visible in the add_n sweep.

Prints CSV in the calibration schema (CLAUDE.md section 9):
gpu,bench,kernel,config,mode,timer,n_runs,mean_us,std_us,min_us,median_us,module_loading
Run under each torch build used by the workloads (torch version goes into config).

--set conv measures the FP32 convolution ops of the Diffusion Policy and ACT workloads instead
(cuDNN, TF32 off); write it to torch_dispatch_conv_<env>_env.csv next to the main table.
kernels_per_op for these is counted with torch.profiler, since cuDNN may launch helper kernels.
"""

from __future__ import annotations

import os
import statistics
import sys
import time
from collections.abc import Callable

import torch
import torch.nn.functional as F

DEV = "cuda"
DT = torch.float16
N_OPS = 400
REPS = 20


def rmsnorm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    # Same op sequence as HF Llama/SmolLM RMSNorm: 6 small kernels.
    xf = x.to(torch.float32)
    var = xf.pow(2).mean(-1, keepdim=True)
    xf = xf * torch.rsqrt(var + 1e-6)
    return w * xf.to(x.dtype)


def cases() -> list[tuple[str, str, Callable[[], object], int]]:
    """(kernel id, config, op, kernels per call)."""
    out = []
    for n in (1 << 10, 1 << 14, 1 << 17, 1 << 20, 1 << 22, 1 << 24):
        a = torch.randn(n, device=DEV, dtype=DT)
        b = torch.randn(n, device=DEV, dtype=DT)
        c = torch.empty_like(a)
        out.append(("add_n", f"numel={n}", lambda a=a, b=b, c=c: torch.add(a, b, out=c), 1))
    a = torch.randn(1024, device=DEV, dtype=DT)
    out.append(("add_alloc", "numel=1024", lambda a=a: a + a, 1))
    for m, k, n, tag in ((50, 720, 2048, "smolvla_expert_mlp"), (50, 960, 720, "smolvla_kv_proj"),
                         (1, 4096, 4096, "openvla_decode_qkvo"),
                         (1, 4096, 11008, "openvla_decode_mlp"),
                         (283, 4096, 4096, "openvla_prefill_qkvo")):
        x = torch.randn(m, k, device=DEV, dtype=DT)
        w = torch.randn(n, k, device=DEV, dtype=DT)
        out.append(("linear", f"m={m};k={k};n={n};shape={tag}",
                    lambda x=x, w=w: F.linear(x, w), 1))
    for d in (720, 4096):
        x = torch.randn(1, 50, d, device=DEV, dtype=DT)
        w = torch.ones(d, device=DEV, dtype=DT)
        out.append(("rmsnorm6", f"shape=1x50x{d}", lambda x=x, w=w: rmsnorm(x, w), 6))
    q = torch.randn(1, 15, 50, 64, device=DEV, dtype=DT)
    kv = torch.randn(1, 15, 291, 64, device=DEV, dtype=DT)
    out.append(("sdpa", "q=1x15x50x64;kv=1x15x291x64",
                lambda q=q, kv=kv: F.scaled_dot_product_attention(q, kv, kv), 1))
    return out


def count_kernels(op: Callable[[], object]) -> int:
    from torch.profiler import ProfilerActivity, profile
    op()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        op()
        torch.cuda.synchronize()
    n = sum(1 for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA)
    return max(n, 1)


def conv_cases() -> list[tuple[str, str, Callable[[], object], int]]:
    """FP32 convolutions at Diffusion Policy (UNet Conv1d over the 16-step horizon) and ACT
    (ResNet18 on 480x640) shapes, with bias as in the models."""
    f32 = torch.float32
    out = []
    for c_in, c_out, length, tag in ((512, 512, 16, "diffusion_unet_512"),
                                     (2048, 2048, 4, "diffusion_unet_2048")):
        x = torch.randn(1, c_in, length, device=DEV, dtype=f32)
        w = torch.randn(c_out, c_in, 5, device=DEV, dtype=f32)
        b = torch.randn(c_out, device=DEV, dtype=f32)
        op = lambda x=x, w=w, b=b: F.conv1d(x, w, b, padding=2)  # noqa: E731
        out.append(("conv", f"conv1d;c={c_in}x{c_out};l={length};k=5;shape={tag}", op,
                    count_kernels(op)))
    for c, hw, tag in ((64, (120, 160), "act_layer1"), (64, (24, 24), "diffusion_layer1"),
                       (512, (15, 20), "act_layer4")):
        x = torch.randn(1, c, *hw, device=DEV, dtype=f32)
        w = torch.randn(c, c, 3, 3, device=DEV, dtype=f32)
        op = lambda x=x, w=w: F.conv2d(x, w, padding=1)  # noqa: E731
        out.append(("conv", f"conv2d;c={c};hw={hw[0]}x{hw[1]};k=3;shape={tag}", op,
                    count_kernels(op)))
    return out


def timed(fn: Callable[[], None]) -> tuple[float, float]:
    """Returns (issue_s, total_s) for one call of fn (a chain)."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    t1 = time.perf_counter()
    torch.cuda.synchronize()
    return t1 - t0, time.perf_counter() - t0


def stats(xs: list[float]) -> list[str]:
    return [f"{statistics.fmean(xs):.4f}", f"{statistics.stdev(xs):.4f}", f"{min(xs):.4f}",
            f"{statistics.median(xs):.4f}"]


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["main", "conv"], default="main")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    gpu = torch.cuda.get_device_name(0)
    loading = os.environ.get("CUDA_MODULE_LOADING", "LAZY(default)")
    tv = f"torch={torch.__version__};cuda={torch.version.cuda};n_ops={N_OPS}"
    w = sys.stdout.write
    w("gpu,bench,kernel,config,mode,timer,n_runs,mean_us,std_us,min_us,median_us,"
      "module_loading\n")
    for kid, cfg, op, kpc in (conv_cases() if a.set == "conv" else cases()):

        def chain(op=op):
            for _ in range(N_OPS):
                op()

        for _ in range(3):
            chain()
        issue, eager = [], []
        for _ in range(REPS):
            i, t = timed(chain)
            issue.append(i / N_OPS * 1e6)
            eager.append(t / N_OPS * 1e6)
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
        graph = [timed(g.replay)[1] / N_OPS * 1e6 for _ in range(REPS)]
        full = f"{cfg};kernels_per_op={kpc};{tv}"
        for mode, xs in (("eager", eager), ("issue", issue), ("graph", graph)):
            w(",".join([gpu, "torch_dispatch", kid, full, mode, "host", str(REPS), *stats(xs),
                        loading]) + "\n")
        sys.stdout.flush()
        del g


if __name__ == "__main__":
    main()
