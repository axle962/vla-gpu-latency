"""Launch-aware latency model, version 1: same-GPU prediction of eager, CUDA-graph and hybrid
step time from a traced kernel list plus calibration microbenchmarks.

Inputs
  traces/<model>_<config>/<run>/   kernels.csv.gz (per GPU event: duration, op, step), meta.json
  results/calib/<gpu run>/         launch_eager_loading.csv (C++ launch_latency) and
                                   torch_dispatch_<env>_env.csv (workloads/torch_dispatch.py)

Model (CLAUDE.md section 4.2, revised by the A100 calibration)
  Eager: replay the step on two timelines. The CPU spends d_k on kernel k, where d_k is the
  per-op PyTorch dispatch cost of k's aten op class (calibration table) plus a per-kernel
  Python residual p. The GPU starts k at max(cpu_k, gpu_end_{k-1} + g_eager) and runs it for
  its traced duration t_k; g_eager is the GPU-side per-kernel cost of stream launches (spin
  chains). Host-device copies (HtoD/DtoH memcpy) are synchronous: the CPU waits for the GPU.
    T_eager = max(cpu_end, gpu_end)
  Graph: T_graph = L_graph + sum_k (t_k + delta_node), over the captured kernels.
  Hybrid (graph-capable workloads): eager prefix + graph loop.

Fitted parameter: p (seconds of Python/framework code per kernel not covered by single-op
dispatch costs), fitted on ONE config per model and held fixed for the others. Everything else
comes from microbenchmarks or the trace.

Baselines
  gpu_only      sum of kernel durations (what a per-kernel roofline/VLA-Perf-style model
                predicts at best: no launch, no host costs)
  driver_launch two-timeline replay with d_k = bare cudaLaunchKernel cost, no Python
  table         two-timeline replay with the per-op dispatch table, p = 0
  full          table + fitted p

Usage (login node is fine; ~1 GB RAM peak):
  envs/smolvla/bin/python model/launch_model.py > results/predictions_same_gpu.csv
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# aten op -> dispatch-table kernel id (workloads/torch_dispatch.py)
OP_CLASS = {
    "aten::mm": "linear", "aten::addmm": "linear", "aten::bmm": "linear",
    "aten::linear": "linear", "aten::matmul": "linear", "aten::baddbmm": "linear",
    "aten::_efficient_attention_forward": "sdpa", "aten::_flash_attention_forward": "sdpa",
    "aten::_scaled_dot_product_efficient_attention": "sdpa",
    "aten::_scaled_dot_product_flash_attention": "sdpa",
    "aten::scaled_dot_product_attention": "sdpa",
    "aten::copy_": "add_alloc", "aten::to": "add_alloc", "aten::_to_copy": "add_alloc",
    "aten::cat": "add_alloc", "aten::clone": "add_alloc", "aten::contiguous": "add_alloc",
    "aten::index": "add_alloc", "aten::index_put_": "add_alloc", "aten::fill_": "add_alloc",
    "aten::arange": "add_alloc", "aten::full": "add_alloc", "aten::zeros": "add_alloc",
    "aten::ones": "add_alloc", "aten::empty": "add_alloc",
}


@dataclass
class Calib:
    g_eager_us: float      # GPU-side per-kernel cost of eager stream launches
    delta_node_us: float   # per-node cost inside a graph
    graph_call_us: float   # host cost of one cudaGraphLaunch (+ sync)
    launch_call_us: float  # bare cudaLaunchKernel CPU cost
    dispatch_us: dict      # op class -> PyTorch CPU issue cost per op


def load_calib(calib_dir: Path, dispatch_csv: Path) -> Calib:
    rows = list(csv.DictReader(open(calib_dir / "launch_eager_loading.csv")))

    def med(kernel: str, cfg: str, mode: str, timer: str) -> float:
        for r in rows:
            if (r["kernel"], r["config"], r["mode"], r["timer"]) == (kernel, cfg, mode, timer):
                return float(r["median_us"])
        raise KeyError((kernel, cfg, mode, timer))

    n = 100
    ge, dn = [], []
    for spin_ns in (1000, 10000, 15000):
        cfg = f"n={n};spin_ns={spin_ns}"
        spin_total = n * spin_ns / 1e3
        ge.append((med("spin_chain_total", cfg, "eager", "event") - spin_total) / n)
        dn.append((med("spin_chain_total", cfg, "graph", "event") - spin_total) / n)
    disp: dict[str, list[float]] = {}
    for r in csv.DictReader(open(dispatch_csv)):
        if r["mode"] != "issue":
            continue
        cfg = r["config"]
        kpo = int(next(x for x in cfg.split(";") if x.startswith("kernels_per_op=")).split("=")[1])
        kid = r["kernel"]
        if kid == "add_n" and "numel=1024;" not in cfg:
            continue  # issue cost does not depend on size; use the tiny case
        if kid == "linear" and "m=1;" in cfg:
            kid = "linear_gemv"
        disp.setdefault(kid, []).append(float(r["median_us"]) / kpo)
    dispatch = {k: float(np.median(v)) for k, v in disp.items()}
    dispatch.setdefault("linear_gemv", dispatch["linear"])
    return Calib(
        g_eager_us=float(np.median(ge)),
        delta_node_us=float(np.median(dn)),
        graph_call_us=med("host_graph_call", "n=1", "graph", "host"),
        launch_call_us=med("host_launch_call", "batch=256", "eager", "host"),
        dispatch_us=dispatch,
    )


def op_class(op: str, dims: str) -> str:
    c = OP_CLASS.get(op, "add_n")
    if c == "linear" and dims.startswith("[[1, "):
        return "linear_gemv"
    return c


def simulate_eager(t_us: np.ndarray, d_us: np.ndarray, sync: np.ndarray, g_eager: float) -> float:
    cpu = 0.0
    gpu_end = 0.0
    for t, d, s in zip(t_us, d_us, sync):
        cpu += d
        start = max(cpu, gpu_end + g_eager)
        gpu_end = start + t
        if s:
            cpu = gpu_end
    return max(cpu, gpu_end)


def find_loop(names: list[str], n_iter: int, min_period: int = 50) -> int:
    """Length P of the repeating tail: the last n_iter * P events repeat with period P.
    Returns the best P (highest match fraction, ties to the smallest)."""
    best, best_score = 0, 0.0
    a = np.array(names, dtype=object)
    for p in range(min_period, len(a) // n_iter + 1):
        tail = a[-n_iter * p:]
        score = float((tail[p:] == tail[:-p]).mean())
        if score > best_score + 1e-9:
            best, best_score = p, score
    return best if best_score > 0.95 else 0


@dataclass
class StepTrace:
    t_us: np.ndarray
    cls: list
    sync: np.ndarray
    loop_start: int  # index where the repeated loop (denoise / decode) starts


def load_step(run: Path, n_iter: int) -> StepTrace:
    k = pd.read_csv(run / "kernels.csv.gz",
                    usecols=["cat", "name", "dur_us", "op", "op_input_dims", "step"])
    k = k[k.cat != "gpu_memset"]
    s = k[k.step == k.step.min()].reset_index(drop=True)
    sync = s.name.str.contains("HtoD|DtoH", regex=True).to_numpy()
    cls = [op_class(o, d) for o, d in zip(s.op.fillna(""), s.op_input_dims.fillna(""))]
    p = find_loop(s.name.tolist(), n_iter)
    return StepTrace(s.dur_us.to_numpy(float), cls, sync, len(s) - n_iter * p if p else -1)


def eager(tr: StepTrace, c: Calib, sl: slice, mode: str, p_us: float = 0.0) -> float:
    t, sync = tr.t_us[sl], tr.sync[sl]
    if mode == "gpu_only":
        return float(t.sum())
    if mode == "driver_launch":
        d = np.full(len(t), c.launch_call_us)
        return simulate_eager(t, d, sync, c.g_eager_us)
    d = np.array([c.dispatch_us[x] for x in tr.cls[sl]]) + p_us
    return simulate_eager(t, d, sync, c.g_eager_us)


def graph(tr: StepTrace, c: Calib, sl: slice) -> float:
    t = tr.t_us[sl]
    return c.graph_call_us + float((t + c.delta_node_us).sum())


def fit_p(tr: StepTrace, c: Calib, target_us: float) -> float:
    lo, hi = 0.0, 200.0
    for _ in range(50):
        mid = (lo + hi) / 2
        if eager(tr, c, slice(None), "table", mid) < target_us:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def main() -> None:
    calib_dir = ROOT / "results/calib/NVIDIA_A100-SXM4-40GB_b-19-11_20261110-231542"
    disp_dir = next((ROOT / "results/calib").glob("NVIDIA_A100*_b-19-28_*"))
    runs = []
    for meta_p in sorted((ROOT / "traces").glob("*/*/meta.json")):
        m = json.loads(meta_p.read_text())
        if "A100" not in m["gpu"]:
            continue
        runs.append((meta_p.parent, m))

    out = csv.writer(sys.stdout)
    out.writerow(["model", "config", "run", "quantity", "measured_ms", "predictor",
                  "predicted_ms", "error_pct", "fit_on_this_run"])
    for model in ("smolvla", "openvla-7b"):
        mruns = [(r, m) for r, m in runs if m["model"] == model]
        env = "smolvla" if model == "smolvla" else "openvla"
        c = load_calib(calib_dir, disp_dir / f"torch_dispatch_{env}_env.csv")
        print(f"# {model}: {c}", file=sys.stderr)
        traces = {}
        for r, m in mruns:
            n_iter = m.get("num_steps", 10) if model == "smolvla" else m["action_tokens"] - 1
            traces[r] = load_step(r, n_iter)
            print(f"#   {r.parent.name}/{r.name}: {len(traces[r].t_us)} events, loop starts at "
                  f"{traces[r].loop_start}", file=sys.stderr)
        # Fit p on the baseline config only (the first run of it).
        if model == "smolvla":
            fit_run = next(r for r, m in mruns
                           if (m["cameras"], m["num_steps"], m["chunk_size"]) == (3, 10, 50))
        else:
            fit_run = mruns[0][0]
        fit_meta = dict(mruns)[fit_run]
        p_us = fit_p(traces[fit_run], c, fit_meta["eager_ms_median"] * 1e3)
        print(f"#   fitted Python residual p = {p_us:.2f} us per kernel on {fit_run.name}",
              file=sys.stderr)

        for r, m in mruns:
            tr = traces[r]
            cfg = r.parent.name
            full, pre = slice(None), slice(0, tr.loop_start)
            loop = slice(tr.loop_start, None)
            q: dict[str, tuple[float, dict]] = {}
            preds_eager = {mode: eager(tr, c, full, mode) for mode in
                           ("gpu_only", "driver_launch", "table")}
            preds_eager["full"] = eager(tr, c, full, "table", p_us)
            q["eager"] = (m["eager_ms_median"], preds_eager)
            if tr.loop_start > 0:
                pre_key = "eager_prefix_ms_median" if model == "smolvla" else \
                    "eager_prefill_ms_median"
                if m.get(pre_key):
                    q["eager_prefix"] = (m[pre_key], {
                        "gpu_only": eager(tr, c, pre, "gpu_only"),
                        "driver_launch": eager(tr, c, pre, "driver_launch"),
                        "table": eager(tr, c, pre, "table"),
                        "full": eager(tr, c, pre, "table", p_us)})
                loop_key = "eager_denoise_ms_median" if model == "smolvla" else \
                    "static_decode_ms_median"
                if m.get(loop_key):
                    q["eager_loop"] = (m[loop_key], {
                        "gpu_only": eager(tr, c, loop, "gpu_only"),
                        "driver_launch": eager(tr, c, loop, "driver_launch"),
                        "table": eager(tr, c, loop, "table"),
                        "full": eager(tr, c, loop, "table", p_us)})
                g_key = "graph_denoise_ms_median" if model == "smolvla" else \
                    "graph_decode_ms_median"
                if m.get(g_key):
                    gp = graph(tr, c, loop)
                    q["graph_loop"] = (m[g_key], {"gpu_only": eager(tr, c, loop, "gpu_only"),
                                                  "full": gp})
                    h_key = "hybrid_ms_median" if model == "smolvla" else \
                        "static_hybrid_ms_median"
                    if m.get(h_key):
                        q["hybrid"] = (m[h_key], {
                            "gpu_only": eager(tr, c, full, "gpu_only"),
                            "full": eager(tr, c, pre, "table", p_us) + gp})
            for name, (meas_ms, preds) in q.items():
                for pred_name, pred_us in preds.items():
                    pred_ms = pred_us / 1e3
                    out.writerow([model, cfg, r.name, name, f"{meas_ms:.3f}", pred_name,
                                  f"{pred_ms:.3f}", f"{100 * (pred_ms - meas_ms) / meas_ms:.1f}",
                                  int(r == fit_run)])


if __name__ == "__main__":
    main()
