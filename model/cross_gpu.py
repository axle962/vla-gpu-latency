"""Cross-GPU latency model (Project 2, flavor B): predict a VLA step's eager, CUDA-graph and
hybrid latency on a TARGET GPU from a kernel trace taken on a SOURCE GPU, plus calibration
microbenchmarks on the target. Builds on the same-GPU replay in launch_model.py.

What comes from where
  source trace     per-kernel durations s_k (one profiled step), op class, sync points
  source fit       b (profiler bias per traced kernel) and p (Python time per kernel), both
                   fitted on the source's baseline run only (launch_model.fit_b / fit_p)
  target calib     C++ launch_latency (g_eager, delta_node, graph call), bandwidth sweep
                   (BW_DRAM), PyTorch dispatch table of the target host, and the small-kernel
                   latency set from bench/kernels (graph-mode durations of tiny RMSNorm,
                   softmax, RoPE, SiLU, elementwise kernels at the traced shapes)
  never used       any trace or end-to-end timing from the target

Kernel-time transfer (s = bias-corrected source duration, tau = latency threshold)
  same      t = s                                        (kernels are GPU invariant)
  spec_bw   t = s * BWspec_S / BWspec_T                   (spec-ratio baseline)
  bw        t = s * BW_S / BW_T                           (measured DRAM bandwidth ratio)
  lat       t = s * r_lat                                 (all kernels latency bound)
  lat_bw    t = r_lat * min(s, tau) + r_bw * max(s - tau, 0)
where r_lat is the geometric mean of target/source small-kernel times (our C++ kernels) and
r_bw = BW_S/BW_T.
  slot_class  per-kernel graph slot u = t + delta_S scaled by its op class's measured slot
              ratio r_c (PyTorch graph-mode microbenchmarks on both hosts, SLOT_ROWS)
  slot        as slot_class up to tau_c (the source slot of the class's calibration op), the
              rest of the slot scaled by r_bw (transfer_slots)
Batch-1 VLA kernels are mostly a few microseconds long; their duration is set by kernel start,
a few dependent memory round trips and the clock, not by bandwidth (A100 vs 2080 Ti: the same
small-kernel times despite 2.5x the DRAM bandwidth). Traced durations also carry a profiler
bias that differs per GPU (~1.0 us A100, ~1.2 us 2080 Ti, ~0.5 us TITAN V per kernel), so
slots, not raw traced durations, are what transfers.

Host side on the target: per-op dispatch costs from the target host's table; Python residual
p_T = p_S * h, with h the median target/source ratio of the dispatch-table costs.

Usage (login node is fine, ~1 GB RAM):
  envs/smolvla/bin/python model/cross_gpu.py > results/predictions_cross_gpu.csv
"""

from __future__ import annotations

import csv
import re
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calib import bandwidth_params, device_info  # noqa: E402
from launch_model import (Calib, StepTrace, corrected, eager, fit_b, fit_p, graph,  # noqa: E402
                          load_calib, load_step)

ROOT = Path(__file__).resolve().parent.parent
TAU_US = 5.0
# Small kernels (graph-mode event timing in results/bench/*/kernels.csv) that define r_lat:
# everything under ~6 us on the source, i.e. the regime of 90% of the traced kernels.
SMALL_SRC = ("expert_rmsnorm", "vlm_rmsnorm", "expert_self_attn", "expert_cross_attn",
             "expert_q_rope", "expert_k_rope", "expert_mlp_gate", "expert_residual_f32",
             "expert_cast_f32_f16", "expert_cast_f16_f32", "scalar_fill")


@dataclass
class GPU:
    key: str               # short name used in file globs
    name: str
    calib: Calib
    bw_dram: float         # measured, GB/s
    bw_spec: float         # spec sheet (device_info), GB/s
    small_us: dict         # src tag -> graph-mode duration of the small kernel set (C++)
    dispatch_med: dict     # op class -> dispatch cost (same as calib.dispatch_us)
    slot_us: dict          # op class -> PyTorch graph-mode time per kernel (slot)
    cold_slot_us: dict     # op class -> cold-cache slot (workloads/torch_probe.py), or {}
    py_us: dict            # host Python probes (us per call), or {}
    clk_mhz: float         # sustained SM clock under the C++ flow chain (clock_probe.sh), or nan
    cpp_slot_us: dict      # op class -> summed C++ replay slot time of the flow step, or {}


def _one(pattern: str) -> Path:
    hits = sorted(ROOT.glob(pattern))
    if not hits:
        raise FileNotFoundError(pattern)
    return hits[0]


def small_kernels(bench_dir: Path) -> dict:
    out = {}
    for r in csv.DictReader(open(bench_dir / "kernels.csv")):
        if r["mode"] != "graph" or r["timer"] != "event" or "gemm" in r["kernel"]:
            continue
        src = next((x.split("=")[1] for x in r["config"].split(";") if x.startswith("src=")), "")
        if src in SMALL_SRC:
            out[src] = float(r["median_us"])
    return out


# Graph-mode rows of the PyTorch dispatch calibration (workloads/torch_dispatch.py) that
# define each op class's slot time: the real PyTorch kernels, replayed in a CUDA graph, no
# profiler. (kernel id, config substring) per class.
SLOT_ROWS = {
    "add_n": [("add_n", "numel=1024;"), ("add_n", "numel=16384;"), ("rmsnorm6", "1x50x720")],
    "add_alloc": [("add_alloc", "numel=1024;")],
    "linear": [("linear", "smolvla_expert_mlp"), ("linear", "smolvla_kv_proj")],
    "linear_gemv": [("linear", "openvla_decode_qkvo")],
    "sdpa": [("sdpa", "q=1x15x50x64")],
}


def slot_table(dispatch_csv: Path) -> dict:
    rows = [r for r in csv.DictReader(open(dispatch_csv)) if r["mode"] == "graph"]
    out = {}
    for cls, keys in SLOT_ROWS.items():
        vals = []
        for kid, sub in keys:
            for r in rows:
                if r["kernel"] == kid and sub in r["config"]:
                    kpo = int(r["config"].split("kernels_per_op=")[1].split(";")[0])
                    vals.append(float(r["median_us"]) / kpo)
        out[cls] = float(np.exp(np.mean(np.log(vals))))
    return out


# Cold-cache rows of workloads/torch_probe.py per op class (kernel id, config substring).
COLD_ROWS = {
    "add_n": [("add_cold", "50x720;dtype=float16"), ("add_cold", "241x960"),
              ("add_f32_cold", ""), ("cast_cold", ""), ("rmsnorm6_cold", "")],
    "add_alloc": [("cast_cold", ""), ("add_cold", "50x720;dtype=float16")],
    "linear": [("linear_cold", "n=2048"), ("linear_cold", "n=720")],
}


def probe_tables(probe_csv: Path | None) -> tuple[dict, dict]:
    if probe_csv is None:
        return {}, {}
    rows = list(csv.DictReader(open(probe_csv)))
    py = {r["kernel"]: float(r["median_us"]) for r in rows if r["mode"] == "host"}
    cold = {}
    for cls, keys in COLD_ROWS.items():
        vals = []
        for kid, sub in keys:
            for r in rows:
                if r["mode"] == "graph" and r["kernel"] == kid and sub in r["config"]:
                    kpo = int(r["config"].split("kernels_per_op=")[1].split(";")[0])
                    vals.append(float(r["median_us"]) / kpo)
        if vals:
            cold[cls] = float(np.exp(np.mean(np.log(vals))))
    return cold, py


def sustained_clock(calib_dir: Path) -> float:
    f = calib_dir / "sm_clock_summary.txt"
    if not f.exists():
        return float("nan")
    m = re.search(r"SM clock under load: median (\d+) MHz", f.read_text())
    return float(m.group(1)) if m else float("nan")


# C++ replay categories (results/bench/*/perop_flow_cublas.csv) per traced op class.
CPP_CATS = {"linear": ["gemm"], "linear_gemv": ["gemm"],
            "add_n": ["elementwise", "reduce", "scan", "memset", "softmax"],
            "add_alloc": ["elementwise", "reduce", "scan", "memset", "softmax"],
            "sdpa": ["gemm", "softmax"]}


def cpp_slots(bench_dir: Path) -> dict:
    f = bench_dir / "perop_flow_cublas.csv"
    if not f.exists():
        return {}
    tot: dict = {}
    for r in csv.DictReader(open(f)):
        tot[r["cat"]] = tot.get(r["cat"], 0.0) + float(r["graph_us_per_launch"])
    return {c: sum(tot.get(k, 0.0) for k in cats) for c, cats in CPP_CATS.items()}


def load_gpu(key: str, calib_glob: str, dispatch_glob: str, bench_glob: str,
             probe_glob: str) -> GPU:
    cdir = _one(calib_glob)
    info = device_info(cdir / "device_info.csv")
    bw = bandwidth_params(_one(calib_glob + "/bandwidth.csv"), float(info["l2_bytes"]))
    dcsv = _one(dispatch_glob + "/torch_dispatch_smolvla_env.csv")
    c = load_calib(cdir, dcsv)
    return GPU(key, str(info["gpu"]), c, bw["bw_dram_gbps"],
               float(info["theoretical_dram_bw_mbps"]) / 1e3,
               small_kernels(_one(bench_glob)), c.dispatch_us, slot_table(dcsv),
               *probe_tables(max(ROOT.glob(probe_glob), default=None)),
               sustained_clock(cdir), cpp_slots(_one(bench_glob)))


GPUS = {
    "A100": dict(calib_glob="results/calib/NVIDIA_A100-SXM4-40GB_b-19-11_*",
                 dispatch_glob="results/calib/NVIDIA_A100-SXM4-40GB_b-19-28_*",
                 bench_glob="results/bench/NVIDIA_A100*",
                 probe_glob="results/calib/NVIDIA_A100*/torch_probe_smolvla_env.csv"),
    "2080Ti": dict(calib_glob="results/calib/NVIDIA_GeForce_RTX_2080_Ti_cuda2_*",
                   dispatch_glob="results/calib/NVIDIA_GeForce_RTX_2080_Ti_cuda2_*",
                   bench_glob="results/bench/NVIDIA_GeForce_RTX_2080_Ti_cuda2_*",
                   probe_glob="results/calib/NVIDIA_GeForce_RTX_2080_Ti_cuda2_*/torch_probe_*.csv"),
    "TITANV": dict(calib_glob="results/calib/NVIDIA_TITAN_V_cuda3_*",
                   dispatch_glob="results/calib/NVIDIA_TITAN_V_cuda3_*",
                   bench_glob="results/bench/NVIDIA_TITAN_V_cuda3_*",
                   probe_glob="results/calib/NVIDIA_TITAN_V_cuda3_*/torch_probe_*.csv"),
    # CIMS cuda5, Ada sm_89, 36 MB L2 (2026-11-13): similar DRAM bandwidth to the 2080 Ti.
    "RTX4070": dict(calib_glob="results/calib/NVIDIA_GeForce_RTX_4070_cuda5_*",
                    dispatch_glob="results/calib/NVIDIA_GeForce_RTX_4070_cuda5_*",
                    bench_glob="results/bench/NVIDIA_GeForce_RTX_4070_cuda5_*",
                    probe_glob="results/calib/NVIDIA_GeForce_RTX_4070_cuda5_*/torch_probe_*.csv"),
}
TRACE_KEY = {"A100": "A100", "2080Ti": "RTX_2080_Ti", "TITANV": "TITAN_V", "RTX4070": "RTX_4070"}


def r_lat(src: GPU, tgt: GPU) -> float:
    common = sorted(set(src.small_us) & set(tgt.small_us))
    return float(np.exp(np.mean([np.log(tgt.small_us[k] / src.small_us[k]) for k in common])))


def py_ratio(src: GPU, tgt: GPU) -> float:
    """Host Python speed ratio (target/source) from the torch_probe host probes."""
    common = sorted(set(src.py_us) & set(tgt.py_us))
    return float(np.exp(np.mean([np.log(tgt.py_us[k] / src.py_us[k]) for k in common])))


def host_ratio(src: GPU, tgt: GPU) -> float:
    common = sorted(set(src.dispatch_med) & set(tgt.dispatch_med))
    return float(np.median([tgt.dispatch_med[k] / src.dispatch_med[k] for k in common]))


def transfer(s: np.ndarray, mode: str, src: GPU, tgt: GPU, tau: float = TAU_US) -> np.ndarray:
    if mode == "same":
        return s
    if mode == "spec_bw":
        return s * src.bw_spec / tgt.bw_spec
    rb = src.bw_dram / tgt.bw_dram
    if mode == "bw":
        return s * rb
    rl = r_lat(src, tgt)
    if mode == "lat":
        return s * rl
    if mode == "lat_bw":
        return rl * np.minimum(s, tau) + rb * np.maximum(s - tau, 0.0)
    raise ValueError(mode)


def transfer_slots(tr: StepTrace, src: GPU, tgt: GPU, tail_bw: bool = True,
                   cold: bool = False) -> np.ndarray:
    """Proposed transfer. A kernel's graph slot on the source is u = t + delta_S (t already
    bias corrected). Up to the size of its class's calibration op (tau_c = source slot time of
    that op), the slot scales with the measured class ratio r_c = slot_T / slot_S; the part
    beyond tau_c is data streaming and scales with the DRAM bandwidth ratio. Returns kernel
    durations on the target, t_T = u_T - delta_T."""
    u = tr.t_us + src.calib.delta_node_us
    cls = np.array(tr.cls)
    st = {**src.slot_us, **(src.cold_slot_us if cold else {})}
    tt = {**tgt.slot_us, **(tgt.cold_slot_us if cold else {})}
    tau = np.array([st[c] for c in cls])
    r = np.array([tt[c] / st[c] for c in cls])
    rb = src.bw_dram / tgt.bw_dram if tail_bw else 1.0
    u_t = r * np.minimum(u, tau) + (rb if tail_bw else r) * np.maximum(u - tau, 0.0)
    return np.maximum(u_t - tgt.calib.delta_node_us, 0.5)


def transfer_ratio(tr: StepTrace, src: GPU, tgt: GPU, ratio: dict) -> np.ndarray:
    """Scale each kernel's graph slot by a per-op-class ratio (target/source)."""
    u = tr.t_us + src.calib.delta_node_us
    r = np.array([ratio[c] for c in tr.cls])
    return np.maximum(u * r - tgt.calib.delta_node_us, 0.5)


def runs_for(gpu_key: str) -> list[tuple[Path, dict]]:
    out = []
    for meta_p in sorted((ROOT / "traces").glob(f"smolvla_*/*{TRACE_KEY[gpu_key]}*/meta.json")):
        out.append((meta_p.parent, json.loads(meta_p.read_text())))
    return out


def cfg_of(m: dict) -> tuple:
    return (m["cameras"], m["num_steps"], m["chunk_size"])


def measured_by_cfg(gpu_key: str) -> dict:
    """One measured run per config on the target: the one with the smallest eager std
    (cuda2 has an OS-scheduling slow mode on some unpinned runs, see CLAUDE.md)."""
    best: dict = {}
    for r, m in runs_for(gpu_key):
        if not m.get("graph_denoise_ms_median"):
            continue
        k = cfg_of(m)
        if k not in best or m["eager_ms_std"] < best[k][1]["eager_ms_std"]:
            best[k] = (r, m)
    return best


QUANT = {"eager": "eager_ms_median", "eager_prefix": "eager_prefix_ms_median",
         "eager_loop": "eager_denoise_ms_median", "graph_loop": "graph_denoise_ms_median",
         "hybrid": "hybrid_ms_median"}


def predict(tr: StepTrace, c: Calib, p_us: float) -> dict:
    full, pre, loop = slice(None), slice(0, tr.loop_start), slice(tr.loop_start, None)
    g = graph(tr, c, loop)
    pre_t = eager(tr, c, pre, "table", p_us)
    return {"eager": eager(tr, c, full, "table", p_us), "eager_prefix": pre_t,
            "eager_loop": eager(tr, c, loop, "table", p_us), "graph_loop": g,
            "hybrid": pre_t + g}


def main() -> None:
    gpus = {k: load_gpu(k, **v) for k, v in GPUS.items()}
    for k, g in gpus.items():
        print(f"# {k}: {g.name}, BW {g.bw_dram:.0f} GB/s (spec {g.bw_spec:.0f}), "
              f"g_eager {g.calib.g_eager_us:.2f} us, delta_node {g.calib.delta_node_us:.2f} us, "
              f"small kernels {len(g.small_us)}", file=sys.stderr)
    out = csv.writer(sys.stdout)
    out.writerow(["source", "target", "config", "quantity", "unit", "measured",
                  "kernel_transfer", "predicted", "error_pct", "r_lat", "r_bw", "host_ratio"])
    for sk, src in gpus.items():
        src_runs = measured_by_cfg(sk)
        base_r, base_m = src_runs[(3, 10, 50)]
        base_tr = load_step(base_r, base_m["num_steps"])
        b_us = fit_b(base_tr, src.calib, base_m["graph_denoise_ms_median"] * 1e3)
        p_us = fit_p(corrected(base_tr, b_us), src.calib, base_m["eager_ms_median"] * 1e3)
        print(f"# source {sk}: b = {b_us:.3f} us, p = {p_us:.2f} us (fit on {base_r.name})",
              file=sys.stderr)
        traces = {k: corrected(load_step(r, m["num_steps"]), b_us)
                  for k, (r, m) in src_runs.items()}
        for tk, tgt in gpus.items():
            h = host_ratio(src, tgt)
            rl = r_lat(src, tgt) if src.small_us and tgt.small_us else float("nan")
            rb = src.bw_dram / tgt.bw_dram
            for cfg, (tr_r, tr_m) in measured_by_cfg(tk).items():
                if cfg not in traces:
                    continue
                tr = traces[cfg]
                has_probe = bool(src.cold_slot_us and tgt.cold_slot_us)
                for mode in ("same", "spec_bw", "bw", "lat", "lat_bw", "slot_class",
                             "slot", "slot_cold", "slot_cold_py", "clock", "slot_cpp"):
                    if mode == "clock" and not (np.isfinite(src.clk_mhz)
                                                and np.isfinite(tgt.clk_mhz)):
                        continue
                    if mode == "slot_cpp" and not (src.cpp_slot_us and tgt.cpp_slot_us):
                        continue
                    if mode.startswith("lat") and not np.isfinite(rl):
                        continue
                    if mode.startswith("slot_cold") and not has_probe:
                        continue
                    if mode == "clock":
                        k = src.clk_mhz / tgt.clk_mhz
                        t_new = transfer_ratio(tr, src, tgt, {c: k for c in set(tr.cls)})
                    elif mode == "slot_cpp":
                        t_new = transfer_ratio(tr, src, tgt, {
                            c: tgt.cpp_slot_us[c] / src.cpp_slot_us[c] for c in set(tr.cls)})
                    elif mode.startswith("slot_cold"):
                        t_new = transfer_slots(tr, src, tgt, tail_bw=False, cold=True)
                    elif mode.startswith("slot"):
                        t_new = transfer_slots(tr, src, tgt, tail_bw=(mode == "slot"))
                    else:
                        t_new = transfer(tr.t_us, mode, src, tgt)
                    t2 = replace(tr, t_us=t_new)
                    hp = py_ratio(src, tgt) if mode == "slot_cold_py" else h
                    pred = predict(t2, tgt.calib, p_us * hp)
                    vals = {q: (tr_m[QUANT[q]], us / 1e3, "ms") for q, us in pred.items()}
                    # CUDA-graph speedups: loop only, and end to end (eager vs hybrid)
                    vals["speedup_loop"] = (tr_m["eager_denoise_ms_median"]
                                            / tr_m["graph_denoise_ms_median"],
                                            pred["eager_loop"] / pred["graph_loop"], "x")
                    vals["speedup_e2e"] = (tr_m["eager_ms_median"] / tr_m["hybrid_ms_median"],
                                           pred["eager"] / pred["hybrid"], "x")
                    for q, (meas, pv, unit) in vals.items():
                        out.writerow([sk, tk, "cam{}_steps{}_chunk{}".format(*cfg), q, unit,
                                      f"{meas:.3f}", mode, f"{pv:.3f}",
                                      f"{100 * (pv - meas) / meas:.1f}",
                                      f"{rl:.3f}", f"{rb:.3f}", f"{h:.3f}"])


if __name__ == "__main__":
    main()
