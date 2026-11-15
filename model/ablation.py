"""Ablation: how much target-side calibration does the cross-GPU model need?

Each level adds one kind of target measurement to the one before (the source GPU always has
its full trace, b and p fitted on its baseline as in model/cross_gpu.py):

  L0 spec only        kernels: clock ratio (driver max SM clock, device_info) up to TAU_US,
                      spec DRAM bandwidth ratio beyond; launch constants (g_eager, delta_node,
                      graph launch), dispatch table and p of the SOURCE (as if on the same host)
  L1 + C++ microbench kernels: measured C++ small-kernel ratio + measured bandwidth (lat_bw);
                      launch constants of the target (bench/calib launch_latency); host of the
                      source
  L2 + dispatch table host: target's PyTorch dispatch table, p x host_ratio (as cross_gpu.py)
  L3 + graph slots    kernels: per-op-class PyTorch graph-mode slot ratios (slot_class)
  L4 final (slot_drv) host: p x framework_ratio (driver-corrected)

Prints markdown (MAPE per level and quantity over cross-GPU pairs, exact GPU ranking) and
writes results/ablation_calibration.csv (one row per prediction).

Usage: envs/smolvla/bin/python model/ablation.py [--workload smolvla] > results/ablation_calibration.md
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calib import device_info  # noqa: E402
import cross_gpu as cg  # noqa: E402
from launch_model import corrected, fit_b, fit_p  # noqa: E402

LEVELS = ["L0_spec", "L1_cpp_microbench", "L2_dispatch_table", "L3_graph_slots",
          "L4_final_slot_drv"]


def spec_clock_mhz(key: str) -> float:
    info = device_info(cg._one(cg.GPUS[key]["calib_glob"]) / "device_info.csv")
    return float(info["sm_clock_khz"]) / 1e3


def spec_transfer(t: np.ndarray, src: cg.GPU, tgt: cg.GPU, clk: dict) -> np.ndarray:
    rl = clk[src.key] / clk[tgt.key]
    rb = src.bw_spec / tgt.bw_spec
    return rl * np.minimum(t, cg.TAU_US) + rb * np.maximum(t - cg.TAU_US, 0.0)


def kendall(a: list, b: list) -> float:
    n, s = len(a), 0
    for i in range(n):
        for j in range(i + 1, n):
            s += np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])
    return 2 * s / (n * (n - 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", default="smolvla")
    ap.add_argument("--csv", default=str(cg.ROOT / "results/ablation_calibration.csv"))
    a = ap.parse_args()
    wl, spec = a.workload, cg.WORKLOADS[a.workload]
    gpus = {k: cg.load_gpu(k, **v) for k, v in cg.GPUS.items()}
    clk = {k: spec_clock_mhz(k) for k in gpus}
    print(f"# spec clocks (MHz): {clk}", file=sys.stderr)
    p_smolvla: dict = {}
    rows = []
    for sk, src in gpus.items():
        src_runs = cg.measured_by_cfg(sk, wl)
        if spec["base"] not in src_runs:
            continue
        quant = spec["quant"]
        base_r, base_m = src_runs[spec["base"]]
        base_tr = cg.load_wl_step(base_r, base_m, wl)
        b_us = fit_b(base_tr, src.calib, base_m[quant["graph_loop"]] * 1e3)
        p_us = fit_p(corrected(base_tr, b_us), src.calib, base_m["eager_ms_median"] * 1e3)
        if wl == "smolvla":
            p_smolvla[sk] = p_us
        traces = {k: corrected(cg.load_wl_step(r, m, wl), b_us) for k, (r, m) in src_runs.items()}
        for tk, tgt in gpus.items():
            if tk == sk:
                continue
            for cfg, (_, tm) in cg.measured_by_cfg(tk, wl).items():
                if cfg not in traces:
                    continue
                tr = traces[cfg]
                for lev in LEVELS:
                    if lev == "L0_spec":
                        t_new, c, p = spec_transfer(tr.t_us, src, tgt, clk), src.calib, p_us
                    elif lev == "L1_cpp_microbench":
                        t_new = cg.transfer(tr.t_us, "lat_bw", src, tgt)
                        c = replace(src.calib, g_eager_us=tgt.calib.g_eager_us,
                                    delta_node_us=tgt.calib.delta_node_us,
                                    graph_call_us=tgt.calib.graph_call_us,
                                    launch_call_us=tgt.calib.launch_call_us)
                        p = p_us
                    elif lev == "L2_dispatch_table":
                        t_new = cg.transfer(tr.t_us, "lat_bw", src, tgt)
                        c, p = tgt.calib, p_us * cg.host_ratio(src, tgt)
                    elif lev == "L3_graph_slots":
                        t_new = cg.transfer_slots(tr, src, tgt, tail_bw=False)
                        c, p = tgt.calib, p_us * cg.host_ratio(src, tgt)
                    else:
                        t_new = cg.transfer_slots(tr, src, tgt, tail_bw=False)
                        c, p = tgt.calib, p_us * cg.framework_ratio(src, tgt)
                    pred = cg.predict(replace(tr, t_us=t_new), c, p)
                    vals = {q: (tm[key], pred[q] / 1e3) for q, key in quant.items()}
                    if spec["loop"]:
                        vals["speedup_loop"] = (tm["eager_denoise_ms_median"]
                                                / tm["graph_denoise_ms_median"],
                                                pred["eager_loop"] / pred["graph_loop"])
                    else:
                        vals["speedup_e2e"] = (tm["eager_ms_median"] / tm[quant["graph_loop"]],
                                               pred["eager"] / pred["graph_loop"])
                    for q, (meas, pv) in vals.items():
                        rows.append({"workload": wl, "source": sk, "target": tk,
                                     "config": spec["name"].format(*cfg), "level": lev,
                                     "quantity": q, "measured": meas, "predicted": pv,
                                     "error_pct": 100 * (pv - meas) / meas})
    d = pd.DataFrame(rows)
    d.to_csv(a.csv, index=False)
    d["ape"] = d.error_pct.abs()
    print(f"Workload: `{wl}`, cross-GPU pairs only (source != target), "
          f"{d.source.nunique()} source GPUs.\n")
    print("## MAPE (%) per calibration level\n")
    t = d.pivot_table(index="level", columns="quantity", values="ape", aggfunc="mean")
    print(t.reindex(LEVELS).round(1).to_markdown(), "\n")

    # Ranking of all GPUs per (source, config, quantity, level); the source's own measured
    # value counts as its prediction.
    print("## Exact ranking of all GPUs (%)\n")
    out = []
    for lev in LEVELS:
        ex, taus = [], []
        for (sk, cfg, q), g in d[d.level == lev].groupby(["source", "config", "quantity"]):
            own = d[(d.target == sk) & (d.config == cfg) & (d.quantity == q)].measured
            if own.empty or g.target.nunique() < 2:
                continue
            meas = list(g.measured) + [own.iloc[0]]
            pred = list(g.predicted) + [own.iloc[0]]
            taus.append(kendall(meas, pred))
            ex.append(taus[-1] == 1.0)
        out.append({"level": lev, "cases": len(ex), "exact_ranking_pct": 100 * np.mean(ex),
                    "mean_kendall_tau": np.mean(taus)})
    print(pd.DataFrame(out).round(2).to_markdown(index=False), "\n")


if __name__ == "__main__":
    main()
