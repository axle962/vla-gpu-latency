"""Summarize results/predictions_cross_gpu.csv (model/cross_gpu.py) for the paper:

  1. MAPE per quantity and kernel-transfer variant, cross-GPU pairs only (source != target)
  2. MAPE per (source, target) pair for the proposed variant
  3. GPU ordering: for every source GPU, config and quantity, does the predicted ranking of
     the target GPUs (including the source itself) match the measured one? Reports the
     fraction of exact rankings and the mean Kendall tau, per variant.
  4. The eager vs graph ordering inversion (TITAN V fastest eager, slowest graph): predicted?

Usage: envs/smolvla/bin/python model/report_cross_gpu.py [--variant slot] [csv]
Writes markdown to stdout.
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
QUANT_ORDER = ["eager", "eager_prefix", "eager_loop", "graph_loop", "hybrid", "speedup_loop",
               "speedup_e2e"]


def kendall(a: list[float], b: list[float]) -> float:
    s = n = 0
    for i, j in combinations(range(len(a)), 2):
        da, db = a[i] - a[j], b[i] - b[j]
        s += (da * db > 0) - (da * db < 0)
        n += 1
    return s / n if n else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default=str(ROOT / "results/predictions_cross_gpu.csv"))
    ap.add_argument("--variant", default="slot_class")
    a = ap.parse_args()
    d = pd.read_csv(a.csv)
    d["ape"] = d.error_pct.abs()
    x = d[d.source != d.target]
    variants = [v for v in ["same", "spec_bw", "bw", "lat", "lat_bw", "slot_class", "slot",
                            "slot_cold", "slot_cold_py", "clock", "slot_cpp"] if v in set(d.kernel_transfer)]

    print("## MAPE (%) over cross-GPU pairs and configs\n")
    t = x.pivot_table(index="quantity", columns="kernel_transfer", values="ape", aggfunc="mean")
    t = t.reindex([q for q in QUANT_ORDER if q in t.index])[variants]
    print(t.round(1).to_markdown(), "\n")

    print(f"## MAPE (%) per pair, variant `{a.variant}`\n")
    v = x[x.kernel_transfer == a.variant]
    t = v.pivot_table(index=["source", "target"], columns="quantity", values="ape",
                      aggfunc="mean")
    print(t[[q for q in QUANT_ORDER if q in t.columns]].round(1).to_markdown(), "\n")

    print("## Same-GPU baseline (source == target), variant-independent\n")
    s = d[(d.source == d.target) & (d.kernel_transfer == "same")]
    t = s.pivot_table(index="source", columns="quantity", values="ape", aggfunc="mean")
    print(t[[q for q in QUANT_ORDER if q in t.columns]].round(1).to_markdown(), "\n")

    print("## GPU ordering (all GPUs ranked per source, config, quantity)\n")
    rows = []
    for var in variants:
        g = d[d.kernel_transfer == var]
        exact, taus = [], []
        for (src, cfg, q), grp in g.groupby(["source", "config", "quantity"]):
            if grp.target.nunique() < 3:
                continue
            taus.append(kendall(grp.measured.tolist(), grp.predicted.tolist()))
            exact.append(taus[-1] == 1.0)
        if not taus:
            continue  # variant lacks data for some GPU (e.g. clock before every GPU has one)
        rows.append({"variant": var, "cases": len(taus),
                     "exact_ranking_pct": 100 * sum(exact) / len(exact),
                     "mean_kendall_tau": sum(taus) / len(taus)})
    print(pd.DataFrame(rows).round(3).to_markdown(index=False), "\n")

    print("## Eager vs graph ordering inversion, baseline config (cam3_steps10_chunk50)\n")
    b = d[(d.config == "cam3_steps10_chunk50") & d.quantity.isin(["eager", "graph_loop"])
          & d.kernel_transfer.isin(["spec_bw", a.variant])]
    meas = b[b.kernel_transfer == a.variant].groupby(["quantity", "target"]).measured.first()
    print("measured (ms):", {q: meas[q].round(1).to_dict() for q in ["eager", "graph_loop"]})
    for var in ["spec_bw", a.variant]:
        for src in sorted(b.source.unique()):
            p = b[(b.kernel_transfer == var) & (b.source == src)] \
                .set_index(["quantity", "target"]).predicted
            ok = all(list(p[q].sort_values().index) == list(meas[q].sort_values().index)
                     for q in ["eager", "graph_loop"])
            print(f"- {var}, source {src}: eager order {list(p['eager'].sort_values().index)}, "
                  f"graph order {list(p['graph_loop'].sort_values().index)}, "
                  f"{'matches' if ok else 'differs from'} measured")
    print()


if __name__ == "__main__":
    main()
