"""Paper figures from results/predictions_cross_gpu.csv (model/cross_gpu.py).

  paper/figures/pred_vs_meas.pdf   predicted vs measured, cross-GPU pairs, baseline vs proposed
  paper/figures/mape_variants.pdf  MAPE per kernel-transfer variant for four quantities

Usage: envs/smolvla/bin/python model/figures.py [--variant slot_class]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper/figures"
# Categorical slots 1 to 3 of the reference palette, fixed per target GPU.
GPU_COLOR = {"A100": "#2a78d6", "2080Ti": "#eb6834", "TITANV": "#1baf7a", "RTX4070": "#8a5cd1"}
GPU_LABEL = {"A100": "A100", "2080Ti": "RTX 2080 Ti", "TITANV": "TITAN V", "RTX4070": "RTX 4070"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
VARIANT_LABEL = {"same": "unchanged", "spec_bw": "spec BW ratio", "bw": "measured BW ratio",
                 "lat": "C++ latency ratio", "lat_bw": "C++ latency + BW",
                 "slot_class": "slot per op class", "slot": "slot + BW tail",
                 "slot_cold": "cold slot", "slot_cold_py": "cold slot + Python host",
                 "clock": "sustained clock ratio", "slot_cpp": "C++ replay slot ratio",
                 "slot_drv": "slot + driver-corrected host"}


def style(ax) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def pred_vs_meas(d: pd.DataFrame, variant: str, tag: str = "") -> None:
    x = d[(d.source != d.target) & (d.unit == "ms")
          & d.quantity.isin(["eager", "graph_loop", "hybrid"])]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.4), sharex=True, sharey=True)
    for ax, var in zip(axes, ["spec_bw", variant]):
        v = x[x.kernel_transfer == var]
        for tgt, g in v.groupby("target"):
            ax.scatter(g.measured, g.predicted, s=22, color=GPU_COLOR[tgt],
                       edgecolor="white", linewidth=0.8, label=GPU_LABEL[tgt], zorder=3)
        lo, hi = 15, 1200
        ax.plot([lo, hi], [lo, hi], color=INK2, linewidth=1, zorder=2)
        for f in (0.8, 1.25):
            ax.plot([lo, hi], [lo * f, hi * f], color=INK2, linewidth=0.6, ls=":", zorder=2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        style(ax)
        mape = v.error_pct.abs().mean()
        ax.set_title(f"{VARIANT_LABEL[var]} (MAPE {mape:.1f}%)", fontsize=9, color=INK)
        ax.set_xlabel("measured latency (ms)", fontsize=8, color=INK2)
    axes[0].set_ylabel("predicted latency (ms)", fontsize=8, color=INK2)
    axes[1].legend(title="target GPU", fontsize=7, title_fontsize=7, frameon=False,
                   loc="upper left")
    pairs = x[["source", "target"]].drop_duplicates().shape[0]
    fig.text(0.5, -0.02, f"eager step, graph loop and hybrid step; {pairs} source/target pairs x "
             f"{x.config.nunique()} configs; dotted lines: +/-20%", ha="center", fontsize=7,
             color=INK2)
    fig.tight_layout()
    fig.savefig(OUT / f"pred_vs_meas{tag}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"pred_vs_meas{tag}.png", dpi=200, bbox_inches="tight")


def mape_variants(d: pd.DataFrame, variant: str, tag: str = "") -> None:
    x = d[d.source != d.target]
    order = [v for v in VARIANT_LABEL if v in set(x.kernel_transfer)]
    qs = [("eager", "eager step"), ("graph_loop", "graph loop"), ("hybrid", "hybrid step"),
          ("speedup_loop", "graph speedup (loop)")]
    fig, axes = plt.subplots(1, 4, figsize=(9.0, 2.9), sharey=True)
    for ax, (q, title) in zip(axes, qs):
        m = x[x.quantity == q].groupby("kernel_transfer").error_pct \
            .apply(lambda e: e.abs().mean()).reindex(order)
        colors = [GPU_COLOR["A100"] if v == variant else "#b7b6b0" for v in order]
        ax.barh(range(len(order)), m.values, color=colors, height=0.7)
        for i, val in enumerate(m.values):
            ax.text(val + 1, i, f"{val:.0f}", va="center", fontsize=7, color=INK2)
        ax.set_yticks(range(len(order)), [VARIANT_LABEL[v] for v in order])
        ax.invert_yaxis()
        style(ax)
        ax.grid(axis="y", visible=False)
        ax.set_title(title, fontsize=9, color=INK)
        ax.set_xlabel("MAPE (%)", fontsize=8, color=INK2)
        ax.set_xlim(0, max(80, m.max() * 1.15))
    fig.tight_layout()
    fig.savefig(OUT / f"mape_variants{tag}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"mape_variants{tag}.png", dpi=200, bbox_inches="tight")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="slot_class")
    ap.add_argument("--csv", default=str(ROOT / "results/predictions_cross_gpu.csv"))
    ap.add_argument("--gpus", default="", help="comma list of GPU keys to keep (default: all)")
    ap.add_argument("--tag", default="", help="suffix for the output file names, e.g. _4gpu")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(a.csv)
    if a.gpus:
        keep = a.gpus.split(",")
        d = d[d.source.isin(keep) & d.target.isin(keep)]
    pred_vs_meas(d, a.variant, a.tag)
    mape_variants(d, a.variant, a.tag)
    print("wrote", *sorted(p.name for p in OUT.glob("*")))


if __name__ == "__main__":
    main()
