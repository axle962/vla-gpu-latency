"""Figure for the hosting-placement study: one horizontal stacked bar per placement (prefix,
KV handoff, action-expert loop), eager next to best mode (CUDA graph where the loop runs on a
GPU). Bars are the parts timed alone; the number at the end of each bar is the measured
end-to-end step (parts and end-to-end agree within 3% in every run so far).

  python model/placement_figure.py RUN [--gpu1-baseline RUN] [--out paper/figures/placement_X]

--gpu1-baseline adds a "1 GPU (GPU 1)" row from a run made with only GPU 1 visible, for hosts
whose two GPUs differ in speed (cuda2: GPU 0 runs the graph loop in 54 ms, GPU 1 in 40 ms).
Writes <out>.pdf and <out>.png.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# Reference categorical palette, slots 1 to 3 in order (validated: CVD and normal-vision
# separation pass; aqua is below 3:1 on the surface, so every bar also carries a legend entry
# and a value label).
COLORS = {"prefix": "#2a78d6", "handoff": "#eb6834", "loop": "#1baf7a"}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

LABELS = {
    "gpu": "1 GPU",
    "gpu-gpu": "prefix GPU 0, loop GPU 1",
    "gpu-cpu": "prefix GPU, loop CPU",
    "cpu-gpu": "prefix CPU, loop GPU",
    "cpu": "CPU only",
}


def medians(run: Path) -> pd.DataFrame:
    s = pd.read_csv(run / "summary.csv")
    return s.pivot(index="placement", columns="mode", values="median_ms")


def rows_for(run: Path, gpu1: Path | None, extras: list[str] = ()) -> list[dict]:
    med = medians(run)
    order = list(json.loads((run / "meta.json").read_text())["placements"])
    two_gpus = gpu1 is not None

    def v(p, m, src=med):
        return float(src.loc[p, m]) if m in src.columns and pd.notna(src.loc[p, m]) else None

    out = []
    for p in order:
        label = LABELS[p] + (" (GPU 0)" if p == "gpu" and two_gpus else "")
        out.append({"label": label, "src": med, "p": p})
        if p == "gpu" and two_gpus:
            out.append({"label": LABELS[p] + " (GPU 1)", "src": medians(gpu1), "p": "gpu"})
        if p == "gpu":
            for spec in extras:  # RUN=PLACEMENT=LABEL, inserted after the colocated row
                erun, ep, elabel = spec.split("=", 2)
                out.append({"label": elabel, "src": medians(Path(erun)), "p": ep})
    rows = []
    for r in out:
        src, p = r["src"], r["p"]
        pre, hand = v(p, "prefix", src), v(p, "handoff", src) or 0.0
        eager_loop, graph_loop = v(p, "denoise_eager", src), v(p, "denoise_graph", src)
        rows.append({
            "label": r["label"],
            "eager": (pre, hand, eager_loop, v(p, "e2e_eager", src)),
            "best": (pre, hand, graph_loop or eager_loop,
                     v(p, "e2e_hybrid", src) if graph_loop else v(p, "e2e_eager", src)),
            "graph": graph_loop is not None,
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--gpu1-baseline", default=None)
    ap.add_argument("--extra", action="append", default=[], metavar="RUN=PLACEMENT=LABEL",
                    help="add a row from another run (e.g. gpu-gpu from a two-GPU run)")
    ap.add_argument("--title", default=None)
    ap.add_argument("--out", default="paper/figures/placement")
    args = ap.parse_args()
    run = Path(args.run)
    meta = json.loads((run / "meta.json").read_text())
    rows = rows_for(run, Path(args.gpu1_baseline) if args.gpu1_baseline else None, args.extra)

    plt.rcParams.update({"font.size": 9, "font.family": "sans-serif"})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 0.42 * len(rows) + 1.3), sharey=True,
                             facecolor=SURFACE)
    xmax = max(r[k][3] for r in rows for k in ("eager", "best")) * 1.2
    y = list(range(len(rows)))[::-1]
    for ax, key, title in (
        (axes[0], "eager", "Eager PyTorch"),
        (axes[1], "best", "Best mode (CUDA graph for a GPU loop)"),
    ):
        ax.set_facecolor(SURFACE)
        for yi, r in zip(y, rows):
            pre, hand, loop, e2e = r[key]
            left = 0.0
            for part, w in (("prefix", pre), ("handoff", hand), ("loop", loop)):
                if w:
                    ax.barh(yi, w, left=left, height=0.62, color=COLORS[part],
                            edgecolor=SURFACE, linewidth=1.0)
                    left += w
            ax.text(left + xmax * 0.012, yi, f"{e2e:,.0f} ms", va="center", ha="left",
                    color=INK, fontsize=8.5)
        ax.set_xlim(0, xmax)
        ax.set_title(title, loc="left", color=INK, fontsize=9.5)
        ax.set_xlabel("ms per inference (median)", color=INK2)
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=INK2, length=0)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([r["label"] for r in rows], color=INK)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[k]) for k in COLORS]
    fig.legend(handles, ["prefix (vision + VLM)", "KV handoff", "action-expert loop"],
               loc="upper left", ncol=3, frameon=False, bbox_to_anchor=(0.01, 0.985),
               labelcolor=INK)
    gpu = meta["gpus"][0] if meta["gpus"] else "CPU"
    title = args.title or f"SmolVLA hosting placements, {gpu} ({meta['host'].split('.')[0]})"
    fig.suptitle(title, x=0.01, y=1.03, ha="left", color=INK, fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=200, bbox_inches="tight", facecolor=SURFACE)
    print("wrote", f"{out}.pdf", f"{out}.png")


if __name__ == "__main__":
    main()
