"""GPU timeline of the SmolVLA denoising loop, eager vs CUDA graph, on one time axis (slides).

Preferred source: an Nsight Systems export of workloads/nsys_smolvla.py (--nsys file.sqlite):
low overhead, kernels from CUPTI_ACTIVITY_KIND_KERNEL, loops delimited by the NVTX ranges
eager_loop_1 / graph_loop_1 (second instance, after a warm first one). Each row shows the 10
flow steps of one loop.

Fallback (no argument): the torch.profiler traces in traces/, one flow step each. That
profiler records every op on the host and roughly doubles the eager time, so the fallback
overstates the gaps; use it only when no nsys export exists.

Usage: envs/smolvla/bin/python model/timeline_figure.py [--nsys <run>.sqlite | --csv
  results/nsys/smolvla_loop_2080Ti_cuda2_kernels.csv.gz]  (--nsys also saves that CSV)
Writes paper/figures/timeline_loop.{pdf,png} (nsys) or timeline_flow_step.{pdf,png}.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from figures import GRID, INK, INK2, OUT, ROOT  # noqa: E402

KERNEL = "#2a78d6"
DEFAULT_RUN = "traces/smolvla_cam3_steps10_chunk50/NVIDIA_GeForce_RTX_2080_Ti_cuda2_20261112-170823"


def from_nsys(db: Path) -> tuple[list, str]:
    con = sqlite3.connect(db)
    k = pd.read_sql("SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL", con)
    names = pd.read_sql("SELECT id, value FROM StringIds", con).set_index("id").value
    nv = pd.read_sql("SELECT start, end, text, textId FROM NVTX_EVENTS WHERE end IS NOT NULL", con)
    nv["label"] = nv.text.fillna(nv.textId.map(names))
    gpu = pd.read_sql("SELECT name FROM TARGET_INFO_GPU", con).name.iloc[0]
    rows = []
    for label, title in (("eager_loop_1", "Eager PyTorch"), ("graph_loop_1", "CUDA graph")):
        r = nv[nv.label == label].iloc[0]
        d = k[(k.start >= r.start) & (k.end <= r.end)].sort_values("start")
        d = d.assign(start=(d.start - d.start.min()) / 1e6, dur=(d.end - d.start) / 1e6)
        rows.append((title, d))
    return rows, f"10 flow steps, {len(rows[1][1])} kernels each, {gpu}, Nsight Systems"


def from_profiler(run: Path) -> tuple[list, str]:
    meta = json.loads((run / "meta.json").read_text())
    k = int(round(meta["graph_trace_kernels_per_replay"] / meta["num_steps"]))
    rows = []
    for title, f in (("Eager PyTorch", "kernels.csv.gz"), ("CUDA graph", "kernels_graph.csv.gz")):
        d = pd.read_csv(run / f)
        d = d[(d.cat == "kernel") & (d.step == d.step.min())].sort_values("ts_us").tail(k)
        rows.append((title, d.assign(start=(d.ts_us - d.ts_us.min()) / 1e3, dur=d.dur_us / 1e3)))
    gpu = meta["gpu"].replace("NVIDIA ", "").replace("GeForce ", "")
    return rows, f"one flow step = {k} kernels, {gpu}, torch.profiler (inflates eager gaps)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsys", default="")
    ap.add_argument("--csv", default="", help="kernel intervals saved by an earlier --nsys run")
    ap.add_argument("run", nargs="?", default=DEFAULT_RUN)
    a = ap.parse_args()
    csv = ROOT / "results/nsys/smolvla_loop_2080Ti_cuda2_kernels.csv.gz"
    if a.nsys:
        rows, note = from_nsys(Path(a.nsys))
        stem = "timeline_loop"
        pd.concat([d.assign(mode=t)[["mode", "start", "dur"]] for t, d in rows]).to_csv(
            csv, index=False)
        (csv.parent / "smolvla_loop_2080Ti_cuda2_note.txt").write_text(note + "\n")
    elif a.csv:
        d = pd.read_csv(a.csv)
        rows = [(t, d[d["mode"] == t]) for t in ("Eager PyTorch", "CUDA graph")]
        note = (Path(a.csv).parent / "smolvla_loop_2080Ti_cuda2_note.txt").read_text().strip()
        stem = "timeline_loop"
    else:
        rows, note = from_profiler(ROOT / a.run)
        stem = "timeline_flow_step"
    span = max(float((d.start + d.dur).max()) for _, d in rows)
    # Unprofiled loop times of the same GPU and config (traced times run ~1.4x longer).
    meta = json.loads((ROOT / DEFAULT_RUN / "meta.json").read_text())
    untraced = [meta["eager_denoise_ms_median"], meta["graph_denoise_ms_median"]] \
        if (a.nsys or a.csv) \
        else [None, None]

    fig, ax = plt.subplots(figsize=(7.2, 1.9))
    for i, (label, d) in enumerate(rows):
        y = 1 - i
        ax.broken_barh(list(zip(d.start, d.dur)), (y - 0.3, 0.6), facecolors=KERNEL, linewidth=0)
        end = float((d.start + d.dur).max())
        txt = f"{end:.0f} ms traced, GPU busy {100 * float(d.dur.sum()) / end:.0f}%"
        if untraced[i]:
            txt += f"\n{untraced[i]:.0f} ms without profiler"
        ax.text(end + span * 0.01, y, txt, va="center", ha="left", fontsize=8, color=INK)
    ax.set_yticks([1, 0], [r[0] for r in rows], fontsize=8, color=INK)
    ax.set_ylim(-0.6, 1.6)
    ax.set_xlim(0, span * 1.4)
    ax.set_xlabel(f"time (ms); {note}", fontsize=8, color=INK2)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(INK2)
    ax.tick_params(axis="x", colors=INK2, labelsize=8)
    ax.tick_params(axis="y", length=0)
    ax.grid(True, axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=200, bbox_inches="tight")
    print("wrote", OUT / f"{stem}.pdf")


if __name__ == "__main__":
    main()
