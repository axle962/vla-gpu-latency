"""Turn calibration CSVs (results/calib/<run>/) into per-GPU model parameters.

Parameters extracted, all in microseconds or GB/s:
  host_launch_us     CPU cost of one cudaLaunchKernel call.
  eager_spacing_us   GPU-side spacing between back-to-back empty kernels (eager, n=1000).
  graph_node_us      delta_node: per-node spacing inside a replayed graph (n=1000).
  graph_launch_us    CPU cost of cudaGraphLaunch for a 1000-node graph.
  roundtrip_us       one empty kernel plus synchronize (eager, n=1, host clock).
  graph_fixed_us     L_graph: replay of a 1-node graph plus synchronize (host clock).
  bw_l2_gbps         median bandwidth over working sets well inside L2.
  bw_dram_gbps       median bandwidth over working sets >= 8x L2.
  l2_knee_mb         working set where bandwidth falls halfway from the L2 to the DRAM plateau.
  crossover_spin_us  spin duration where the eager chain is within 10% of the graph chain.

Usage: uv run python model/calib.py results/calib/*/
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _cfg(series: pd.Series) -> pd.DataFrame:
    """Expand the `key=value;key=value` config column into numeric columns."""
    rows = [
        {k: float(v) for k, v in (kv.split("=", 1) for kv in c.split(";") if kv)} for c in series
    ]
    return pd.DataFrame(rows, index=series.index)


def _pick(df: pd.DataFrame, kernel: str, mode: str, timer: str, **cfg: int) -> float:
    sel = df[(df.kernel == kernel) & (df["mode"] == mode) & (df.timer == timer)]
    for k, v in cfg.items():
        sel = sel[sel[k] == v]
    if sel.empty:
        return float("nan")
    return float(sel.median_us.iloc[0])


def launch_params(path: Path) -> dict[str, float]:
    df = pd.read_csv(path)
    df = df.join(_cfg(df.config))
    out = {
        "host_launch_us": _pick(df, "host_launch_call", "eager", "host"),
        "eager_spacing_us": _pick(df, "empty_chain_per_kernel", "eager", "event", n=1000),
        "graph_node_us": _pick(df, "empty_chain_per_kernel", "graph", "event", n=1000),
        "graph_launch_us": _pick(df, "host_graph_call", "graph", "host", n=1000),
        "roundtrip_us": _pick(df, "empty_chain_total", "eager", "host", n=1),
        "graph_fixed_us": _pick(df, "empty_chain_total", "graph", "host", n=1),
    }
    spin = df[(df.kernel == "spin_chain_total") & (df.timer == "host")]
    piv = spin.pivot_table(index="spin_ns", columns="mode", values="median_us")
    if {"eager", "graph"} <= set(piv.columns):
        ratio = piv.eager / piv.graph
        below = ratio[ratio <= 1.10]
        out["crossover_spin_us"] = float(below.index.min()) / 1e3 if len(below) else float("nan")
        out["graph_speedup_spin0"] = float(ratio.iloc[0])
    return out


def bandwidth_params(path: Path, l2_bytes: float) -> dict[str, float]:
    df = pd.read_csv(path)
    df = df.join(_cfg(df.config))
    df["gbps"] = df.moved / (df.median_us * 1e-6) / 1e9
    l2_part = df[df.bytes <= 0.25 * l2_bytes]
    dram_part = df[df.bytes >= 8 * l2_bytes]
    bw_l2 = float(l2_part.gbps.median()) if len(l2_part) else float("nan")
    bw_dram = float(dram_part.gbps.median()) if len(dram_part) else float("nan")
    knee = float("nan")
    if np.isfinite(bw_l2) and np.isfinite(bw_dram):
        half = 0.5 * (bw_l2 + bw_dram)
        after = df[(df.bytes > 0.25 * l2_bytes) & (df.gbps < half)]
        if len(after):
            knee = float(after.bytes.min()) / 2**20
    return {"bw_l2_gbps": bw_l2, "bw_dram_gbps": bw_dram, "l2_knee_mb": knee}


def device_info(path: Path) -> dict[str, float | str]:
    """Long-form gpu,key,value rows; each device block starts with key device_index."""
    df = pd.read_csv(path)
    starts = list(df.index[df.key == "device_index"]) + [len(df)]
    first = df.iloc[starts[0] : starts[1]]  # device 0 only
    info: dict[str, float | str] = {"gpu": str(first.gpu.iloc[0])}
    info.update({k: float(v) for k, v in zip(first.key, first.value)})
    return info


def summarize_run(run: Path) -> dict[str, float | str]:
    info = device_info(run / "device_info.csv")
    row: dict[str, float | str] = {
        "run": run.name,
        "gpu": info["gpu"],
        "sm_count": info["sm_count"],
        "l2_mb": float(info["l2_bytes"]) / 2**20,
        "spec_dram_gbps": float(info["theoretical_dram_bw_mbps"]) / 1e3,
    }
    launch_csv = run / "launch_eager_loading.csv"
    if launch_csv.exists():
        row.update(launch_params(launch_csv))
    bw_csv = run / "bandwidth.csv"
    if bw_csv.exists():
        row.update(bandwidth_params(bw_csv, float(info["l2_bytes"])))
    return row


def main(argv: list[str]) -> None:
    runs = [Path(a) for a in argv if (Path(a) / "device_info.csv").exists()]
    if not runs:
        sys.exit("usage: calib.py results/calib/<run>/ [...]")
    table = pd.DataFrame([summarize_run(r) for r in runs])
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(table.set_index("run").T.to_string(float_format=lambda x: f"{x:.3g}"))


if __name__ == "__main__":
    main(sys.argv[1:])
