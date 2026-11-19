"""Hosting placement report: measured latency per placement and predicted vs measured.

Reads one run of workloads/placement.py (results/placement/<run>/) and writes report.md there
(or --out). Three parts:

1. Measured medians per placement and the slowdown against the colocated GPU baseline.
2. Copy calibration per device pair: fixed cost per tensor (the 4 KB copy) and bandwidth
   (least-squares slope over copies of 16 MB and more).
3. Prediction of each split placement from parts that never ran split:
     T(A, B) = T_prefix(A colocated) + T_handoff(A -> B) + T_denoise(B colocated)
     T_handoff = n_tensors * t0 + bytes / BW          (from the copy calibration and KV size)
   for the eager loop and for the graph loop, with percentage error. A second GPU (cuda:1) is
   assumed to match cuda:0 (true on cuda2 and cuda3: two identical cards). The prediction has
   no term for the dtype cast on a CPU destination or for contention between the halves, so
   its error is the measured size of those effects.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

COLOCATED = {"cuda:0": "gpu", "cuda:1": "gpu", "cpu": "cpu"}


def fit_copy(calib: pd.DataFrame) -> dict[tuple[str, str], tuple[float, float]]:
    """(src, dst) -> (t0_ms per tensor, bandwidth in bytes per ms)."""
    out = {}
    for (src, dst), g in calib.groupby(["src", "dst"]):
        g = g.sort_values("bytes")
        t0 = float(g.median_ms.iloc[0])
        big = g[g.bytes >= 16 << 20]
        x, y = big.bytes.astype(float), big.median_ms.astype(float)
        slope = ((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum()
        out[(src, dst)] = (t0, 1.0 / slope if slope > 0 else float("inf"))
    return out


def md(df: pd.DataFrame) -> str:
    """Markdown table without the optional tabulate dependency (not in every env)."""
    cols = [str(c) for c in df.columns]
    rows = [[str(v) for v in r] for r in df.itertuples(index=False)]
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def pct(pred: float, meas: float) -> str:
    return f"{100 * (pred - meas) / meas:+.1f}%"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="results/placement/<run> directory")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    run = Path(args.run)
    summ = pd.read_csv(run / "summary.csv")
    calib = pd.read_csv(run / "copy_calib.csv")
    meta = json.loads((run / "meta.json").read_text())
    med = summ.pivot(index="placement", columns="mode", values="median_ms")
    med = med.reindex([p for p in meta["placements"] if p in med.index])

    def get(p: str, mode: str) -> float | None:
        if p in med.index and mode in med.columns and pd.notna(med.loc[p, mode]):
            return float(med.loc[p, mode])
        return None

    lines = [
        f"# Hosting placement report: {meta['model']}",
        "",
        f"Host `{meta['host']}` ({meta['cpu']}, {meta['torch_threads']} torch threads, "
        f"{meta['affinity_cpus']} CPUs in the affinity mask), GPUs: "
        f"{', '.join(meta['gpus']) or 'none'}, torch {meta['torch']}. "
        f"{meta['cameras']} cameras, {meta['num_steps']} flow steps, chunk {meta['chunk_size']}. "
        f"Precision per device: {meta['dtype_by_device']}.",
        "",
        "## Measured (median ms)",
        "",
    ]
    cols = ["prefix", "handoff", "denoise_eager", "denoise_graph", "e2e_eager", "e2e_hybrid"]
    cols = [c for c in cols if c in med.columns]
    base_e, base_h = get("gpu", "e2e_eager"), get("gpu", "e2e_hybrid")
    rows = []
    for p in med.index:
        info = meta["placements"][p]
        r = {"placement": p, "prefix on": info["dev_prefix"], "denoise on": info["dev_denoise"]}
        for c in cols:
            v = get(p, c)
            r[c] = "" if v is None else f"{v:.2f}"
        e, h = get(p, "e2e_eager"), get(p, "e2e_hybrid")
        r["eager vs gpu"] = f"{e / base_e:.2f}x" if e and base_e else ""
        r["best vs gpu best"] = f"{(h or e) / (base_h or base_e):.2f}x" if base_e and e else ""
        r["max diff vs first"] = f"{info['max_abs_diff_vs_first']:.1e}"
        rows.append(r)
    lines += [md(pd.DataFrame(rows)), ""]
    lines += [
        "`best` is e2e_hybrid where the denoise device is a GPU (CUDA graph), else e2e_eager.",
        "",
    ]

    fits = fit_copy(calib) if len(calib) else {}
    if fits:
        lines += ["## Copy calibration", ""]
        rows = [
            {"src": s, "dst": d, "per-tensor t0 (us)": f"{t0 * 1e3:.1f}",
             "bandwidth (GB/s)": f"{bw * 1e3 / 1e9:.2f}"}
            for (s, d), (t0, bw) in fits.items()
        ]
        lines += [md(pd.DataFrame(rows)), ""]

        lines += ["## Split placements predicted from colocated parts", ""]
        rows = []
        for p in med.index:
            info = meta["placements"][p]
            a, b = info["dev_prefix"], info["dev_denoise"]
            if a == b or (a, b) not in fits:
                continue
            t0, bw = fits[(a, b)]
            hand_pred = info["kv_tensors"] * t0 + info["kv_bytes"] / bw
            hand = get(p, "handoff")
            r = {
                "placement": p,
                "KV MB": f"{info['kv_bytes'] / 1e6:.1f}",
                "tensors": info["kv_tensors"],
                "handoff pred": f"{hand_pred:.2f}",
                "handoff meas": f"{hand:.2f}" if hand else "",
                "err": pct(hand_pred, hand) if hand else "",
            }
            pre = get(COLOCATED[a], "prefix")
            for label, part, e2e in (
                ("eager", "denoise_eager", "e2e_eager"),
                ("hybrid", "denoise_graph", "e2e_hybrid"),
            ):
                den, meas = get(COLOCATED[b], part), get(p, e2e)
                if pre is None or den is None or meas is None:
                    r[f"{label} pred"] = r[f"{label} meas"] = r[f"{label} err"] = ""
                    continue
                pred = pre + hand_pred + den
                r[f"{label} pred"] = f"{pred:.1f}"
                r[f"{label} meas"] = f"{meas:.1f}"
                r[f"{label} err"] = pct(pred, meas)
            rows.append(r)
        if rows:
            lines += [md(pd.DataFrame(rows)), ""]
        else:
            lines += ["No split placement with its colocated baselines in this run.", ""]

    lines += ["## Additivity inside each placement", ""]
    rows = []
    for p in med.index:
        parts = [get(p, "prefix"), get(p, "handoff") or 0.0, get(p, "denoise_eager")]
        e = get(p, "e2e_eager")
        if None in parts or e is None:
            continue
        rows.append({
            "placement": p, "sum of parts": f"{sum(parts):.1f}", "e2e_eager": f"{e:.1f}",
            "gap": pct(sum(parts), e),
        })
    lines += [md(pd.DataFrame(rows)), ""]
    lines += [
        "A negative gap means the end-to-end step costs more than its parts timed alone "
        "(for example cold caches after the handoff, or the two halves competing for host "
        "threads).",
        "",
    ]

    out = Path(args.out) if args.out else run / "report.md"
    out.write_text("\n".join(lines))
    print("\n".join(lines))
    print("wrote", out)


if __name__ == "__main__":
    main()
