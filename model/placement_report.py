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
    """(src, dst) -> (t0_ms per tensor, bandwidth in bytes per ms). t0 is the per-copy cost of
    32 small copies queued with one sync (the handoff pattern) when the run measured it, else
    the single synced 4 KB copy (older runs; overstates t0 for async GPU-GPU copies)."""
    if "batch" not in calib.columns:
        calib = calib.assign(batch=1)
    out = {}
    for (src, dst), g in calib.groupby(["src", "dst"]):
        b = g[g.batch > 1]
        g = g[g.batch == 1].sort_values("bytes")
        t0 = float(b.median_ms.iloc[0]) if len(b) else float(g.median_ms.iloc[0])
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
    ap.add_argument(
        "--device-baseline", action="append", default=[], metavar="DEV=RUN",
        help="take the colocated baseline for DEV (e.g. cuda:1) from the `gpu` placement of "
        "another run made with only that GPU visible; needed when the two GPUs differ in speed",
    )
    args = ap.parse_args()
    run = Path(args.run)
    summ = pd.read_csv(run / "summary.csv")
    calib = pd.read_csv(run / "copy_calib.csv")
    meta = json.loads((run / "meta.json").read_text())
    med = summ.pivot(index="placement", columns="mode", values="median_ms")
    med = med.reindex([p for p in meta["placements"] if p in med.index])

    ext: dict[str, pd.DataFrame] = {}
    for spec in args.device_baseline:
        dev, d = spec.split("=", 1)
        s2 = pd.read_csv(Path(d) / "summary.csv")
        ext[dev] = s2[s2.placement == "gpu"].set_index("mode").median_ms

    def colocated(dev: str, mode: str) -> float | None:
        """Time of `mode` with both halves on dev: from --device-baseline if given, else from
        this run's colocated placement (cuda:1 then reuses cuda:0's numbers)."""
        if dev in ext:
            v = ext[dev].get(mode)
            return None if v is None or pd.isna(v) else float(v)
        return get(COLOCATED[dev], mode)

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
    first = next(iter(meta["placements"]))
    e0, e1 = get(first, "e2e_eager"), get(first, "e2e_eager_repeat")
    if e0 and e1:
        ok = abs(e1 - e0) / e0 < 0.03
        lines += [
            f"Drift check: `{first}` e2e_eager {e0:.1f} ms first, {e1:.1f} ms after all other "
            f"placements ({pct(e1, e0)}). "
            + ("Within 3%: placements are comparable." if ok else
               "**More than 3%: later placements ran under different conditions; treat "
               "cross-placement comparisons with care.**"),
            "",
        ]
    else:
        lines += [
            "No drift check in this run (older script). Runs made before the fix of "
            "2026-11-19 profiled each placement right after timing it, which slowed every "
            "later placement; compare only within a placement.",
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
            pre = colocated(a, "prefix")
            for label, part, e2e in (
                ("eager", "denoise_eager", "e2e_eager"),
                ("hybrid", "denoise_graph", "e2e_hybrid"),
            ):
                den, meas = colocated(b, part), get(p, e2e)
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
            base = ", ".join(f"{k} from `{v}`" for k, v in
                             (x.split("=", 1) for x in args.device_baseline))
            lines += [
                "Colocated baselines: "
                + (base + "; others from this run." if base else
                   "all from this run (a second GPU is assumed as fast as cuda:0)."),
                "",
            ]
        else:
            lines += ["No split placement with its colocated baselines in this run.", ""]

    if "pipe_interval" in med.columns and med["pipe_interval"].notna().any():
        lines += ["## Pipelined requests", ""]
        rows = []
        for p in med.index:
            pi = get(p, "pipe_interval")
            if pi is None:
                continue
            info = meta["placements"][p]
            seq, pre, loop = get(p, "e2e_hybrid"), get(p, "prefix"), get(p, "denoise_graph")
            hand = get(p, "handoff") or 0.0
            bound = max(pre, hand + loop)  # two-stage pipeline: the slower stage sets the pace
            diff = info.get("pipe_max_abs_diff_vs_sequential", float("nan"))
            rows.append({
                "placement": p,
                "sequential (ms/request)": f"{seq:.1f}",
                "pipelined (ms/request)": f"{pi:.1f}",
                "throughput gain": f"{seq / pi:.2f}x",
                "bound max(prefix, handoff+loop)": f"{bound:.1f}",
                "measured vs bound": pct(pi, bound),
                "max diff vs sequential": f"{diff:.1e}",
            })
        lines += [md(pd.DataFrame(rows)), ""]
        lines += [
            "The prefix of request t+1 runs while the graph loop of request t replays (same GPU: "
            "its own stream; two GPUs: the second GPU). The bound assumes the stages do not slow "
            "each other; on one GPU they share the SMs, so the measured interval sits above it by "
            "the contention. Requests alternate between two observations and every output is "
            "compared with the sequential one (0 means no race on the shared KV buffers).",
            "",
        ]

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
