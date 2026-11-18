"""Multi-policy hosting report: measured latency per hosted policy and predicted vs measured.

Reads one run of workloads/multipolicy.py (results/multipolicy/<run>/: latency.csv,
configs.json, meta.json) and writes report.md and summary.csv there.

Prediction (the "hosting model"): each policy copy is a client of a closed queueing network
with two stations, the host CPU and the GPU, and it alternates between them: a step spends
Z = T_solo - D on the host (Python and dispatch, which the eager step is dominated by) and D on
the GPU (the kernels' busy time, from the solo profile, with the profiler's ~0.95 us per
kernel removed). Nothing in it is measured with more than one policy running.
  procs, procs-mps  the host is a delay station (every process has its own cores), the GPU a
                    single FIFO server (contexts time-slice, kernels do not overlap)
  threads           both stations are single FIFO servers (the GIL serializes the host side)
Solved with the Bard-Schweitzer approximate MVA (one client per class):
  R_ik = D_ik (1 + sum_{j != i} Q_jk),  X_i = 1 / sum_k R_ik,  Q_ik = X_i R_ik,
predicted step latency T_i = sum_k R_ik, throughput X_i. Two bounds frame it: no interference
(T_solo) and the model. Under MPS kernels of different processes may run concurrently, so the
model is an upper bound there and the gap to it is the measured overlap.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

MODES = ["eager", "graph", "loop"]
MIN_STEPS = 5  # copies with fewer timed steps in the window are left out of the prediction


def mva(classes: list[tuple[float, float]], cpu_queue: bool, iters: int = 5000):
    """classes: (Z_i host ms, D_i GPU ms) per client. Returns (T_i ms, X_i steps/ms)."""
    n = len(classes)
    qc, qg = [0.0] * n, [0.0] * n
    rc = rg = x = None
    for _ in range(iters):
        rc, rg, x = [], [], []
        for i, (z, d) in enumerate(classes):
            og = sum(qg[j] for j in range(n) if j != i)
            oc = sum(qc[j] for j in range(n) if j != i)
            rg.append(d * (1 + og))
            rc.append(z * (1 + oc) if cpu_queue else z)
            x.append(1.0 / (rc[-1] + rg[-1]) if rc[-1] + rg[-1] > 0 else 0.0)
        qc_n = [x[i] * rc[i] for i in range(n)]
        qg_n = [x[i] * rg[i] for i in range(n)]
        delta = max(abs(a - b) for a, b in zip(qc + qg, qc_n + qg_n))
        qc = [0.5 * a + 0.5 * b for a, b in zip(qc, qc_n)]
        qg = [0.5 * a + 0.5 * b for a, b in zip(qg, qg_n)]
        if delta < 1e-10:
            break
    return [rc[i] + rg[i] for i in range(n)], x


def pct(pred: float, meas: float) -> float:
    return 100.0 * (pred - meas) / meas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="results/multipolicy/<run> directory")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    run = Path(args.run)
    lat = pd.read_csv(run / "latency.csv")
    cfgs = {c["config"]: c for c in json.loads((run / "configs.json").read_text())}
    meta = json.loads((run / "meta.json").read_text())
    order = [c for c in cfgs if c in set(lat.config)]

    # Middle 80% of each phase's window, per copy.
    rows = []
    for (cfg, mode, w, ci, pol), g in lat.groupby(["config", "mode", "worker", "copy", "policy"]):
        dur = cfgs[cfg]["duration_s"]
        m = g[(g.t_s >= 0.1 * dur) & (g.t_s <= 0.9 * dur)].wall_ms
        if len(m) == 0:
            m = g.wall_ms
        rows.append({
            "config": cfg, "hosting": cfgs[cfg]["hosting"], "n_policies": len(cfgs[cfg]["policies"]),
            "batch": cfgs[cfg]["batch"], "mode": mode, "worker": w, "copy": ci, "policy": pol,
            "n": len(m), "median_ms": m.median(), "mean_ms": m.mean(), "p95_ms": m.quantile(0.95),
            "min_ms": m.min(), "steps_per_s": len(m) / (0.8 * dur),
        })
    summ = pd.DataFrame(rows)
    summ.to_csv(run / "summary.csv", index=False)

    # Solo reference: procs with one policy, per policy and mode.
    solo: dict[tuple[str, str], dict] = {}
    for cfg, c in cfgs.items():
        if c["hosting"] == "procs" and len(c["policies"]) == 1 and cfg in order:
            pol = c["policies"][0]
            prof = c["workers"].get("0", c["workers"].get(0, {})).get("profile", {})
            for mode in MODES:
                s = summ[(summ.config == cfg) & (summ["mode"] == mode)]
                if len(s) == 0:
                    continue
                t = float(s.median_ms.iloc[0])
                p = prof.get(mode, {})
                d = p.get("gpu_busy_ms_per_step_corrected")
                solo[(pol, mode)] = {
                    "config": cfg, "T": t, "p95": float(s.p95_ms.iloc[0]),
                    "D": d, "D_raw": p.get("gpu_busy_ms_per_step"),
                    "kernels": p.get("kernels_per_step"), "Z": (t - d) if d is not None else None,
                }

    lines = [
        "# Multi-policy hosting on one GPU",
        "",
        f"GPU `{meta['gpu']}` (driver {meta['driver']}), host `{meta['host']}` ({meta['cpu']}, "
        f"{meta['affinity_cpus']} CPUs in the affinity mask), torch {meta['torch']} cu"
        f"{meta['cuda_runtime']}. Policy settings: {json.dumps(meta['policy_args'])}. "
        "Closed loop (a new step as soon as the previous one returned), median of the middle "
        "80% of each window.",
        "",
        "## Policies alone (procs x1)",
        "",
    ]
    srows = []
    for (pol, mode), s in solo.items():
        srows.append({
            "policy": pol, "mode": mode, "step ms": f"{s['T']:.1f}", "p95 ms": f"{s['p95']:.1f}",
            "GPU busy ms (profiled)": "" if s["D_raw"] is None else f"{s['D_raw']:.1f}",
            "GPU busy ms (corrected D)": "" if s["D"] is None else f"{s['D']:.1f}",
            "host ms (Z = step - D)": "" if s["Z"] is None else f"{s['Z']:.1f}",
            "GPU share": "" if s["D"] is None else f"{s['D'] / s['T']:.2f}",
            "kernels/step": "" if s["kernels"] is None else f"{s['kernels']:.0f}",
        })
    lines += [pd.DataFrame(srows).to_markdown(index=False) if srows else "(no solo run)", ""]
    lines += [
        "D removes the profiler's 0.95 us per kernel (CLAUDE.md 0.1); Z is what the step spends "
        "off the GPU (Python, dispatch, syncs). A policy with a small GPU share leaves room for "
        "others.",
        "",
        "## Measured: every hosted configuration",
        "",
    ]
    mrows = []
    pred_rows = []
    for cfg in order:
        c = cfgs[cfg]
        hosting, pols, batch = c["hosting"], c["policies"], c["batch"]
        for mode in MODES:
            s = summ[(summ.config == cfg) & (summ["mode"] == mode)]
            if len(s) == 0:
                continue
            agg = s.steps_per_s.sum() * batch
            for pol, sp in s.groupby("policy", sort=False):
                med = sp.median_ms.median()
                p95 = sp.p95_ms.median()
                ref = solo.get((pol, mode))
                r = {
                    "config": cfg, "mode": mode, "policy": pol, "copies": len(sp),
                    "median ms": f"{med:.1f}", "p95 ms": f"{p95:.1f}",
                    "vs alone": f"{med / ref['T']:.2f}x" if ref else "",
                    "steps/s (all policies)": f"{agg:.2f}",
                    "steps/copy": int(sp.n.median()),
                }
                if batch > 1:
                    r["copies"] = f"batch {batch}"
                mrows.append(r)
            # Prediction from parts measured alone.
            if hosting in ("procs", "procs-mps", "threads") and len(pols) > 1:
                refs = [solo.get((p, mode)) for p in pols]
                if any(r is None or r["D"] is None for r in refs):
                    continue
                classes = [(max(r["Z"], 0.0), max(r["D"], 1e-6)) for r in refs]
                t_pred, x_pred = mva(classes, cpu_queue=(hosting == "threads"))
                for pol in dict.fromkeys(pols):
                    idx = [i for i, p in enumerate(pols) if p == pol]
                    if s[s.policy == pol].n.median() < MIN_STEPS:
                        continue  # starved copies: a median of 1 or 2 steps says nothing
                    meas = s[s.policy == pol].median_ms.median()
                    ref = solo[(pol, mode)]
                    pred = sum(t_pred[i] for i in idx) / len(idx)
                    pred_rows.append({
                        "config": cfg, "hosting": hosting, "mode": mode, "policy": pol,
                        "alone ms": ref["T"], "model ms": pred, "measured ms": meas,
                        "model err %": pct(pred, meas),
                        "no-interference err %": pct(ref["T"], meas),
                        "pred steps/s": sum(x_pred[i] for i in idx) * 1e3,
                        "meas steps/s": s[s.policy == pol].steps_per_s.sum(),
                    })
    lines += [pd.DataFrame(mrows).to_markdown(index=False), ""]
    lines += [
        "`vs alone` is the slowdown against the same policy and mode hosted alone (procs x1). "
        "`steps/s` sums every hosted copy (times the batch for batch hosting). `steps/copy` is "
        f"the number of timed steps behind the median; rows under {MIN_STEPS} are starved "
        "copies (the GPU time-slicer or the GIL let one copy run for seconds) and are left out "
        "of the prediction tables.",
        "",
        "## Predicted from the policies alone (closed queueing network, host + GPU)",
        "",
    ]
    if pred_rows:
        pr = pd.DataFrame(pred_rows)
        show = pr.copy()
        for col in ("alone ms", "model ms", "measured ms", "pred steps/s", "meas steps/s"):
            show[col] = show[col].map(lambda v: f"{v:.1f}")
        for col in ("model err %", "no-interference err %"):
            show[col] = show[col].map(lambda v: f"{v:+.1f}")
        lines += [show.to_markdown(index=False), ""]
        pr["ape_model"] = pr["model err %"].abs()
        pr["ape_none"] = pr["no-interference err %"].abs()
        mape = pr.groupby(["hosting", "mode"]).agg(
            n=("config", "size"), model_MAPE=("ape_model", "mean"), no_interference_MAPE=("ape_none", "mean")
        ).round(1)
        lines += ["MAPE (%) of the per-policy latency:", "", mape.to_markdown(), ""]
        pr.drop(columns=["ape_model", "ape_none"]).to_csv(run / "predictions.csv", index=False)
        lines += [
            "Reading: for `procs` the model assumes the GPU serializes the contexts' kernels; a "
            "measured latency below the model means kernels of different processes overlapped "
            "(expected under `procs-mps`, where the model is an upper bound). For `threads` "
            "the model also serializes the host side (GIL); PyTorch releases the GIL inside its "
            "C++ ops, so the measurement should sit between the two bounds.",
            "",
        ]
    else:
        lines += ["No prediction: a solo run with --profile is needed for every policy.", ""]

    checks = []
    for cfg in order:
        for w, info in cfgs[cfg]["workers"].items():
            for ci, cp in enumerate(info["copies"]):
                checks.append({
                    "config": cfg, "worker": w, "copy": ci, "policy": cp.get("hf_model", ""),
                    "split vs stock": cp.get("split_vs_stock_max_abs_diff", ""),
                    "graph vs eager": cp.get("graph_vs_eager_max_abs_diff", ""),
                })
    ck = pd.DataFrame(checks)
    worst = ck[["split vs stock", "graph vs eager"]].apply(pd.to_numeric, errors="coerce").max()
    lines += [
        "## Correctness",
        "",
        f"Largest |diff| over all copies: split loop vs stock step {worst['split vs stock']:.1e}, "
        f"graph vs eager {worst['graph vs eager']:.1e} ({len(ck)} policy copies loaded).",
        "",
    ]
    mps = [c["mps"] for c in cfgs.values() if c.get("mps")]
    if mps:
        lines += [f"MPS state recorded by the procs-mps runs: {sorted(set(mps))}", ""]
    out = Path(args.out) if args.out else run / "report.md"
    out.write_text("\n".join(lines))
    print("\n".join(lines))
    print("wrote", out)


if __name__ == "__main__":
    main()
