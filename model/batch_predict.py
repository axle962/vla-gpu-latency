"""Batch-size sweep (traces/smolvlabatch_b<B>_cam3_steps10_chunk50, batch 1 = the normal
baseline): does the model hold as kernels get longer and the CUDA-graph speedup shrinks?

b and p are fitted on each GPU's batch-1 baseline only (as in model/cross_gpu.py) and held
fixed for every batch size:
  same-GPU   the batch-B trace of a GPU predicts that GPU's batch-B eager / graph loop / hybrid
  cross-GPU  the source's batch-B trace, transferred with slot_drv, predicts the target's
             batch-B measurements (no target traces)
Also reports the batch size at which the loop's graph speedup crosses 2x (log-linear
interpolation in B), measured vs predicted.

Usage: envs/smolvla/bin/python model/batch_predict.py > results/batch_predictions.md
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cross_gpu as cg  # noqa: E402
from launch_model import corrected, fit_b, fit_p  # noqa: E402

BATCHES = [1, 2, 4, 8, 16]


def runs(key: str) -> dict:
    """batch -> (run dir, meta); batch 1 from the baseline config (smallest eager std)."""
    out = {1: cg.measured_by_cfg(key, "smolvla")[(3, 10, 50)]}
    for b in BATCHES[1:]:
        hits = sorted((cg.ROOT / "traces").glob(
            f"smolvlabatch_b{b}_cam3_steps10_chunk50/*{cg.TRACE_KEY[key]}*/meta.json"))
        if hits:
            m = [(h.parent, json.loads(h.read_text())) for h in hits]
            out[b] = min(m, key=lambda x: x[1]["eager_ms_median"])
    return out


def crossover(bs: list, sp: list, level: float = 2.0) -> float:
    for i in range(1, len(bs)):
        if sp[i - 1] >= level > sp[i]:
            f = (np.log(sp[i - 1]) - np.log(level)) / (np.log(sp[i - 1]) - np.log(sp[i]))
            return float(np.exp(np.log(bs[i - 1]) + f * (np.log(bs[i]) - np.log(bs[i - 1]))))
    return float("nan")  # > largest batch (or never above)


def main() -> None:
    keys = [k for k in cg.GPUS if len(runs(k)) > 1]
    gpus = {k: cg.load_gpu(k, **cg.GPUS[k]) for k in keys}
    R = {k: runs(k) for k in keys}
    fit = {}
    for k in keys:
        r, m = R[k][1]
        tr = cg.load_wl_step(r, m, "smolvla")
        b = fit_b(tr, gpus[k].calib, m["graph_denoise_ms_median"] * 1e3)
        p = fit_p(corrected(tr, b), gpus[k].calib, m["eager_ms_median"] * 1e3)
        fit[k] = (b, p)
    rows = []
    for s in keys:
        S = gpus[s]
        b, p = fit[s]
        for B, (r, m) in R[s].items():
            tr = corrected(cg.load_wl_step(r, m, "smolvla"), b)
            for t in keys:
                if B not in R[t]:
                    continue
                T, tm = gpus[t], R[t][B][1]
                if s == t:
                    pr = cg.predict(tr, S.calib, p)
                else:
                    t_new = cg.transfer_slots(tr, S, T, tail_bw=False)
                    pr = cg.predict(replace(tr, t_us=t_new), T.calib, p * cg.framework_ratio(S, T))
                vals = {"eager": (tm["eager_ms_median"], pr["eager"] / 1e3),
                        "eager_loop": (tm["eager_denoise_ms_median"], pr["eager_loop"] / 1e3),
                        "graph_loop": (tm["graph_denoise_ms_median"], pr["graph_loop"] / 1e3),
                        "hybrid": (tm["hybrid_ms_median"], pr["hybrid"] / 1e3),
                        "speedup_loop": (tm["eager_denoise_ms_median"] / tm["graph_denoise_ms_median"],
                                         pr["eager_loop"] / pr["graph_loop"])}
                for q, (meas, pv) in vals.items():
                    rows.append({"source": s, "target": t, "batch": B, "quantity": q,
                                 "measured": meas, "predicted": pv,
                                 "error_pct": 100 * (pv - meas) / meas})
    d = pd.DataFrame(rows)
    d.to_csv(cg.ROOT / "results/batch_predictions.csv", index=False)
    d["ape"] = d.error_pct.abs()
    d["kind"] = np.where(d.source == d.target, "same-GPU", "cross-GPU")
    print("# SmolVLA batch sweep (model/batch_predict.py)\n")
    print("b and p fitted on each GPU's batch-1 baseline, held fixed for all batches; cross-GPU "
          "uses slot_drv.\n")
    print("## Measured (ms; speedup = eager loop / graph loop)\n")
    me = d[d.source == d.target].pivot_table(index=["target", "batch"], columns="quantity",
                                             values="measured")
    print(me[["eager", "eager_loop", "graph_loop", "hybrid", "speedup_loop"]].round(1)
          .to_markdown(), "\n")
    print("## MAPE (%) by batch\n")
    t = d.pivot_table(index=["kind", "batch"], columns="quantity", values="ape")
    print(t[["eager", "eager_loop", "graph_loop", "hybrid", "speedup_loop"]].round(1)
          .to_markdown(), "\n")
    print("## Batch size where the loop's graph speedup falls below 2x\n")
    out = []
    for t in keys:
        bs = sorted(R[t])
        meas = d[(d.source == t) & (d.target == t) & (d.quantity == "speedup_loop")] \
            .set_index("batch").measured.reindex(bs)
        row = {"GPU": t, "measured": crossover(bs, list(meas))}
        for s in keys:
            pr = d[(d.source == s) & (d.target == t) & (d.quantity == "speedup_loop")] \
                .set_index("batch").predicted.reindex(bs)
            row[f"pred. from {s}"] = crossover(bs, list(pr))
        out.append(row)
    print(pd.DataFrame(out).round(1).to_markdown(index=False), "\n")
    print("nan: not crossed within the measured batches (speedup still >= 2 at the largest).\n")


if __name__ == "__main__":
    main()
