"""torch.compile of the SmolVLA denoising loop (workloads/trace_smolvla_compile.py): measured
comparison with eager and our manual CUDA graph, and cross-GPU prediction of the compiled
times with the final model (slot_drv), from the source GPU's compiled trace only.

The compiled trace (kernels.csv.gz, profiled compile_fused = Inductor kernels launched one by
one) has the same kernels as the compile_graph replay, with ops attached: cuBLAS/CUTLASS GEMMs
under aten::mm/bmm (linear class), Triton kernels as fused elementwise/norm work (add_n class).
Per source GPU, fitted on its own compiled run only:
  b_c  profiler bias, so the graph prediction matches its compile_graph time
  p_c  host time per kernel beyond the dispatch table, so the eager replay matches compile_fused
       (Inductor's Python wrapper launches each Triton kernel)
Targets: kernels by transfer_slots (per-class graph-slot ratios), p_c x framework_ratio.

Usage: envs/smolvla/bin/python model/compile_predict.py > results/compile_predictions.md
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cross_gpu as cg  # noqa: E402
from launch_model import corrected, fit_b, fit_p, load_step  # noqa: E402

CFGS = {1: "smolvlacompile_cam3_steps10_chunk50", 8: "smolvlacompile_cam3_steps10_chunk50_b8"}


def runs(key: str, batch: int) -> list:
    hits = sorted((cg.ROOT / "traces" / CFGS[batch]).glob(f"*{cg.TRACE_KEY[key]}*/meta.json"))
    return [(h.parent, json.loads(h.read_text())) for h in hits]


def main() -> None:
    keys = [k for k in cg.GPUS if runs(k, 1)]
    gpus = {k: cg.load_gpu(k, **cg.GPUS[k]) for k in keys}
    meas, rows = [], []
    for B in CFGS:
        R = {k: min(runs(k, B), key=lambda x: x[1]["eager_ms_std"]) for k in keys if runs(k, B)}
        for k, (r, m) in R.items():
            meas.append({"GPU": k, "batch": B,
                         "eager loop": m["eager_denoise_ms_median"],
                         "compiled, eager launch": m["eager_ms_median"],
                         "manual CUDA graph": m["graph_denoise_ms_median"],
                         "compiled + CUDA graph": m["compile_graph_ms_median"],
                         "kernels (compiled)": m["graph_trace_kernels_per_replay"],
                         "max |diff| vs eager": m["compile_graph_max_abs_diff"],
                         "first compile (s)": m["compile_seconds"]["compile_fused"]})
        for s, (r, m) in R.items():
            S = gpus[s]
            tr0 = replace(load_step(r, 1), loop_start=0)
            b = fit_b(tr0, S.calib, m["compile_graph_ms_median"] * 1e3)
            tr = corrected(tr0, b)
            p = fit_p(tr, S.calib, m["eager_ms_median"] * 1e3)
            for t, (_, tm) in R.items():
                T = gpus[t]
                if s == t:
                    pr = cg.predict(tr, S.calib, p)
                else:
                    t_new = cg.transfer_slots(tr, S, T, tail_bw=False)
                    pr = cg.predict(replace(tr, t_us=t_new), T.calib, p * cg.framework_ratio(S, T))
                for q, key, pv in (("compiled + CUDA graph", "compile_graph_ms_median",
                                    pr["graph_loop"]),
                                   ("compiled, eager launch", "eager_ms_median", pr["eager"])):
                    rows.append({"batch": B, "source": s, "target": t, "quantity": q,
                                 "measured": tm[key], "predicted": pv / 1e3,
                                 "error_pct": 100 * (pv / 1e3 - tm[key]) / tm[key],
                                 "b_us": b, "p_us": p})
    d = pd.DataFrame(rows)
    d.to_csv(cg.ROOT / "results/compile_predictions.csv", index=False)
    print("# torch.compile of the SmolVLA denoising loop (10 flow steps)\n")
    print("## Measured (ms)\n")
    print(pd.DataFrame(meas).round(4).to_markdown(index=False), "\n")
    print("## Cross-GPU prediction error (%), slot_drv, from each source's compiled trace\n")
    x = d[d.source != d.target]
    print(x.pivot_table(index=["batch", "quantity", "target"], columns="source",
                        values="error_pct").round(1).to_markdown(), "\n")
    x = x.assign(ape=x.error_pct.abs())
    print("MAPE (%):", x.groupby(["batch", "quantity"]).ape.mean().round(1).to_dict(), "\n")
    print("Fitted per source (us): " + ", ".join(
        f"{r.source} b{r.batch}: b={r.b_us:.2f}, p={r.p_us:.1f}"
        for r in d[d.source == d.target].drop_duplicates(["source", "batch"]).itertuples()))


if __name__ == "__main__":
    main()
