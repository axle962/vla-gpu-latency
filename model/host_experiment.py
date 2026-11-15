"""Host CPU experiment (scripts/host_contention.sh): same GPU, host slowed down in controlled
ways. Source = condition "alone"; targets = smt_alu, smt_mem, share. Kernel times stay the
source's (same GPU); the target's launch constants and dispatch table are its own
measurements, and the per-kernel Python residual p (fitted on the source baseline) is scaled by
each candidate host ratio:
  none        p unchanged (a spec-only model knows nothing about the host)
  dispatch    median ratio of the PyTorch dispatch table (cross_gpu.host_ratio, slot_class)
  framework   dispatch ratio after removing the C++ launch cost (cross_gpu.framework_ratio,
              slot_drv, the final model)
  python      Python probe ratio (cross_gpu.py_ratio, slot_cold_py)
Prints markdown; writes results/host_experiment.csv.

Usage: envs/smolvla/bin/python model/host_experiment.py > results/host_experiment.md
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cross_gpu as cg  # noqa: E402
from launch_model import corrected, fit_b, fit_p  # noqa: E402

CONDS = ["alone", "smt_alu", "smt_mem", "share"]


def main() -> None:
    bench = "results/bench/NVIDIA_GeForce_RTX_2080_Ti_cuda2_20261112-175909"
    for c in CONDS:
        g = f"results/calib/HOSTEXP-{c}-NVIDIA-GeForce-RTX-2080-Ti-*"
        if not list(cg.ROOT.glob(g)):
            continue
        key = f"host_{c}"
        cg.GPUS[key] = dict(calib_glob=g, dispatch_glob=g, bench_glob=bench,
                            probe_glob=g + "/torch_probe_smolvla_env.csv")
        cg.TRACE_KEY[key] = f"HOSTEXP-{c}-"
    keys = [k for k in cg.GPUS if k.startswith("host_")]
    gpus = {k: cg.load_gpu(k, **cg.GPUS[k]) for k in keys}
    src = gpus["host_alone"]
    rows = []
    ratios = []
    for wl in ("smolvla", "act"):
        spec = cg.WORKLOADS[wl]
        src_runs = cg.measured_by_cfg("host_alone", wl)
        base_r, base_m = src_runs[spec["base"]]
        quant = spec["quant"]
        tr0 = cg.load_wl_step(base_r, base_m, wl)
        b = fit_b(tr0, src.calib, base_m[quant["graph_loop"]] * 1e3)
        p = fit_p(corrected(tr0, b), src.calib, base_m["eager_ms_median"] * 1e3)
        traces = {k: corrected(cg.load_wl_step(r, m, wl), b) for k, (r, m) in src_runs.items()}
        for tk, tgt in gpus.items():
            if tk == "host_alone":
                continue
            hr = {"none": 1.0, "dispatch": cg.host_ratio(src, tgt),
                  "framework": cg.framework_ratio(src, tgt), "python": cg.py_ratio(src, tgt)}
            for cfg, (_, tm) in cg.measured_by_cfg(tk, wl).items():
                if cfg not in traces:
                    continue
                sm = src_runs[cfg][1]
                if wl == "smolvla" and cfg == spec["base"]:
                    ratios.append({"condition": tk[5:], **{k: round(v, 3) for k, v in hr.items()},
                                   "measured eager ratio": round(tm["eager_ms_median"]
                                                                 / sm["eager_ms_median"], 3)})
                for name, h in hr.items():
                    pr = cg.predict(traces[cfg], tgt.calib, p * h)
                    vals = {q: (tm[k], pr[q] / 1e3) for q, k in quant.items()}
                    for q, (meas, pv) in vals.items():
                        rows.append({"workload": wl, "config": spec["name"].format(*cfg),
                                     "condition": tk[5:], "host_scaling": name, "quantity": q,
                                     "measured_ms": meas, "source_ms": sm[quant[q]],
                                     "predicted_ms": pv, "error_pct": 100 * (pv - meas) / meas})
    d = pd.DataFrame(rows)
    d.to_csv(cg.ROOT / "results/host_experiment.csv", index=False)
    print("# Host CPU experiment (same RTX 2080 Ti, host slowed down)\n")
    print("## Host ratios (target/alone) vs the measured SmolVLA eager slowdown\n")
    print(pd.DataFrame(ratios).to_markdown(index=False), "\n")
    print("## Measured (ms)\n")
    m = d[d.host_scaling == "none"].pivot_table(
        index=["workload", "config", "quantity"], columns="condition", values="measured_ms")
    src_ms = d[d.host_scaling == "none"].groupby(["workload", "config", "quantity"]).source_ms.first()
    m.insert(0, "alone", src_ms)
    print(m.round(1).to_markdown(), "\n")
    print("## Prediction error (%) per host scaling\n")
    e = d.pivot_table(index=["workload", "config", "quantity", "condition"],
                      columns="host_scaling", values="error_pct")
    print(e[["none", "dispatch", "framework", "python"]].round(1).to_markdown(), "\n")
    d["ape"] = d.error_pct.abs()
    print("## MAPE (%) over conditions and configs\n")
    print(d.pivot_table(index=["workload", "quantity"], columns="host_scaling", values="ape")
          [["none", "dispatch", "framework", "python"]].round(1).to_markdown(), "\n")


if __name__ == "__main__":
    main()
