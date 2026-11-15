"""Rodinia sanity check: cross-GPU kernel-time transfer on classic throughput kernels
(scripts/rodinia.sh -> results/rodinia/<gpu>_<host>_<stamp>/<bench>.csv, nsys per-kernel sums).

For each benchmark, kernel name and (source, target) pair, the source's mean kernel time is
transferred with the same rules the VLA model uses for its kernels:
  same        unchanged
  spec_bw     x spec DRAM bandwidth ratio (spec-ratio baseline)
  spec_clock  x spec SM clock ratio
  spec_mix    clock ratio for the first TAU_US, spec bandwidth ratio beyond (ablation L0)
  meas_mix    measured C++ small-kernel ratio for the first TAU_US, measured bandwidth beyond
              (ablation L1, cross_gpu "lat_bw")
The prediction for a benchmark is sum(instances x predicted mean) over its kernels; its error is
reported against the target's measured total kernel time. Only kernel time is compared (the
Rodinia hosts' CPU phases and copies are not modeled here).

Usage: envs/smolvla/bin/python model/rodinia_predict.py > results/rodinia_predictions.md
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cross_gpu as cg  # noqa: E402
from ablation import spec_clock_mhz  # noqa: E402

GPU_KEY = {"A100": "A100", "RTX_2080_Ti": "2080Ti", "TITAN_V": "TITANV", "RTX_4070": "RTX4070"}
# srad_v2 reads out of bounds at the image border (compute-sanitizer: invalid __global__ read
# in srad_cuda_1); the read faults on Turing/Ada (no kernels run) but not on the TITAN V.
EXCLUDE = {"srad_v2"}
MODES = ["same", "spec_bw", "spec_clock", "spec_mix", "meas_mix"]


def load() -> pd.DataFrame:
    rows = []
    for run in sorted((cg.ROOT / "results/rodinia").glob("*")):
        key = next((v for k, v in GPU_KEY.items() if k in run.name), None)
        if key is None:
            continue
        for f in sorted(run.glob("*.csv")):
            if f.stem in EXCLUDE or f.stat().st_size == 0:
                continue
            d = pd.read_csv(f)
            d = d.rename(columns={"Total Time (ns)": "total_ns", "Instances": "n",
                                  "Avg (ns)": "avg_ns", "Name": "kernel"})
            for r in d.itertuples():
                rows.append({"gpu": key, "run": run.name, "bench": f.stem, "kernel": r.kernel,
                             "n": int(r.n), "avg_us": r.avg_ns / 1e3})
    d = pd.DataFrame(rows)
    # Latest run per GPU only.
    last = d.groupby("gpu").run.max()
    return d[d.run.isin(last)]


def main() -> None:
    d = load()
    keys = sorted(d.gpu.unique())
    gpus = {k: cg.load_gpu(k, **cg.GPUS[k]) for k in keys}
    clk = {k: spec_clock_mhz(k) for k in keys}
    preds = []
    for s in keys:
        for t in keys:
            if s == t:
                continue
            S, T = gpus[s], gpus[t]
            rl = cg.r_lat(S, T)
            for bench, g in d.groupby("bench"):
                gs = g[g.gpu == s].set_index("kernel")
                gt = g[g.gpu == t].set_index("kernel")
                common = gs.index.intersection(gt.index)
                if common.empty:
                    continue
                x = gs.loc[common, "avg_us"].to_numpy()
                n_t = gt.loc[common, "n"].to_numpy()
                meas = float((gt.loc[common, "avg_us"] * n_t).sum())
                tau = cg.TAU_US
                p = {"same": x,
                     "spec_bw": x * S.bw_spec / T.bw_spec,
                     "spec_clock": x * clk[s] / clk[t],
                     "spec_mix": clk[s] / clk[t] * np.minimum(x, tau)
                     + S.bw_spec / T.bw_spec * np.maximum(x - tau, 0),
                     "meas_mix": rl * np.minimum(x, tau)
                     + S.bw_dram / T.bw_dram * np.maximum(x - tau, 0)}
                for mode, v in p.items():
                    pred = float((v * n_t).sum())
                    preds.append({"source": s, "target": t, "bench": bench, "mode": mode,
                                  "measured_ms": meas / 1e3, "predicted_ms": pred / 1e3,
                                  "error_pct": 100 * (pred - meas) / meas,
                                  "median_kernel_us": float(np.median(x))})
    p = pd.DataFrame(preds)
    p.to_csv(cg.ROOT / "results/rodinia_predictions.csv", index=False)
    p["ape"] = p.error_pct.abs()
    print("# Rodinia: cross-GPU kernel-time transfer (model/rodinia_predict.py)\n")
    print(f"GPUs: {', '.join(keys)}. Error of total kernel time per benchmark, all ordered "
          "pairs.\n")
    print("## Measured total kernel time (ms) and median kernel duration\n")
    m = d.assign(t=d.avg_us * d.n / 1e3).groupby(["bench", "gpu"]).t.sum().unstack()
    med = d.groupby("bench").avg_us.median().rename("median kernel (us, all GPUs)")
    print(pd.concat([m, med], axis=1).round(2).to_markdown(), "\n")
    print("## MAPE (%) per benchmark and transfer rule\n")
    t = p.pivot_table(index="bench", columns="mode", values="ape")[MODES]
    t.loc["all"] = p.pivot_table(columns="mode", values="ape")[MODES].iloc[0]
    print(t.round(1).to_markdown(), "\n")


if __name__ == "__main__":
    main()
