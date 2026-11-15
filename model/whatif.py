"""Spec-only what-if predictions for GPUs we cannot run (H100, H200, L40S, RTX 4090, RTX PRO
6000), from the A100 traces, and a comparison with the public Yotta Labs OpenVLA numbers.

Kernel transfer is level L0 of model/ablation.py: per kernel, the first TAU_US scale with the
SM clock ratio and the rest with the DRAM bandwidth ratio (spec sheets only). Over our four
real GPUs this level has 38% MAPE on the SmolVLA graph loop and 13% on hybrid
(results/ablation_calibration.md), which is the error bar for these numbers.

The host is held fixed at the A100's (its dispatch table, launch constants and fitted p), so
"eager" answers: what would this GPU give behind the A100 cloud host and its software? For the
Yotta Labs OpenVLA latencies (results/external/yotta_vla_benchmarks_2026.csv) the script also
solves for the host time per kernel (p) that the measured latency implies, given the predicted
GPU-side kernel times: a GPU-side model cannot make the H200 slower than the H100, so any such
gap must come from the host or software stack.

Usage: envs/smolvla/bin/python model/whatif.py > results/whatif.md
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
from launch_model import corrected, eager, fit_b, fit_p, graph, load_calib, load_step  # noqa: E402

ROOT = cg.ROOT
# Spec sheets (boost SM clock in MHz, DRAM bandwidth in GB/s). Source A100 values are from
# its device_info (1410 MHz, 1555 GB/s). The others are public NVIDIA spec-sheet values,
# TODO verify before quoting: H100 SXM5 80GB, H200 SXM 141GB, L40S, RTX 4090, RTX PRO 6000
# Blackwell (workstation). B300 is left out (no reliable public clock/bandwidth figures).
SPECS = {
    "A100": (1410, 1555),
    "H100 SXM": (1980, 3350),
    "H200": (1980, 4800),
    "L40S": (2520, 864),
    "RTX 4090": (2520, 1008),
    "RTX PRO 6000": (2617, 1792),
}
YOTTA_NAME = {"H100 SXM": "H100", "H200": "H200", "RTX 4090": "RTX 4090",
              "RTX PRO 6000": "RTX PRO 6000"}


def spec_t(t: np.ndarray, tgt: str) -> np.ndarray:
    (c0, b0), (c1, b1) = SPECS["A100"], SPECS[tgt]
    return (c0 / c1) * np.minimum(t, cg.TAU_US) + (b0 / b1) * np.maximum(t - cg.TAU_US, 0.0)


def openvla() -> None:
    run = max((ROOT / "traces/openvla7b_fp16").glob("*b-19-28*"))
    m = json.loads((run / "meta.json").read_text())
    calib_dir = ROOT / "results/calib/NVIDIA_A100-SXM4-40GB_b-19-11_20261110-231542"
    disp = next((ROOT / "results/calib").glob("NVIDIA_A100*_b-19-28_*")) / \
        "torch_dispatch_openvla_env.csv"
    c = load_calib(calib_dir, disp)
    tr0 = load_step(run, m["action_tokens"] - 1)
    b = fit_b(tr0, c, m["graph_decode_ms_median"] * 1e3)
    tr = corrected(tr0, b)
    p = fit_p(tr, c, m["eager_ms_median"] * 1e3)
    full, pre, loop = slice(None), slice(0, tr.loop_start), slice(tr.loop_start, None)
    yotta = pd.read_csv(ROOT / "results/external/yotta_vla_benchmarks_2026.csv")
    yotta = yotta[yotta.model == "OpenVLA"].set_index("gpu").latency_ms
    print("## OpenVLA-7B (FP16, 7 action tokens), source A100 trace "
          f"({run.name}, {len(tr.t_us)} kernels; b = {b:.2f} us, p = {p:.1f} us)\n")
    print(f"Measured on the A100 (our host, {m['cpu']}, torch {m['torch']}): eager "
          f"{m['eager_ms_median']:.0f} ms, graph decode {m['graph_decode_ms_median']:.1f} ms, "
          f"eager prefill + graph decode {m['static_hybrid_ms_median']:.0f} ms.\n")
    rows = []
    for g in SPECS:
        t2 = replace(tr, t_us=spec_t(tr.t_us, g))
        gpu_ms = t2.t_us.sum() / 1e3
        e = eager(t2, c, full, "table", p) / 1e3
        gd = graph(t2, c, loop) / 1e3
        hy = (eager(t2, c, pre, "table", p) + graph(t2, c, loop)) / 1e3
        row = {"GPU": g, "kernel time (ms)": gpu_ms, "eager, A100 host (ms)": e,
               "graph decode (ms)": gd, "prefill + graph decode (ms)": hy}
        name = YOTTA_NAME.get(g)
        if name in yotta.index:
            y = float(yotta[name])
            row["Yotta measured (ms)"] = y
            row["host per kernel implied (us)"] = (
                fit_p(t2, c, y * 1e3) + float(np.median(list(c.dispatch_us.values())))
                if y * 1e3 > eager(t2, c, full, "table", 0.0) else float("nan"))
        rows.append(row)
    d = pd.DataFrame(rows)
    print(d.round(1).to_markdown(index=False), "\n")
    print(f"Host time per kernel on the A100 host: p + median dispatch = "
          f"{p + float(np.median(list(c.dispatch_us.values()))):.1f} us. "
          "'implied' = the same quantity solved from the Yotta latency with this GPU's "
          "predicted kernel times (nan: the latency is below the zero-Python bound).\n")


def smolvla() -> None:
    gpus = {"A100": cg.load_gpu("A100", **cg.GPUS["A100"])}
    src = gpus["A100"]
    runs = cg.measured_by_cfg("A100", "smolvla")
    r, m = runs[(3, 10, 50)]
    tr0 = cg.load_wl_step(r, m, "smolvla")
    b = fit_b(tr0, src.calib, m["graph_denoise_ms_median"] * 1e3)
    tr = corrected(tr0, b)
    p = fit_p(tr, src.calib, m["eager_ms_median"] * 1e3)
    print("## SmolVLA (FP16, 3 cameras, 10 flow steps, chunk 50), source A100 trace\n")
    print(f"Measured on the A100: eager {m['eager_ms_median']:.0f} ms, graph loop "
          f"{m['graph_denoise_ms_median']:.1f} ms, hybrid {m['hybrid_ms_median']:.0f} ms.\n")
    rows = []
    for g in SPECS:
        t2 = replace(tr, t_us=spec_t(tr.t_us, g))
        pr = cg.predict(t2, src.calib, p)
        rows.append({"GPU": g, "eager, A100 host (ms)": pr["eager"] / 1e3,
                     "graph loop (ms)": pr["graph_loop"] / 1e3, "hybrid (ms)": pr["hybrid"] / 1e3,
                     "loop graph speedup": pr["eager_loop"] / pr["graph_loop"]})
    print(pd.DataFrame(rows).round(1).to_markdown(index=False), "\n")


def main() -> None:
    print("# Spec-only what-if predictions (model/whatif.py)\n")
    print("Kernel transfer from spec sheets only (clock ratio for the first 5 us of each kernel, "
          "DRAM bandwidth ratio beyond); host fixed at the A100's. Error bar from "
          "results/ablation_calibration.md, level L0 on our four GPUs: SmolVLA graph loop 38%, "
          "hybrid 13%, eager 34% MAPE.\n")
    smolvla()
    openvla()


if __name__ == "__main__":
    main()
