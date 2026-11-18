"""LIBERO latency vs task success (workloads/libero_eval.py -> results/libero/*.csv).

1. Success rate per policy, precision and simulated delay d (control steps of 50 ms at 20 Hz),
   with Wilson 95% intervals, and the paired change versus d = 0: every delay runs the same
   (task, episode) scenes and noise seeds, so pairs are compared episode by episode
   (counts of episodes that flipped, and an exact McNemar test).
2. What each GPU and execution mode buys: the measured latency of the policy on that GPU
   (traces/*/meta.json; eager step, eager prefix + CUDA-graph loop) converted to a delay
   d = ceil(latency / 50 ms), and the success rate interpolated at that d.

Usage: envs/smolvla/bin/python model/libero_analysis.py > results/libero_analysis.md
"""

from __future__ import annotations

import glob
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CONTROL_MS = 50.0  # LIBERO policies act at 20 Hz
TRACE_PREFIX = {"smolvla": "smolvla_cam3_steps10_chunk50", "pi0": "pi0_libero_fp16_steps10_chunk50",
                "pi05": "pi05_libero_fp16_steps10_chunk50"}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def mcnemar_p(b: int, c: int) -> float:
    """Exact two-sided McNemar test on the discordant pairs b (lost) and c (gained)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main() -> None:
    files = sorted(glob.glob(str(ROOT / "results/libero/*.csv")))
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    d = d.drop_duplicates(["policy", "precision", "exec_horizon", "task_id", "episode",
                          "delay_steps"], keep="last")
    print("# LIBERO-Long: inference delay vs task success\n")
    print(f"Suite {', '.join(sorted(d.suite.unique()))}; delay d in control steps of "
          f"{CONTROL_MS:.0f} ms (20 Hz). Paired: same scenes and noise seeds at every d.\n")

    print("## Success rate per delay\n")
    rows = []
    for (pol, prec, H), g in d.groupby(["policy", "precision", "exec_horizon"]):
        base = g[g.delay_steps == 0].set_index(["task_id", "episode"]).success
        for dl, gg in g.groupby("delay_steps"):
            k, n = int(gg.success.sum()), len(gg)
            lo, hi = wilson(k, n)
            row = {"policy": pol, "precision": prec, "exec_horizon": H, "delay_steps": dl,
                   "delay_ms": dl * CONTROL_MS, "episodes": n, "success_pct": 100 * k / n,
                   "ci95": f"{100 * lo:.0f} to {100 * hi:.0f}"}
            if dl != 0 and len(base):
                s = gg.set_index(["task_id", "episode"]).success
                j = base.to_frame("b").join(s.to_frame("s"), how="inner")
                lost = int(((j.b == 1) & (j.s == 0)).sum())
                gained = int(((j.b == 0) & (j.s == 1)).sum())
                row.update({"lost_vs_d0": lost, "gained_vs_d0": gained,
                            "mcnemar_p": round(mcnemar_p(lost, gained), 4)})
            rows.append(row)
    r = pd.DataFrame(rows)
    print(r.round(1).to_markdown(index=False), "\n")

    print("## Per task at d = 0 (success / episodes)\n")
    t = d[d.delay_steps == 0].groupby(["policy", "precision", "exec_horizon", "task_id"]).success \
        .agg(lambda x: f"{int(x.sum())}/{len(x)}").unstack()
    print(t.to_markdown(), "\n")

    print("## What each GPU and execution mode buys (FP16 sweeps)\n")
    curves = {(p, H): g.groupby("delay_steps").success.mean()
              for (p, prec, H), g in d.groupby(["policy", "precision", "exec_horizon"])
              if prec == "fp16"}
    out = []
    for (pol, H), curve in curves.items():
        for mp in sorted(glob.glob(str(ROOT / "traces" / TRACE_PREFIX[pol] / "*" / "meta.json"))):
            m = json.loads(Path(mp).read_text())
            if "HOSTEXP" in mp:
                continue
            gpu = m["gpu"].replace("NVIDIA ", "").replace("GeForce ", "")
            for mode, key in (("eager", "eager_ms_median"), ("prefix + graph loop", "hybrid_ms_median")):
                lat = m.get(key)
                if not lat:
                    continue
                dl = math.ceil(lat / CONTROL_MS)
                xs, ys = np.array(curve.index, float), curve.to_numpy()
                succ = float(np.interp(dl, xs, ys)) if dl <= xs.max() else float("nan")
                out.append({"policy": pol, "exec_horizon": H, "GPU": gpu, "mode": mode,
                            "latency_ms": lat,
                            "delay_steps": dl, "expected_success_pct": 100 * succ})
    o = pd.DataFrame(out).groupby(["policy", "exec_horizon", "GPU", "mode"]).median(numeric_only=True)
    print(o.round(1).to_markdown(), "\n")
    print("nan: the delay exceeds the largest simulated d.\n")


if __name__ == "__main__":
    main()
