"""RoboTwin 2.0 latency vs task success (workloads/robotwin_delay_eval.py on the cluster A100).

Sweep: lerobot/smolvla_robotwin on beat_block_hammer, simulated inference delay d in control
steps of 40 ms (25 Hz), 100 episodes per d split over two jobs (episode seeds 1000.. with noise
seed 10000, 1052.. with 20000). Every d runs the same episode seeds and noise seeds, so episodes
are paired by (half, index) and compared with an exact McNemar test, as in libero_analysis.py.

Usage: python3 model/robotwin_analysis.py > results/robotwin_analysis.md
"""

from __future__ import annotations

import glob
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from libero_analysis import mcnemar_p, wilson  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CONTROL_MS = 40.0  # RoboTwin runs at 25 Hz
TRACE_PREFIX = "smolvla_cam3_steps10_chunk50"


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, runs = [], []
    for f in sorted(glob.glob(str(ROOT / "results/robotwin/sweep_d*_s*_*/eval_info.json"))):
        m = re.search(r"sweep_d(\d+)_s(\d+)_(\d+)", f)
        dl, seed, job = (int(x) for x in m.groups())
        e = json.loads(Path(f).read_text())
        info = json.loads((Path(f).parent / "delay_info.json").read_text())
        assert info["delay_steps"] == dl
        succ = e["per_task"][0]["metrics"]["successes"]
        for i, s in enumerate(succ):
            rows.append({"delay_steps": dl, "half": seed, "episode": i, "success": int(bool(s))})
        o = e["overall"]
        runs.append({"delay_steps": dl, "episode_seed": seed, "noise_seed": info["noise_seed"],
                     "job": job, "episodes": o["n_episodes"], "successes": o["n_success"],
                     "inferences": info["inferences"],
                     "infer_ms_median": info["infer_ms_median"],
                     "wall_s_per_episode": o["eval_ep_s"]})
    return pd.DataFrame(rows), pd.DataFrame(runs).sort_values(["delay_steps", "episode_seed"])


def main() -> None:
    d, runs = load()
    print("# RoboTwin 2.0: inference delay vs task success\n")
    print("SmolVLA (`lerobot/smolvla_robotwin`, 480x640, 3 cameras), task `beat_block_hammer`, "
          f"A100, 4 parallel envs. Delay d in control steps of {CONTROL_MS:.0f} ms (25 Hz), "
          "naive async execution (the next chunk is planned d steps early and starts at index d, "
          "so a 50-action chunk executes 50 - d actions). Paired: same episode and noise seeds "
          "at every d.\n")

    print("## Runs\n")
    print(runs.round(1).to_markdown(index=False), "\n")

    print("## Success rate per delay\n")
    key = ["half", "episode"]
    base = d[d.delay_steps == 0].set_index(key).success
    out = []
    for dl, g in d.groupby("delay_steps"):
        k, n = int(g.success.sum()), len(g)
        lo, hi = wilson(k, n)
        row = {"delay_steps": dl, "delay_ms": dl * CONTROL_MS, "episodes": n,
               "success_pct": 100 * k / n, "ci95": f"{100 * lo:.0f} to {100 * hi:.0f}"}
        if dl != 0:
            j = base.to_frame("b").join(g.set_index(key).success.to_frame("s"), how="inner")
            lost = int(((j.b == 1) & (j.s == 0)).sum())
            gained = int(((j.b == 0) & (j.s == 1)).sum())
            row.update({"lost_vs_d0": lost, "gained_vs_d0": gained,
                        "mcnemar_p": round(mcnemar_p(lost, gained), 4)})
        out.append(row)
    r = pd.DataFrame(out)
    print(r.round(1).to_markdown(index=False), "\n")

    # Episodes that succeed or fail at every delay: how much of the outcome is the scene.
    w = d.pivot_table(index=key, columns="delay_steps", values="success")
    always, never = int((w.min(axis=1) == 1).sum()), int((w.max(axis=1) == 0).sum())
    print(f"Scene effect: {always} of {len(w)} episodes succeed at every d, {never} fail at "
          f"every d; {len(w) - always - never} change outcome with d.\n")

    print("## Success per half (seed block)\n")
    t = d.groupby(["delay_steps", "half"]).success.agg(lambda x: f"{int(x.sum())}/{len(x)}") \
        .unstack()
    print(t.to_markdown(), "\n")

    print("## What each GPU and execution mode buys\n")
    print("Latency from the SmolVLA traces (3 cameras, 10 flow steps, chunk 50; the traced "
          "inputs are resized to 512, as RoboTwin's 480x640 frames are), converted to "
          f"d = ceil(latency / {CONTROL_MS:.0f} ms); success interpolated linearly between "
          "the measured delays.\n")
    curve = d.groupby("delay_steps").success.mean()
    xs, ys = np.array(curve.index, float), curve.to_numpy()
    rows = []
    for mp in sorted(glob.glob(str(ROOT / "traces" / TRACE_PREFIX / "*" / "meta.json"))):
        if "HOSTEXP" in mp:
            continue
        m = json.loads(Path(mp).read_text())
        gpu = m["gpu"].replace("NVIDIA ", "").replace("GeForce ", "")
        for mode, k in (("eager", "eager_ms_median"), ("prefix + graph loop", "hybrid_ms_median")):
            lat = m.get(k)
            if not lat:
                continue
            dl = math.ceil(lat / CONTROL_MS)
            succ = float(np.interp(dl, xs, ys)) if dl <= xs.max() else float("nan")
            rows.append({"GPU": gpu, "mode": mode, "latency_ms": lat, "delay_steps": dl,
                         "expected_success_pct": 100 * succ})
    o = pd.DataFrame(rows).groupby(["GPU", "mode"]).median(numeric_only=True)
    print(o.round(1).to_markdown(), "\n")
    print("The in-eval policy call (`infer_ms_median` above, eager, includes lerobot "
          "pre/post-processing) is the A100's eager latency in the actual harness.\n")

    print("## Reading\n")
    print("- No delay up to 20 steps (800 ms) lowers success on this task; the point estimates "
          "rise with d, but no change is significant (paired McNemar p >= 0.2). Unlike "
          "LIBERO-Long (results/libero_analysis.md), CUDA graphs buy no task success here: "
          "the curve is flat, so the GPU/mode table above only restates it.\n"
          "- Confound: with a 50-action chunk and naive async execution, a larger d also means "
          "replanning every 50 - d steps instead of 50 (more closed-loop feedback, visible as "
          "more inferences per run). The sweep cannot separate staleness from replanning rate; "
          "a control would replan every 50 - d steps with no delay.\n"
          "- Outcomes are noisy per episode: only "
          f"{always + never} of {len(w)} seeds have the same outcome at every d, so even "
          "paired, 100 episodes resolve only effects of roughly 15 points or more.\n"
          "- beat_block_hammer is a slow, quasi-static task; one task does not establish that "
          "RoboTwin as a whole is delay insensitive.\n")


if __name__ == "__main__":
    main()
