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
TRACE_RTW = "smolvlarobotwin_cam3_steps10_chunk50"  # the RoboTwin checkpoint itself (A100)


def _read(f: str) -> tuple[dict, dict, list[int]]:
    e = json.loads(Path(f).read_text())
    info = json.loads((Path(f).parent / "delay_info.json").read_text())
    return e, info, [int(bool(x)) for x in e["per_task"][0]["metrics"]["successes"]]


TASKS = ("beat_block_hammer", "lift_pot")


def run_patterns(task: str) -> tuple[tuple[str, str], ...]:
    """(kind, directory regex) of a task's delay runs and replanning controls. beat_block_hammer
    (the first sweep) uses sweep_d<d>_s<seed>_<job> / ctrl_h<H>_s<seed>_<job>; later tasks
    <task>_d<d>_s<seed>_<job> / <task>_ctrlh<H>_s<seed>_<job>."""
    if task == "beat_block_hammer":
        return (("sweep", r"sweep_d(\d+)_s(\d+)_(\d+)$"), ("ctrl", r"ctrl_h(\d+)_s(\d+)_(\d+)$"))
    return (("sweep", rf"{task}_d(\d+)_s(\d+)_(\d+)$"), ("ctrl", rf"{task}_ctrlh(\d+)_s(\d+)_(\d+)$"))


def load(task: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Delay runs and replanning controls (no delay, replan every H steps, the same replanning
    period as delay d = 50 - H) of one task; only finished runs (eval_info.json written)."""
    rows, ctrl, runs = [], [], []
    files = sorted(glob.glob(str(ROOT / "results/robotwin/*/eval_info.json")))
    for kind, pat in run_patterns(task):
        for f in files:
            m = re.search(pat, Path(f).parent.name)
            if not m:
                continue
            x, seed, job = (int(v) for v in m.groups())
            e, info, succ = _read(f)
            if kind == "sweep":
                assert info["delay_steps"] == x
                dl, h = x, info["exec_horizon"]
            else:
                assert info["delay_steps"] == 0 and info["exec_horizon"] == x
                dl, h = 0, x
            period = h - dl  # control steps between observations used for planning
            for i, s in enumerate(succ):
                r = {"delay_steps": dl, "horizon": h, "replan_period": period, "half": seed,
                     "episode": i, "success": s}
                (rows if kind == "sweep" else ctrl).append(r)
            o = e["overall"]
            runs.append({"run": kind, "delay_steps": dl, "replan_period": period,
                         "episode_seed": seed, "noise_seed": info["noise_seed"], "job": job,
                         "episodes": o["n_episodes"], "successes": o["n_success"],
                         "inferences": info["inferences"],
                         "infer_ms_median": info["infer_ms_median"],
                         "wall_s_per_episode": o["eval_ep_s"]})
    if not runs:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    runs = pd.DataFrame(runs).sort_values(["run", "delay_steps", "replan_period", "episode_seed"],
                                          ascending=[False, True, False, True])
    return pd.DataFrame(rows), pd.DataFrame(ctrl), runs


def paired(a: pd.Series, b: pd.Series) -> dict:
    j = a.to_frame("a").join(b.to_frame("b"), how="inner")
    lost = int(((j.a == 1) & (j.b == 0)).sum())
    gained = int(((j.a == 0) & (j.b == 1)).sum())
    return {"n": len(j), "lost": lost, "gained": gained, "mcnemar_p": round(mcnemar_p(lost, gained), 4)}


def task_report(task: str) -> None:
    d, ctrl, runs = load(task)
    if d.empty:
        return
    if 0 not in set(d.delay_steps):
        print(f"## Task `{task}`: incomplete (no finished d = 0 run yet)\n")
        return
    print(f"## Task `{task}`\n")
    print("### Runs\n")
    print(runs.round(1).to_markdown(index=False), "\n")

    print("### Success rate per delay\n")
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

    print("### Success per half (seed block)\n")
    t = d.groupby(["delay_steps", "half"]).success.agg(lambda x: f"{int(x.sum())}/{len(x)}") \
        .unstack()
    print(t.to_markdown(), "\n")

    print("### Staleness vs replanning rate (control runs)\n")
    print("A delay d makes the policy plan every 50 - d steps from an observation d steps old. "
          "The control replans every H = 50 - d steps with no delay (synchronous, the chunk's "
          "first H actions), so it observes at the same steps; the only difference is staleness. "
          "Same episode and noise seeds, paired by episode.\n")
    out = []
    for h, g in (ctrl.groupby("horizon", sort=False) if len(ctrl) else []):
        dl = 50 - h
        c = g.set_index(key).success
        k, n = int(c.sum()), len(c)
        lo, hi = wilson(k, n)
        dd = d[d.delay_steps == dl].set_index(key).success
        row = {"replan_every": h, "control_success_pct": 100 * k / n,
               "control_ci95": f"{100 * lo:.0f} to {100 * hi:.0f}",
               f"delay_d_success_pct": 100 * dd.mean() if len(dd) else float("nan"),
               "delay_d": dl}
        v0 = paired(base, c)
        row.update({"ctrl_vs_d0 (lost/gained, p)": f"{v0['lost']}/{v0['gained']}, {v0['mcnemar_p']}"})
        if len(dd):
            v = paired(c, dd)
            row.update({"delay_vs_ctrl (lost/gained, p)": f"{v['lost']}/{v['gained']}, {v['mcnemar_p']}"})
        out.append(row)
    if out:
        print(pd.DataFrame(out).sort_values("replan_every", ascending=False).round(1)
              .to_markdown(index=False), "\n")

    print("### What each GPU and execution mode buys\n")
    print("Latency from the SmolVLA traces (3 cameras, 10 flow steps, chunk 50; the traced "
          "inputs are resized to 512, as RoboTwin's 480x640 frames are), converted to "
          f"d = ceil(latency / {CONTROL_MS:.0f} ms); success interpolated linearly between "
          "the measured delays.\n")
    curve = d.groupby("delay_steps").success.mean()
    xs, ys = np.array(curve.index, float), curve.to_numpy()
    rows = []
    for mp in sorted(glob.glob(str(ROOT / "traces" / TRACE_PREFIX / "*" / "meta.json"))
                     + glob.glob(str(ROOT / "traces" / TRACE_RTW / "*" / "meta.json"))):
        if "HOSTEXP" in mp:
            continue
        m = json.loads(Path(mp).read_text())
        gpu = m["gpu"].replace("NVIDIA ", "").replace("GeForce ", "")
        if TRACE_RTW in mp:
            gpu += " (smolvla_robotwin)"
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


def main() -> None:
    print("# RoboTwin 2.0: inference delay vs task success\n")
    print("SmolVLA (`lerobot/smolvla_robotwin`, 480x640, 3 cameras) on the A100, 4 parallel envs. "
          f"Delay d in control steps of {CONTROL_MS:.0f} ms (25 Hz), naive async execution (the "
          "next chunk is planned d steps early and starts at index d, so a 50-action chunk "
          "executes 50 - d actions). Paired: same episode and noise seeds at every d; seed "
          "blocks (`half`) are the jobs a condition is split into.\n")
    for task in TASKS:
        task_report(task)

    print("## Task screen (d = 0, 8 episodes each, seeds 1000..1007)\n")
    sc = []
    for f in sorted(glob.glob(str(ROOT / "results/robotwin/screen_*_d0_*/eval_info.json"))):
        task = re.search(r"screen_(.+)_d0_(\d+)", f).group(1)
        e, info, succ = _read(f)
        lo, hi = wilson(sum(succ), len(succ))
        sc.append({"task": task, "success": f"{sum(succ)}/{len(succ)}",
                   "ci95": f"{100 * lo:.0f} to {100 * hi:.0f}",
                   "wall_s_per_episode": round(e["overall"]["eval_ep_s"])})
    if sc:
        print(pd.DataFrame(sc).to_markdown(index=False), "\n")

    print("## Reading (beat_block_hammer)\n")
    print("- No delay up to 20 steps (800 ms) lowers success on beat_block_hammer (paired "
          "McNemar p >= 0.2 vs d = 0).\n"
          "- The control settles the confound: at the same replanning period, a stale plan "
          "(delay d) and a fresh plan (no delay) succeed equally often, episode for episode "
          "(discordant pairs 15/15 at d = 10, 13/14 at d = 20). Staleness costs nothing here; "
          "the small rise with d comes from replanning more often (also not significant).\n"
          "- So on this task CUDA graphs buy no task success, unlike LIBERO-Long "
          "(results/libero_analysis.md); their value is latency and energy only. The "
          "smolvla_robotwin checkpoint has the base model's latency (A100 eager 378 ms, graph "
          "loop 35.4 ms, hybrid 94 ms; graph output bit-exact).\n"
          "- Outcomes are noisy per episode (71 of 100 seeds change outcome across d), so 100 "
          "paired episodes resolve only effects of about 15 points or more.\n"
          "- One slow, quasi-static task. Of three screened alternatives only lift_pot has a "
          "usable success rate for a delay study (handover_block 1/8, stack_blocks_two 0/8).\n")

if __name__ == "__main__":
    main()
