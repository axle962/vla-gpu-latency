"""Graph-worthiness rule: predict the CUDA-graph speedup of a region before capturing it.

Capturing a region as a CUDA graph takes engineering (static buffers, no host syncs, constant
uploads moved out, buffers kept alive), so the question a practitioner asks first is whether
it would pay off. This rule answers it from eager mode only:

  T_e  eager wall time of the region                       (time it, no profiler)
  K    kernel launches in the region                        (one profiled eager step)
  D    summed GPU duration of those kernels in eager mode   (same profiled step)

  T_graph ~= D + K * eps        S = T_e / T_graph          worth it if S >= threshold

eps is the net per-kernel cost of a kernel in a graph relative to its traced eager duration:
the graph node cost (0.3 to 1.1 us in our C++ calibration) minus the profiler's bias on each
traced duration (about 1 us). With D / K = mean kernel time t and T_e / K = eager time per
kernel c (mostly host dispatch), S ~= c / (t + eps): graphs pay off when the host spends much
longer per kernel than the GPU does.

Where eps comes from (each a separate prediction column, all evaluated on the same runs):

  global    one value, fitted on ONE run (2080 Ti SmolVLA baseline), used everywhere
  per_gpu   one value per GPU, fitted on that GPU's SmolVLA baseline run
  lofo      leave one policy family out: per GPU, the median eps of the runs of all OTHER
            families on that GPU (no run of the predicted family is used)
  logo      leave one GPU out: the median of the other GPUs' per_gpu values (nothing measured
            on the predicted GPU)
  micro     no workload run at all: eps from workloads/eps_probe.py on that GPU (graph time
            per kernel minus CUPTI-traced eager duration on chains of small ops); only for
            GPUs with a results/calib/*/eps_probe_*_env.csv

Baselines: no_eps (S = T_e / D, no per-kernel term) and busy (S = 1 / gpu_busy_frac of the
whole step, what a utilization counter suggests).

Evaluation set: every run with an eager per-kernel trace and a measured graph time of the same
region, EXCEPT the SmolVLA baseline config (smolvla_cam3_steps10_chunk50) on every GPU, which
the global and per_gpu fits use. Workloads: SmolVLA (camera, step, chunk, batch and RoboTwin
variants), SmolVLA under torch.compile (eager launches vs compile + CUDA graph: fused Triton
kernels), pi0, pi0.5, ACT, Diffusion Policy (default and autotuned cuDNN), and the
autoregressive decoders OpenVLA-7B and pi0-FAST (eager static-cache decode vs its graph).

Regions. Flow/DDPM loops: the trailing run of num_steps identical blocks of kernel names in the
eager step (template arguments stripped). ACT and torch.compile: the whole profiled step.
Decoders: the tokens between the first and last occurrence of a kernel that runs exactly once
per generated token (argmax for OpenVLA, a cache copy for pi0-FAST); their eager trace is of
the stock dynamic-cache decoder (about 3% more kernels per token than the static-cache path
that is timed and graphed), stated as a limitation.

Statistics: runs repeat some (config, GPU) situations, so errors are also aggregated per
situation, and 95% confidence intervals come from a bootstrap that resamples configs.
Decisions are scored at every threshold from 1.2x to 15x, with the runs whose measured speedup
lies within 25% of the threshold reported separately (the hard cases).

  python model/graph_worthiness.py [--threshold 1.5] [--out results/graph_worthiness]
writes <out>.csv, <out>.md and paper/figures/graph_worthiness{,_threshold}.{pdf,png}.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from launch_model import load_calib  # noqa: E402

BASE_CONFIG = "smolvla_cam3_steps10_chunk50"
FIT_RUN = f"traces/{BASE_CONFIG}/NVIDIA_GeForce_RTX_2080_Ti_cuda2_*"
GPU_SHORT = {"A100": "A100", "2080 Ti": "2080 Ti", "TITAN V": "TITAN V", "4070": "RTX 4070"}
CALIB_KEY = {"A100": "NVIDIA_A100", "2080 Ti": "NVIDIA_GeForce_RTX_2080_Ti",
             "TITAN V": "NVIDIA_TITAN_V", "RTX 4070": "NVIDIA_GeForce_RTX_4070"}
POLICY = {"smolvla": "SmolVLA", "smolvlabatch": "SmolVLA", "smolvlarobotwin": "SmolVLA",
          "smolvlacompile": "SmolVLA compiled", "pi0": "pi0", "pi05": "pi0.5", "act": "ACT",
          "diffusion": "Diffusion Policy", "diffusionbench": "Diffusion Policy",
          "openvla7b": "OpenVLA", "pi0fast": "pi0-FAST"}
VARIANTS = ["global", "per_gpu", "lofo", "logo", "micro", "no_eps", "busy"]
VARIANT_LABEL = {"global": "one eps (1 run)", "per_gpu": "eps per GPU", "lofo":
                 "leave policy out", "logo": "leave GPU out", "micro": "eps from probe",
                 "no_eps": "T_e/D (no eps)", "busy": "1/busy_frac"}
HARD_BAND = 1.25


def short_gpu(name: str) -> str:
    for k, v in GPU_SHORT.items():
        if k in name:
            return v
    return name


def strip(names: pd.Series) -> list[str]:
    return names.str.split("<").str[0].str.split("(").str[0].tolist()


def load_run(meta_path: str) -> dict | None:
    d = Path(meta_path).parent
    m = json.loads(Path(meta_path).read_text())
    family = d.parent.name.split("_")[0]
    if family not in POLICY:
        return None
    kc = d / "kernels.csv.gz"
    if not kc.exists():
        return None
    if family == "smolvlacompile":
        if "compile_graph_ms_median" not in m:
            return None
        t_e, t_g, region = m["eager_ms_median"], m["compile_graph_ms_median"], "step"
    elif not m.get("graph_ok", False):
        return None
    elif "static_decode_ms_median" in m and "graph_decode_ms_median" in m:
        t_e, t_g, region = m["static_decode_ms_median"], m["graph_decode_ms_median"], "decode"
    elif "graph_denoise_ms_median" in m and "eager_denoise_ms_median" in m:
        t_e, t_g, region = m["eager_denoise_ms_median"], m["graph_denoise_ms_median"], "loop"
    elif "graph_ms_median" in m:
        t_e, t_g, region = m["eager_ms_median"], m["graph_ms_median"], "step"
    else:
        return None
    k = pd.read_csv(kc, usecols=["cat", "name", "dur_us", "step", "seq"])
    k = k[k.cat == "kernel"].sort_values(["step", "seq"])
    ks, ds = [], []
    for _, s in k.groupby("step"):
        s = s.reset_index(drop=True)
        if region == "loop":
            n = loop_kernels(strip(s.name), int(m.get("num_steps") or 0))
            sel = s.iloc[-n:] if n else None
        elif region == "decode":
            ntok = int(m.get("decode_steps") or m.get("action_tokens") or 0)
            sel = decode_tokens(s, ntok)
        else:
            sel = s
        if sel is None or not len(sel):
            return None
        ks.append(len(sel))
        ds.append(sel.dur_us.sum())
    return {
        "run": str(d), "workload": family, "policy": POLICY[family], "config": d.parent.name,
        "gpu": short_gpu(m["gpu"]), "region": region, "K": int(np.median(ks)),
        "T_e_ms": t_e, "D_ms": float(np.median(ds)) / 1e3, "T_g_ms": t_g,
        "busy_frac": m.get("gpu_busy_frac", np.nan),
        "host_cond": d.name.split("-")[1] if d.name.startswith("HOSTEXP-") else "",
    }


def loop_kernels(names: list[str], reps: int) -> int | None:
    """Kernels in the trailing loop of `reps` iterations: the smallest block length L such that
    the last reps - 1 blocks of L kernel names are identical (the first iteration may differ,
    e.g. Diffusion Policy's first DDPM step); the loop is then the last reps * L kernels.
    None if there is no such block."""
    if reps < 3:
        return None
    n = len(names)
    for L in range(20, n // reps + 1):
        tail = names[n - L:]
        if names[n - 2 * L:n - L] != tail:
            continue
        if all(names[n - (r + 1) * L:n - r * L] == tail for r in range(2, reps - 1)):
            return L * reps
    return None


def decode_tokens(s: pd.DataFrame, ntok: int) -> pd.DataFrame | None:
    """Kernels of decode steps 2..ntok: between the first and the last occurrence of a kernel
    that runs exactly once per generated token (the most regularly spaced such kernel)."""
    if ntok < 3:
        return None
    vc = s.name.value_counts()
    best, best_cv = None, np.inf
    for name in vc[vc == ntok].index:
        pos = np.flatnonzero((s.name == name).to_numpy())
        gaps = np.diff(pos)
        cv = gaps.std() / gaps.mean()
        if cv < best_cv:
            best, best_cv = pos, cv
    if best is None:
        return None
    return s.iloc[best[0] + 1:best[-1] + 1]


def probe_eps() -> dict[str, float]:
    """eps per GPU from workloads/eps_probe.py (median over its ops, per kernel)."""
    out: dict[str, float] = {}
    for gname, key in CALIB_KEY.items():
        files = sorted(glob.glob(f"results/calib/{key}*/eps_probe_*_env.csv"))
        if not files:
            continue
        rows = [r for r in csv.DictReader(open(files[-1])) if r["mode"] == "eps"]
        if rows:
            out[gname] = float(np.median([float(r["median_us"]) for r in rows]))
    return out


def node_cost() -> dict[str, float]:
    """C++ graph node cost per GPU (bench/calib launch_latency spin chains), for the table."""
    out = {}
    for gname, key in CALIB_KEY.items():
        for d in sorted(glob.glob(f"results/calib/{key}*"), reverse=True):
            d = Path(d)
            disp = sorted(Path("results/calib").glob(f"{key}*/torch_dispatch_smolvla_env.csv"))
            if (d / "launch_eager_loading.csv").exists() and disp:
                out[gname] = load_calib(d, disp[0]).delta_node_us
                break
    return out


def run_eps(r) -> float:
    return (r.T_g_ms - r.D_ms) / r.K * 1e3


def bootstrap_ci(err: pd.Series, groups: pd.Series, n: int = 2000, seed: int = 0):
    """95% CI of the MAPE, resampling groups (configs) with replacement."""
    rng = np.random.default_rng(seed)
    a = np.abs(err.to_numpy())
    g = groups.to_numpy()
    keys = np.unique(g)
    idx = {k: np.flatnonzero(g == k) for k in keys}
    vals = []
    for _ in range(n):
        pick = rng.choice(keys, size=len(keys), replace=True)
        vals.append(np.concatenate([a[idx[k]] for k in pick]).mean())
    return np.percentile(vals, [2.5, 97.5])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=1.5, help="speedup that makes it worth it")
    ap.add_argument("--out", default="results/graph_worthiness")
    ap.add_argument("--fig", default="paper/figures/graph_worthiness")
    args = ap.parse_args()

    rows = [r for f in sorted(glob.glob("traces/*/*/meta.json")) if (r := load_run(f))]
    df = pd.DataFrame(rows).dropna(subset=["D_ms"])
    df["S_meas"] = df.T_e_ms / df.T_g_ms
    df["eps_run_us"] = df.apply(run_eps, axis=1)
    df["c_us"] = df.T_e_ms / df.K * 1e3
    df["t_us"] = df.D_ms / df.K * 1e3

    # --- eps per variant -------------------------------------------------------------------
    fit = df[df.run.apply(lambda r: Path(r).match(FIT_RUN))].iloc[0]
    eps_global = run_eps(fit)
    normal = df[df.host_cond == ""]
    base = normal[normal.config == BASE_CONFIG].sort_values("run").groupby("gpu").first()
    eps_gpu = {g: run_eps(r) for g, r in base.iterrows()}
    eps_micro = probe_eps()
    df["eps_global"] = eps_global
    df["eps_per_gpu"] = df.gpu.map(eps_gpu)
    df["eps_logo"] = df.gpu.map(
        {g: float(np.median([v for h, v in eps_gpu.items() if h != g])) for g in eps_gpu})
    df["eps_lofo"] = [
        float(normal[(normal.gpu == r.gpu) & (normal.policy != r.policy)].eps_run_us.median())
        for r in df.itertuples()]
    df["eps_micro"] = df.gpu.map(eps_micro)
    for v in ("global", "per_gpu", "lofo", "logo", "micro"):
        df[f"S_{v}"] = df.T_e_ms / (df.D_ms + df.K * df[f"eps_{v}"] / 1e3)
    df["S_no_eps"] = df.T_e_ms / df.D_ms
    df["S_busy"] = 1.0 / df.busy_frac
    for v in VARIANTS:
        df[f"err_{v}"] = 100 * (df[f"S_{v}"] - df.S_meas) / df.S_meas

    ev = df[(df.config != BASE_CONFIG) & (df.host_cond == "")].copy()
    stress = df[df.host_cond != ""].copy()
    variants = [v for v in VARIANTS if ev[f"S_{v}"].notna().all()]
    if "micro" not in variants and ev.S_micro.notna().any():
        variants.insert(variants.index("no_eps"), "micro")  # partial: report its own subset

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(f"{out}.csv", index=False)

    def mape(s):
        s = s.dropna()
        return float(np.mean(np.abs(s))) if len(s) else np.nan

    # Per-situation errors: median over repeat runs of the same (config, GPU).
    sit = ev.groupby(["config", "gpu"], as_index=False).agg(
        policy=("policy", "first"), S_meas=("S_meas", "median"),
        **{f"err_{v}": (f"err_{v}", "median") for v in VARIANTS})

    n_pol, n_gpu = ev.policy.nunique(), ev.gpu.nunique()
    lines = [
        "# Graph-worthiness rule: predicted vs measured CUDA-graph speedup",
        "",
        "`S = T_e / (D + K*eps)` from eager mode only (region wall time, kernel count, summed "
        "kernel time of one profiled eager step). No graph is captured to make a prediction.",
        "",
        f"Evaluation set: {len(ev)} runs, {len(sit)} distinct (config, GPU) situations, "
        f"{n_pol} policies ({', '.join(sorted(ev.policy.unique()))}), {n_gpu} GPUs; measured "
        f"speedups {ev.S_meas.min():.2f}x to {ev.S_meas.max():.1f}x. The SmolVLA baseline "
        f"config ({BASE_CONFIG}), which the global and per-GPU fits use, is excluded on every "
        "GPU.",
        "",
        "## Accuracy by variant",
        "",
        "MAPE of the predicted speedup over runs and over situations (median over repeats), "
        "with a 95% bootstrap CI that resamples configs.",
        "",
    ]
    acc = []
    for v in variants:
        e = ev[f"err_{v}"].dropna()
        g = ev.loc[e.index, "config"]
        lo, hi = bootstrap_ci(e, g)
        sub = "" if len(e) == len(ev) else f" ({ev.loc[e.index].gpu.nunique()} GPUs, {len(e)} runs)"
        acc.append({"variant": VARIANT_LABEL[v] + sub, "eps source": {
            "global": f"{eps_global:+.2f} us, 1 run",
            "per_gpu": "1 SmolVLA run per GPU", "lofo": "other policies' runs, same GPU",
            "logo": "other GPUs only", "micro": "eps_probe.py, no VLA",
            "no_eps": "-", "busy": "-"}[v],
            "MAPE runs": f"{mape(e):.1f}%", "95% CI": f"{lo:.1f} to {hi:.1f}%",
            "MAPE situations": f"{mape(sit[f'err_{v}']):.1f}%",
            "max |err|": f"{np.abs(e).max():.0f}%"})
    lines += [md(pd.DataFrame(acc)), ""]

    node = node_cost()
    eg = [{"GPU": g, "runs": int((ev.gpu == g).sum()),
           "eps per GPU (us)": f"{eps_gpu.get(g, np.nan):+.2f}",
           "eps leave-GPU-out": f"{ev[ev.gpu == g].eps_logo.iloc[0]:+.2f}",
           "eps probe": f"{eps_micro[g]:+.2f}" if g in eps_micro else "pending",
           "C++ node cost (us)": f"{node[g]:.2f}" if g in node else "-",
           **{f"MAPE {VARIANT_LABEL[v]}": f"{mape(ev[ev.gpu == g][f'err_{v}']):.1f}%"
              for v in ("global", "per_gpu", "lofo", "logo")}}
          for g in sorted(ev.gpu.unique())]
    lines += ["## By GPU", "",
              "eps = graph node cost minus the profiler's bias on each traced kernel, so it can "
              "be negative. The C++ node cost is shown for reference.", "",
              md(pd.DataFrame(eg)), ""]

    pol = []
    for p, g in ev.groupby("policy"):
        pol.append({"policy": p, "runs": len(g), "GPUs": g.gpu.nunique(),
                    "S measured": f"{g.S_meas.min():.2f} to {g.S_meas.max():.2f}",
                    "t (us/kernel)": f"{g.t_us.median():.1f}",
                    "c (us/kernel)": f"{g.c_us.median():.1f}",
                    **{VARIANT_LABEL[v]: f"{mape(g[f'err_{v}']):.1f}%" for v in variants}})
    lines += ["## By policy", "",
              "t = D/K, mean traced kernel time; c = T_e/K, eager time per kernel. "
              "S ~= c / (t + eps).", "", md(pd.DataFrame(pol)), ""]

    # --- decisions over thresholds ---------------------------------------------------------
    taus = np.round(np.geomspace(1.2, 15, 25), 3)
    dec = []
    for tau in taus:
        worthy = ev.S_meas >= tau
        hard = (ev.S_meas >= tau / HARD_BAND) & (ev.S_meas <= tau * HARD_BAND)
        rec = {"tau": tau, "worthy": int(worthy.sum()), "hard": int(hard.sum())}
        for v in variants + ["always"]:
            if v == "always":
                pred = pd.Series(True, index=ev.index)
                ok = pd.Series(True, index=ev.index)
            else:
                pred = ev[f"S_{v}"] >= tau
                ok = ev[f"S_{v}"].notna()
            right = (pred == worthy) & ok
            rec[f"acc_{v}"] = right[ok].mean()
            rec[f"hard_{v}"] = right[ok & hard].mean() if (ok & hard).any() else np.nan
        dec.append(rec)
    dec = pd.DataFrame(dec)
    dec.to_csv(f"{out}_thresholds.csv", index=False)
    show = [1.5, 2, 3, 5, 10]
    tab = []
    for t in show:
        worthy = ev.S_meas >= t
        hard = (ev.S_meas >= t / HARD_BAND) & (ev.S_meas <= t * HARD_BAND)
        rec = {"threshold": f"{t:g}x", "worth it": f"{int(worthy.sum())}/{len(ev)}",
               f"hard (within {HARD_BAND:g}x)": int(hard.sum())}
        for v in variants:
            pred = ev[f"S_{v}"] >= t
            ok = ev[f"S_{v}"].notna()
            r_all = ((pred == worthy) & ok)[ok].mean()
            r_h = ((pred == worthy) & ok)[ok & hard].mean() if (ok & hard).any() else np.nan
            rec[VARIANT_LABEL[v]] = f"{100 * r_all:.0f}% / {100 * r_h:.0f}%" \
                if not np.isnan(r_h) else f"{100 * r_all:.0f}% / -"
        rec["always capture"] = f"{100 * worthy.mean():.0f}%"
        tab.append(rec)
    worst = {v: dec[f"acc_{v}"].min() for v in variants}
    lines += ["## Decisions: capture if predicted S >= threshold", "",
              "Accuracy over all runs / over the hard runs (measured speedup within a factor "
              f"{HARD_BAND:g} of the threshold). 'Always capture' is right for the worth-it "
              "runs only.", "", md(pd.DataFrame(tab)), "",
              "Lowest accuracy over 25 thresholds from 1.2x to 15x: " +
              ", ".join(f"{VARIANT_LABEL[v]} {100 * worst[v]:.0f}%" for v in variants) + ".", ""]

    # rank agreement within each GPU (which regions benefit most)
    rk = []
    for g, gg in ev.groupby("gpu"):
        rk.append({"GPU": g, "runs": len(gg), **{
            VARIANT_LABEL[v]: f"{gg[f'S_{v}'].rank().corr(gg.S_meas.rank()):.3f}"
            for v in variants if gg[f"S_{v}"].notna().all()}})
    lines += ["## Ranking", "", "Spearman correlation of predicted and measured speedup over "
              "the runs of each GPU.", "", md(pd.DataFrame(rk)), ""]

    worst_runs = ev.reindex(ev.err_per_gpu.abs().sort_values(ascending=False).index).head(6)
    lines += ["## Largest errors (eps per GPU)", "",
              md(worst_runs[["config", "gpu", "S_meas", "S_per_gpu", "err_per_gpu", "t_us",
                             "c_us"]].round(2)), ""]
    lines += window_section(ev)
    lines += stress_section(stress, df)
    Path(f"{out}.md").write_text("\n".join(lines))
    print("\n".join(lines))
    figure(ev, args.threshold, args.fig, variant="lofo")
    figure_thresholds(dec, variants, f"{args.fig}_threshold")


def window_section(ev: pd.DataFrame) -> list[str]:
    """Long profiles: a 20-flow-step SmolVLA profile (about 25k kernels per step, three steps
    per session) inflates the traced kernel durations on the 2080 Ti and A100 hosts. Because the
    loop is periodic, K and D per iteration can come from a short profile instead: here the
    5-step run of the same GPU, scaled to 20 iterations."""
    rows = []
    for g, gg in ev.groupby("gpu"):
        long = gg[gg.config == "smolvla_cam3_steps20_chunk50"]
        short = gg[gg.config == "smolvla_cam3_steps5_chunk50"]
        if not len(long) or not len(short):
            continue
        lr, sr = long.iloc[0], short.iloc[0]
        k = sr.K * 4
        d = sr.D_ms * 4
        s_short = lr.T_e_ms / (d + k * lr.eps_per_gpu / 1e3)
        rows.append({"GPU": g, "t traced, 5 steps (us)": f"{sr.t_us:.2f}",
                     "t traced, 20 steps (us)": f"{lr.t_us:.2f}", "S measured": f"{lr.S_meas:.2f}",
                     "20-step profile": f"{lr.err_per_gpu:+.1f}%",
                     "5-step profile x 4": f"{100 * (s_short - lr.S_meas) / lr.S_meas:+.1f}%"})
    if not rows:
        return []
    return ["## Profile a short window", "",
            "SmolVLA with 20 flow steps, eps per GPU. Long profiling sessions inflate the "
            "traced kernel durations on the 2080 Ti and A100 hosts (CUPTI); K and D of a "
            "periodic loop can come from a short profile scaled to the full loop.", "",
            md(pd.DataFrame(rows)), ""]


def stress_section(stress: pd.DataFrame, df: pd.DataFrame) -> list[str]:
    """Host-contention runs (scripts/host_contention.sh, 2080 Ti, one core): 'alone' pinned to
    one idle core, 'smt_alu' / 'smt_mem' a hog on the SMT sibling, 'share' time-slicing the
    core. Two ways to feed the rule: everything from the contended run, or T_e from the
    contended run and K, D from a profile taken on the quiet host (the alone run)."""
    if not len(stress):
        return []
    rows = []
    for (cfg, cond), g in stress.groupby(["config", "host_cond"]):
        r = g.iloc[0]
        q = stress[(stress.config == cfg) & (stress.host_cond == "alone")]
        if not len(q):
            continue
        q = q.iloc[0]
        s_q = r.T_e_ms / (q.D_ms + q.K * r.eps_per_gpu / 1e3)
        rows.append({"config": cfg, "host": cond, "eager slowdown": f"{r.T_e_ms / q.T_e_ms:.2f}x",
                     "t traced (us)": f"{r.t_us:.2f}", "S measured": f"{r.S_meas:.2f}",
                     "profile under contention": f"{r.err_per_gpu:+.1f}%",
                     "profile on quiet host": f"{100 * (s_q - r.S_meas) / r.S_meas:+.1f}%"})
    if not rows:
        return []
    return ["## Stress test: host contention (2080 Ti, one core)", "",
            "Not in the evaluation set above. Contention inflates the CUPTI-traced kernel "
            "durations (t), so D is biased when the profile is taken on a busy host; T_e "
            "should be measured under deployment conditions, D and K on a quiet host. eps per "
            "GPU as above.", "", md(pd.DataFrame(rows)), ""]


def md(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    out += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(out)


SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#8a63d2", "#0f9fb5"]
MARKERS = ["o", "D", "v", "s", "^", "P", "X"]
POLICY_ORDER = ["SmolVLA", "SmolVLA compiled", "pi0", "pi0.5", "ACT", "Diffusion Policy",
                "OpenVLA", "pi0-FAST"]


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2)


def figure(ev: pd.DataFrame, threshold: float, path: str, variant: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, NullFormatter

    # Seven categorical slots; the two decoders share one (they are one kind: autoregressive
    # decode), told apart by marker.
    slot = {"SmolVLA": 0, "SmolVLA compiled": 5, "pi0": 1, "pi0.5": 2, "ACT": 3,
            "Diffusion Policy": 4, "OpenVLA": 6, "pi0-FAST": 6}
    mk = {"SmolVLA": "o", "SmolVLA compiled": "P", "pi0": "D", "pi0.5": "v", "ACT": "s",
          "Diffusion Policy": "^", "OpenVLA": "X", "pi0-FAST": "*"}
    col = f"S_{variant}"
    fig, ax = plt.subplots(figsize=(5.4, 4.6), facecolor=SURFACE)
    _style(ax)
    lo, hi = 0.8, ev[["S_meas", col]].max().max() * 1.25
    ax.plot([lo, hi], [lo, hi], color=INK2, linewidth=0.8, zorder=1)
    ax.axhline(threshold, color=GRID, linewidth=1, zorder=0)
    ax.axvline(threshold, color=GRID, linewidth=1, zorder=0)
    for pol in POLICY_ORDER:
        g = ev[ev.policy == pol]
        if not len(g):
            continue
        ax.scatter(g.S_meas, g[col], s=70 if mk[pol] == "*" else 32, marker=mk[pol],
                   color=PALETTE[slot[pol]], edgecolor=SURFACE, linewidth=0.8,
                   label=f"{pol} ({len(g)})", zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ticks = [t for t in (1, 2, 5, 10, 20) if lo <= t <= hi]
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(FixedLocator(ticks))
        axis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}x"))
        axis.set_minor_formatter(NullFormatter())
    m = float(np.mean(np.abs(ev[f"err_{variant}"])))
    ax.set_xlabel("measured CUDA-graph speedup", color=INK2)
    ax.set_ylabel("predicted from eager mode only", color=INK2)
    ax.set_title(f"{len(ev)} held-out runs, {ev.gpu.nunique()} GPUs, eps without the "
                 f"predicted policy: MAPE {m:.1f}%", loc="left", color=INK, fontsize=9)
    ax.text(threshold * 1.03, lo * 1.05, f"threshold {threshold:g}x", color=INK2, fontsize=7.5)
    ax.legend(frameon=False, fontsize=7.5, labelcolor=INK, loc="upper left")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{path}.{ext}", dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def figure_thresholds(dec: pd.DataFrame, variants: list[str], path: str) -> None:
    """Wrong decisions vs threshold. 'Always capture' (wrong for 32 to 93% of the runs over this
    range) is left out so the differences between the predictors stay visible."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    show = [v for v in ("lofo", "no_eps", "busy") if v in variants]
    label = {**VARIANT_LABEL, "lofo": "rule, eps without the predicted policy"}
    colors = {"lofo": PALETTE[0], "no_eps": PALETTE[1], "busy": PALETTE[3]}
    fig, ax = plt.subplots(figsize=(5.4, 3.0), facecolor=SURFACE)
    _style(ax)
    for v in show:
        ax.plot(dec.tau, 100 * (1 - dec[f"acc_{v}"]), color=colors[v], linewidth=2,
                label=label[v])
    ax.set_xscale("log")
    ax.set_xticks([1.5, 2, 3, 5, 10])
    ax.set_xticklabels(["1.5x", "2x", "3x", "5x", "10x"])
    ax.minorticks_off()
    ax.set_ylim(0, 20)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_xlabel("decision threshold (capture if predicted speedup is at least this)",
                  color=INK2)
    ax.set_ylabel("wrong decisions (% of runs)", color=INK2)
    ax.legend(frameon=False, fontsize=7.5, labelcolor=INK, loc="upper left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{path}.{ext}", dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
