"""Graph-worthiness rule: predict the CUDA-graph speedup of a region before capturing it.

Capturing a region as a CUDA graph takes engineering (static buffers, no host syncs, constant
uploads moved out, buffers kept alive), so the question a practitioner asks first is whether
it would pay off. This rule answers it from eager mode only:

  T_e  eager wall time of the region                       (time it, no profiler)
  K    kernel launches in the region                        (one profiled step)
  D    summed GPU duration of those kernels in eager mode   (same profiled step)

  T_graph ~= D + K * eps        S = T_e / T_graph          worth it if S >= threshold

eps is the net per-kernel cost of a kernel in a graph relative to its traced eager duration:
the graph node cost (about 0.3 to 0.8 us) minus the profiler's bias on each traced duration
(about 1 us, CLAUDE.md "profiler bias"). It is fitted on ONE run (the 2080 Ti SmolVLA baseline)
and held fixed for every other workload and GPU. With D / K = mean kernel time t and
T_e / K = eager time per kernel c (mostly host dispatch), S ~= c / (t + eps): graphs pay off
when the host spends much longer per kernel than the GPU does.

Data: every trace under traces/ with an eager per-kernel table and a measured graph time of
the same region (SmolVLA and its batch/RoboTwin variants, pi0, pi0.5, ACT, Diffusion Policy
default and autotuned; torch.compile runs excluded, their graph runs different kernels). The
captured region is the tail of each eager step (the loop; the whole step for ACT); K comes
from the eager trace alone: the loop is the trailing run of num_steps identical blocks of
kernel names (one per flow or DDPM step), found by checking candidate block lengths. (The graph
replay's own kernel count is a few percent higher, for example 11,900 vs 11,480 for SmolVLA,
so taking that many trailing eager kernels would reach into the prefix.)

Baselines: (a) S = T_e / D, no per-kernel term; (b) S = 1 / gpu_busy_frac of the whole step.

  python model/graph_worthiness.py [--threshold 1.5] [--out results/graph_worthiness]
writes <out>.csv, <out>.md and paper/figures/graph_worthiness.{pdf,png}.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

FIT_RUN = "traces/smolvla_cam3_steps10_chunk50/NVIDIA_GeForce_RTX_2080_Ti_cuda2_*"
GPU_SHORT = {"A100": "A100", "2080 Ti": "2080 Ti", "TITAN V": "TITAN V", "4070": "RTX 4070"}


def short_gpu(name: str) -> str:
    for k, v in GPU_SHORT.items():
        if k in name:
            return v
    return name


def load_run(meta_path: str) -> dict | None:
    d = Path(meta_path).parent
    m = json.loads(Path(meta_path).read_text())
    family = d.parent.name.split("_")[0]
    if family in ("smolvlacompile", "openvla7b", "pi0fast") or not m.get("graph_ok", False):
        return None
    if "graph_denoise_ms_median" in m and "eager_denoise_ms_median" in m:
        t_e, t_g, region = m["eager_denoise_ms_median"], m["graph_denoise_ms_median"], "loop"
    elif "graph_ms_median" in m:
        t_e, t_g, region = m["eager_ms_median"], m["graph_ms_median"], "step"
    else:
        return None
    kc = d / "kernels.csv.gz"
    if not kc.exists():
        return None
    k = pd.read_csv(kc, usecols=["cat", "name", "dur_us", "step", "seq"])
    k = k[k.cat == "kernel"].sort_values(["step", "seq"])
    reps = int(m.get("num_steps") or 0) if region == "loop" else 0
    ks, ds = [], []
    for _, s in k.groupby("step"):
        # Compare kernel names without template arguments: some differ between iterations of
        # the same loop (Diffusion Policy), the kernel sequence does not.
        base_names = s.name.str.split("<").str[0].str.split("(").str[0].tolist()
        n = loop_kernels(base_names, reps) if region == "loop" else len(s)
        if not n:
            return None
        ks.append(n)
        ds.append(s.dur_us.iloc[-n:].sum())
    n = int(np.median(ks))
    d_region = pd.Series(ds)
    return {
        "run": str(d),
        "workload": family,
        "config": d.parent.name,
        "gpu": short_gpu(m["gpu"]),
        "region": region,
        "K": n,
        "T_e_ms": t_e,
        "D_ms": float(d_region.median()) / 1e3,
        "T_g_ms": t_g,
        "busy_frac": m.get("gpu_busy_frac", np.nan),
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=1.5, help="speedup that makes it worth it")
    ap.add_argument("--out", default="results/graph_worthiness")
    ap.add_argument("--fig", default="paper/figures/graph_worthiness")
    args = ap.parse_args()

    rows = [r for f in sorted(glob.glob("traces/*/*/meta.json")) if (r := load_run(f))]
    df = pd.DataFrame(rows).dropna(subset=["D_ms"])
    df["S_meas"] = df.T_e_ms / df.T_g_ms

    fit = df[df.run.apply(lambda r: Path(r).match(FIT_RUN))].iloc[0]
    eps_us = (fit.T_g_ms - fit.D_ms) / fit.K * 1e3  # one run, then held fixed
    df["T_g_pred_ms"] = df.D_ms + df.K * eps_us / 1e3
    df["S_pred"] = df.T_e_ms / df.T_g_pred_ms
    # Variant: eps calibrated once per machine, on that GPU's SmolVLA baseline run.
    base = df[df.config == "smolvla_cam3_steps10_chunk50"].groupby("gpu").first()
    eps_gpu = ((base.T_g_ms - base.D_ms) / base.K * 1e3).to_dict()
    df["eps_gpu_us"] = df.gpu.map(eps_gpu)
    df["S_pred_gpu"] = df.T_e_ms / (df.D_ms + df.K * df.eps_gpu_us / 1e3)
    df["is_fit_gpu"] = df.run.isin(base.run)
    df["S_noeps"] = df.T_e_ms / df.D_ms
    df["S_busy"] = 1.0 / df.busy_frac
    for c in ("S_pred", "S_pred_gpu", "S_noeps", "S_busy"):
        df[f"err_{c}"] = 100 * (df[c] - df.S_meas) / df.S_meas
    df["c_us"] = df.T_e_ms / df.K * 1e3
    df["t_us"] = df.D_ms / df.K * 1e3
    held = df[df.run != fit.run]
    worthy = held.S_meas >= args.threshold
    pred_worthy = held.S_pred >= args.threshold

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(f"{out}.csv", index=False)

    def mape(s):
        return float(np.mean(np.abs(s)))

    lines = [
        "# Graph-worthiness rule: predicted vs measured CUDA-graph speedup",
        "",
        f"`S = T_e / (D + K*eps)`, eps = {eps_us:+.3f} us per kernel, fitted on one run "
        f"(`{Path(fit.run).parent.name}`, {fit.gpu}) and held fixed. {len(held)} held-out runs, "
        f"{held.workload.nunique()} workload families, {held.gpu.nunique()} GPUs. Inputs come "
        "from eager mode only (region wall time, kernel count, summed kernel time of one "
        "profiled step); no graph is captured.",
        "",
        "## Accuracy (held-out runs)",
        "",
    ]
    acc = []
    for (w,), g in held.groupby(["workload"]):
        acc.append({
            "workload": w, "runs": len(g),
            "S measured (range)": f"{g.S_meas.min():.2f} to {g.S_meas.max():.2f}",
            "rule MAPE": f"{mape(g.err_S_pred):.1f}%",
            "per-machine eps": f"{mape(g[~g.is_fit_gpu].err_S_pred_gpu):.1f}%",
            "no eps (T_e/D)": f"{mape(g.err_S_noeps):.1f}%",
            "1/busy_frac": f"{mape(g.err_S_busy):.1f}%",
        })
    acc.append({
        "workload": "**all**", "runs": len(held),
        "S measured (range)": f"{held.S_meas.min():.2f} to {held.S_meas.max():.2f}",
        "rule MAPE": f"**{mape(held.err_S_pred):.1f}%**",
        "per-machine eps": f"**{mape(held[~held.is_fit_gpu].err_S_pred_gpu):.1f}%**",
        "no eps (T_e/D)": f"{mape(held.err_S_noeps):.1f}%",
        "1/busy_frac": f"{mape(held.err_S_busy):.1f}%",
    })
    lines += [md(pd.DataFrame(acc)), ""]
    per_gpu = [{"GPU": gname, "runs": len(g), "rule MAPE": f"{mape(g.err_S_pred):.1f}%",
                "median signed error": f"{np.median(g.err_S_pred):+.1f}%",
                "eps for this GPU (us)": f"{eps_gpu.get(gname, np.nan):+.2f}",
                "per-machine eps MAPE": f"{mape(g[~g.is_fit_gpu].err_S_pred_gpu):.1f}%"}
               for gname, g in held.groupby("gpu")]
    lines += ["By GPU (per-machine eps: fitted on that GPU's SmolVLA baseline, evaluated on "
              "its other runs):", "", md(pd.DataFrame(per_gpu)), ""]

    tp = int((worthy & pred_worthy).sum())
    tn = int((~worthy & ~pred_worthy).sum())
    fp = int((~worthy & pred_worthy).sum())
    fn = int((worthy & ~pred_worthy).sum())
    lines += [
        f"## Decision: capture if predicted S >= {args.threshold}",
        "",
        f"Correct for {tp + tn} of {len(held)} held-out runs ({100 * (tp + tn) / len(held):.0f}%): "
        f"{tp} worth it and predicted so, {tn} not worth it and predicted so, "
        f"{fp} predicted worth it but not, {fn} worth it but predicted not.",
        "",
    ]
    wrong = held[worthy != pred_worthy]
    if len(wrong):
        lines += ["Misclassified:", "", md(wrong[["config", "gpu", "S_meas", "S_pred"]].round(2)),
                  ""]

    lines += [
        "## Reading",
        "",
        "With c = T_e/K (eager time per kernel, mostly host dispatch) and t = D/K (mean kernel "
        "time), S ~= c / (t + eps). Per workload (median over GPUs):",
        "",
    ]
    rd = held.groupby("workload").agg(c_us=("c_us", "median"), t_us=("t_us", "median"),
                                      S_meas=("S_meas", "median")).round(2).reset_index()
    lines += [md(rd), ""]
    Path(f"{out}.md").write_text("\n".join(lines))
    print("\n".join(lines))
    figure(held, eps_us, args.threshold, args.fig)


def md(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    out += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(out)


POLICY = {"smolvla": "SmolVLA", "smolvlabatch": "SmolVLA", "smolvlarobotwin": "SmolVLA",
          "pi0": "pi0", "pi05": "pi0.5", "act": "ACT", "diffusion": "Diffusion Policy",
          "diffusionbench": "Diffusion Policy"}


def figure(held: pd.DataFrame, eps_us: float, threshold: float, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, NullFormatter

    # Reference categorical palette, slots 1 to 5 in fixed order (validated for CVD and normal
    # vision); one slot per policy, its variants (batch, RoboTwin, autotuned) share it.
    palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
    surface, ink, ink2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    h = held[~held.is_fit_gpu].copy()
    h["policy"] = h.workload.map(POLICY)
    order = ["SmolVLA", "pi0", "pi0.5", "ACT", "Diffusion Policy"]
    markers = ["o", "D", "v", "s", "^"]
    fig, ax = plt.subplots(figsize=(5.2, 4.4), facecolor=surface)
    ax.set_facecolor(surface)
    lo, hi = 0.8, h[["S_meas", "S_pred_gpu"]].max().max() * 1.25
    ax.plot([lo, hi], [lo, hi], color=ink2, linewidth=0.8, zorder=1)
    ax.axhline(threshold, color=grid, linewidth=1, zorder=0)
    ax.axvline(threshold, color=grid, linewidth=1, zorder=0)
    for i, pol in enumerate(order):
        g = h[h.policy == pol]
        ax.scatter(g.S_meas, g.S_pred_gpu, s=34, marker=markers[i], color=palette[i],
                   edgecolor=surface, linewidth=0.8, label=f"{pol} ({len(g)})", zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ticks = [t for t in (1, 2, 5, 10, 20) if lo <= t <= hi]
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(FixedLocator(ticks))
        axis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}x"))
        axis.set_minor_formatter(NullFormatter())
    mape = float(np.mean(np.abs(h.err_S_pred_gpu)))
    ax.set_xlabel("measured CUDA-graph speedup", color=ink2)
    ax.set_ylabel("predicted from eager mode only", color=ink2)
    ax.set_title(f"Graph-worthiness rule: {len(h)} held-out runs, 4 GPUs, MAPE {mape:.1f}%",
                 loc="left", color=ink, fontsize=9.5)
    ax.text(threshold * 1.03, lo * 1.05, f"threshold {threshold:g}x", color=ink2, fontsize=7.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(grid)
    ax.tick_params(colors=ink2)
    ax.legend(frameon=False, fontsize=8, labelcolor=ink, loc="upper left")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{path}.{ext}", dpi=200, facecolor=surface, bbox_inches="tight")


if __name__ == "__main__":
    main()
