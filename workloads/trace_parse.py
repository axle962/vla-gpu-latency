"""Parse a torch.profiler chrome trace (trace.json.gz) into per-kernel and per-op tables.

Needs only pandas, so it also runs on CPU nodes. The login node has ~2 GB of RAM, too little
for a 100+ MB trace: re-parse on a compute node, e.g.
  sbatch -p n2c48m24 scripts/cluster/reparse.sbatch traces/smolvla_*/*/

Step boundaries come from the CPU-side record_function("vla_step_<i>") annotations
(cat == "user_annotation"). The profiler also emits a GPU-side copy of each annotation
(cat == "gpu_user_annotation"); counting both doubled the number of steps in early traces.
"""

from __future__ import annotations

import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

GPU_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}


def _load(path: Path) -> list[dict]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        return json.load(f)["traceEvents"]


def _innermost_ops(events: list[dict], launches: list[dict]) -> dict[int, dict]:
    """Map each launch's correlation id to the innermost cpu_op enclosing it on the same
    thread. Uses only timestamps, so it does not depend on how a torch version fills the
    "External id" field (torch 2.2 links kernels to the wrong op through it)."""
    by_tid: dict = defaultdict(list)
    for e in events:
        if e.get("cat") == "cpu_op" and e.get("ph") == "X":
            by_tid[e.get("tid")].append(e)
    out: dict[int, dict] = {}
    lby_tid: dict = defaultdict(list)
    for e in launches:
        lby_tid[e.get("tid")].append(e)
    for tid, ls in lby_tid.items():
        ops = sorted(by_tid.get(tid, []), key=lambda e: (e["ts"], -e["dur"]))
        stack: list[dict] = []
        i = 0
        for la in ls:  # launches are sorted by ts
            t = la["ts"]
            while i < len(ops) and ops[i]["ts"] <= t:
                while stack and stack[-1]["ts"] + stack[-1]["dur"] < ops[i]["ts"]:
                    stack.pop()
                stack.append(ops[i])
                i += 1
            while stack and stack[-1]["ts"] + stack[-1]["dur"] < t:
                stack.pop()
            if stack:
                out[la["args"]["correlation"]] = stack[-1]
    return out


def parse_trace(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    events = _load(path)
    steps = sorted(
        (
            e
            for e in events
            if e.get("ph") == "X"
            and e.get("cat") == "user_annotation"
            and str(e.get("name", "")).startswith("vla_step_")
        ),
        key=lambda e: e["ts"],
    )
    # Launch API calls (runtime and driver) carry the correlation id of the GPU work they
    # enqueue. GPU work lags the CPU, so steps and ops are assigned through these calls.
    launches = sorted(
        (
            e
            for e in events
            if e.get("cat") in ("cuda_runtime", "cuda_driver")
            and e.get("ph") == "X"
            and e.get("args", {}).get("correlation") is not None
        ),
        key=lambda e: e["ts"],
    )
    corr_op = _innermost_ops(events, launches)
    rows = []
    for e in events:
        if e.get("cat") not in GPU_CATS or e.get("ph") != "X":
            continue
        a = e.get("args", {})
        op = corr_op.get(a.get("correlation"), {})
        rows.append(
            {
                "cat": e["cat"],
                "name": e["name"],
                "ts_us": float(e["ts"]),
                "dur_us": float(e["dur"]),
                "correlation": a.get("correlation"),
                "stream": a.get("stream"),
                "grid": "x".join(map(str, a.get("grid", []))),
                "block": "x".join(map(str, a.get("block", []))),
                "regs_per_thread": a.get("registers per thread"),
                "smem_bytes": a.get("shared memory"),
                "blocks_per_sm": a.get("blocks per SM"),
                "warps_per_sm": a.get("warps per SM"),
                "est_occupancy_pct": a.get("est. achieved occupancy %"),
                "op": op.get("name", ""),
                "op_input_dims": json.dumps(op.get("args", {}).get("Input Dims", [])),
                "op_input_types": json.dumps(op.get("args", {}).get("Input type", [])),
            }
        )
    k = pd.DataFrame(rows).sort_values("ts_us").reset_index(drop=True)
    corr_step: dict[int, int] = {}
    for i, s in enumerate(steps):
        t0, t1 = s["ts"], s["ts"] + s["dur"]
        for e in launches:
            if t0 <= e["ts"] <= t1:
                corr_step[e["args"]["correlation"]] = i
    k["step"] = k.correlation.map(corr_step)
    k = k[k.step.notna()].copy()
    k["step"] = k.step.astype(int)
    k["seq"] = k.groupby("step").cumcount()
    k["gap_before_us"] = k.groupby("step").ts_us.diff() - k.groupby("step").dur_us.shift()
    cpu_calls = defaultdict(int)
    for e in launches:
        s = corr_step.get(e["args"]["correlation"])
        if s is not None:
            cpu_calls[(s, e["name"])] += 1
    launch_df = pd.DataFrame(
        [{"step": s, "api": n, "count": c} for (s, n), c in sorted(cpu_calls.items())]
    )
    return k, launch_df


def summarize_trace(out: Path) -> dict:
    """Parse out/trace.json.gz, write kernels.csv.gz, ops.csv and launch_api_calls.csv, and
    return the trace-derived summary fields for meta.json."""
    out = Path(out)
    kernels, launches = parse_trace(out / "trace.json.gz")
    n_steps = int(kernels.step.nunique())
    kernels.to_csv(out / "kernels.csv.gz", index=False)
    launches.to_csv(out / "launch_api_calls.csv", index=False)
    kk = kernels[kernels.cat == "kernel"]
    ops = (
        kk.groupby(["op", "op_input_dims"])
        .agg(kernels=("name", "size"), gpu_us=("dur_us", "sum"))
        .reset_index()
        .sort_values("gpu_us", ascending=False)
    )
    ops["kernels"] = ops["kernels"] / n_steps
    ops["gpu_us"] = ops["gpu_us"] / n_steps
    ops.to_csv(out / "ops.csv", index=False)
    per_step = kk.groupby("step").agg(n=("name", "size"), busy_us=("dur_us", "sum"))
    calls = launches.groupby("api")["count"].sum() / n_steps
    first = kernels.groupby("step").ts_us.min()
    last = (kernels.ts_us + kernels.dur_us).groupby(kernels.step).max()
    return {
        "prof_steps_parsed": n_steps,
        "kernels_per_step": float(per_step.n.median()),
        "memcpy_per_step": float((kernels.cat == "gpu_memcpy").sum() / n_steps),
        "gpu_busy_ms_per_step": float(per_step.busy_us.median()) / 1e3,
        "gpu_span_ms_per_step": float((last - first).median()) / 1e3,
        "kernel_dur_us_median": float(kk.dur_us.median()),
        "kernel_dur_us_p90": float(kk.dur_us.quantile(0.9)),
        "frac_kernels_under_5us": float((kk.dur_us < 5).mean()),
        "kernel_gap_us_median": float(kk.gap_before_us.median()),
        "cpu_api_calls_per_step": {k: float(v) for k, v in calls.items()},
    }


def reparse(out: Path) -> None:
    """Rebuild the trace tables and refresh the trace-derived fields of meta.json."""
    out = Path(out)
    meta_p = out / "meta.json"
    meta = json.loads(meta_p.read_text())
    meta.update(summarize_trace(out))
    meta["gpu_busy_frac"] = meta["gpu_busy_ms_per_step"] / meta["eager_ms_median"]
    meta_p.write_text(json.dumps(meta, indent=2))
    print(out, json.dumps({k: meta[k] for k in ("kernels_per_step", "gpu_busy_ms_per_step",
                                               "gpu_busy_frac", "eager_ms_median")}))


if __name__ == "__main__":
    for d in sys.argv[1:]:
        reparse(Path(d))
