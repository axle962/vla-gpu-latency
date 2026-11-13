"""Turn an Nsight Compute raw CSV (`ncu --csv --page raw`) into a per-kernel feature table.

Output columns (one row per profiled kernel, in launch order):
  seq, kernel, grid, block, regs, smem_bytes, waves, theo_occ_pct, achieved_occ_pct,
  dur_us, cycles, dram_read_B, dram_write_B, l2_B, l2_hit_pct, tensor_inst, inst
  tensor_inst = HMMA instructions (each 16x8x8 on Turing; FLOPs depend on the shape).
The metric set needs 3 counter passes (~0.23 s per kernel on the 2080 Ti); throughput-percent
and L1 metrics were dropped because they add passes. FLOPs are not collected: the SASS-instrumented FLOP counters (smsp__sass_thread_inst_*) need
extra passes and made a 1,300-kernel run take tens of minutes. FLOPs follow exactly from the
shapes in bench/chains/*.csv (GEMM 2*b*m*n*k, attention 4*b*m*n*k).

Usage
  python profile/ncu_features.py <ncu_raw.csv> -o <features.csv.gz> [--chain bench/chains/x.csv]
With --chain the kernels are also matched to the replay rows (same launch order) and the
row index, category and op are added; this works for chain_replay runs with --gemm=own.
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import pandas as pd

M = {
    "gpu__time_duration.sum": "dur_ns",
    "sm__cycles_elapsed.avg": "cycles",
    "dram__bytes_read.sum": "dram_read_B",
    "dram__bytes_write.sum": "dram_write_B",
    "lts__t_bytes.sum": "l2_B",
    "lts__t_sector_hit_rate.pct": "l2_hit_pct",
    "sm__warps_active.avg.pct_of_peak_sustained_active": "achieved_occ_pct",
    "sm__maximum_warps_per_active_cycle_pct": "theo_occ_pct",
    "sm__inst_executed_pipe_tensor_op_hmma.sum": "tensor_inst",
    "smsp__inst_executed.sum": "inst",
    "launch__grid_size": "grid_size",
    "launch__block_size": "block_size",
    "launch__registers_per_thread": "regs",
    "launch__shared_mem_per_block_static": "smem_static",
    "launch__shared_mem_per_block_dynamic": "smem_dynamic",
    "launch__waves_per_multiprocessor": "waves",
}
METRICS = ",".join(M)  # scripts/run_ncu.sh reads this list


def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(",", "", regex=False), errors="coerce")


def load(path: Path) -> pd.DataFrame:
    text = path.read_text(errors="replace").splitlines()
    start = next(i for i, l in enumerate(text) if l.startswith('"ID"'))
    df = pd.read_csv(io.StringIO("\n".join(text[start:])), low_memory=False)
    units = df.iloc[0]
    df = df.iloc[1:].reset_index(drop=True)  # second row holds the units
    out = pd.DataFrame({"seq": range(len(df)), "kernel": df["Kernel Name"]})
    out["grid"] = df.get("Grid Size")
    out["block"] = df.get("Block Size")
    for k, v in M.items():
        if k not in df:
            continue
        col = to_num(df[k])
        u = str(units.get(k, ""))
        if k == "gpu__time_duration.sum":  # normalize to ns
            col = col * {"usecond": 1e3, "msecond": 1e6, "second": 1e9}.get(u, 1.0)
        if v.endswith("_B"):  # normalize bytes
            col = col * {"Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9, "byte": 1.0}.get(u, 1.0)
        out[v] = col
    out["dur_us"] = out.pop("dur_ns") / 1e3
    out["smem_bytes"] = out.get("smem_static", 0) + out.get("smem_dynamic", 0)
    keep = ["seq", "kernel", "grid", "block", "regs", "smem_bytes", "waves", "theo_occ_pct",
            "achieved_occ_pct", "dur_us", "cycles", "dram_read_B", "dram_write_B", "l2_B",
            "l2_hit_pct", "tensor_inst", "inst"]
    return out[[c for c in keep if c in out]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("raw", type=Path)
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--chain", type=Path, help="replay list to align with (gemm=own runs)")
    a = ap.parse_args()
    f = load(a.raw)
    if a.chain is not None:
        ch = pd.read_csv(a.chain)
        ch = ch[~((ch["cat"] == "memcpy") | (ch["cat"] == "memset"))].reset_index(drop=True)
        # Our GEMM launcher may add a split-K reduce kernel: align by skipping those.
        main_k = f[~f.kernel.str.contains("splitk_reduce")].reset_index()
        if len(main_k) == len(ch):
            f.loc[main_k["index"], "chain_idx"] = ch["idx"].values
            f.loc[main_k["index"], "cat"] = ch["cat"].values
            f.loc[main_k["index"], "op"] = ch["op"].values
            f.loc[main_k["index"], "traced_us"] = ch["traced_us"].values
            f[["chain_idx", "cat", "op", "traced_us"]] = f[
                ["chain_idx", "cat", "op", "traced_us"]].ffill()  # reduce kernels inherit
        else:
            print(f"warning: {len(main_k)} kernels vs {len(ch)} chain rows; not aligned",
                  file=sys.stderr)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    f.to_csv(a.out, index=False)
    tot = f.dur_us.sum()
    print(f"{a.raw.name}: {len(f)} kernels, {tot:.0f} us under ncu, DRAM "
          f"{(f.dram_read_B.sum() + f.dram_write_B.sum()) / 1e6:.1f} MB, L2 "
          f"{f.l2_B.sum() / 1e6:.1f} MB, median l2 hit {f.l2_hit_pct.median():.1f}%",
          file=sys.stderr)


if __name__ == "__main__":
    main()
