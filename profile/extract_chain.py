"""Turn a traced VLA step into a kernel list that bench/chains/chain_replay.cu can replay.

One row per launch (op call) in trace order. Kernels that share a correlation id (one CUDA
launch call) are merged; extra library kernels (e.g. cuBLAS split-K reduce) are folded into the
row of the op that launched them, so `traced_us` is the GPU time of the whole op.

Columns
  idx,cat,op,b,m,n,k,numel,n_in,in_bytes,out_bytes,bias,weight,kind,traced_us,traced_kernels
  cat: gemm | attention | softmax | reduce | norm | scan | elementwise | memcpy | memset
  gemm:       C[b,m,n] = A[b,m,k] B[b,k,n] (+bias), in_bytes = element size; weight=1 means B is
              a parameter (distinct buffer per row, so weights stream from DRAM as in the model)
  attention:  b = batch*heads, m = query len, n = key len, k = head dim (FP16)
  softmax/reduce/norm: m = rows, n = row length, in_bytes = element size
  elementwise: numel outputs, n_in tensor inputs, in_bytes/out_bytes = element sizes
  memcpy:     numel bytes, kind HtoD | DtoH | DtoD; memset: numel bytes
Rows whose shape could not be read fall back to elementwise with numel = launched threads
(`op` keeps the original name so the fallback fraction can be reported).

Regions
  full  the whole traced step (prefix + denoising loop)
  flow  one flow-matching step of the denoising loop, found as the repeating period at the
        end of the step (SmolVLA: 1,148 kernels)

Usage
  python profile/extract_chain.py traces/<cfg>/<run> --region flow -o bench/chains/<name>.csv
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from math import prod
from pathlib import Path

import pandas as pd

ELEM = {"c10::Half": 2, "c10::BFloat16": 2, "float": 4, "double": 8, "long int": 8,
        "int": 4, "bool": 1, "unsigned char": 1, "signed char": 1, "short int": 2}
GEMM_OPS = {"aten::mm", "aten::addmm", "aten::bmm", "aten::baddbmm"}
ATTN_OPS = {"aten::_efficient_attention_forward", "aten::_flash_attention_forward",
            "aten::_scaled_dot_product_efficient_attention",
            "aten::_scaled_dot_product_flash_attention"}
REDUCE_OPS = {"aten::mean", "aten::sum", "aten::min", "aten::max", "aten::amax", "aten::amin",
              "aten::var", "aten::norm", "aten::linalg_vector_norm", "aten::argmax", "aten::any",
              "aten::all", "aten::prod", "aten::std"}
COLS = ["idx", "cat", "op", "b", "m", "n", "k", "numel", "n_in", "in_bytes", "out_bytes", "bias",
        "weight", "kind", "traced_us", "traced_kernels"]


def parse(s):
    try:
        return ast.literal_eval(s) if isinstance(s, str) else []
    except (ValueError, SyntaxError):
        return []


def is_shape(x):
    return isinstance(x, list) and all(isinstance(v, int) for v in x)


def tensors(dims, types):
    """(shape, elem_bytes) of each tensor input; TensorLists are flattened."""
    out = []
    for i, d in enumerate(dims):
        t = types[i] if i < len(types) else ""
        if is_shape(d) and t in ELEM:
            out.append((d, ELEM[t]))
        elif isinstance(d, list) and d and all(is_shape(x) for x in d):  # TensorList
            out += [(x, 2) for x in d]  # element type is not recorded for lists
    return out


def threads(r):
    try:
        g = [int(v) for v in str(r.grid).split("x")]
        bl = [int(v) for v in str(r.block).split("x")]
        return max(1, prod(g) * prod(bl))
    except ValueError:
        return 1


def row_for(r, dims, types):
    """Classify one op launch. Returns a dict with the replay parameters."""
    op = r["op"] if isinstance(r["op"], str) else ""
    ts = tensors(dims, types)
    e = dict(cat="elementwise", op=op or "unlabeled", b=1, m=0, n=0, k=0, numel=0, n_in=1,
             in_bytes=4, out_bytes=4, bias=0, weight=0, kind="")
    if r["cat"] == "gpu_memcpy":
        kind = "HtoD" if "HtoD" in r["name"] else "DtoH" if "DtoH" in r["name"] else "DtoD"
        nbytes = max((prod(s) * eb for s, eb in ts), default=0)
        return {**e, "cat": "memcpy", "kind": kind, "numel": max(nbytes, 4)}
    if r["cat"] == "gpu_memset":
        nbytes = max((prod(s) * eb for s, eb in ts), default=0)
        return {**e, "cat": "memset", "numel": max(nbytes, 4)}
    if op in GEMM_OPS and len(ts) >= 2:
        sh = [s for s, _ in ts]
        eb = ts[-1][1]
        if op in ("aten::mm",) and len(sh[0]) == 2:
            (m, k), (_, n) = sh[0], sh[1]
            return {**e, "cat": "gemm", "m": m, "n": n, "k": k, "in_bytes": eb, "weight": 1}
        if op == "aten::addmm" and len(sh) >= 3:
            (m, k), (_, n) = sh[1], sh[2]
            return {**e, "cat": "gemm", "m": m, "n": n, "k": k, "in_bytes": eb, "bias": 1,
                    "weight": 1}
        if op in ("aten::bmm", "aten::baddbmm"):
            a, bm = (sh[-2], sh[-1])
            if len(a) == 3:
                return {**e, "cat": "gemm", "b": a[0], "m": a[1], "k": a[2], "n": bm[2],
                        "in_bytes": eb, "bias": int(op == "aten::baddbmm")}
    if op in ATTN_OPS and len(ts) >= 3:
        (bq, sq, h, d), (_, skv, _, _) = ts[0][0], ts[1][0]
        return {**e, "cat": "attention", "b": bq * h, "m": sq, "n": skv, "k": d, "in_bytes": 2}
    if op == "aten::cudnn_convolution" and len(ts) >= 2:
        (nb, c, hh, ww), (o, _, kh, kw) = ts[0][0], ts[1][0]
        m = nb * (hh // kh) * (ww // kw)  # patch embedding: stride == kernel
        return {**e, "cat": "gemm", "m": m, "n": o, "k": c * kh * kw, "in_bytes": ts[0][1],
                "weight": 1}
    if ts:
        s0, eb0 = ts[0]
        rows, cols = (prod(s0[:-1]) if len(s0) > 1 else 1), (s0[-1] if s0 else 1)
        if op == "aten::_softmax":
            return {**e, "cat": "softmax", "m": rows, "n": cols, "in_bytes": eb0}
        if op == "aten::native_layer_norm":
            return {**e, "cat": "norm", "m": rows, "n": cols, "in_bytes": eb0}
        if op in REDUCE_OPS:
            return {**e, "cat": "reduce", "m": rows, "n": cols, "in_bytes": eb0}
        if op == "aten::cumsum":
            return {**e, "cat": "scan", "numel": max(prod(s0), 1), "in_bytes": eb0}
        if op == "aten::cat":
            return {**e, "numel": sum(prod(s) for s, _ in ts), "n_in": 1, "in_bytes": 2,
                    "out_bytes": 2}
        numel = max(prod(s) for s, _ in ts)
        if op in ("aten::gather", "aten::index_select", "aten::take"):
            numel = prod(ts[-1][0])  # output has the index's size, not the source table's
        if numel > 1:
            if op in ("aten::copy_", "aten::_to_copy", "aten::to") and len(ts) >= 2:
                return {**e, "numel": numel, "n_in": 1, "in_bytes": ts[1][1],
                        "out_bytes": ts[0][1]}
            inplace = op.endswith("_")
            n_in = len(ts)
            return {**e, "numel": numel, "n_in": max(1, min(n_in, 3)),
                    "in_bytes": max(eb for _, eb in ts), "out_bytes": eb0 if inplace or
                    op not in ("aten::where",) else ts[-1][1]}
    # Fallback: shape unknown (fill_ on a scalar, arange, unlabeled kernels).
    n = threads(r)
    return {**e, "numel": n, "n_in": 0 if op in ("aten::fill_", "aten::arange") else 1}


def flow_region(k: pd.DataFrame) -> tuple[int, int]:
    """[start, end) of one flow step: the repeating period at the end of the kernel list."""
    key = (k["name"].str.replace("void ", "", regex=False)
           .str.replace(r"<.*", "", regex=True).str.slice(0, 50) + "|" + k["grid"].astype(str))
    kn = key.tolist()
    n = len(kn)
    for p in range(200, n // 3):
        a = n - 3 * p
        if kn[a:a + p] == kn[a + p:a + 2 * p] == kn[a + 2 * p:a + 3 * p]:
            s = n - 2 * p
            while s - p >= 0 and kn[s - p:s] == kn[s:s + p]:
                s -= p
            # take the middle period of the loop (away from warm-up/tail effects)
            steps = (n - s) // p
            start = s + p * (steps // 2)
            return start, start + p
    raise SystemExit("no repeating flow step found")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run", type=Path)
    ap.add_argument("--region", choices=["full", "flow"], default="flow")
    ap.add_argument("--step", type=int, default=0)
    ap.add_argument("-o", "--out", type=Path, required=True)
    a = ap.parse_args()

    ev = pd.read_csv(a.run / "kernels.csv.gz")
    ev = ev[ev.step == a.step].reset_index(drop=True)
    kern = ev[ev.cat == "kernel"].reset_index(drop=True)
    if a.region == "flow":
        s, t = flow_region(kern)
        t0, t1 = kern.ts_us.iloc[s], kern.ts_us.iloc[t]
        ev = ev[(ev.ts_us >= t0) & (ev.ts_us < t1)].reset_index(drop=True)
    ev = ev.sort_values("ts_us", kind="stable").reset_index(drop=True)

    rows = []
    for _, grp in ev.groupby("correlation", sort=False):
        r = grp.iloc[0]
        dims, types = parse(r.op_input_dims), parse(r.op_input_types)
        d = row_for(r, dims, types)
        d["traced_us"] = round(float(grp.dur_us.sum()), 3)
        d["traced_kernels"] = int((grp.cat == "kernel").sum())
        d["_ts"] = r.ts_us
        rows.append(d)
    out = pd.DataFrame(rows).sort_values("_ts").drop(columns="_ts").reset_index(drop=True)
    out.insert(0, "idx", range(len(out)))
    out = out[COLS]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)

    fb = out[(out.cat == "elementwise") & (out.op.isin(["unlabeled"]) | (out.n_in == 0))]
    summ = out.groupby("cat").agg(rows=("idx", "size"), traced_us=("traced_us", "sum"))
    print(f"{a.run} region={a.region}: {len(out)} launches, "
          f"{int(out.traced_kernels.sum())} kernels, traced GPU {out.traced_us.sum():.1f} us",
          file=sys.stderr)
    print(summ.to_string(), file=sys.stderr)
    print(f"shape fallbacks: {len(fb)} rows, {fb.traced_us.sum():.1f} us", file=sys.stderr)


if __name__ == "__main__":
    main()
