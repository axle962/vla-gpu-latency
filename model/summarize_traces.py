"""Collect meta.json of every traced run into one CSV (one row per run) for the model.

Reads only the small meta.json files, so it is safe on the login node.
  python3 model/summarize_traces.py [traces_dir] > results/traces_summary.csv
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

COLS = [
    "model", "gpu", "host", "cpu", "torch", "dtype", "cameras", "num_steps", "chunk_size",
    "prefix_len", "eager_ms_median", "eager_ms_std", "eager_prefix_ms_median",
    "eager_denoise_ms_median", "graph_denoise_ms_median", "hybrid_ms_median",
    "eager_prefill_ms_median", "decode_ms_per_token", "kernels_per_step",
    "gpu_busy_ms_per_step", "gpu_busy_frac", "kernel_dur_us_median",
    "frac_kernels_under_5us", "memcpy_per_step", "run",
]


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "traces")
    w = csv.DictWriter(sys.stdout, fieldnames=COLS + ["graph_denoise_speedup",
                                                     "hybrid_speedup"])
    w.writeheader()
    for meta_p in sorted(root.glob("*/*/meta.json")):
        m = json.loads(meta_p.read_text())
        row = {c: m.get(c, "") for c in COLS}
        row["run"] = str(meta_p.parent.relative_to(root))
        if m.get("graph_denoise_ms_median"):
            row["graph_denoise_speedup"] = round(
                m["eager_denoise_ms_median"] / m["graph_denoise_ms_median"], 3)
            row["hybrid_speedup"] = round(m["eager_ms_median"] / m["hybrid_ms_median"], 3)
        w.writerow(row)


if __name__ == "__main__":
    main()
