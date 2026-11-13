"""Run one eager SmolVLA step inside a CUDA profiler range, for Nsight Compute.

Builds the policy exactly as workloads/trace_smolvla.py does (same `build`, FP16, same inputs),
warms up, then runs one stock `sample_actions` step between cudaProfilerStart/Stop, so that
`ncu --profile-from-start off` collects counters for exactly the kernels of one step (same
kernel sequence as `kernels.csv.gz` of the trace).

  ncu --profile-from-start off ... python profile/ncu_smolvla.py [--cameras 3 ...]
(scripts/run_ncu.sh wraps this.)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "workloads"))
from trace_smolvla import (  # noqa: E402
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    build,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cameras", type=int, default=3)
    ap.add_argument("--num-steps", type=int, default=10)
    ap.add_argument("--chunk-size", type=int, default=50)
    ap.add_argument("--dtype", choices=["fp16", "stock"], default="fp16")
    ap.add_argument("--warmup", type=int, default=3)
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    policy, batch, noise, _ = build(args)
    m = policy.model

    @torch.no_grad()
    def full_step():
        images, img_masks = policy.prepare_images(batch)
        state = policy.prepare_state(batch)
        return m.sample_actions(
            images, img_masks, batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK],
            state, noise=noise,
        )

    for _ in range(args.warmup):
        full_step()
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    out = full_step()
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()
    print(f"profiled one step, output {tuple(out.shape)}", file=sys.stderr)


if __name__ == "__main__":
    main()
