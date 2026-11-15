"""GPU energy per SmolVLA inference: eager vs eager prefix + CUDA-graph loop (hybrid).

For each mode the step runs back to back for --seconds; energy comes from the GPU's cumulative
energy counter (NVML nvmlDeviceGetTotalEnergyConsumption, mJ, Volta and newer), with power
also sampled every 20 ms for the mean/peak. An idle window first gives the GPU's idle power,
so energy above idle is reported too. Only the GPU is measured (no CPU package energy: RAPL is
not readable without root on CIMS). The GPU must have no other process.

Usage (inside envs/smolvla; CUDA_VISIBLE_DEVICES = one nvidia-smi index, which is also the NVML
index):
  python workloads/energy_smolvla.py [--seconds 20] > results/energy/<gpu>_<host>_<stamp>.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import sys
import threading
import time
from pathlib import Path

import pynvml
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trace_smolvla import CudaConstMemo, build  # noqa: E402

from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks  # noqa: E402
from lerobot.utils.constants import (  # noqa: E402
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
)


class PowerSampler:
    def __init__(self, h, period_s: float = 0.02) -> None:
        self.h, self.period, self.samples = h, period_s, []
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.samples.append(pynvml.nvmlDeviceGetPowerUsage(self.h) / 1e3)
            time.sleep(self.period)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *a) -> None:
        self._stop.set()
        self._t.join()


def measure(h, fn, seconds: float) -> dict:
    torch.cuda.synchronize()
    with PowerSampler(h) as ps:
        e0 = pynvml.nvmlDeviceGetTotalEnergyConsumption(h)
        t0 = time.perf_counter()
        n = 0
        while time.perf_counter() - t0 < seconds:
            fn()
            torch.cuda.synchronize()
            n += 1
        t1 = time.perf_counter()
        e1 = pynvml.nvmlDeviceGetTotalEnergyConsumption(h)
    s = ps.samples or [float("nan")]
    return {"steps": n, "wall_s": t1 - t0, "energy_j": (e1 - e0) / 1e3,
            "power_mean_w": sum(s) / len(s), "power_max_w": max(s),
            "sm_clock_mhz": pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cameras", type=int, default=3)
    ap.add_argument("--num-steps", type=int, default=10)
    ap.add_argument("--chunk-size", type=int, default=50)
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seconds", type=float, default=20.0)
    args = ap.parse_args()

    pynvml.nvmlInit()
    idx = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
    h = pynvml.nvmlDeviceGetHandleByIndex(idx)
    assert torch.cuda.get_device_name(0) in pynvml.nvmlDeviceGetName(h), "NVML index mismatch"

    torch.backends.cuda.matmul.allow_tf32 = False
    policy, batch, noise, _ = build(args)
    m, cfg = policy.model, policy.config

    @torch.no_grad()
    def prefix():
        images, img_masks = policy.prepare_images(batch)
        state = policy.prepare_state(batch)
        embs, pad, att = m.embed_prefix(images, img_masks, batch[OBS_LANGUAGE_TOKENS],
                                        batch[OBS_LANGUAGE_ATTENTION_MASK], state=state)
        _, kv = m.vlm_with_expert.forward(
            attention_mask=make_att_2d_masks(pad, att), position_ids=torch.cumsum(pad, 1) - 1,
            past_key_values=None, inputs_embeds=[embs, None], use_cache=cfg.use_cache,
            fill_kv_cache=True)
        return pad, kv

    @torch.no_grad()
    def full_step():
        images, img_masks = policy.prepare_images(batch)
        state = policy.prepare_state(batch)
        return m.sample_actions(images, img_masks, batch[OBS_LANGUAGE_TOKENS],
                                batch[OBS_LANGUAGE_ATTENTION_MASK], state, noise=noise)

    dt = -1.0 / cfg.num_steps
    times = [torch.full((args.batch,), 1.0 + s * dt, dtype=torch.float32, device="cuda")
             for s in range(cfg.num_steps)]

    @torch.no_grad()
    def denoise(pad, kv, x0):
        x_t = x0
        for t in times:
            x_t = x_t + dt * m.denoise_step(x_t=x_t, prefix_pad_masks=pad, past_key_values=kv,
                                            timestep=t)
        return x_t

    pad0, kv0 = prefix()
    static_pad = pad0.clone()
    static_kv = {i: {k: v.clone() for k, v in layer.items()} for i, layer in kv0.items()}
    static_x = noise.clone()
    g = torch.cuda.CUDAGraph()
    memo = CudaConstMemo()
    with memo.active():
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                denoise(static_pad, static_kv, static_x)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        with torch.cuda.graph(g):
            denoise(static_pad, static_kv, static_x)

    def hybrid():
        pad, kv = prefix()
        static_pad.copy_(pad)
        for i, layer in kv.items():
            for k, v in layer.items():
                static_kv[i][k].copy_(v)
        g.replay()

    for _ in range(3):
        full_step()
        hybrid()
    torch.cuda.synchronize()
    time.sleep(5)  # let power settle before the idle window
    rows = [("idle", measure(h, lambda: time.sleep(0.05), args.seconds))]
    rows.append(("eager", measure(h, full_step, args.seconds)))
    rows.append(("hybrid", measure(h, hybrid, args.seconds)))
    rows.append(("graph_loop", measure(h, g.replay, args.seconds)))

    idle_w = rows[0][1]["energy_j"] / rows[0][1]["wall_s"]
    w = csv.writer(sys.stdout)
    w.writerow(["gpu", "host", "mode", "batch", "steps", "wall_s", "ms_per_step", "energy_j",
                "j_per_step", "j_per_step_above_idle", "power_mean_w", "power_max_w",
                "idle_w", "sm_clock_mhz_end"])
    for mode, r in rows:
        ms = 1e3 * r["wall_s"] / max(r["steps"], 1)
        jps = r["energy_j"] / max(r["steps"], 1)
        w.writerow([torch.cuda.get_device_name(0), platform.node().split(".")[0], mode,
                    args.batch, r["steps"], f"{r['wall_s']:.3f}", f"{ms:.3f}",
                    f"{r['energy_j']:.3f}", f"{jps:.4f}",
                    f"{jps - idle_w * ms / 1e3:.4f}", f"{r['power_mean_w']:.2f}",
                    f"{r['power_max_w']:.2f}", f"{idle_w:.2f}", r["sm_clock_mhz"]])


if __name__ == "__main__":
    main()
