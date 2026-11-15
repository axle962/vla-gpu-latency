"""Low-overhead GPU timeline of the SmolVLA denoising loop, eager vs CUDA graph, for slides.

torch.profiler with shapes roughly doubles the eager step time (each op is recorded on the
host), so the timeline figure uses Nsight Systems instead (CUDA API + kernel trace only).
After warm-up, two eager loops and two graph replays run inside cudaProfilerStart/Stop, each
in an NVTX range ("eager_loop_<i>", "graph_loop_<i>"); model/timeline_figure.py --nsys reads
the exported SQLite file.

Usage (on cuda2, inside envs/smolvla; nsys from module cuda-12.4):
  nsys profile --trace=cuda,nvtx --capture-range=cudaProfilerApi --capture-range-end=stop \
    -o results/nsys/<name> python workloads/nsys_smolvla.py
  nsys export --type sqlite -o results/nsys/<name>.sqlite results/nsys/<name>.nsys-rep
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trace_smolvla import CudaConstMemo, build  # noqa: E402

from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks  # noqa: E402
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS  # noqa: E402


class Args:
    cameras, num_steps, chunk_size, dtype, batch = 3, 10, 50, "fp16", 1


@torch.no_grad()
def main() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    policy, batch, noise, _ = build(Args)
    m, cfg = policy.model, policy.config
    images, img_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    embs, pad, att = m.embed_prefix(images, img_masks, batch[OBS_LANGUAGE_TOKENS],
                                    batch[OBS_LANGUAGE_ATTENTION_MASK], state=state)
    _, kv = m.vlm_with_expert.forward(
        attention_mask=make_att_2d_masks(pad, att), position_ids=torch.cumsum(pad, 1) - 1,
        past_key_values=None, inputs_embeds=[embs, None], use_cache=cfg.use_cache,
        fill_kv_cache=True)
    dt = -1.0 / cfg.num_steps
    times = [torch.full((1,), 1.0 + s * dt, dtype=torch.float32, device="cuda")
             for s in range(cfg.num_steps)]

    def denoise(x):
        for t in times:
            x = x + dt * m.denoise_step(x_t=x, prefix_pad_masks=pad, past_key_values=kv, timestep=t)
        return x

    g = torch.cuda.CUDAGraph()
    with CudaConstMemo().active():
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                denoise(noise)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        with torch.cuda.graph(g):
            denoise(noise)
    for _ in range(5):
        denoise(noise)
        g.replay()
    torch.cuda.synchronize()

    torch.cuda.cudart().cudaProfilerStart()
    for i in range(2):
        for name, fn in (("eager_loop", lambda: denoise(noise)), ("graph_loop", g.replay)):
            torch.cuda.synchronize()
            time.sleep(0.05)
            torch.cuda.nvtx.range_push(f"{name}_{i}")
            fn()
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()
    torch.cuda.cudart().cudaProfilerStop()


if __name__ == "__main__":
    main()
