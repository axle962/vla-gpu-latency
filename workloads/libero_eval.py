"""LIBERO closed-loop evaluation with a simulated inference delay (latency vs task success).

Mirrors lerobot's rollout (scripts/lerobot_eval.py: preprocess_observation, add_envs_task, the
LIBERO env processors, the policy pre/post-processors), with whole action chunks instead of
select_action's queue so that inference latency can be modeled in control steps.

Delay model (naive asynchronous execution, as the baseline in real-time chunking work): a
robot keeps moving while the policy computes. With a delay of d control steps (latency divided
by the 50 ms control period at LIBERO's 20 Hz), inference for the next chunk starts when d
actions of the current chunk remain, from the observation at that moment; the robot executes
those d actions; then the new chunk takes over from its index d (its first d actions belong to
the past). d = 0 is ordinary synchronous chunked execution. The first chunk of an episode is
synchronous. The simulator pauses during inference, so d is set by the flag, not by the
measured wall time (which is recorded too).

Episodes: task k, episode e starts from LIBERO's fixed init state e and seeds torch/numpy with
1000k + e (the policies sample flow-matching noise at every inference), so every delay sees the
same scenes and the same noise stream (paired comparison).

Usage (envs/smolvla for smolvla, envs/pi for pi0/pi05; MUJOCO_GL=egl):
  python workloads/libero_eval.py --policy pi0 --precision fp16 --suite libero_10 \
     --tasks 0-9 --episodes 5 --delays 0,5,10 --out results/libero/<name>.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import sys
import time
from collections import deque
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np  # noqa: E402
import torch  # noqa: E402

from lerobot.envs.configs import LiberoEnv  # noqa: E402
from lerobot.envs.factory import make_env, make_env_pre_post_processors  # noqa: E402
from lerobot.envs.utils import add_envs_task, preprocess_observation  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.utils.constants import ACTION  # noqa: E402

REPOS = {"smolvla": "lerobot/smolvla_libero", "pi0": "lerobot/pi0_libero_finetuned_v044",
         "pi05": "lerobot/pi05_libero_finetuned_v044"}


def load_policy(name: str, precision: str):
    from lerobot.configs.policies import PreTrainedConfig
    repo = REPOS[name]
    cfg = PreTrainedConfig.from_pretrained(repo)
    cfg.device = "cuda"
    if hasattr(cfg, "compile_model"):
        cfg.compile_model = False
    if name == "smolvla":
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy as P
        policy = P.from_pretrained(repo, config=cfg).cuda().eval()
        if precision == "fp16":
            policy.model.vlm_with_expert.to(torch.float16)
        # "fp32"/"bf16": keep the checkpoint's own dtypes (stock lerobot behavior)
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from trace_pi import KEEP_FP32, add_fp16_casts
        if name == "pi0":
            from lerobot.policies.pi0.modeling_pi0 import PI0Policy as P
        else:
            from lerobot.policies.pi05.modeling_pi05 import PI05Policy as P
        policy = P.from_pretrained(repo, config=cfg).cuda().eval()
        if precision == "fp16":
            pw = policy.model.paligemma_with_expert
            pw.to(torch.float16)
            for n, p in pw.named_parameters():
                if any(k in n for k in KEEP_FP32):
                    p.data = p.data.to(torch.float32)
            add_fp16_casts(pw)
    pre, post = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=repo,
        preprocessor_overrides={"device_processor": {"device": "cuda"}})
    return policy, pre, post


def parse_ids(s: str) -> list[int]:
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=list(REPOS), required=True)
    ap.add_argument("--precision", choices=["fp16", "bf16", "fp32"], default="fp16")
    ap.add_argument("--suite", default="libero_10")
    ap.add_argument("--tasks", default="0-9")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--delays", default="0")
    ap.add_argument("--exec-horizon", type=int, default=0, help="0 = policy n_action_steps")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    policy, pre, post = load_policy(args.policy, args.precision)
    H = args.exec_horizon or policy.config.n_action_steps
    delays = [int(x) for x in args.delays.split(",")]
    assert max(delays) < H, "delay must be shorter than the execution horizon"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    new = not out.exists()
    f = open(out, "a", newline="")
    w = csv.writer(f)
    if new:
        w.writerow(["policy", "precision", "suite", "task_id", "episode", "delay_steps",
                    "exec_horizon", "success", "steps", "inferences", "infer_ms_median",
                    "episode_s", "gpu",
                    "host", "task"])

    for tid in parse_ids(args.tasks):
        env_cfg = LiberoEnv(task=args.suite, task_ids=[tid])
        env_pre, env_post = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy.config)
        for d in delays:
            # fresh env per delay so episode e always gets init state e
            env = make_env(env_cfg, n_envs=1)[args.suite][tid]
            max_steps = env.call("_max_episode_steps")[0]
            task_text = env.call("task_description")[0] if hasattr(env, "call") else ""
            for e in range(args.episodes):
                obs, _ = env.reset(seed=[e])
                policy.reset()
                # Same flow-matching noise stream for every delay (paired comparison).
                torch.manual_seed(1000 * tid + e)
                np.random.seed(1000 * tid + e)
                t_ep = time.perf_counter()
                infer_ms = []

                def infer(raw):
                    o = preprocess_observation(raw)
                    o = add_envs_task(env, o)
                    o = env_pre(o)
                    o = pre(o)
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    with torch.inference_mode():
                        chunk = policy.predict_action_chunk(o)
                    torch.cuda.synchronize()
                    infer_ms.append(1e3 * (time.perf_counter() - t0))
                    chunk = post(chunk)  # (1, chunk, A), unnormalized
                    acts = []
                    for i in range(chunk.shape[1]):
                        a = env_post({ACTION: chunk[:, i]})[ACTION]
                        acts.append(a.to("cpu").numpy())
                    return acts

                queue = deque(infer(obs)[:H])
                pending = None
                success, step = False, 0
                while step < max_steps:
                    if d > 0 and pending is None and len(queue) == d:
                        pending = infer(obs)          # planned from the current observation
                    if not queue:
                        if d == 0:
                            queue = deque(infer(obs)[:H])
                        else:
                            if pending is None:       # cannot happen when d < H
                                pending = infer(obs)
                            queue = deque(pending[d:d + H])
                            pending = None
                    obs, rew, term, trunc, info = env.step(queue.popleft())
                    step += 1
                    if "final_info" in info:
                        success = bool(info["final_info"]["is_success"][0])
                    if bool(term[0]) or bool(trunc[0]):
                        break
                w.writerow([args.policy, args.precision, args.suite, tid, e, d, H, int(success),
                            step, len(infer_ms), f"{np.median(infer_ms):.1f}",
                            f"{time.perf_counter() - t_ep:.1f}",
                            torch.cuda.get_device_name(0), platform.node().split(".")[0],
                            task_text])
                f.flush()
                print(f"task {tid} delay {d} ep {e}: success={int(success)} steps={step} "
                      f"inferences={len(infer_ms)} infer {np.median(infer_ms):.0f} ms, "
                      f"episode {time.perf_counter() - t_ep:.0f} s", flush=True)
            env.close()
    f.close()


if __name__ == "__main__":
    main()
