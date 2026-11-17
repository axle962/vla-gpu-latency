"""lerobot-eval (lerobot 0.7) with a simulated inference delay, for RoboTwin 2.0.

Same delay model as workloads/libero_eval.py (naive asynchronous execution): with a delay of d
control steps, inference for the next chunk starts when d actions of the current chunk remain,
from the observation at that moment; the robot executes those d actions; then the new chunk takes
over from its index d. d = 0 is ordinary synchronous chunked execution (identical to lerobot's
own select_action queue). The first chunk of an episode is synchronous. The simulator pauses
during inference, so d is set by the flag, not by measured wall time (recorded too).

Unlike the LIBERO harness this does not re-implement the rollout: it runs lerobot's own
`lerobot-eval` (env, rename map, processors, per-episode seeds, success metric) and only wraps
the policy that `make_policy` returns, replacing `select_action`. Episode seeds come from
lerobot (`--seed`, default 1000, one per episode), so every delay sees the same scenes; torch is
reseeded at every rollout start (policy.reset) so the flow-matching noise stream is the same for
every delay as well (paired comparison).

Usage (envs/robotwin, from the RoboTwin root, RoboTwin on PYTHONPATH):
  ROBOTWIN_DELAY=5 python workloads/robotwin_delay_eval.py <lerobot-eval arguments>
Optional: ROBOTWIN_EXEC_H (execution horizon, default the policy's n_action_steps).
Writes <output_dir>/delay_info.json (delay, horizon, inference count and times).
"""

from __future__ import annotations

import json
import os
import time
import types
from collections import deque

import torch

import lerobot.scripts.lerobot_eval as le

DELAY = int(os.environ.get("ROBOTWIN_DELAY", "0"))
EXEC_H = int(os.environ.get("ROBOTWIN_EXEC_H", "0"))
STATS = {"delay_steps": DELAY, "exec_horizon": None, "rollouts": 0, "inferences": 0,
         "infer_ms": [], "select_calls": 0}


def wrap_policy(policy):
    H = EXEC_H or policy.config.n_action_steps
    assert DELAY < H, "delay must be shorter than the execution horizon"
    STATS["exec_horizon"] = H
    orig_reset = policy.reset
    state = {"queue": deque(), "pending": None, "start": True}

    sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)

    def infer(batch):
        sync()
        t0 = time.perf_counter()
        with torch.inference_mode():
            chunk = policy.predict_action_chunk(batch)  # (B, chunk, A)
        sync()
        STATS["infer_ms"].append(1e3 * (time.perf_counter() - t0))
        STATS["inferences"] += 1
        return chunk

    def reset(self):
        orig_reset()
        state["queue"], state["pending"], state["start"] = deque(), None, True
        # same noise stream for every delay: seed by rollout index
        torch.manual_seed(10_000 + STATS["rollouts"])
        STATS["rollouts"] += 1

    def select_action(self, batch, **kwargs):
        # Called once per control step with the current observation; same order as the
        # LIBERO harness loop. Queue entries are (B, A) actions.
        STATS["select_calls"] += 1
        q = state["queue"]
        if state["start"]:  # first chunk of an episode: synchronous
            q.extend(infer(batch).transpose(0, 1)[:H])
            state["start"] = False
        if DELAY > 0 and state["pending"] is None and len(q) == DELAY:
            state["pending"] = infer(batch)  # planned from the current observation
        if not q:
            if DELAY == 0:
                q.extend(infer(batch).transpose(0, 1)[:H])
            else:
                q.extend(state["pending"].transpose(0, 1)[DELAY:DELAY + H])
                state["pending"] = None
        return q.popleft()

    policy.reset = types.MethodType(reset, policy)
    policy.select_action = types.MethodType(select_action, policy)
    return policy


_orig_make_policy = le.make_policy


def make_policy(*args, **kwargs):
    return wrap_policy(_orig_make_policy(*args, **kwargs))


le.make_policy = make_policy


def _dump():
    import sys
    out = None
    for a in sys.argv:
        if a.startswith("--output_dir="):
            out = a.split("=", 1)[1]
    ms = sorted(STATS["infer_ms"])
    info = dict(STATS, infer_ms_median=ms[len(ms) // 2] if ms else None,
                infer_ms_first=STATS["infer_ms"][:3])
    info.pop("infer_ms")
    print("delay_info", json.dumps(info), flush=True)
    if out:
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "delay_info.json"), "w") as f:
            json.dump(info, f, indent=1)


if __name__ == "__main__":
    try:
        le.main()
    finally:
        _dump()
