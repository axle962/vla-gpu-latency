"""Check workloads/robotwin_delay_eval.py against the LIBERO harness loop (libero_eval.py) with a
fake policy; CPU only:  envs/robotwin/bin/python workloads/test_robotwin_delay.py"""
import importlib, os, sys, types
sys.argv = ["x"]
import torch
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
for d in (0, 1, 5, 20):
    os.environ["ROBOTWIN_DELAY"] = str(d)
    import robotwin_delay_eval as m
    m = importlib.reload(m)
    class P:
        config = types.SimpleNamespace(n_action_steps=50)
        def __init__(s): s.k = 0; s.plan_steps = []; s.step = 0
        def reset(s): pass
        def predict_action_chunk(s, batch):
            s.plan_steps.append(s.step); c = torch.arange(50.) + 1000 * s.k; s.k += 1
            return c.view(1, 50, 1)
    p = m.wrap_policy(P()); p.reset()
    seq = []
    for t in range(160):
        p.step = t; seq.append(int(p.select_action({})[0, 0]))
    # expected (LIBERO loop): chunk0 [0..49]; then for d=0 chunk k at step 50k, actions 0..49;
    # for d>0 chunk k planned at step 50k - d, executed from index d for steps 50k..50k+49
    # reference: the LIBERO harness loop (workloads/libero_eval.py), with the same fake chunks
    from collections import deque
    k = 0; plans = []
    def inf(t):
        global k
        plans.append(t); c = [1000 * k + i for i in range(50)]; k += 1; return c
    queue = deque(inf(0)[:50]); pending = None; exp = []
    for t in range(160):
        if d > 0 and pending is None and len(queue) == d: pending = inf(t)
        if not queue:
            if d == 0: queue = deque(inf(t)[:50])
            else: queue = deque(pending[d:d + 50]); pending = None
        exp.append(queue.popleft())
    ok = seq == exp and p.plan_steps == plans
    print(f"d={d}: ok={ok} plan_steps={p.plan_steps} first={seq[48:53]} inferences={m.STATS['inferences']}")
