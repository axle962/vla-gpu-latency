# Graph-worthiness rule

Predict the CUDA-graph speedup of a region of a PyTorch policy from eager mode only, before
doing the work of capturing it.

```
S = T_e / (D + K * eps)  ~=  c / (t + eps)
```

- `T_e`: eager wall time of the region (time it without the profiler, under deployment conditions)
- `K`, `D`: kernel count and summed traced kernel time of the region in one profiled eager step
- `eps`: per-kernel cost of a graph node minus the profiler's bias per traced kernel (one value per machine)
- `c = T_e / K` (host time per kernel, 26 to 44 us on our hosts), `t = D / K` (mean kernel time)

## Results (`results/graph_worthiness.md`)

114 held-out runs, 95 distinct (config, GPU) situations, 8 policies, 4 GPUs, speedups 0.99x to 24.3x:

| eps from | MAPE (95% CI) |
|---|---|
| one SmolVLA run per GPU | 1.8% (1.0 to 2.8%) |
| other policies on the same GPU (leave policy out) | 2.5% (1.5 to 3.8%) |
| one run on one GPU, used everywhere | 7.1% (4.6 to 9.9%) |
| other GPUs only (leave GPU out) | 12.7% (8.6 to 17.1%) |
| no eps, `T_e / D` | 7.1% |
| `1 / gpu_busy_frac` | 16.9% |

The leave-policy-out version makes the right capture decision for at least 94% of the runs at every
threshold from 1.2x to 15x, and 91% of the hard runs (measured speedup within a factor 1.25 of
the threshold) at 5x and 10x.

## How to profile

1. **Profile a short window.** Long CUPTI sessions inflate traced kernel durations on some
   hosts: a 20-step SmolVLA profile is 18 to 25% off; a 5-step profile scaled to 20 steps is
   within 0.2%.
2. **Profile on a quiet host**, and time `T_e` under deployment conditions. Under CPU contention
   the traced durations grow (t 3.0 to 4.1 us for SmolVLA) and the rule is 19 to 25% low; a
   profile from the quiet host fixes it (within +8%).
3. Calibrate `eps` once per machine: on one representative workload, or with
   `scripts/run_eps_probe.sh` (no VLA needed; results pending).

## Running

```
python model/graph_worthiness.py      # -> results/graph_worthiness{.csv,.md,_thresholds.csv}
                                      #    paper/figures/graph_worthiness{,_threshold}.{pdf,png}
CUDA_VISIBLE_DEVICES=1 WORK=/scratch/$USER scripts/run_eps_probe.sh   # CIMS, idle GPU, ~1 min
```

The A100 variant of the probe command (Slurm, about 0.05 GPU-hours) is in the header of
`scripts/run_eps_probe.sh`. The paper text is `paper/graph_worthiness_section.tex`
(`\input` it after the cross-GPU results).
