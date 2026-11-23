# Graph-worthiness rule: predicted vs measured CUDA-graph speedup

`S = T_e / (D + K*eps)` from eager mode only (region wall time, kernel count, summed kernel time of one profiled eager step). No graph is captured to make a prediction.

Evaluation set: 114 runs, 95 distinct (config, GPU) situations, 8 policies (ACT, Diffusion Policy, OpenVLA, SmolVLA, SmolVLA compiled, pi0, pi0-FAST, pi0.5), 4 GPUs; measured speedups 0.99x to 24.3x. The SmolVLA baseline config (smolvla_cam3_steps10_chunk50), which the global and per-GPU fits use, is excluded on every GPU.

## Accuracy by variant

MAPE of the predicted speedup over runs and over situations (median over repeats), with a 95% bootstrap CI that resamples configs.

| variant | eps source | MAPE runs | 95% CI | MAPE situations | max |err| |
|---|---|---|---|---|---|
| one eps (1 run) | +0.52 us, 1 run | 7.1% | 4.6 to 9.9% | 7.7% | 38% |
| eps per GPU | 1 SmolVLA run per GPU | 1.8% | 1.0 to 2.8% | 1.8% | 25% |
| leave policy out | other policies' runs, same GPU | 2.5% | 1.5 to 3.8% | 2.7% | 25% |
| leave GPU out | other GPUs only | 12.7% | 8.6 to 17.1% | 13.7% | 38% |
| T_e/D (no eps) | - | 7.1% | 4.6 to 9.7% | 7.7% | 30% |
| 1/busy_frac | - | 16.9% | 11.0 to 23.7% | 18.4% | 62% |

## By GPU

eps = graph node cost minus the profiler's bias on each traced kernel, so it can be negative. The C++ node cost is shown for reference.

| GPU | runs | eps per GPU (us) | eps leave-GPU-out | eps probe | C++ node cost (us) | MAPE one eps (1 run) | MAPE eps per GPU | MAPE leave policy out | MAPE leave GPU out |
|---|---|---|---|---|---|---|---|---|---|
| 2080 Ti | 32 | +0.52 | -0.35 | pending | 1.07 | 1.9% | 1.9% | 2.0% | 12.5% |
| A100 | 17 | -0.65 | +0.52 | pending | 0.29 | 14.6% | 2.7% | 4.3% | 14.6% |
| RTX 4070 | 34 | -0.35 | +0.52 | pending | 0.28 | 12.6% | 1.8% | 2.8% | 12.6% |
| TITAN V | 31 | +0.80 | -0.35 | pending | 0.77 | 2.2% | 1.1% | 1.7% | 12.0% |

## By policy

t = D/K, mean traced kernel time; c = T_e/K, eager time per kernel. S ~= c / (t + eps).

| policy | runs | GPUs | S measured | t (us/kernel) | c (us/kernel) | one eps (1 run) | eps per GPU | leave policy out | leave GPU out | T_e/D (no eps) | 1/busy_frac |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ACT | 12 | 4 | 1.63 to 6.77 | 13.0 | 31.5 | 3.5% | 2.6% | 2.4% | 7.3% | 4.4% | 4.4% |
| Diffusion Policy | 36 | 4 | 0.99 to 1.20 | 40.8 | 43.7 | 0.9% | 0.4% | 0.4% | 1.8% | 0.9% | 1.1% |
| OpenVLA | 2 | 1 | 2.80 to 2.87 | 9.4 | 25.7 | 8.7% | 3.5% | 3.6% | 8.7% | 3.6% | 13.1% |
| SmolVLA | 41 | 4 | 1.52 to 24.26 | 3.7 | 34.2 | 14.1% | 2.3% | 4.6% | 24.5% | 14.2% | 30.8% |
| SmolVLA compiled | 9 | 3 | 1.12 to 10.89 | 8.8 | 36.0 | 4.2% | 3.0% | 3.1% | 11.2% | 5.5% | 5.5% |
| pi0 | 6 | 3 | 2.90 to 11.69 | 4.8 | 41.0 | 8.5% | 2.4% | 1.7% | 15.0% | 6.8% | 48.2% |
| pi0-FAST | 4 | 3 | 1.73 to 3.49 | 13.4 | 39.8 | 4.7% | 2.1% | 2.1% | 6.7% | 3.5% | 3.5% |
| pi0.5 | 4 | 3 | 3.17 to 10.93 | 5.1 | 41.0 | 7.7% | 1.7% | 1.0% | 15.1% | 7.1% | 49.1% |

## Decisions: capture if predicted S >= threshold

Accuracy over all runs / over the hard runs (measured speedup within a factor 1.25 of the threshold). 'Always capture' is right for the worth-it runs only.

| threshold | worth it | hard (within 1.25x) | one eps (1 run) | eps per GPU | leave policy out | leave GPU out | T_e/D (no eps) | 1/busy_frac | always capture |
|---|---|---|---|---|---|---|---|---|---|
| 1.5x | 77/114 | 6 | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 100% | 98% / 83% | 68% |
| 2x | 72/114 | 12 | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 100% | 97% / 83% | 63% |
| 3x | 57/114 | 18 | 99% / 94% | 99% / 94% | 99% / 94% | 98% / 89% | 97% / 83% | 96% / 78% | 50% |
| 5x | 39/114 | 11 | 99% / 91% | 100% / 100% | 99% / 91% | 95% / 45% | 96% / 55% | 91% / 82% | 34% |
| 10x | 22/114 | 23 | 92% / 61% | 98% / 91% | 98% / 91% | 91% / 57% | 98% / 91% | 89% / 57% | 19% |

Lowest accuracy over 25 thresholds from 1.2x to 15x: one eps (1 run) 93%, eps per GPU 97%, leave policy out 94%, leave GPU out 90%, T_e/D (no eps) 92%, 1/busy_frac 82%.

## Ranking

Spearman correlation of predicted and measured speedup over the runs of each GPU.

| GPU | runs | one eps (1 run) | eps per GPU | leave policy out | leave GPU out | T_e/D (no eps) | 1/busy_frac |
|---|---|---|---|---|---|---|---|
| 2080 Ti | 32 | 0.991 | 0.991 | 0.993 | 0.996 | 0.995 | 0.975 |
| A100 | 17 | 0.988 | 0.988 | 0.988 | 0.988 | 0.988 | 0.968 |
| RTX 4070 | 34 | 0.990 | 0.990 | 0.990 | 0.990 | 0.990 | 0.946 |
| TITAN V | 31 | 0.973 | 0.973 | 0.973 | 0.973 | 0.973 | 0.946 |

## Largest errors (eps per GPU)

| config | gpu | S_meas | S_per_gpu | err_per_gpu | t_us | c_us |
|---|---|---|---|---|---|---|
| smolvla_cam3_steps20_chunk50 | 2080 Ti | 10.24 | 7.71 | -24.71 | 4.11 | 35.7 |
| smolvla_cam3_steps20_chunk50 | A100 | 9.05 | 7.42 | -18.03 | 4.38 | 27.63 |
| act_res240x320_chunk100 | A100 | 2.96 | 3.42 | 15.69 | 8.52 | 26.94 |
| smolvla_cam1_steps10_chunk50 | 2080 Ti | 14.1 | 12.1 | -14.17 | 3.24 | 45.46 |
| smolvlacompile_cam3_steps10_chunk50 | RTX 4070 | 10.83 | 11.77 | 8.66 | 4.43 | 47.97 |
| smolvlacompile_cam3_steps10_chunk50 | RTX 4070 | 10.89 | 11.79 | 8.32 | 4.44 | 48.17 |

## Profile a short window

SmolVLA with 20 flow steps, eps per GPU. Long profiling sessions inflate the traced kernel durations on the 2080 Ti and A100 hosts (CUPTI); K and D of a periodic loop can come from a short profile scaled to the full loop.

| GPU | t traced, 5 steps (us) | t traced, 20 steps (us) | S measured | 20-step profile | 5-step profile x 4 |
|---|---|---|---|---|---|
| 2080 Ti | 2.98 | 4.11 | 10.24 | -24.7% | -0.2% |
| A100 | 3.70 | 4.38 | 9.05 | -18.0% | +0.2% |
| RTX 4070 | 2.42 | 2.43 | 22.67 | -2.7% | -2.4% |
| TITAN V | 4.52 | 4.57 | 4.48 | -0.1% | +0.8% |

## Stress test: host contention (2080 Ti, one core)

Not in the evaluation set above. Contention inflates the CUPTI-traced kernel durations (t), so D is biased when the profile is taken on a busy host; T_e should be measured under deployment conditions, D and K on a quiet host. eps per GPU as above.

| config | host | eager slowdown | t traced (us) | S measured | profile under contention | profile on quiet host |
|---|---|---|---|---|---|---|
| act_res480x640_chunk100 | alone | 1.00x | 14.53 | 2.48 | -1.5% | -1.5% |
| act_res480x640_chunk100 | share | 1.98x | 14.10 | 3.83 | +30.5% | +26.7% |
| act_res480x640_chunk100 | smt_alu | 1.62x | 14.20 | 3.12 | +29.6% | +26.7% |
| act_res480x640_chunk100 | smt_mem | 1.90x | 14.13 | 4.61 | +3.2% | +0.5% |
| smolvla_cam1_steps10_chunk50 | alone | 1.00x | 2.74 | 10.81 | -0.5% | -0.5% |
| smolvla_cam1_steps10_chunk50 | share | 1.98x | 3.79 | 20.22 | -20.3% | +5.5% |
| smolvla_cam1_steps10_chunk50 | smt_alu | 1.69x | 3.54 | 18.05 | -19.3% | +0.7% |
| smolvla_cam1_steps10_chunk50 | smt_mem | 1.86x | 3.79 | 19.99 | -24.3% | +0.1% |
| smolvla_cam3_steps10_chunk50 | alone | 1.00x | 2.96 | 10.16 | -0.3% | -0.3% |
| smolvla_cam3_steps10_chunk50 | share | 1.92x | 4.13 | 18.10 | -19.4% | +7.7% |
| smolvla_cam3_steps10_chunk50 | smt_alu | 1.68x | 4.12 | 16.93 | -24.6% | +0.6% |
| smolvla_cam3_steps10_chunk50 | smt_mem | 1.84x | 4.11 | 18.55 | -24.4% | +0.7% |
