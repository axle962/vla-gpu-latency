# Graph-worthiness rule: predicted vs measured CUDA-graph speedup

`S = T_e / (D + K*eps)`, eps = +0.519 us per kernel, fitted on one run (`smolvla_cam3_steps10_chunk50`, 2080 Ti) and held fixed. 120 held-out runs, 8 workload families, 4 GPUs. Inputs come from eager mode only (region wall time, kernel count, summed kernel time of one profiled step); no graph is captured.

## Accuracy (held-out runs)

| workload | runs | S measured (range) | rule MAPE | per-machine eps | no eps (T_e/D) | 1/busy_frac |
|---|---|---|---|---|---|---|
| act | 16 | 1.63 to 6.77 | 6.7% | 6.0% | 8.2% | 8.2% |
| diffusion | 20 | 0.99 to 1.04 | 0.8% | 0.4% | 0.8% | 0.8% |
| diffusionbench | 16 | 0.99 to 1.20 | 1.0% | 0.4% | 1.0% | 1.5% |
| pi0 | 6 | 2.90 to 11.69 | 8.5% | 2.4% | 6.8% | 48.2% |
| pi05 | 4 | 3.17 to 10.93 | 7.7% | 1.6% | 7.1% | 49.1% |
| smolvla | 42 | 3.90 to 24.26 | 17.3% | 5.7% | 15.8% | 27.8% |
| smolvlabatch | 12 | 1.52 to 17.19 | 6.5% | 0.9% | 8.3% | 39.8% |
| smolvlarobotwin | 4 | 4.21 to 23.86 | 16.5% | 1.0% | 17.0% | 29.0% |
| **all** | 120 | 0.99 to 24.26 | **9.1%** | **3.1%** | 8.9% | 20.1% |

By GPU (per-machine eps: fitted on that GPU's SmolVLA baseline, evaluated on its other runs):

| GPU | runs | rule MAPE | median signed error | eps for this GPU (us) | per-machine eps MAPE |
|---|---|---|---|---|---|
| 2080 Ti | 42 | 6.2% | -0.6% | +0.51 | 6.2% |
| A100 | 18 | 17.5% | -27.1% | -0.65 | 2.4% |
| RTX 4070 | 31 | 14.6% | -14.0% | -0.35 | 1.4% |
| TITAN V | 29 | 2.3% | +1.1% | +0.80 | 0.9% |

## Decision: capture if predicted S >= 1.5

Correct for 120 of 120 held-out runs (100%): 84 worth it and predicted so, 36 not worth it and predicted so, 0 predicted worth it but not, 0 worth it but predicted not.

## Reading

With c = T_e/K (eager time per kernel, mostly host dispatch) and t = D/K (mean kernel time), S ~= c / (t + eps). Per workload (median over GPUs):

| workload | c_us | t_us | S_meas |
|---|---|---|---|
| act | 35.56 | 13.58 | 2.76 |
| diffusion | 56.57 | 55.85 | 1.0 |
| diffusionbench | 39.69 | 38.92 | 1.03 |
| pi0 | 40.95 | 4.78 | 8.84 |
| pi05 | 41.05 | 5.14 | 7.98 |
| smolvla | 35.08 | 3.69 | 10.21 |
| smolvlabatch | 34.22 | 6.47 | 3.8 |
| smolvlarobotwin | 30.78 | 3.33 | 9.46 |
