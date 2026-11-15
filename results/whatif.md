# Spec-only what-if predictions (model/whatif.py)

Kernel transfer from spec sheets only (clock ratio for the first 5 us of each kernel, DRAM bandwidth ratio beyond); host fixed at the A100's. Error bar from results/ablation_calibration.md, level L0 on our four GPUs: SmolVLA graph loop 38%, hybrid 13%, eager 34% MAPE.

## SmolVLA (FP16, 3 cameras, 10 flow steps, chunk 50), source A100 trace

Measured on the A100: eager 383 ms, graph loop 35.6 ms, hybrid 95 ms.

| GPU          |   eager, A100 host (ms) |   graph loop (ms) |   hybrid (ms) |   loop graph speedup |
|:-------------|------------------------:|------------------:|--------------:|---------------------:|
| A100         |                   382.7 |              35.6 |          90.5 |                  9.2 |
| H100 SXM     |                   382.6 |              25.4 |          80.3 |                 12.9 |
| H200         |                   382.6 |              24.9 |          79.8 |                 13.1 |
| L40S         |                   383.2 |              25.9 |          81.3 |                 12.6 |
| RTX 4090     |                   383   |              25   |          80.2 |                 13.1 |
| RTX PRO 6000 |                   382.6 |              21.9 |          76.8 |                 14.9 | 

## OpenVLA-7B (FP16, 7 action tokens), source A100 trace (NVIDIA_A100-SXM4-40GB_b-19-28_20261111-223532, 11553 kernels; b = 0.66 us, p = 16.7 us)

Measured on the A100 (our host, Intel(R) Xeon(R) CPU @ 2.20GHz, torch 2.2.0+cu121): eager 311 ms, graph decode 85.2 ms, eager prefill + graph decode 155 ms.

| GPU          |   kernel time (ms) |   eager, A100 host (ms) |   graph decode (ms) |   prefill + graph decode (ms) |   Yotta measured (ms) |   host per kernel implied (us) |
|:-------------|-------------------:|------------------------:|--------------------:|------------------------------:|----------------------:|-------------------------------:|
| A100         |              119.7 |                   311.3 |                85.2 |                         144.4 |                 nan   |                          nan   |
| H100 SXM     |               63.7 |                   311   |                47   |                         105.8 |                 173.1 |                           21.4 |
| H200         |               51.6 |                   311   |                38.8 |                          97.6 |                 272.7 |                           30   |
| L40S         |              174.6 |                   323.9 |               121.2 |                         192.7 |                 nan   |                          nan   |
| RTX 4090     |              152.3 |                   317.2 |               106.2 |                         171.3 |                 276   |                           29.2 |
| RTX PRO 6000 |               93   |                   311.2 |                66.3 |                         125.3 |                 173.2 |                           20.7 | 

Host time per kernel on the A100 host: p + median dispatch = 33.3 us. 'implied' = the same quantity solved from the Yotta latency with this GPU's predicted kernel times (nan: the latency is below the zero-Python bound).

