# Multi-policy hosting on one GPU

GPU `NVIDIA GeForce RTX 4070` (driver 595.104.02), host `cuda5.cims.nyu.edu` (Intel(R) Xeon(R) CPU E5-2650 v2 @ 2.60GHz, 8 CPUs in the affinity mask), torch 2.10.0+cu128 cu12.8. Policy settings: {"cameras": 3, "num_steps": 10, "chunk_size": 50, "diffusion_steps": 10, "act_hw": [480, 640]}. Closed loop (a new step as soon as the previous one returned), median of the middle 80% of each window.

## Policies alone (procs x1)

| policy   | mode   |   step ms |   p95 ms |   GPU busy ms (profiled) |   GPU busy ms (corrected D) |   host ms (Z = step - D) |   GPU share |   kernels/step |
|:---------|:-------|----------:|---------:|-------------------------:|----------------------------:|-------------------------:|------------:|---------------:|
| smolvla  | eager  |     606.6 |    614.9 |                     54.9 |                        42.4 |                    564.2 |        0.07 |          13200 |
| smolvla  | graph  |     117.6 |    122.4 |                     52.9 |                        39.9 |                     77.7 |        0.34 |          13620 |
| act      | eager  |      15.7 |     16.6 |                      3.7 |                         3.4 |                     12.3 |        0.21 |            347 |
| act      | graph  |       3.8 |      3.8 |                      3.6 |                         3.3 |                      0.5 |        0.87 |            349 |

D removes the profiler's 0.95 us per kernel (CLAUDE.md 0.1); Z is what the step spends off the GPU (Python, dispatch, syncs). A policy with a small GPU share leaves room for others.

## Measured: every hosted configuration

| config                | mode   | policy   | copies   |   median ms |   p95 ms | vs alone   |   steps/s (all policies) |   steps/copy |
|:----------------------|:-------|:---------|:---------|------------:|---------:|:-----------|-------------------------:|-------------:|
| procs_smolvla         | eager  | smolvla  | 1        |       606.6 |    614.9 | 1.00x      |                     1.67 |           20 |
| procs_smolvla         | graph  | smolvla  | 1        |       117.6 |    122.4 | 1.00x      |                     8.5  |          102 |
| procs_smolvlax2       | eager  | smolvla  | 2        |       626.2 |    628.6 | 1.03x      |                     3.17 |           19 |
| procs_smolvlax2       | graph  | smolvla  | 2        |       154.2 |    156.6 | 1.31x      |                    12.83 |           77 |
| procs_smolvlax4       | eager  | smolvla  | 4        |       674.2 |    679   | 1.11x      |                     5.92 |           18 |
| procs_smolvlax4       | graph  | smolvla  | 4        |       248.8 |    250.1 | 2.12x      |                    16    |           48 |
| procs-mps_smolvlax2   | eager  | smolvla  | 2        |       643.9 |    653.6 | 1.06x      |                     3    |           18 |
| procs-mps_smolvlax2   | graph  | smolvla  | 2        |       128.6 |    131   | 1.09x      |                    15.5  |           93 |
| procs-mps_smolvlax4   | eager  | smolvla  | 4        |       667.1 |    668.8 | 1.10x      |                     6    |           18 |
| procs-mps_smolvlax4   | graph  | smolvla  | 4        |       162.5 |    163.8 | 1.38x      |                    24.67 |           74 |
| threads_smolvlax2     | eager  | smolvla  | 2        |       824.6 |    833   | 1.36x      |                     2.5  |           15 |
| threads_smolvlax2     | graph  | smolvla  | 2        |       145.9 |    156   | 1.24x      |                    13.5  |           81 |
| threads_smolvlax4     | eager  | smolvla  | 4        |      4431   |   4630.1 | 7.30x      |                     1    |            3 |
| threads_smolvlax4     | graph  | smolvla  | 4        |       660.1 |    673.1 | 5.61x      |                     6.08 |           18 |
| procs_act             | eager  | act      | 1        |        15.7 |     16.6 | 1.00x      |                    63    |          756 |
| procs_act             | graph  | act      | 1        |         3.8 |      3.8 | 1.00x      |                   263.5  |         3162 |
| procs_actx2           | eager  | act      | 2        |        16.6 |     17.4 | 1.06x      |                   119.92 |          719 |
| procs_actx2           | graph  | act      | 2        |         7.8 |      8.6 | 2.06x      |                   255.67 |         1534 |
| procs_actx4           | eager  | act      | 4        |        18.8 |     19.7 | 1.20x      |                   212.42 |          637 |
| procs_actx4           | graph  | act      | 4        |        15.6 |     16.5 | 4.15x      |                   254.33 |          763 |
| procs-mps_actx2       | eager  | act      | 2        |        15.9 |     16.9 | 1.02x      |                   124.25 |          745 |
| procs-mps_actx2       | graph  | act      | 2        |         6.9 |      7.8 | 1.82x      |                   287.92 |         1727 |
| procs-mps_actx4       | eager  | act      | 4        |        17.7 |     18.1 | 1.13x      |                   226.67 |          676 |
| procs-mps_actx4       | graph  | act      | 4        |        13.5 |     14.7 | 3.58x      |                   294.33 |          883 |
| threads_actx2         | eager  | act      | 2        |        24   |     54.3 | 1.53x      |                    68.17 |          409 |
| threads_actx2         | graph  | act      | 2        |         6.8 |      7.7 | 1.80x      |                   291.83 |         1751 |
| threads_actx4         | eager  | act      | 4        |       134   |    138.8 | 8.53x      |                    29.83 |           89 |
| threads_actx4         | graph  | act      | 4        |        13.2 |     14.3 | 3.50x      |                   301.83 |          905 |
| batch_smolvlax4_b4    | eager  | smolvla  | batch 4  |       656.7 |    663.2 | 1.08x      |                     6.33 |           19 |
| batch_smolvlax4_b4    | graph  | smolvla  | batch 4  |       183   |    186   | 1.56x      |                    21.67 |           65 |
| procs_smolvla+act     | eager  | smolvla  | 1        |       616.4 |    622.1 | 1.02x      |                    60.83 |           19 |
| procs_smolvla+act     | eager  | act      | 1        |        16.8 |     17.5 | 1.07x      |                    60.83 |          711 |
| procs_smolvla+act     | graph  | smolvla  | 1        |       172.8 |    175.8 | 1.47x      |                   183.25 |           69 |
| procs_smolvla+act     | graph  | act      | 1        |         4.5 |      8.2 | 1.19x      |                   183.25 |         2130 |
| procs-mps_smolvla+act | eager  | smolvla  | 1        |       627.2 |    630.3 | 1.03x      |                    62.92 |           19 |
| procs-mps_smolvla+act | eager  | act      | 1        |        16.2 |     17   | 1.03x      |                    62.92 |          736 |
| procs-mps_smolvla+act | graph  | smolvla  | 1        |       166.8 |    169.9 | 1.42x      |                   217.17 |           72 |
| procs-mps_smolvla+act | graph  | act      | 1        |         4.6 |      6.8 | 1.21x      |                   217.17 |         2534 |

`vs alone` is the slowdown against the same policy and mode hosted alone (procs x1). `steps/s` sums every hosted copy (times the batch for batch hosting). `steps/copy` is the number of timed steps behind the median; rows under 5 are starved copies (the GPU time-slicer or the GIL let one copy run for seconds) and are left out of the prediction tables.

## Predicted from the policies alone (closed queueing network, host + GPU)

| config                | hosting   | mode   | policy   |   alone ms |   model ms |   measured ms |   model err % |   no-interference err % |   pred steps/s |   meas steps/s |
|:----------------------|:----------|:-------|:---------|-----------:|-----------:|--------------:|--------------:|------------------------:|---------------:|---------------:|
| procs_smolvlax2       | procs     | eager  | smolvla  |      606.6 |      609.8 |         626.2 |          -2.6 |                    -3.1 |            3.3 |            3.2 |
| procs_smolvlax2       | procs     | graph  | smolvla  |      117.6 |      134.5 |         154.2 |         -12.8 |                   -23.8 |           14.9 |           12.8 |
| procs_smolvlax4       | procs     | eager  | smolvla  |      606.6 |      617.6 |         674.2 |          -8.4 |                   -10   |            6.5 |            5.9 |
| procs_smolvlax4       | procs     | graph  | smolvla  |      117.6 |      187.9 |         248.8 |         -24.5 |                   -52.7 |           21.3 |           16   |
| procs-mps_smolvlax2   | procs-mps | eager  | smolvla  |      606.6 |      609.8 |         643.9 |          -5.3 |                    -5.8 |            3.3 |            3   |
| procs-mps_smolvlax2   | procs-mps | graph  | smolvla  |      117.6 |      134.5 |         128.6 |           4.5 |                    -8.6 |           14.9 |           15.5 |
| procs-mps_smolvlax4   | procs-mps | eager  | smolvla  |      606.6 |      617.6 |         667.1 |          -7.4 |                    -9.1 |            6.5 |            6   |
| procs-mps_smolvlax4   | procs-mps | graph  | smolvla  |      117.6 |      187.9 |         162.5 |          15.6 |                   -27.7 |           21.3 |           24.7 |
| threads_smolvlax2     | threads   | eager  | smolvla  |      606.6 |     1150.9 |         824.6 |          39.6 |                   -26.4 |            1.7 |            2.5 |
| threads_smolvlax2     | threads   | graph  | smolvla  |      117.6 |      184.9 |         145.9 |          26.7 |                   -19.4 |           10.8 |           13.5 |
| threads_smolvlax4     | threads   | graph  | smolvla  |      117.6 |      329   |         660.1 |         -50.2 |                   -82.2 |           12.2 |            6.1 |
| procs_actx2           | procs     | eager  | act      |       15.7 |       16.6 |          16.6 |          -0.1 |                    -5.2 |          120.8 |          119.9 |
| procs_actx2           | procs     | graph  | act      |        3.8 |        6.8 |           7.8 |         -12.1 |                   -51.4 |          293.5 |          255.7 |
| procs_actx4           | procs     | eager  | act      |       15.7 |       19.3 |          18.8 |           3.1 |                   -16.3 |          206.8 |          212.4 |
| procs_actx4           | procs     | graph  | act      |        3.8 |       13.3 |          15.6 |         -15.2 |                   -75.9 |          301.9 |          254.3 |
| procs-mps_actx2       | procs-mps | eager  | act      |       15.7 |       16.6 |          15.9 |           3.8 |                    -1.5 |          120.8 |          124.2 |
| procs-mps_actx2       | procs-mps | graph  | act      |        3.8 |        6.8 |           6.9 |          -0.8 |                   -45.2 |          293.5 |          287.9 |
| procs-mps_actx4       | procs-mps | eager  | act      |       15.7 |       19.3 |          17.7 |           9.2 |                   -11.4 |          206.8 |          226.7 |
| procs-mps_actx4       | procs-mps | graph  | act      |        3.8 |       13.3 |          13.5 |          -1.8 |                   -72.1 |          301.9 |          294.3 |
| threads_actx2         | threads   | eager  | act      |       15.7 |       26.7 |          24   |          11.3 |                   -34.7 |           74.8 |           68.2 |
| threads_actx2         | threads   | graph  | act      |        3.8 |        6.8 |           6.8 |           0.7 |                   -44.5 |          292.6 |          291.8 |
| threads_actx4         | threads   | eager  | act      |       15.7 |       50.5 |         134   |         -62.3 |                   -88.3 |           79.2 |           29.8 |
| threads_actx4         | threads   | graph  | act      |        3.8 |       13.3 |          13.2 |           0.8 |                   -71.4 |          301.5 |          301.8 |
| procs_smolvla+act     | procs     | eager  | smolvla  |      606.6 |      616.2 |         616.4 |          -0   |                    -1.6 |            1.6 |            1.6 |
| procs_smolvla+act     | procs     | eager  | act      |       15.7 |       16   |          16.8 |          -4.7 |                    -6.4 |           62.6 |           59.2 |
| procs_smolvla+act     | procs     | graph  | smolvla  |      117.6 |      153.9 |         172.8 |         -10.9 |                   -31.9 |            6.5 |            5.8 |
| procs_smolvla+act     | procs     | graph  | act      |        3.8 |        5.4 |           4.5 |          20.4 |                   -15.9 |          185.5 |          177.5 |
| procs-mps_smolvla+act | procs-mps | eager  | smolvla  |      606.6 |      616.2 |         627.2 |          -1.7 |                    -3.3 |            1.6 |            1.6 |
| procs-mps_smolvla+act | procs-mps | eager  | act      |       15.7 |       16   |          16.2 |          -1.2 |                    -3   |           62.6 |           61.3 |
| procs-mps_smolvla+act | procs-mps | graph  | smolvla  |      117.6 |      153.9 |         166.8 |          -7.7 |                   -29.5 |            6.5 |            6   |
| procs-mps_smolvla+act | procs-mps | graph  | act      |        3.8 |        5.4 |           4.6 |          18.3 |                   -17.4 |          185.5 |          211.2 |

MAPE (%) of the per-policy latency:

|                        |   n |   model_MAPE |   no_interference_MAPE |
|:-----------------------|----:|-------------:|-----------------------:|
| ('procs', 'eager')     |   6 |          3.2 |                    7.1 |
| ('procs', 'graph')     |   6 |         16   |                   41.9 |
| ('procs-mps', 'eager') |   6 |          4.8 |                    5.7 |
| ('procs-mps', 'graph') |   6 |          8.1 |                   33.4 |
| ('threads', 'eager')   |   3 |         37.7 |                   49.8 |
| ('threads', 'graph')   |   4 |         19.6 |                   54.4 |

Reading: for `procs` the model assumes the GPU serializes the contexts' kernels; a measured latency below the model means kernels of different processes overlapped (expected under `procs-mps`, where the model is an upper bound). For `threads` the model also serializes the host side (GIL); PyTorch releases the GIL inside its C++ ops, so the measurement should sit between the two bounds.

## Correctness

Largest |diff| over all copies: split loop vs stock step 0.0e+00, graph vs eager 0.0e+00 (43 policy copies loaded).

MPS state recorded by the procs-mps runs: ['daemon up, servers: 2078451', 'daemon up, servers: 2095322', 'daemon up, servers: 2107983']
