# Multi-policy hosting on one GPU

GPU `NVIDIA RTX PRO 6000 Blackwell Server Edition` (driver 580.173.02), host `alphagpu54` (Intel(R) Xeon(R) 6960P, 16 CPUs in the affinity mask), torch 2.10.0+cu128 cu12.8. Policy settings: {"cameras": 3, "num_steps": 10, "chunk_size": 50, "diffusion_steps": 10, "act_hw": [480, 640]}. Closed loop (a new step as soon as the previous one returned), median of the middle 80% of each window.

## Policies alone (procs x1)

| policy   | mode   |   step ms |   p95 ms |   GPU busy ms (profiled) |   GPU busy ms (corrected D) |   host ms (Z = step - D) |   GPU share |   kernels/step |
|:---------|:-------|----------:|---------:|-------------------------:|----------------------------:|-------------------------:|------------:|---------------:|
| smolvla  | eager  |     180.9 |    182.3 |                     41   |                        28.1 |                    152.8 |        0.16 |          13542 |
| smolvla  | graph  |      61.3 |     61.6 |                     40.6 |                        27.4 |                     33.9 |        0.45 |          13962 |
| act      | eager  |       5   |      5   |                      5.3 |                         4.9 |                      0   |        0.99 |            373 |
| act      | graph  |       4.2 |      4.3 |                      5.3 |                         5   |                     -0.8 |        1.19 |            375 |

D removes the profiler's 0.95 us per kernel (CLAUDE.md 0.1); Z is what the step spends off the GPU (Python, dispatch, syncs). A policy with a small GPU share leaves room for others.

## Measured: every hosted configuration

| config                | mode   | policy   | copies   |   median ms |   p95 ms | vs alone   |   steps/s (all policies) |
|:----------------------|:-------|:---------|:---------|------------:|---------:|:-----------|-------------------------:|
| procs_smolvla         | eager  | smolvla  | 1        |       180.9 |    182.3 | 1.00x      |                     5.58 |
| procs_smolvla         | graph  | smolvla  | 1        |        61.3 |     61.6 | 1.00x      |                    16.33 |
| procs_smolvlax2       | eager  | smolvla  | 2        |       181.1 |    183   | 1.00x      |                    11.08 |
| procs_smolvlax2       | graph  | smolvla  | 2        |       100   |    101.4 | 1.63x      |                    20    |
| procs_smolvlax4       | eager  | smolvla  | 4        |       268.2 |    271.6 | 1.48x      |                    15    |
| procs_smolvlax4       | graph  | smolvla  | 4        |       191.1 |    195.1 | 3.12x      |                    20.67 |
| procs-mps_smolvlax2   | eager  | smolvla  | 2        |       176.7 |    178.9 | 0.98x      |                    11.42 |
| procs-mps_smolvlax2   | graph  | smolvla  | 2        |        71.4 |     71.9 | 1.17x      |                    28.08 |
| procs-mps_smolvlax4   | eager  | smolvla  | 4        |       168.6 |    179.9 | 0.93x      |                    23.75 |
| procs-mps_smolvlax4   | graph  | smolvla  | 4        |        86.2 |     87.2 | 1.41x      |                    46.5  |
| threads_smolvlax2     | eager  | smolvla  | 2        |       384.6 |    485.2 | 2.13x      |                     5    |
| threads_smolvlax2     | graph  | smolvla  | 2        |        86.6 |    104.2 | 1.41x      |                    23.25 |
| threads_smolvlax4     | eager  | smolvla  | 4        |      1492.8 |   1643.6 | 8.25x      |                     2.67 |
| threads_smolvlax4     | graph  | smolvla  | 4        |       165.8 |    218.7 | 2.71x      |                    24.75 |
| procs_act             | eager  | act      | 1        |         5   |      5   | 1.00x      |                   200.17 |
| procs_act             | graph  | act      | 1        |         4.2 |      4.3 | 1.00x      |                   237.67 |
| procs_actx2           | eager  | act      | 2        |        10.1 |     10.4 | 2.03x      |                   197.83 |
| procs_actx2           | graph  | act      | 2        |         8.6 |      8.8 | 2.05x      |                   232.83 |
| procs_actx4           | eager  | act      | 4        |        20.3 |     20.4 | 4.08x      |                   196.67 |
| procs_actx4           | graph  | act      | 4        |        17.1 |     17.2 | 4.10x      |                   233.42 |
| procs-mps_actx2       | eager  | act      | 2        |         5.8 |      6   | 1.16x      |                   344.58 |
| procs-mps_actx2       | graph  | act      | 2        |         5.4 |      5.7 | 1.29x      |                   369    |
| procs-mps_actx4       | eager  | act      | 4        |         8.3 |      8.7 | 1.67x      |                   481.67 |
| procs-mps_actx4       | graph  | act      | 4        |         8   |      8.3 | 1.91x      |                   502    |
| threads_actx2         | eager  | act      | 2        |        10.7 |     16.6 | 2.14x      |                   175.5  |
| threads_actx2         | graph  | act      | 2        |         5.2 |      5.4 | 1.24x      |                   385    |
| threads_actx4         | eager  | act      | 4        |        39.9 |     62.2 | 8.01x      |                    93.92 |
| threads_actx4         | graph  | act      | 4        |         7.8 |      8.1 | 1.86x      |                   515.67 |
| batch_smolvlax4_b4    | eager  | smolvla  | batch 4  |       191.9 |    193   | 1.06x      |                    21    |
| batch_smolvlax4_b4    | graph  | smolvla  | batch 4  |        79.8 |     80   | 1.30x      |                    50    |
| procs_smolvla+act     | eager  | smolvla  | 1        |       207.2 |    208.1 | 1.15x      |                   135.08 |
| procs_smolvla+act     | eager  | act      | 1        |         7.8 |      8.7 | 1.56x      |                   135.08 |
| procs_smolvla+act     | graph  | smolvla  | 1        |       131.3 |    131.6 | 2.14x      |                   152.33 |
| procs_smolvla+act     | graph  | act      | 1        |         6.7 |      8.8 | 1.61x      |                   152.33 |
| procs-mps_smolvla+act | eager  | smolvla  | 1        |       161.4 |    173.5 | 0.89x      |                   192    |
| procs-mps_smolvla+act | eager  | act      | 1        |         5.3 |      6.3 | 1.06x      |                   192    |
| procs-mps_smolvla+act | graph  | smolvla  | 1        |        79.3 |     81   | 1.29x      |                   225.42 |
| procs-mps_smolvla+act | graph  | act      | 1        |         4.6 |      5.4 | 1.10x      |                   225.42 |

`vs alone` is the slowdown against the same policy and mode hosted alone (procs x1). `steps/s` sums every hosted copy (times the batch for batch hosting).

## Predicted from the policies alone (closed queueing network, host + GPU)

| config                | hosting   | mode   | policy   |   alone ms |   model ms |   measured ms |   model err % |   no-interference err % |   pred steps/s |   meas steps/s |
|:----------------------|:----------|:-------|:---------|-----------:|-----------:|--------------:|--------------:|------------------------:|---------------:|---------------:|
| procs_smolvlax2       | procs     | eager  | smolvla  |      180.9 |      185.9 |         181.1 |           2.6 |                    -0.2 |           10.8 |           11.1 |
| procs_smolvlax2       | procs     | graph  | smolvla  |       61.3 |       76.5 |         100   |         -23.5 |                   -38.7 |           26.1 |           20   |
| procs_smolvlax4       | procs     | eager  | smolvla  |      180.9 |      201.1 |         268.2 |         -25   |                   -32.6 |           19.9 |           15   |
| procs_smolvlax4       | procs     | graph  | smolvla  |       61.3 |      120.2 |         191.1 |         -37.1 |                   -67.9 |           33.3 |           20.7 |
| procs-mps_smolvlax2   | procs-mps | eager  | smolvla  |      180.9 |      185.9 |         176.7 |           5.2 |                     2.3 |           10.8 |           11.4 |
| procs-mps_smolvlax2   | procs-mps | graph  | smolvla  |       61.3 |       76.5 |          71.4 |           7.1 |                   -14.2 |           26.1 |           28.1 |
| procs-mps_smolvlax4   | procs-mps | eager  | smolvla  |      180.9 |      201.1 |         168.6 |          19.3 |                     7.3 |           19.9 |           23.8 |
| procs-mps_smolvlax4   | procs-mps | graph  | smolvla  |       61.3 |      120.2 |          86.2 |          39.4 |                   -28.9 |           33.3 |           46.5 |
| threads_smolvlax2     | threads   | eager  | smolvla  |      180.9 |      321.7 |         384.6 |         -16.4 |                   -53   |            6.2 |            5   |
| threads_smolvlax2     | threads   | graph  | smolvla  |       61.3 |       92.4 |          86.6 |           6.8 |                   -29.2 |           21.6 |           23.2 |
| threads_smolvlax4     | threads   | eager  | smolvla  |      180.9 |      619.5 |        1492.8 |         -58.5 |                   -87.9 |            6.5 |            2.7 |
| threads_smolvlax4     | threads   | graph  | smolvla  |       61.3 |      155.7 |         165.8 |          -6.1 |                   -63.1 |           25.7 |           24.8 |
| procs_actx2           | procs     | eager  | act      |        5   |        9.9 |          10.1 |          -2.1 |                   -50.7 |          202.1 |          197.8 |
| procs_actx2           | procs     | graph  | act      |        4.2 |        9.9 |           8.6 |          15.7 |                   -51.3 |          201.6 |          232.8 |
| procs_actx4           | procs     | eager  | act      |        5   |       19.8 |          20.3 |          -2.8 |                   -75.5 |          202.4 |          196.7 |
| procs_actx4           | procs     | graph  | act      |        4.2 |       19.8 |          17.1 |          15.9 |                   -75.6 |          201.6 |          233.4 |
| procs-mps_actx2       | procs-mps | eager  | act      |        5   |        9.9 |           5.8 |          70.7 |                   -14   |          202.1 |          344.6 |
| procs-mps_actx2       | procs-mps | graph  | act      |        4.2 |        9.9 |           5.4 |          83.7 |                   -22.7 |          201.6 |          369   |
| procs-mps_actx4       | procs-mps | eager  | act      |        5   |       19.8 |           8.3 |         138.2 |                   -39.9 |          202.4 |          481.7 |
| procs-mps_actx4       | procs-mps | graph  | act      |        4.2 |       19.8 |           8   |         149.1 |                   -47.6 |          201.6 |          502   |
| threads_actx2         | threads   | eager  | act      |        5   |        9.9 |          10.7 |          -7.3 |                   -53.3 |          202.1 |          175.5 |
| threads_actx2         | threads   | graph  | act      |        4.2 |        9.9 |           5.2 |          91.2 |                   -19.6 |          201.6 |          385   |
| threads_actx4         | threads   | eager  | act      |        5   |       19.8 |          39.9 |         -50.5 |                   -87.5 |          202.4 |           93.9 |
| threads_actx4         | threads   | graph  | act      |        4.2 |       19.8 |           7.8 |         155.9 |                   -46.2 |          201.6 |          515.7 |
| procs_smolvla+act     | procs     | eager  | smolvla  |      180.9 |      208.7 |         207.2 |           0.8 |                   -12.7 |            4.8 |            4.8 |
| procs_smolvla+act     | procs     | eager  | act      |        5   |        6.3 |           7.8 |         -18.7 |                   -35.8 |          158.6 |          130.2 |
| procs_smolvla+act     | procs     | graph  | smolvla  |       61.3 |       88.6 |         131.3 |         -32.5 |                   -53.3 |           11.3 |            7.7 |
| procs_smolvla+act     | procs     | graph  | act      |        4.2 |        8   |           6.7 |          19.2 |                   -38   |          124.6 |          144.7 |
| procs-mps_smolvla+act | procs-mps | eager  | smolvla  |      180.9 |      208.7 |         161.4 |          29.4 |                    12.1 |            4.8 |            6.1 |
| procs-mps_smolvla+act | procs-mps | eager  | act      |        5   |        6.3 |           5.3 |          19.8 |                    -5.3 |          158.6 |          185.9 |
| procs-mps_smolvla+act | procs-mps | graph  | smolvla  |       61.3 |       88.6 |          79.3 |          11.7 |                   -22.8 |           11.3 |           12.7 |
| procs-mps_smolvla+act | procs-mps | graph  | act      |        4.2 |        8   |           4.6 |          74.5 |                    -9.2 |          124.6 |          212.8 |

MAPE (%) of the per-policy latency:

|                        |   n |   model_MAPE |   no_interference_MAPE |
|:-----------------------|----:|-------------:|-----------------------:|
| ('procs', 'eager')     |   6 |          8.7 |                   34.6 |
| ('procs', 'graph')     |   6 |         24   |                   54.2 |
| ('procs-mps', 'eager') |   6 |         47.1 |                   13.5 |
| ('procs-mps', 'graph') |   6 |         60.9 |                   24.2 |
| ('threads', 'eager')   |   4 |         33.2 |                   70.4 |
| ('threads', 'graph')   |   4 |         65   |                   39.5 |

Reading: for `procs` the model assumes the GPU serializes the contexts' kernels; a measured latency below the model means kernels of different processes overlapped (expected under `procs-mps`, where the model is an upper bound). For `threads` the model also serializes the host side (GIL); PyTorch releases the GIL inside its C++ ops, so the measurement should sit between the two bounds.

## Correctness

Largest |diff| over all copies: split loop vs stock step 0.0e+00, graph vs eager 0.0e+00 (43 policy copies loaded).

MPS state recorded by the procs-mps runs: ['daemon up, servers: 3064250', 'daemon up, servers: 3073866', 'daemon up, servers: 3078583']
