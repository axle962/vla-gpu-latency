# Multi-policy hosting on one GPU

GPU `NVIDIA TITAN V` (driver 580.196.07), host `cuda3.cims.nyu.edu` (Intel(R) Xeon(R) Gold 5118 CPU @ 2.30GHz, 24 CPUs in the affinity mask), torch 2.10.0+cu128 cu12.8. Policy settings: {"cameras": 3, "num_steps": 10, "chunk_size": 50, "diffusion_steps": 10, "act_hw": [480, 640]}. Closed loop (a new step as soon as the previous one returned), median of the middle 80% of each window.

## Policies alone (procs x1)

| policy   | mode   |   step ms |   p95 ms |   GPU busy ms (profiled) |   GPU busy ms (corrected D) |   host ms (Z = step - D) |   GPU share |   kernels/step |
|:---------|:-------|----------:|---------:|-------------------------:|----------------------------:|-------------------------:|------------:|---------------:|
| smolvla  | eager  |     337.3 |    337.9 |                     92.6 |                        79.8 |                    257.5 |        0.24 |          13430 |
| smolvla  | graph  |     125.7 |    126.1 |                    107.9 |                        94.7 |                     31   |        0.75 |          13850 |
| act      | eager  |       8.8 |      8.9 |                      5.8 |                         5.4 |                      3.4 |        0.62 |            363 |
| act      | graph  |       6   |      6   |                      6.3 |                         6   |                     -0   |        1    |            365 |

D removes the profiler's 0.95 us per kernel (CLAUDE.md 0.1); Z is what the step spends off the GPU (Python, dispatch, syncs). A policy with a small GPU share leaves room for others.

## Measured: every hosted configuration

| config                | mode   | policy   | copies   |   median ms |   p95 ms | vs alone   |   steps/s (all policies) |   steps/copy |
|:----------------------|:-------|:---------|:---------|------------:|---------:|:-----------|-------------------------:|-------------:|
| procs_smolvla         | eager  | smolvla  | 1        |       337.3 |    337.9 | 1.00x      |                     2.92 |           35 |
| procs_smolvla         | graph  | smolvla  | 1        |       125.7 |    126.1 | 1.00x      |                     8    |           96 |
| procs_smolvlax2       | eager  | smolvla  | 2        |       357.4 |    359.8 | 1.06x      |                     5.58 |           33 |
| procs_smolvlax2       | graph  | smolvla  | 2        |       238.4 |    239.6 | 1.90x      |                     8.33 |           50 |
| procs_smolvlax4       | eager  | smolvla  | 4        |       476.2 |    476.8 | 1.41x      |                     8.33 |           25 |
| procs_smolvlax4       | graph  | smolvla  | 4        |       460.8 |    461.8 | 3.67x      |                     8.67 |           26 |
| procs-mps_smolvlax2   | eager  | smolvla  | 2        |       342.4 |    355.2 | 1.02x      |                     5.83 |           35 |
| procs-mps_smolvlax2   | graph  | smolvla  | 2        |       159.3 |    160   | 1.27x      |                    12.58 |           75 |
| procs-mps_smolvlax4   | eager  | smolvla  | 4        |       365.8 |    389   | 1.08x      |                    10.75 |           32 |
| procs-mps_smolvlax4   | graph  | smolvla  | 4        |       230.7 |    232   | 1.84x      |                    17.33 |           52 |
| threads_smolvlax2     | eager  | smolvla  | 2        |       537   |    539.3 | 1.59x      |                     3.75 |           22 |
| threads_smolvlax2     | graph  | smolvla  | 2        |       160.2 |    175   | 1.27x      |                    12.25 |           73 |
| threads_smolvlax4     | eager  | smolvla  | 4        |      1581.3 |   1609.2 | 4.69x      |                     2.33 |            7 |
| threads_smolvlax4     | graph  | smolvla  | 4        |       227.1 |    240.1 | 1.81x      |                    17.42 |           52 |
| procs_act             | eager  | act      | 1        |         8.8 |      8.9 | 1.00x      |                   113.25 |         1359 |
| procs_act             | graph  | act      | 1        |         6   |      6   | 1.00x      |                   167.33 |         2008 |
| procs_actx2           | eager  | act      | 2        |        13.7 |     15.6 | 1.55x      |                   147.83 |          887 |
| procs_actx2           | graph  | act      | 2        |        13.4 |     13.4 | 2.23x      |                   149.67 |          898 |
| procs_actx4           | eager  | act      | 4        |        26.6 |     26.8 | 3.01x      |                   150.33 |          451 |
| procs_actx4           | graph  | act      | 4        |        26.7 |     26.8 | 4.47x      |                   149.67 |          449 |
| procs-mps_actx2       | eager  | act      | 2        |         9.9 |     10.3 | 1.12x      |                   202    |         1212 |
| procs-mps_actx2       | graph  | act      | 2        |         9.8 |     10.2 | 1.65x      |                   203.42 |         1220 |
| procs-mps_actx4       | eager  | act      | 4        |        18.3 |     18.8 | 2.07x      |                   218.67 |          656 |
| procs-mps_actx4       | graph  | act      | 4        |        17.1 |     17.7 | 2.86x      |                   234    |          702 |
| threads_actx2         | eager  | act      | 2        |        15.2 |     15.6 | 1.72x      |                   131.75 |          790 |
| threads_actx2         | graph  | act      | 2        |         9.8 |     10.2 | 1.65x      |                   203.5  |         1221 |
| threads_actx4         | eager  | act      | 4        |        56   |     58.8 | 6.34x      |                    70.92 |          214 |
| threads_actx4         | graph  | act      | 4        |        16.8 |     17.3 | 2.81x      |                   238.58 |          716 |
| batch_smolvlax4_b4    | eager  | smolvla  | batch 4  |       396.2 |    396.9 | 1.17x      |                    10    |           30 |
| batch_smolvlax4_b4    | graph  | smolvla  | batch 4  |       213.8 |    214.6 | 1.70x      |                    18.67 |           56 |
| procs_smolvla+act     | eager  | smolvla  | 1        |       389.2 |    390.5 | 1.15x      |                   100.17 |           31 |
| procs_smolvla+act     | eager  | act      | 1        |         9.8 |     12.7 | 1.11x      |                   100.17 |         1171 |
| procs_smolvla+act     | graph  | smolvla  | 1        |       269   |    269.6 | 2.14x      |                    89.25 |           44 |
| procs_smolvla+act     | graph  | act      | 1        |        12   |     13.4 | 2.01x      |                    89.25 |         1027 |
| procs-mps_smolvla+act | eager  | smolvla  | 1        |       346   |    346.7 | 1.03x      |                   113.5  |           35 |
| procs-mps_smolvla+act | eager  | act      | 1        |         8.8 |     11.3 | 1.00x      |                   113.5  |         1327 |
| procs-mps_smolvla+act | graph  | smolvla  | 1        |       203.2 |    205   | 1.62x      |                   133.92 |           60 |
| procs-mps_smolvla+act | graph  | act      | 1        |         7.3 |     11.1 | 1.21x      |                   133.92 |         1547 |

`vs alone` is the slowdown against the same policy and mode hosted alone (procs x1). `steps/s` sums every hosted copy (times the batch for batch hosting). `steps/copy` is the number of timed steps behind the median; rows under 5 are starved copies (the GPU time-slicer or the GIL let one copy run for seconds) and are left out of the prediction tables.

## Predicted from the policies alone (closed queueing network, host + GPU)

| config                | hosting   | mode   | policy   |   alone ms |   model ms |   measured ms |   model err % |   no-interference err % |   pred steps/s |   meas steps/s |
|:----------------------|:----------|:-------|:---------|-----------:|-----------:|--------------:|--------------:|------------------------:|---------------:|---------------:|
| procs_smolvlax2       | procs     | eager  | smolvla  |      337.3 |      360.1 |         357.4 |           0.7 |                    -5.6 |            5.6 |            5.6 |
| procs_smolvlax2       | procs     | graph  | smolvla  |      125.7 |      206.2 |         238.4 |         -13.5 |                   -47.3 |            9.7 |            8.3 |
| procs_smolvlax4       | procs     | eager  | smolvla  |      337.3 |      435.1 |         476.2 |          -8.6 |                   -29.2 |            9.2 |            8.3 |
| procs_smolvlax4       | procs     | graph  | smolvla  |      125.7 |      387.1 |         460.8 |         -16   |                   -72.7 |           10.3 |            8.7 |
| procs-mps_smolvlax2   | procs-mps | eager  | smolvla  |      337.3 |      360.1 |         342.4 |           5.2 |                    -1.5 |            5.6 |            5.8 |
| procs-mps_smolvlax2   | procs-mps | graph  | smolvla  |      125.7 |      206.2 |         159.3 |          29.4 |                   -21.1 |            9.7 |           12.6 |
| procs-mps_smolvlax4   | procs-mps | eager  | smolvla  |      337.3 |      435.1 |         365.8 |          18.9 |                    -7.8 |            9.2 |           10.8 |
| procs-mps_smolvlax4   | procs-mps | graph  | smolvla  |      125.7 |      387.1 |         230.7 |          67.8 |                   -45.5 |           10.3 |           17.3 |
| threads_smolvlax2     | threads   | eager  | smolvla  |      337.3 |      565.6 |         537   |           5.3 |                   -37.2 |            3.5 |            3.8 |
| threads_smolvlax2     | threads   | graph  | smolvla  |      125.7 |      209.4 |         160.2 |          30.7 |                   -21.5 |            9.6 |           12.2 |
| threads_smolvlax4     | threads   | eager  | smolvla  |      337.3 |     1057.8 |        1581.3 |         -33.1 |                   -78.7 |            3.8 |            2.3 |
| threads_smolvlax4     | threads   | graph  | smolvla  |      125.7 |      389.9 |         227.1 |          71.7 |                   -44.6 |           10.3 |           17.4 |
| procs_actx2           | procs     | eager  | act      |        8.8 |       12.8 |          13.7 |          -6.1 |                   -35.5 |          155.7 |          147.8 |
| procs_actx2           | procs     | graph  | act      |        6   |       12   |          13.4 |         -10.1 |                   -55.2 |          166.7 |          149.7 |
| procs_actx4           | procs     | eager  | act      |        8.8 |       22.7 |          26.6 |         -14.5 |                   -66.8 |          175.9 |          150.3 |
| procs_actx4           | procs     | graph  | act      |        6   |       24   |          26.7 |         -10.2 |                   -77.6 |          166.7 |          149.7 |
| procs-mps_actx2       | procs-mps | eager  | act      |        8.8 |       12.8 |           9.9 |          30   |                   -10.6 |          155.7 |          202   |
| procs-mps_actx2       | procs-mps | graph  | act      |        6   |       12   |           9.8 |          22   |                   -39.2 |          166.7 |          203.4 |
| procs-mps_actx4       | procs-mps | eager  | act      |        8.8 |       22.7 |          18.3 |          24.4 |                   -51.7 |          175.9 |          218.7 |
| procs-mps_actx4       | procs-mps | graph  | act      |        6   |       24   |          17.1 |          40.4 |                   -65   |          166.7 |          234   |
| threads_actx2         | threads   | eager  | act      |        8.8 |       13.6 |          15.2 |         -10.4 |                   -41.8 |          147.2 |          131.8 |
| threads_actx2         | threads   | graph  | act      |        6   |       12   |           9.8 |          22.1 |                   -39.2 |          166.7 |          203.5 |
| threads_actx4         | threads   | eager  | act      |        8.8 |       23.6 |          56   |         -57.8 |                   -84.2 |          169.4 |           70.9 |
| threads_actx4         | threads   | graph  | act      |        6   |       24   |          16.8 |          43.1 |                   -64.4 |          166.7 |          238.6 |
| procs_smolvla+act     | procs     | eager  | smolvla  |      337.3 |      392   |         389.2 |           0.7 |                   -13.3 |            2.6 |            2.6 |
| procs_smolvla+act     | procs     | eager  | act      |        8.8 |       10.7 |           9.8 |           9.3 |                    -9.8 |           93.5 |           97.6 |
| procs_smolvla+act     | procs     | graph  | smolvla  |      125.7 |      220.4 |         269   |         -18.1 |                   -53.3 |            4.5 |            3.7 |
| procs_smolvla+act     | procs     | graph  | act      |        6   |       11.2 |          12   |          -7.3 |                   -50.4 |           89.7 |           85.6 |
| procs-mps_smolvla+act | procs-mps | eager  | smolvla  |      337.3 |      392   |         346   |          13.3 |                    -2.5 |            2.6 |            2.9 |
| procs-mps_smolvla+act | procs-mps | eager  | act      |        8.8 |       10.7 |           8.8 |          21.5 |                     0.2 |           93.5 |          110.6 |
| procs-mps_smolvla+act | procs-mps | graph  | smolvla  |      125.7 |      220.4 |         203.2 |           8.5 |                   -38.1 |            4.5 |            5   |
| procs-mps_smolvla+act | procs-mps | graph  | act      |        6   |       11.2 |           7.3 |          53.8 |                   -17.6 |           89.7 |          128.9 |

MAPE (%) of the per-policy latency:

|                        |   n |   model_MAPE |   no_interference_MAPE |
|:-----------------------|----:|-------------:|-----------------------:|
| ('procs', 'eager')     |   6 |          6.7 |                   26.7 |
| ('procs', 'graph')     |   6 |         12.5 |                   59.4 |
| ('procs-mps', 'eager') |   6 |         18.9 |                   12.4 |
| ('procs-mps', 'graph') |   6 |         37   |                   37.8 |
| ('threads', 'eager')   |   4 |         26.7 |                   60.5 |
| ('threads', 'graph')   |   4 |         41.9 |                   42.4 |

Reading: for `procs` the model assumes the GPU serializes the contexts' kernels; a measured latency below the model means kernels of different processes overlapped (expected under `procs-mps`, where the model is an upper bound). For `threads` the model also serializes the host side (GIL); PyTorch releases the GIL inside its C++ ops, so the measurement should sit between the two bounds.

## Correctness

Largest |diff| over all copies: split loop vs stock step 0.0e+00, graph vs eager 0.0e+00 (43 policy copies loaded).

MPS state recorded by the procs-mps runs: ['daemon up, servers: 970746', 'daemon up, servers: 972615', 'daemon up, servers: 974494']
