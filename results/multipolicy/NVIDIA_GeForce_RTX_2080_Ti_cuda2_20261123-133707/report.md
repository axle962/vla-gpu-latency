# Multi-policy hosting on one GPU

GPU `NVIDIA GeForce RTX 2080 Ti` (driver 595.104.02), host `cuda2.cims.nyu.edu` (Intel(R) Xeon(R) CPU E5-2660 v3 @ 2.60GHz, 20 CPUs in the affinity mask), torch 2.10.0+cu128 cu12.8. Policy settings: {"cameras": 3, "num_steps": 10, "chunk_size": 50, "diffusion_steps": 10, "act_hw": [480, 640]}. Closed loop (a new step as soon as the previous one returned), median of the middle 80% of each window.

## Policies alone (procs x1)

| policy   | mode   |   step ms |   p95 ms |   GPU busy ms (profiled) |   GPU busy ms (corrected D) |   host ms (Z = step - D) |   GPU share |   kernels/step |
|:---------|:-------|----------:|---------:|-------------------------:|----------------------------:|-------------------------:|------------:|---------------:|
| smolvla  | eager  |     484.5 |    487.7 |                     89.9 |                        77.2 |                    407.3 |        0.16 |          13346 |
| smolvla  | graph  |     135.4 |    135.8 |                     96.5 |                        83.4 |                     52   |        0.62 |          13766 |
| act      | eager  |      12.3 |     13.1 |                      6.7 |                         6.4 |                      5.9 |        0.52 |            356 |
| act      | graph  |       6.8 |      6.9 |                      6.8 |                         6.5 |                      0.3 |        0.95 |            358 |

D removes the profiler's 0.95 us per kernel (CLAUDE.md 0.1); Z is what the step spends off the GPU (Python, dispatch, syncs). A policy with a small GPU share leaves room for others.

## Measured: every hosted configuration

| config                | mode   | policy   | copies   |   median ms |   p95 ms | vs alone   |   steps/s (all policies) |   steps/copy |
|:----------------------|:-------|:---------|:---------|------------:|---------:|:-----------|-------------------------:|-------------:|
| procs_smolvla         | eager  | smolvla  | 1        |       484.5 |    487.7 | 1.00x      |                     2.08 |           25 |
| procs_smolvla         | graph  | smolvla  | 1        |       135.4 |    135.8 | 1.00x      |                     7.42 |           89 |
| procs_smolvlax2       | eager  | smolvla  | 2        |       512.1 |    513.9 | 1.06x      |                     4    |           24 |
| procs_smolvlax2       | graph  | smolvla  | 2        |       241.2 |    241.9 | 1.78x      |                     8.33 |           50 |
| procs_smolvlax4       | eager  | smolvla  | 4        |       561.8 |    569.9 | 1.16x      |                     7.08 |           21 |
| procs_smolvlax4       | graph  | smolvla  | 4        |       440.2 |    440.9 | 3.25x      |                     9.33 |           28 |
| procs-mps_smolvlax2   | eager  | smolvla  | 2        |       481.2 |    493.3 | 0.99x      |                     4.08 |           24 |
| procs-mps_smolvlax2   | graph  | smolvla  | 2        |       161.7 |    162.6 | 1.19x      |                    12.42 |           74 |
| procs-mps_smolvlax4   | eager  | smolvla  | 4        |       494.5 |    506.4 | 1.02x      |                     7.67 |           24 |
| procs-mps_smolvlax4   | graph  | smolvla  | 4        |       230.9 |    234.5 | 1.71x      |                    17.17 |           52 |
| threads_smolvlax2     | eager  | smolvla  | 2        |       771.7 |    786.4 | 1.59x      |                     2.67 |           16 |
| threads_smolvlax2     | graph  | smolvla  | 2        |       186.3 |    191.7 | 1.38x      |                    10.67 |           64 |
| threads_smolvlax4     | eager  | smolvla  | 4        |      1487   |   1501.4 | 3.07x      |                     2.67 |            8 |
| threads_smolvlax4     | graph  | smolvla  | 4        |       246.1 |    256.7 | 1.82x      |                    16.08 |           48 |
| procs_act             | eager  | act      | 1        |        12.3 |     13.1 | 1.00x      |                    80.67 |          968 |
| procs_act             | graph  | act      | 1        |         6.8 |      6.9 | 1.00x      |                   146.5  |         1758 |
| procs_actx2           | eager  | act      | 2        |        15.6 |     16.3 | 1.27x      |                   127.75 |          766 |
| procs_actx2           | graph  | act      | 2        |        15.1 |     15.1 | 2.22x      |                   132.5  |          795 |
| procs_actx4           | eager  | act      | 4        |        30.7 |     30.7 | 2.49x      |                   130.67 |          392 |
| procs_actx4           | graph  | act      | 4        |        30.2 |     30.3 | 4.44x      |                   132.33 |          397 |
| procs-mps_actx2       | eager  | act      | 2        |        14.1 |     14.3 | 1.14x      |                   143.5  |          861 |
| procs-mps_actx2       | graph  | act      | 2        |        11.2 |     11.7 | 1.65x      |                   178.67 |         1072 |
| procs-mps_actx4       | eager  | act      | 4        |        20.4 |     21.2 | 1.66x      |                   196.5  |          589 |
| procs-mps_actx4       | graph  | act      | 4        |        19.7 |     20.6 | 2.90x      |                   203    |          609 |
| threads_actx2         | eager  | act      | 2        |        18.6 |     19   | 1.51x      |                   107.42 |          644 |
| threads_actx2         | graph  | act      | 2        |        11.1 |     11.6 | 1.64x      |                   179.83 |         1079 |
| threads_actx4         | eager  | act      | 4        |        47   |     50.8 | 3.82x      |                    84.33 |          254 |
| threads_actx4         | graph  | act      | 4        |        19.5 |     20.4 | 2.87x      |                   205.33 |          616 |
| batch_smolvlax4_b4    | eager  | smolvla  | batch 4  |       579.6 |    590.3 | 1.20x      |                     6.67 |           20 |
| batch_smolvlax4_b4    | graph  | smolvla  | batch 4  |       260.3 |    261.8 | 1.92x      |                    15.67 |           47 |
| procs_smolvla+act     | eager  | smolvla  | 1        |       553   |    556.4 | 1.14x      |                    76.92 |           22 |
| procs_smolvla+act     | eager  | act      | 1        |        12.9 |     15.9 | 1.05x      |                    76.92 |          901 |
| procs_smolvla+act     | graph  | smolvla  | 1        |       250.9 |    253.6 | 1.85x      |                    78.08 |           48 |
| procs_smolvla+act     | graph  | act      | 1        |        13.3 |     16.7 | 1.96x      |                    78.08 |          889 |
| procs-mps_smolvla+act | eager  | smolvla  | 1        |       491.2 |    492.6 | 1.01x      |                    66.08 |           24 |
| procs-mps_smolvla+act | eager  | act      | 1        |        15.6 |     15.7 | 1.27x      |                    66.08 |          769 |
| procs-mps_smolvla+act | graph  | smolvla  | 1        |       210.8 |    213.5 | 1.56x      |                   118.25 |           57 |
| procs-mps_smolvla+act | graph  | act      | 1        |         8.5 |     11.2 | 1.25x      |                   118.25 |         1362 |

`vs alone` is the slowdown against the same policy and mode hosted alone (procs x1). `steps/s` sums every hosted copy (times the batch for batch hosting). `steps/copy` is the number of timed steps behind the median; rows under 5 are starved copies (the GPU time-slicer or the GIL let one copy run for seconds) and are left out of the prediction tables.

## Predicted from the policies alone (closed queueing network, host + GPU)

| config                | hosting   | mode   | policy   |   alone ms |   model ms |   measured ms |   model err % |   no-interference err % |   pred steps/s |   meas steps/s |
|:----------------------|:----------|:-------|:---------|-----------:|-----------:|--------------:|--------------:|------------------------:|---------------:|---------------:|
| procs_smolvlax2       | procs     | eager  | smolvla  |      484.5 |      498.6 |         512.1 |          -2.6 |                    -5.4 |            4   |            4   |
| procs_smolvlax2       | procs     | graph  | smolvla  |      135.4 |      196.8 |         241.2 |         -18.4 |                   -43.9 |           10.2 |            8.3 |
| procs_smolvlax4       | procs     | eager  | smolvla  |      484.5 |      542.1 |         561.8 |          -3.5 |                   -13.8 |            7.4 |            7.1 |
| procs_smolvlax4       | procs     | graph  | smolvla  |      135.4 |      348.3 |         440.2 |         -20.9 |                   -69.2 |           11.5 |            9.3 |
| procs-mps_smolvlax2   | procs-mps | eager  | smolvla  |      484.5 |      498.6 |         481.2 |           3.6 |                     0.7 |            4   |            4.1 |
| procs-mps_smolvlax2   | procs-mps | graph  | smolvla  |      135.4 |      196.8 |         161.7 |          21.7 |                   -16.3 |           10.2 |           12.4 |
| procs-mps_smolvlax4   | procs-mps | eager  | smolvla  |      484.5 |      542.1 |         494.5 |           9.6 |                    -2   |            7.4 |            7.7 |
| procs-mps_smolvlax4   | procs-mps | graph  | smolvla  |      135.4 |      348.3 |         230.9 |          50.9 |                   -41.4 |           11.5 |           17.2 |
| threads_smolvlax2     | threads   | eager  | smolvla  |      484.5 |      859.2 |         771.7 |          11.3 |                   -37.2 |            2.3 |            2.7 |
| threads_smolvlax2     | threads   | graph  | smolvla  |      135.4 |      208.4 |         186.3 |          11.8 |                   -27.3 |            9.6 |           10.7 |
| threads_smolvlax4     | threads   | eager  | smolvla  |      484.5 |     1652.7 |        1487   |          11.1 |                   -67.4 |            2.4 |            2.7 |
| threads_smolvlax4     | threads   | graph  | smolvla  |      135.4 |      361.9 |         246.1 |          47   |                   -45   |           11.1 |           16.1 |
| procs_actx2           | procs     | eager  | act      |       12.3 |       16.4 |          15.6 |           5.2 |                   -21.1 |          122   |          127.8 |
| procs_actx2           | procs     | graph  | act      |        6.8 |       13.1 |          15.1 |         -13.2 |                   -54.9 |          152.6 |          132.5 |
| procs_actx4           | procs     | eager  | act      |       12.3 |       27.3 |          30.7 |         -10.8 |                   -59.9 |          146.3 |          130.7 |
| procs_actx4           | procs     | graph  | act      |        6.8 |       26   |          30.2 |         -14   |                   -77.5 |          154.1 |          132.3 |
| procs-mps_actx2       | procs-mps | eager  | act      |       12.3 |       16.4 |          14.1 |          16.5 |                   -12.6 |          122   |          143.5 |
| procs-mps_actx2       | procs-mps | graph  | act      |        6.8 |       13.1 |          11.2 |          17   |                   -39.3 |          152.6 |          178.7 |
| procs-mps_actx4       | procs-mps | eager  | act      |       12.3 |       27.3 |          20.4 |          34.3 |                   -39.6 |          146.3 |          196.5 |
| procs-mps_actx4       | procs-mps | graph  | act      |        6.8 |       26   |          19.7 |          31.8 |                   -65.5 |          154.1 |          203   |
| threads_actx2         | threads   | eager  | act      |       12.3 |       18.5 |          18.6 |          -0.9 |                   -34   |          108.4 |          107.4 |
| threads_actx2         | threads   | graph  | act      |        6.8 |       13.1 |          11.1 |          17.9 |                   -38.8 |          152.6 |          179.8 |
| threads_actx4         | threads   | eager  | act      |       12.3 |       30.8 |          47   |         -34.4 |                   -73.8 |          129.8 |           84.3 |
| threads_actx4         | threads   | graph  | act      |        6.8 |       26   |          19.5 |          33.2 |                   -65.1 |          154.1 |          205.3 |
| procs_smolvla+act     | procs     | eager  | smolvla  |      484.5 |      528.6 |         553   |          -4.4 |                   -12.4 |            1.9 |            1.8 |
| procs_smolvla+act     | procs     | eager  | act      |       12.3 |       13.8 |          12.9 |           6.7 |                    -4.7 |           72.7 |           75.1 |
| procs_smolvla+act     | procs     | graph  | smolvla  |      135.4 |      216.4 |         250.9 |         -13.8 |                   -46   |            4.6 |            4   |
| procs_smolvla+act     | procs     | graph  | act      |        6.8 |       11.7 |          13.3 |         -12.2 |                   -49   |           85.3 |           74.1 |
| procs-mps_smolvla+act | procs-mps | eager  | smolvla  |      484.5 |      528.6 |         491.2 |           7.6 |                    -1.4 |            1.9 |            2   |
| procs-mps_smolvla+act | procs-mps | eager  | act      |       12.3 |       13.8 |          15.6 |         -11.7 |                   -21.1 |           72.7 |           64.1 |
| procs-mps_smolvla+act | procs-mps | graph  | smolvla  |      135.4 |      216.4 |         210.8 |           2.7 |                   -35.8 |            4.6 |            4.8 |
| procs-mps_smolvla+act | procs-mps | graph  | act      |        6.8 |       11.7 |           8.5 |          38.2 |                   -19.8 |           85.3 |          113.5 |

MAPE (%) of the per-policy latency:

|                        |   n |   model_MAPE |   no_interference_MAPE |
|:-----------------------|----:|-------------:|-----------------------:|
| ('procs', 'eager')     |   6 |          5.5 |                   19.5 |
| ('procs', 'graph')     |   6 |         15.4 |                   56.8 |
| ('procs-mps', 'eager') |   6 |         13.9 |                   12.9 |
| ('procs-mps', 'graph') |   6 |         27   |                   36.3 |
| ('threads', 'eager')   |   4 |         14.5 |                   53.1 |
| ('threads', 'graph')   |   4 |         27.5 |                   44.1 |

Reading: for `procs` the model assumes the GPU serializes the contexts' kernels; a measured latency below the model means kernels of different processes overlapped (expected under `procs-mps`, where the model is an upper bound). For `threads` the model also serializes the host side (GIL); PyTorch releases the GIL inside its C++ ops, so the measurement should sit between the two bounds.

## Correctness

Largest |diff| over all copies: split loop vs stock step 0.0e+00, graph vs eager 0.0e+00 (43 policy copies loaded).

MPS state recorded by the procs-mps runs: ['daemon up, servers: 3700634', 'daemon up, servers: 3706265', 'daemon up, servers: 3710445']
