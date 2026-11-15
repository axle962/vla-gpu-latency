# Rodinia: cross-GPU kernel-time transfer (model/rodinia_predict.py)

GPUs: 2080Ti, RTX4070, TITANV. Error of total kernel time per benchmark, all ordered pairs.

## Measured total kernel time (ms) and median kernel duration

| bench                |   2080Ti |   RTX4070 |   TITANV |   median kernel (us, all GPUs) |
|:---------------------|---------:|----------:|---------:|-------------------------------:|
| backprop             |     2.21 |      1.68 |     0.79 |                         499.61 |
| gaussian             |    58.27 |     46.97 |   112.55 |                          23.48 |
| lavaMD               |    68.24 |     49.74 |     3.41 |                       49739.8  |
| lud                  |    10.08 |      5.22 |     7.73 |                          16.87 |
| nw                   |     3.54 |      2.49 |     3.53 |                           6.89 |
| particlefilter_float |   506.58 |    446.14 |   184.09 |                        7107.48 |
| pathfinder           |     0.14 |      0.1  |     0.1  |                          20.15 |
| streamcluster        |   210.55 |    245.91 |   193.26 |                         130.69 | 

## MAPE (%) per benchmark and transfer rule

| bench                |   same |   spec_bw |   spec_clock |   spec_mix |   meas_mix |
|:---------------------|-------:|----------:|-------------:|-----------:|-----------:|
| backprop             |   77.1 |      71   |        108.2 |       71.2 |       67.1 |
| gaussian             |   63.8 |      86.6 |         37.3 |       80.4 |       85.6 |
| lavaMD               |  585.1 |     517.8 |        795.2 |      517.8 |      493.6 |
| lud                  |   45.9 |      62.4 |         25.9 |       49.7 |       54   |
| nw                   |   24.1 |      42.9 |         12.9 |       11.5 |       20.8 |
| particlefilter_float |   77.6 |      70.3 |        121.8 |       70.3 |       66   |
| pathfinder           |   23.1 |      37.2 |         37.5 |       27.3 |       30.6 |
| streamcluster        |   16.2 |       3   |         56.2 |        2.4 |        2.5 |
| all                  |  114.1 |     111.4 |        149.4 |      103.8 |      102.5 | 

