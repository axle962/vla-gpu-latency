# SmolVLA batch sweep (model/batch_predict.py)

b and p fitted on each GPU's batch-1 baseline, held fixed for all batches; cross-GPU uses slot_drv.

## Measured (ms; speedup = eager loop / graph loop)

|                 |   eager |   eager_loop |   graph_loop |   hybrid |   speedup_loop |
|:----------------|--------:|-------------:|-------------:|---------:|---------------:|
| ('2080Ti', 1)   |   484.4 |        403.5 |         39.8 |    115   |           10.1 |
| ('2080Ti', 2)   |   518.5 |        415.2 |         49.2 |    141.8 |            8.4 |
| ('2080Ti', 4)   |   545.8 |        409.3 |         65.6 |    203.2 |            6.2 |
| ('2080Ti', 8)   |   614.5 |        403.2 |        111.8 |    325.6 |            3.6 |
| ('2080Ti', 16)  |   800.6 |        390.7 |        184.2 |    601.7 |            2.1 |
| ('RTX4070', 1)  |   607   |        510.2 |         22.8 |    115.9 |           22.4 |
| ('RTX4070', 2)  |   636   |        527.7 |         30.7 |    132   |           17.2 |
| ('RTX4070', 4)  |   659.9 |        523.3 |         45.9 |    182   |           11.4 |
| ('RTX4070', 8)  |   742.8 |        530.6 |         72.5 |    283.5 |            7.3 |
| ('RTX4070', 16) |   877.9 |        510.2 |        134   |    504.1 |            3.8 |
| ('TITANV', 1)   |   340.5 |        281.5 |         61.4 |    118.9 |            4.6 |
| ('TITANV', 2)   |   365.6 |        287.7 |         75.8 |    152.7 |            3.8 |
| ('TITANV', 4)   |   397   |        284   |         88.8 |    200.6 |            3.2 |
| ('TITANV', 8)   |   473.4 |        288.3 |        122.4 |    305.9 |            2.4 |
| ('TITANV', 16)  |   628.7 |        278.9 |        184   |    533.1 |            1.5 | 

## MAPE (%) by batch

|                   |   eager |   eager_loop |   graph_loop |   hybrid |   speedup_loop |
|:------------------|--------:|-------------:|-------------:|---------:|---------------:|
| ('cross-GPU', 1)  |    21.9 |         23.7 |         18.4 |     13   |           17.2 |
| ('cross-GPU', 2)  |    20.7 |         24.1 |         12.8 |     11.4 |           22.4 |
| ('cross-GPU', 4)  |    18.1 |         25.8 |          5.7 |     15.7 |           29.8 |
| ('cross-GPU', 8)  |    17.8 |         24.1 |         13.7 |     29.4 |           36.3 |
| ('cross-GPU', 16) |    26.7 |         20.5 |         27   |     42.1 |           47   |
| ('same-GPU', 1)   |     0   |          1.3 |          0   |      2.4 |            1.3 |
| ('same-GPU', 2)   |     1.8 |          3.4 |          0.7 |      1   |            2.7 |
| ('same-GPU', 4)   |     4.9 |          5.9 |          1   |      1.8 |            4.9 |
| ('same-GPU', 8)   |     2.9 |          3.3 |          1.6 |      2.1 |            2.5 |
| ('same-GPU', 16)  |     4.3 |          5   |          1.1 |      2.2 |            3.9 | 

## Batch size where the loop's graph speedup falls below 2x

| GPU     |   measured |   pred. from 2080Ti |   pred. from TITANV |   pred. from RTX4070 |
|:--------|-----------:|--------------------:|--------------------:|---------------------:|
| 2080Ti  |      nan   |               nan   |               nan   |                 10.2 |
| TITANV  |       10.3 |                 7.9 |                10.8 |                  4.8 |
| RTX4070 |      nan   |               nan   |               nan   |                nan   | 

nan: not crossed within the measured batches (speedup still >= 2 at the largest).

