Workload: `diffusion`

## MAPE (%) over cross-GPU pairs and configs

| quantity     |   same |   spec_bw |   bw |   lat |   lat_bw |   slot_class |   slot |   slot_cold |   slot_cold_py |   clock |   slot_cpp |   slot_drv |
|:-------------|-------:|----------:|-----:|------:|---------:|-------------:|-------:|------------:|---------------:|--------:|-----------:|-----------:|
| eager        |   43.4 |      60.1 | 59.5 |  78   |     57.5 |         38.6 |   54.1 |        38.5 |           41.1 |    97.5 |       84.9 |       39.8 |
| eager_prefix |   34.3 |      37.8 | 37.5 |  33.4 |     36.6 |         34.6 |   34.9 |        34.6 |           22.5 |    32.8 |       34.1 |       29.5 |
| eager_loop   |   43.8 |      60.5 | 59.9 |  78.8 |     57.9 |         39   |   54.5 |        39   |           41.5 |    98.4 |       85.7 |       40.2 |
| graph_loop   |   48.6 |      65.6 | 64.6 |  88.6 |     62.3 |         41.9 |   57.4 |        41.8 |           41.8 |   110.2 |       95   |       41.9 |
| hybrid       |   48   |      64.9 | 64   |  87.6 |     61.6 |         41.3 |   56.7 |        41.2 |           41.3 |   108.9 |       93.9 |       41.3 |
| speedup_loop |   11.6 |      12.2 | 11.2 |  31.2 |     10   |          6   |    6   |         5.9 |            1   |    43   |       31.8 |        3.6 |
| speedup_e2e  |   11.5 |      12.1 | 11.2 |  30.3 |     10   |          6   |    6   |         6   |            1.2 |    41.4 |       30.8 |        3.7 | 

## MAPE (%) per pair, variant `slot_drv`

|                       |   eager |   eager_prefix |   eager_loop |   graph_loop |   hybrid |   speedup_loop |   speedup_e2e |
|:----------------------|--------:|---------------:|-------------:|-------------:|---------:|---------------:|--------------:|
| ('2080Ti', 'A100')    |     6.8 |           11.4 |          6.8 |          7.9 |      7.9 |            1.2 |           1.2 |
| ('2080Ti', 'RTX4070') |    47.2 |           68.8 |         47.8 |         53.2 |     52.4 |           11.4 |          11.2 |
| ('2080Ti', 'TITANV')  |    16.4 |           18.1 |         16.7 |         18.2 |     18   |            1.8 |           2   |
| ('A100', '2080Ti')    |     9.5 |           13.4 |          9.1 |          8.6 |      8.6 |            0.4 |           0.8 |
| ('A100', 'RTX4070')   |    47.8 |           65.9 |         48.4 |         49.1 |     48.4 |            1.2 |           1.2 |
| ('A100', 'TITANV')    |     9.6 |           15   |          9.8 |         11.2 |     11.1 |            1.6 |           1.7 |
| ('RTX4070', '2080Ti') |   111   |           30.5 |        111.9 |        111.6 |    109.7 |            0.2 |           0.6 |
| ('RTX4070', 'A100')   |    91.4 |           28.7 |         92.6 |         94.4 |     93   |            1   |           0.8 |
| ('RTX4070', 'TITANV') |    71.3 |           23.4 |         71.9 |         70.6 |     69.6 |            0.8 |           1   |
| ('TITANV', '2080Ti')  |    23.6 |            8.8 |         23.4 |         22.9 |     22.5 |            0.4 |           0.9 |
| ('TITANV', 'A100')    |    14.4 |            8.3 |         14.6 |         13.2 |     12.9 |            1.3 |           1.4 |
| ('TITANV', 'RTX4070') |    29.1 |           61.8 |         29.7 |         42.3 |     41.6 |           21.8 |          21.5 | 

## Same-GPU baseline (source == target), variant-independent

| source   |   eager |   eager_prefix |   eager_loop |   graph_loop |   hybrid |   speedup_loop |   speedup_e2e |
|:---------|--------:|---------------:|-------------:|-------------:|---------:|---------------:|--------------:|
| 2080Ti   |     1   |           13.8 |          0.4 |          0   |      0.1 |            0.4 |           0.9 |
| A100     |     0.6 |            9.7 |          0.7 |          0   |      0.1 |            0.6 |           0.8 |
| RTX4070  |     0.6 |           16.6 |          0.4 |          0.2 |      0.3 |            0.2 |           0.2 |
| TITANV   |     2.2 |            9.3 |          2   |          0.2 |      0.1 |            1.9 |           2   | 

## GPU ordering (all GPUs ranked per source, config, quantity)

| variant      |   cases |   exact_ranking_pct |   mean_kendall_tau |
|:-------------|--------:|--------------------:|-------------------:|
| same         |     112 |              17.857 |              0.429 |
| spec_bw      |     112 |              22.321 |              0.565 |
| bw           |     112 |              19.643 |              0.558 |
| lat          |     112 |              14.286 |              0.116 |
| lat_bw       |     112 |              21.429 |              0.56  |
| slot_class   |     112 |              22.321 |              0.732 |
| slot         |     112 |              13.393 |              0.512 |
| slot_cold    |     112 |              14.286 |              0.705 |
| slot_cold_py |     112 |              13.393 |              0.704 |
| clock        |     112 |              11.607 |              0.064 |
| slot_cpp     |     112 |               8.929 |             -0.135 |
| slot_drv     |     112 |              25     |              0.747 | 

## Eager vs graph ordering inversion, baseline config (steps100_h16)

measured (ms): {'eager': {'2080Ti': 1485.3, 'A100': 1598.0, 'RTX4070': 3401.1, 'TITANV': 1448.3}, 'graph_loop': {'2080Ti': 1491.4, 'A100': 1555.0, 'RTX4070': 3384.8, 'TITANV': 1446.1}}
- spec_bw, source 2080Ti: eager order ['A100', 'TITANV', '2080Ti', 'RTX4070'], graph order ['A100', 'TITANV', '2080Ti', 'RTX4070'], differs from measured
- spec_bw, source A100: eager order ['A100', 'TITANV', '2080Ti', 'RTX4070'], graph order ['A100', 'TITANV', '2080Ti', 'RTX4070'], differs from measured
- spec_bw, source RTX4070: eager order ['A100', 'TITANV', '2080Ti', 'RTX4070'], graph order ['A100', 'TITANV', '2080Ti', 'RTX4070'], differs from measured
- spec_bw, source TITANV: eager order ['A100', 'TITANV', '2080Ti', 'RTX4070'], graph order ['A100', 'TITANV', '2080Ti', 'RTX4070'], differs from measured
- slot_drv, source 2080Ti: eager order ['TITANV', '2080Ti', 'A100', 'RTX4070'], graph order ['TITANV', 'A100', '2080Ti', 'RTX4070'], differs from measured
- slot_drv, source A100: eager order ['TITANV', 'A100', '2080Ti', 'RTX4070'], graph order ['TITANV', 'A100', '2080Ti', 'RTX4070'], differs from measured
- slot_drv, source RTX4070: eager order ['TITANV', 'A100', '2080Ti', 'RTX4070'], graph order ['TITANV', 'A100', '2080Ti', 'RTX4070'], differs from measured
- slot_drv, source TITANV: eager order ['TITANV', '2080Ti', 'A100', 'RTX4070'], graph order ['TITANV', 'A100', '2080Ti', 'RTX4070'], differs from measured

