Workload: `diffusionbench`

## MAPE (%) over cross-GPU pairs and configs

| quantity     |   same |   spec_bw |   bw |   lat |   lat_bw |   slot_class |   slot |   slot_cold |   slot_cold_py |   clock |   slot_cpp |   slot_drv |
|:-------------|-------:|----------:|-----:|------:|---------:|-------------:|-------:|------------:|---------------:|--------:|-----------:|-----------:|
| eager        |   36.5 |      43.8 | 45.9 |  47.5 |     44.1 |         47.4 |   45.5 |        47.6 |           30.8 |    57   |       51.6 |       41.4 |
| eager_prefix |   43.7 |      43.7 | 43.7 |  42.2 |     43.7 |         43.7 |   43.7 |        43.7 |           27   |    41.5 |       41.9 |       36.5 |
| eager_loop   |   36.6 |      43.9 | 46.1 |  47.9 |     44.3 |         47.6 |   45.7 |        47.8 |           31.2 |    57.5 |       52   |       41.6 |
| graph_loop   |    8.2 |      24.4 | 28   |  27.8 |     24.5 |         27   |   26.7 |        27.3 |           27.3 |    42.1 |       33.8 |       27   |
| hybrid       |    8   |      24.7 | 28.1 |  26.8 |     24.8 |         27.2 |   26.9 |        27.5 |           27   |    40.9 |       32.7 |       27.1 |
| speedup_loop |   31.8 |      19.5 | 18.5 |  57.6 |     20   |         22.6 |   19.5 |        22.5 |           17.1 |    74.1 |       67.6 |       17.1 |
| speedup_e2e  |   30.9 |      19.1 | 18.1 |  55.3 |     19.5 |         22   |   19   |        22   |           16.8 |    70.6 |       64.6 |       16.7 | 

## MAPE (%) per pair, variant `slot_drv`

|                       |   eager |   eager_prefix |   eager_loop |   graph_loop |   hybrid |   speedup_loop |   speedup_e2e |
|:----------------------|--------:|---------------:|-------------:|-------------:|---------:|---------------:|--------------:|
| ('2080Ti', 'RTX4070') |    82.8 |           64.3 |         83.2 |          6.9 |      8   |           71.4 |          69.4 |
| ('2080Ti', 'TITANV')  |    25.4 |           12.9 |         25.8 |         28.1 |     27.8 |            3.2 |           3.2 |
| ('RTX4070', '2080Ti') |    12.4 |           39   |         12   |          5.6 |      6.2 |            6.8 |           6.8 |
| ('RTX4070', 'TITANV') |    30.2 |           30.4 |         30.2 |         31.9 |     31.9 |            2.4 |           2.4 |
| ('TITANV', '2080Ti')  |    30   |           10.2 |         30.6 |         39.9 |     39   |            6.8 |           6.8 |
| ('TITANV', 'RTX4070') |    67.6 |           62.3 |         67.7 |         49.8 |     50   |           12   |          11.7 | 

## Same-GPU baseline (source == target), variant-independent

| source   |   eager |   eager_prefix |   eager_loop |   graph_loop |   hybrid |   speedup_loop |   speedup_e2e |
|:---------|--------:|---------------:|-------------:|-------------:|---------:|---------------:|--------------:|
| 2080Ti   |     7.8 |           11.6 |          7.8 |          0.2 |      0.1 |            7.8 |           7.8 |
| RTX4070  |     0.8 |            8.9 |          0.6 |          0   |      0.1 |            0.6 |           0.6 |
| TITANV   |     2   |            7.2 |          1.9 |          0   |      0.1 |            1.9 |           1.9 | 

## GPU ordering (all GPUs ranked per source, config, quantity)

| variant      |   cases |   exact_ranking_pct |   mean_kendall_tau |
|:-------------|--------:|--------------------:|-------------------:|
| same         |      84 |              13.095 |              0.119 |
| spec_bw      |      84 |              13.095 |             -0.19  |
| bw           |      84 |              13.095 |             -0.19  |
| lat          |      84 |              40.476 |              0.444 |
| lat_bw       |      84 |              13.095 |             -0.19  |
| slot_class   |      84 |              13.095 |             -0.19  |
| slot         |      84 |              13.095 |             -0.19  |
| slot_cold    |      84 |              13.095 |             -0.19  |
| slot_cold_py |      84 |               9.524 |             -0.27  |
| clock        |      84 |              40.476 |              0.444 |
| slot_cpp     |      84 |              40.476 |              0.444 |
| slot_drv     |      84 |              13.095 |             -0.19  | 

## Eager vs graph ordering inversion, baseline config (steps100_h16)

measured (ms): {'eager': {'2080Ti': 1268.5, 'RTX4070': 1160.1, 'TITANV': 1214.9}, 'graph_loop': {'2080Ti': 1093.2, 'RTX4070': 1079.9, 'TITANV': 1214.8}}
- spec_bw, source 2080Ti: eager order ['TITANV', '2080Ti', 'RTX4070'], graph order ['TITANV', '2080Ti', 'RTX4070'], differs from measured
- spec_bw, source RTX4070: eager order ['TITANV', '2080Ti', 'RTX4070'], graph order ['TITANV', '2080Ti', 'RTX4070'], differs from measured
- spec_bw, source TITANV: eager order ['TITANV', '2080Ti', 'RTX4070'], graph order ['TITANV', '2080Ti', 'RTX4070'], differs from measured
- slot_drv, source 2080Ti: eager order ['TITANV', '2080Ti', 'RTX4070'], graph order ['TITANV', '2080Ti', 'RTX4070'], differs from measured
- slot_drv, source RTX4070: eager order ['TITANV', '2080Ti', 'RTX4070'], graph order ['TITANV', '2080Ti', 'RTX4070'], differs from measured
- slot_drv, source TITANV: eager order ['TITANV', '2080Ti', 'RTX4070'], graph order ['TITANV', '2080Ti', 'RTX4070'], differs from measured

