Workload: `diffusionbench`

## MAPE (%) over cross-GPU pairs and configs

| quantity     |   same |   spec_bw |   bw |   lat |   lat_bw |   slot_class |   slot |   slot_cold |   slot_cold_py |   clock |   slot_cpp |   slot_drv |
|:-------------|-------:|----------:|-----:|------:|---------:|-------------:|-------:|------------:|---------------:|--------:|-----------:|-----------:|
| eager        |   32.5 |      44.4 | 46.2 |  40.6 |     44.7 |         41.6 |   45.4 |        41.8 |           28.2 |    50.7 |       45.9 |       36.8 |
| eager_prefix |   38.5 |      38.5 | 38.5 |  36.9 |     38.5 |         38.8 |   38.5 |        38.8 |           24.6 |    36   |       36.9 |       32.7 |
| eager_loop   |   32.7 |      44.7 | 46.4 |  41   |     45   |         41.8 |   45.7 |        42   |           28.6 |    51.2 |       46.4 |       37   |
| graph_loop   |    9.8 |      33.1 | 35.8 |  25.2 |     32.3 |         25.5 |   32.3 |        25.7 |           25.7 |    39.3 |       31.8 |       25.5 |
| hybrid       |    9.6 |      33.1 | 35.8 |  24.3 |     32.3 |         25.6 |   32.4 |        25.8 |           25.4 |    38.2 |       30.9 |       25.5 |
| speedup_loop |   25.7 |      27.7 | 26.6 |  46.5 |     24.4 |         18.3 |   19.6 |        18.3 |           13.9 |    59.8 |       54.4 |       13.9 |
| speedup_e2e  |   24.9 |      26.9 | 25.8 |  44.5 |     23.8 |         17.8 |   19.2 |        17.8 |           13.7 |    56.9 |       52   |       13.6 | 

## MAPE (%) per pair, variant `slot_drv`

|                       |   eager |   eager_prefix |   eager_loop |   graph_loop |   hybrid |   speedup_loop |   speedup_e2e |
|:----------------------|--------:|---------------:|-------------:|-------------:|---------:|---------------:|--------------:|
| ('2080Ti', 'A100')    |    20.6 |            9   |         21   |         21   |     20.8 |            0.3 |           0.4 |
| ('2080Ti', 'RTX4070') |    82.8 |           64.3 |         83.2 |          6.9 |      8   |           71.4 |          69.4 |
| ('2080Ti', 'TITANV')  |    25.4 |           12.9 |         25.8 |         28.1 |     27.8 |            3.2 |           3.2 |
| ('RTX4070', '2080Ti') |    12.4 |           39   |         12   |          5.6 |      6.2 |            6.8 |           6.8 |
| ('RTX4070', 'A100')   |    26.4 |           35   |         26.3 |         25.4 |     25.8 |            1.3 |           0.8 |
| ('RTX4070', 'TITANV') |    30.2 |           30.4 |         30.2 |         31.9 |     31.9 |            2.4 |           2.4 |
| ('TITANV', '2080Ti')  |    30   |           10.2 |         30.6 |         39.9 |     39   |            6.8 |           6.8 |
| ('TITANV', 'A100')    |     8.1 |            7.8 |          8.2 |         11   |     10.4 |            2.4 |           2   |
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
| same         |      84 |               9.524 |              0.048 |
| spec_bw      |      84 |               8.333 |             -0.258 |
| bw           |      84 |               8.333 |             -0.258 |
| lat          |      84 |              22.619 |              0.353 |
| lat_bw       |      84 |               8.333 |             -0.258 |
| slot_class   |      84 |               9.524 |             -0.127 |
| slot         |      84 |              10.714 |             -0.242 |
| slot_cold    |      84 |               9.524 |             -0.135 |
| slot_cold_py |      84 |               3.571 |             -0.256 |
| clock        |      84 |              22.619 |              0.353 |
| slot_cpp     |      84 |              21.429 |              0.258 |
| slot_drv     |      84 |               9.524 |             -0.127 | 

## Eager vs graph ordering inversion, baseline config (steps100_h16)

measured (ms): {'eager': {'2080Ti': 1268.5, 'RTX4070': 1160.1, 'TITANV': 1214.9}, 'graph_loop': {'2080Ti': 1093.2, 'RTX4070': 1079.9, 'TITANV': 1214.8}}
- spec_bw, source 2080Ti: eager order ['TITANV', '2080Ti', 'RTX4070'], graph order ['TITANV', '2080Ti', 'RTX4070'], differs from measured
- spec_bw, source RTX4070: eager order ['TITANV', '2080Ti', 'RTX4070'], graph order ['TITANV', '2080Ti', 'RTX4070'], differs from measured
- spec_bw, source TITANV: eager order ['TITANV', '2080Ti', 'RTX4070'], graph order ['TITANV', '2080Ti', 'RTX4070'], differs from measured
- slot_drv, source 2080Ti: eager order ['TITANV', '2080Ti', 'RTX4070'], graph order ['TITANV', '2080Ti', 'RTX4070'], differs from measured
- slot_drv, source RTX4070: eager order ['TITANV', '2080Ti', 'RTX4070'], graph order ['TITANV', '2080Ti', 'RTX4070'], differs from measured
- slot_drv, source TITANV: eager order ['TITANV', '2080Ti', 'RTX4070'], graph order ['TITANV', '2080Ti', 'RTX4070'], differs from measured

