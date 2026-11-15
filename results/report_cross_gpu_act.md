Workload: `act`

## MAPE (%) over cross-GPU pairs and configs

| quantity    |   same |   spec_bw |   bw |   lat |   lat_bw |   slot_class |   slot |   slot_cold |   slot_cold_py |   clock |   slot_cpp |   slot_drv |
|:------------|-------:|----------:|-----:|------:|---------:|-------------:|-------:|------------:|---------------:|--------:|-----------:|-----------:|
| eager       |   31.1 |      32.1 | 32   |  29.7 |     31.6 |         31   |   31.7 |        30.9 |           16.4 |    28.7 |       29.9 |       28   |
| graph_loop  |   17   |      75.3 | 76.5 |  22.5 |     52.1 |         12.2 |   42.6 |        12.2 |           12.2 |    34.4 |       20.2 |       12.2 |
| speedup_e2e |   21.3 |      46.4 | 47.1 |  49.5 |     23.4 |         23.8 |   15   |        25.1 |           16.3 |    66.2 |       41.4 |       21.2 | 

## MAPE (%) per pair, variant `slot_drv`

|                       |   eager |   graph_loop |   speedup_e2e |
|:----------------------|--------:|-------------:|--------------:|
| ('2080Ti', 'A100')    |     9.6 |          7.4 |          11.4 |
| ('2080Ti', 'RTX4070') |    38.7 |         16.2 |          22.8 |
| ('2080Ti', 'TITANV')  |     2.2 |          8.2 |           7   |
| ('A100', '2080Ti')    |    27.4 |          3.8 |          24.3 |
| ('A100', 'RTX4070')   |    64.6 |         23   |          44.2 |
| ('A100', 'TITANV')    |    21   |         14.6 |           6   |
| ('RTX4070', '2080Ti') |    31.1 |         12.8 |          18.8 |
| ('RTX4070', 'A100')   |    39.5 |         19.2 |          28.8 |
| ('RTX4070', 'TITANV') |    28.2 |          7   |          23.8 |
| ('TITANV', '2080Ti')  |    12.5 |          5.3 |          17.8 |
| ('TITANV', 'A100')    |     8   |         11   |          18   |
| ('TITANV', 'RTX4070') |    53.6 |         17.8 |          31.2 | 

## Same-GPU baseline (source == target), variant-independent

| source   |   eager |   graph_loop |   speedup_e2e |
|:---------|--------:|-------------:|--------------:|
| 2080Ti   |     4.7 |          1   |           5.8 |
| A100     |     8   |          0.2 |           7.8 |
| RTX4070  |     0   |          2.2 |           2.2 |
| TITANV   |     7.3 |          0.5 |           7.8 | 

## GPU ordering (all GPUs ranked per source, config, quantity)

| variant      |   cases |   exact_ranking_pct |   mean_kendall_tau |
|:-------------|--------:|--------------------:|-------------------:|
| same         |      24 |              50     |              0.833 |
| spec_bw      |      24 |              29.167 |              0.486 |
| bw           |      24 |              29.167 |              0.486 |
| lat          |      24 |              45.833 |              0.694 |
| lat_bw       |      24 |              41.667 |              0.604 |
| slot_class   |      24 |              50     |              0.833 |
| slot         |      24 |              45.833 |              0.625 |
| slot_cold    |      24 |              58.333 |              0.833 |
| slot_cold_py |      24 |              33.333 |              0.75  |
| clock        |      24 |              45.833 |              0.681 |
| slot_cpp     |      24 |              45.833 |              0.806 |
| slot_drv     |      24 |              50     |              0.833 | 

## Eager vs graph ordering inversion, baseline config (res480x640_chunk100)

measured (ms): {'eager': {'2080Ti': 12.6, 'A100': 10.7, 'RTX4070': 15.4, 'TITANV': 9.0}, 'graph_loop': {'2080Ti': 5.3, 'A100': 4.3, 'RTX4070': 4.4, 'TITANV': 5.5}}
- spec_bw, source 2080Ti: eager order ['TITANV', 'A100', '2080Ti', 'RTX4070'], graph order ['A100', 'TITANV', '2080Ti', 'RTX4070'], differs from measured
- spec_bw, source A100: eager order ['A100', 'TITANV', '2080Ti', 'RTX4070'], graph order ['A100', 'TITANV', '2080Ti', 'RTX4070'], differs from measured
- spec_bw, source RTX4070: eager order ['TITANV', 'A100', '2080Ti', 'RTX4070'], graph order ['A100', 'TITANV', '2080Ti', 'RTX4070'], differs from measured
- spec_bw, source TITANV: eager order ['TITANV', 'A100', '2080Ti', 'RTX4070'], graph order ['A100', 'TITANV', '2080Ti', 'RTX4070'], differs from measured
- slot_drv, source 2080Ti: eager order ['TITANV', 'A100', '2080Ti', 'RTX4070'], graph order ['RTX4070', 'A100', '2080Ti', 'TITANV'], differs from measured
- slot_drv, source A100: eager order ['TITANV', 'A100', '2080Ti', 'RTX4070'], graph order ['RTX4070', 'A100', '2080Ti', 'TITANV'], differs from measured
- slot_drv, source RTX4070: eager order ['A100', 'TITANV', '2080Ti', 'RTX4070'], graph order ['RTX4070', 'A100', '2080Ti', 'TITANV'], differs from measured
- slot_drv, source TITANV: eager order ['TITANV', 'A100', '2080Ti', 'RTX4070'], graph order ['RTX4070', 'A100', '2080Ti', 'TITANV'], differs from measured

