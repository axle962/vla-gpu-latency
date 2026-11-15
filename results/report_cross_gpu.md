Workload: `smolvla`

## MAPE (%) over cross-GPU pairs and configs

| quantity     |   same |   spec_bw |   bw |   lat |   lat_bw |   slot_class |   slot |   slot_cold |   slot_cold_py |   clock |   slot_cpp |   slot_drv |
|:-------------|-------:|----------:|-----:|------:|---------:|-------------:|-------:|------------:|---------------:|--------:|-----------:|-----------:|
| eager        |   24.9 |      25.5 | 25.4 |  24.7 |     25.3 |         24.4 |   25.4 |        24.3 |           16.2 |    24.4 |       24.6 |       20.2 |
| eager_prefix |   21.8 |      23.6 | 23.5 |  20.5 |     23   |         20.8 |   23.2 |        20   |           14.7 |    21.8 |       20.1 |       16.7 |
| eager_loop   |   26.1 |      26.1 | 26.1 |  26.1 |     26.1 |         26.1 |   26.1 |        26.1 |           16.6 |    26.1 |       26.1 |       21.7 |
| graph_loop   |   46.9 |      96.4 | 98.4 |  20.9 |     30.9 |         15.3 |   38.5 |        18.3 |           18.3 |    27.9 |       24.7 |       15.3 |
| hybrid       |   29.1 |      44.6 | 45.2 |  19.8 |     24.1 |         15.5 |   25.4 |        16.6 |            7.7 |    22.7 |       20.3 |       13   |
| speedup_loop |   35.9 |      64   | 65   |  30.1 |     27.4 |         25.9 |   24.2 |        30.1 |           31.9 |    38.1 |       24.8 |       22   |
| speedup_e2e  |   16.7 |      23.4 | 23.9 |  14.9 |     12.5 |         13.2 |   10.4 |        15   |           15   |    18.5 |       12.8 |       12.1 | 

## MAPE (%) per pair, variant `slot_drv`

|                       |   eager |   eager_prefix |   eager_loop |   graph_loop |   hybrid |   speedup_loop |   speedup_e2e |
|:----------------------|--------:|---------------:|-------------:|-------------:|---------:|---------------:|--------------:|
| ('2080Ti', 'A100')    |     5   |            6.2 |          3.6 |          6.2 |      2.3 |            9   |           3.3 |
| ('2080Ti', 'RTX4070') |    37.3 |           31.3 |         39.5 |         10.2 |     26.1 |           26.7 |           8.9 |
| ('2080Ti', 'TITANV')  |     6.1 |           15.3 |          4.8 |         16.6 |      3.4 |           25.7 |           8.7 |
| ('A100', '2080Ti')    |     7.5 |            3.7 |         10   |          4   |      3.4 |           10.1 |           6.6 |
| ('A100', 'RTX4070')   |    44   |           36.9 |         46.6 |          9.2 |     30   |           35.4 |          10.8 |
| ('A100', 'TITANV')    |     6.4 |            3   |          8.6 |         19.6 |     11.8 |           35.3 |          20.7 |
| ('RTX4070', '2080Ti') |    32.1 |           16.5 |         33.9 |          4.6 |     12.9 |           30.7 |          22.1 |
| ('RTX4070', 'A100')   |    32.3 |           15.2 |         34.6 |          6.2 |     11.4 |           31.5 |          23.4 |
| ('RTX4070', 'TITANV') |    20   |           20.8 |         27.9 |         24   |      7.3 |            5.1 |          16.7 |
| ('TITANV', '2080Ti')  |     6.6 |            6.7 |          5.1 |         21.7 |      5.5 |           21.9 |           9.3 |
| ('TITANV', 'A100')    |     8.1 |            7.3 |          7.1 |         29   |      6.6 |           27.9 |          13.6 |
| ('TITANV', 'RTX4070') |    37.3 |           37.5 |         38.4 |         32   |     35.6 |            4.9 |           1.2 | 

## Same-GPU baseline (source == target), variant-independent

| source   |   eager |   eager_prefix |   eager_loop |   graph_loop |   hybrid |   speedup_loop |   speedup_e2e |
|:---------|--------:|---------------:|-------------:|-------------:|---------:|---------------:|--------------:|
| 2080Ti   |     4.5 |            6.8 |          5.7 |          0.2 |      4.9 |            5.5 |           4.3 |
| A100     |     1.4 |            5.4 |          1.8 |          0.4 |      4   |            2.2 |           3.1 |
| RTX4070  |     1.3 |            2.1 |          2.4 |          1.7 |      1.5 |            3.2 |           2.2 |
| TITANV   |     1.5 |            4.7 |          1.8 |          0.3 |      1.8 |            2   |           1.8 | 

## GPU ordering (all GPUs ranked per source, config, quantity)

| variant      |   cases |   exact_ranking_pct |   mean_kendall_tau |
|:-------------|--------:|--------------------:|-------------------:|
| same         |     196 |              48.98  |              0.721 |
| spec_bw      |     196 |              26.02  |              0.515 |
| bw           |     196 |              27.041 |              0.515 |
| lat          |     196 |              72.449 |              0.847 |
| lat_bw       |     196 |              55.102 |              0.788 |
| slot_class   |     196 |              64.286 |              0.83  |
| slot         |     196 |              36.224 |              0.724 |
| slot_cold    |     196 |              66.837 |              0.833 |
| slot_cold_py |     196 |              39.796 |              0.729 |
| clock        |     196 |              60.204 |              0.767 |
| slot_cpp     |     196 |              56.633 |              0.806 |
| slot_drv     |     196 |              64.796 |              0.83  | 

## Eager vs graph ordering inversion, baseline config (cam3_steps10_chunk50)

measured (ms): {'eager': {'2080Ti': 484.4, 'A100': 382.7, 'RTX4070': 607.0, 'TITANV': 340.5}, 'graph_loop': {'2080Ti': 39.8, 'A100': 35.6, 'RTX4070': 22.8, 'TITANV': 61.4}}
- spec_bw, source 2080Ti: eager order ['TITANV', 'A100', '2080Ti', 'RTX4070'], graph order ['A100', 'TITANV', 'RTX4070', '2080Ti'], differs from measured
- spec_bw, source A100: eager order ['TITANV', 'A100', '2080Ti', 'RTX4070'], graph order ['A100', 'TITANV', '2080Ti', 'RTX4070'], differs from measured
- spec_bw, source RTX4070: eager order ['TITANV', 'A100', '2080Ti', 'RTX4070'], graph order ['A100', 'RTX4070', 'TITANV', '2080Ti'], differs from measured
- spec_bw, source TITANV: eager order ['TITANV', 'A100', '2080Ti', 'RTX4070'], graph order ['A100', 'TITANV', '2080Ti', 'RTX4070'], differs from measured
- slot_drv, source 2080Ti: eager order ['TITANV', 'A100', '2080Ti', 'RTX4070'], graph order ['RTX4070', 'A100', '2080Ti', 'TITANV'], matches measured
- slot_drv, source A100: eager order ['TITANV', 'A100', '2080Ti', 'RTX4070'], graph order ['RTX4070', 'A100', '2080Ti', 'TITANV'], matches measured
- slot_drv, source RTX4070: eager order ['A100', 'TITANV', '2080Ti', 'RTX4070'], graph order ['RTX4070', 'A100', '2080Ti', 'TITANV'], differs from measured
- slot_drv, source TITANV: eager order ['TITANV', 'A100', '2080Ti', 'RTX4070'], graph order ['RTX4070', 'A100', '2080Ti', 'TITANV'], matches measured

