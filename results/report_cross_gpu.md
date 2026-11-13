## MAPE (%) over cross-GPU pairs and configs

| quantity     |   same |   spec_bw |   bw |   lat |   lat_bw |   slot_class |   slot |   slot_cold |   slot_cold_py |   clock |   slot_cpp |
|:-------------|-------:|----------:|-----:|------:|---------:|-------------:|-------:|------------:|---------------:|--------:|-----------:|
| eager        |   24.9 |      25.5 | 25.4 |  24.7 |     25.3 |         24.4 |   25.4 |        24.3 |           16.2 |    24.4 |       24.6 |
| eager_prefix |   21.8 |      23.6 | 23.5 |  20.5 |     23.1 |         20.8 |   23.2 |        20   |           14.7 |    21.8 |       20.1 |
| eager_loop   |   26.1 |      26.1 | 26.1 |  26.1 |     26.1 |         26.1 |   26.1 |        26.1 |           16.6 |    26.1 |       26.1 |
| graph_loop   |   46.9 |      96.4 | 98.4 |  20.9 |     30.9 |         15.3 |   38.5 |        18.3 |           18.3 |    27.9 |       24.7 |
| hybrid       |   29.1 |      44.6 | 45.2 |  19.8 |     24.1 |         15.4 |   25.4 |        16.5 |            7.7 |    22.7 |       20.3 |
| speedup_loop |   35.9 |      64   | 65   |  30.1 |     27.4 |         25.9 |   24.2 |        30.1 |           32   |    38.1 |       24.8 |
| speedup_e2e  |   16.7 |      23.4 | 23.9 |  14.9 |     12.5 |         13.2 |   10.4 |        15.1 |           15   |    18.5 |       12.8 | 

## MAPE (%) per pair, variant `slot_class`

|                       |   eager |   eager_prefix |   eager_loop |   graph_loop |   hybrid |   speedup_loop |   speedup_e2e |
|:----------------------|--------:|---------------:|-------------:|-------------:|---------:|---------------:|--------------:|
| ('2080Ti', 'A100')    |     1.3 |            2.4 |          2.9 |          6.2 |      1.4 |            4.1 |           1.5 |
| ('2080Ti', 'RTX4070') |    61.6 |           53.8 |         64.4 |         10.2 |     43.5 |           49.2 |          12.6 |
| ('2080Ti', 'TITANV')  |    10   |           17.9 |          9   |         16.6 |      2.8 |           30.8 |          11.4 |
| ('A100', '2080Ti')    |     4.3 |            9.1 |          5.3 |          4   |      6   |            6   |           4.8 |
| ('A100', 'RTX4070')   |    56   |           47.7 |         58.9 |          9.2 |     38.5 |           46.8 |          12.7 |
| ('A100', 'TITANV')    |     8.8 |            2.3 |         11.1 |         19.6 |     10.9 |           38.5 |          22.3 |
| ('RTX4070', '2080Ti') |    36   |           19.3 |         38.2 |          4.6 |     14.7 |           35.1 |          25.1 |
| ('RTX4070', 'A100')   |    32.7 |           15.2 |         35.1 |          6.2 |     11.4 |           32.1 |          24   |
| ('RTX4070', 'TITANV') |    20.3 |           21.1 |         28.4 |         24   |      7.3 |            5.7 |          17.2 |
| ('TITANV', '2080Ti')  |    10.3 |           10.2 |          8.8 |         21.7 |      4.6 |           25   |          10.9 |
| ('TITANV', 'A100')    |    10.1 |            9.5 |          9.1 |         29   |      5.7 |           29.5 |          14.4 |
| ('TITANV', 'RTX4070') |    41.2 |           41.1 |         42.4 |         32   |     38.4 |            7.9 |           2   | 

## Same-GPU baseline (source == target), variant-independent

| source   |   eager |   eager_prefix |   eager_loop |   graph_loop |   hybrid |   speedup_loop |   speedup_e2e |
|:---------|--------:|---------------:|-------------:|-------------:|---------:|---------------:|--------------:|
| 2080Ti   |     4.5 |            6.9 |          5.7 |          0.2 |      5   |            5.5 |           4.4 |
| A100     |     1.4 |            5.4 |          1.8 |          0.4 |      4.1 |            2.2 |           3.1 |
| RTX4070  |     1.3 |            2.1 |          2.4 |          1.7 |      1.6 |            3.2 |           2.3 |
| TITANV   |     1.5 |            4.6 |          1.8 |          0.3 |      1.7 |            2   |           1.8 | 

## GPU ordering (all GPUs ranked per source, config, quantity)

| variant      |   cases |   exact_ranking_pct |   mean_kendall_tau |
|:-------------|--------:|--------------------:|-------------------:|
| same         |     196 |              50     |              0.724 |
| spec_bw      |     196 |              26.02  |              0.515 |
| bw           |     196 |              27.041 |              0.515 |
| lat          |     196 |              72.449 |              0.847 |
| lat_bw       |     196 |              55.102 |              0.788 |
| slot_class   |     196 |              64.286 |              0.83  |
| slot         |     196 |              36.224 |              0.724 |
| slot_cold    |     196 |              64.286 |              0.825 |
| slot_cold_py |     196 |              40.306 |              0.73  |
| clock        |     196 |              58.673 |              0.762 |
| slot_cpp     |     196 |              56.633 |              0.806 | 

## Eager vs graph ordering inversion, baseline config (cam3_steps10_chunk50)

measured (ms): {'eager': {'2080Ti': 484.4, 'A100': 382.7, 'RTX4070': 607.0, 'TITANV': 340.5}, 'graph_loop': {'2080Ti': 39.8, 'A100': 35.6, 'RTX4070': 22.8, 'TITANV': 61.4}}
- spec_bw, source 2080Ti: eager order ['TITANV', 'A100', '2080Ti', 'RTX4070'], graph order ['A100', 'TITANV', 'RTX4070', '2080Ti'], differs from measured
- spec_bw, source A100: eager order ['TITANV', 'A100', '2080Ti', 'RTX4070'], graph order ['A100', 'TITANV', '2080Ti', 'RTX4070'], differs from measured
- spec_bw, source RTX4070: eager order ['TITANV', 'A100', '2080Ti', 'RTX4070'], graph order ['A100', 'RTX4070', 'TITANV', '2080Ti'], differs from measured
- spec_bw, source TITANV: eager order ['TITANV', 'A100', '2080Ti', 'RTX4070'], graph order ['A100', 'TITANV', '2080Ti', 'RTX4070'], differs from measured
- slot_class, source 2080Ti: eager order ['TITANV', 'A100', '2080Ti', 'RTX4070'], graph order ['RTX4070', 'A100', '2080Ti', 'TITANV'], matches measured
- slot_class, source A100: eager order ['TITANV', 'A100', '2080Ti', 'RTX4070'], graph order ['RTX4070', 'A100', '2080Ti', 'TITANV'], matches measured
- slot_class, source RTX4070: eager order ['A100', 'TITANV', '2080Ti', 'RTX4070'], graph order ['RTX4070', 'A100', '2080Ti', 'TITANV'], differs from measured
- slot_class, source TITANV: eager order ['TITANV', 'A100', '2080Ti', 'RTX4070'], graph order ['RTX4070', 'A100', '2080Ti', 'TITANV'], matches measured

