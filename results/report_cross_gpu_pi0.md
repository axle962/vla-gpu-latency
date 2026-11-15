Workload: `pi0`

## MAPE (%) over cross-GPU pairs and configs

| quantity     |   same |   spec_bw |   bw |   lat |   lat_bw |   slot_class |   slot |   slot_cold |   slot_cold_py |   clock |   slot_cpp |   slot_drv |
|:-------------|-------:|----------:|-----:|------:|---------:|-------------:|-------:|------------:|---------------:|--------:|-----------:|-----------:|
| eager        |   27.9 |      29.8 | 29.7 |  26.2 |     29.6 |         23.5 |   29.7 |        23.3 |           18   |    24.7 |       24.5 |       17.8 |
| eager_prefix |   16.4 |      22.9 | 22.9 |  26.2 |     22.3 |         24   |   22.6 |        19.9 |           27.6 |    30   |       23.9 |       22.2 |
| eager_loop   |   32.6 |      32.6 | 32.6 |  32.6 |     32.6 |         32.6 |   32.6 |        32.6 |           19.1 |    32.6 |       32.6 |       26.5 |
| graph_loop   |   49.5 |      69.1 | 73.3 |  15.3 |     44   |         12.8 |   46.8 |        19.1 |           19.1 |     4.4 |       14.4 |       12.8 |
| hybrid       |   18.2 |      27.9 | 30.2 |  13.5 |     22.9 |         11.8 |   23.8 |         8   |           14.2 |    18.2 |       12.4 |       11.2 |
| speedup_loop |   40.2 |      44.3 | 47.5 |  29.7 |     31.7 |         38   |   30.4 |        39.9 |           37.1 |    30.8 |       26.3 |       31   |
| speedup_e2e  |   12.8 |      11.6 | 13.4 |  20.4 |      9.7 |         24.2 |    9.2 |        23.6 |           17.3 |    25.1 |       21.6 |       22.3 | 

## MAPE (%) per pair, variant `slot_drv`

|                       |   eager |   eager_prefix |   eager_loop |   graph_loop |   hybrid |   speedup_loop |   speedup_e2e |
|:----------------------|--------:|---------------:|-------------:|-------------:|---------:|---------------:|--------------:|
| ('2080Ti', 'RTX4070') |    30   |            1   |         42.5 |          7.1 |      2.3 |           53.3 |          33.1 |
| ('2080Ti', 'TITANV')  |    13.3 |           31.8 |          5.4 |         17.1 |     10.8 |           27.1 |           2.3 |
| ('RTX4070', '2080Ti') |    23.2 |            0.4 |         33.4 |          8.1 |      1.4 |           38.4 |          24.3 |
| ('RTX4070', 'TITANV') |     1.6 |           59.5 |         29.5 |         11.5 |     29.1 |           20.3 |          23.8 |
| ('TITANV', '2080Ti')  |     8.3 |           30.9 |          1.8 |         21   |     18.1 |           15.9 |          12.1 |
| ('TITANV', 'RTX4070') |    30.5 |            9.5 |         46.3 |         11.7 |      5.6 |           31   |          38.2 | 

## Same-GPU baseline (source == target), variant-independent

| source   |   eager |   eager_prefix |   eager_loop |   graph_loop |   hybrid |   speedup_loop |   speedup_e2e |
|:---------|--------:|---------------:|-------------:|-------------:|---------:|---------------:|--------------:|
| 2080Ti   |       0 |            6.6 |          3.1 |            0 |      5.8 |            3.1 |           6.2 |
| RTX4070  |       0 |            1.2 |          1.1 |            0 |      1.1 |            1.1 |           1.1 |
| TITANV   |       0 |            7.7 |          4.1 |            0 |      4.6 |            4.1 |           4.8 | 

## GPU ordering (all GPUs ranked per source, config, quantity)

| variant      |   cases |   exact_ranking_pct |   mean_kendall_tau |
|:-------------|--------:|--------------------:|-------------------:|
| same         |      21 |              71.429 |              0.81  |
| spec_bw      |      21 |              71.429 |              0.651 |
| bw           |      21 |              71.429 |              0.651 |
| lat          |      21 |              71.429 |              0.651 |
| lat_bw       |      21 |              85.714 |              0.905 |
| slot_class   |      21 |              76.19  |              0.683 |
| slot         |      21 |              85.714 |              0.905 |
| slot_cold    |      21 |              80.952 |              0.746 |
| slot_cold_py |      21 |              71.429 |              0.619 |
| clock        |      21 |              71.429 |              0.651 |
| slot_cpp     |      21 |              71.429 |              0.651 |
| slot_drv     |      21 |              76.19  |              0.683 | 

## Eager vs graph ordering inversion, baseline config (steps10_chunk50)

measured (ms): {'eager': {'2080Ti': 507.3, 'RTX4070': 591.5, 'TITANV': 355.3}, 'graph_loop': {'2080Ti': 54.6, 'RTX4070': 37.5, 'TITANV': 83.4}}
- spec_bw, source 2080Ti: eager order ['TITANV', '2080Ti', 'RTX4070'], graph order ['TITANV', '2080Ti', 'RTX4070'], differs from measured
- spec_bw, source RTX4070: eager order ['TITANV', '2080Ti', 'RTX4070'], graph order ['TITANV', 'RTX4070', '2080Ti'], differs from measured
- spec_bw, source TITANV: eager order ['TITANV', '2080Ti', 'RTX4070'], graph order ['TITANV', '2080Ti', 'RTX4070'], differs from measured
- slot_drv, source 2080Ti: eager order ['TITANV', '2080Ti', 'RTX4070'], graph order ['RTX4070', '2080Ti', 'TITANV'], matches measured
- slot_drv, source RTX4070: eager order ['TITANV', '2080Ti', 'RTX4070'], graph order ['RTX4070', '2080Ti', 'TITANV'], matches measured
- slot_drv, source TITANV: eager order ['TITANV', '2080Ti', 'RTX4070'], graph order ['RTX4070', '2080Ti', 'TITANV'], matches measured

