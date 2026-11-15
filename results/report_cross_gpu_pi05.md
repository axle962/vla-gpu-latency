Workload: `pi05`

## MAPE (%) over cross-GPU pairs and configs

| quantity     |   same |   spec_bw |   bw |   lat |   lat_bw |   slot_class |   slot |   slot_cold |   slot_cold_py |   clock |   slot_cpp |   slot_drv |
|:-------------|-------:|----------:|-----:|------:|---------:|-------------:|-------:|------------:|---------------:|--------:|-----------:|-----------:|
| eager        |   22   |      23.9 | 24   |  20   |     23.8 |         17.1 |   23.8 |        16.9 |           13.1 |    19   |       18.3 |       11.9 |
| eager_prefix |   10.6 |      17.4 | 17.4 |  25   |     16.8 |         26.9 |   17.1 |        23.2 |           32.5 |    30.8 |       24.2 |       28.5 |
| eager_loop   |   28.4 |      28.4 | 28.4 |  28.4 |     28.4 |         28.4 |   28.4 |        28.4 |           12.8 |    28.4 |       28.4 |       24.6 |
| graph_loop   |   39.8 |      58.2 | 62.2 |   9.6 |     34.5 |         10.4 |   36   |        16.1 |           16.1 |     4.3 |        7.7 |       10.4 |
| hybrid       |   12.8 |      23.9 | 26.3 |  12.9 |     17.9 |         16.5 |   18.5 |        12.8 |           19.6 |    20.1 |       13.4 |       17.8 |
| speedup_loop |   30.4 |      34.2 | 37.2 |  29.1 |     22.8 |         37.5 |   21.4 |        39.3 |           28.2 |    32.8 |       26.4 |       30.8 |
| speedup_e2e  |   12.1 |       8.5 |  9.8 |  25.2 |      8.8 |         28.8 |    8   |        27.5 |           18.2 |    31.1 |       26.6 |       27.3 | 

## MAPE (%) per pair, variant `slot_drv`

|                       |   eager |   eager_prefix |   eager_loop |   graph_loop |   hybrid |   speedup_loop |   speedup_e2e |
|:----------------------|--------:|---------------:|-------------:|-------------:|---------:|---------------:|--------------:|
| ('2080Ti', 'RTX4070') |    20.2 |           11.2 |         31.6 |         14.3 |     11.9 |           53.5 |          36.5 |
| ('2080Ti', 'TITANV')  |     5.8 |           29.3 |          3.4 |         13.3 |     10.7 |           11.4 |           4.5 |
| ('RTX4070', '2080Ti') |    15.7 |           12.8 |         26.9 |         17.1 |     13.1 |           37.6 |          25.5 |
| ('RTX4070', 'TITANV') |     0   |           72.5 |         29.5 |          0   |     40.9 |           29.4 |          29.1 |
| ('TITANV', '2080Ti')  |     0.8 |           27.8 |         10.5 |         15.8 |     16.1 |            4.5 |          18.2 |
| ('TITANV', 'RTX4070') |    29.1 |           17.2 |         45.6 |          1.8 |     14   |           48.3 |          50.2 | 

## Same-GPU baseline (source == target), variant-independent

| source   |   eager |   eager_prefix |   eager_loop |   graph_loop |   hybrid |   speedup_loop |   speedup_e2e |
|:---------|--------:|---------------:|-------------:|-------------:|---------:|---------------:|--------------:|
| 2080Ti   |       0 |            6.1 |          2.9 |            0 |      5.2 |            2.9 |           5.4 |
| RTX4070  |       0 |            2.8 |          1.5 |            0 |      2.3 |            1.5 |           2.3 |
| TITANV   |       0 |            6.8 |          3.3 |            0 |      4   |            3.3 |           4.1 | 

## GPU ordering (all GPUs ranked per source, config, quantity)

| variant      |   cases |   exact_ranking_pct |   mean_kendall_tau |
|:-------------|--------:|--------------------:|-------------------:|
| same         |      21 |              85.714 |              0.905 |
| spec_bw      |      21 |              85.714 |              0.746 |
| bw           |      21 |              85.714 |              0.746 |
| lat          |      21 |              71.429 |              0.683 |
| lat_bw       |      21 |             100     |              1     |
| slot_class   |      21 |              71.429 |              0.587 |
| slot         |      21 |             100     |              1     |
| slot_cold    |      21 |              80.952 |              0.651 |
| slot_cold_py |      21 |              71.429 |              0.492 |
| clock        |      21 |              71.429 |              0.619 |
| slot_cpp     |      21 |              71.429 |              0.651 |
| slot_drv     |      21 |              76.19  |              0.619 | 

## Eager vs graph ordering inversion, baseline config (steps10_chunk50)

measured (ms): {'eager': {'2080Ti': 564.5, 'RTX4070': 712.7, 'TITANV': 425.0}, 'graph_loop': {'2080Ti': 64.8, 'RTX4070': 48.4, 'TITANV': 94.6}}
- spec_bw, source 2080Ti: eager order ['TITANV', '2080Ti', 'RTX4070'], graph order ['TITANV', '2080Ti', 'RTX4070'], differs from measured
- spec_bw, source RTX4070: eager order ['TITANV', '2080Ti', 'RTX4070'], graph order ['TITANV', 'RTX4070', '2080Ti'], differs from measured
- spec_bw, source TITANV: eager order ['TITANV', '2080Ti', 'RTX4070'], graph order ['TITANV', '2080Ti', 'RTX4070'], differs from measured
- slot_drv, source 2080Ti: eager order ['TITANV', '2080Ti', 'RTX4070'], graph order ['RTX4070', '2080Ti', 'TITANV'], matches measured
- slot_drv, source RTX4070: eager order ['TITANV', '2080Ti', 'RTX4070'], graph order ['RTX4070', '2080Ti', 'TITANV'], matches measured
- slot_drv, source TITANV: eager order ['TITANV', '2080Ti', 'RTX4070'], graph order ['RTX4070', '2080Ti', 'TITANV'], matches measured

