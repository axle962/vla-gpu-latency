Workload: `smolvla`

## MAPE (%) over cross-GPU pairs and configs

| quantity     |   same |   spec_bw |   bw |   lat |   lat_bw |   slot_class |   slot |   slot_cold |   slot_cold_py |   clock |   slot_cpp |   slot_drv |
|:-------------|-------:|----------:|-----:|------:|---------:|-------------:|-------:|------------:|---------------:|--------:|-----------:|-----------:|
| eager        |    6.6 |       7.1 |  7   |   7.4 |      6.9 |          7.5 |    7   |         7.3 |           20.1 |     7.6 |        7.2 |        6.6 |
| eager_prefix |    6.3 |       5.5 |  5.1 |  10.3 |      5.2 |          8.5 |    5.6 |         8   |           16.7 |    11.8 |        8.4 |        7   |
| eager_loop   |    7.7 |       7.7 |  7.7 |   7.7 |      7.7 |          7.7 |    7.7 |         7.7 |           20.7 |     7.7 |        7.7 |        6.5 |
| graph_loop   |   37.9 |      63.8 | 64.2 |  23.2 |     23.8 |         16.2 |   21.2 |        22.1 |           22.1 |    33.4 |       21.3 |       16.2 |
| hybrid       |   15.5 |      26.6 | 26.8 |  11.3 |      8.9 |          5.2 |    7.4 |         7.9 |            8.2 |    18.5 |        9.2 |        5.5 |
| speedup_loop |   45.9 |      68.5 | 68.9 |  30.5 |     31.2 |         22.3 |   26.3 |        29.1 |           41.8 |    38.5 |       27.4 |       21.7 |
| speedup_e2e  |   22   |      26.3 | 26.5 |  16.3 |     15.6 |         10.9 |   13   |        14.4 |           20.4 |    19.6 |       14.1 |       10.4 | 

## MAPE (%) per pair, variant `slot_drv`

|                      |   eager |   eager_prefix |   eager_loop |   graph_loop |   hybrid |   speedup_loop |   speedup_e2e |
|:---------------------|--------:|---------------:|-------------:|-------------:|---------:|---------------:|--------------:|
| ('2080Ti', 'A100')   |     5   |            6.2 |          3.6 |          6.2 |      2.3 |            9   |           3.3 |
| ('2080Ti', 'TITANV') |     6.1 |           15.3 |          4.8 |         16.6 |      3.4 |           25.7 |           8.7 |
| ('A100', '2080Ti')   |     7.5 |            3.7 |         10   |          4   |      3.4 |           10.1 |           6.6 |
| ('A100', 'TITANV')   |     6.4 |            3   |          8.6 |         19.6 |     11.8 |           35.3 |          20.7 |
| ('TITANV', '2080Ti') |     6.6 |            6.7 |          5.1 |         21.7 |      5.5 |           21.9 |           9.3 |
| ('TITANV', 'A100')   |     8.1 |            7.3 |          7.1 |         29   |      6.6 |           27.9 |          13.6 | 

## Same-GPU baseline (source == target), variant-independent

| source   |   eager |   eager_prefix |   eager_loop |   graph_loop |   hybrid |   speedup_loop |   speedup_e2e |
|:---------|--------:|---------------:|-------------:|-------------:|---------:|---------------:|--------------:|
| 2080Ti   |     4.5 |            6.8 |          5.7 |          0.2 |      4.9 |            5.5 |           4.3 |
| A100     |     1.4 |            5.4 |          1.8 |          0.4 |      4   |            2.2 |           3.1 |
| TITANV   |     1.5 |            4.7 |          1.8 |          0.3 |      1.8 |            2   |           1.8 | 

## GPU ordering (all GPUs ranked per source, config, quantity)

| variant      |   cases |   exact_ranking_pct |   mean_kendall_tau |
|:-------------|--------:|--------------------:|-------------------:|
| same         |     147 |              59.864 |              0.732 |
| spec_bw      |     147 |              34.694 |              0.565 |
| bw           |     147 |              35.374 |              0.569 |
| lat          |     147 |              88.435 |              0.923 |
| lat_bw       |     147 |              66.667 |              0.778 |
| slot_class   |     147 |              78.912 |              0.859 |
| slot         |     147 |              51.02  |              0.673 |
| slot_cold    |     147 |              81.633 |              0.878 |
| slot_cold_py |     147 |              50.34  |              0.621 |
| clock        |     147 |              70.748 |              0.769 |
| slot_cpp     |     147 |              70.068 |              0.8   |
| slot_drv     |     147 |              78.912 |              0.859 | 

## Eager vs graph ordering inversion, baseline config (cam3_steps10_chunk50)

measured (ms): {'eager': {'2080Ti': 484.4, 'A100': 382.7, 'TITANV': 340.5}, 'graph_loop': {'2080Ti': 39.8, 'A100': 35.6, 'TITANV': 61.4}}
- spec_bw, source 2080Ti: eager order ['TITANV', 'A100', '2080Ti'], graph order ['A100', 'TITANV', '2080Ti'], differs from measured
- spec_bw, source A100: eager order ['TITANV', 'A100', '2080Ti'], graph order ['A100', 'TITANV', '2080Ti'], differs from measured
- spec_bw, source TITANV: eager order ['TITANV', 'A100', '2080Ti'], graph order ['A100', 'TITANV', '2080Ti'], differs from measured
- slot_drv, source 2080Ti: eager order ['TITANV', 'A100', '2080Ti'], graph order ['A100', '2080Ti', 'TITANV'], matches measured
- slot_drv, source A100: eager order ['TITANV', 'A100', '2080Ti'], graph order ['A100', '2080Ti', 'TITANV'], matches measured
- slot_drv, source TITANV: eager order ['TITANV', 'A100', '2080Ti'], graph order ['A100', '2080Ti', 'TITANV'], matches measured

