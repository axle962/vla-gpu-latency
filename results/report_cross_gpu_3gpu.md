## MAPE (%) over cross-GPU pairs and configs

| quantity     |   same |   spec_bw |   bw |   lat |   lat_bw |   slot_class |   slot |   slot_cold |   slot_cold_py |   clock |   slot_cpp |   slot_drv |
|:-------------|-------:|----------:|-----:|------:|---------:|-------------:|-------:|------------:|---------------:|--------:|-----------:|-----------:|
| eager        |    6.6 |       7.1 |  7   |   7.4 |      6.9 |          7.5 |    7   |         7.3 |           20.1 |     7.6 |        7.2 |        6.6 |
| eager_prefix |    6.3 |       5.6 |  5.2 |  10.4 |      5.2 |          8.6 |    5.7 |         8   |           16.8 |    11.8 |        8.5 |        7.1 |
| eager_loop   |    7.7 |       7.7 |  7.7 |   7.7 |      7.7 |          7.7 |    7.7 |         7.7 |           20.7 |     7.7 |        7.7 |        6.5 |
| graph_loop   |   37.9 |      63.8 | 64.2 |  23.2 |     23.8 |         16.2 |   21.2 |        22.1 |           22.1 |    33.4 |       21.3 |       16.2 |
| hybrid       |   15.5 |      26.6 | 26.8 |  11.3 |      8.9 |          5.2 |    7.4 |         7.8 |            8.2 |    18.5 |        9.3 |        5.5 |
| speedup_loop |   46   |      68.5 | 68.9 |  30.5 |     31.3 |         22.3 |   26.3 |        29.1 |           41.8 |    38.5 |       27.4 |       21.7 |
| speedup_e2e  |   22   |      26.3 | 26.5 |  16.3 |     15.6 |         10.9 |   13   |        14.3 |           20.3 |    19.7 |       14.1 |       10.4 | 

## MAPE (%) per pair, variant `slot_class`

|                      |   eager |   eager_prefix |   eager_loop |   graph_loop |   hybrid |   speedup_loop |   speedup_e2e |
|:---------------------|--------:|---------------:|-------------:|-------------:|---------:|---------------:|--------------:|
| ('2080Ti', 'A100')   |     1.3 |            2.4 |          2.9 |          6.2 |      1.4 |            4.1 |           1.5 |
| ('2080Ti', 'TITANV') |    10   |           17.9 |          9   |         16.6 |      2.8 |           30.8 |          11.4 |
| ('A100', '2080Ti')   |     4.3 |            9.1 |          5.3 |          4   |      6   |            6   |           4.8 |
| ('A100', 'TITANV')   |     8.8 |            2.3 |         11.1 |         19.6 |     10.9 |           38.5 |          22.3 |
| ('TITANV', '2080Ti') |    10.3 |           10.2 |          8.8 |         21.7 |      4.6 |           25   |          10.9 |
| ('TITANV', 'A100')   |    10.1 |            9.5 |          9.1 |         29   |      5.7 |           29.5 |          14.4 | 

## Same-GPU baseline (source == target), variant-independent

| source   |   eager |   eager_prefix |   eager_loop |   graph_loop |   hybrid |   speedup_loop |   speedup_e2e |
|:---------|--------:|---------------:|-------------:|-------------:|---------:|---------------:|--------------:|
| 2080Ti   |     4.5 |            6.9 |          5.7 |          0.2 |      5   |            5.5 |           4.4 |
| A100     |     1.4 |            5.4 |          1.8 |          0.4 |      4.1 |            2.2 |           3.1 |
| TITANV   |     1.5 |            4.6 |          1.8 |          0.3 |      1.7 |            2   |           1.8 | 

## GPU ordering (all GPUs ranked per source, config, quantity)

| variant      |   cases |   exact_ranking_pct |   mean_kendall_tau |
|:-------------|--------:|--------------------:|-------------------:|
| same         |     147 |              61.224 |              0.741 |
| spec_bw      |     147 |              34.694 |              0.565 |
| bw           |     147 |              35.374 |              0.569 |
| lat          |     147 |              88.435 |              0.923 |
| lat_bw       |     147 |              66.667 |              0.778 |
| slot_class   |     147 |              78.912 |              0.859 |
| slot         |     147 |              51.02  |              0.673 |
| slot_cold    |     147 |              78.231 |              0.855 |
| slot_cold_py |     147 |              51.02  |              0.624 |
| clock        |     147 |              68.707 |              0.755 |
| slot_cpp     |     147 |              70.068 |              0.8   |
| slot_drv     |     147 |              79.592 |              0.864 | 

## Eager vs graph ordering inversion, baseline config (cam3_steps10_chunk50)

measured (ms): {'eager': {'2080Ti': 484.4, 'A100': 382.7, 'TITANV': 340.5}, 'graph_loop': {'2080Ti': 39.8, 'A100': 35.6, 'TITANV': 61.4}}
- spec_bw, source 2080Ti: eager order ['TITANV', 'A100', '2080Ti'], graph order ['A100', 'TITANV', '2080Ti'], differs from measured
- spec_bw, source A100: eager order ['TITANV', 'A100', '2080Ti'], graph order ['A100', 'TITANV', '2080Ti'], differs from measured
- spec_bw, source TITANV: eager order ['TITANV', 'A100', '2080Ti'], graph order ['A100', 'TITANV', '2080Ti'], differs from measured
- slot_class, source 2080Ti: eager order ['TITANV', 'A100', '2080Ti'], graph order ['A100', '2080Ti', 'TITANV'], matches measured
- slot_class, source A100: eager order ['TITANV', 'A100', '2080Ti'], graph order ['A100', '2080Ti', 'TITANV'], matches measured
- slot_class, source TITANV: eager order ['TITANV', 'A100', '2080Ti'], graph order ['A100', '2080Ti', 'TITANV'], matches measured

