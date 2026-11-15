Workload: `pi0fast`

## MAPE (%) over cross-GPU pairs and configs

| quantity     |   same |   spec_bw |   bw |   lat |   lat_bw |   slot_class |   slot |   slot_cold |   slot_cold_py |   clock |   slot_cpp |   slot_drv |
|:-------------|-------:|----------:|-----:|------:|---------:|-------------:|-------:|------------:|---------------:|--------:|-----------:|-----------:|
| eager        |   25.9 |      25.9 | 25.9 |  21.9 |     25.9 |         25.9 |   25.9 |        25.9 |           15.7 |    24.1 |       22   |       19.2 |
| eager_prefix |   22   |      29.7 | 29.5 |  37.3 |     29   |         34.6 |   29.2 |        29.8 |           34.7 |    41.4 |       34.1 |       33.4 |
| eager_loop   |   25.8 |      25.8 | 25.8 |  22.2 |     25.8 |         25.8 |   25.8 |        25.8 |           15.6 |    23.8 |       21.7 |       19.6 |
| graph_loop   |    2.5 |      18.7 | 22   |  32.1 |     12.2 |          9.3 |   14.3 |         9   |            9   |    46.2 |       37.6 |        9.3 |
| hybrid       |    3.9 |      18.6 | 21.9 |  31.9 |     12.4 |          7.2 |   14.4 |         7.3 |            6.8 |    46.1 |       37.5 |        7.1 |
| speedup_loop |   26.3 |      12.8 | 13.8 |  56.5 |     16.2 |         16.9 |   13.4 |        16.7 |           14.2 |    71.1 |       61   |       13.1 |
| speedup_e2e  |   26.9 |      14   | 14.4 |  53.8 |     17.2 |         19.4 |   14.6 |        19   |           15   |    67.2 |       58   |       14.6 | 

## MAPE (%) per pair, variant `slot_drv`

|                       |   eager |   eager_prefix |   eager_loop |   graph_loop |   hybrid |   speedup_loop |   speedup_e2e |
|:----------------------|--------:|---------------:|-------------:|-------------:|---------:|---------------:|--------------:|
| ('2080Ti', 'RTX4070') |    27.8 |            9   |         28.5 |         12.2 |     11.4 |           14.5 |          14.7 |
| ('2080Ti', 'TITANV')  |     0.4 |           70.8 |          1.1 |          5.6 |      7.8 |            4.3 |           6.8 |
| ('RTX4070', '2080Ti') |    28.3 |            7.1 |         27.2 |         13.2 |      6.7 |           16.2 |          23.2 |
| ('RTX4070', 'TITANV') |    26.1 |           67   |         25.7 |         10.9 |      8.3 |           16.6 |          19.5 |
| ('TITANV', '2080Ti')  |     0   |           29.8 |          1.6 |          3.9 |      0.6 |            5.8 |           0.6 |
| ('TITANV', 'RTX4070') |    32.6 |           16.6 |         33.5 |         10   |      8   |           21.3 |          22.8 | 

## Same-GPU baseline (source == target), variant-independent

| source   |   eager |   eager_prefix |   eager_loop |   graph_loop |   hybrid |   speedup_loop |   speedup_e2e |
|:---------|--------:|---------------:|-------------:|-------------:|---------:|---------------:|--------------:|
| 2080Ti   |       0 |           16.3 |          1.6 |            0 |      7.1 |            1.6 |           6.6 |
| RTX4070  |       0 |            0.7 |          0.5 |            0 |      0.5 |            0.5 |           0.5 |
| TITANV   |       0 |            5.5 |          0.7 |            0 |      0.5 |            0.7 |           0.5 | 

## GPU ordering (all GPUs ranked per source, config, quantity)

| variant      |   cases |   exact_ranking_pct |   mean_kendall_tau |
|:-------------|--------:|--------------------:|-------------------:|
| same         |      21 |              71.429 |              0.714 |
| spec_bw      |      21 |              71.429 |              0.524 |
| bw           |      21 |              71.429 |              0.524 |
| lat          |      21 |              66.667 |              0.714 |
| lat_bw       |      21 |              71.429 |              0.524 |
| slot_class   |      21 |              61.905 |              0.587 |
| slot         |      21 |              71.429 |              0.524 |
| slot_cold    |      21 |              61.905 |              0.587 |
| slot_cold_py |      21 |              57.143 |              0.492 |
| clock        |      21 |              61.905 |              0.683 |
| slot_cpp     |      21 |              61.905 |              0.683 |
| slot_drv     |      21 |              66.667 |              0.587 | 

## Eager vs graph ordering inversion, baseline config (decode256)

measured (ms): {'eager': {'2080Ti': 8107.7, 'RTX4070': 10610.5, 'TITANV': 5625.8}, 'graph_loop': {'2080Ti': 3145.8, 'RTX4070': 2986.6, 'TITANV': 3192.4}}
- spec_bw, source 2080Ti: eager order ['TITANV', '2080Ti', 'RTX4070'], graph order ['TITANV', '2080Ti', 'RTX4070'], differs from measured
- spec_bw, source RTX4070: eager order ['TITANV', '2080Ti', 'RTX4070'], graph order ['TITANV', '2080Ti', 'RTX4070'], differs from measured
- spec_bw, source TITANV: eager order ['TITANV', '2080Ti', 'RTX4070'], graph order ['TITANV', '2080Ti', 'RTX4070'], differs from measured
- slot_drv, source 2080Ti: eager order ['TITANV', '2080Ti', 'RTX4070'], graph order ['2080Ti', 'RTX4070', 'TITANV'], differs from measured
- slot_drv, source RTX4070: eager order ['TITANV', '2080Ti', 'RTX4070'], graph order ['2080Ti', 'TITANV', 'RTX4070'], differs from measured
- slot_drv, source TITANV: eager order ['TITANV', '2080Ti', 'RTX4070'], graph order ['2080Ti', 'TITANV', 'RTX4070'], differs from measured

