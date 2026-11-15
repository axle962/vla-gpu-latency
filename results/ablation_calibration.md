Workload: `smolvla`, cross-GPU pairs only (source != target), 4 source GPUs.

## MAPE (%) per calibration level

| level             |   eager |   eager_loop |   eager_prefix |   graph_loop |   hybrid |   speedup_loop |
|:------------------|--------:|-------------:|---------------:|-------------:|---------:|---------------:|
| L0_spec           |    34.4 |         35.9 |           27.7 |         38.2 |     12.6 |           71.8 |
| L1_cpp_microbench |    34.3 |         35.9 |           27.5 |         30.9 |     10.2 |           66.2 |
| L2_dispatch_table |    25.3 |         26.1 |           23.1 |         30.9 |     24.1 |           27.4 |
| L3_graph_slots    |    24.4 |         26.1 |           20.8 |         15.3 |     15.5 |           25.9 |
| L4_final_slot_drv |    20.2 |         21.7 |           16.7 |         15.3 |     13   |           22   | 

## Exact ranking of all GPUs (%)

| level             |   cases |   exact_ranking_pct |   mean_kendall_tau |
|:------------------|--------:|--------------------:|-------------------:|
| L0_spec           |     168 |               19.64 |               0.4  |
| L1_cpp_microbench |     168 |               16.67 |               0.34 |
| L2_dispatch_table |     168 |               57.14 |               0.78 |
| L3_graph_slots    |     168 |               63.1  |               0.82 |
| L4_final_slot_drv |     168 |               64.29 |               0.82 | 

