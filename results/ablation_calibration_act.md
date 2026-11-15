Workload: `act`, cross-GPU pairs only (source != target), 4 source GPUs.

## MAPE (%) per calibration level

| level             |   eager |   graph_loop |   speedup_e2e |
|:------------------|--------:|-------------:|--------------:|
| L0_spec           |    30.1 |         50.2 |          67   |
| L1_cpp_microbench |    30.1 |         52.1 |          71   |
| L2_dispatch_table |    31.6 |         52.1 |          23.4 |
| L3_graph_slots    |    31   |         12.2 |          23.8 |
| L4_final_slot_drv |    28   |         12.2 |          21.2 | 

## Exact ranking of all GPUs (%)

| level             |   cases |   exact_ranking_pct |   mean_kendall_tau |
|:------------------|--------:|--------------------:|-------------------:|
| L0_spec           |      24 |                0    |              -0.14 |
| L1_cpp_microbench |      24 |                0    |              -0.17 |
| L2_dispatch_table |      24 |               37.5  |               0.58 |
| L3_graph_slots    |      24 |               41.67 |               0.81 |
| L4_final_slot_drv |      24 |               41.67 |               0.81 | 

