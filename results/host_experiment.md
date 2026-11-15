# Host CPU experiment (same RTX 2080 Ti, host slowed down)

## Host ratios (target/alone) vs the measured SmolVLA eager slowdown

| condition   |   none |   dispatch |   framework |   python |   measured eager ratio |
|:------------|-------:|-----------:|------------:|---------:|-----------------------:|
| smt_alu     |      1 |      1.976 |       2.152 |    1.951 |                  1.651 |
| smt_mem     |      1 |      1.969 |       1.945 |    2.347 |                  1.82  |
| share       |      1 |      1.7   |       1.74  |    1.735 |                  1.927 | 

## Measured (ms)

|                                                     |   alone |   share |   smt_alu |   smt_mem |
|:----------------------------------------------------|--------:|--------:|----------:|----------:|
| ('act', 'res480x640_chunk100', 'eager')             |    13.1 |    26   |      21.2 |      24.8 |
| ('act', 'res480x640_chunk100', 'graph_loop')        |     5.3 |     6.8 |       6.8 |       5.4 |
| ('smolvla', 'cam1_steps10_chunk50', 'eager')        |   459.9 |   909.6 |     774.4 |     853.2 |
| ('smolvla', 'cam1_steps10_chunk50', 'eager_loop')   |   401.9 |   797.6 |     679.1 |     747.7 |
| ('smolvla', 'cam1_steps10_chunk50', 'eager_prefix') |    55.7 |   108.9 |      91.9 |     100.1 |
| ('smolvla', 'cam1_steps10_chunk50', 'graph_loop')   |    37.2 |    39.4 |      37.6 |      37.4 |
| ('smolvla', 'cam1_steps10_chunk50', 'hybrid')       |    92.2 |   142.6 |     130.8 |     139.2 |
| ('smolvla', 'cam3_steps10_chunk50', 'eager')        |   489.4 |   943.1 |     808   |     890.7 |
| ('smolvla', 'cam3_steps10_chunk50', 'eager_loop')   |   404.3 |   777.9 |     680   |     745.7 |
| ('smolvla', 'cam3_steps10_chunk50', 'eager_prefix') |    75.3 |   155.3 |     125.5 |     138.7 |
| ('smolvla', 'cam3_steps10_chunk50', 'graph_loop')   |    39.8 |    43   |      40.2 |      40.2 |
| ('smolvla', 'cam3_steps10_chunk50', 'hybrid')       |   115.8 |   186.1 |     166.4 |     179.3 | 

## Prediction error (%) per host scaling

|                                                                |   none |   dispatch |   framework |   python |
|:---------------------------------------------------------------|-------:|-----------:|------------:|---------:|
| ('act', 'res480x640_chunk100', 'eager', 'share')               |  -38.1 |      -17.9 |       -16.7 |    -16.9 |
| ('act', 'res480x640_chunk100', 'eager', 'smt_alu')             |   -9.1 |       25.5 |        31.7 |     24.6 |
| ('act', 'res480x640_chunk100', 'eager', 'smt_mem')             |  -24.9 |        4.4 |         3.6 |     15.8 |
| ('act', 'res480x640_chunk100', 'graph_loop', 'share')          |  -22.3 |      -22.3 |       -22.3 |    -22.3 |
| ('act', 'res480x640_chunk100', 'graph_loop', 'smt_alu')        |  -22.2 |      -22.2 |       -22.2 |    -22.2 |
| ('act', 'res480x640_chunk100', 'graph_loop', 'smt_mem')        |   -1.6 |       -1.6 |        -1.6 |     -1.6 |
| ('smolvla', 'cam1_steps10_chunk50', 'eager', 'share')          |  -42.2 |      -19.5 |       -18.3 |    -18.4 |
| ('smolvla', 'cam1_steps10_chunk50', 'eager', 'smt_alu')        |   -6.5 |       30.7 |        37.4 |     29.7 |
| ('smolvla', 'cam1_steps10_chunk50', 'eager', 'smt_mem')        |  -22.7 |       10.8 |         9.9 |     23.8 |
| ('smolvla', 'cam1_steps10_chunk50', 'eager_loop', 'share')     |  -41.9 |      -18.9 |       -17.6 |    -17.8 |
| ('smolvla', 'cam1_steps10_chunk50', 'eager_loop', 'smt_alu')   |   -5.5 |       32   |        38.8 |     31.1 |
| ('smolvla', 'cam1_steps10_chunk50', 'eager_loop', 'smt_mem')   |  -22   |       11.9 |        11   |     25   |
| ('smolvla', 'cam1_steps10_chunk50', 'eager_prefix', 'share')   |  -43.1 |      -21.7 |       -20.5 |    -20.6 |
| ('smolvla', 'cam1_steps10_chunk50', 'eager_prefix', 'smt_alu') |  -10   |       25.6 |        32   |     24.7 |
| ('smolvla', 'cam1_steps10_chunk50', 'eager_prefix', 'smt_mem') |  -23.6 |        8.8 |         7.9 |     21.4 |
| ('smolvla', 'cam1_steps10_chunk50', 'graph_loop', 'share')     |   -5.8 |       -5.8 |        -5.8 |     -5.8 |
| ('smolvla', 'cam1_steps10_chunk50', 'graph_loop', 'smt_alu')   |   -0.8 |       -0.8 |        -0.8 |     -0.8 |
| ('smolvla', 'cam1_steps10_chunk50', 'graph_loop', 'smt_mem')   |    1   |        1   |         1   |      1   |
| ('smolvla', 'cam1_steps10_chunk50', 'hybrid', 'share')         |  -30.5 |      -14.2 |       -13.2 |    -13.4 |
| ('smolvla', 'cam1_steps10_chunk50', 'hybrid', 'smt_alu')       |   -8.2 |       16.8 |        21.3 |     16.1 |
| ('smolvla', 'cam1_steps10_chunk50', 'hybrid', 'smt_mem')       |  -18   |        5.3 |         4.7 |     14.4 |
| ('smolvla', 'cam3_steps10_chunk50', 'eager', 'share')          |  -41.9 |      -19.3 |       -18   |    -18.2 |
| ('smolvla', 'cam3_steps10_chunk50', 'eager', 'smt_alu')        |   -7.4 |       29.4 |        36   |     28.4 |
| ('smolvla', 'cam3_steps10_chunk50', 'eager', 'smt_mem')        |  -23.2 |        9.9 |         9.1 |     22.8 |
| ('smolvla', 'cam3_steps10_chunk50', 'eager_loop', 'share')     |  -40.4 |      -16.9 |       -15.5 |    -15.7 |
| ('smolvla', 'cam3_steps10_chunk50', 'eager_loop', 'smt_alu')   |   -5.6 |       31.9 |        38.6 |     30.9 |
| ('smolvla', 'cam3_steps10_chunk50', 'eager_loop', 'smt_mem')   |  -21.8 |       12.2 |        11.3 |     25.4 |
| ('smolvla', 'cam3_steps10_chunk50', 'eager_prefix', 'share')   |  -45.5 |      -26.4 |       -25.3 |    -25.5 |
| ('smolvla', 'cam3_steps10_chunk50', 'eager_prefix', 'smt_alu') |  -14.9 |       18.3 |        24.3 |     17.5 |
| ('smolvla', 'cam3_steps10_chunk50', 'eager_prefix', 'smt_mem') |  -26.9 |        2.9 |         2.2 |     14.6 |
| ('smolvla', 'cam3_steps10_chunk50', 'graph_loop', 'share')     |   -7.7 |       -7.7 |        -7.7 |     -7.7 |
| ('smolvla', 'cam3_steps10_chunk50', 'graph_loop', 'smt_alu')   |   -0.8 |       -0.8 |        -0.8 |     -0.8 |
| ('smolvla', 'cam3_steps10_chunk50', 'graph_loop', 'smt_mem')   |    0.2 |        0.2 |         0.2 |      0.2 |
| ('smolvla', 'cam3_steps10_chunk50', 'hybrid', 'share')         |  -33.2 |      -17.3 |       -16.4 |    -16.5 |
| ('smolvla', 'cam3_steps10_chunk50', 'hybrid', 'smt_alu')       |  -11.9 |       13.2 |        17.7 |     12.6 |
| ('smolvla', 'cam3_steps10_chunk50', 'hybrid', 'smt_mem')       |  -21   |        2.1 |         1.5 |     11.1 | 

## MAPE (%) over conditions and configs

|                             |   none |   dispatch |   framework |   python |
|:----------------------------|-------:|-----------:|------------:|---------:|
| ('act', 'eager')            |   24   |       15.9 |        17.4 |     19.1 |
| ('act', 'graph_loop')       |   15.4 |       15.4 |        15.4 |     15.4 |
| ('smolvla', 'eager')        |   24   |       19.9 |        21.4 |     23.6 |
| ('smolvla', 'eager_loop')   |   22.9 |       20.6 |        22.2 |     24.3 |
| ('smolvla', 'eager_prefix') |   27.3 |       17.3 |        18.7 |     20.7 |
| ('smolvla', 'graph_loop')   |    2.7 |        2.7 |         2.7 |      2.7 |
| ('smolvla', 'hybrid')       |   20.5 |       11.5 |        12.5 |     14   | 

