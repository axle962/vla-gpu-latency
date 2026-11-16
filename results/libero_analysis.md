# LIBERO-Long: inference delay vs task success

Suite libero_10; delay d in control steps of 50 ms (20 Hz). Paired: same scenes and noise seeds at every d.

## Success rate per delay

| policy   | precision   |   delay_steps |   delay_ms |   episodes |   success_pct | ci95     |   lost_vs_d0 |   gained_vs_d0 |   mcnemar_p |
|:---------|:------------|--------------:|-----------:|-----------:|--------------:|:---------|-------------:|---------------:|------------:|
| pi0      | bf16        |             0 |          0 |        100 |            39 | 30 to 49 |          nan |            nan |       nan   |
| pi0      | fp16        |             0 |          0 |        100 |            38 | 29 to 48 |          nan |            nan |       nan   |
| pi0      | fp16        |             2 |        100 |        100 |            37 | 28 to 47 |           12 |             11 |         1   |
| pi0      | fp16        |             5 |        250 |        100 |            34 | 25 to 44 |           15 |             11 |         0.6 |
| pi0      | fp16        |            10 |        500 |        100 |            27 | 19 to 36 |           22 |             11 |         0.1 |
| pi0      | fp16        |            15 |        750 |        100 |            22 | 15 to 31 |           27 |             11 |         0   |
| pi0      | fp16        |            20 |       1000 |        100 |            15 | 9 to 23  |           31 |              8 |         0   |
| pi05     | bf16        |             0 |          0 |        100 |            83 | 74 to 89 |          nan |            nan |       nan   |
| pi05     | fp16        |             0 |          0 |        100 |            87 | 79 to 92 |          nan |            nan |       nan   |
| pi05     | fp16        |             2 |        100 |        100 |            79 | 70 to 86 |           15 |              7 |         0.1 |
| pi05     | fp16        |             5 |        250 |        100 |            76 | 67 to 83 |           18 |              7 |         0   |
| pi05     | fp16        |            10 |        500 |        100 |            63 | 53 to 72 |           31 |              7 |         0   |
| pi05     | fp16        |            15 |        750 |        100 |            65 | 55 to 74 |           26 |              4 |         0   |
| pi05     | fp16        |            20 |       1000 |        100 |            48 | 38 to 58 |           45 |              6 |         0   |
| smolvla  | fp16        |             0 |          0 |        100 |            50 | 40 to 60 |          nan |            nan |       nan   |
| smolvla  | fp16        |             2 |        100 |        100 |            49 | 39 to 59 |           16 |             15 |         1   |
| smolvla  | fp16        |             5 |        250 |        100 |            51 | 41 to 61 |           15 |             16 |         1   |
| smolvla  | fp16        |            10 |        500 |        100 |            44 | 35 to 54 |           21 |             15 |         0.4 |
| smolvla  | fp16        |            15 |        750 |        100 |            44 | 35 to 54 |           22 |             16 |         0.4 |
| smolvla  | fp16        |            20 |       1000 |        100 |            34 | 25 to 44 |           27 |             11 |         0   |
| smolvla  | fp32        |             0 |          0 |        100 |            58 | 48 to 67 |          nan |            nan |       nan   | 

## Per task at d = 0 (success / episodes)

|                     | 0    | 1     | 2    | 3    | 4    | 5     | 6    | 7    | 8    | 9    |
|:--------------------|:-----|:------|:-----|:-----|:-----|:------|:-----|:-----|:-----|:-----|
| ('pi0', 'bf16')     | 4/10 | 6/10  | 2/10 | 7/10 | 0/10 | 7/10  | 4/10 | 2/10 | 2/10 | 5/10 |
| ('pi0', 'fp16')     | 4/10 | 6/10  | 2/10 | 7/10 | 0/10 | 7/10  | 3/10 | 2/10 | 2/10 | 5/10 |
| ('pi05', 'bf16')    | 7/10 | 10/10 | 9/10 | 7/10 | 8/10 | 10/10 | 9/10 | 8/10 | 7/10 | 8/10 |
| ('pi05', 'fp16')    | 8/10 | 10/10 | 9/10 | 9/10 | 9/10 | 10/10 | 9/10 | 8/10 | 7/10 | 8/10 |
| ('smolvla', 'fp16') | 2/10 | 7/10  | 8/10 | 8/10 | 4/10 | 5/10  | 3/10 | 2/10 | 3/10 | 8/10 |
| ('smolvla', 'fp32') | 5/10 | 7/10  | 9/10 | 8/10 | 2/10 | 8/10  | 3/10 | 3/10 | 6/10 | 7/10 | 

## What each GPU and execution mode buys (FP16 sweeps)

|                                                      |   latency_ms |   delay_steps |   expected_success_pct |
|:-----------------------------------------------------|-------------:|--------------:|-----------------------:|
| ('pi0', 'RTX 2080 Ti', 'eager')                      |        507.3 |          11   |                   26   |
| ('pi0', 'RTX 2080 Ti', 'prefix + graph loop')        |        210.8 |           5   |                   34   |
| ('pi0', 'RTX 4070', 'eager')                         |        598.8 |          12.5 |                   24.5 |
| ('pi0', 'RTX 4070', 'prefix + graph loop')           |        198.6 |           4   |                   35   |
| ('pi0', 'TITAN V', 'eager')                          |        359.5 |           8   |                   29.8 |
| ('pi0', 'TITAN V', 'prefix + graph loop')            |        196.5 |           4   |                   35   |
| ('pi05', 'RTX 2080 Ti', 'eager')                     |        564.5 |          12   |                   63.8 |
| ('pi05', 'RTX 2080 Ti', 'prefix + graph loop')       |        228.7 |           5   |                   76   |
| ('pi05', 'RTX 4070', 'eager')                        |        712.7 |          15   |                   65   |
| ('pi05', 'RTX 4070', 'prefix + graph loop')          |        229.2 |           5   |                   76   |
| ('pi05', 'TITAN V', 'eager')                         |        425   |           9   |                   65.6 |
| ('pi05', 'TITAN V', 'prefix + graph loop')           |        218.7 |           5   |                   76   |
| ('smolvla', 'A100-SXM4-40GB', 'eager')               |        382.7 |           8   |                   46.8 |
| ('smolvla', 'A100-SXM4-40GB', 'prefix + graph loop') |         94.8 |           2   |                   49   |
| ('smolvla', 'RTX 2080 Ti', 'eager')                  |        491.3 |          10   |                   44   |
| ('smolvla', 'RTX 2080 Ti', 'prefix + graph loop')    |        116   |           3   |                   49.7 |
| ('smolvla', 'RTX 4070', 'eager')                     |        611.8 |          13   |                   44   |
| ('smolvla', 'RTX 4070', 'prefix + graph loop')       |        116.9 |           3   |                   49.7 |
| ('smolvla', 'TITAN V', 'eager')                      |        343   |           7   |                   48.2 |
| ('smolvla', 'TITAN V', 'prefix + graph loop')        |        119.3 |           3   |                   49.7 | 

nan: the delay exceeds the largest simulated d.

