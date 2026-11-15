# torch.compile of the SmolVLA denoising loop (10 flow steps)

## Measured (ms)

| GPU     |   batch |   eager loop |   compiled, eager launch |   manual CUDA graph |   compiled + CUDA graph |   kernels (compiled) |   max |diff| vs eager |   first compile (s) |
|:--------|--------:|-------------:|-------------------------:|--------------------:|------------------------:|---------------------:|----------------------:|--------------------:|
| 2080Ti  |       1 |      413.008 |                 118.962  |             40.0866 |                 20.8135 |                 3190 |                0.0008 |             31.9901 |
| TITANV  |       1 |      289.919 |                  68.2572 |             61.3853 |                 30.9327 |                 3190 |                0.0009 |             29.0774 |
| RTX4070 |       1 |      503.519 |                 150.283  |             22.7778 |                 13.803  |                 3120 |                0.0007 |             33.1732 |
| 2080Ti  |       8 |      401.469 |                 112.839  |            111.254  |                 64.1011 |                 3310 |                0.0009 |            296.495  |
| TITANV  |       8 |      285.543 |                  71.2166 |            122.158  |                 63.4907 |                 3780 |                0.0011 |            269.44   |
| RTX4070 |       8 |      618.541 |                 186.044  |             72.8438 |                 46.0006 |                 3300 |                0.0012 |            324.767  | 

## Cross-GPU prediction error (%), slot_drv, from each source's compiled trace

|                                          |   2080Ti |   RTX4070 |   TITANV |
|:-----------------------------------------|---------:|----------:|---------:|
| (1, 'compiled + CUDA graph', '2080Ti')   |    nan   |       2.9 |     19.6 |
| (1, 'compiled + CUDA graph', 'RTX4070')  |     -2.6 |     nan   |     15.5 |
| (1, 'compiled + CUDA graph', 'TITANV')   |    -16.3 |     -13.9 |    nan   |
| (1, 'compiled, eager launch', '2080Ti')  |    nan   |     -28.6 |    -17.5 |
| (1, 'compiled, eager launch', 'RTX4070') |     31.8 |     nan   |     13   |
| (1, 'compiled, eager launch', 'TITANV')  |     20.1 |     -12.3 |    nan   |
| (8, 'compiled + CUDA graph', '2080Ti')   |    nan   |      12.4 |    -20.2 |
| (8, 'compiled + CUDA graph', 'RTX4070')  |    -10.2 |     nan   |    -29.2 |
| (8, 'compiled + CUDA graph', 'TITANV')   |     25.6 |      41   |    nan   |
| (8, 'compiled, eager launch', '2080Ti')  |    nan   |      -6.3 |     -9.9 |
| (8, 'compiled, eager launch', 'RTX4070') |      1.9 |     nan   |     -4.5 |
| (8, 'compiled, eager launch', 'TITANV')  |     15.1 |      28.8 |    nan   | 

MAPE (%): {(1, 'compiled + CUDA graph'): 11.8, (1, 'compiled, eager launch'): 20.6, (8, 'compiled + CUDA graph'): 23.1, (8, 'compiled, eager launch'): 11.1} 

Fitted per source (us): 2080Ti b1: b=0.47, p=16.1, TITANV b1: b=-0.14, p=6.3, RTX4070 b1: b=0.33, p=11.8, 2080Ti b8: b=0.70, p=14.8, TITANV b8: b=0.01, p=3.8, RTX4070 b8: b=0.20, p=21.4
