# torch.compile of the SmolVLA denoising loop (10 flow steps)

## Measured (ms)

| GPU     |   batch |   eager loop |   compiled, eager launch |   manual CUDA graph |   compiled + CUDA graph |   kernels (compiled) |   max |diff| vs eager |   first compile (s) |
|:--------|--------:|-------------:|-------------------------:|--------------------:|------------------------:|---------------------:|----------------------:|--------------------:|
| 2080Ti  |       1 |      519.802 |                 147.561  |             40.0307 |                 21.0284 |                 3190 |                0.0008 |             33.0749 |
| TITANV  |       1 |      289.919 |                  68.2572 |             61.3853 |                 30.9327 |                 3190 |                0.0009 |             29.0774 |
| RTX4070 |       1 |      503.519 |                 150.283  |             22.7778 |                 13.803  |                 3120 |                0.0007 |             33.1732 |
| 2080Ti  |       8 |      401.469 |                 112.839  |            111.254  |                 64.1011 |                 3310 |                0.0009 |            296.495  |
| TITANV  |       8 |      285.543 |                  71.2166 |            122.158  |                 63.4907 |                 3780 |                0.0011 |            269.44   |
| RTX4070 |       8 |      618.541 |                 186.044  |             72.8438 |                 46.0006 |                 3300 |                0.0012 |            324.767  | 

## Cross-GPU prediction error (%), slot_drv, from each source's compiled trace

|                                          |   2080Ti |   RTX4070 |   TITANV |
|:-----------------------------------------|---------:|----------:|---------:|
| (1, 'compiled + CUDA graph', '2080Ti')   |    nan   |       1.9 |     18.4 |
| (1, 'compiled + CUDA graph', 'RTX4070')  |     -1.7 |     nan   |     15.5 |
| (1, 'compiled + CUDA graph', 'TITANV')   |    -15.4 |     -13.9 |    nan   |
| (1, 'compiled, eager launch', '2080Ti')  |    nan   |     -42.4 |    -33.5 |
| (1, 'compiled, eager launch', 'RTX4070') |     60.5 |     nan   |     13   |
| (1, 'compiled, eager launch', 'TITANV')  |     47.8 |     -12.3 |    nan   |
| (8, 'compiled + CUDA graph', '2080Ti')   |    nan   |      12.4 |    -20.2 |
| (8, 'compiled + CUDA graph', 'RTX4070')  |    -10.2 |     nan   |    -29.2 |
| (8, 'compiled + CUDA graph', 'TITANV')   |     25.6 |      41   |    nan   |
| (8, 'compiled, eager launch', '2080Ti')  |    nan   |      -6.3 |     -9.9 |
| (8, 'compiled, eager launch', 'RTX4070') |      1.9 |     nan   |     -4.5 |
| (8, 'compiled, eager launch', 'TITANV')  |     15.1 |      28.8 |    nan   | 

MAPE (%): {(1, 'compiled + CUDA graph'): 11.1, (1, 'compiled, eager launch'): 34.9, (8, 'compiled + CUDA graph'): 23.1, (8, 'compiled, eager launch'): 11.1} 

Fitted per source (us): 2080Ti b1: b=0.40, p=25.1, TITANV b1: b=-0.14, p=6.3, RTX4070 b1: b=0.33, p=11.8, 2080Ti b8: b=0.70, p=14.8, TITANV b8: b=0.01, p=3.8, RTX4070 b8: b=0.20, p=21.4
