# RoboTwin 2.0: inference delay vs task success

SmolVLA (`lerobot/smolvla_robotwin`, 480x640, 3 cameras), task `beat_block_hammer`, A100, 4 parallel envs. Delay d in control steps of 40 ms (25 Hz), naive async execution (the next chunk is planned d steps early and starts at index d, so a 50-action chunk executes 50 - d actions). Paired: same episode and noise seeds at every d.

## Runs

|   delay_steps |   episode_seed |   noise_seed |   job |   episodes |   successes |   inferences |   infer_ms_median |   wall_s_per_episode |
|--------------:|---------------:|-------------:|------:|-----------:|------------:|-------------:|------------------:|---------------------:|
|             0 |           1000 |        10000 | 34666 |         52 |          34 |          229 |             328.6 |                299.9 |
|             0 |           1052 |        20000 | 34667 |         48 |          29 |          274 |             328.2 |                330.5 |
|             3 |           1000 |        10000 | 34668 |         52 |          34 |          316 |             328.6 |                313.5 |
|             3 |           1052 |        20000 | 34669 |         48 |          29 |          274 |             324.7 |                328.9 |
|            10 |           1000 |        10000 | 34670 |         52 |          41 |          253 |             332.2 |                217.2 |
|            10 |           1052 |        20000 | 34671 |         48 |          29 |          360 |             325.3 |                322.8 |
|            20 |           1000 |        10000 | 34689 |         52 |          38 |          423 |             333.8 |                244.1 |
|            20 |           1052 |        20000 | 34673 |         48 |          35 |          375 |             327.2 |                236.6 | 

## Success rate per delay

|   delay_steps |   delay_ms |   episodes |   success_pct | ci95     |   lost_vs_d0 |   gained_vs_d0 |   mcnemar_p |
|--------------:|-----------:|-----------:|--------------:|:---------|-------------:|---------------:|------------:|
|             0 |          0 |        100 |            63 | 53 to 72 |          nan |            nan |       nan   |
|             3 |        120 |        100 |            63 | 53 to 72 |           17 |             17 |         1   |
|            10 |        400 |        100 |            70 | 60 to 78 |           18 |             25 |         0.4 |
|            20 |        800 |        100 |            73 | 64 to 81 |           17 |             27 |         0.2 | 

Scene effect: 28 of 100 episodes succeed at every d, 1 fail at every d; 71 change outcome with d.

## Success per half (seed block)

|   delay_steps | 1000   | 1052   |
|--------------:|:-------|:-------|
|             0 | 34/52  | 29/48  |
|             3 | 34/52  | 29/48  |
|            10 | 41/52  | 29/48  |
|            20 | 38/52  | 35/48  | 

## What each GPU and execution mode buys

Latency from the SmolVLA traces (3 cameras, 10 flow steps, chunk 50; the traced inputs are resized to 512, as RoboTwin's 480x640 frames are), converted to d = ceil(latency / 40 ms); success interpolated linearly between the measured delays.

|                                           |   latency_ms |   delay_steps |   expected_success_pct |
|:------------------------------------------|-------------:|--------------:|-----------------------:|
| ('A100-SXM4-40GB', 'eager')               |        382.7 |            10 |                   70   |
| ('A100-SXM4-40GB', 'prefix + graph loop') |         94.8 |             3 |                   63   |
| ('RTX 2080 Ti', 'eager')                  |        491.3 |            13 |                   70.9 |
| ('RTX 2080 Ti', 'prefix + graph loop')    |        116   |             3 |                   63   |
| ('RTX 4070', 'eager')                     |        611.8 |            16 |                   71.8 |
| ('RTX 4070', 'prefix + graph loop')       |        116.9 |             3 |                   63   |
| ('TITAN V', 'eager')                      |        343   |             9 |                   69   |
| ('TITAN V', 'prefix + graph loop')        |        119.3 |             3 |                   63   | 

The in-eval policy call (`infer_ms_median` above, eager, includes lerobot pre/post-processing) is the A100's eager latency in the actual harness.

## Reading

- No delay up to 20 steps (800 ms) lowers success on this task; the point estimates rise with d, but no change is significant (paired McNemar p >= 0.2). Unlike LIBERO-Long (results/libero_analysis.md), CUDA graphs buy no task success here: the curve is flat, so the GPU/mode table above only restates it.
- Confound: with a 50-action chunk and naive async execution, a larger d also means replanning every 50 - d steps instead of 50 (more closed-loop feedback, visible as more inferences per run). The sweep cannot separate staleness from replanning rate; a control would replan every 50 - d steps with no delay.
- Outcomes are noisy per episode: only 29 of 100 seeds have the same outcome at every d, so even paired, 100 episodes resolve only effects of roughly 15 points or more.
- beat_block_hammer is a slow, quasi-static task; one task does not establish that RoboTwin as a whole is delay insensitive.

