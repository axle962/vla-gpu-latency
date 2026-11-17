# RoboTwin 2.0: inference delay vs task success

SmolVLA (`lerobot/smolvla_robotwin`, 480x640, 3 cameras), task `beat_block_hammer`, A100, 4 parallel envs. Delay d in control steps of 40 ms (25 Hz), naive async execution (the next chunk is planned d steps early and starts at index d, so a 50-action chunk executes 50 - d actions). Paired: same episode and noise seeds at every d.

## Runs

| run   |   delay_steps |   replan_period |   episode_seed |   noise_seed |   job |   episodes |   successes |   inferences |   infer_ms_median |   wall_s_per_episode |
|:------|--------------:|----------------:|---------------:|-------------:|------:|-----------:|------------:|-------------:|------------------:|---------------------:|
| sweep |             0 |              50 |           1000 |        10000 | 34666 |         52 |          34 |          229 |             328.6 |                299.9 |
| sweep |             0 |              50 |           1052 |        20000 | 34667 |         48 |          29 |          274 |             328.2 |                330.5 |
| sweep |             3 |              47 |           1000 |        10000 | 34668 |         52 |          34 |          316 |             328.6 |                313.5 |
| sweep |             3 |              47 |           1052 |        20000 | 34669 |         48 |          29 |          274 |             324.7 |                328.9 |
| sweep |            10 |              40 |           1000 |        10000 | 34670 |         52 |          41 |          253 |             332.2 |                217.2 |
| sweep |            10 |              40 |           1052 |        20000 | 34671 |         48 |          29 |          360 |             325.3 |                322.8 |
| sweep |            20 |              30 |           1000 |        10000 | 34689 |         52 |          38 |          423 |             333.8 |                244.1 |
| sweep |            20 |              30 |           1052 |        20000 | 34673 |         48 |          35 |          375 |             327.2 |                236.6 |
| ctrl  |             0 |              40 |           1000 |        10000 | 34876 |         52 |          35 |          340 |             324.5 |                290.2 |
| ctrl  |             0 |              40 |           1052 |        20000 | 34877 |         48 |          35 |          263 |             324.5 |                263.1 |
| ctrl  |             0 |              30 |           1000 |        10000 | 34878 |         52 |          37 |          383 |             319.8 |                246.8 |
| ctrl  |             0 |              30 |           1052 |        20000 | 34879 |         48 |          35 |          382 |             326   |                247.8 | 

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

## Staleness vs replanning rate (control runs)

A delay d makes the policy plan every 50 - d steps from an observation d steps old. The control replans every H = 50 - d steps with no delay (synchronous, the chunk's first H actions), so it observes at the same steps; the only difference is staleness. Same episode and noise seeds, paired by episode.

|   replan_every |   control_success_pct | control_ci95   |   delay_d_success_pct |   delay_d | ctrl_vs_d0 (lost/gained, p)   | delay_vs_ctrl (lost/gained, p)   |
|---------------:|----------------------:|:---------------|----------------------:|----------:|:------------------------------|:---------------------------------|
|             40 |                    70 | 60 to 78       |                    70 |        10 | 17/24, 0.3489                 | 15/15, 1.0                       |
|             30 |                    72 | 63 to 80       |                    73 |        20 | 17/26, 0.2221                 | 13/14, 1.0                       | 

## Task screen (d = 0, 8 episodes each, seeds 1000..1007)

| task             | success   | ci95     |   wall_s_per_episode |
|:-----------------|:----------|:---------|---------------------:|
| handover_block   | 1/8       | 2 to 47  |                  605 |
| lift_pot         | 5/8       | 31 to 86 |                  427 |
| stack_blocks_two | 0/8       | 0 to 32  |                  661 | 

## What each GPU and execution mode buys

Latency from the SmolVLA traces (3 cameras, 10 flow steps, chunk 50; the traced inputs are resized to 512, as RoboTwin's 480x640 frames are), converted to d = ceil(latency / 40 ms); success interpolated linearly between the measured delays.

|                                                              |   latency_ms |   delay_steps |   expected_success_pct |
|:-------------------------------------------------------------|-------------:|--------------:|-----------------------:|
| ('A100-SXM4-40GB', 'eager')                                  |        382.7 |            10 |                   70   |
| ('A100-SXM4-40GB', 'prefix + graph loop')                    |         94.8 |             3 |                   63   |
| ('A100-SXM4-40GB (smolvla_robotwin)', 'eager')               |        377.6 |            10 |                   70   |
| ('A100-SXM4-40GB (smolvla_robotwin)', 'prefix + graph loop') |         93.9 |             3 |                   63   |
| ('RTX 2080 Ti', 'eager')                                     |        491.3 |            13 |                   70.9 |
| ('RTX 2080 Ti', 'prefix + graph loop')                       |        116   |             3 |                   63   |
| ('RTX 4070', 'eager')                                        |        611.8 |            16 |                   71.8 |
| ('RTX 4070', 'prefix + graph loop')                          |        116.9 |             3 |                   63   |
| ('TITAN V', 'eager')                                         |        343   |             9 |                   69   |
| ('TITAN V', 'prefix + graph loop')                           |        119.3 |             3 |                   63   | 

The in-eval policy call (`infer_ms_median` above, eager, includes lerobot pre/post-processing) is the A100's eager latency in the actual harness.

## Reading

- No delay up to 20 steps (800 ms) lowers success on beat_block_hammer (paired McNemar p >= 0.2 vs d = 0).
- The control settles the confound: at the same replanning period, a stale plan (delay d) and a fresh plan (no delay) succeed equally often, episode for episode (discordant pairs 15/15 at d = 10, 13/14 at d = 20). Staleness costs nothing here; the small rise with d comes from replanning more often (also not significant).
- So on this task CUDA graphs buy no task success, unlike LIBERO-Long (results/libero_analysis.md); their value is latency and energy only. The smolvla_robotwin checkpoint has the base model's latency (A100 eager 378 ms, graph loop 35.4 ms, hybrid 94 ms; graph output bit-exact).
- Outcomes are noisy per episode (71 of 100 seeds change outcome across d), so 100 paired episodes resolve only effects of about 15 points or more.
- One slow, quasi-static task. Of three screened alternatives only lift_pot has a usable success rate for a delay study (handover_block 1/8, stack_blocks_two 0/8).

