# GPU energy per SmolVLA inference (workloads/energy_smolvla.py)

GPU board energy only (NVML energy counter), 20 s per mode, back-to-back steps; eager = stock step, hybrid = eager prefix + CUDA-graph loop, graph loop = the loop alone. Per robot observation at batch 1; at batch 8 the step serves 8 observations.

|                    |   eager ms |   hybrid ms |   eager J |   hybrid J |   energy saving (eager/hybrid) |   eager W |   hybrid W |   graph loop J |
|:-------------------|-----------:|------------:|----------:|-----------:|-------------------------------:|----------:|-----------:|---------------:|
| (1, 'RTX 2080 Ti') |     471.24 |      114.7  |     19.05 |      10.04 |                           1.9  |     69.6  |     164.34 |           3.71 |
| (1, 'RTX 4070')    |     733.53 |      133.49 |     31.45 |      11.47 |                           2.74 |     41.46 |      84.34 |           2.75 |
| (1, 'TITAN V')     |     343    |      118.96 |     16.83 |      10.28 |                           1.64 |     48.64 |      84.79 |           4.17 |
| (8, 'RTX 2080 Ti') |     642.96 |      328.83 |     58.15 |      43.45 |                           1.34 |    166.15 |     242.4  |          15.11 |
| (8, 'RTX 4070')    |     865.07 |      311.24 |     78.56 |      52.52 |                           1.5  |     87.37 |     165.82 |          14.05 |
| (8, 'TITAN V')     |     467.58 |      305.54 |     56.07 |      51.76 |                           1.08 |    119.38 |     169.66 |          15.66 | 

