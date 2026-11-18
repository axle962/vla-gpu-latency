# Multi-policy hosting on one GPU

GPU `NVIDIA RTX PRO 6000 Blackwell Server Edition` (driver 580.173.02), host `alphagpu54` (Intel(R) Xeon(R) 6960P, 16 CPUs in the affinity mask), torch 2.10.0+cu128 cu12.8. Policy settings: {"cameras": 3, "num_steps": 10, "chunk_size": 50, "diffusion_steps": 10, "act_hw": [480, 640]}. Closed loop (a new step as soon as the previous one returned), median of the middle 80% of each window.

## Policies alone (procs x1)

| policy    | mode   |   step ms |   p95 ms |   GPU busy ms (profiled) |   GPU busy ms (corrected D) |   host ms (Z = step - D) |   GPU share |   kernels/step |
|:----------|:-------|----------:|---------:|-------------------------:|----------------------------:|-------------------------:|------------:|---------------:|
| smolvla   | eager  |     173.7 |    174.8 |                     40.1 |                        27.3 |                    146.4 |        0.16 |          13542 |
| smolvla   | graph  |      59.6 |     59.8 |                     39.3 |                        26.1 |                     33.6 |        0.44 |          13962 |
| smolvla   | loop   |      32.4 |     32.5 |                     27.9 |                        16.4 |                     16   |        0.51 |          12080 |
| act       | eager  |       4.9 |      4.9 |                      5.1 |                         4.7 |                      0.1 |        0.97 |            373 |
| act       | graph  |       4.1 |      4.1 |                      5.2 |                         4.8 |                     -0.7 |        1.17 |            375 |
| act       | loop   |       4.1 |      4.1 |                      5.2 |                         4.8 |                     -0.8 |        1.19 |            375 |
| diffusion | eager  |     136.5 |    136.6 |                    131.6 |                       129   |                      7.5 |        0.95 |           2764 |
| diffusion | graph  |     132   |    132.2 |                    131.6 |                       128.9 |                      3.2 |        0.98 |           2904 |
| diffusion | loop   |     130.5 |    130.5 |                    130.5 |                       127.9 |                      2.6 |        0.98 |           2759 |

D removes the profiler's 0.95 us per kernel (CLAUDE.md 0.1); Z is what the step spends off the GPU (Python, dispatch, syncs). A policy with a small GPU share leaves room for others.

## Measured: every hosted configuration

| config                          | mode   | policy    | copies   |   median ms |   p95 ms | vs alone   |   steps/s (all policies) |   steps/copy |
|:--------------------------------|:-------|:----------|:---------|------------:|---------:|:-----------|-------------------------:|-------------:|
| procs_smolvla                   | eager  | smolvla   | 1        |       173.7 |    174.8 | 1.00x      |                     5.88 |           47 |
| procs_smolvla                   | graph  | smolvla   | 1        |        59.6 |     59.8 | 1.00x      |                    16.75 |          134 |
| procs_smolvla                   | loop   | smolvla   | 1        |        32.4 |     32.5 | 1.00x      |                    30.88 |          247 |
| procs_smolvlax2                 | eager  | smolvla   | 2        |       181.1 |    182.8 | 1.04x      |                    11.12 |           44 |
| procs_smolvlax2                 | graph  | smolvla   | 2        |        98.5 |     98.9 | 1.65x      |                    20.5  |           82 |
| procs_smolvlax2                 | loop   | smolvla   | 2        |        60.4 |     60.6 | 1.87x      |                    33    |          132 |
| procs_smolvlax4                 | eager  | smolvla   | 4        |       263.3 |    263.8 | 1.52x      |                    15    |           30 |
| procs_smolvlax4                 | graph  | smolvla   | 4        |       186.1 |    187.7 | 3.12x      |                    21.5  |           43 |
| procs_smolvlax4                 | loop   | smolvla   | 4        |       121.1 |    121.5 | 3.74x      |                    33    |           66 |
| procs_smolvlax8                 | eager  | smolvla   | 8        |       512.2 |    514   | 2.95x      |                    15.88 |           16 |
| procs_smolvlax8                 | graph  | smolvla   | 8        |       374.9 |    379.2 | 6.29x      |                    21.25 |           21 |
| procs_smolvlax8                 | loop   | smolvla   | 8        |        28.2 |     28.2 | 0.87x      |                     1    |            1 |
| procs-mps_smolvlax2             | eager  | smolvla   | 2        |       170.3 |    173   | 0.98x      |                    11.62 |           46 |
| procs-mps_smolvlax2             | graph  | smolvla   | 2        |        69.7 |     70.2 | 1.17x      |                    28.62 |          114 |
| procs-mps_smolvlax2             | loop   | smolvla   | 2        |        41.3 |     41.5 | 1.28x      |                    48.5  |          194 |
| procs-mps_smolvlax4             | eager  | smolvla   | 4        |       162.4 |    164.4 | 0.93x      |                    24.62 |           49 |
| procs-mps_smolvlax4             | graph  | smolvla   | 4        |        83.2 |     83.8 | 1.40x      |                    48.25 |           96 |
| procs-mps_smolvlax4             | loop   | smolvla   | 4        |        47.7 |     47.9 | 1.47x      |                    83.88 |          168 |
| procs-mps_smolvlax8             | eager  | smolvla   | 8        |       155.3 |    157.7 | 0.89x      |                    51.62 |           52 |
| procs-mps_smolvlax8             | graph  | smolvla   | 8        |       111.1 |    112.2 | 1.86x      |                    72.12 |           72 |
| procs-mps_smolvlax8             | loop   | smolvla   | 8        |        56.7 |     57.2 | 1.75x      |                   140.75 |          141 |
| threads_smolvlax2               | eager  | smolvla   | 2        |       373.4 |    476.7 | 2.15x      |                     5.12 |           20 |
| threads_smolvlax2               | graph  | smolvla   | 2        |        78.4 |     80.1 | 1.31x      |                    25.75 |          103 |
| threads_smolvlax2               | loop   | smolvla   | 2        |        38.4 |     38.5 | 1.19x      |                    52    |          208 |
| threads_smolvlax4               | eager  | smolvla   | 4        |      1426.5 |   1589.5 | 8.21x      |                     2.88 |            6 |
| threads_smolvlax4               | graph  | smolvla   | 4        |       169.2 |    225   | 2.84x      |                    24.5  |           49 |
| threads_smolvlax4               | loop   | smolvla   | 4        |        46   |     46.1 | 1.42x      |                    87    |          174 |
| threads_smolvlax8               | eager  | smolvla   | 8        |     39431.6 |  39431.6 | 227.05x    |                     1    |            1 |
| threads_smolvlax8               | graph  | smolvla   | 8        |      5433.8 |   5623.9 | 91.15x     |                     2    |            2 |
| threads_smolvlax8               | loop   | smolvla   | 8        |        54.3 |     55.3 | 1.68x      |                   147    |          147 |
| procs_act                       | eager  | act       | 1        |         4.9 |      4.9 | 1.00x      |                   205.75 |         1646 |
| procs_act                       | graph  | act       | 1        |         4.1 |      4.1 | 1.00x      |                   242.38 |         1939 |
| procs_act                       | loop   | act       | 1        |         4.1 |      4.1 | 1.00x      |                   245.62 |         1965 |
| procs_actx2                     | eager  | act       | 2        |         9.5 |      9.6 | 1.96x      |                   209.88 |          839 |
| procs_actx2                     | graph  | act       | 2        |         8.3 |      8.5 | 2.01x      |                   241    |          964 |
| procs_actx2                     | loop   | act       | 2        |         8.2 |      8.4 | 2.03x      |                   242.38 |          969 |
| procs_actx4                     | eager  | act       | 4        |        19.1 |     19.2 | 3.93x      |                   209.5  |          419 |
| procs_actx4                     | graph  | act       | 4        |        16.9 |     17   | 4.09x      |                   237.25 |          474 |
| procs_actx4                     | loop   | act       | 4        |        16.5 |     16.6 | 4.05x      |                   242.5  |          485 |
| procs_actx8                     | eager  | act       | 8        |        38.3 |     38.6 | 7.89x      |                   209    |          209 |
| procs_actx8                     | graph  | act       | 8        |        33.7 |     33.9 | 8.16x      |                   238.5  |          238 |
| procs_actx8                     | loop   | act       | 8        |        33.1 |     33.3 | 8.13x      |                   241.5  |          241 |
| procs-mps_actx2                 | eager  | act       | 2        |         5.9 |      6   | 1.21x      |                   340.5  |         1362 |
| procs-mps_actx2                 | graph  | act       | 2        |         5.4 |      5.6 | 1.30x      |                   371.75 |         1487 |
| procs-mps_actx2                 | loop   | act       | 2        |         5.4 |      5.6 | 1.32x      |                   371.62 |         1486 |
| procs-mps_actx4                 | eager  | act       | 4        |         8.2 |      8.5 | 1.69x      |                   485    |          973 |
| procs-mps_actx4                 | graph  | act       | 4        |         7.9 |      8.3 | 1.93x      |                   502.88 |         1006 |
| procs-mps_actx4                 | loop   | act       | 4        |         8   |      8.3 | 1.96x      |                   502.38 |         1005 |
| procs-mps_actx8                 | eager  | act       | 8        |        14.1 |     14.6 | 2.91x      |                   564.38 |          566 |
| procs-mps_actx8                 | graph  | act       | 8        |        14.1 |     14.6 | 3.42x      |                   567    |          567 |
| procs-mps_actx8                 | loop   | act       | 8        |        14.1 |     14.7 | 3.47x      |                   566.25 |          566 |
| threads_actx2                   | eager  | act       | 2        |        10.3 |     12.2 | 2.12x      |                   189.38 |          757 |
| threads_actx2                   | graph  | act       | 2        |         5.2 |      5.4 | 1.25x      |                   387.5  |         1550 |
| threads_actx2                   | loop   | act       | 2        |         5.2 |      5.4 | 1.27x      |                   387.25 |         1549 |
| threads_actx4                   | eager  | act       | 4        |        39.3 |     58.2 | 8.10x      |                    95.62 |          192 |
| threads_actx4                   | graph  | act       | 4        |         7.7 |      8   | 1.87x      |                   518    |         1036 |
| threads_actx4                   | loop   | act       | 4        |         7.7 |      8   | 1.90x      |                   517.25 |         1034 |
| threads_actx8                   | eager  | act       | 8        |      1141.6 |   1221.1 | 235.20x    |                     7    |            7 |
| threads_actx8                   | graph  | act       | 8        |        13.8 |     14.4 | 3.35x      |                   578    |          578 |
| threads_actx8                   | loop   | act       | 8        |        13.9 |     14.5 | 3.41x      |                   575.25 |          575 |
| procs_diffusion                 | eager  | diffusion | 1        |       136.5 |    136.6 | 1.00x      |                     7.38 |           59 |
| procs_diffusion                 | graph  | diffusion | 1        |       132   |    132.2 | 1.00x      |                     7.5  |           60 |
| procs_diffusion                 | loop   | diffusion | 1        |       130.5 |    130.5 | 1.00x      |                     7.75 |           62 |
| procs_diffusionx2               | eager  | diffusion | 2        |       288.8 |    288.9 | 2.12x      |                     7    |           28 |
| procs_diffusionx2               | graph  | diffusion | 2        |       281.3 |    281.4 | 2.13x      |                     7    |           28 |
| procs_diffusionx2               | loop   | diffusion | 2        |       278.5 |    278.8 | 2.13x      |                     7.25 |           29 |
| procs_diffusionx4               | eager  | diffusion | 4        |       577.9 |    578.1 | 4.23x      |                     7    |           14 |
| procs_diffusionx4               | graph  | diffusion | 4        |       562.7 |    563.3 | 4.26x      |                     7    |           14 |
| procs_diffusionx4               | loop   | diffusion | 4        |       558.8 |    558.9 | 4.28x      |                     7.5  |           15 |
| procs_diffusionx8               | eager  | diffusion | 8        |      1164   |   1172   | 8.53x      |                     7    |            7 |
| procs_diffusionx8               | graph  | diffusion | 8        |      1133.3 |   1133.4 | 8.58x      |                     7    |            7 |
| procs_diffusionx8               | loop   | diffusion | 8        |      1123.9 |   1124.2 | 8.61x      |                     7.12 |            7 |
| procs-mps_diffusionx2           | eager  | diffusion | 2        |       140   |    141.8 | 1.03x      |                    14.25 |           57 |
| procs-mps_diffusionx2           | graph  | diffusion | 2        |       134.2 |    136.2 | 1.02x      |                    14.75 |           59 |
| procs-mps_diffusionx2           | loop   | diffusion | 2        |       132.3 |    132.9 | 1.01x      |                    15.12 |           60 |
| procs-mps_diffusionx4           | eager  | diffusion | 4        |       156.6 |    157.1 | 1.15x      |                    25.5  |           51 |
| procs-mps_diffusionx4           | graph  | diffusion | 4        |       151.9 |    152.4 | 1.15x      |                    26.12 |           52 |
| procs-mps_diffusionx4           | loop   | diffusion | 4        |       149.8 |    150.3 | 1.15x      |                    27    |           54 |
| procs-mps_diffusionx8           | eager  | diffusion | 8        |       221.9 |    223   | 1.63x      |                    36    |           36 |
| procs-mps_diffusionx8           | graph  | diffusion | 8        |       218.1 |    219.3 | 1.65x      |                    37    |           37 |
| procs-mps_diffusionx8           | loop   | diffusion | 8        |       215.6 |    216.7 | 1.65x      |                    37    |           37 |
| threads_diffusionx2             | eager  | diffusion | 2        |       139.8 |    142.7 | 1.02x      |                    14.25 |           57 |
| threads_diffusionx2             | graph  | diffusion | 2        |       133.7 |    134.5 | 1.01x      |                    15    |           60 |
| threads_diffusionx2             | loop   | diffusion | 2        |       134   |    134.8 | 1.03x      |                    14.75 |           59 |
| threads_diffusionx4             | eager  | diffusion | 4        |       253   |    367.4 | 1.85x      |                    14.88 |           30 |
| threads_diffusionx4             | graph  | diffusion | 4        |       153.3 |    154.7 | 1.16x      |                    26.25 |           52 |
| threads_diffusionx4             | loop   | diffusion | 4        |       149.5 |    150   | 1.15x      |                    27    |           54 |
| threads_diffusionx8             | eager  | diffusion | 8        |      5446.3 |   5523.3 | 39.89x     |                     1.25 |            1 |
| threads_diffusionx8             | graph  | diffusion | 8        |       217.9 |    219   | 1.65x      |                    36.62 |           37 |
| threads_diffusionx8             | loop   | diffusion | 8        |       215.6 |    216.7 | 1.65x      |                    37    |           37 |
| batch_smolvlax4_b4              | eager  | smolvla   | batch 4  |       180.2 |    181.8 | 1.04x      |                    22.5  |           45 |
| batch_smolvlax4_b4              | graph  | smolvla   | batch 4  |        77   |     77.5 | 1.29x      |                    52    |          104 |
| batch_smolvlax4_b4              | loop   | smolvla   | batch 4  |        33.8 |     34.1 | 1.04x      |                   118    |          236 |
| batch_smolvlax8_b8              | eager  | smolvla   | batch 8  |       202.4 |    204.1 | 1.17x      |                    39    |           39 |
| batch_smolvlax8_b8              | graph  | smolvla   | batch 8  |        98.6 |    100.2 | 1.65x      |                    81    |           81 |
| batch_smolvlax8_b8              | loop   | smolvla   | batch 8  |        39.3 |     39.3 | 1.21x      |                   205    |          205 |
| procs_smolvla+act               | eager  | smolvla   | 1        |       203.2 |    203.7 | 1.17x      |                   137.88 |           40 |
| procs_smolvla+act               | eager  | act       | 1        |         7.6 |      8.6 | 1.56x      |                   137.88 |         1063 |
| procs_smolvla+act               | graph  | smolvla   | 1        |       127.9 |    129.3 | 2.15x      |                   155.38 |           63 |
| procs_smolvla+act               | graph  | act       | 1        |         6.6 |      8.8 | 1.61x      |                   155.38 |         1180 |
| procs_smolvla+act               | loop   | smolvla   | 1        |        60.4 |     61   | 1.86x      |                   132.5  |          132 |
| procs_smolvla+act               | loop   | act       | 1        |         8.7 |      8.8 | 2.13x      |                   132.5  |          928 |
| procs-mps_smolvla+act           | eager  | smolvla   | 1        |       171.7 |    172.8 | 0.99x      |                   199    |           47 |
| procs-mps_smolvla+act           | eager  | act       | 1        |         5.1 |      6.1 | 1.04x      |                   199    |         1545 |
| procs-mps_smolvla+act           | graph  | smolvla   | 1        |        77.8 |     79.1 | 1.30x      |                   226.62 |          103 |
| procs-mps_smolvla+act           | graph  | act       | 1        |         4.6 |      5.4 | 1.11x      |                   226.62 |         1710 |
| procs-mps_smolvla+act           | loop   | smolvla   | 1        |        47.5 |     48.5 | 1.47x      |                   241.38 |          169 |
| procs-mps_smolvla+act           | loop   | act       | 1        |         4.5 |      4.8 | 1.11x      |                   241.38 |         1762 |
| procs_smolvla+diffusion         | eager  | smolvla   | 1        |       247.3 |    249.7 | 1.42x      |                     9.12 |           32 |
| procs_smolvla+diffusion         | eager  | diffusion | 1        |       196.2 |    199.6 | 1.44x      |                     9.12 |           41 |
| procs_smolvla+diffusion         | graph  | smolvla   | 1        |       127.4 |    128.8 | 2.14x      |                    12.38 |           63 |
| procs_smolvla+diffusion         | graph  | diffusion | 1        |       225.6 |    230.3 | 1.71x      |                    12.38 |           36 |
| procs_smolvla+diffusion         | loop   | smolvla   | 1        |        60.8 |     61   | 1.88x      |                    20.12 |          132 |
| procs_smolvla+diffusion         | loop   | diffusion | 1        |       277.7 |    278.7 | 2.13x      |                    20.12 |           29 |
| procs-mps_smolvla+diffusion     | eager  | smolvla   | 1        |       167.2 |    169.3 | 0.96x      |                    12.75 |           48 |
| procs-mps_smolvla+diffusion     | eager  | diffusion | 1        |       147.9 |    152   | 1.08x      |                    12.75 |           54 |
| procs-mps_smolvla+diffusion     | graph  | smolvla   | 1        |        63.6 |     64.3 | 1.07x      |                    22.12 |          125 |
| procs-mps_smolvla+diffusion     | graph  | diffusion | 1        |       154.5 |    160.8 | 1.17x      |                    22.12 |           52 |
| procs-mps_smolvla+diffusion     | loop   | smolvla   | 1        |        36.5 |     36.6 | 1.13x      |                    34.25 |          220 |
| procs-mps_smolvla+diffusion     | loop   | diffusion | 1        |       146.6 |    153.8 | 1.12x      |                    34.25 |           54 |
| procs_act+diffusion             | eager  | act       | 1        |        13.5 |     13.7 | 2.79x      |                    79    |          597 |
| procs_act+diffusion             | eager  | diffusion | 1        |       227.5 |    227.9 | 1.67x      |                    79    |           35 |
| procs_act+diffusion             | graph  | act       | 1        |         8.7 |      8.7 | 2.11x      |                   118.75 |          922 |
| procs_act+diffusion             | graph  | diffusion | 1        |       286.4 |    286.6 | 2.17x      |                   118.75 |           28 |
| procs_act+diffusion             | loop   | act       | 1        |         8.7 |      8.7 | 2.14x      |                   118.5  |          920 |
| procs_act+diffusion             | loop   | diffusion | 1        |       282.8 |    283   | 2.17x      |                   118.5  |           28 |
| procs-mps_act+diffusion         | eager  | act       | 1        |         5   |      5.1 | 1.04x      |                   204    |         1585 |
| procs-mps_act+diffusion         | eager  | diffusion | 1        |       171.5 |    174   | 1.26x      |                   204    |           47 |
| procs-mps_act+diffusion         | graph  | act       | 1        |         4.4 |      4.7 | 1.06x      |                   233.12 |         1820 |
| procs-mps_act+diffusion         | graph  | diffusion | 1        |       176.1 |    181.8 | 1.33x      |                   233.12 |           45 |
| procs-mps_act+diffusion         | loop   | act       | 1        |         4.3 |      4.7 | 1.06x      |                   233.88 |         1825 |
| procs-mps_act+diffusion         | loop   | diffusion | 1        |       173.7 |    177.4 | 1.33x      |                   233.88 |           46 |
| threads_act+diffusion           | eager  | act       | 1        |         5.6 |     11.4 | 1.15x      |                   153.62 |         1182 |
| threads_act+diffusion           | eager  | diffusion | 1        |       172   |    179   | 1.26x      |                   153.62 |           47 |
| threads_act+diffusion           | graph  | act       | 1        |         4.1 |      4.2 | 1.00x      |                   247.12 |         1931 |
| threads_act+diffusion           | graph  | diffusion | 1        |       175.4 |    181.6 | 1.33x      |                   247.12 |           46 |
| threads_act+diffusion           | loop   | act       | 1        |         4.1 |      4.2 | 1.01x      |                   247.12 |         1931 |
| threads_act+diffusion           | loop   | diffusion | 1        |       173.7 |    178.1 | 1.33x      |                   247.12 |           46 |
| procs_smolvla+act+diffusion     | eager  | smolvla   | 1        |       279.7 |    281.2 | 1.61x      |                    64.25 |           29 |
| procs_smolvla+act+diffusion     | eager  | act       | 1        |        18.2 |     18.5 | 3.76x      |                    64.25 |          458 |
| procs_smolvla+act+diffusion     | eager  | diffusion | 1        |       293.6 |    296.6 | 2.15x      |                    64.25 |           27 |
| procs_smolvla+act+diffusion     | graph  | smolvla   | 1        |       198.3 |    203   | 3.33x      |                    96.25 |           40 |
| procs_smolvla+act+diffusion     | graph  | act       | 1        |        11.1 |     13.2 | 2.69x      |                    96.25 |          709 |
| procs_smolvla+act+diffusion     | graph  | diffusion | 1        |       374.1 |    377.4 | 2.83x      |                    96.25 |           21 |
| procs_smolvla+act+diffusion     | loop   | smolvla   | 1        |        91   |     91.4 | 2.81x      |                    90.12 |           87 |
| procs_smolvla+act+diffusion     | loop   | act       | 1        |        13.1 |     13.2 | 3.22x      |                    90.12 |          615 |
| procs_smolvla+act+diffusion     | loop   | diffusion | 1        |       422.4 |    423.7 | 3.24x      |                    90.12 |           19 |
| procs-mps_smolvla+act+diffusion | eager  | smolvla   | 1        |       171.6 |    174.2 | 0.99x      |                   200.5  |           48 |
| procs-mps_smolvla+act+diffusion | eager  | act       | 1        |         5.2 |      6.3 | 1.07x      |                   200.5  |         1511 |
| procs-mps_smolvla+act+diffusion | eager  | diffusion | 1        |       180.6 |    184.1 | 1.32x      |                   200.5  |           45 |
| procs-mps_smolvla+act+diffusion | graph  | smolvla   | 1        |        79.2 |     80.2 | 1.33x      |                   228    |          102 |
| procs-mps_smolvla+act+diffusion | graph  | act       | 1        |         4.7 |      5.4 | 1.13x      |                   228    |         1679 |
| procs-mps_smolvla+act+diffusion | graph  | diffusion | 1        |       184.2 |    190.4 | 1.40x      |                   228    |           43 |
| procs-mps_smolvla+act+diffusion | loop   | smolvla   | 1        |        48.3 |     49.6 | 1.49x      |                   242.88 |          167 |
| procs-mps_smolvla+act+diffusion | loop   | act       | 1        |         4.6 |      4.9 | 1.13x      |                   242.88 |         1731 |
| procs-mps_smolvla+act+diffusion | loop   | diffusion | 1        |       177.6 |    184   | 1.36x      |                   242.88 |           45 |
| procs_smolvlax2+actx2           | eager  | smolvla   | 2        |       283.8 |    286.6 | 1.63x      |                   121.38 |           28 |
| procs_smolvlax2+actx2           | eager  | act       | 2        |        17.8 |     19.3 | 3.66x      |                   121.38 |          457 |
| procs_smolvlax2+actx2           | graph  | smolvla   | 2        |       249.8 |    250.5 | 4.19x      |                   152    |           32 |
| procs_smolvlax2+actx2           | graph  | act       | 2        |        13.5 |     16.3 | 3.27x      |                   152    |          576 |
| procs_smolvlax2+actx2           | loop   | smolvla   | 2        |       120.9 |    121.6 | 3.74x      |                   132.38 |           66 |
| procs_smolvlax2+actx2           | loop   | act       | 2        |        17.4 |     17.5 | 4.28x      |                   132.38 |          463 |
| procs-mps_smolvlax2+actx2       | eager  | smolvla   | 2        |       154.5 |    156.4 | 0.89x      |                   318.25 |           52 |
| procs-mps_smolvlax2+actx2       | eager  | act       | 2        |         6.3 |      7.7 | 1.30x      |                   318.25 |         1221 |
| procs-mps_smolvlax2+actx2       | graph  | smolvla   | 2        |       109.8 |    110.9 | 1.84x      |                   343.25 |           73 |
| procs-mps_smolvlax2+actx2       | graph  | act       | 2        |         5.9 |      7   | 1.44x      |                   343.25 |         1300 |
| procs-mps_smolvlax2+actx2       | loop   | smolvla   | 2        |        72.4 |     73.2 | 2.24x      |                   372.62 |          110 |
| procs-mps_smolvlax2+actx2       | loop   | act       | 2        |         5.8 |      6   | 1.42x      |                   372.62 |         1380 |

`vs alone` is the slowdown against the same policy and mode hosted alone (procs x1). `steps/s` sums every hosted copy (times the batch for batch hosting). `steps/copy` is the number of timed steps behind the median; rows under 5 are starved copies (the GPU time-slicer or the GIL let one copy run for seconds) and are left out of the prediction tables.

## Predicted from the policies alone (closed queueing network, host + GPU)

| config                          | hosting   | mode   | policy    |   alone ms |   model ms |   measured ms |   model err % |   no-interference err % |   pred steps/s |   meas steps/s |
|:--------------------------------|:----------|:-------|:----------|-----------:|-----------:|--------------:|--------------:|------------------------:|---------------:|---------------:|
| procs_smolvlax2                 | procs     | eager  | smolvla   |      173.7 |      178.6 |         181.1 |          -1.4 |                    -4.1 |           11.2 |           11.1 |
| procs_smolvlax2                 | procs     | graph  | smolvla   |       59.6 |       73.8 |          98.5 |         -25   |                   -39.5 |           27.1 |           20.5 |
| procs_smolvlax2                 | procs     | loop   | smolvla   |       32.4 |       42.6 |          60.4 |         -29.4 |                   -46.4 |           47   |           33   |
| procs_smolvlax4                 | procs     | eager  | smolvla   |      173.7 |      193.6 |         263.3 |         -26.5 |                   -34   |           20.7 |           15   |
| procs_smolvlax4                 | procs     | graph  | smolvla   |       59.6 |      114.9 |         186.1 |         -38.2 |                   -68   |           34.8 |           21.5 |
| procs_smolvlax4                 | procs     | loop   | smolvla   |       32.4 |       70.3 |         121.1 |         -41.9 |                   -73.3 |           56.9 |           33   |
| procs_smolvlax8                 | procs     | eager  | smolvla   |      173.7 |      254.9 |         512.2 |         -50.2 |                   -66.1 |           31.4 |           15.9 |
| procs_smolvlax8                 | procs     | graph  | smolvla   |       59.6 |      213.3 |         374.9 |         -43.1 |                   -84.1 |           37.5 |           21.2 |
| procs-mps_smolvlax2             | procs-mps | eager  | smolvla   |      173.7 |      178.6 |         170.3 |           4.9 |                     2   |           11.2 |           11.6 |
| procs-mps_smolvlax2             | procs-mps | graph  | smolvla   |       59.6 |       73.8 |          69.7 |           5.9 |                   -14.5 |           27.1 |           28.6 |
| procs-mps_smolvlax2             | procs-mps | loop   | smolvla   |       32.4 |       42.6 |          41.3 |           3   |                   -21.7 |           47   |           48.5 |
| procs-mps_smolvlax4             | procs-mps | eager  | smolvla   |      173.7 |      193.6 |         162.4 |          19.3 |                     7   |           20.7 |           24.6 |
| procs-mps_smolvlax4             | procs-mps | graph  | smolvla   |       59.6 |      114.9 |          83.2 |          38.1 |                   -28.4 |           34.8 |           48.2 |
| procs-mps_smolvlax4             | procs-mps | loop   | smolvla   |       32.4 |       70.3 |          47.7 |          47.6 |                   -32.1 |           56.9 |           83.9 |
| procs-mps_smolvlax8             | procs-mps | eager  | smolvla   |      173.7 |      254.9 |         155.3 |          64.2 |                    11.8 |           31.4 |           51.6 |
| procs-mps_smolvlax8             | procs-mps | graph  | smolvla   |       59.6 |      213.3 |         111.1 |          91.9 |                   -46.3 |           37.5 |           72.1 |
| procs-mps_smolvlax8             | procs-mps | loop   | smolvla   |       32.4 |      133.2 |          56.7 |         134.8 |                   -43   |           60   |          140.8 |
| threads_smolvlax2               | threads   | eager  | smolvla   |      173.7 |      308.5 |         373.4 |         -17.4 |                   -53.5 |            6.5 |            5.1 |
| threads_smolvlax2               | threads   | graph  | smolvla   |       59.6 |       90.1 |          78.4 |          15   |                   -23.9 |           22.2 |           25.8 |
| threads_smolvlax2               | threads   | loop   | smolvla   |       32.4 |       48.6 |          38.4 |          26.5 |                   -15.7 |           41.2 |           52   |
| threads_smolvlax4               | threads   | eager  | smolvla   |      173.7 |      593.8 |        1426.5 |         -58.4 |                   -87.8 |            6.7 |            2.9 |
| threads_smolvlax4               | threads   | graph  | smolvla   |       59.6 |      152.4 |         169.2 |          -9.9 |                   -64.8 |           26.2 |           24.5 |
| threads_smolvlax4               | threads   | loop   | smolvla   |       32.4 |       80.9 |          46   |          76.1 |                   -29.6 |           49.4 |           87   |
| threads_smolvlax8               | threads   | loop   | smolvla   |       32.4 |      145.7 |          54.3 |         168.2 |                   -40.4 |           54.9 |          147   |
| procs_actx2                     | procs     | eager  | act       |        4.9 |        9.5 |           9.5 |           0.2 |                   -48.9 |          210.1 |          209.9 |
| procs_actx2                     | procs     | graph  | act       |        4.1 |        9.7 |           8.3 |          16.5 |                   -50.2 |          207.1 |          241   |
| procs_actx2                     | procs     | loop   | act       |        4.1 |        9.7 |           8.2 |          17.1 |                   -50.6 |          207.2 |          242.4 |
| procs_actx4                     | procs     | eager  | act       |        4.9 |       18.9 |          19.1 |          -0.8 |                   -74.6 |          211.2 |          209.5 |
| procs_actx4                     | procs     | graph  | act       |        4.1 |       19.3 |          16.9 |          14.5 |                   -75.5 |          207.1 |          237.2 |
| procs_actx4                     | procs     | loop   | act       |        4.1 |       19.3 |          16.5 |          17   |                   -75.3 |          207.2 |          242.5 |
| procs_actx8                     | procs     | eager  | act       |        4.9 |       37.8 |          38.3 |          -1.3 |                   -87.3 |          211.5 |          209   |
| procs_actx8                     | procs     | graph  | act       |        4.1 |       38.6 |          33.7 |          14.7 |                   -87.8 |          207.1 |          238.5 |
| procs_actx8                     | procs     | loop   | act       |        4.1 |       38.6 |          33.1 |          16.7 |                   -87.7 |          207.2 |          241.5 |
| procs-mps_actx2                 | procs-mps | eager  | act       |        4.9 |        9.5 |           5.9 |          62.7 |                   -17   |          210.1 |          340.5 |
| procs-mps_actx2                 | procs-mps | graph  | act       |        4.1 |        9.7 |           5.4 |          79.9 |                   -23.1 |          207.1 |          371.8 |
| procs-mps_actx2                 | procs-mps | loop   | act       |        4.1 |        9.7 |           5.4 |          79.8 |                   -24.2 |          207.2 |          371.6 |
| procs-mps_actx4                 | procs-mps | eager  | act       |        4.9 |       18.9 |           8.2 |         130.5 |                   -40.9 |          211.2 |          485   |
| procs-mps_actx4                 | procs-mps | graph  | act       |        4.1 |       19.3 |           7.9 |         143.1 |                   -48.1 |          207.1 |          502.9 |
| procs-mps_actx4                 | procs-mps | loop   | act       |        4.1 |       19.3 |           8   |         142.5 |                   -48.9 |          207.2 |          502.4 |
| procs-mps_actx8                 | procs-mps | eager  | act       |        4.9 |       37.8 |          14.1 |         168   |                   -65.6 |          211.5 |          564.4 |
| procs-mps_actx8                 | procs-mps | graph  | act       |        4.1 |       38.6 |          14.1 |         173.6 |                   -70.8 |          207.1 |          567   |
| procs-mps_actx8                 | procs-mps | loop   | act       |        4.1 |       38.6 |          14.1 |         173.4 |                   -71.2 |          207.2 |          566.2 |
| threads_actx2                   | threads   | eager  | act       |        4.9 |        9.5 |          10.3 |          -7.3 |                   -52.7 |          210.1 |          189.4 |
| threads_actx2                   | threads   | graph  | act       |        4.1 |        9.7 |           5.2 |          87.2 |                   -20   |          207.1 |          387.5 |
| threads_actx2                   | threads   | loop   | act       |        4.1 |        9.7 |           5.2 |          87   |                   -21.1 |          207.2 |          387.2 |
| threads_actx4                   | threads   | eager  | act       |        4.9 |       18.9 |          39.3 |         -51.8 |                   -87.7 |          211.2 |           95.6 |
| threads_actx4                   | threads   | graph  | act       |        4.1 |       19.3 |           7.7 |         150.1 |                   -46.6 |          207.1 |          518   |
| threads_actx4                   | threads   | loop   | act       |        4.1 |       19.3 |           7.7 |         149.7 |                   -47.3 |          207.2 |          517.2 |
| threads_actx8                   | threads   | eager  | act       |        4.9 |       37.8 |        1141.6 |         -96.7 |                   -99.6 |          211.5 |            7   |
| threads_actx8                   | threads   | graph  | act       |        4.1 |       38.6 |          13.8 |         179.1 |                   -70.2 |          207.1 |          578   |
| threads_actx8                   | threads   | loop   | act       |        4.1 |       38.6 |          13.9 |         177.8 |                   -70.7 |          207.2 |          575.2 |
| procs_diffusionx2               | procs     | eager  | diffusion |      136.5 |      261.8 |         288.8 |          -9.3 |                   -52.7 |            7.6 |            7   |
| procs_diffusionx2               | procs     | graph  | diffusion |      132   |      259.3 |         281.3 |          -7.8 |                   -53.1 |            7.7 |            7   |
| procs_diffusionx2               | procs     | loop   | diffusion |      130.5 |      257   |         278.5 |          -7.7 |                   -53.1 |            7.8 |            7.2 |
| procs_diffusionx4               | procs     | eager  | diffusion |      136.5 |      518   |         577.9 |         -10.4 |                   -76.4 |            7.7 |            7   |
| procs_diffusionx4               | procs     | graph  | diffusion |      132   |      516.2 |         562.7 |          -8.3 |                   -76.5 |            7.7 |            7   |
| procs_diffusionx4               | procs     | loop   | diffusion |      130.5 |      512.1 |         558.8 |          -8.4 |                   -76.6 |            7.8 |            7.5 |
| procs_diffusionx8               | procs     | eager  | diffusion |      136.5 |     1033.1 |        1164   |         -11.2 |                   -88.3 |            7.7 |            7   |
| procs_diffusionx8               | procs     | graph  | diffusion |      132   |     1031.3 |        1133.3 |          -9   |                   -88.3 |            7.8 |            7   |
| procs_diffusionx8               | procs     | loop   | diffusion |      130.5 |     1023.2 |        1123.9 |          -9   |                   -88.4 |            7.8 |            7.1 |
| procs-mps_diffusionx2           | procs-mps | eager  | diffusion |      136.5 |      261.8 |         140   |          87   |                    -2.5 |            7.6 |           14.2 |
| procs-mps_diffusionx2           | procs-mps | graph  | diffusion |      132   |      259.3 |         134.2 |          93.3 |                    -1.6 |            7.7 |           14.8 |
| procs-mps_diffusionx2           | procs-mps | loop   | diffusion |      130.5 |      257   |         132.3 |          94.4 |                    -1.3 |            7.8 |           15.1 |
| procs-mps_diffusionx4           | procs-mps | eager  | diffusion |      136.5 |      518   |         156.6 |         230.8 |                   -12.8 |            7.7 |           25.5 |
| procs-mps_diffusionx4           | procs-mps | graph  | diffusion |      132   |      516.2 |         151.9 |         239.8 |                   -13.1 |            7.7 |           26.1 |
| procs-mps_diffusionx4           | procs-mps | loop   | diffusion |      130.5 |      512.1 |         149.8 |         241.7 |                   -12.9 |            7.8 |           27   |
| procs-mps_diffusionx8           | procs-mps | eager  | diffusion |      136.5 |     1033.1 |         221.9 |         365.7 |                   -38.5 |            7.7 |           36   |
| procs-mps_diffusionx8           | procs-mps | graph  | diffusion |      132   |     1031.3 |         218.1 |         372.9 |                   -39.5 |            7.8 |           37   |
| procs-mps_diffusionx8           | procs-mps | loop   | diffusion |      130.5 |     1023.2 |         215.6 |         374.5 |                   -39.5 |            7.8 |           37   |
| threads_diffusionx2             | threads   | eager  | diffusion |      136.5 |      262   |         139.8 |          87.3 |                    -2.4 |            7.6 |           14.2 |
| threads_diffusionx2             | threads   | graph  | diffusion |      132   |      259.3 |         133.7 |          93.9 |                    -1.3 |            7.7 |           15   |
| threads_diffusionx2             | threads   | loop   | diffusion |      130.5 |      257   |         134   |          91.8 |                    -2.6 |            7.8 |           14.8 |
| threads_diffusionx4             | threads   | eager  | diffusion |      136.5 |      518.1 |         253   |         104.8 |                   -46   |            7.7 |           14.9 |
| threads_diffusionx4             | threads   | graph  | diffusion |      132   |      516.3 |         153.3 |         236.8 |                   -13.8 |            7.7 |           26.2 |
| threads_diffusionx4             | threads   | loop   | diffusion |      130.5 |      512.1 |         149.5 |         242.6 |                   -12.7 |            7.8 |           27   |
| threads_diffusionx8             | threads   | graph  | diffusion |      132   |     1031.3 |         217.9 |         373.2 |                   -39.4 |            7.8 |           36.6 |
| threads_diffusionx8             | threads   | loop   | diffusion |      130.5 |     1023.2 |         215.6 |         374.6 |                   -39.5 |            7.8 |           37   |
| procs_smolvla+act               | procs     | eager  | smolvla   |      173.7 |      200.4 |         203.2 |          -1.4 |                   -14.5 |            5   |            5   |
| procs_smolvla+act               | procs     | eager  | act       |        4.9 |        6.1 |           7.6 |         -19.3 |                   -36.1 |          163.2 |          132.9 |
| procs_smolvla+act               | procs     | graph  | smolvla   |       59.6 |       85.7 |         127.9 |         -33   |                   -53.4 |           11.7 |            7.9 |
| procs_smolvla+act               | procs     | graph  | act       |        4.1 |        7.8 |           6.6 |          16.8 |                   -37.9 |          128.8 |          147.5 |
| procs_smolvla+act               | procs     | loop   | smolvla   |       32.4 |       48.7 |          60.4 |         -19.2 |                   -46.4 |           20.5 |           16.5 |
| procs_smolvla+act               | procs     | loop   | act       |        4.1 |        8.1 |           8.7 |          -7.2 |                   -53.2 |          123.9 |          116   |
| procs-mps_smolvla+act           | procs-mps | eager  | smolvla   |      173.7 |      200.4 |         171.7 |          16.7 |                     1.1 |            5   |            5.9 |
| procs-mps_smolvla+act           | procs-mps | eager  | act       |        4.9 |        6.1 |           5.1 |          21.1 |                    -4.1 |          163.2 |          193.1 |
| procs-mps_smolvla+act           | procs-mps | graph  | smolvla   |       59.6 |       85.7 |          77.8 |          10.1 |                   -23.4 |           11.7 |           12.9 |
| procs-mps_smolvla+act           | procs-mps | graph  | act       |        4.1 |        7.8 |           4.6 |          70   |                    -9.7 |          128.8 |          213.8 |
| procs-mps_smolvla+act           | procs-mps | loop   | smolvla   |       32.4 |       48.7 |          47.5 |           2.5 |                   -31.9 |           20.5 |           21.1 |
| procs-mps_smolvla+act           | procs-mps | loop   | act       |        4.1 |        8.1 |           4.5 |          79.3 |                    -9.5 |          123.9 |          220.2 |
| procs_smolvla+diffusion         | procs     | eager  | smolvla   |      173.7 |      199.7 |         247.3 |         -19.2 |                   -29.8 |            5   |            4   |
| procs_smolvla+diffusion         | procs     | eager  | diffusion |      136.5 |      171   |         196.2 |         -12.9 |                   -30.4 |            5.8 |            5.1 |
| procs_smolvla+diffusion         | procs     | graph  | smolvla   |       59.6 |       85.3 |         127.4 |         -33   |                   -53.2 |           11.7 |            7.9 |
| procs_smolvla+diffusion         | procs     | graph  | diffusion |      132   |      210.2 |         225.6 |          -6.8 |                   -41.5 |            4.8 |            4.5 |
| procs_smolvla+diffusion         | procs     | loop   | smolvla   |       32.4 |       48.5 |          60.8 |         -20.1 |                   -46.8 |           20.6 |           16.5 |
| procs_smolvla+diffusion         | procs     | loop   | diffusion |      130.5 |      216.2 |         277.7 |         -22.1 |                   -53   |            4.6 |            3.6 |
| procs-mps_smolvla+diffusion     | procs-mps | eager  | smolvla   |      173.7 |      199.7 |         167.2 |          19.4 |                     3.8 |            5   |            6   |
| procs-mps_smolvla+diffusion     | procs-mps | eager  | diffusion |      136.5 |      171   |         147.9 |          15.6 |                    -7.7 |            5.8 |            6.8 |
| procs-mps_smolvla+diffusion     | procs-mps | graph  | smolvla   |       59.6 |       85.3 |          63.6 |          34   |                    -6.3 |           11.7 |           15.6 |
| procs-mps_smolvla+diffusion     | procs-mps | graph  | diffusion |      132   |      210.2 |         154.5 |          36.1 |                   -14.5 |            4.8 |            6.5 |
| procs-mps_smolvla+diffusion     | procs-mps | loop   | smolvla   |       32.4 |       48.5 |          36.5 |          33   |                   -11.3 |           20.6 |           27.5 |
| procs-mps_smolvla+diffusion     | procs-mps | loop   | diffusion |      130.5 |      216.2 |         146.6 |          47.5 |                   -11   |            4.6 |            6.8 |
| procs_act+diffusion             | procs     | eager  | act       |        4.9 |        9.4 |          13.5 |         -30.2 |                   -64.2 |          105.9 |           74.6 |
| procs_act+diffusion             | procs     | eager  | diffusion |      136.5 |      263.8 |         227.5 |          15.9 |                   -40   |            3.8 |            4.4 |
| procs_act+diffusion             | procs     | graph  | act       |        4.1 |        9.6 |           8.7 |          10.2 |                   -52.7 |          104.2 |          115.2 |
| procs_act+diffusion             | procs     | graph  | diffusion |      132   |      260.9 |         286.4 |          -8.9 |                   -53.9 |            3.8 |            3.5 |
| procs_act+diffusion             | procs     | loop   | act       |        4.1 |        9.6 |           8.7 |          10.2 |                   -53.3 |          104.1 |          115   |
| procs_act+diffusion             | procs     | loop   | diffusion |      130.5 |      258.3 |         282.8 |          -8.7 |                   -53.9 |            3.9 |            3.5 |
| procs-mps_act+diffusion         | procs-mps | eager  | act       |        4.9 |        9.4 |           5   |          87.4 |                    -3.7 |          105.9 |          198.1 |
| procs-mps_act+diffusion         | procs-mps | eager  | diffusion |      136.5 |      263.8 |         171.5 |          53.8 |                   -20.4 |            3.8 |            5.9 |
| procs-mps_act+diffusion         | procs-mps | graph  | act       |        4.1 |        9.6 |           4.4 |         120.4 |                    -5.2 |          104.2 |          227.5 |
| procs-mps_act+diffusion         | procs-mps | graph  | diffusion |      132   |      260.9 |         176.1 |          48.2 |                   -25   |            3.8 |            5.6 |
| procs-mps_act+diffusion         | procs-mps | loop   | act       |        4.1 |        9.6 |           4.3 |         121.5 |                    -6.1 |          104.1 |          228.1 |
| procs-mps_act+diffusion         | procs-mps | loop   | diffusion |      130.5 |      258.3 |         173.7 |          48.8 |                   -24.9 |            3.9 |            5.8 |
| threads_act+diffusion           | threads   | eager  | act       |        4.9 |        9.4 |           5.6 |          69.3 |                   -13   |          105.8 |          147.8 |
| threads_act+diffusion           | threads   | eager  | diffusion |      136.5 |      263.9 |         172   |          53.4 |                   -20.6 |            3.8 |            5.9 |
| threads_act+diffusion           | threads   | graph  | act       |        4.1 |        9.6 |           4.1 |         132.1 |                    -0.2 |          104.2 |          241.4 |
| threads_act+diffusion           | threads   | graph  | diffusion |      132   |      260.9 |         175.4 |          48.7 |                   -24.7 |            3.8 |            5.8 |
| threads_act+diffusion           | threads   | loop   | act       |        4.1 |        9.6 |           4.1 |         132.4 |                    -1.5 |          104.1 |          241.4 |
| threads_act+diffusion           | threads   | loop   | diffusion |      130.5 |      258.3 |         173.7 |          48.7 |                   -24.9 |            3.9 |            5.8 |
| procs_smolvla+act+diffusion     | procs     | eager  | smolvla   |      173.7 |      227.2 |         279.7 |         -18.7 |                   -37.9 |            4.4 |            3.6 |
| procs_smolvla+act+diffusion     | procs     | eager  | act       |        4.9 |       11.1 |          18.2 |         -38.9 |                   -73.4 |           89.7 |           57.2 |
| procs_smolvla+act+diffusion     | procs     | eager  | diffusion |      136.5 |      310   |         293.6 |           5.6 |                   -53.5 |            3.2 |            3.4 |
| procs_smolvla+act+diffusion     | procs     | graph  | smolvla   |       59.6 |      111.5 |         198.3 |         -43.8 |                   -69.9 |            9   |            5   |
| procs_smolvla+act+diffusion     | procs     | graph  | act       |        4.1 |       13   |          11.1 |          16.9 |                   -62.9 |           77   |           88.6 |
| procs_smolvla+act+diffusion     | procs     | graph  | diffusion |      132   |      351   |         374.1 |          -6.2 |                   -64.7 |            2.8 |            2.6 |
| procs_smolvla+act+diffusion     | procs     | loop   | smolvla   |       32.4 |       65   |          91   |         -28.6 |                   -64.4 |           15.4 |           10.9 |
| procs_smolvla+act+diffusion     | procs     | loop   | act       |        4.1 |       13.3 |          13.1 |           1.1 |                   -68.9 |           75.5 |           76.9 |
| procs_smolvla+act+diffusion     | procs     | loop   | diffusion |      130.5 |      354.7 |         422.4 |         -16   |                   -69.1 |            2.8 |            2.4 |
| procs-mps_smolvla+act+diffusion | procs-mps | eager  | smolvla   |      173.7 |      227.2 |         171.6 |          32.4 |                     1.2 |            4.4 |            6   |
| procs-mps_smolvla+act+diffusion | procs-mps | eager  | act       |        4.9 |       11.1 |           5.2 |         115.2 |                    -6.3 |           89.7 |          188.9 |
| procs-mps_smolvla+act+diffusion | procs-mps | eager  | diffusion |      136.5 |      310   |         180.6 |          71.6 |                   -24.4 |            3.2 |            5.6 |
| procs-mps_smolvla+act+diffusion | procs-mps | graph  | smolvla   |       59.6 |      111.5 |          79.2 |          40.7 |                   -24.7 |            9   |           12.8 |
| procs-mps_smolvla+act+diffusion | procs-mps | graph  | act       |        4.1 |       13   |           4.7 |         178.3 |                   -11.6 |           77   |          209.9 |
| procs-mps_smolvla+act+diffusion | procs-mps | graph  | diffusion |      132   |      351   |         184.2 |          90.5 |                   -28.3 |            2.8 |            5.4 |
| procs-mps_smolvla+act+diffusion | procs-mps | loop   | smolvla   |       32.4 |       65   |          48.3 |          34.5 |                   -33   |           15.4 |           20.9 |
| procs-mps_smolvla+act+diffusion | procs-mps | loop   | act       |        4.1 |       13.3 |           4.6 |         189.2 |                   -11.2 |           75.5 |          216.4 |
| procs-mps_smolvla+act+diffusion | procs-mps | loop   | diffusion |      130.5 |      354.7 |         177.6 |          99.7 |                   -26.5 |            2.8 |            5.6 |
| procs_smolvlax2+actx2           | procs     | eager  | smolvla   |      173.7 |      238.2 |         283.8 |         -16.1 |                   -38.8 |            8.4 |            7   |
| procs_smolvlax2+actx2           | procs     | eager  | act       |        4.9 |       13.2 |          17.8 |         -25.8 |                   -72.7 |          151.8 |          114.4 |
| procs_smolvlax2+actx2           | procs     | graph  | smolvla   |       59.6 |      131.1 |         249.8 |         -47.5 |                   -76.1 |           15.3 |            8   |
| procs_smolvlax2+actx2           | procs     | graph  | act       |        4.1 |       16.8 |          13.5 |          24.8 |                   -69.4 |          118.8 |          144   |
| procs_smolvlax2+actx2           | procs     | loop   | smolvla   |       32.4 |       78.1 |         120.9 |         -35.4 |                   -73.2 |           25.6 |           16.5 |
| procs_smolvlax2+actx2           | procs     | loop   | act       |        4.1 |       17.3 |          17.4 |          -0.5 |                   -76.6 |          115.4 |          115.9 |
| procs-mps_smolvlax2+actx2       | procs-mps | eager  | smolvla   |      173.7 |      238.2 |         154.5 |          54.2 |                    12.4 |            8.4 |           13   |
| procs-mps_smolvlax2+actx2       | procs-mps | eager  | act       |        4.9 |       13.2 |           6.3 |         109.5 |                   -22.8 |          151.8 |          305.2 |
| procs-mps_smolvlax2+actx2       | procs-mps | graph  | smolvla   |       59.6 |      131.1 |         109.8 |          19.4 |                   -45.7 |           15.3 |           18.2 |
| procs-mps_smolvlax2+actx2       | procs-mps | graph  | act       |        4.1 |       16.8 |           5.9 |         183.2 |                   -30.6 |          118.8 |          325   |
| procs-mps_smolvlax2+actx2       | procs-mps | loop   | smolvla   |       32.4 |       78.1 |          72.4 |           8   |                   -55.3 |           25.6 |           27.6 |
| procs-mps_smolvlax2+actx2       | procs-mps | loop   | act       |        4.1 |       17.3 |           5.8 |         199.6 |                   -29.6 |          115.4 |          345   |

MAPE (%) of the per-policy latency:

|                        |   n |   model_MAPE |   no_interference_MAPE |
|:-----------------------|----:|-------------:|-----------------------:|
| ('procs', 'eager')     |  20 |         15.8 |                   51.2 |
| ('procs', 'graph')     |  20 |         21.3 |                   62.9 |
| ('procs', 'loop')      |  19 |         16.7 |                   63.7 |
| ('procs-mps', 'eager') |  20 |         86.5 |                   15.3 |
| ('procs-mps', 'graph') |  20 |        103.5 |                   25.5 |
| ('procs-mps', 'loop')  |  20 |        107.8 |                   27.3 |
| ('threads', 'eager')   |   9 |         60.7 |                   51.5 |
| ('threads', 'graph')   |  10 |        132.6 |                   30.5 |
| ('threads', 'loop')    |  11 |        143.2 |                   27.8 |

Reading: for `procs` the model assumes the GPU serializes the contexts' kernels; a measured latency below the model means kernels of different processes overlapped (expected under `procs-mps`, where the model is an upper bound). For `threads` the model also serializes the host side (GIL); PyTorch releases the GIL inside its C++ ops, so the measurement should sit between the two bounds.

## Correctness

Largest |diff| over all copies: split loop vs stock step 0.0e+00, graph vs eager 0.0e+00 (159 policy copies loaded).

MPS state recorded by the procs-mps runs: ['daemon up, servers: 3428289', 'daemon up, servers: 3433136', 'daemon up, servers: 3437733', 'daemon up, servers: 3441311', 'daemon up, servers: 3442348', 'daemon up, servers: 3443378', 'daemon up, servers: 3444440', 'daemon up, servers: 3445810']
