# vla-gpu-latency

Predicting batch-1 vision-language-action (VLA) policy inference latency, and the speedup
from CUDA-graph capture, on a target GPU from a profile taken on a different source GPU.
The model makes kernel launch overhead and L2 working-set fit explicit terms.

Course project for NYU GPU Architecture and Programming, Fall 2026 (Project 2, flavor B).
The full brief, status and decisions log are in [CLAUDE.md](CLAUDE.md).

## Quick start (CIMS cuda2)

```bash
module load cuda-12.4
make -C bench                 # builds for sm_75
scripts/check_profilers.sh    # do ncu counters and nsys work?
scripts/run_calib.sh          # device info, launch latency, L2/DRAM bandwidth sweep
```

## Slurm cluster

```bash
cp scripts/cluster/site.example.sh scripts/cluster/site.sh && source scripts/cluster/site.sh
make -C bench ARCHS="75 70 80 89" BUILD=build-cluster
sbatch -p g2-standard-12 scripts/cluster/calib.sbatch    # one job per target GPU
```

Results are CSV files under `results/calib/<gpu>_<host>_<time>/`, one schema for all GPUs:
`gpu,bench,kernel,config,mode,timer,n_runs,mean_us,std_us,min_us,median_us,module_loading`.
