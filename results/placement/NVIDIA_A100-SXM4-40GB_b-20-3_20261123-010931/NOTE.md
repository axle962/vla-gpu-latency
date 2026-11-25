# First pipelining run (2026-11-23 01:09 UTC, job 40283): timings only

Run with the placement.py that freed the graph's static noise input (`static_x`) after
capture, so every CUDA-graph replay computed on reused memory: its outputs, and therefore
the pipelining correctness check (both observations gave the same garbage), are invalid. The
timings are valid (same kernels on the same shapes) and agree with the fixed rerun within 1%
(pipelined 60.7 / 60.9 ms vs 61.2 / 61.6 ms). Superseded by
`NVIDIA_A100-SXM4-40GB_b-20-4_20261123-013831`.
