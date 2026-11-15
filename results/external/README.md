# External reference numbers

`yotta_vla_benchmarks_2026.csv`: Yotta Labs and Fractal Labs, "VLA inference GPU benchmarks",
https://www.yottalabs.ai/post/vla-inference-gpu-benchmarks (published 2026-08-10, read
2026-11-15). Batch 1, latency in ms per forward pass from their main table. Not stated on the
page: precision, CUDA graphs or torch.compile, timing method, host CPU or instance type, driver.
Software differs per GPU for OpenVLA (framework_note column), so cross-GPU differences mix
hardware with host and software. The page's claim that the H200 loses "every matchup" is
contradicted by its own Lingbot-VLA row (H200 583 vs H100 884 ms).
