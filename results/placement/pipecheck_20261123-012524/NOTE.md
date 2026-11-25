# Diagnostic that found the static_x bug (job 40291, 1x A100)

`observations differ by: KV 3.5, eager output 0.73, graph output 7.6e4; graph vs eager per
observation 7.6e4`: the two observations differ and the eager loop follows them, but the graph
loop's output was garbage. Cause: `capture_graph` did not keep its static noise input alive.
Fixed in e3ed026 (pipelining) and 02e736c (profiling-hosting).
