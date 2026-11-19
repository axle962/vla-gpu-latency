# First cuda3 run (2026-11-18 01:17, script commit 0b03985): what can be used

Both TITAN Vs idle, pinned to NUMA node 0 (GPU 0's node), 24 torch threads, `--profile`.
Pushed from cuda3 in commit d8f897f.

Problems found in this run, fixed in the script afterwards:

1. **Profiler residue.** The script profiled each placement right after timing it. After the
   first profiler session every later GPU timing is slower: prefix on cuda:0 59.2 ms (`gpu`)
   vs 67.7 to 68.0 ms (later placements), eager loop on cuda:0 280.4 vs 349.0 ms (`cpu-gpu`),
   graph loop on cuda:0 66.5 vs 72.9 ms. So the colocated-vs-split comparisons and the
   "predicted from colocated parts" errors in report.md are not valid for this run.
   Now: all profiling happens after all timing, and the first placement is timed again at the
   end (`e2e_eager_repeat`) as a drift check.
2. **No graph on cuda:1.** `torch.cuda.graph` uses one capture stream per process, created on
   cuda:0 by the first capture; capturing cuda:1 work on it fails
   (`cudaErrorStreamCaptureInvalidated`). Now each capture gets a stream on its own device.
3. **Handoff model.** The copy calibration synced after every copy, which overstates the
   per-tensor cost of GPU-GPU copies queued back to back (+120% handoff error). Now it also
   times 32 small copies with one sync.

Usable as is:
- `gpu` placement (timed first, before any profiling): prefix 59.2, eager loop 280.4, graph
  loop 66.5, eager step 340.6, hybrid 126.2 ms; matches the TITAN V traces (eager 343 ms).
- Handoff times (copies, barely affected by the profiler): GPU->GPU 1.36 ms, GPU->CPU 3.55 ms
  (4.9 MB FP16, 33 tensors); CPU->GPU 4.29 ms (9.9 MB FP32, cast to FP16 on the GPU).
- CPU halves (FP32, 24 threads): prefix 1.97 to 2.02 s, action expert loop 928 to 944 ms.
- Numerics: moving the action expert to the CPU changes the actions by at most 7e-4; moving the
  prefix to the CPU (FP32 instead of FP16) changes them by up to 0.38. To check.
