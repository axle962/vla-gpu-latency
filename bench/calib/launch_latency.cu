// Launch-overhead calibration: the L_launch, L_graph and delta_node terms of the model.
//
// Experiments (bench column = launch):
//   host_launch_call  CPU cost of one cudaLaunchKernel call (empty kernel, no sync).
//   host_graph_call   CPU cost of one cudaGraphLaunch call for an N-node graph.
//   empty_chain       N back-to-back empty kernels, per-kernel time (eager vs graph):
//                     the minimum spacing between dependent kernels on the GPU.
//   roundtrip         One empty kernel plus stream synchronize, host clock.
//   spin_chain        N kernels that each busy-wait D ns, end-to-end host clock, eager vs
//                     graph. Sweeping D shows where the chain stops being launch-bound,
//                     which is the crossover the latency model must predict.
#include <string>

#include "../common/harness.cuh"

__global__ void empty_kernel() {}

__device__ __forceinline__ unsigned long long global_ns() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

// Busy-waits for `ns` nanoseconds. globaltimer resolution is about 32 ns to 1 us depending
// on the GPU, so durations below ~1 us are approximate; the measured chain time is what
// matters, not the nominal D.
__global__ void spin_kernel(unsigned long long ns) {
  if (ns == 0) return;
  unsigned long long t0 = global_ns();
  while (global_ns() - t0 < ns) {
  }
}

int main(int argc, char** argv) {
  const int reps = bench::arg_int(argc, argv, "reps", 30);
  const int warmup = bench::arg_int(argc, argv, "warmup", 5);
  const int chain = bench::arg_int(argc, argv, "chain", 100);
  cudaStream_t s;
  CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  bench::csv_header();
  using clk = std::chrono::steady_clock;

  // Host-side cost of one launch call. A batch of calls is timed and divided, the GPU is
  // drained between repetitions so the launch queue never fills.
  {
    const int batch = 256;
    for (int i = 0; i < warmup; ++i) empty_kernel<<<1, 32, 0, s>>>();
    CUDA_CHECK(cudaStreamSynchronize(s));
    std::vector<double> us;
    for (int r = 0; r < reps; ++r) {
      auto t0 = clk::now();
      for (int i = 0; i < batch; ++i) empty_kernel<<<1, 32, 0, s>>>();
      auto t1 = clk::now();
      CUDA_CHECK(cudaStreamSynchronize(s));
      us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count() / batch);
    }
    CUDA_CHECK(cudaGetLastError());
    bench::csv_row("launch", "host_launch_call", "batch=256", "eager", "host",
                   bench::summarize(us));
  }

  for (int n : {1, 10, 100, 1000}) {
    std::string cfg = "n=" + std::to_string(n);
    auto enqueue_empty = [&](cudaStream_t st) {
      for (int i = 0; i < n; ++i) empty_kernel<<<1, 32, 0, st>>>();
    };
    cudaGraphExec_t g = bench::capture_graph(s, enqueue_empty);
    auto replay = [&](cudaStream_t st) { CUDA_CHECK(cudaGraphLaunch(g, st)); };

    // Per-kernel GPU-side spacing.
    auto e = bench::time_events(s, enqueue_empty, warmup, reps);
    bench::csv_row("launch", "empty_chain_per_kernel", cfg, "eager", "event",
                   bench::scale(e, 1.0 / n));
    auto ge = bench::time_events(s, replay, warmup, reps);
    bench::csv_row("launch", "empty_chain_per_kernel", cfg, "graph", "event",
                   bench::scale(ge, 1.0 / n));

    // End-to-end host latency of the whole chain.
    bench::csv_row("launch", "empty_chain_total", cfg, "eager", "host",
                   bench::time_host(s, enqueue_empty, warmup, reps));
    bench::csv_row("launch", "empty_chain_total", cfg, "graph", "host",
                   bench::time_host(s, replay, warmup, reps));

    // CPU cost of the cudaGraphLaunch call itself.
    std::vector<double> us;
    for (int r = 0; r < reps; ++r) {
      auto t0 = clk::now();
      CUDA_CHECK(cudaGraphLaunch(g, s));
      auto t1 = clk::now();
      CUDA_CHECK(cudaStreamSynchronize(s));
      us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
    bench::csv_row("launch", "host_graph_call", cfg, "graph", "host", bench::summarize(us));
    CUDA_CHECK(cudaGraphExecDestroy(g));
  }

  for (int d_ns : {0, 500, 1000, 2000, 3000, 5000, 7500, 10000, 15000, 20000, 30000, 50000,
                   100000}) {
    std::string cfg = "n=" + std::to_string(chain) + ";spin_ns=" + std::to_string(d_ns);
    auto enqueue_spin = [&](cudaStream_t st) {
      for (int i = 0; i < chain; ++i)
        spin_kernel<<<1, 32, 0, st>>>(static_cast<unsigned long long>(d_ns));
    };
    cudaGraphExec_t g = bench::capture_graph(s, enqueue_spin);
    auto replay = [&](cudaStream_t st) { CUDA_CHECK(cudaGraphLaunch(g, st)); };
    bench::csv_row("launch", "spin_chain_total", cfg, "eager", "host",
                   bench::time_host(s, enqueue_spin, warmup, reps));
    bench::csv_row("launch", "spin_chain_total", cfg, "graph", "host",
                   bench::time_host(s, replay, warmup, reps));
    bench::csv_row("launch", "spin_chain_total", cfg, "eager", "event",
                   bench::time_events(s, enqueue_spin, warmup, reps));
    bench::csv_row("launch", "spin_chain_total", cfg, "graph", "event",
                   bench::time_events(s, replay, warmup, reps));
    CUDA_CHECK(cudaGraphExecDestroy(g));
  }

  CUDA_CHECK(cudaStreamDestroy(s));
  return 0;
}
