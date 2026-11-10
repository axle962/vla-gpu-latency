// Shared timing harness for all benchmarks.
//
// Every binary prints CSV rows with the same columns so results from all GPUs merge cleanly:
//   gpu,bench,kernel,config,mode,timer,n_runs,mean_us,std_us,min_us,median_us,module_loading
// `config` is a semicolon-separated key=value list (never contains commas).
// `mode` is eager or graph. `timer` is event (GPU-side, cudaEvent) or host (wall clock,
// including launch calls and the final synchronize).
#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                              \
  do {                                                                                \
    cudaError_t err_ = (call);                                                        \
    if (err_ != cudaSuccess) {                                                        \
      std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, __LINE__, \
                   cudaGetErrorString(err_));                                         \
      std::exit(1);                                                                   \
    }                                                                                 \
  } while (0)

namespace bench {

struct Stats {
  int n = 0;
  double mean = 0, std = 0, min = 0, median = 0;
};

inline Stats summarize(std::vector<double> v) {
  Stats s;
  s.n = static_cast<int>(v.size());
  if (v.empty()) return s;
  std::sort(v.begin(), v.end());
  double sum = 0;
  for (double x : v) sum += x;
  s.mean = sum / s.n;
  double ss = 0;
  for (double x : v) ss += (x - s.mean) * (x - s.mean);
  s.std = s.n > 1 ? std::sqrt(ss / (s.n - 1)) : 0.0;
  s.min = v.front();
  s.median = s.n % 2 ? v[s.n / 2] : 0.5 * (v[s.n / 2 - 1] + v[s.n / 2]);
  return s;
}

inline Stats scale(Stats s, double f) {
  s.mean *= f;
  s.std *= f;
  s.min *= f;
  s.median *= f;
  return s;
}

inline std::string gpu_name(int dev = 0) {
  cudaDeviceProp p;
  CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
  std::string n = p.name;
  for (char& c : n)
    if (c == ',') c = ' ';
  return n;
}

// CUDA 12 defaults to lazy module loading, which inflates the first launch of each kernel.
// Benchmarks always warm up, and record the setting so the paper can state it.
inline std::string module_loading() {
  const char* s = std::getenv("CUDA_MODULE_LOADING");
  return s ? s : "LAZY(default)";
}

inline void csv_header() {
  std::printf(
      "gpu,bench,kernel,config,mode,timer,n_runs,mean_us,std_us,min_us,median_us,"
      "module_loading\n");
}

inline void csv_row(const std::string& bench, const std::string& kernel, const std::string& config,
                    const char* mode, const char* timer, const Stats& s) {
  static const std::string gpu = gpu_name();
  static const std::string ml = module_loading();
  std::printf("%s,%s,%s,%s,%s,%s,%d,%.4f,%.4f,%.4f,%.4f,%s\n", gpu.c_str(), bench.c_str(),
              kernel.c_str(), config.c_str(), mode, timer, s.n, s.mean, s.std, s.min, s.median,
              ml.c_str());
  std::fflush(stdout);
}

// Times `enqueue(stream)` with cudaEvents around each repetition. Returns microseconds.
template <class F>
Stats time_events(cudaStream_t stream, F&& enqueue, int warmup, int reps) {
  cudaEvent_t a, b;
  CUDA_CHECK(cudaEventCreate(&a));
  CUDA_CHECK(cudaEventCreate(&b));
  for (int i = 0; i < warmup; ++i) enqueue(stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  std::vector<double> us;
  us.reserve(reps);
  for (int i = 0; i < reps; ++i) {
    CUDA_CHECK(cudaEventRecord(a, stream));
    enqueue(stream);
    CUDA_CHECK(cudaEventRecord(b, stream));
    CUDA_CHECK(cudaEventSynchronize(b));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
    us.push_back(1e3 * ms);
  }
  CUDA_CHECK(cudaEventDestroy(a));
  CUDA_CHECK(cudaEventDestroy(b));
  return summarize(us);
}

// Times `enqueue(stream)` plus a stream synchronize with a host wall clock: the latency a
// caller (e.g. a robot control loop) actually observes. Returns microseconds.
template <class F>
Stats time_host(cudaStream_t stream, F&& enqueue, int warmup, int reps) {
  using clk = std::chrono::steady_clock;
  for (int i = 0; i < warmup; ++i) enqueue(stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  std::vector<double> us;
  us.reserve(reps);
  for (int i = 0; i < reps; ++i) {
    auto t0 = clk::now();
    enqueue(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto t1 = clk::now();
    us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
  }
  return summarize(us);
}

// Captures everything `enqueue(stream)` launches into an instantiated CUDA graph.
template <class F>
cudaGraphExec_t capture_graph(cudaStream_t stream, F&& enqueue) {
  cudaGraph_t g;
  cudaGraphExec_t exec;
  CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  enqueue(stream);
  CUDA_CHECK(cudaStreamEndCapture(stream, &g));
  CUDA_CHECK(cudaGraphInstantiate(&exec, g, 0));
  CUDA_CHECK(cudaGraphDestroy(g));
  return exec;
}

inline int arg_int(int argc, char** argv, const char* key, int def) {
  std::string k = std::string("--") + key + "=";
  for (int i = 1; i < argc; ++i)
    if (std::string(argv[i]).rfind(k, 0) == 0) return std::atoi(argv[i] + k.size());
  return def;
}

}  // namespace bench
