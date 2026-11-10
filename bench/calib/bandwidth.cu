// Working-set sweep: effective read bandwidth vs working-set size. The knee locates the
// usable L2 capacity; the plateaus give BW_L2 and BW_DRAM for the model's memory term.
//
// Loads use __ldcg (cache at L2, bypass L1) so small working sets measure L2, not the
// per-SM L1. Each launch reads the working set `passes` times so every launch moves at
// least ~256 MB and launch overhead is negligible.
#include <string>

#include "../common/harness.cuh"

__global__ void read_kernel(const float4* __restrict__ a, size_t n, int passes, float* sink) {
  float acc = 0.f;
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (int p = 0; p < passes; ++p) {
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < n;
         i += stride) {
      float4 v = __ldcg(a + i);
      acc += v.x + v.y + v.z + v.w;
    }
  }
  // Never true for the zero-filled input, but the compiler cannot prove it.
  if (acc == 1234.5f) sink[0] = acc;
}

int main(int argc, char** argv) {
  const int reps = bench::arg_int(argc, argv, "reps", 20);
  const int warmup = bench::arg_int(argc, argv, "warmup", 3);
  const int max_mb = bench::arg_int(argc, argv, "max_mb", 1024);

  cudaDeviceProp p;
  CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
  size_t max_bytes = static_cast<size_t>(max_mb) << 20;
  max_bytes = std::min(max_bytes, p.totalGlobalMem / 4);

  float4* a = nullptr;
  float* sink = nullptr;
  CUDA_CHECK(cudaMalloc(&a, max_bytes));
  CUDA_CHECK(cudaMemset(a, 0, max_bytes));
  CUDA_CHECK(cudaMalloc(&sink, sizeof(float)));
  cudaStream_t s;
  CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));

  const int threads = 256;
  int blocks_per_sm = 0;
  CUDA_CHECK(
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, read_kernel, threads, 0));
  const int grid = blocks_per_sm * p.multiProcessorCount;

  bench::csv_header();
  // Sizes from 256 KB upward in steps of 2^(1/4).
  for (double kb = 256; kb * 1024 <= static_cast<double>(max_bytes); kb *= 1.189207115) {
    size_t bytes = (static_cast<size_t>(kb * 1024) / 4096) * 4096;
    size_t n = bytes / sizeof(float4);
    int passes = static_cast<int>(std::max<size_t>(1, (256ull << 20) / bytes));
    auto enqueue = [&](cudaStream_t st) {
      read_kernel<<<grid, threads, 0, st>>>(a, n, passes, sink);
    };
    bench::Stats t = bench::time_events(s, enqueue, warmup, reps);
    CUDA_CHECK(cudaGetLastError());
    std::string cfg = "bytes=" + std::to_string(bytes) + ";passes=" + std::to_string(passes) +
                      ";grid=" + std::to_string(grid) + ";block=" + std::to_string(threads);
    bench::csv_row("bandwidth", "read_ldcg", cfg, "eager", "event", t);
    // Bandwidth in GB/s from the median time, as a separate row for convenience.
    double gbps = static_cast<double>(bytes) * passes / (t.median * 1e-6) / 1e9;
    std::fprintf(stderr, "%10.2f MB  %9.1f GB/s\n", bytes / 1048576.0, gbps);
  }

  CUDA_CHECK(cudaFree(a));
  CUDA_CHECK(cudaFree(sink));
  CUDA_CHECK(cudaStreamDestroy(s));
  return 0;
}
