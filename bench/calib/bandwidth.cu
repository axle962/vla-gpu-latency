// Working-set sweep: effective read bandwidth vs working-set size. The knee locates the
// usable L2 capacity; the plateaus give BW_L2 and BW_DRAM for the model's memory term.
//
// Loads use __ldcg (cache at L2, bypass L1) so small working sets measure L2, not the
// per-SM L1. Each launch moves at least ~256 MB so launch overhead is negligible.
#include <string>

#include "../common/harness.cuh"

// Every thread in the grid issues `iters` loads, walking the working set with a stride of the
// whole grid and wrapping around. All threads stay busy even when the working set has fewer
// float4 elements than the grid has threads, so small (L2-resident) sizes are not starved of
// parallelism. Bytes moved = grid * block * iters * 16.
__global__ void read_kernel(const float4* __restrict__ a, unsigned n, unsigned step, int iters,
                            float* sink) {
  float acc = 0.f;
  unsigned idx = (blockIdx.x * blockDim.x + threadIdx.x) % n;
#pragma unroll 4
  for (int k = 0; k < iters; ++k) {
    float4 v = __ldcg(a + idx);
    acc += v.x + v.y + v.z + v.w;
    idx += step;  // step = (grid size) mod n, both < n, so one subtraction wraps it
    if (idx >= n) idx -= n;
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
    unsigned n = static_cast<unsigned>(bytes / sizeof(float4));
    const unsigned long long total_threads = static_cast<unsigned long long>(grid) * threads;
    unsigned step = static_cast<unsigned>(total_threads % n);
    // At least 256 MB moved and at least one full sweep of the working set.
    unsigned long long loads = std::max<unsigned long long>((256ull << 20) / 16, n);
    int iters = static_cast<int>((loads + total_threads - 1) / total_threads);
    size_t moved = static_cast<size_t>(total_threads) * iters * 16;
    auto enqueue = [&](cudaStream_t st) {
      read_kernel<<<grid, threads, 0, st>>>(a, n, step, iters, sink);
    };
    bench::Stats t = bench::time_events(s, enqueue, warmup, reps);
    CUDA_CHECK(cudaGetLastError());
    std::string cfg = "bytes=" + std::to_string(bytes) + ";moved=" + std::to_string(moved) +
                      ";grid=" + std::to_string(grid) + ";block=" + std::to_string(threads);
    bench::csv_row("bandwidth", "read_ldcg", cfg, "eager", "event", t);
    // Human-readable bandwidth from the median time (stderr; the CSV has the raw times).
    double gbps = static_cast<double>(moved) / (t.median * 1e-6) / 1e9;
    std::fprintf(stderr, "%10.2f MB  %9.1f GB/s\n", bytes / 1048576.0, gbps);
  }

  CUDA_CHECK(cudaFree(a));
  CUDA_CHECK(cudaFree(sink));
  CUDA_CHECK(cudaStreamDestroy(s));
  return 0;
}
