// deviceQuery-style dump of the properties the latency model uses, as gpu,key,value CSV.
// Run on every source and target GPU; the model reads specs from these files, not from
// hand-copied spec sheets.
#include <cstdio>
#include <string>

#include "../common/harness.cuh"

static void kv(const std::string& gpu, const char* key, long long v) {
  std::printf("%s,%s,%lld\n", gpu.c_str(), key, v);
}

int main() {
  int n = 0;
  CUDA_CHECK(cudaGetDeviceCount(&n));
  int drv = 0, rt = 0;
  CUDA_CHECK(cudaDriverGetVersion(&drv));
  CUDA_CHECK(cudaRuntimeGetVersion(&rt));
  std::printf("gpu,key,value\n");
  for (int d = 0; d < n; ++d) {
    CUDA_CHECK(cudaSetDevice(d));
    cudaDeviceProp p;
    CUDA_CHECK(cudaGetDeviceProperties(&p, d));
    std::string gpu = bench::gpu_name(d);
    auto attr = [&](cudaDeviceAttr a) {
      int v = 0;
      CUDA_CHECK(cudaDeviceGetAttribute(&v, a, d));
      return v;
    };
    int mem_khz = attr(cudaDevAttrMemoryClockRate);
    int bus_bits = attr(cudaDevAttrGlobalMemoryBusWidth);
    kv(gpu, "device_index", d);
    kv(gpu, "cc_major", p.major);
    kv(gpu, "cc_minor", p.minor);
    kv(gpu, "sm_count", p.multiProcessorCount);
    kv(gpu, "l2_bytes", p.l2CacheSize);
    kv(gpu, "persisting_l2_max_bytes", p.persistingL2CacheMaxSize);
    kv(gpu, "global_mem_bytes", static_cast<long long>(p.totalGlobalMem));
    kv(gpu, "smem_per_sm_bytes", static_cast<long long>(p.sharedMemPerMultiprocessor));
    kv(gpu, "smem_per_block_optin_bytes", static_cast<long long>(p.sharedMemPerBlockOptin));
    kv(gpu, "regs_per_sm", p.regsPerMultiprocessor);
    kv(gpu, "max_threads_per_sm", p.maxThreadsPerMultiProcessor);
    kv(gpu, "max_blocks_per_sm", p.maxBlocksPerMultiProcessor);
    kv(gpu, "warp_size", p.warpSize);
    kv(gpu, "sm_clock_khz", attr(cudaDevAttrClockRate));
    kv(gpu, "mem_clock_khz", mem_khz);
    kv(gpu, "mem_bus_bits", bus_bits);
    // Double data rate: bytes/s = 2 * clock * bus_width / 8. Approximate for GDDR6X/HBM;
    // the measured bandwidth sweep is the number the model should trust.
    kv(gpu, "theoretical_dram_bw_mbps",
       static_cast<long long>(2.0 * mem_khz * 1e3 * (bus_bits / 8.0) / 1e6));
    kv(gpu, "async_engines", p.asyncEngineCount);
    kv(gpu, "concurrent_kernels", p.concurrentKernels);
    kv(gpu, "ecc_enabled", p.ECCEnabled);
    kv(gpu, "pci_bus_id", p.pciBusID);
    kv(gpu, "driver_version", drv);
    kv(gpu, "runtime_version", rt);
  }
  return 0;
}
