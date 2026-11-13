// Replays a VLA step rebuilt from a trace (bench/chains/*.csv, written by
// profile/extract_chain.py) as a standalone CUDA C++ program, eager and as a CUDA graph.
//
// Each row of the list is one launch of the traced step with its category and shape. Rows run
// our kernels from bench/kernels/kernels.cuh (GEMMs optionally through cuBLAS instead). Data
// layout (with --gemm=cublas, attention also runs as cuBLAS QK^T, our softmax, cuBLAS PV,
// like PyTorch's math SDPA path; with --gemm=own everything is our code). Data layout:
//   - every GEMM whose B operand is a model parameter (weight=1) gets its own weight buffer,
//     so a step streams its weights from DRAM as the real model does;
//   - activations rotate through four shared buffers (small and hot, as in the real model).
// Buffers are zero-filled: timing of these kernels does not depend on the values.
//
// Host copies: in eager mode, HtoD copies come from pageable memory (like torch.tensor(list)
// .to("cuda")) and every DtoH copy is followed by a stream synchronize (like .item()). Graph
// mode skips HtoD and DtoH rows, as the captured SmolVLA denoising loop does.
//
// Usage
//   build/chain_replay --list=bench/chains/smolvla_flow.csv [--steps=10] [--gemm=own|cublas]
//                      [--reps=20] [--warmup=3] [--per-op=out.csv] [--profile-once]
//   --per-op       also time every launch alone (CUDA graph of 20 copies, per-launch us)
//   --profile-once run one eager pass between cudaProfilerStart/Stop and exit (for
//                  `ncu --profile-from-start off`)
//   --weight-pool-mb=N  L2 working-set experiment: all GEMM weights share one pool of N MB
//                  (each weight at the next offset, wrapping), so the step touches about
//                  N MB of distinct weight data instead of the model's full weights. Shapes
//                  and kernels are unchanged. 0 (default) = one buffer per weight.
#include <cublas_v2.h>
#include <cuda_profiler_api.h>

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "../common/harness.cuh"
#include "../kernels/kernels.cuh"

#define CUBLAS_CHECK(call)                                                             \
  do {                                                                                 \
    cublasStatus_t st_ = (call);                                                       \
    if (st_ != CUBLAS_STATUS_SUCCESS) {                                                \
      std::fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)st_, __FILE__, __LINE__); \
      std::exit(1);                                                                    \
    }                                                                                  \
  } while (0)

struct Op {
  int idx = 0;
  std::string cat, op, kind;
  long long b = 1, m = 0, n = 0, k = 0, numel = 0;
  int n_in = 1, in_bytes = 4, out_bytes = 4, bias = 0, weight = 0;
  double traced_us = 0;
  int traced_kernels = 0;
  void* w = nullptr;   // weight buffer (GEMM with weight=1)
  void* wb = nullptr;  // bias buffer
};

static std::string arg_str(int argc, char** argv, const char* key, const char* def) {
  std::string k = std::string("--") + key + "=";
  for (int i = 1; i < argc; ++i)
    if (std::string(argv[i]).rfind(k, 0) == 0) return argv[i] + k.size();
  return def;
}
static bool has_flag(int argc, char** argv, const char* key) {
  std::string k = std::string("--") + key;
  for (int i = 1; i < argc; ++i)
    if (k == argv[i]) return true;
  return false;
}

static std::vector<Op> load(const std::string& path) {
  std::ifstream f(path);
  if (!f) {
    std::fprintf(stderr, "cannot open %s\n", path.c_str());
    std::exit(1);
  }
  std::string line;
  std::getline(f, line);  // header
  std::vector<Op> ops;
  while (std::getline(f, line)) {
    std::vector<std::string> c;
    std::stringstream ss(line);
    std::string x;
    while (std::getline(ss, x, ',')) c.push_back(x);
    while (c.size() < 16) c.push_back("");
    Op o;
    o.idx = std::stoi(c[0]);
    o.cat = c[1];
    o.op = c[2];
    o.b = std::stoll(c[3]);
    o.m = std::stoll(c[4]);
    o.n = std::stoll(c[5]);
    o.k = std::stoll(c[6]);
    o.numel = std::stoll(c[7]);
    o.n_in = std::stoi(c[8]);
    o.in_bytes = std::stoi(c[9]);
    o.out_bytes = std::stoi(c[10]);
    o.bias = std::stoi(c[11]);
    o.weight = std::stoi(c[12]);
    o.kind = c[13];
    o.traced_us = c[14].empty() ? 0 : std::stod(c[14]);
    o.traced_kernels = c[15].empty() ? 0 : std::stoi(c[15]);
    ops.push_back(o);
  }
  return ops;
}

// Bytes any one activation operand of `o` needs.
static size_t act_bytes(const Op& o) {
  auto mx = [](long long a, long long b) { return a > b ? a : b; };
  long long e = 0;
  if (o.cat == "gemm")
    e = mx(o.b * o.m * o.k, mx(o.weight ? 0 : o.b * o.k * o.n, o.b * o.m * o.n)) * o.in_bytes;
  else if (o.cat == "attention")
    e = mx(o.b * o.m * o.k, o.b * o.n * o.k) * 2;
  else if (o.cat == "softmax" || o.cat == "reduce" || o.cat == "norm")
    e = o.m * o.n * o.in_bytes;
  else if (o.cat == "scan")
    e = o.numel * o.in_bytes;
  else if (o.cat == "elementwise")
    e = o.numel * mx(o.in_bytes, o.out_bytes);
  else
    e = o.numel;  // memcpy, memset: bytes
  return (size_t)e;
}

struct Ctx {
  void* act[4] = {};
  void* host_pageable = nullptr;
  void* host_pinned = nullptr;
  cublasHandle_t blas = nullptr;
  void* scores = nullptr;  // attention scores for the cuBLAS math path
  bool cublas = false;
  bool sync_dtoh = true;
};

static void blas_gemm(const Ctx& c, const Op& o, const void* A, const void* B, void* C,
                      long long sB, cudaStream_t s) {
  const float one = 1.f, zero = 0.f;
  const int M = (int)o.m, N = (int)o.n, K = (int)o.k;
  CUBLAS_CHECK(cublasSetStream(c.blas, s));
  if (o.in_bytes == 2)
    CUBLAS_CHECK(cublasGemmStridedBatchedEx(c.blas, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &one, B,
                                            CUDA_R_16F, N, sB, A, CUDA_R_16F, K,
                                            (long long)M * K, &zero, C, CUDA_R_16F, N,
                                            (long long)M * N, (int)o.b, CUBLAS_COMPUTE_32F,
                                            CUBLAS_GEMM_DEFAULT));
  else
    CUBLAS_CHECK(cublasSgemmStridedBatched(c.blas, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &one,
                                           (const float*)B, N, sB, (const float*)A, K,
                                           (long long)M * K, &zero, (float*)C, N,
                                           (long long)M * N, (int)o.b));
}

// Enqueues one row. Returns false if the row was skipped (host copies in graph mode).
static bool run_op(const Ctx& c, const Op& o, int i, bool graph, cudaStream_t s) {
  void* a0 = c.act[i & 3];
  void* a1 = c.act[(i + 1) & 3];
  void* a2 = c.act[(i + 2) & 3];
  void* out = c.act[(i + 3) & 3];
  if (o.cat == "gemm") {
    const void* B = o.weight ? o.w : a1;
    long long sB = o.weight ? 0 : o.k * o.n;
    if (c.cublas) {
      blas_gemm(c, o, a0, B, out, sB, s);
    } else if (o.in_bytes == 2) {
      vk::gemm_f16<__half>((const __half*)a0, (const __half*)B, (__half*)out,
                           (const __half*)o.wb, (int)o.b, (int)o.m, (int)o.n, (int)o.k,
                           o.m * o.k, sB, o.m * o.n, s);
    } else {
      vk::gemm_f32((const float*)a0, (const float*)B, (float*)out, (const float*)o.wb, (int)o.b,
                   (int)o.m, (int)o.n, (int)o.k, o.m * o.k, sB, o.m * o.n, s);
    }
  } else if (o.cat == "attention" && c.cublas) {
    // Math-path attention with library GEMMs: S = Q K^T (cuBLAS), softmax (ours), O = P V.
    const float one = 1.f, zero = 0.f;
    const int M = (int)o.m, N = (int)o.n, D = (int)o.k, B = (int)o.b;
    CUBLAS_CHECK(cublasSetStream(c.blas, s));
    CUBLAS_CHECK(cublasGemmStridedBatchedEx(c.blas, CUBLAS_OP_T, CUBLAS_OP_N, N, M, D, &one, a1,
                                            CUDA_R_16F, D, (long long)N * D, a0, CUDA_R_16F, D,
                                            (long long)M * D, &zero, c.scores, CUDA_R_16F, N,
                                            (long long)M * N, B, CUBLAS_COMPUTE_32F,
                                            CUBLAS_GEMM_DEFAULT));
    vk::softmax(c.scores, c.scores, B * M, N, 2, s);
    CUBLAS_CHECK(cublasGemmStridedBatchedEx(c.blas, CUBLAS_OP_N, CUBLAS_OP_N, D, M, N, &one, a2,
                                            CUDA_R_16F, D, (long long)N * D, c.scores,
                                            CUDA_R_16F, N, (long long)M * N, &zero, out,
                                            CUDA_R_16F, D, (long long)M * D, B,
                                            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
  } else if (o.cat == "attention") {
    vk::attention((const __half*)a0, (const __half*)a1, (const __half*)a2, (__half*)out,
                  (int)o.b, (int)o.m, (int)o.n, (int)o.k, s);
  } else if (o.cat == "softmax") {
    vk::softmax(a0, out, (int)o.m, (int)o.n, o.in_bytes, s);
  } else if (o.cat == "reduce") {
    vk::row_mean(a0, (float*)out, (int)o.m, (int)o.n, o.in_bytes, s);
  } else if (o.cat == "norm") {
    vk::norm<false>(a0, a1, a2, out, (int)o.m, (int)o.n, o.in_bytes == 2 ? 2 : 4, s);
  } else if (o.cat == "scan") {
    vk::scan(a0, out, (int)o.numel, o.in_bytes == 2 ? 2 : 4, s);
  } else if (o.cat == "elementwise") {
    vk::elementwise(a0, a1, a2, out, o.numel, o.n_in, o.in_bytes, o.out_bytes, s);
  } else if (o.cat == "memset") {
    CUDA_CHECK(cudaMemsetAsync(out, 0, (size_t)o.numel, s));
  } else if (o.cat == "memcpy") {
    if (o.kind == "DtoD") {
      CUDA_CHECK(cudaMemcpyAsync(out, a0, (size_t)o.numel, cudaMemcpyDeviceToDevice, s));
    } else if (graph) {
      return false;
    } else if (o.kind == "HtoD") {
      CUDA_CHECK(cudaMemcpyAsync(out, c.host_pageable, (size_t)o.numel,
                                 cudaMemcpyHostToDevice, s));
    } else {
      CUDA_CHECK(cudaMemcpyAsync(c.host_pinned, a0, (size_t)o.numel, cudaMemcpyDeviceToHost, s));
      if (c.sync_dtoh) CUDA_CHECK(cudaStreamSynchronize(s));
    }
  } else {
    std::fprintf(stderr, "unknown category %s (row %d)\n", o.cat.c_str(), o.idx);
    std::exit(1);
  }
  return true;
}

int main(int argc, char** argv) {
  const std::string list = arg_str(argc, argv, "list", "bench/chains/smolvla_flow.csv");
  const std::string gemm = arg_str(argc, argv, "gemm", "own");
  const std::string per_op = arg_str(argc, argv, "per-op", "");
  const int steps = bench::arg_int(argc, argv, "steps", 1);
  const int reps = bench::arg_int(argc, argv, "reps", 20);
  const int warmup = bench::arg_int(argc, argv, "warmup", 3);
  const bool profile_once = has_flag(argc, argv, "profile-once");
  const int pool_mb = bench::arg_int(argc, argv, "weight-pool-mb", 0);

  std::vector<Op> ops = load(list);
  std::string name = list.substr(list.find_last_of('/') + 1);
  name = name.substr(0, name.find('.'));

  Ctx c;
  c.cublas = gemm == "cublas";
  c.sync_dtoh = bench::arg_int(argc, argv, "sync-dtoh", 1) != 0;
  size_t amax = 1 << 20, hmax = 1 << 12, wbytes = 0;
  double traced = 0;
  int traced_k = 0;
  for (auto& o : ops) {
    amax = std::max(amax, act_bytes(o));
    if (o.cat == "memcpy") hmax = std::max(hmax, (size_t)o.numel);
    traced += o.traced_us;
    traced_k += o.traced_kernels;
  }
  for (int i = 0; i < 4; ++i) {
    CUDA_CHECK(cudaMalloc(&c.act[i], amax));
    CUDA_CHECK(cudaMemset(c.act[i], 0, amax));
  }
  size_t pool_bytes = 0, pool_off = 0;
  char* pool = nullptr;
  if (pool_mb > 0) {
    size_t wmax = 0;
    for (auto& o : ops)
      if (o.cat == "gemm" && o.weight) wmax = std::max(wmax, (size_t)(o.k * o.n * o.in_bytes));
    pool_bytes = std::max((size_t)pool_mb << 20, wmax);
    CUDA_CHECK(cudaMalloc(&pool, pool_bytes));
    CUDA_CHECK(cudaMemset(pool, 0, pool_bytes));
  }
  for (auto& o : ops) {
    if (o.cat != "gemm" || !o.weight) continue;
    size_t wb = (size_t)o.k * o.n * o.in_bytes;
    if (pool) {
      pool_off = (pool_off + 255) & ~(size_t)255;  // keep 256-byte alignment for cuBLAS/WMMA
      if (pool_off + wb > pool_bytes) pool_off = 0;
      o.w = pool + pool_off;
      pool_off += wb;
    } else {
      CUDA_CHECK(cudaMalloc(&o.w, wb));
      CUDA_CHECK(cudaMemset(o.w, 0, wb));
    }
    wbytes += wb;
    if (o.bias) {
      CUDA_CHECK(cudaMalloc(&o.wb, (size_t)o.n * o.in_bytes));
      CUDA_CHECK(cudaMemset(o.wb, 0, (size_t)o.n * o.in_bytes));
    }
  }
  size_t smax = 0;
  for (auto& o : ops)
    if (o.cat == "attention") smax = std::max(smax, (size_t)(o.b * o.m * o.n * 2));
  if (smax) CUDA_CHECK(cudaMalloc(&c.scores, smax));
  c.host_pageable = std::calloc(hmax, 1);
  CUDA_CHECK(cudaMallocHost(&c.host_pinned, hmax));
  CUBLAS_CHECK(cublasCreate(&c.blas));
  vk::gemm_workspace_init();  // split-K partials; must precede graph capture
  void* ws;
  const size_t ws_bytes = 32u << 20;
  CUDA_CHECK(cudaMalloc(&ws, ws_bytes));
  CUBLAS_CHECK(cublasSetWorkspace(c.blas, ws, ws_bytes));
  cudaStream_t s;
  CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  CUDA_CHECK(cudaDeviceSynchronize());

  int run_eager = 0, run_graph = 0;
  auto eager = [&](cudaStream_t st) {
    run_eager = 0;
    for (int t = 0; t < steps; ++t)
      for (size_t i = 0; i < ops.size(); ++i) run_eager += run_op(c, ops[i], (int)i, false, st);
  };
  auto graph_body = [&](cudaStream_t st) {
    run_graph = 0;
    for (int t = 0; t < steps; ++t)
      for (size_t i = 0; i < ops.size(); ++i) run_graph += run_op(c, ops[i], (int)i, true, st);
  };

  std::fprintf(stderr,
               "%s: %zu rows (%d traced kernels, traced GPU %.1f us per step), weights %.1f MB, "
               "activation buffers 4 x %.2f MB, gemm=%s, steps=%d, weight pool %s\n",
               name.c_str(), ops.size(), traced_k, traced, wbytes / 1e6, amax / 1e6,
               gemm.c_str(), steps,
               pool ? (std::to_string(pool_bytes >> 20) + " MB").c_str() : "off");

  if (profile_once) {
    eager(s);  // warm up (module loading, cuBLAS heuristics)
    CUDA_CHECK(cudaStreamSynchronize(s));
    CUDA_CHECK(cudaProfilerStart());
    eager(s);
    CUDA_CHECK(cudaStreamSynchronize(s));
    CUDA_CHECK(cudaProfilerStop());
    return 0;
  }

  bench::csv_header();
  auto cfg = [&](int launched) {
    char b[256];
    std::snprintf(b, sizeof b, "gemm=%s;steps=%d;rows=%zu;launched=%d;traced_kernels=%d;"
                  "traced_gpu_us=%.1f%s",
                  gemm.c_str(), steps, ops.size(), launched, traced_k * steps, traced * steps,
                  pool_mb > 0 ? (";weight_pool_mb=" + std::to_string(pool_mb)).c_str() : "");
    return std::string(b);
  };
  auto ev = bench::time_events(s, eager, warmup, reps);
  bench::csv_row("chain", name, cfg(run_eager), "eager", "event", ev);
  auto ho = bench::time_host(s, eager, warmup, reps);
  bench::csv_row("chain", name, cfg(run_eager), "eager", "host", ho);
  cudaGraphExec_t g = bench::capture_graph(s, graph_body);
  auto replay = [&](cudaStream_t st) { CUDA_CHECK(cudaGraphLaunch(g, st)); };
  bench::csv_row("chain", name, cfg(run_graph), "graph", "event",
                 bench::time_events(s, replay, warmup, reps));
  bench::csv_row("chain", name, cfg(run_graph), "graph", "host",
                 bench::time_host(s, replay, warmup, reps));
  CUDA_CHECK(cudaGraphExecDestroy(g));

  if (!per_op.empty()) {
    // Each row alone: a graph of R copies, so launch cost is the graph node cost only.
    const int R = 20;
    FILE* f = std::fopen(per_op.c_str(), "w");
    std::fprintf(f, "idx,cat,op,gemm,traced_us,traced_kernels,graph_us_per_launch\n");
    for (size_t i = 0; i < ops.size(); ++i) {
      const Op& o = ops[i];
      if (o.cat == "memcpy" && o.kind != "DtoD") continue;
      auto body = [&](cudaStream_t st) {
        for (int r = 0; r < R; ++r) run_op(c, o, (int)i, true, st);
      };
      cudaGraphExec_t gi = bench::capture_graph(s, body);
      auto rp = [&](cudaStream_t st) { CUDA_CHECK(cudaGraphLaunch(gi, st)); };
      auto st = bench::time_events(s, rp, 2, 5);
      std::fprintf(f, "%d,%s,%s,%s,%.3f,%d,%.4f\n", o.idx, o.cat.c_str(), o.op.c_str(),
                   gemm.c_str(), o.traced_us, o.traced_kernels, st.median / R);
      CUDA_CHECK(cudaGraphExecDestroy(gi));
    }
    std::fclose(f);
    std::fprintf(stderr, "wrote %s\n", per_op.c_str());
  }
  return 0;
}
