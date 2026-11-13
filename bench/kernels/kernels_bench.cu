// Standalone kernel benchmarks at the shapes traced from SmolVLA (bench/chains/*.csv):
// our kernels (bench/kernels/kernels.cuh) and cuBLAS for the GEMMs, eager vs CUDA graph.
//
// Every case is first checked for correctness (GEMMs against cuBLAS, the rest against a CPU
// reference); a failed check prints FAIL on stderr and the binary exits with status 1.
// Timing: a chain of --chain launches of the same kernel (default 50), timed with cudaEvents
// (eager and graph) and with a host clock (eager); rows report microseconds PER KERNEL.
//
//   build/kernels_bench [--reps=30] [--warmup=5] [--chain=50] > kernels.csv
#include <cublas_v2.h>

#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "../common/harness.cuh"
#include "kernels.cuh"

#define CUBLAS_CHECK(call)                                                       \
  do {                                                                           \
    cublasStatus_t st_ = (call);                                                 \
    if (st_ != CUBLAS_STATUS_SUCCESS) {                                          \
      std::fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)st_, __FILE__, __LINE__); \
      std::exit(1);                                                              \
    }                                                                            \
  } while (0)

using bench::Stats;

static int g_reps, g_warmup, g_chain;
static bool g_fail = false;
static cudaStream_t g_stream;
static cublasHandle_t g_blas;

// Device buffer with host copy helpers.
template <class T>
struct Buf {
  T* d = nullptr;
  size_t n = 0;
  explicit Buf(size_t n_, uint32_t seed = 1, float scale = 1.f) : n(n_) {
    CUDA_CHECK(cudaMalloc(&d, n * sizeof(T)));
    vk::init(d, n, seed, scale);
    CUDA_CHECK(cudaDeviceSynchronize());  // init runs on the legacy stream; g_stream is non-blocking
  }
  ~Buf() { cudaFree(d); }
  std::vector<float> host() const {
    std::vector<T> h(n);
    CUDA_CHECK(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
    std::vector<float> f(n);
    for (size_t i = 0; i < n; ++i) f[i] = to_float(h[i]);
    return f;
  }
  static float to_float(float x) { return x; }
  static float to_float(__half x) { return __half2float(x); }
};

static void check(const std::string& what, const std::vector<float>& got,
                  const std::vector<float>& ref, float tol) {
  double max_err = 0, max_ref = 1e-6;
  for (size_t i = 0; i < ref.size(); ++i) {
    max_err = std::max(max_err, (double)std::fabs(got[i] - ref[i]));
    max_ref = std::max(max_ref, (double)std::fabs(ref[i]));
  }
  double rel = max_err / max_ref;
  bool ok = rel <= tol && std::isfinite(rel) && max_ref > 1e-3;
  std::fprintf(stderr, "%s %-48s max rel err %.2e (tol %.0e, max |ref| %.3g)\n",
               ok ? "ok  " : "FAIL", what.c_str(), rel, tol, max_ref);
  if (!ok) g_fail = true;
}

// Times one kernel: chain of g_chain launches, eager (event + host) and graph (event).
static void time_case(const std::string& kernel, const std::string& config,
                      const std::function<void(cudaStream_t)>& launch) {
  auto chain = [&](cudaStream_t s) {
    for (int i = 0; i < g_chain; ++i) launch(s);
  };
  const double per = 1.0 / g_chain;
  std::string cfg = config + ";chain=" + std::to_string(g_chain);
  bench::csv_row("kernels", kernel, cfg, "eager", "event",
                 bench::scale(bench::time_events(g_stream, chain, g_warmup, g_reps), per));
  bench::csv_row("kernels", kernel, cfg, "eager", "host",
                 bench::scale(bench::time_host(g_stream, chain, g_warmup, g_reps), per));
  cudaGraphExec_t g = bench::capture_graph(g_stream, chain);
  auto replay = [&](cudaStream_t s) { CUDA_CHECK(cudaGraphLaunch(g, s)); };
  bench::csv_row("kernels", kernel, cfg, "graph", "event",
                 bench::scale(bench::time_events(g_stream, replay, g_warmup, g_reps), per));
  CUDA_CHECK(cudaGraphExecDestroy(g));
}

// Row-major C = A B via column-major cuBLAS: C^T = B^T A^T.
static void blas_f16(const __half* A, const __half* B, __half* C, int batch, int M, int N, int K,
                     cudaStream_t s) {
  const float one = 1.f, zero = 0.f;
  CUBLAS_CHECK(cublasSetStream(g_blas, s));
  CUBLAS_CHECK(cublasGemmStridedBatchedEx(g_blas, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &one, B,
                                          CUDA_R_16F, N, (long long)K * N, A, CUDA_R_16F, K,
                                          (long long)M * K, &zero, C, CUDA_R_16F, N,
                                          (long long)M * N, batch, CUBLAS_COMPUTE_32F,
                                          CUBLAS_GEMM_DEFAULT));
}
static void blas_f32(const float* A, const float* B, float* C, int batch, int M, int N, int K,
                     cudaStream_t s) {
  const float one = 1.f, zero = 0.f;
  CUBLAS_CHECK(cublasSetStream(g_blas, s));
  CUBLAS_CHECK(cublasSgemmStridedBatched(g_blas, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &one, B, N,
                                         (long long)K * N, A, K, (long long)M * K, &zero, C, N,
                                         (long long)M * N, batch));
}

static std::string mnk(int b, int m, int n, int k) {
  return "b=" + std::to_string(b) + ";m=" + std::to_string(m) + ";n=" + std::to_string(n) +
         ";k=" + std::to_string(k);
}

static void gemm_case_f16(int b, int M, int N, int K, const char* where) {
  Buf<__half> A((size_t)b * M * K, 11, 1.f), B((size_t)b * K * N, 12, 1.f);
  Buf<__half> C((size_t)b * M * N), R((size_t)b * M * N);
  vk::gemm_f16<__half>(A.d, B.d, C.d, nullptr, b, M, N, K, (long long)M * K, (long long)K * N,
                       (long long)M * N, g_stream);
  blas_f16(A.d, B.d, R.d, b, M, N, K, g_stream);
  CUDA_CHECK(cudaStreamSynchronize(g_stream));
  std::string cfg = mnk(b, M, N, K) + ";dtype=f16;src=" + where;
  check("gemm_wmma_f16 " + cfg, C.host(), R.host(), 2e-2f);
  time_case("gemm_wmma_f16", cfg, [&](cudaStream_t s) {
    vk::gemm_f16<__half>(A.d, B.d, C.d, nullptr, b, M, N, K, (long long)M * K, (long long)K * N,
                         (long long)M * N, s);
  });
  time_case("cublas_gemm_f16", cfg,
            [&](cudaStream_t s) { blas_f16(A.d, B.d, R.d, b, M, N, K, s); });
}

static void gemm_case_f32(int b, int M, int N, int K, const char* where) {
  Buf<float> A((size_t)b * M * K, 21, 1.f), B((size_t)b * K * N, 22, 1.f);
  Buf<float> C((size_t)b * M * N), R((size_t)b * M * N);
  vk::gemm_f32(A.d, B.d, C.d, nullptr, b, M, N, K, (long long)M * K, (long long)K * N,
               (long long)M * N, g_stream);
  blas_f32(A.d, B.d, R.d, b, M, N, K, g_stream);
  CUDA_CHECK(cudaStreamSynchronize(g_stream));
  std::string cfg = mnk(b, M, N, K) + ";dtype=f32;src=" + where;
  check("gemm_simt_f32 " + cfg, C.host(), R.host(), 1e-4f);
  time_case("gemm_simt_f32", cfg, [&](cudaStream_t s) {
    vk::gemm_f32(A.d, B.d, C.d, nullptr, b, M, N, K, (long long)M * K, (long long)K * N,
                 (long long)M * N, s);
  });
  time_case("cublas_sgemm", cfg,
            [&](cudaStream_t s) { blas_f32(A.d, B.d, R.d, b, M, N, K, s); });
}

static void norm_case(int rows, int cols, bool rms, const char* where) {
  Buf<__half> x((size_t)rows * cols, 31, 4.f), w(cols, 32, 2.f), bb(cols, 33, 1.f),
      y((size_t)rows * cols);
  auto run = [&](cudaStream_t s) {
    if (rms)
      vk::norm<true>(x.d, w.d, nullptr, y.d, rows, cols, 2, s);
    else
      vk::norm<false>(x.d, w.d, bb.d, y.d, rows, cols, 2, s);
  };
  run(g_stream);
  CUDA_CHECK(cudaStreamSynchronize(g_stream));
  auto hx = x.host(), hw = w.host(), hb = bb.host();
  std::vector<float> ref(hx.size());
  for (int r = 0; r < rows; ++r) {
    double s = 0, ss = 0;
    for (int c = 0; c < cols; ++c) {
      s += hx[r * cols + c];
      ss += (double)hx[r * cols + c] * hx[r * cols + c];
    }
    double mean = rms ? 0 : s / cols, var = rms ? ss / cols : ss / cols - mean * mean;
    double rstd = 1.0 / std::sqrt(var + 1e-6);
    for (int c = 0; c < cols; ++c)
      ref[r * cols + c] = (float)((hx[r * cols + c] - mean) * rstd * hw[c] + (rms ? 0 : hb[c]));
  }
  std::string cfg = "rows=" + std::to_string(rows) + ";cols=" + std::to_string(cols) +
                    ";dtype=f16;src=" + where;
  const char* name = rms ? "rmsnorm" : "layernorm";
  check(std::string(name) + " " + cfg, y.host(), ref, 1e-2f);
  time_case(name, cfg, run);
}

static void softmax_case(int rows, int cols, const char* where) {
  Buf<float> x((size_t)rows * cols, 41, 8.f), y((size_t)rows * cols);
  auto run = [&](cudaStream_t s) { vk::softmax(x.d, y.d, rows, cols, 4, s); };
  run(g_stream);
  CUDA_CHECK(cudaStreamSynchronize(g_stream));
  auto hx = x.host();
  std::vector<float> ref(hx.size());
  for (int r = 0; r < rows; ++r) {
    float m = -INFINITY;
    for (int c = 0; c < cols; ++c) m = std::max(m, hx[r * cols + c]);
    double s = 0;
    for (int c = 0; c < cols; ++c) s += std::exp(hx[r * cols + c] - m);
    for (int c = 0; c < cols; ++c) ref[r * cols + c] = (float)(std::exp(hx[r * cols + c] - m) / s);
  }
  std::string cfg = "rows=" + std::to_string(rows) + ";cols=" + std::to_string(cols) +
                    ";dtype=f32;src=" + where;
  check("softmax " + cfg, y.host(), ref, 1e-4f);
  time_case("softmax", cfg, run);
}

static void rope_case(int tokens, int heads, int D, const char* where) {
  Buf<__half> x((size_t)tokens * heads * D, 51, 2.f);
  Buf<float> cs((size_t)tokens * D / 2, 52, 2.f), sn((size_t)tokens * D / 2, 53, 2.f);
  auto hx = x.host(), hc = cs.host(), hs = sn.host();
  vk::rope(x.d, cs.d, sn.d, tokens, heads, D, g_stream);
  CUDA_CHECK(cudaStreamSynchronize(g_stream));
  std::vector<float> ref(hx);
  for (int t = 0; t < tokens; ++t)
    for (int h = 0; h < heads; ++h)
      for (int d = 0; d < D / 2; ++d) {
        size_t o = ((size_t)t * heads + h) * D;
        float x1 = hx[o + d], x2 = hx[o + d + D / 2], c = hc[t * D / 2 + d], s = hs[t * D / 2 + d];
        ref[o + d] = x1 * c - x2 * s;
        ref[o + d + D / 2] = x2 * c + x1 * s;
      }
  std::string cfg = "tokens=" + std::to_string(tokens) + ";heads=" + std::to_string(heads) +
                    ";d=" + std::to_string(D) + ";dtype=f16;src=" + where;
  check("rope " + cfg, x.host(), ref, 1e-2f);
  // In place: repeated launches keep rotating the same data, which is fine for timing.
  time_case("rope", cfg,
            [&](cudaStream_t s) { vk::rope(x.d, cs.d, sn.d, tokens, heads, D, s); });
}

static void attention_case(int BH, int M, int N, int D, const char* where) {
  Buf<__half> Q((size_t)BH * M * D, 61, 2.f), K((size_t)BH * N * D, 62, 2.f),
      V((size_t)BH * N * D, 63, 2.f), O((size_t)BH * M * D);
  auto run = [&](cudaStream_t s) { vk::attention(Q.d, K.d, V.d, O.d, BH, M, N, D, s); };
  run(g_stream);
  CUDA_CHECK(cudaStreamSynchronize(g_stream));
  auto hq = Q.host(), hk = K.host(), hv = V.host();
  std::vector<float> ref((size_t)BH * M * D);
  const int check_rows = std::min(M, 8);  // CPU reference on the first rows of every head
  std::vector<float> got_all = O.host(), got, refc;
  for (int b = 0; b < BH; ++b)
    for (int i = 0; i < check_rows; ++i) {
      std::vector<double> p(N);
      double m = -1e30;
      for (int j = 0; j < N; ++j) {
        double s = 0;
        for (int d = 0; d < D; ++d)
          s += (double)hq[((size_t)b * M + i) * D + d] * hk[((size_t)b * N + j) * D + d];
        p[j] = s / std::sqrt((double)D);
        m = std::max(m, p[j]);
      }
      double l = 0;
      for (int j = 0; j < N; ++j) l += (p[j] = std::exp(p[j] - m));
      for (int d = 0; d < D; ++d) {
        double o = 0;
        for (int j = 0; j < N; ++j) o += p[j] * hv[((size_t)b * N + j) * D + d];
        refc.push_back((float)(o / l));
        got.push_back(got_all[((size_t)b * M + i) * D + d]);
      }
    }
  std::string cfg = "bh=" + std::to_string(BH) + ";m=" + std::to_string(M) +
                    ";n=" + std::to_string(N) + ";d=" + std::to_string(D) +
                    ";dtype=f16;src=" + where;
  check("attention " + cfg, got, refc, 2e-2f);
  time_case("attention", cfg, run);
}

template <vk::Act ACT>
static void act_case(const char* name, long long n, const char* where) {
  Buf<__half> x(n, 71, 4.f), g(n, 72, 2.f), y(n);
  auto run = [&](cudaStream_t s) { vk::activation<ACT>(x.d, g.d, y.d, n, s); };
  run(g_stream);
  CUDA_CHECK(cudaStreamSynchronize(g_stream));
  auto hx = x.host(), hg = g.host();
  std::vector<float> ref(n);
  for (long long i = 0; i < n; ++i) {
    float v = hx[i];
    if (ACT == vk::Act::GeLU)
      ref[i] = 0.5f * v * (1.f + std::tanh(0.7978845608f * (v + 0.044715f * v * v * v)));
    else
      ref[i] = v / (1.f + std::exp(-v)) * (ACT == vk::Act::SiLUMul ? hg[i] : 1.f);
  }
  std::string cfg = "numel=" + std::to_string(n) + ";dtype=f16;src=" + where;
  check(std::string(name) + " " + cfg, y.host(), ref, 1e-2f);
  time_case(name, cfg, run);
}

static void elementwise_case(long long n, int nin, int ib, int ob, const char* where) {
  void *a, *b, *c, *o;
  size_t mx = (size_t)n * 8;
  CUDA_CHECK(cudaMalloc(&a, mx));
  CUDA_CHECK(cudaMalloc(&b, mx));
  CUDA_CHECK(cudaMalloc(&c, mx));
  CUDA_CHECK(cudaMalloc(&o, mx));
  CUDA_CHECK(cudaMemset(a, 0, mx));
  CUDA_CHECK(cudaMemset(b, 0, mx));
  CUDA_CHECK(cudaMemset(c, 0, mx));
  CUDA_CHECK(cudaDeviceSynchronize());
  std::string cfg = "numel=" + std::to_string(n) + ";n_in=" + std::to_string(nin) +
                    ";in_bytes=" + std::to_string(ib) + ";out_bytes=" + std::to_string(ob) +
                    ";src=" + where;
  time_case("elementwise", cfg,
            [&](cudaStream_t s) { vk::elementwise(a, b, c, o, n, nin, ib, ob, s); });
  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
  cudaFree(o);
}

int main(int argc, char** argv) {
  g_reps = bench::arg_int(argc, argv, "reps", 30);
  g_warmup = bench::arg_int(argc, argv, "warmup", 5);
  g_chain = bench::arg_int(argc, argv, "chain", 50);
  CUDA_CHECK(cudaStreamCreateWithFlags(&g_stream, cudaStreamNonBlocking));
  vk::gemm_workspace_init();  // split-K partials; must precede graph capture
  CUBLAS_CHECK(cublasCreate(&g_blas));
  // Fixed workspace so cuBLAS never allocates during graph capture.
  void* ws;
  const size_t ws_bytes = 32u << 20;
  CUDA_CHECK(cudaMalloc(&ws, ws_bytes));
  CUBLAS_CHECK(cublasSetWorkspace(g_blas, ws, ws_bytes));
  bench::csv_header();

  // SmolVLA action expert (flow step), FP16 weights: M = 50 action tokens.
  gemm_case_f16(1, 50, 2048, 720, "expert_mlp_up");
  gemm_case_f16(1, 50, 720, 2048, "expert_mlp_down");
  gemm_case_f16(1, 50, 960, 720, "expert_q");
  gemm_case_f16(1, 50, 320, 720, "expert_kv");
  gemm_case_f16(1, 50, 720, 960, "expert_o");
  gemm_case_f16(1, 241, 320, 320, "expert_prefix_kv");
  gemm_case_f16(15, 50, 64, 291, "expert_attn_pv");
  // SmolVLA vision tower (prefix), FP16: 1024 patches of 768.
  gemm_case_f16(1, 1024, 768, 768, "vision_qkvo");
  gemm_case_f16(1, 1024, 3072, 768, "vision_mlp_up");
  gemm_case_f16(1, 1024, 768, 3072, "vision_mlp_down");
  // FP32 GEMMs in the flow step (time embedding MLP, attention scores in FP32).
  gemm_case_f32(1, 50, 720, 1440, "expert_time_mlp");
  gemm_case_f32(1, 50, 720, 720, "expert_time_mlp2");
  gemm_case_f32(15, 50, 291, 64, "expert_attn_qk");

  norm_case(50, 720, true, "expert_rmsnorm");
  norm_case(241, 960, true, "vlm_rmsnorm");
  norm_case(1024, 768, false, "vision_layernorm");
  softmax_case(15 * 50, 241, "expert_self_attn");
  softmax_case(15 * 50, 291, "expert_cross_attn");
  rope_case(50, 15, 64, "expert_q_rope");
  rope_case(50, 5, 64, "expert_k_rope");
  attention_case(12, 1024, 1024, 64, "vision_sdpa");
  attention_case(15, 50, 291, 64, "expert_attn_fused");
  act_case<vk::Act::SiLUMul>("silu_mul", 50 * 2048, "expert_mlp_gate");
  act_case<vk::Act::GeLU>("gelu", 1024 * 3072, "vision_mlp_act");
  elementwise_case(50 * 720, 2, 4, 4, "expert_residual_f32");
  elementwise_case(50 * 720, 1, 4, 2, "expert_cast_f32_f16");
  elementwise_case(50 * 15 * 64, 1, 2, 4, "expert_cast_f16_f32");
  elementwise_case(1024 * 768, 2, 2, 2, "vision_residual_f16");
  elementwise_case(1, 0, 4, 4, "scalar_fill");

  cublasDestroy(g_blas);
  cudaFree(ws);
  if (g_fail) {
    std::fprintf(stderr, "correctness check FAILED\n");
    return 1;
  }
  std::fprintf(stderr, "all correctness checks passed\n");
  return 0;
}
