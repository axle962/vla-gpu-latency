// Our own CUDA kernels for the operations in the traced VLA steps (SmolVLA, OpenVLA):
// GEMM (FP16 tensor-core WMMA and FP32 SIMT, batched, optional bias), fused attention,
// softmax, row reductions, LayerNorm, RMSNorm, RoPE, activations, generic elementwise, scan.
//
// Design goals: code we control (course rule: do not rely only on cuBLAS), sm_75 baseline with
// no BF16/FP8, and launch-heavy batch-1 shapes (M = 50 action tokens, odd K such as 241).
// Kernels favour simplicity and correctness over peak speed; cuBLAS is the reference in
// kernels_bench.cu. All launchers take a stream so chains can be captured into CUDA graphs.
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <cstdint>

namespace vk {

// ---------------------------------------------------------------- element type helpers
template <int B> struct ElemT;
template <> struct ElemT<1> { using T = uint8_t; };
template <> struct ElemT<2> { using T = __half; };
template <> struct ElemT<4> { using T = float; };
template <> struct ElemT<8> { using T = double; };

template <class T> __device__ __forceinline__ float to_f(T x) { return static_cast<float>(x); }
template <> __device__ __forceinline__ float to_f<__half>(__half x) { return __half2float(x); }
template <class T> __device__ __forceinline__ T from_f(float x) { return static_cast<T>(x); }
template <> __device__ __forceinline__ __half from_f<__half>(float x) { return __float2half(x); }
template <> __device__ __forceinline__ uint8_t from_f<uint8_t>(float x) {
  return static_cast<uint8_t>(x > 0.f);
}

__device__ __forceinline__ float warp_sum(float v) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
__device__ __forceinline__ float warp_max(float v) {
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}
// Block-wide reduction (sum or max) for blockDim.x a multiple of 32, <= 1024.
template <bool MAX>
__device__ float block_reduce(float v) {
  __shared__ float red[32];
  __shared__ float result;
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = blockDim.x >> 5;
  v = MAX ? warp_max(v) : warp_sum(v);
  if (lane == 0) red[wid] = v;
  __syncthreads();
  if (wid == 0) {
    float x = lane < nw ? red[lane] : (MAX ? -INFINITY : 0.f);
    x = MAX ? warp_max(x) : warp_sum(x);
    if (lane == 0) result = x;
  }
  __syncthreads();
  float r = result;
  __syncthreads();  // red/result may be reused by a following call
  return r;
}

// ---------------------------------------------------------------- init (deterministic data)
template <class T>
__global__ void init_kernel(T* p, size_t n, uint32_t seed, float scale) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  for (; i < n; i += (size_t)gridDim.x * blockDim.x) {
    uint32_t h = (uint32_t)i * 2654435761u ^ seed;
    h ^= h >> 15;
    h *= 2246822519u;
    h ^= h >> 13;
    float u = (h & 0xffffff) / 16777216.f - 0.5f;  // [-0.5, 0.5)
    p[i] = from_f<T>(u * scale);
  }
}
template <class T>
inline void init(T* p, size_t n, uint32_t seed, float scale, cudaStream_t s = 0) {
  init_kernel<T><<<1024, 256, 0, s>>>(p, n, seed, scale);
}

// ---------------------------------------------------------------- GEMM shared state
// Split-K workspace (FP32 partial sums). Allocate it once with gemm_workspace_init() before
// any CUDA-graph capture: cudaMalloc is not allowed while a stream is capturing.
inline float*& ws_ptr() { static float* p = nullptr; return p; }
inline size_t& ws_floats() { static size_t n = 0; return n; }
inline void gemm_workspace_init(size_t bytes = 64u << 20) {
  if (!ws_ptr() && cudaMalloc(&ws_ptr(), bytes) == cudaSuccess) ws_floats() = bytes / 4;
}
inline int num_sms() {
  static int n = 0;
  if (!n) cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, 0);
  return n;
}
// Number of K splits so that a small grid (batch-1 shapes: M = 50) still fills the GPU.
// Each split keeps at least 128 of K; the partials must fit in the workspace.
inline int choose_splits(long long blocks, int K, long long out_elems, int bk) {
  if (K < 256 || !ws_ptr()) return 1;
  long long target = 2LL * num_sms();
  long long s = (target + blocks - 1) / blocks;
  s = std::min<long long>(s, std::min(K / 128, 16));
  while (s > 1 && s * out_elems > (long long)ws_floats()) --s;
  if (s <= 1) return 1;
  int chunk = ((K + s - 1) / s + bk - 1) / bk * bk;
  return (K + chunk - 1) / chunk;
}
inline int split_chunk(int K, int splits, int bk) {
  return ((K + splits - 1) / splits + bk - 1) / bk * bk;
}

// out[b][i] = sum_s ws[s][b][i] (+ bias[i % N]); finishes a split-K GEMM.
template <class TC>
__global__ void splitk_reduce(const float* __restrict__ ws, TC* __restrict__ C,
                              const TC* __restrict__ bias, long long per_split, int splits,
                              int N) {
  long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (i >= per_split) return;
  float v = 0.f;
  for (int s = 0; s < splits; ++s) v += ws[s * per_split + i];
  if (bias) v += to_f<TC>(bias[i % N]);
  C[i] = from_f<TC>(v);
}

// ---------------------------------------------------------------- GEMM, FP16 tensor cores
// C[b] (MxN, row-major) = A[b] (MxK) * B[b] (KxN) (+ bias[N]); FP32 accumulate.
// 64x64 block tile, BK = 32, 4 warps each computing 32x32 with 2x2 WMMA 16x16x16 fragments.
// Arbitrary M, N, K (zero-padded tiles), so odd sizes like K = 241 work; 16-byte vector tile
// loads when K and N are multiples of 8. blockIdx.z = batch * splits + split: with splits > 1
// each block covers one K chunk and writes FP32 partials to `ws` (reduced by splitk_reduce).
namespace gemm_cfg {
constexpr int BM = 64, BN = 64, BK = 32, PAD = 8, THREADS = 128;
}

template <class TC>
__global__ void __launch_bounds__(gemm_cfg::THREADS)
    gemm_wmma_f16(const __half* __restrict__ A, const __half* __restrict__ B, TC* __restrict__ C,
                  const TC* __restrict__ bias, float* __restrict__ ws, int M, int N, int K,
                  long long sA, long long sB, long long sC, int splits, int kchunk, bool vec) {
#if __CUDA_ARCH__ >= 700
  using namespace nvcuda;
  using namespace gemm_cfg;
  __shared__ __align__(32) __half As[BM][BK + PAD];
  __shared__ __align__(32) __half Bs[BK][BN + PAD];
  __shared__ __align__(32) float Cs[BM][BN + 4];
  const int bz = blockIdx.z / splits, split = blockIdx.z % splits;
  A += bz * sA;
  B += bz * sB;
  const int kb = split * kchunk, ke = min(K, kb + kchunk);
  const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
  const int warp = threadIdx.x >> 5, wm = (warp >> 1) * 32, wn = (warp & 1) * 32;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j) wmma::fill_fragment(acc[i][j], 0.f);
  const __half zero = __float2half(0.f);
  for (int k0 = kb; k0 < ke; k0 += BK) {
    if (vec) {  // 8 halves per load; K, N multiples of 8 so vectors never straddle an edge
      for (int e = threadIdx.x; e < BM * BK / 8; e += THREADS) {
        int r = e / (BK / 8), c = (e % (BK / 8)) * 8, gm = m0 + r, gk = k0 + c;
        uint4 v = make_uint4(0, 0, 0, 0);
        if (gm < M && gk < ke) v = *reinterpret_cast<const uint4*>(A + (long long)gm * K + gk);
        *reinterpret_cast<uint4*>(&As[r][c]) = v;
      }
      for (int e = threadIdx.x; e < BK * BN / 8; e += THREADS) {
        int r = e / (BN / 8), c = (e % (BN / 8)) * 8, gk = k0 + r, gn = n0 + c;
        uint4 v = make_uint4(0, 0, 0, 0);
        if (gk < ke && gn < N) v = *reinterpret_cast<const uint4*>(B + (long long)gk * N + gn);
        *reinterpret_cast<uint4*>(&Bs[r][c]) = v;
      }
    } else {
      for (int e = threadIdx.x; e < BM * BK; e += THREADS) {
        int r = e / BK, c = e % BK, gm = m0 + r, gk = k0 + c;
        As[r][c] = (gm < M && gk < ke) ? A[(long long)gm * K + gk] : zero;
      }
      for (int e = threadIdx.x; e < BK * BN; e += THREADS) {
        int r = e / BN, c = e % BN, gk = k0 + r, gn = n0 + c;
        Bs[r][c] = (gk < ke && gn < N) ? B[(long long)gk * N + gn] : zero;
      }
    }
    __syncthreads();
    for (int kk = 0; kk < BK; kk += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa[2];
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fb[2];
      for (int i = 0; i < 2; ++i) wmma::load_matrix_sync(fa[i], &As[wm + 16 * i][kk], BK + PAD);
      for (int j = 0; j < 2; ++j) wmma::load_matrix_sync(fb[j], &Bs[kk][wn + 16 * j], BN + PAD);
      for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) wmma::mma_sync(acc[i][j], fa[i], fb[j], acc[i][j]);
    }
    __syncthreads();
  }
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j)
      wmma::store_matrix_sync(&Cs[wm + 16 * i][wn + 16 * j], acc[i][j], BN + 4,
                              wmma::mem_row_major);
  __syncthreads();
  const int batches = gridDim.z / splits;
  float* P = ws + ((long long)split * batches + bz) * M * N;
  C += bz * sC;
  for (int e = threadIdx.x; e < BM * BN; e += THREADS) {
    int r = e / BN, c = e % BN, gm = m0 + r, gn = n0 + c;
    if (gm < M && gn < N) {
      float v = Cs[r][c];
      if (splits > 1) {
        P[(long long)gm * N + gn] = v;
      } else {
        if (bias) v += to_f<TC>(bias[gn]);
        C[(long long)gm * N + gn] = from_f<TC>(v);
      }
    }
  }
#else
  __trap();  // WMMA needs sm_70+
#endif
}

// ---------------------------------------------------------------- GEMM, FP32 SIMT
// 64x64 tile, BK = 16, 256 threads, each thread computes a 4x4 strided micro-tile.
// Same batch/split-K convention as gemm_wmma_f16.
__global__ void __launch_bounds__(256)
    gemm_simt_f32(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
                  const float* __restrict__ bias, float* __restrict__ ws, int M, int N, int K,
                  long long sA, long long sB, long long sC, int splits, int kchunk) {
  constexpr int BM = 64, BN = 64, BK = 16;
  __shared__ float As[BK][BM + 1];
  __shared__ float Bs[BK][BN + 1];
  const int bz = blockIdx.z / splits, split = blockIdx.z % splits;
  A += bz * sA;
  B += bz * sB;
  const int kb = split * kchunk, ke = min(K, kb + kchunk);
  const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
  const int tx = threadIdx.x & 15, ty = threadIdx.x >> 4;
  float acc[4][4] = {};
  for (int k0 = kb; k0 < ke; k0 += BK) {
    for (int e = threadIdx.x; e < BM * BK; e += 256) {
      int r = e / BK, c = e % BK, gm = m0 + r, gk = k0 + c;
      As[c][r] = (gm < M && gk < ke) ? A[(long long)gm * K + gk] : 0.f;
    }
    for (int e = threadIdx.x; e < BK * BN; e += 256) {
      int r = e / BN, c = e % BN, gk = k0 + r, gn = n0 + c;
      Bs[r][c] = (gk < ke && gn < N) ? B[(long long)gk * N + gn] : 0.f;
    }
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < BK; ++kk) {
      float a[4], b[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) a[i] = As[kk][ty + 16 * i];
#pragma unroll
      for (int j = 0; j < 4; ++j) b[j] = Bs[kk][tx + 16 * j];
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
    }
    __syncthreads();
  }
  const int batches = gridDim.z / splits;
  float* P = ws + ((long long)split * batches + bz) * M * N;
  C += bz * sC;
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      int gm = m0 + ty + 16 * i, gn = n0 + tx + 16 * j;
      if (gm < M && gn < N) {
        if (splits > 1)
          P[(long long)gm * N + gn] = acc[i][j];
        else
          C[(long long)gm * N + gn] = acc[i][j] + (bias ? bias[gn] : 0.f);
      }
    }
}

// Launchers. batch >= 1; strides in elements (0 = operand shared across the batch). With a
// split-K workspace (gemm_workspace_init) small grids are split along K automatically; the
// output must then be dense (sC == M * N), which holds for every caller here.
template <class TC>
inline void gemm_f16(const __half* A, const __half* B, TC* C, const TC* bias, int batch, int M,
                     int N, int K, long long sA, long long sB, long long sC, cudaStream_t s) {
  using namespace gemm_cfg;
  const int gx = (N + BN - 1) / BN, gy = (M + BM - 1) / BM;
  const long long out = (long long)batch * M * N;
  const int splits = sC == (long long)M * N ? choose_splits((long long)gx * gy * batch, K, out, BK) : 1;
  const int chunk = splits > 1 ? split_chunk(K, splits, BK) : K;
  const bool vec = K % 8 == 0 && N % 8 == 0 && ((uintptr_t)A % 16 == 0) && ((uintptr_t)B % 16 == 0) &&
                   sA % 8 == 0 && sB % 8 == 0;
  gemm_wmma_f16<TC><<<dim3(gx, gy, batch * splits), THREADS, 0, s>>>(
      A, B, C, bias, ws_ptr(), M, N, K, sA, sB, sC, splits, chunk, vec);
  if (splits > 1)
    splitk_reduce<TC><<<(int)((out + 255) / 256), 256, 0, s>>>(ws_ptr(), C, bias, out, splits, N);
}
inline void gemm_f32(const float* A, const float* B, float* C, const float* bias, int batch,
                     int M, int N, int K, long long sA, long long sB, long long sC,
                     cudaStream_t s) {
  const int gx = (N + 63) / 64, gy = (M + 63) / 64;
  const long long out = (long long)batch * M * N;
  const int splits = sC == (long long)M * N ? choose_splits((long long)gx * gy * batch, K, out, 16) : 1;
  const int chunk = splits > 1 ? split_chunk(K, splits, 16) : K;
  gemm_simt_f32<<<dim3(gx, gy, batch * splits), 256, 0, s>>>(A, B, C, bias, ws_ptr(), M, N, K,
                                                            sA, sB, sC, splits, chunk);
  if (splits > 1)
    splitk_reduce<float><<<(int)((out + 255) / 256), 256, 0, s>>>(ws_ptr(), C, bias, out, splits, N);
}

// ---------------------------------------------------------------- generic elementwise
// out[i] = f(in0[i], in1[i], in2[i]) with NIN inputs (0 = fill). The arithmetic is a cheap
// fused multiply-add chain: batch-1 elementwise ops are memory/launch bound, so the bytes
// moved (set by the element types) are what matters for timing.
template <class TI, class TO, int NIN>
__global__ void elementwise_kernel(const TI* __restrict__ a, const TI* __restrict__ b,
                                   const TI* __restrict__ c, TO* __restrict__ out, long long n) {
  long long i0 = (blockIdx.x * (long long)blockDim.x + threadIdx.x) * 4;
#pragma unroll
  for (int u = 0; u < 4; ++u) {
    long long i = i0 + u;
    if (i < n) {
      float v = 1.f;
      if (NIN >= 1) v = to_f<TI>(a[i]);
      if (NIN >= 2) v = fmaf(v, 0.5f, to_f<TI>(b[i]));
      if (NIN >= 3) v = fmaf(v, 0.5f, to_f<TI>(c[i]));
      out[i] = from_f<TO>(v * 0.999f);
    }
  }
}
template <class TI, class TO>
inline void elementwise_t(const void* a, const void* b, const void* c, void* out, long long n,
                          int nin, cudaStream_t s) {
  if (n <= 0) return;
  int blocks = (int)((n + 511) / 512);
  auto A = (const TI*)a, B = (const TI*)b, C = (const TI*)c;
  auto O = (TO*)out;
  switch (nin) {
    case 0: elementwise_kernel<TI, TO, 0><<<blocks, 128, 0, s>>>(A, B, C, O, n); break;
    case 1: elementwise_kernel<TI, TO, 1><<<blocks, 128, 0, s>>>(A, B, C, O, n); break;
    case 2: elementwise_kernel<TI, TO, 2><<<blocks, 128, 0, s>>>(A, B, C, O, n); break;
    default: elementwise_kernel<TI, TO, 3><<<blocks, 128, 0, s>>>(A, B, C, O, n); break;
  }
}
template <class TI>
inline void elementwise_o(const void* a, const void* b, const void* c, void* out, long long n,
                          int nin, int out_bytes, cudaStream_t s) {
  switch (out_bytes) {
    case 1: elementwise_t<TI, uint8_t>(a, b, c, out, n, nin, s); break;
    case 2: elementwise_t<TI, __half>(a, b, c, out, n, nin, s); break;
    case 8: elementwise_t<TI, double>(a, b, c, out, n, nin, s); break;
    default: elementwise_t<TI, float>(a, b, c, out, n, nin, s); break;
  }
}
// Runtime dispatch on element sizes (1, 2, 4 or 8 bytes).
inline void elementwise(const void* a, const void* b, const void* c, void* out, long long n,
                        int nin, int in_bytes, int out_bytes, cudaStream_t s) {
  switch (in_bytes) {
    case 1: elementwise_o<uint8_t>(a, b, c, out, n, nin, out_bytes, s); break;
    case 2: elementwise_o<__half>(a, b, c, out, n, nin, out_bytes, s); break;
    case 8: elementwise_o<double>(a, b, c, out, n, nin, out_bytes, s); break;
    default: elementwise_o<float>(a, b, c, out, n, nin, out_bytes, s); break;
  }
}

// ---------------------------------------------------------------- activations (FP16)
enum class Act { SiLU, GeLU, SiLUMul };
template <Act ACT>
__global__ void act_kernel(const __half* __restrict__ x, const __half* __restrict__ g,
                           __half* __restrict__ y, long long n) {
  long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (i >= n) return;
  float v = __half2float(x[i]), r;
  if (ACT == Act::GeLU) {
    r = 0.5f * v * (1.f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
  } else {
    r = v / (1.f + __expf(-v));
    if (ACT == Act::SiLUMul) r *= __half2float(g[i]);
  }
  y[i] = __float2half(r);
}
template <Act ACT>
inline void activation(const __half* x, const __half* g, __half* y, long long n, cudaStream_t s) {
  act_kernel<ACT><<<(int)((n + 255) / 256), 256, 0, s>>>(x, g, y, n);
}

// ---------------------------------------------------------------- row ops (one block per row)
// Row reduction (mean of the row). out[r] in FP32.
template <class T>
__global__ void row_mean_kernel(const T* __restrict__ x, float* __restrict__ out, int cols) {
  const T* row = x + (long long)blockIdx.x * cols;
  float s = 0.f;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) s += to_f<T>(row[c]);
  s = block_reduce<false>(s);
  if (threadIdx.x == 0) out[blockIdx.x] = s / cols;
}
inline int row_threads(int cols) { return cols >= 1024 ? 256 : cols >= 256 ? 128 : 64; }
inline void row_mean(const void* x, float* out, int rows, int cols, int elem_bytes,
                     cudaStream_t s) {
  if (rows <= 0) return;
  int t = row_threads(cols);
  switch (elem_bytes) {
    case 2: row_mean_kernel<__half><<<rows, t, 0, s>>>((const __half*)x, out, cols); break;
    case 8: row_mean_kernel<double><<<rows, t, 0, s>>>((const double*)x, out, cols); break;
    case 1: row_mean_kernel<uint8_t><<<rows, t, 0, s>>>((const uint8_t*)x, out, cols); break;
    default: row_mean_kernel<float><<<rows, t, 0, s>>>((const float*)x, out, cols); break;
  }
}

// Softmax along rows, in place-safe (x and y may alias).
template <class T>
__global__ void softmax_kernel(const T* __restrict__ x, T* __restrict__ y, int cols) {
  const long long off = (long long)blockIdx.x * cols;
  float m = -INFINITY;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) m = fmaxf(m, to_f<T>(x[off + c]));
  m = block_reduce<true>(m);
  float s = 0.f;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) s += __expf(to_f<T>(x[off + c]) - m);
  s = block_reduce<false>(s);
  const float inv = 1.f / s;
  for (int c = threadIdx.x; c < cols; c += blockDim.x)
    y[off + c] = from_f<T>(__expf(to_f<T>(x[off + c]) - m) * inv);
}
inline void softmax(const void* x, void* y, int rows, int cols, int elem_bytes, cudaStream_t s) {
  if (rows <= 0) return;
  int t = row_threads(cols);
  if (elem_bytes == 2)
    softmax_kernel<__half><<<rows, t, 0, s>>>((const __half*)x, (__half*)y, cols);
  else
    softmax_kernel<float><<<rows, t, 0, s>>>((const float*)x, (float*)y, cols);
}

// LayerNorm (RMS = false) or RMSNorm (RMS = true) with weight (and bias for LayerNorm).
template <class T, bool RMS>
__global__ void norm_kernel(const T* __restrict__ x, const T* __restrict__ w,
                            const T* __restrict__ b, T* __restrict__ y, int cols, float eps) {
  const long long off = (long long)blockIdx.x * cols;
  float s = 0.f, ss = 0.f;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    float v = to_f<T>(x[off + c]);
    s += v;
    ss += v * v;
  }
  float mean = RMS ? 0.f : block_reduce<false>(s) / cols;
  float msq = block_reduce<false>(ss) / cols;
  float var = RMS ? msq : fmaxf(msq - mean * mean, 0.f);
  float rstd = rsqrtf(var + eps);
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    float v = (to_f<T>(x[off + c]) - mean) * rstd * to_f<T>(w[c]);
    if (!RMS && b) v += to_f<T>(b[c]);
    y[off + c] = from_f<T>(v);
  }
}
template <bool RMS>
inline void norm(const void* x, const void* w, const void* b, void* y, int rows, int cols,
                 int elem_bytes, cudaStream_t s, float eps = 1e-6f) {
  if (rows <= 0) return;
  int t = row_threads(cols);
  if (elem_bytes == 2)
    norm_kernel<__half, RMS><<<rows, t, 0, s>>>((const __half*)x, (const __half*)w,
                                                (const __half*)b, (__half*)y, cols, eps);
  else
    norm_kernel<float, RMS><<<rows, t, 0, s>>>((const float*)x, (const float*)w,
                                               (const float*)b, (float*)y, cols, eps);
}

// ---------------------------------------------------------------- RoPE (rotate-half form)
// x: [tokens, heads, D] FP16; cos/sin: [tokens, D/2] FP32. In place.
__global__ void rope_kernel(__half* __restrict__ x, const float* __restrict__ cs,
                            const float* __restrict__ sn, int tokens, int heads, int D) {
  const int half_d = D / 2;
  long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  long long total = (long long)tokens * heads * half_d;
  if (i >= total) return;
  int d = i % half_d;
  long long th = i / half_d;  // token * heads + head
  int t = (int)(th / heads);
  __half* row = x + th * D;
  float x1 = __half2float(row[d]), x2 = __half2float(row[d + half_d]);
  float c = cs[t * half_d + d], s = sn[t * half_d + d];
  row[d] = __float2half(x1 * c - x2 * s);
  row[d + half_d] = __float2half(x2 * c + x1 * s);
}
inline void rope(__half* x, const float* cs, const float* sn, int tokens, int heads, int D,
                 cudaStream_t s) {
  long long total = (long long)tokens * heads * (D / 2);
  rope_kernel<<<(int)((total + 255) / 256), 256, 0, s>>>(x, cs, sn, tokens, heads, D);
}

// ---------------------------------------------------------------- attention (FP16)
// O[b, i, :] = softmax(Q[b, i, :] K[b]^T / sqrt(D)) V[b]; Q: [B, M, D], K, V: [B, N, D].
// Tiled, FlashAttention-style (online softmax, K/V tiles in shared memory, nothing of size N
// stored). On the 2080 Ti it is ~9x slower than PyTorch's memory-efficient attention for the
// vision tower (a WMMA version was slower still: its shared memory left one 2-warp block per
// SM). Block = 4 warps x 4 query rows = 16 rows; key tiles of 32 keys, one key per lane
// for the scores, D/32 output dims per lane. Requires D % 32 == 0 and D <= 128.
constexpr int ATTN_WARPS = 4, ATTN_ROWS = 4, ATTN_TILE = 32, ATTN_MAXD = 128;
// D is a template parameter so the per-lane accumulators stay in registers.
template <int D>
__global__ void __launch_bounds__(ATTN_WARPS * 32)
    attention_kernel(const __half* __restrict__ Q, const __half* __restrict__ K,
                     const __half* __restrict__ V, __half* __restrict__ O, int M, int N,
                     float scale) {
  __shared__ float qs[ATTN_WARPS * ATTN_ROWS][D];
  __shared__ __half ks[ATTN_TILE][D + 2];  // +2 halves: odd word stride, no bank conflicts
  __shared__ __half vs[ATTN_TILE][D];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, bh = blockIdx.y;
  const int row0 = blockIdx.x * ATTN_WARPS * ATTN_ROWS;
  constexpr int dpl = D / 32;  // output dims per lane
  const __half* Qb = Q + (long long)bh * M * D;
  const __half* Kb = K + (long long)bh * N * D;
  const __half* Vb = V + (long long)bh * N * D;
  for (int e = threadIdx.x; e < ATTN_WARPS * ATTN_ROWS * D; e += blockDim.x) {
    int r = e / D, d = e % D, gi = row0 + r;
    qs[r][d] = gi < M ? __half2float(Qb[(long long)gi * D + d]) * scale : 0.f;
  }
  float m[ATTN_ROWS], l[ATTN_ROWS], o[ATTN_ROWS][dpl];
#pragma unroll
  for (int r = 0; r < ATTN_ROWS; ++r) {
    m[r] = -INFINITY;
    l[r] = 0.f;
#pragma unroll
    for (int c = 0; c < dpl; ++c) o[r][c] = 0.f;
  }
  for (int t0 = 0; t0 < N; t0 += ATTN_TILE) {
    __syncthreads();
    for (int e = threadIdx.x; e < ATTN_TILE * D; e += blockDim.x) {
      int j = e / D, d = e % D, gj = t0 + j;
      __half z = __float2half(0.f);
      ks[j][d] = gj < N ? Kb[(long long)gj * D + d] : z;
      vs[j][d] = gj < N ? Vb[(long long)gj * D + d] : z;
    }
    __syncthreads();
    const bool valid = t0 + lane < N;
#pragma unroll
    for (int r = 0; r < ATTN_ROWS; ++r) {
      const float* q = qs[warp * ATTN_ROWS + r];
      float s = 0.f;
#pragma unroll 16
      for (int d = 0; d < D; ++d) s = fmaf(q[d], __half2float(ks[lane][d]), s);
      s = valid ? s : -INFINITY;
      float mn = fmaxf(m[r], warp_max(s));
      float p = valid ? __expf(s - mn) : 0.f;
      float corr = __expf(m[r] - mn);  // exp(-inf) = 0 on the first tile
      l[r] = l[r] * corr + warp_sum(p);
      m[r] = mn;
#pragma unroll
      for (int c = 0; c < dpl; ++c) o[r][c] *= corr;
#pragma unroll 8
      for (int j = 0; j < ATTN_TILE; ++j) {
        float pj = __shfl_sync(0xffffffffu, p, j);
#pragma unroll
        for (int c = 0; c < dpl; ++c) o[r][c] = fmaf(pj, __half2float(vs[j][lane + 32 * c]), o[r][c]);
      }
    }
  }
#pragma unroll
  for (int r = 0; r < ATTN_ROWS; ++r) {
    int gi = row0 + warp * ATTN_ROWS + r;
    if (gi < M)
#pragma unroll
      for (int c = 0; c < dpl; ++c)
        O[((long long)bh * M + gi) * D + lane + 32 * c] = __float2half(o[r][c] / l[r]);
  }
}
inline void attention(const __half* Q, const __half* K, const __half* V, __half* O, int BH, int M,
                      int N, int D, cudaStream_t s) {
  if (D % 32 != 0 || D > ATTN_MAXD || N <= 0) return;  // unsupported head dim: skip
  const float sc = rsqrtf((float)D);
  const int rows = ATTN_WARPS * ATTN_ROWS;
  const dim3 grid((M + rows - 1) / rows, BH), block(ATTN_WARPS * 32);
  switch (D) {
    case 32: attention_kernel<32><<<grid, block, 0, s>>>(Q, K, V, O, M, N, sc); break;
    case 64: attention_kernel<64><<<grid, block, 0, s>>>(Q, K, V, O, M, N, sc); break;
    case 96: attention_kernel<96><<<grid, block, 0, s>>>(Q, K, V, O, M, N, sc); break;
    default: attention_kernel<128><<<grid, block, 0, s>>>(Q, K, V, O, M, N, sc); break;
  }
}

// ---------------------------------------------------------------- inclusive scan (one block)
template <class T>
__global__ void scan_kernel(const T* __restrict__ x, T* __restrict__ y, int n) {
  __shared__ float part[1024];
  const int t = threadIdx.x, per = (n + blockDim.x - 1) / blockDim.x;
  const int b = t * per, e = min(n, b + per);
  float s = 0.f;
  for (int i = b; i < e; ++i) s += to_f<T>(x[i]);
  part[t] = s;
  __syncthreads();
  for (int o = 1; o < blockDim.x; o <<= 1) {  // Hillis-Steele over the per-thread sums
    float v = t >= o ? part[t - o] : 0.f;
    __syncthreads();
    part[t] += v;
    __syncthreads();
  }
  float run = t ? part[t - 1] : 0.f;
  for (int i = b; i < e; ++i) {
    run += to_f<T>(x[i]);
    y[i] = from_f<T>(run);
  }
}
inline void scan(const void* x, void* y, int n, int elem_bytes, cudaStream_t s) {
  int t = n >= 1024 ? 1024 : ((n + 31) / 32) * 32;
  if (t < 32) t = 32;
  if (elem_bytes == 2)
    scan_kernel<__half><<<1, t, 0, s>>>((const __half*)x, (__half*)y, n);
  else
    scan_kernel<float><<<1, t, 0, s>>>((const float*)x, (float*)y, n);
}

}  // namespace vk
