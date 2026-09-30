// Prototype (2026-10-01): the atomic additions of the spreading of PME,
// order 4, on the grid of MDIR for a system of the suite (Cellulose:
// 270 x 126 x 126). The first points and the weights of the particles are
// computed on the host, as a kernel of weights would give them, so that
// only the additions are timed. Variants:
//
//   lanes4    4 threads a particle along z, 16 atomics each (MDIR before)
//   warp      a warp a particle: lanes (x: 2, y: 4, z: 4), 2 atomics each,
//             the grid in the order of the FFT (MDIR now)
//   brick     a warp a particle: lanes (x: 4, y: 4, z: 2), the grid in
//             bricks of 4 x 4 in x-y with z inside them, so that the 16
//             points of a particle in x-y at one z are 16 consecutive
//             values of at most 4 bricks; then a copy into the order of
//             the FFT
//
// each with f32 atomics and with i32 atomics in fixed point (which the
// deterministic mode would also take). The grids are compared with one
// summed on the host in double precision.
//
//   spread <system.bin> [k1 k2 k3]
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

static const float SCALE = 16777216.0f;  // 2^24: fixed point of the i32 variants

__device__ __forceinline__ void addf(float *p, float v) {
  asm volatile("red.relaxed.gpu.global.add.f32 [%0], %1;" :: "l"(p), "f"(v) : "memory");
}
__device__ __forceinline__ void addi(int *p, int v) {
  asm volatile("red.relaxed.gpu.global.add.s32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}

// Particle data: start (3 ints) and weights (12 floats, x already times q).
struct P { int s[3]; float w[12]; };

template <bool FIX>
__global__ void lanes4(int n, const P *ps, int k1, int k2, int k3, float *gf, int *gi) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  int i = t >> 2, c = t & 3;
  if (i >= n) return;
  P p = ps[i];
  int z = p.s[2] + c; z -= (z >= k3) * k3;
  for (int a = 0; a < 4; ++a) {
    int x = p.s[0] + a; x -= (x >= k1) * k1;
    for (int b = 0; b < 4; ++b) {
      int y = p.s[1] + b; y -= (y >= k2) * k2;
      float v = p.w[a] * p.w[4 + b] * p.w[8 + c];
      size_t at = ((size_t)x * k2 + y) * k3 + z;
      if (FIX) addi(gi + at, __float2int_rn(v * SCALE)); else addf(gf + at, v);
    }
  }
}

template <bool FIX>
__global__ void warp(int n, const P *ps, int k1, int k2, int k3, float *gf, int *gi) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  int i = t >> 5, l = t & 31;
  if (i >= n) return;
  const P &p = ps[i];
  int c = l & 3, b = (l >> 2) & 3, a0 = l >> 4;
  int z = p.s[2] + c; z -= (z >= k3) * k3;
  int y = p.s[1] + b; y -= (y >= k2) * k2;
  float wyz = p.w[4 + b] * p.w[8 + c];
  for (int h = 0; h < 2; ++h) {
    int a = a0 + 2 * h;
    int x = p.s[0] + a; x -= (x >= k1) * k1;
    float v = p.w[a] * wyz;
    size_t at = ((size_t)x * k2 + y) * k3 + z;
    if (FIX) addi(gi + at, __float2int_rn(v * SCALE)); else addf(gf + at, v);
  }
}

// Bricks: point (x, y, z) at (((x / 4) * B2 + y / 4) * k3 + z) * 16 + (y % 4) * 4 + x % 4,
// with B2 = ceil(k2 / 4).
template <bool FIX>
__global__ void brick(int n, const P *ps, int k1, int k2, int k3, int b2, float *gf, int *gi) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  int i = t >> 5, l = t & 31;
  if (i >= n) return;
  const P &p = ps[i];
  int a = l & 3, b = (l >> 2) & 3, c0 = l >> 4;
  int x = p.s[0] + a; x -= (x >= k1) * k1;
  int y = p.s[1] + b; y -= (y >= k2) * k2;
  float wxy = p.w[a] * p.w[4 + b];
  size_t xy = ((size_t)(x >> 2) * b2 + (y >> 2)) * k3 * 16 + (y & 3) * 4 + (x & 3);
  for (int h = 0; h < 2; ++h) {
    int c = c0 + 2 * h;
    int z = p.s[2] + c; z -= (z >= k3) * k3;
    float v = wxy * p.w[8 + c];
    size_t at = xy + (size_t)z * 16;
    if (FIX) addi(gi + at, __float2int_rn(v * SCALE)); else addf(gf + at, v);
  }
}

// From bricks to the order of the FFT, a thread a point.
template <bool FIX>
__global__ void unbrick(int k1, int k2, int k3, int b2, const float *bf, const int *bi, float *out) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t total = (size_t)k1 * k2 * k3;
  if (t >= total) return;
  int z = t % k3; size_t r = t / k3; int y = r % k2; int x = r / k2;
  size_t at = ((size_t)(x >> 2) * b2 + (y >> 2)) * k3 * 16 + (size_t)z * 16 + (y & 3) * 4 + (x & 3);
  out[t] = FIX ? bi[at] * (1.0f / SCALE) : bf[at];
}
template <bool dummy>
__global__ void fromFixed(size_t total, const int *gi, float *out) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < total) out[t] = gi[t] * (1.0f / SCALE);
}

// The weights of a particle, a thread each: its first points and the
// B-splines of order 4 along x, y, z (x times the charge), from positions
// in f64, the fraction along the cell in f64 and the rest in f32, as the
// template of MDIR computes them. With SLOPES, their derivatives as well
// (for the gathering), into a second array.
template <bool SLOPES>
__global__ void weights(int n, const double *x, const float *q, double3 il, int3 K, P *ps, float *slopes) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  P p;
  double ilv[3] = {il.x, il.y, il.z}; int Kv[3] = {K.x, K.y, K.z};
  for (int k = 0; k < 3; ++k) {
    double s = x[3 * i + k] * ilv[k];
    double frac = s - floor(s);
    float u = (float)frac * Kv[k];
    float fu = floorf(u), w = u - fu;
    int b = (int)fu;
    int st = b - 3; st += (st < 0) * Kv[k];
    p.s[k] = st;
    float w2 = w * w, w3 = w2 * w, omw = 1.f - w;
    float m0 = omw * omw * omw * (1.f / 6.f);
    float m1 = (4.f - 6.f * w2 + 3.f * w3) * (1.f / 6.f);
    float m2 = (1.f + 3.f * w + 3.f * w2 - 3.f * w3) * (1.f / 6.f);
    float m3 = w3 * (1.f / 6.f);
    float sc = k == 0 ? q[i] : 1.f;
    p.w[4 * k] = m0 * sc; p.w[4 * k + 1] = m1 * sc; p.w[4 * k + 2] = m2 * sc; p.w[4 * k + 3] = m3 * sc;
    if (SLOPES) {
      float *d = slopes + 12 * (size_t)i + 4 * k;
      d[0] = -0.5f * omw * omw; d[1] = 1.5f * w2 - 2.f * w; d[2] = 0.5f + w - 1.5f * w2; d[3] = 0.5f * w2;
    }
  }
  ps[i] = p;
}

// The same, into arrays by component (15 values a particle, each array
// written by consecutive threads at once), and the spreading into bricks
// that reads them.
__global__ void weightsSoA(int n, const double *x, const float *q, double3 il, int3 K, int *starts, float *w) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  double ilv[3] = {il.x, il.y, il.z}; int Kv[3] = {K.x, K.y, K.z};
#pragma unroll
  for (int k = 0; k < 3; ++k) {
    double s = x[3 * i + k] * ilv[k];
    double frac = s - floor(s);
    float u = (float)frac * Kv[k];
    float fu = floorf(u), v = u - fu;
    int st = (int)fu - 3; st += (st < 0) * Kv[k];
    starts[(size_t)k * n + i] = st;
    float v2 = v * v, v3 = v2 * v, omv = 1.f - v;
    float sc = k == 0 ? q[i] : 1.f;
    w[(size_t)(4 * k) * n + i] = omv * omv * omv * (1.f / 6.f) * sc;
    w[(size_t)(4 * k + 1) * n + i] = (4.f - 6.f * v2 + 3.f * v3) * (1.f / 6.f) * sc;
    w[(size_t)(4 * k + 2) * n + i] = (1.f + 3.f * v + 3.f * v2 - 3.f * v3) * (1.f / 6.f) * sc;
    w[(size_t)(4 * k + 3) * n + i] = v3 * (1.f / 6.f) * sc;
  }
}
__global__ void brickSoA(int n, const int *starts, const float *w, int k1, int k2, int k3, int b2, float *gf) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  int i = t >> 5, l = t & 31;
  if (i >= n) return;
  int a = l & 3, b = (l >> 2) & 3, c0 = l >> 4;
  int x = starts[i] + a; x -= (x >= k1) * k1;
  int y = starts[n + i] + b; y -= (y >= k2) * k2;
  int z0 = starts[2 * (size_t)n + i];
  float wxy = w[(size_t)a * n + i] * w[(size_t)(4 + b) * n + i];
  size_t xy = ((size_t)(x >> 2) * b2 + (y >> 2)) * k3 * 16 + (y & 3) * 4 + (x & 3);
  for (int h = 0; h < 2; ++h) {
    int c = c0 + 2 * h;
    int z = z0 + c; z -= (z >= k3) * k3;
    addf(gf + xy + (size_t)z * 16, wxy * w[(size_t)(8 + c) * n + i]);
  }
}

int main(int argc, char **argv) {
  FILE *fp = fopen(argv[1], "rb");
  int n, ntypes; double box[3];
  fread(&n, 4, 1, fp); fread(&ntypes, 4, 1, fp); fread(box, 8, 3, fp);
  std::vector<double> x(3 * n); std::vector<float> q(n);
  fread(x.data(), 8, 3 * n, fp); fread(q.data(), 4, n, fp); fclose(fp);
  for (float &c : q) c /= std::sqrt(332.0522f);
  int K[3] = {270, 126, 126};
  if (argc > 4) for (int k = 0; k < 3; ++k) K[k] = atoi(argv[2 + k]);
  for (int i = 0; i < 3 * n; ++i) { int k = i % 3; x[i] -= box[k] * std::floor(x[i] / box[k]); }

  // The spatial order of MDIR: cells of 5 A, x first.
  std::vector<int> order(n);
  {
    int nc[3]; for (int k = 0; k < 3; ++k) nc[k] = std::max(1, (int)(box[k] / 5.0));
    std::vector<std::pair<long, int>> key(n);
    for (int i = 0; i < n; ++i) { int c[3]; for (int k = 0; k < 3; ++k) c[k] = std::min(nc[k] - 1, (int)(x[3*i+k] / box[k] * nc[k]));
      key[i] = {((long)c[2] * nc[1] + c[1]) * nc[0] + c[0], i}; }
    std::sort(key.begin(), key.end());
    for (int i = 0; i < n; ++i) order[i] = key[i].second;
  }
  std::vector<P> ps(n);
  size_t total = (size_t)K[0] * K[1] * K[2];
  std::vector<double> ref(total, 0.0);
  for (int pi = 0; pi < n; ++pi) {
    int i = order[pi]; P &p = ps[pi];
    for (int k = 0; k < 3; ++k) {
      double u = x[3*i+k] / box[k] * K[k];
      double fu = std::floor(u), w = u - fu;
      int b = (int)fu;
      int s = ((b - 3) % K[k] + K[k]) % K[k];
      p.s[k] = s;
      // B-spline of order 4 at w, for the points s .. s + 3.
      double m[4];
      m[3] = w * w * w / 6.0;
      m[2] = (1.0 + 3.0 * w + 3.0 * w * w - 3.0 * w * w * w) / 6.0;
      m[1] = (4.0 - 6.0 * w * w + 3.0 * w * w * w) / 6.0;
      m[0] = (1.0 - w) * (1.0 - w) * (1.0 - w) / 6.0;
      for (int e = 0; e < 4; ++e) p.w[4 * k + e] = (float)(m[e] * (k == 0 ? q[i] : 1.0));
    }
    for (int a = 0; a < 4; ++a) for (int b = 0; b < 4; ++b) for (int c = 0; c < 4; ++c) {
      int gx = (p.s[0] + a) % K[0], gy = (p.s[1] + b) % K[1], gz = (p.s[2] + c) % K[2];
      ref[((size_t)gx * K[1] + gy) * K[2] + gz] += (double)p.w[a] * p.w[4 + b] * p.w[8 + c];
    }
  }
  int b2 = (K[1] + 3) / 4, b1 = (K[0] + 3) / 4;
  size_t bricks = (size_t)b1 * b2 * K[2] * 16;
  P *dps; float *gf, *out, *bf; int *gi, *bi;
  CK(cudaMalloc(&dps, sizeof(P) * n)); cudaMemcpy(dps, ps.data(), sizeof(P) * n, cudaMemcpyHostToDevice);
  CK(cudaMalloc(&gf, 4 * total)); CK(cudaMalloc(&gi, 4 * total)); CK(cudaMalloc(&out, 4 * total));
  CK(cudaMalloc(&bf, 4 * bricks)); CK(cudaMalloc(&bi, 4 * bricks));
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  const int reps = 200;
  auto time = [&](auto launch) {
    launch(); CK(cudaDeviceSynchronize());
    cudaEventRecord(e0); for (int r = 0; r < reps; ++r) launch(); cudaEventRecord(e1); CK(cudaEventSynchronize(e1));
    float ms; cudaEventElapsedTime(&ms, e0, e1); return 1000.0 * ms / reps;
  };
  std::vector<float> host(total);
  auto check = [&](const char *name, double us, const float *d) {
    cudaMemcpy(host.data(), d, 4 * total, cudaMemcpyDeviceToHost);
    double num = 0, den = 0;
    for (size_t t = 0; t < total; ++t) { double e = host[t] - ref[t]; num += e * e; den += ref[t] * ref[t]; }
    printf("%-12s %8.1f us   relative difference %.2e\n", name, us, std::sqrt(num / den));
    printf("csv,spread,%s,,,%.2f\n", name, us);
  };
  int T = 128;
  auto blocks = [&](size_t threads) { return (unsigned)((threads + T - 1) / T); };
  unsigned gridBlocks = blocks(total);
  double t;
  t = time([&] { cudaMemsetAsync(gf, 0, 4 * total); lanes4<false><<<blocks(4 * (size_t)n), T>>>(n, dps, K[0], K[1], K[2], gf, gi); });
  check("lanes4 f32", t, gf);
  t = time([&] { cudaMemsetAsync(gi, 0, 4 * total); lanes4<true><<<blocks(4 * (size_t)n), T>>>(n, dps, K[0], K[1], K[2], gf, gi);
                 fromFixed<true><<<gridBlocks, T>>>(total, gi, out); });
  check("lanes4 i32", t, out);
  t = time([&] { cudaMemsetAsync(gf, 0, 4 * total); warp<false><<<blocks(32 * (size_t)n), T>>>(n, dps, K[0], K[1], K[2], gf, gi); });
  check("warp f32", t, gf);
  t = time([&] { cudaMemsetAsync(gi, 0, 4 * total); warp<true><<<blocks(32 * (size_t)n), T>>>(n, dps, K[0], K[1], K[2], gf, gi);
                 fromFixed<true><<<gridBlocks, T>>>(total, gi, out); });
  check("warp i32", t, out);
  t = time([&] { cudaMemsetAsync(bf, 0, 4 * bricks); brick<false><<<blocks(32 * (size_t)n), T>>>(n, dps, K[0], K[1], K[2], b2, bf, bi);
                 unbrick<false><<<gridBlocks, T>>>(K[0], K[1], K[2], b2, bf, bi, out); });
  check("brick f32", t, out);
  t = time([&] { cudaMemsetAsync(bi, 0, 4 * bricks); brick<true><<<blocks(32 * (size_t)n), T>>>(n, dps, K[0], K[1], K[2], b2, bf, bi);
                 unbrick<true><<<gridBlocks, T>>>(K[0], K[1], K[2], b2, bf, bi, out); });
  check("brick i32", t, out);
  {
    std::vector<double> xs(3 * n); std::vector<float> qs(n);
    for (int pi = 0; pi < n; ++pi) { int i = order[pi]; for (int k = 0; k < 3; ++k) xs[3*pi+k] = x[3*i+k]; qs[pi] = q[i]; }
    double *dx; float *dq, *dsl; P *dps2;
    CK(cudaMalloc(&dx, 24 * n)); CK(cudaMalloc(&dq, 4 * n)); CK(cudaMalloc(&dsl, 48 * (size_t)n)); CK(cudaMalloc(&dps2, sizeof(P) * n));
    cudaMemcpy(dx, xs.data(), 24 * n, cudaMemcpyHostToDevice); cudaMemcpy(dq, qs.data(), 4 * n, cudaMemcpyHostToDevice);
    double3 il = make_double3(1 / box[0], 1 / box[1], 1 / box[2]); int3 Kd = make_int3(K[0], K[1], K[2]);
    double tw = time([&] { weights<false><<<blocks(n), T>>>(n, dx, dq, il, Kd, dps2, dsl); });
    double ts = time([&] { weights<true><<<blocks(n), T>>>(n, dx, dq, il, Kd, dps2, dsl); });
    printf("weights %.1f us, with slopes %.1f us\n", tw, ts);
    int *dst; float *dw; CK(cudaMalloc(&dst, 12 * (size_t)n)); CK(cudaMalloc(&dw, 48 * (size_t)n));
    double tsoa = time([&] { weightsSoA<<<blocks(n), T>>>(n, dx, dq, il, Kd, dst, dw); });
    double tall = time([&] { cudaMemsetAsync(bf, 0, 4 * bricks); weightsSoA<<<blocks(n), T>>>(n, dx, dq, il, Kd, dst, dw);
      brickSoA<<<blocks(32 * (size_t)n), T>>>(n, dst, dw, K[0], K[1], K[2], b2, bf);
      unbrick<false><<<gridBlocks, T>>>(K[0], K[1], K[2], b2, bf, bi, out); });
    printf("weights by component %.1f us; clearing, weights, bricks, and the copy %.1f us\n", tsoa, tall);
    check("all, from positions", tall, out);
    std::vector<P> back(n); cudaMemcpy(back.data(), dps2, sizeof(P) * n, cudaMemcpyDeviceToHost);
    double worst = 0; int starts = 0;
    for (int i = 0; i < n; ++i) { for (int k = 0; k < 3; ++k) starts += back[i].s[k] != ps[i].s[k];
      for (int e = 0; e < 12; ++e) worst = std::max(worst, (double)std::fabs(back[i].w[e] - ps[i].w[e])); }
    printf("weights against the host: %d starts differ, largest difference of a weight %.2e\n", starts, worst);
    printf("csv,spread,weights,,,%.2f\ncsv,spread,weights with slopes,,,%.2f\n", tw, ts);
  }
  // The parts alone.
  t = time([&] { brick<true><<<blocks(32 * (size_t)n), T>>>(n, dps, K[0], K[1], K[2], b2, bf, bi); });
  printf("brick i32 additions alone %.1f us\n", t);
  t = time([&] { unbrick<true><<<gridBlocks, T>>>(K[0], K[1], K[2], b2, bf, bi, out); });
  printf("unbrick alone %.1f us\n", t);
  t = time([&] { cudaMemsetAsync(gf, 0, 4 * total); });
  printf("clearing the grid %.1f us\n", t);
  return 0;
}
