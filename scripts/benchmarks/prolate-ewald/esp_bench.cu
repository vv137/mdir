// Prototype of the reciprocal sum of PME with a tensor-product window of P
// points (B-splines of order 4 or the PSWF of ESP, P = 4..8), for the
// measurement of issue #334; a stand-alone program, not part of MDIR.  The kernels follow the
// structure of MDIR's order-4 kernels (D102, D104, D108, D109): places as
// first point and fraction, f64 fraction along the cell, spreading a warp a
// particle into bricks of 4 x 4 points with z inside by atomics, the
// transposed copy into the grid that leaves the bricks zero, cuFFT, the
// product with separable tables, gathering with weights recomputed from
// the fractions.  The weights are polynomials in t = 2s - 1 by Horner.
//
// The tiled layouts (tiled4, tiled8) bin the particles by their first point
// each step and add into a tile of (B + P - 1)^3 points in block memory,
// the binned, blocked spreading of cuFINUFFT (Shih et al., IPDPSW 2021,
// doi:10.1109/IPDPSW52791.2021.00105).
// The ESP window and splitting follow Liang et al., Nat. Commun. 17, 6681
// (2026), eqs. 3 to 16; written for this measurement, not copied.
//
// esp_bench CASE PREC(f|d) REPS LAYOUT(brick|direct|tiled4|tiled8) GATHER_LANES ORDER(input|sorted) FORCES_OUT
#include <cuda_runtime.h>
#include <cufft.h>
#include <cub/cub.cuh>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <string>
#include <cmath>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1);} } while (0)
#define CF(x) do { cufftResult e = (x); if (e != CUFFT_SUCCESS) { fprintf(stderr, "%s:%d cufft %d\n", __FILE__, __LINE__, (int)e); exit(1);} } while (0)

struct Case {
  int N, K1, K2, K3, P, D, kind, ncheb;
  double L[3], rc, off, xscale;
  std::vector<double> x, q, w, d, B1, B2, B3, cheb, chebd;
};

static Case readCase(const char *path) {
  Case c;
  FILE *f = fopen(path, "rb");
  if (!f) { perror(path); exit(1); }
  int h[8];
  fread(h, 4, 8, f);
  c.N = h[0]; c.K1 = h[1]; c.K2 = h[2]; c.K3 = h[3]; c.P = h[4]; c.D = h[5]; c.kind = h[6]; c.ncheb = h[7];
  double g[6];
  fread(g, 8, 6, f);
  for (int a = 0; a < 3; ++a) c.L[a] = g[a];
  c.rc = g[3]; c.off = g[4]; c.xscale = g[5];
  auto rd = [&](std::vector<double> &v, size_t n) { v.resize(n); fread(v.data(), 8, n, f); };
  rd(c.x, 3 * (size_t)c.N); rd(c.q, c.N); rd(c.w, c.P * c.D); rd(c.d, c.P * c.D);
  rd(c.B1, c.K1); rd(c.B2, c.K2); rd(c.B3, c.K3 / 2 + 1); rd(c.cheb, c.ncheb); rd(c.chebd, c.ncheb);
  fclose(f);
  return c;
}

// Coefficients in the type of the grid, so that no kernel converts from f64
// (a conversion runs at the f64 rate of a consumer device).
__constant__ float fW[8 * 24], fDW[8 * 24], fCheb[64], fChebD[64];
__constant__ double dW[8 * 24], dDW[8 * 24], dCheb[64], dChebD[64];
template <typename R> struct Co;
template <> struct Co<float> {
  __device__ static const float *W() { return fW; } __device__ static const float *DW() { return fDW; }
  __device__ static const float *Ch() { return fCheb; } __device__ static const float *ChD() { return fChebD; }
  static void up(const std::vector<double> &w, const std::vector<double> &d, const std::vector<double> &ch, const std::vector<double> &chd) {
    std::vector<float> a(w.begin(), w.end()), b(d.begin(), d.end()), c(ch.begin(), ch.end()), g(chd.begin(), chd.end());
    CK(cudaMemcpyToSymbol(fW, a.data(), 4 * a.size())); CK(cudaMemcpyToSymbol(fDW, b.data(), 4 * b.size()));
    CK(cudaMemcpyToSymbol(fCheb, c.data(), 4 * c.size())); CK(cudaMemcpyToSymbol(fChebD, g.data(), 4 * g.size()));
  }
};
template <> struct Co<double> {
  __device__ static const double *W() { return dW; } __device__ static const double *DW() { return dDW; }
  __device__ static const double *Ch() { return dCheb; } __device__ static const double *ChD() { return dChebD; }
  static void up(const std::vector<double> &w, const std::vector<double> &d, const std::vector<double> &ch, const std::vector<double> &chd) {
    CK(cudaMemcpyToSymbol(dW, w.data(), 8 * w.size())); CK(cudaMemcpyToSymbol(dDW, d.data(), 8 * d.size()));
    CK(cudaMemcpyToSymbol(dCheb, ch.data(), 8 * ch.size())); CK(cudaMemcpyToSymbol(dChebD, chd.data(), 8 * chd.size()));
  }
};

template <typename R> struct Cplx;
template <> struct Cplx<float> { typedef cufftComplex T; };
template <> struct Cplx<double> { typedef cufftDoubleComplex T; };

template <typename R, int D>
__device__ __forceinline__ R horner(const R *c, int j, R t) {
  R acc = c[j * D + D - 1];
#pragma unroll
  for (int k = D - 2; k >= 0; --k) acc = fma(acc, t, c[j * D + k]);
  return acc;
}

template <typename R>
__global__ void places(int N, const double *x, double3 inv, int3 K, R off, int *first, R *s) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  double iv[3] = {inv.x, inv.y, inv.z};
  int Ks[3] = {K.x, K.y, K.z};
#pragma unroll
  for (int a = 0; a < 3; ++a) {
    double f = x[3 * i + a] * iv[a];
    f -= floor(f);
    R u = (R)f * (R)Ks[a];
    R v = u - off;
    R fl = floor(v);
    int fi = (int)fl;
    fi %= Ks[a];
    if (fi < 0) fi += Ks[a];
    first[a * N + i] = fi;
    s[a * N + i] = v - fl;
  }
}

// Spreading: a warp a particle.  Lanes 0..3P-1 compute the 3P weights (the
// first axis times the charge); every point takes its three by shuffles.
// BRICK: lanes run x fastest within a 4 x 4 brick; DIRECT: z fastest, the
// grid of the transform itself.
template <typename R, int P, int D, bool BRICK>
__global__ void spread(int N, const R *q, const int *first, const R *s, R *out, int K1, int K2, int K3, int b2) {
  int item = blockIdx.x * blockDim.x + threadIdx.x;
  int i = item >> 5;
  int lane = threadIdx.x & 31;
  if (i >= N) return;  // whole warps leave together
  int axis = lane / P;
  int j = lane - axis * P;
  R w = 0;
  if (lane < 3 * P) {
    R t = 2 * s[axis * N + i] - 1;
    w = horner<R, D>(Co<R>::W(), j, t);
    if (axis == 0) w *= q[i];
  }
  int f1 = first[i], f2 = first[N + i], f3 = first[2 * N + i];
  constexpr int P3 = P * P * P;
#pragma unroll
  for (int pass = 0; pass < (P3 + 31) / 32; ++pass) {
    int k = pass * 32 + lane;
    int a, b, c;
    if (BRICK) { c = k / (P * P); int ab = k - c * P * P; b = ab / P; a = ab - b * P; }
    else { a = k / (P * P); int bc = k - a * P * P; b = bc / P; c = bc - b * P; }
    bool in = k < P3;
    if (!in) { a = 0; b = 0; c = 0; }
    R wx = __shfl_sync(0xffffffffu, w, a);
    R wy = __shfl_sync(0xffffffffu, w, P + b);
    R wz = __shfl_sync(0xffffffffu, w, 2 * P + c);
    if (in) {
      int g1 = f1 + a; if (g1 >= K1) g1 -= K1;
      int g2 = f2 + b; if (g2 >= K2) g2 -= K2;
      int g3 = f3 + c; if (g3 >= K3) g3 -= K3;
      unsigned at;
      if (BRICK) at = ((unsigned)((g1 >> 2) * b2 + (g2 >> 2)) * K3 + g3) * 16u + (g2 & 3) * 4 + (g1 & 3);
      else at = ((unsigned)g1 * K2 + g2) * K3 + g3;
      atomicAdd(out + at, wx * wy * wz);
    }
  }
}

// The sum of a gathering over the P^3 points of a particle.  G = 32: the
// lanes 0..3P-1 compute the 3P weights and slopes once, and the others take
// them by shuffles; otherwise each lane computes what it uses.
template <typename R, int P, int D, int G, typename RowF>
__device__ __forceinline__ void gatherSum(int lane, R t1, R t2, R t3, RowF rowf, int c0, int wrap, R &gx, R &gy, R &gz) {
  R wz[P], dz[P];
  if constexpr (G == 32) {
    int ax = lane / P, jj = lane - ax * P;
    R tt = ax == 0 ? t1 : (ax == 1 ? t2 : t3);
    R mw = 0, md = 0;
    if (lane < 3 * P) { mw = horner<R, D>(Co<R>::W(), jj, tt); md = horner<R, D>(Co<R>::DW(), jj, tt); }
#pragma unroll
    for (int c = 0; c < P; ++c) { wz[c] = __shfl_sync(0xffffffffu, mw, 2 * P + c); dz[c] = __shfl_sync(0xffffffffu, md, 2 * P + c); }
#pragma unroll
    for (int base = 0; base < P * P; base += 32) {
      int ab = base + lane;
      bool in = ab < P * P;
      int a = in ? ab / P : 0, b = in ? ab - a * P : 0;
      R wx = __shfl_sync(0xffffffffu, mw, a), dx = __shfl_sync(0xffffffffu, md, a);
      R wy = __shfl_sync(0xffffffffu, mw, P + b), dy = __shfl_sync(0xffffffffu, md, P + b);
      if (in) {
        const R *row = rowf(a, b);
        R sz = 0, sd = 0;
#pragma unroll
        for (int c = 0; c < P; ++c) { int g3 = c0 + c; if (g3 >= wrap) g3 -= wrap; R v = row[g3]; sz = fma(v, wz[c], sz); sd = fma(v, dz[c], sd); }
        gx = fma(dx * wy, sz, gx); gy = fma(wx * dy, sz, gy); gz = fma(wx * wy, sd, gz);
      }
    }
  } else {
#pragma unroll
    for (int c = 0; c < P; ++c) { wz[c] = horner<R, D>(Co<R>::W(), c, t3); dz[c] = horner<R, D>(Co<R>::DW(), c, t3); }
    for (int ab = lane; ab < P * P; ab += G) {
      int a = ab / P, b = ab - a * P;
      R wx = horner<R, D>(Co<R>::W(), a, t1), dx = horner<R, D>(Co<R>::DW(), a, t1);
      R wy = horner<R, D>(Co<R>::W(), b, t2), dy = horner<R, D>(Co<R>::DW(), b, t2);
      const R *row = rowf(a, b);
      R sz = 0, sd = 0;
#pragma unroll
      for (int c = 0; c < P; ++c) { int g3 = c0 + c; if (g3 >= wrap) g3 -= wrap; R v = row[g3]; sz = fma(v, wz[c], sz); sd = fma(v, dz[c], sd); }
      gx = fma(dx * wy, sz, gx); gy = fma(wx * dy, sz, gy); gz = fma(wx * wy, sd, gz);
    }
  }
}

// Pitches of the shared tile of the tiled gathering (P0 of #337): the row
// pitch S1 and the plane pitch S2 of the tile, index a S2 + b S1 + c, chosen
// at compile time to minimize the largest number of the 32 lanes of a warp
// of one particle (lanes ab = a P + b, a pass a 32 lanes) that fall on one
// bank of the 32; T and T^2 unpadded.
__host__ __device__ constexpr int bankConflict(int P, int S1, int S2) {
  int worst = 0;
  for (int base = 0; base < P * P; base += 32) {
    int count[32] = {};
    for (int ab = base; ab < P * P && ab < base + 32; ++ab) {
      int bank = ((ab / P) * S2 + (ab % P) * S1) % 32;
      if (++count[bank] > worst) worst = count[bank];
    }
  }
  return worst;
}
__host__ __device__ constexpr int tilePitch(int P, int B, bool pad, bool plane) {
  int T = B + P - 1;
  if (!pad) return plane ? T * T : T;
  int best1 = T, best2 = T * T, bestC = 1 << 30, bestSize = 1 << 30;
  for (int s1 = T; s1 <= T + 8; ++s1)
    for (int s2 = T * s1; s2 <= T * s1 + 40; ++s2) {
      int c = bankConflict(P, s1, s2), size = (T - 1) * s2 + (T - 1) * s1 + T;
      if (c < bestC || (c == bestC && size < bestSize)) { bestC = c; bestSize = size; best1 = s1; best2 = s2; }
    }
  return plane ? best2 : best1;
}
__host__ __device__ constexpr int tileWords(int P, int B, bool pad) {
  int T = B + P - 1;
  return (T - 1) * tilePitch(P, B, pad, true) + (T - 1) * tilePitch(P, B, pad, false) + T;
}

// Tiled spreading and gathering: the particles are binned by their first
// point into bins of B^3 points (a counting sort each step); a block takes a
// bin and adds into a tile of (B + P - 1)^3 points in block memory, then adds
// the tile to the grid with global atomics (a point is in at most 8 tiles).
__global__ void binCount(int N, const int *first, int B, int nb1, int nb2, int nb3, int *count, int *bin, int *rank) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  int b = ((first[i] / B) * nb2 + first[N + i] / B) * nb3 + first[2 * N + i] / B;
  bin[i] = b;
  rank[i] = atomicAdd(count + b, 1);
}
__global__ void binScatter(int N, const int *bin, const int *rank, const int *start, int *list) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  list[start[bin[i]] + rank[i]] = i;
}

template <typename R, int P, int D, int B>
__global__ void spreadTiled(int N, const R *q, const int *first, const R *s, const int *start, const int *count, const int *list,
                            R *grid, int K1, int K2, int K3, int nb2, int nb3) {
  constexpr int T = B + P - 1, T3 = T * T * T, P3 = P * P * P;
  extern __shared__ unsigned char smraw[];
  R *tile = reinterpret_cast<R *>(smraw);
  int bin = blockIdx.x;
  int bz = bin % nb3, by = (bin / nb3) % nb2, bx = bin / (nb3 * nb2);
  int o1 = bx * B, o2 = by * B, o3 = bz * B;
  for (int t = threadIdx.x; t < T3; t += blockDim.x) tile[t] = 0;
  __syncthreads();
  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
  int s0 = start[bin], n = count[bin];
  int axis = lane / P, j = lane - axis * P;
  for (int k = warp; k < n; k += nw) {
    int i = list[s0 + k];
    R w = 0;
    if (lane < 3 * P) {
      R t = 2 * s[axis * N + i] - 1;
      w = horner<R, D>(Co<R>::W(), j, t);
      if (axis == 0) w *= q[i];
    }
    int l1 = first[i] - o1, l2 = first[N + i] - o2, l3 = first[2 * N + i] - o3;
#pragma unroll
    for (int pass = 0; pass < (P3 + 31) / 32; ++pass) {
      int kk = pass * 32 + lane;
      int a = kk / (P * P); int bc = kk - a * P * P; int b = bc / P; int c = bc - b * P;
      bool in = kk < P3;
      if (!in) { a = 0; b = 0; c = 0; }
      R wx = __shfl_sync(0xffffffffu, w, a);
      R wy = __shfl_sync(0xffffffffu, w, P + b);
      R wz = __shfl_sync(0xffffffffu, w, 2 * P + c);
      if (in) atomicAdd(tile + ((l1 + a) * T + (l2 + b)) * T + (l3 + c), wx * wy * wz);
    }
  }
  __syncthreads();
  for (int t = threadIdx.x; t < T3; t += blockDim.x) {
    R v = tile[t];
    if (v != 0) {
      int c = t % T, b = (t / T) % T, a = t / (T * T);
      int g1 = o1 + a; if (g1 >= K1) g1 -= K1;
      int g2 = o2 + b; if (g2 >= K2) g2 -= K2;
      int g3 = o3 + c; if (g3 >= K3) g3 -= K3;
      atomicAdd(grid + ((size_t)g1 * K2 + g2) * K3 + g3, v);
    }
  }
}

template <typename R, int P, int D, int B, int G, bool PAD = false>
__global__ void gatherTiled(int N, const R *q, const int *first, const R *s, const int *start, const int *count, const int *list,
                            const R *grid, R *force, int K1, int K2, int K3, int nb2, int nb3, R kx, R ky, R kz) {
  constexpr int T = B + P - 1, T3 = T * T * T;
  constexpr int S1 = tilePitch(P, B, PAD, false), S2 = tilePitch(P, B, PAD, true);
  extern __shared__ unsigned char smraw[];
  R *tile = reinterpret_cast<R *>(smraw);
  int bin = blockIdx.x;
  int bz = bin % nb3, by = (bin / nb3) % nb2, bx = bin / (nb3 * nb2);
  int o1 = bx * B, o2 = by * B, o3 = bz * B;
  int n = count[bin];
  if (n == 0) return;
  for (int t = threadIdx.x; t < T3; t += blockDim.x) {
    int c = t % T, b = (t / T) % T, a = t / (T * T);
    int g1 = o1 + a; if (g1 >= K1) g1 -= K1;
    int g2 = o2 + b; if (g2 >= K2) g2 -= K2;
    int g3 = o3 + c; if (g3 >= K3) g3 -= K3;
    tile[a * S2 + b * S1 + c] = grid[((size_t)g1 * K2 + g2) * K3 + g3];
  }
  __syncthreads();
  int s0 = start[bin];
  int groups = blockDim.x / G, gid = threadIdx.x / G, lane = threadIdx.x - gid * G;
  int rounds = (n + groups - 1) / groups;
  for (int r = 0; r < rounds; ++r) {
    int k = r * groups + gid;
    bool live = k < n;
    int i = list[s0 + (live ? k : n - 1)];
    R t1 = 2 * s[i] - 1, t2 = 2 * s[N + i] - 1, t3 = 2 * s[2 * N + i] - 1;
    int l1 = first[i] - o1, l2 = first[N + i] - o2, l3 = first[2 * N + i] - o3;
    R gx = 0, gy = 0, gz = 0;
    gatherSum<R, P, D, G>(lane, t1, t2, t3, [&](int a, int b) { return tile + (l1 + a) * S2 + (l2 + b) * S1 + l3; }, 0, T, gx, gy, gz);
#pragma unroll
    for (int o = G / 2; o; o >>= 1) {
      gx += __shfl_down_sync(0xffffffffu, gx, o, G);
      gy += __shfl_down_sync(0xffffffffu, gy, o, G);
      gz += __shfl_down_sync(0xffffffffu, gz, o, G);
    }
    if (live && lane == 0) {
      R qi = q[i];
      force[3 * i] = -qi * kx * gx; force[3 * i + 1] = -qi * ky * gy; force[3 * i + 2] = -qi * kz * gz;
    }
  }
}

// The transposed copy of MDIR (D108): a block a brick and 32 slabs along z.
template <typename R>
__global__ void copyBricks(R *bricks, R *real, int K1, int K2, int K3, int b2, int zchunks) {
  __shared__ R tile[544];
  int bxy = blockIdx.x / zchunks, zc = blockIdx.x - bxy * zchunks;
  int z0 = zc * 32;
  int bx1 = bxy / b2, by1 = bxy - bx1 * b2;
  size_t slab0 = (size_t)bxy * K3;
  int tx = threadIdx.x;
  for (int j = 0; j < 4; ++j) {
    int e = j * 128 + tx, zz = e >> 4, inxy = e & 15, z = z0 + zz;
    if (z < K3) tile[zz * 17 + inxy] = bricks[(slab0 + z) * 16 + inxy];
  }
  __syncthreads();
  for (int j = 0; j < 4; ++j) {
    int e = j * 128 + tx, zz = e >> 4, inxy = e & 15, z = z0 + zz;
    if (z < K3) bricks[(slab0 + z) * 16 + inxy] = 0;
  }
  int warp = tx >> 5, lane = tx & 31, z = z0 + lane;
  for (int pass = 0; pass < 4; ++pass) {
    int r = pass * 4 + warp, inx = r & 3, iny = r >> 2;
    int g1 = bx1 * 4 + inx, g2 = by1 * 4 + iny;
    if (g1 < K1 && g2 < K2 && z < K3) real[((size_t)g1 * K2 + g2) * K3 + z] = tile[lane * 17 + r];
  }
}

template <typename R>
__device__ __forceinline__ R clenshaw(const R *a, int n, R y) {
  R b1 = 0, b2 = 0, y2 = 2 * y;
  for (int k = n - 1; k >= 1; --k) { R t = fma(y2, b1, a[k] - b2); b2 = b1; b1 = t; }
  return fma(y, b1, a[0] - b2);
}

// The product with the influence function; with ENERGY also the energy and
// the virial (block sums in R, added in f64).
template <typename R, bool ENERGY>
__global__ void product(typename Cplx<R>::T *g, const R *t1, const R *t2, const R *t3, const R *m1, const R *m2, const R *m3,
                        int K1, int K2, int NZ, int K3, int kind, R xscale2, int ncheb, double *acc) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t tot = (size_t)K1 * K2 * NZ;
  R e = 0, w[6] = {0, 0, 0, 0, 0, 0};
  if (idx < tot) {
    int k3 = idx % NZ; size_t r = idx / NZ; int k2 = r % K2; int k1 = r / K2;
    R a = m1[k1], b = m2[k2], c = m3[k3];
    R msq = a * a + b * b + c * c;
    R G = 0, gl = 0;
    if (idx != 0) {
      R B = t1[k1] * t2[k2] * t3[k3];
      if (kind == 0) {
        G = B / msq;
        if (ENERGY) gl = -2 * xscale2 * msq;  // d ln(chi_hat)/d ln(xi) of the Gaussian
      } else {
        R x2 = xscale2 * msq;
        if (x2 < 1) {
          R y = 2 * x2 - 1;
          R ch = clenshaw<R>(Co<R>::Ch(), ncheb, y);
          G = B * ch / msq;
          if (ENERGY) gl = clenshaw<R>(Co<R>::ChD(), ncheb, y) / ch;
        }
      }
    }
    typename Cplx<R>::T v = g[idx];
    if (ENERGY) {
      R h = (k3 == 0 || (2 * k3 == K3)) ? (R)0.5 : (R)1;
      e = h * G * (v.x * v.x + v.y * v.y);
      R f = (idx != 0) ? (gl - 2) / msq : 0;
      R mm[3] = {a, b, c};
      w[0] = e * (1 + f * mm[0] * mm[0]); w[1] = e * (1 + f * mm[1] * mm[1]); w[2] = e * (1 + f * mm[2] * mm[2]);
      w[3] = e * f * mm[0] * mm[1]; w[4] = e * f * mm[0] * mm[2]; w[5] = e * f * mm[1] * mm[2];
    }
    v.x *= G; v.y *= G;
    g[idx] = v;
  }
  if (ENERGY) {
    __shared__ R red[7][4];
    R vals[7] = {e, w[0], w[1], w[2], w[3], w[4], w[5]};
    for (int k = 0; k < 7; ++k) {
      R v = vals[k];
      for (int o = 16; o; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
      if ((threadIdx.x & 31) == 0) red[k][threadIdx.x >> 5] = v;
    }
    __syncthreads();
    if (threadIdx.x < 7) {
      double s = 0;
      for (int k = 0; k < 4; ++k) s += (double)red[threadIdx.x][k];
      atomicAdd(acc + threadIdx.x, s);
    }
  }
}

// Gathering: G lanes a particle, each over rows (a, b) along z.
template <typename R, int P, int D, int G>
__global__ void gather(int N, const R *q, const int *first, const R *s, const R *grid, R *force, int K1, int K2, int K3, R kx, R ky, R kz) {
  int item = blockIdx.x * blockDim.x + threadIdx.x;
  int i = item / G;
  int lane = item - i * G;
  bool live = i < N;
  int ii = live ? i : N - 1;
  R t1 = 2 * s[ii] - 1, t2 = 2 * s[N + ii] - 1, t3 = 2 * s[2 * N + ii] - 1;
  int f1 = first[ii], f2 = first[N + ii], f3 = first[2 * N + ii];
  R gx = 0, gy = 0, gz = 0;
  gatherSum<R, P, D, G>(lane, t1, t2, t3, [&](int a, int b) {
    int g1 = f1 + a; if (g1 >= K1) g1 -= K1;
    int g2 = f2 + b; if (g2 >= K2) g2 -= K2;
    return grid + ((size_t)g1 * K2 + g2) * K3; }, f3, K3, gx, gy, gz);
#pragma unroll
  for (int o = G / 2; o; o >>= 1) {
    gx += __shfl_down_sync(0xffffffffu, gx, o, G);
    gy += __shfl_down_sync(0xffffffffu, gy, o, G);
    gz += __shfl_down_sync(0xffffffffu, gz, o, G);
  }
  if (live && lane == 0) {
    R qi = q[i];
    force[3 * i] = -qi * kx * gx; force[3 * i + 1] = -qi * ky * gy; force[3 * i + 2] = -qi * kz * gz;
  }
}

// The measurement P0 of #337: the gathering with the weights and slopes of
// each particle precomputed in memory, so that it does only the P^3
// accumulation. w[((a * 2 + d) * P + j) * N + i]: axis a, value (d = 0) or
// slope (d = 1), point j, particle i.
template <typename R, int P, int D>
__global__ void precompute(int N, const R *s, R *w) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
#pragma unroll
  for (int a = 0; a < 3; ++a) {
    R t = 2 * s[a * N + i] - 1;
#pragma unroll
    for (int j = 0; j < P; ++j) {
      w[((a * 2 + 0) * P + j) * (size_t)N + i] = horner<R, D>(Co<R>::W(), j, t);
      w[((a * 2 + 1) * P + j) * (size_t)N + i] = horner<R, D>(Co<R>::DW(), j, t);
    }
  }
}

// G lanes a particle over the rows (a, b); each grid value is read once
// and gives the three components (sz for x and y, sd for z).
template <typename R, int P, int G, typename RowF>
__device__ __forceinline__ void gatherSumPre(int lane, int i, int N, const R *w, RowF rowf, int c0, int wrap,
                                             R &gx, R &gy, R &gz) {
  R wz[P], dz[P];
#pragma unroll
  for (int c = 0; c < P; ++c) { wz[c] = w[((4) * P + c) * (size_t)N + i]; dz[c] = w[((5) * P + c) * (size_t)N + i]; }
  for (int ab = lane; ab < P * P; ab += G) {
    int a = ab / P, b = ab - a * P;
    R wx = w[((0) * P + a) * (size_t)N + i], dx = w[((1) * P + a) * (size_t)N + i];
    R wy = w[((2) * P + b) * (size_t)N + i], dy = w[((3) * P + b) * (size_t)N + i];
    const R *row = rowf(a, b);
    R sz = 0, sd = 0;
#pragma unroll
    for (int c = 0; c < P; ++c) { int g3 = c0 + c; if (g3 >= wrap) g3 -= wrap; R v = row[g3]; sz = fma(v, wz[c], sz); sd = fma(v, dz[c], sd); }
    gx = fma(dx * wy, sz, gx); gy = fma(wx * dy, sz, gy); gz = fma(wx * wy, sd, gz);
  }
}

template <typename R, int P, int G>
__global__ void gatherPre(int N, const R *q, const int *first, const R *w, const R *grid, R *force, int K1, int K2, int K3, R kx, R ky, R kz) {
  int item = blockIdx.x * blockDim.x + threadIdx.x;
  int i = item / G, lane = item - i * G;
  bool live = i < N;
  int ii = live ? i : N - 1;
  int f1 = first[ii], f2 = first[N + ii], f3 = first[2 * N + ii];
  R gx = 0, gy = 0, gz = 0;
  gatherSumPre<R, P, G>(lane, ii, N, w, [&](int a, int b) {
    int g1 = f1 + a; if (g1 >= K1) g1 -= K1;
    int g2 = f2 + b; if (g2 >= K2) g2 -= K2;
    return grid + ((size_t)g1 * K2 + g2) * K3; }, f3, K3, gx, gy, gz);
#pragma unroll
  for (int o = G / 2; o; o >>= 1) {
    gx += __shfl_down_sync(0xffffffffu, gx, o, G);
    gy += __shfl_down_sync(0xffffffffu, gy, o, G);
    gz += __shfl_down_sync(0xffffffffu, gz, o, G);
  }
  if (live && lane == 0) { R qi = q[i]; force[3 * i] = -qi * kx * gx; force[3 * i + 1] = -qi * ky * gy; force[3 * i + 2] = -qi * kz * gz; }
}

template <typename R, int P, int B, int G, bool PAD = false>
__global__ void gatherPreTiled(int N, const R *q, const int *first, const R *w, const int *start, const int *count, const int *list,
                               const R *grid, R *force, int K1, int K2, int K3, int nb2, int nb3, R kx, R ky, R kz) {
  constexpr int T = B + P - 1, T3 = T * T * T;
  constexpr int S1 = tilePitch(P, B, PAD, false), S2 = tilePitch(P, B, PAD, true);
  extern __shared__ unsigned char smraw[];
  R *tile = reinterpret_cast<R *>(smraw);
  int bin = blockIdx.x;
  int bz = bin % nb3, by = (bin / nb3) % nb2, bx = bin / (nb3 * nb2);
  int o1 = bx * B, o2 = by * B, o3 = bz * B;
  int n = count[bin];
  if (n == 0) return;
  for (int t = threadIdx.x; t < T3; t += blockDim.x) {
    int c = t % T, b = (t / T) % T, a = t / (T * T);
    int g1 = o1 + a; if (g1 >= K1) g1 -= K1;
    int g2 = o2 + b; if (g2 >= K2) g2 -= K2;
    int g3 = o3 + c; if (g3 >= K3) g3 -= K3;
    tile[a * S2 + b * S1 + c] = grid[((size_t)g1 * K2 + g2) * K3 + g3];
  }
  __syncthreads();
  int s0 = start[bin];
  int groups = blockDim.x / G, gid = threadIdx.x / G, lane = threadIdx.x - gid * G;
  for (int r = 0; r < (n + groups - 1) / groups; ++r) {
    int k = r * groups + gid;
    bool live = k < n;
    int i = list[s0 + (live ? k : n - 1)];
    int l1 = first[i] - o1, l2 = first[N + i] - o2, l3 = first[2 * N + i] - o3;
    R gx = 0, gy = 0, gz = 0;
    gatherSumPre<R, P, G>(lane, i, N, w, [&](int a, int b) { return tile + (l1 + a) * S2 + (l2 + b) * S1 + l3; }, 0, T, gx, gy, gz);
#pragma unroll
    for (int o = G / 2; o; o >>= 1) {
      gx += __shfl_down_sync(0xffffffffu, gx, o, G);
      gy += __shfl_down_sync(0xffffffffu, gy, o, G);
      gz += __shfl_down_sync(0xffffffffu, gz, o, G);
    }
    if (live && lane == 0) { R qi = q[i]; force[3 * i] = -qi * kx * gx; force[3 * i + 1] = -qi * ky * gy; force[3 * i + 2] = -qi * kz * gz; }
  }
}

template <typename R> struct FFT;
template <> struct FFT<float> {
  static void plans(cufftHandle &f, cufftHandle &b, int K1, int K2, int K3) { CF(cufftPlan3d(&f, K1, K2, K3, CUFFT_R2C)); CF(cufftPlan3d(&b, K1, K2, K3, CUFFT_C2R)); }
  static void fwd(cufftHandle p, float *r, cufftComplex *c) { CF(cufftExecR2C(p, r, c)); }
  static void bwd(cufftHandle p, cufftComplex *c, float *r) { CF(cufftExecC2R(p, c, r)); }
};
template <> struct FFT<double> {
  static void plans(cufftHandle &f, cufftHandle &b, int K1, int K2, int K3) { CF(cufftPlan3d(&f, K1, K2, K3, CUFFT_D2Z)); CF(cufftPlan3d(&b, K1, K2, K3, CUFFT_Z2D)); }
  static void fwd(cufftHandle p, double *r, cufftDoubleComplex *c) { CF(cufftExecD2Z(p, r, c)); }
  static void bwd(cufftHandle p, cufftDoubleComplex *c, double *r) { CF(cufftExecZ2D(p, c, r)); }
};

struct Opts { int reps; bool brick; int glanes; bool sorted; const char *fout; int tiled; };

template <typename R, int P, int D>
void run(Case &c, Opts o) {
  int N = c.N, K1 = c.K1, K2 = c.K2, K3 = c.K3, NZ = K3 / 2 + 1;
  std::vector<int> perm(N);
  for (int i = 0; i < N; ++i) perm[i] = i;
  if (o.sorted) {
    // cells of about 1 nm, z fastest, as a spatial order of the particles
    std::vector<long> key(N);
    for (int i = 0; i < N; ++i) {
      long k = 0;
      for (int a = 0; a < 3; ++a) {
        double f = c.x[3 * i + a] / c.L[a]; f -= floor(f);
        int nc = std::max(1, (int)(c.L[a] / 1.0));
        k = k * 4096 + (long)(f * nc);
      }
      key[i] = k;
    }
    std::stable_sort(perm.begin(), perm.end(), [&](int a, int b) { return key[a] < key[b]; });
  }
  std::vector<double> xs(3 * (size_t)N);
  std::vector<R> qs(N);
  for (int i = 0; i < N; ++i) { for (int a = 0; a < 3; ++a) xs[3 * i + a] = c.x[3 * perm[i] + a]; qs[i] = (R)c.q[perm[i]]; }
  double *dx; R *dq, *ds, *dgrid, *dbricks, *dforce; int *dfirst;
  typename Cplx<R>::T *dc;
  int b1 = (K1 + 3) / 4, b2 = (K2 + 3) / 4;
  size_t nbr = (size_t)b1 * b2 * K3 * 16;
  CK(cudaMalloc(&dx, 8 * 3 * (size_t)N));
  CK(cudaMalloc(&dq, sizeof(R) * N));
  CK(cudaMalloc(&ds, sizeof(R) * 3 * N));
  CK(cudaMalloc(&dfirst, 4 * 3 * N));
  CK(cudaMalloc(&dgrid, sizeof(R) * (size_t)K1 * K2 * K3));
  CK(cudaMalloc(&dbricks, sizeof(R) * nbr));
  CK(cudaMalloc(&dforce, sizeof(R) * 3 * N));
  CK(cudaMalloc(&dc, sizeof(typename Cplx<R>::T) * (size_t)K1 * K2 * NZ));
  CK(cudaMemset(dbricks, 0, sizeof(R) * nbr));
  CK(cudaMemcpy(dx, xs.data(), 8 * 3 * (size_t)N, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dq, qs.data(), sizeof(R) * N, cudaMemcpyHostToDevice));
  Co<R>::up(c.w, c.d, c.cheb, c.chebd);
  // tables: B along each edge (prefactor on the first) and m along each edge
  std::vector<R> t1(K1), t2(K2), t3(NZ), m1(K1), m2(K2), m3(NZ);
  for (int k = 0; k < K1; ++k) { t1[k] = (R)c.B1[k]; m1[k] = (R)((k <= K1 / 2 ? k : k - K1) / c.L[0]); }
  for (int k = 0; k < K2; ++k) { t2[k] = (R)c.B2[k]; m2[k] = (R)((k <= K2 / 2 ? k : k - K2) / c.L[1]); }
  for (int k = 0; k < NZ; ++k) { t3[k] = (R)c.B3[k]; m3[k] = (R)(k / c.L[2]); }
  R *dt1, *dt2, *dt3, *dm1, *dm2, *dm3; double *dacc;
  CK(cudaMalloc(&dt1, sizeof(R) * K1)); CK(cudaMalloc(&dt2, sizeof(R) * K2)); CK(cudaMalloc(&dt3, sizeof(R) * NZ));
  CK(cudaMalloc(&dm1, sizeof(R) * K1)); CK(cudaMalloc(&dm2, sizeof(R) * K2)); CK(cudaMalloc(&dm3, sizeof(R) * NZ));
  CK(cudaMalloc(&dacc, 8 * 8));
  CK(cudaMemcpy(dt1, t1.data(), sizeof(R) * K1, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dt2, t2.data(), sizeof(R) * K2, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dt3, t3.data(), sizeof(R) * NZ, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dm1, m1.data(), sizeof(R) * K1, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dm2, m2.data(), sizeof(R) * K2, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dm3, m3.data(), sizeof(R) * NZ, cudaMemcpyHostToDevice));
  cufftHandle pf, pb;
  FFT<R>::plans(pf, pb, K1, K2, K3);
  double3 inv = make_double3(1 / c.L[0], 1 / c.L[1], 1 / c.L[2]);
  int3 Kv = make_int3(K1, K2, K3);
  R xscale2 = (R)(c.kind == 0 ? c.xscale : c.xscale * c.xscale);
  const int T = 128;
  int gN = (N + T - 1) / T, gW = (int)(((size_t)N * 32 + T - 1) / T), zch = (K3 + 31) / 32;
  int gP = (int)(((size_t)K1 * K2 * NZ + T - 1) / T), gG = (int)(((size_t)N * o.glanes + T - 1) / T);
  enum { E_PLACES, E_SPREAD, E_COPY, E_FWD, E_PROD, E_BWD, E_GATHER, E_N };
  const char *names[E_N] = {"places", "spread", "copy", "fft_fwd", "product", "fft_bwd", "gather"};
  cudaEvent_t ev[E_N + 1];
  for (int k = 0; k <= E_N; ++k) CK(cudaEventCreate(&ev[k]));
  std::vector<std::vector<float>> times(E_N + 2);
  auto launchGather = [&]() {
    R kx = (R)(K1 / c.L[0]), ky = (R)(K2 / c.L[1]), kz = (R)(K3 / c.L[2]);
    switch (o.glanes) {
      case 4: gather<R, P, D, 4><<<gG, T>>>(N, dq, dfirst, ds, dgrid, dforce, K1, K2, K3, kx, ky, kz); break;
      case 8: gather<R, P, D, 8><<<gG, T>>>(N, dq, dfirst, ds, dgrid, dforce, K1, K2, K3, kx, ky, kz); break;
      case 16: gather<R, P, D, 16><<<gG, T>>>(N, dq, dfirst, ds, dgrid, dforce, K1, K2, K3, kx, ky, kz); break;
      default: gather<R, P, D, 32><<<gG, T>>>(N, dq, dfirst, ds, dgrid, dforce, K1, K2, K3, kx, ky, kz); break;
    }
  };
  // bins of the tiled kernels
  int Bt = o.tiled ? o.tiled : 8;
  int nb1 = (K1 + Bt - 1) / Bt, nb2 = (K2 + Bt - 1) / Bt, nb3 = (K3 + Bt - 1) / Bt, nb = nb1 * nb2 * nb3;
  int *dcount, *dstart, *dbin, *drank, *dlist; void *dtemp = nullptr; size_t ntemp = 0;
  CK(cudaMalloc(&dcount, 4 * nb)); CK(cudaMalloc(&dstart, 4 * nb));
  CK(cudaMalloc(&dbin, 4 * N)); CK(cudaMalloc(&drank, 4 * N)); CK(cudaMalloc(&dlist, 4 * N));
  CK(cub::DeviceScan::ExclusiveSum(dtemp, ntemp, dcount, dstart, nb));
  CK(cudaMalloc(&dtemp, ntemp));
  int Tt = Bt + P - 1;
  size_t smem = sizeof(R) * Tt * Tt * Tt;
  auto spreadT = [&]() {
    if (Bt == 4) { CK(cudaFuncSetAttribute(spreadTiled<R, P, D, 4>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem)); spreadTiled<R, P, D, 4><<<nb, 256, smem>>>(N, dq, dfirst, ds, dstart, dcount, dlist, dgrid, K1, K2, K3, nb2, nb3); }
    else { CK(cudaFuncSetAttribute(spreadTiled<R, P, D, 8>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem)); spreadTiled<R, P, D, 8><<<nb, 256, smem>>>(N, dq, dfirst, ds, dstart, dcount, dlist, dgrid, K1, K2, K3, nb2, nb3); }
  };
  auto gatherT = [&]() {
    R kx = (R)(K1 / c.L[0]), ky = (R)(K2 / c.L[1]), kz = (R)(K3 / c.L[2]);
#define GT(BB, GG) { CK(cudaFuncSetAttribute(gatherTiled<R, P, D, BB, GG>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem)); \
    gatherTiled<R, P, D, BB, GG><<<nb, 256, smem>>>(N, dq, dfirst, ds, dstart, dcount, dlist, dgrid, dforce, K1, K2, K3, nb2, nb3, kx, ky, kz); }
    if (Bt == 4) { if (o.glanes == 4) GT(4, 4) else if (o.glanes == 8) GT(4, 8) else if (o.glanes == 16) GT(4, 16) else GT(4, 32) }
    else { if (o.glanes == 4) GT(8, 4) else if (o.glanes == 8) GT(8, 8) else if (o.glanes == 16) GT(8, 16) else GT(8, 32) }
#undef GT
  };
  auto step = [&](bool energy, bool timed) {
    if (timed) CK(cudaEventRecord(ev[0]));
    places<R><<<gN, T>>>(N, dx, inv, Kv, (R)c.off, dfirst, ds);
    if (timed) CK(cudaEventRecord(ev[1]));
    if (o.tiled) {
      CK(cudaMemsetAsync(dcount, 0, 4 * nb));
      binCount<<<gN, T>>>(N, dfirst, Bt, nb1, nb2, nb3, dcount, dbin, drank);
      CK(cub::DeviceScan::ExclusiveSum(dtemp, ntemp, dcount, dstart, nb));
      binScatter<<<gN, T>>>(N, dbin, drank, dstart, dlist);
      CK(cudaMemsetAsync(dgrid, 0, sizeof(R) * (size_t)K1 * K2 * K3));
      spreadT();
    } else if (o.brick) spread<R, P, D, true><<<gW, T>>>(N, dq, dfirst, ds, dbricks, K1, K2, K3, b2);
    else { CK(cudaMemsetAsync(dgrid, 0, sizeof(R) * (size_t)K1 * K2 * K3)); spread<R, P, D, false><<<gW, T>>>(N, dq, dfirst, ds, dgrid, K1, K2, K3, b2); }
    if (timed) CK(cudaEventRecord(ev[2]));
    if (o.brick && !o.tiled) copyBricks<R><<<b1 * b2 * zch, T>>>(dbricks, dgrid, K1, K2, K3, b2, zch);
    if (timed) CK(cudaEventRecord(ev[3]));
    FFT<R>::fwd(pf, dgrid, dc);
    if (timed) CK(cudaEventRecord(ev[4]));
    if (energy) { CK(cudaMemsetAsync(dacc, 0, 64)); product<R, true><<<gP, T>>>(dc, dt1, dt2, dt3, dm1, dm2, dm3, K1, K2, NZ, K3, c.kind, xscale2, c.ncheb, dacc); }
    else product<R, false><<<gP, T>>>(dc, dt1, dt2, dt3, dm1, dm2, dm3, K1, K2, NZ, K3, c.kind, xscale2, c.ncheb, dacc);
    if (timed) CK(cudaEventRecord(ev[5]));
    FFT<R>::bwd(pb, dc, dgrid);
    if (timed) CK(cudaEventRecord(ev[6]));
    if (o.tiled) gatherT(); else launchGather();
    if (timed) CK(cudaEventRecord(ev[7]));
  };
  for (int r = 0; r < 20; ++r) step(false, false);
  CK(cudaDeviceSynchronize());
  for (int r = 0; r < o.reps; ++r) {
    step(false, true);
    CK(cudaEventSynchronize(ev[E_N]));
    float tot = 0;
    for (int k = 0; k < E_N; ++k) { float ms; CK(cudaEventElapsedTime(&ms, ev[k], ev[k + 1])); times[k].push_back(ms * 1000); tot += ms * 1000; }
    times[E_N].push_back(tot);
  }
  // back to back, no events between the kernels: the time of the sequence
  cudaEvent_t a0, a1; CK(cudaEventCreate(&a0)); CK(cudaEventCreate(&a1));
  CK(cudaEventRecord(a0));
  for (int r = 0; r < o.reps; ++r) step(false, false);
  CK(cudaEventRecord(a1)); CK(cudaEventSynchronize(a1));
  float seq; CK(cudaEventElapsedTime(&seq, a0, a1));
  // with energy and virial
  CK(cudaEventRecord(a0));
  for (int r = 0; r < o.reps; ++r) step(true, false);
  CK(cudaEventRecord(a1)); CK(cudaEventSynchronize(a1));
  float seqE; CK(cudaEventElapsedTime(&seqE, a0, a1));
  CK(cudaDeviceSynchronize());
  double acc[8];
  CK(cudaMemcpy(acc, dacc, 64, cudaMemcpyDeviceToHost));
  // P0 (#337): the gathering alone, the grid as the last step left it,
  // with the weights computed in the kernel and precomputed in memory,
  // from the grid in global memory and from tiles in block memory (the
  // bins of the step, B = o.tiled). Medians of o.reps launches.
  double p0[7] = {0, 0, 0, 0, 0, 0, 0};
  if (getenv("ESP_P0") && o.tiled) {
    R *dw;
    CK(cudaMalloc(&dw, sizeof(R) * 6 * P * (size_t)N));
    R kx = (R)(K1 / c.L[0]), ky = (R)(K2 / c.L[1]), kz = (R)(K3 / c.L[2]);
    int G = o.glanes;
    size_t padded = sizeof(R) * tileWords(P, 4, true);
    int gGP = (int)(((size_t)N * G + T - 1) / T);
    auto timeIt = [&](auto launch) {
      std::vector<float> v;
      for (int r = 0; r < 10; ++r) launch();
      for (int r = 0; r < o.reps; ++r) {
        CK(cudaEventRecord(a0)); launch(); CK(cudaEventRecord(a1)); CK(cudaEventSynchronize(a1));
        float ms; CK(cudaEventElapsedTime(&ms, a0, a1)); v.push_back(ms * 1000);
      }
      std::sort(v.begin(), v.end());
      return (double)v[v.size() / 2];
    };
    p0[0] = timeIt([&] { launchGather(); });
    p0[1] = timeIt([&] { gatherT(); });
    p0[2] = timeIt([&] { precompute<R, P, D><<<gN, T>>>(N, ds, dw); });
#define PRE(GG) \
    p0[3] = timeIt([&] { gatherPre<R, P, GG><<<gGP, T>>>(N, dq, dfirst, dw, dgrid, dforce, K1, K2, K3, kx, ky, kz); }); \
    CK(cudaFuncSetAttribute(gatherPreTiled<R, P, 4, GG>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem)); \
    p0[4] = timeIt([&] { gatherPreTiled<R, P, 4, GG><<<nb, 256, smem>>>(N, dq, dfirst, dw, dstart, dcount, dlist, dgrid, dforce, K1, K2, K3, nb2, nb3, kx, ky, kz); }); \
    CK(cudaFuncSetAttribute(gatherPreTiled<R, P, 4, GG, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)padded)); \
    p0[6] = timeIt([&] { gatherPreTiled<R, P, 4, GG, true><<<nb, 256, padded>>>(N, dq, dfirst, dw, dstart, dcount, dlist, dgrid, dforce, K1, K2, K3, nb2, nb3, kx, ky, kz); }); \
    CK(cudaFuncSetAttribute(gatherTiled<R, P, D, 4, GG, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)padded)); \
    p0[5] = timeIt([&] { gatherTiled<R, P, D, 4, GG, true><<<nb, 256, padded>>>(N, dq, dfirst, ds, dstart, dcount, dlist, dgrid, dforce, K1, K2, K3, nb2, nb3, kx, ky, kz); });
    if (o.tiled == 4) {
      if (G == 4) { PRE(4) } else if (G == 8) { PRE(8) } else if (G == 16) { PRE(16) } else { PRE(32) }
    }
#undef PRE
    CK(cudaFree(dw));
  }
  printf("{\"p0_gather_us\": %.2f, \"p0_gather_tiled_us\": %.2f, \"p0_precompute_us\": %.2f, \"p0_gather_pre_us\": %.2f, \"p0_gather_pre_tiled_us\": %.2f, "
         "\"p0_gather_tiled_padded_us\": %.2f, \"p0_gather_pre_tiled_padded_us\": %.2f, \"p0_pitches\": [%d, %d], \"p0_conflicts\": [%d, %d], ",
         p0[0], p0[1], p0[2], p0[3], p0[4], p0[5], p0[6], tilePitch(P, 4, true, false), tilePitch(P, 4, true, true),
         bankConflict(P, P + 3, (P + 3) * (P + 3)), bankConflict(P, tilePitch(P, 4, true, false), tilePitch(P, 4, true, true)));
  printf("\"N\": %d, \"K\": [%d, %d, %d], \"P\": %d, \"D\": %d, \"kind\": %d, \"prec\": \"%s\", \"layout\": \"%s\", \"glanes\": %d, \"order\": \"%s\"",
         N, K1, K2, K3, P, D, c.kind, sizeof(R) == 4 ? "f32" : "f64", o.tiled ? (o.tiled == 4 ? "tiled4" : "tiled8") : o.brick ? "brick" : "direct", o.glanes, o.sorted ? "sorted" : "input");
  for (int k = 0; k <= E_N; ++k) {
    auto v = times[k];
    std::sort(v.begin(), v.end());
    printf(", \"%s_us\": %.2f", k < E_N ? names[k] : "sum", v[v.size() / 2]);
  }
  printf(", \"sequence_us\": %.2f, \"sequence_energy_us\": %.2f, \"energy_kJ\": %.9f, \"virial\": [%.6f, %.6f, %.6f, %.6f, %.6f, %.6f]}\n",
         1000 * seq / o.reps, 1000 * seqE / o.reps, acc[0], acc[1], acc[2], acc[3], acc[4], acc[5], acc[6]);
  if (o.fout && strcmp(o.fout, "-") != 0) {
    std::vector<R> fr(3 * (size_t)N);
    CK(cudaMemcpy(fr.data(), dforce, sizeof(R) * 3 * N, cudaMemcpyDeviceToHost));
    std::vector<double> fo(3 * (size_t)N);
    for (int i = 0; i < N; ++i) for (int a = 0; a < 3; ++a) fo[3 * perm[i] + a] = fr[3 * i + a];
    FILE *f = fopen(o.fout, "wb"); fwrite(fo.data(), 8, fo.size(), f); fwrite(acc, 8, 7, f); fclose(f);
  }
  cufftDestroy(pf); cufftDestroy(pb);
  cudaFree(dx); cudaFree(dq); cudaFree(ds); cudaFree(dfirst); cudaFree(dgrid); cudaFree(dbricks); cudaFree(dforce); cudaFree(dc);
  cudaFree(dt1); cudaFree(dt2); cudaFree(dt3); cudaFree(dm1); cudaFree(dm2); cudaFree(dm3); cudaFree(dacc);
}

template <typename R, int P>
void runD(Case &c, Opts o) {
  switch (c.D) {
#define DC(d) case d: run<R, P, d>(c, o); break;
    DC(4) DC(5) DC(6) DC(7) DC(8) DC(9) DC(10) DC(11) DC(12)
    default: fprintf(stderr, "degree %d\n", c.D); exit(1);
  }
}

template <typename R>
void runP(Case &c, Opts o) {
  switch (c.P) {
    case 4: runD<R, 4>(c, o); break;
    case 5: runD<R, 5>(c, o); break;
    case 6: runD<R, 6>(c, o); break;
    case 7: runD<R, 7>(c, o); break;
    case 8: runD<R, 8>(c, o); break;
    default: fprintf(stderr, "P %d\n", c.P); exit(1);
  }
}

int main(int argc, char **argv) {
  if (argc < 8) { fprintf(stderr, "esp_bench CASE f|d REPS brick|direct GLANES input|sorted FORCES\n"); return 2; }
  Case c = readCase(argv[1]);
  Opts o{atoi(argv[3]), strcmp(argv[4], "brick") == 0, atoi(argv[5]), strcmp(argv[6], "sorted") == 0, argv[7], strcmp(argv[4], "tiled4") == 0 ? 4 : strcmp(argv[4], "tiled8") == 0 ? 8 : 0};
  if (argv[2][0] == 'd') runP<double>(c, o); else runP<float>(c, o);
  return 0;
}
