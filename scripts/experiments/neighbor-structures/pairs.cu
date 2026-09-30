// Prototype (2026-09-30, D82): the loop over pairs of JAC with the neighbor matrix (16 lanes per
// particle, as MDIR lowers it) against a tile structure of 8 (full list, 8x4
// lanes per warp), both in f32 with LJ from type tables and the direct sum of
// PME. Measures the kernels only; the lists are built on the host.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

static int n, ntypes;
static double box[3];
static std::vector<double> x;
static std::vector<float> q, ta, tb;
static std::vector<int> type, exoff, exl;

static const float RC = 8.0f, BETA = 0.3945f;
static float REACH = 10.0f;

static double mi(double d, double l) { return d - l * std::nearbyint(d / l); }
static double dist2(int i, int j) {
  double s = 0;
  for (int k = 0; k < 3; ++k) { double d = mi(x[3*i+k] - x[3*j+k], box[k]); s += d * d; }
  return s;
}
static bool excluded(int i, int j) {
  for (int k = exoff[i]; k < exoff[i+1]; ++k) if (exl[k] == j) return true;
  return false;
}

__device__ __forceinline__ void pairForce(float3 xi, float qi, int ti, float3 xj, float qj, int tj,
                                          float3 L, float3 iL, const float *A, const float *B, int nt,
                                          float3 &f) {
  float dx = xi.x - xj.x, dy = xi.y - xj.y, dz = xi.z - xj.z;
  dx -= L.x * rintf(dx * iL.x); dy -= L.y * rintf(dy * iL.y); dz -= L.z * rintf(dz * iL.z);
  float r2 = dx*dx + dy*dy + dz*dz;
  if (r2 < RC * RC) {
    float ri = rsqrtf(r2), r2i = ri * ri, r6i = r2i * r2i * r2i;
    float a = A[ti * nt + tj], b = B[ti * nt + tj];
    float r = r2 * ri;
    float fs = (12.f * a * r6i - 6.f * b) * r6i * r2i;
    float br = BETA * r;
    fs += qi * qj * (erfcf(br) * ri + 1.1283792f * BETA * __expf(-br * br)) * r2i;
    f.x += fs * dx; f.y += fs * dy; f.z += fs * dz;
  }
}

// The matrix: 16 lanes per particle.
__global__ void matrixKernel(int n, int W, const int *counts, const int *index, const float *xs,
                             const float *qs, const int *ts, const float *A, const float *B, int nt,
                             float3 L, float3 iL, float *force) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  int i = t / 16, lane = t % 16;
  bool valid = i < n;
  float3 f = {0, 0, 0};
  if (valid) {
    float3 xi = {xs[3*i], xs[3*i+1], xs[3*i+2]};
    float qi = qs[i]; int ti = ts[i];
    int c = counts[i];
    for (int e = lane; e < c; e += 16) {
      int j = index[(size_t)i * W + e];
      if (j == i) continue;
      float3 xj = {xs[3*j], xs[3*j+1], xs[3*j+2]};
      pairForce(xi, qi, ti, xj, qs[j], ts[j], L, iL, A, B, nt, f);
    }
  }
  for (int o = 8; o; o >>= 1) {
    f.x += __shfl_xor_sync(0xffffffff, f.x, o);
    f.y += __shfl_xor_sync(0xffffffff, f.y, o);
    f.z += __shfl_xor_sync(0xffffffff, f.z, o);
  }
  if (valid && lane == 0) { force[3*i] = f.x; force[3*i+1] = f.y; force[3*i+2] = f.z; }
}

// Tiles: a warp per target tile, lane = 4 u + q; the lanes of u take the
// sources q and q + 4 of each record.
__global__ void tileKernel(int T, const int *rowStart, const int *recTile, const unsigned long long *recMask,
                           const float4 *xt, const int *tt, const int *members, const float *A,
                           const float *B, int nt, float3 L, float3 iL, float *force) {
  int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  int lane = threadIdx.x % 32;
  if (warp >= T) return;
  int u = lane >> 2, qq = lane & 3;
  float4 pi = xt[warp * 8 + u];
  int ti = tt[warp * 8 + u];
  float3 xi = {pi.x, pi.y, pi.z};
  float3 f = {0, 0, 0};
  int end = rowStart[warp + 1];
  for (int r = rowStart[warp]; r < end; ++r) {
    int b = recTile[r];
    unsigned long long m = recMask[r];
    float4 ps = xt[b * 8 + (lane & 7)];
    int ts = tt[b * 8 + (lane & 7)];
    unsigned bits = (unsigned)(m >> (u * 8)) & 0xffu;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      int v = qq + 4 * h;
      float4 pj;
      pj.x = __shfl_sync(0xffffffff, ps.x, v); pj.y = __shfl_sync(0xffffffff, ps.y, v);
      pj.z = __shfl_sync(0xffffffff, ps.z, v); pj.w = __shfl_sync(0xffffffff, ps.w, v);
      int tj = __shfl_sync(0xffffffff, ts, v);
      if (bits >> v & 1u)
        pairForce(xi, pi.w, ti, make_float3(pj.x, pj.y, pj.z), pj.w, tj, L, iL, A, B, nt, f);
    }
  }
  for (int o = 1; o < 4; o <<= 1) {
    f.x += __shfl_xor_sync(0xffffffff, f.x, o);
    f.y += __shfl_xor_sync(0xffffffff, f.y, o);
    f.z += __shfl_xor_sync(0xffffffff, f.z, o);
  }
  int i = members[warp * 8 + u];
  if (qq == 0 && i >= 0) { force[3*i] = f.x; force[3*i+1] = f.y; force[3*i+2] = f.z; }
}

// 8 x 4: records of a target tile of 8 and a half of a source tile (4); a
// lane takes one pair of a record.
__global__ void halfKernel(int T, const int *rowStart, const int *recTile, const unsigned *recMask,
                           const float4 *xt, const int *tt, const int *members, const float *A,
                           const float *B, int nt, float3 L, float3 iL, float *force) {
  int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  int lane = threadIdx.x % 32;
  if (warp >= T) return;
  int u = lane >> 2, qq = lane & 3;
  float4 pi = xt[warp * 8 + u];
  int ti = tt[warp * 8 + u];
  float3 xi = {pi.x, pi.y, pi.z};
  float3 f = {0, 0, 0};
  int end = rowStart[warp + 1];
  for (int r = rowStart[warp]; r < end; ++r) {
    int b = recTile[r];              // 4 * tile + half
    unsigned m = recMask[r];
    float4 pj = xt[b * 4 + qq];
    int tj = tt[b * 4 + qq];
    if (m >> (u * 4 + qq) & 1u)
      pairForce(xi, pi.w, ti, make_float3(pj.x, pj.y, pj.z), pj.w, tj, L, iL, A, B, nt, f);
  }
  for (int o = 1; o < 4; o <<= 1) {
    f.x += __shfl_xor_sync(0xffffffff, f.x, o);
    f.y += __shfl_xor_sync(0xffffffff, f.y, o);
    f.z += __shfl_xor_sync(0xffffffff, f.z, o);
  }
  int i = members[warp * 8 + u];
  if (qq == 0 && i >= 0) { force[3*i] = f.x; force[3*i+1] = f.y; force[3*i+2] = f.z; }
}

// As halfKernel, with the records of a target tile split over the K warps of
// a block, summed in shared memory.
template <int K>
__global__ void splitKernel(int T, const int *rowStart, const int *recTile, const unsigned *recMask,
                            const float4 *xt, const int *tt, const int *members, const float *A,
                            const float *B, int nt, float3 L, float3 iL, float *force) {
  __shared__ float3 part[K][32];
  int tile = blockIdx.x, w = threadIdx.x / 32, lane = threadIdx.x % 32;
  int u = lane >> 2, qq = lane & 3;
  float4 pi = xt[tile * 8 + u];
  int ti = tt[tile * 8 + u];
  float3 xi = {pi.x, pi.y, pi.z};
  float3 f = {0, 0, 0};
  int end = rowStart[tile + 1];
  for (int r = rowStart[tile] + w; r < end; r += K) {
    int b = recTile[r];
    unsigned m = recMask[r];
    float4 pj = xt[b * 4 + qq];
    int tj = tt[b * 4 + qq];
    if (m >> (u * 4 + qq) & 1u)
      pairForce(xi, pi.w, ti, make_float3(pj.x, pj.y, pj.z), pj.w, tj, L, iL, A, B, nt, f);
  }
  for (int o = 1; o < 4; o <<= 1) {
    f.x += __shfl_xor_sync(0xffffffff, f.x, o);
    f.y += __shfl_xor_sync(0xffffffff, f.y, o);
    f.z += __shfl_xor_sync(0xffffffff, f.z, o);
  }
  part[w][lane] = f;
  __syncthreads();
  if (w == 0 && qq == 0) {
    for (int k = 1; k < K; ++k) { f.x += part[k][lane].x; f.y += part[k][lane].y; f.z += part[k][lane].z; }
    int i = members[tile * 8 + u];
    if (i >= 0) { force[3*i] = f.x; force[3*i+1] = f.y; force[3*i+2] = f.z; }
  }
}

// Prunes the rows of the outer matrix to the entries within `reach2` now,
// keeping their order: 32 lanes per particle, compacted with a ballot.
__global__ void pruneKernel(int n, int W, const int *counts, const int *index, const float *xs,
                            float3 L, float3 iL, float reach2, int Win, int *countsIn, int *indexIn) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  int i = t / 32, lane = t % 32;
  if (i >= n) return;
  float3 xi = {xs[3*i], xs[3*i+1], xs[3*i+2]};
  int c = counts[i], kept = 0;
  for (int base = 0; base < c; base += 32) {
    int e = base + lane;
    bool keep = false; int j = 0;
    if (e < c) {
      j = index[(size_t)i * W + e];
      if (j == i) keep = true;  // an excluded entry stays marked
      else {
        float dx = xi.x - xs[3*j], dy = xi.y - xs[3*j+1], dz = xi.z - xs[3*j+2];
        dx -= L.x * rintf(dx * iL.x); dy -= L.y * rintf(dy * iL.y); dz -= L.z * rintf(dz * iL.z);
        keep = dx*dx + dy*dy + dz*dz <= reach2;
      }
    }
    unsigned ballot = __ballot_sync(0xffffffff, keep);
    if (keep) indexIn[(size_t)i * Win + kept + __popc(ballot & ((1u << lane) - 1))] = j;
    kept += __popc(ballot);
  }
  if (lane == 0) countsIn[i] = kept;
}

__device__ __forceinline__ float3 pairF(float3 xi, float qi, int ti, float3 xj, float qj, int tj,
                                        float3 L, float3 iL, const float *A, const float *B, int nt) {
  float3 f = {0, 0, 0};
  pairForce(xi, qi, ti, xj, qj, tj, L, iL, A, B, nt, f);
  return f;
}

// Half list, 8 x 4: each pair once; the force on the source is summed over
// the 8 targets with shuffles and added with atomics (float, or fixed point
// in 64-bit integers at the scale 2^32 when FIXED).
template <int K, bool FIXED>
__global__ void halfListKernel(int T, const int *rowStart, const int *recTile, const unsigned *recMask,
                               const float4 *xt, const int *tt, const int *members, const float *A,
                               const float *B, int nt, float3 L, float3 iL, float *force,
                               unsigned long long *fixedForce) {
  int tile = blockIdx.x, w = threadIdx.x / 32, lane = threadIdx.x % 32;
  int u = lane >> 2, qq = lane & 3;
  float4 pi = xt[tile * 8 + u];
  int ti = tt[tile * 8 + u];
  float3 xi = {pi.x, pi.y, pi.z};
  float3 fi = {0, 0, 0};
  int end = rowStart[tile + 1];
  const float scale = 4294967296.0f;
  for (int r = rowStart[tile] + w; r < end; r += K) {
    int b = recTile[r];
    unsigned m = recMask[r];
    float4 pj = xt[b * 4 + qq];
    int tj = tt[b * 4 + qq];
    float3 f = {0, 0, 0};
    if (m >> (u * 4 + qq) & 1u)
      f = pairF(xi, pi.w, ti, make_float3(pj.x, pj.y, pj.z), pj.w, tj, L, iL, A, B, nt);
    fi.x += f.x; fi.y += f.y; fi.z += f.z;
    for (int o = 4; o < 32; o <<= 1) {
      f.x += __shfl_xor_sync(0xffffffff, f.x, o);
      f.y += __shfl_xor_sync(0xffffffff, f.y, o);
      f.z += __shfl_xor_sync(0xffffffff, f.z, o);
    }
    if (u == 0) {
      int j = members[b * 4 + qq];
      if (j >= 0) {
        if (FIXED) {
          atomicAdd(&fixedForce[3*j], (unsigned long long)(long long)llrintf(-f.x * scale));
          atomicAdd(&fixedForce[3*j+1], (unsigned long long)(long long)llrintf(-f.y * scale));
          atomicAdd(&fixedForce[3*j+2], (unsigned long long)(long long)llrintf(-f.z * scale));
        } else {
          atomicAdd(&force[3*j], -f.x); atomicAdd(&force[3*j+1], -f.y); atomicAdd(&force[3*j+2], -f.z);
        }
      }
    }
  }
  for (int o = 1; o < 4; o <<= 1) {
    fi.x += __shfl_xor_sync(0xffffffff, fi.x, o);
    fi.y += __shfl_xor_sync(0xffffffff, fi.y, o);
    fi.z += __shfl_xor_sync(0xffffffff, fi.z, o);
  }
  int i = members[tile * 8 + u];
  if (qq == 0 && i >= 0) {
    if (FIXED) {
      atomicAdd(&fixedForce[3*i], (unsigned long long)(long long)llrintf(fi.x * scale));
      atomicAdd(&fixedForce[3*i+1], (unsigned long long)(long long)llrintf(fi.y * scale));
      atomicAdd(&fixedForce[3*i+2], (unsigned long long)(long long)llrintf(fi.z * scale));
    } else {
      atomicAdd(&force[3*i], fi.x); atomicAdd(&force[3*i+1], fi.y); atomicAdd(&force[3*i+2], fi.z);
    }
  }
}

int main(int argc, char **argv) {
  if (argc > 2) REACH = atof(argv[2]);
  FILE *fp = fopen(argv[1], "rb");
  fread(&n, 4, 1, fp); fread(&ntypes, 4, 1, fp); fread(box, 8, 3, fp);
  x.resize(3 * n); q.resize(n); type.resize(n); ta.resize(ntypes * ntypes); tb.resize(ntypes * ntypes);
  fread(x.data(), 8, 3 * n, fp); fread(q.data(), 4, n, fp); fread(type.data(), 4, n, fp);
  fread(ta.data(), 4, ntypes * ntypes, fp); fread(tb.data(), 4, ntypes * ntypes, fp);
  exoff.resize(n + 1); fread(exoff.data(), 4, n + 1, fp);
  exl.resize(exoff[n]); fread(exl.data(), 4, exoff[n], fp);
  fclose(fp);
  for (int i = 0; i < 3 * n; ++i) { int k = i % 3; x[i] -= box[k] * std::floor(x[i] / box[k]); }

  // The spatial order of MDIR: cells of half the reach, x first, then by index.
  {
    int nc[3]; for (int k = 0; k < 3; ++k) nc[k] = std::max(1, (int)(box[k] / (REACH / 2)));
    std::vector<std::pair<long, int>> key(n);
    for (int i = 0; i < n; ++i) {
      int c[3]; for (int k = 0; k < 3; ++k) c[k] = std::min(nc[k] - 1, (int)(x[3*i+k] / box[k] * nc[k]));
      key[i] = {((long)c[2] * nc[1] + c[1]) * nc[0] + c[0], i};
    }
    std::sort(key.begin(), key.end());
    std::vector<int> perm(n), inv(n);
    for (int i = 0; i < n; ++i) { perm[i] = key[i].second; inv[perm[i]] = i; }
    std::vector<double> x2(3 * n); std::vector<float> q2(n); std::vector<int> t2(n);
    std::vector<int> off2(n + 1, 0), l2;
    for (int i = 0; i < n; ++i) {
      int o = perm[i];
      for (int k = 0; k < 3; ++k) x2[3*i+k] = x[3*o+k];
      q2[i] = q[o]; t2[i] = type[o];
      for (int e = exoff[o]; e < exoff[o+1]; ++e) l2.push_back(inv[exl[e]]);
      off2[i+1] = l2.size();
    }
    x = x2; q = q2; type = t2; exoff = off2; exl = l2;
  }

  // Cell list of width reach / 2 for the host searches.
  int nc[3]; for (int k = 0; k < 3; ++k) nc[k] = std::max(3, (int)(box[k] / (REACH / 2)));
  int ncells = nc[0] * nc[1] * nc[2];
  std::vector<std::vector<int>> cells(ncells);
  auto cellOf = [&](const double *p, int *c) { for (int k = 0; k < 3; ++k) c[k] = std::min(nc[k] - 1, (int)(p[k] / box[k] * nc[k])); };
  for (int i = 0; i < n; ++i) { int c[3]; cellOf(&x[3*i], c); cells[(c[2] * nc[1] + c[1]) * nc[0] + c[0]].push_back(i); }
  double R2 = (double)REACH * REACH;

  // The matrix, full, with excluded entries marked by the particle itself.
  std::vector<std::vector<int>> rows(n);
  size_t within = 0, listed = 0;
  for (int i = 0; i < n; ++i) {
    int c[3]; cellOf(&x[3*i], c);
    for (int dz = -2; dz <= 2; ++dz) for (int dy = -2; dy <= 2; ++dy) for (int dx = -2; dx <= 2; ++dx) {
      int cc = (((c[2]+dz+nc[2])%nc[2]) * nc[1] + (c[1]+dy+nc[1])%nc[1]) * nc[0] + (c[0]+dx+nc[0])%nc[0];
      for (int j : cells[cc]) if (j != i) {
        double d2 = dist2(i, j);
        if (d2 <= R2) { rows[i].push_back(excluded(i, j) ? i : j); ++listed; if (d2 < RC * RC) ++within; }
      }
    }
    std::sort(rows[i].begin(), rows[i].end());
  }
  int W = 0; for (auto &r : rows) W = std::max(W, (int)r.size());
  std::vector<int> counts(n), index((size_t)n * W, 0);
  for (int i = 0; i < n; ++i) { counts[i] = rows[i].size(); std::copy(rows[i].begin(), rows[i].end(), index.begin() + (size_t)i * W); }
  printf("matrix: width %d, entries %zu, within the cutoff %.1f%%\n", W, listed, 100.0 * within / listed);

  // Tiles: columns in x-y of spacing (8 / rho)^(1/3), sorted by z, cut into 8.
  double rho = n / (box[0] * box[1] * box[2]);
  double a = std::cbrt(8.0 / rho);
  int ncx = std::max(1, (int)(box[0] / a)), ncy = std::max(1, (int)(box[1] / a));
  std::vector<std::vector<int>> cols(ncx * ncy);
  for (int i = 0; i < n; ++i) {
    int cx = std::min(ncx - 1, (int)(x[3*i] / box[0] * ncx)), cy = std::min(ncy - 1, (int)(x[3*i+1] / box[1] * ncy));
    cols[cy * ncx + cx].push_back(i);
  }
  std::vector<int> members;
  std::vector<int> tileCol;
  for (int c = 0; c < ncx * ncy; ++c) {
    auto &v = cols[c];
    std::sort(v.begin(), v.end(), [&](int i, int j) { return x[3*i+2] < x[3*j+2]; });
    for (size_t s = 0; s < v.size(); s += 8) {
      for (size_t k = 0; k < 8; ++k) members.push_back(s + k < v.size() ? v[s + k] : -1);
      tileCol.push_back(c);
    }
  }
  int T = members.size() / 8;
  // Bounding boxes (in the unwrapped z of the column; x-y within the column).
  std::vector<double> lo(3 * T), hi(3 * T);
  for (int t = 0; t < T; ++t) for (int k = 0; k < 3; ++k) { lo[3*t+k] = 1e30; hi[3*t+k] = -1e30; }
  for (int t = 0; t < T; ++t) for (int u = 0; u < 8; ++u) { int i = members[8*t+u]; if (i < 0) continue;
    for (int k = 0; k < 3; ++k) { lo[3*t+k] = std::min(lo[3*t+k], x[3*i+k]); hi[3*t+k] = std::max(hi[3*t+k], x[3*i+k]); } }
  std::vector<int> rowStart(T + 1, 0), recTile; std::vector<unsigned long long> recMask;
  size_t tilePairs = 0, maskBits = 0;
  for (int A = 0; A < T; ++A) {
    for (int B = 0; B < T; ++B) {
      // Bounding-box distance in the minimum image.
      double s = 0;
      for (int k = 0; k < 3; ++k) {
        double cA = 0.5 * (lo[3*A+k] + hi[3*A+k]), cB = 0.5 * (lo[3*B+k] + hi[3*B+k]);
        double d = std::fabs(mi(cA - cB, box[k])) - 0.5 * (hi[3*A+k] - lo[3*A+k]) - 0.5 * (hi[3*B+k] - lo[3*B+k]);
        if (d > 0) s += d * d;
      }
      if (s > R2) continue;
      unsigned long long m = 0;
      for (int u = 0; u < 8; ++u) { int i = members[8*A+u]; if (i < 0) continue;
        for (int v = 0; v < 8; ++v) { int j = members[8*B+v]; if (j < 0 || j == i) continue;
          if (dist2(i, j) <= R2 && !excluded(i, j)) m |= 1ull << (u * 8 + v); } }
      if (m) { recTile.push_back(B); recMask.push_back(m); ++tilePairs; maskBits += __builtin_popcountll(m); }
    }
    rowStart[A + 1] = recTile.size();
  }
  printf("tiles: %d (%.1f%% padding), records %zu, %.1f pairs per record, %.1f%% of the 8x8 slots used\n",
         T, 100.0 * (8.0 * T - n) / (8.0 * T), tilePairs, (double)maskBits / tilePairs, 100.0 * maskBits / (64.0 * tilePairs));
  // The records of 8 x 4: each record split into its halves of sources.
  std::vector<int> row4(T + 1, 0), rec4; std::vector<unsigned> mask4;
  for (int A = 0; A < T; ++A) {
    for (int r = rowStart[A]; r < rowStart[A + 1]; ++r)
      for (int h = 0; h < 2; ++h) {
        unsigned m = 0;
        for (int u = 0; u < 8; ++u) for (int v = 0; v < 4; ++v)
          if (recMask[r] >> (u * 8 + 4 * h + v) & 1ull) m |= 1u << (u * 4 + v);
        if (m) { rec4.push_back(2 * recTile[r] + h); mask4.push_back(m); }
      }
    row4[A + 1] = rec4.size();
  }
  printf("8x4: records %zu, %.1f%% of the slots used\n", rec4.size(), 100.0 * maskBits / (32.0 * rec4.size()));
  // The half list: a record of tiles A <= B once, and of a tile with itself
  // only the pairs u < v.
  std::vector<int> rowH(T + 1, 0), recH; std::vector<unsigned> maskH;
  for (int A = 0; A < T; ++A) {
    for (int r = row4[A]; r < row4[A + 1]; ++r) {
      int B = rec4[r] / 2, h = rec4[r] % 2;
      if (B < A) continue;
      unsigned m = mask4[r];
      if (B == A)
        for (int u = 0; u < 8; ++u) for (int v = 0; v < 4; ++v) if (4 * h + v <= u) m &= ~(1u << (u * 4 + v));
      if (m) { recH.push_back(rec4[r]); maskH.push_back(m); }
    }
    rowH[A + 1] = recH.size();
  }
  printf("half list: records %zu\n", recH.size());
  std::vector<float4> xt(8 * T); std::vector<int> tt(8 * T);
  for (int s = 0; s < 8 * T; ++s) { int i = members[s];
    if (i < 0) { xt[s] = make_float4(-1e4f, -1e4f, -1e4f, 0.f); tt[s] = 0; }
    else { xt[s] = make_float4(x[3*i], x[3*i+1], x[3*i+2], q[i]); tt[s] = type[i]; } }

  std::vector<float> xs(3 * n); for (int i = 0; i < 3 * n; ++i) xs[i] = x[i];
  float *dx, *dq, *dA, *dB, *df1, *df2; int *dt, *dcounts, *dindex, *drow, *drec, *dtt, *dmem; unsigned long long *dmask; float4 *dxt;
  CK(cudaMalloc(&dx, 12 * n)); CK(cudaMalloc(&dq, 4 * n)); CK(cudaMalloc(&dt, 4 * n));
  CK(cudaMalloc(&dA, 4 * ta.size())); CK(cudaMalloc(&dB, 4 * tb.size()));
  CK(cudaMalloc(&dcounts, 4 * n)); CK(cudaMalloc(&dindex, 4 * index.size()));
  CK(cudaMalloc(&df1, 12 * n)); CK(cudaMalloc(&df2, 12 * n));
  CK(cudaMalloc(&drow, 4 * (T + 1))); CK(cudaMalloc(&drec, 4 * recTile.size())); CK(cudaMalloc(&dmask, 8 * recMask.size()));
  CK(cudaMalloc(&dxt, 16 * xt.size())); CK(cudaMalloc(&dtt, 4 * tt.size())); CK(cudaMalloc(&dmem, 4 * members.size()));
  cudaMemcpy(dx, xs.data(), 12 * n, cudaMemcpyHostToDevice); cudaMemcpy(dq, q.data(), 4 * n, cudaMemcpyHostToDevice);
  cudaMemcpy(dt, type.data(), 4 * n, cudaMemcpyHostToDevice);
  cudaMemcpy(dA, ta.data(), 4 * ta.size(), cudaMemcpyHostToDevice); cudaMemcpy(dB, tb.data(), 4 * tb.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(dcounts, counts.data(), 4 * n, cudaMemcpyHostToDevice); cudaMemcpy(dindex, index.data(), 4 * index.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(drow, rowStart.data(), 4 * (T + 1), cudaMemcpyHostToDevice); cudaMemcpy(drec, recTile.data(), 4 * recTile.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(dmask, recMask.data(), 8 * recMask.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(dxt, xt.data(), 16 * xt.size(), cudaMemcpyHostToDevice); cudaMemcpy(dtt, tt.data(), 4 * tt.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(dmem, members.data(), 4 * members.size(), cudaMemcpyHostToDevice);
  int *drow4, *drec4; unsigned *dmask4;
  CK(cudaMalloc(&drow4, 4 * (T + 1))); CK(cudaMalloc(&drec4, 4 * rec4.size())); CK(cudaMalloc(&dmask4, 4 * mask4.size()));
  cudaMemcpy(drow4, row4.data(), 4 * (T + 1), cudaMemcpyHostToDevice); cudaMemcpy(drec4, rec4.data(), 4 * rec4.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(dmask4, mask4.data(), 4 * mask4.size(), cudaMemcpyHostToDevice);
  float3 L = make_float3(box[0], box[1], box[2]), iL = make_float3(1 / box[0], 1 / box[1], 1 / box[2]);

  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  const int reps = 1000;
  auto time = [&](auto launch) {
    launch(); CK(cudaDeviceSynchronize());
    cudaEventRecord(e0); for (int r = 0; r < reps; ++r) launch(); cudaEventRecord(e1); CK(cudaEventSynchronize(e1));
    float ms; cudaEventElapsedTime(&ms, e0, e1); return 1000.0 * ms / reps;
  };
  double tm = time([&] { matrixKernel<<<(16 * n + 127) / 128, 128>>>(n, W, dcounts, dindex, dx, dq, dt, dA, dB, ntypes, L, iL, df1); });
  double tl = time([&] { tileKernel<<<(32 * T + 127) / 128, 128>>>(T, drow, drec, dmask, dxt, dtt, dmem, dA, dB, ntypes, L, iL, df2); });
  double th = time([&] { halfKernel<<<(32 * T + 127) / 128, 128>>>(T, drow4, drec4, dmask4, dxt, dtt, dmem, dA, dB, ntypes, L, iL, df2); });
  printf("8x4 tiles %.1f us\n", th);
  printf("csv,pairs,tiles 8x4 full,%.2f,,%.2f\n", REACH, th);
  double t2 = time([&] { splitKernel<2><<<T, 64>>>(T, drow4, drec4, dmask4, dxt, dtt, dmem, dA, dB, ntypes, L, iL, df2); });
  double t4 = time([&] { splitKernel<4><<<T, 128>>>(T, drow4, drec4, dmask4, dxt, dtt, dmem, dA, dB, ntypes, L, iL, df2); });
  double t8 = time([&] { splitKernel<8><<<T, 256>>>(T, drow4, drec4, dmask4, dxt, dtt, dmem, dA, dB, ntypes, L, iL, df2); });
  printf("8x4 split over 2/4/8 warps: %.1f / %.1f / %.1f us\n", t2, t4, t8);
  printf("csv,pairs,tiles 8x4 full over 4 warps,%.2f,,%.2f\n", REACH, t4);
  for (float rin : {8.25f, 8.5f, 9.0f}) {
    int *dci, *dii; CK(cudaMalloc(&dci, 4 * n)); CK(cudaMalloc(&dii, 4 * index.size()));
    double tp = time([&] { pruneKernel<<<(32 * n + 127) / 128, 128>>>(n, W, dcounts, dindex, dx, L, iL, rin * rin, W, dci, dii); });
    double tk = time([&] { matrixKernel<<<(16 * n + 127) / 128, 128>>>(n, W, dci, dii, dx, dq, dt, dA, dB, ntypes, L, iL, df2); });
    printf("inner reach %.2f: prune %.1f us, loop over the pruned matrix %.1f us\n", rin, tp, tk);
    printf("csv,prune,prune from outer,%.2f,%.2f,%.2f\ncsv,prune,loop over pruned,%.2f,%.2f,%.2f\n", REACH, rin, tp, REACH, rin, tk);
    cudaFree(dci); cudaFree(dii);
  }
  int *drowH, *drecH; unsigned *dmaskH; unsigned long long *dfix;
  CK(cudaMalloc(&drowH, 4 * (T + 1))); CK(cudaMalloc(&drecH, 4 * recH.size())); CK(cudaMalloc(&dmaskH, 4 * maskH.size()));
  CK(cudaMalloc(&dfix, 24 * n));
  cudaMemcpy(drowH, rowH.data(), 4 * (T + 1), cudaMemcpyHostToDevice); cudaMemcpy(drecH, recH.data(), 4 * recH.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(dmaskH, maskH.data(), 4 * maskH.size(), cudaMemcpyHostToDevice);
  double hf = time([&] { cudaMemsetAsync(df2, 0, 12 * n); halfListKernel<4, false><<<T, 128>>>(T, drowH, drecH, dmaskH, dxt, dtt, dmem, dA, dB, ntypes, L, iL, df2, dfix); });
  std::vector<float> fh(3 * n); cudaMemcpy(fh.data(), df2, 12 * n, cudaMemcpyDeviceToHost);
  double hx = time([&] { cudaMemsetAsync(dfix, 0, 24 * n); halfListKernel<4, true><<<T, 128>>>(T, drowH, drecH, dmaskH, dxt, dtt, dmem, dA, dB, ntypes, L, iL, df2, dfix); });
  printf("half list (with the clearing): float atomics %.1f us, fixed point %.1f us\n", hf, hx);
  printf("csv,pairs,tiles 8x4 half f32 atomics,%.2f,,%.2f\ncsv,pairs,tiles 8x4 half fixed point,%.2f,,%.2f\n", REACH, hf, REACH, hx);
  std::vector<float> f1(3 * n), f2(3 * n);
  cudaMemcpy(f1.data(), df1, 12 * n, cudaMemcpyDeviceToHost); f2 = fh;
  double num = 0, den = 0;
  for (int i = 0; i < 3 * n; ++i) { num += (f1[i] - f2[i]) * (double)(f1[i] - f2[i]); den += (double)f1[i] * f1[i]; }
  printf("reach %.1f: matrix %.1f us, tiles %.1f us, relative difference of the forces %.2e\n", REACH, tm, tl, std::sqrt(num / den));
  printf("csv,pairs,matrix 16 lanes,%.2f,,%.2f\ncsv,pairs,tiles 8x8 full,%.2f,,%.2f\n", REACH, tm, REACH, tl);
  return 0;
}
