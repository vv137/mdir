// Prototype (2026-10-01, D82): the loop over pairs with the cluster pair
// list of [Pall2013] as its GPU layout has it, against the neighbor matrix
// of MDIR (16 lanes per particle). Lennard-Jones from type tables and the
// direct sum of PME, in f32; the lists are built on the host and only the
// kernels are timed.
//
// Particles are binned in columns in x-y and sorted by z; each column is cut
// into super-clusters of 64, and a super-cluster into 8 clusters of 8 by
// halving in x, then y, then z. The list of a super-cluster holds the
// clusters j within the reach of one of its clusters i, each pair of
// clusters once (i <= j; within one cluster only u < v): a half list. An
// entry carries a mask of 8 bits for the clusters i that it pairs with, and
// masks of the excluded pairs where there are any.
//
// The kernel is a warp per super-cluster: lane = 8 h + u holds particle u of
// each of the 8 clusters i and takes the particles h and h + 4 of each
// cluster j. The forces on i stay in registers over the whole list; those on
// j are summed over u with shuffles after the 8 clusters i and added with
// atomics, once per particle of j per entry.
//
//   supercluster <system.bin> [reach]
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
#ifndef NO_IMAGE
  dx -= L.x * rintf(dx * iL.x); dy -= L.y * rintf(dy * iL.y); dz -= L.z * rintf(dz * iL.z);
#endif
  float r2 = dx*dx + dy*dy + dz*dz;
  if (r2 < RC * RC) {
    float ri = rsqrtf(r2), r2i = ri * ri, r6i = r2i * r2i * r2i;
    float a = A[ti * nt + tj], b = B[ti * nt + tj];
    float r = r2 * ri;
    float fs = (12.f * a * r6i - 6.f * b) * r6i * r2i;
    float br = BETA * r;
#ifdef FAST_ERFC
    // erfc(x) = t (a1 + t (a2 + t (a3 + t (a4 + t a5)))) exp(-x^2), t = 1 / (1 + p x)
    // (Abramowitz and Stegun 7.1.26); the exponential is that of the force.
    float ex = __expf(-br * br);
    float t = __frcp_rn(1.f + 0.3275911f * br);
    float poly = t * (0.254829592f + t * (-0.284496736f + t * (1.421413741f + t * (-1.453152027f + t * 1.061405429f))));
    fs += qi * qj * (poly * ri + 1.1283792f * BETA) * ex * r2i;
#else
    fs += qi * qj * (erfcf(br) * ri + 1.1283792f * BETA * __expf(-br * br)) * r2i;
#endif
    f.x += fs * dx; f.y += fs * dy; f.z += fs * dz;
  }
}

// The matrix of MDIR: 16 lanes per particle, full list.
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

// The cluster pair list: a warp per super-cluster. `slots` holds positions
// and charges by slot (8 per cluster); padding slots are far away and
// uncharged. `force` is by slot.
__global__ void superKernel(int S, const int *rowStart, const int *entCluster,
                            const unsigned char *entImask, const int *entMasks,
                            const unsigned long long *masks, const float4 *slots, const int *slotType,
                            const float *A, const float *B, int nt, float3 L, float3 iL, float *force) {
  int s = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  if (s >= S) return;
  int lane = threadIdx.x % 32, u = lane & 7, h = lane >> 3;
  float3 xi[8]; float qi[8]; int ti[8]; float3 fi[8];
#pragma unroll
  for (int c = 0; c < 8; ++c) {
    float4 p = slots[(8 * s + c) * 8 + u];
    xi[c] = make_float3(p.x, p.y, p.z); qi[c] = p.w; ti[c] = slotType[(8 * s + c) * 8 + u];
    fi[c] = make_float3(0, 0, 0);
  }
  for (int e = rowStart[s]; e < rowStart[s + 1]; ++e) {
    int cj = entCluster[e];
    unsigned im = entImask[e];
    int mo = entMasks[e];
    float4 p0 = slots[cj * 8 + h], p1 = slots[cj * 8 + h + 4];
    int t0 = slotType[cj * 8 + h], t1 = slotType[cj * 8 + h + 4];
    float3 x0 = make_float3(p0.x, p0.y, p0.z), x1 = make_float3(p1.x, p1.y, p1.z);
    float3 f0 = {0, 0, 0}, f1 = {0, 0, 0};
#pragma unroll
    for (int c = 0; c < 8; ++c) {
      if (!(im >> c & 1u)) continue;
      unsigned long long m = mo < 0 ? ~0ull : masks[mo * 8 + c];
      if (m >> (u * 8 + h) & 1ull) {
        float3 f = {0, 0, 0};
        pairForce(xi[c], qi[c], ti[c], x0, p0.w, t0, L, iL, A, B, nt, f);
        fi[c].x += f.x; fi[c].y += f.y; fi[c].z += f.z;
        f0.x -= f.x; f0.y -= f.y; f0.z -= f.z;
      }
      if (m >> (u * 8 + h + 4) & 1ull) {
        float3 f = {0, 0, 0};
        pairForce(xi[c], qi[c], ti[c], x1, p1.w, t1, L, iL, A, B, nt, f);
        fi[c].x += f.x; fi[c].y += f.y; fi[c].z += f.z;
        f1.x -= f.x; f1.y -= f.y; f1.z -= f.z;
      }
    }
    for (int o = 1; o < 8; o <<= 1) {
      f0.x += __shfl_xor_sync(0xffffffff, f0.x, o); f0.y += __shfl_xor_sync(0xffffffff, f0.y, o);
      f0.z += __shfl_xor_sync(0xffffffff, f0.z, o);
      f1.x += __shfl_xor_sync(0xffffffff, f1.x, o); f1.y += __shfl_xor_sync(0xffffffff, f1.y, o);
      f1.z += __shfl_xor_sync(0xffffffff, f1.z, o);
    }
    if (u == 0) {
      float *a = force + 3 * (cj * 8 + h), *b = force + 3 * (cj * 8 + h + 4);
      atomicAdd(a, f0.x); atomicAdd(a + 1, f0.y); atomicAdd(a + 2, f0.z);
      atomicAdd(b, f1.x); atomicAdd(b + 1, f1.y); atomicAdd(b + 2, f1.z);
    }
  }
#pragma unroll
  for (int c = 0; c < 8; ++c) {
    for (int o = 8; o < 32; o <<= 1) {
      fi[c].x += __shfl_xor_sync(0xffffffff, fi[c].x, o);
      fi[c].y += __shfl_xor_sync(0xffffffff, fi[c].y, o);
      fi[c].z += __shfl_xor_sync(0xffffffff, fi[c].z, o);
    }
    if (h == 0) {
      float *a = force + 3 * ((8 * s + c) * 8 + u);
      atomicAdd(a, fi[c].x); atomicAdd(a + 1, fi[c].y); atomicAdd(a + 2, fi[c].z);
    }
  }
}

// Groups of G = 32 particles, as the neighbor list of [SalomonFerrer2013]
// describes (groups of 16 or 32 that share the atoms within the extended
// cutoff of any of them, from a half shell): a warp per group, lane k holds
// particle k of the group. The entries of the list are particles j, 32 at
// a time; they turn around the warp by one lane per step, with the force on
// them, so that each lane meets each j once, and return to their lanes
// after 32 steps, where the force on them is added with an atomic. Each
// entry carries a mask of the lanes it pairs with (exclusions, the pairs of
// the group with itself once, empty lanes).
__global__ void groupKernel(int Gn, const int *rowStart, const int *entSlot, const unsigned *entMask,
                            const float4 *slots, const int *slotType, const float *A, const float *B,
                            int nt, float3 L, float3 iL, float *force) {
  int g = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  if (g >= Gn) return;
  int lane = threadIdx.x % 32;
  float4 pi = slots[32 * g + lane];
  float3 xi = make_float3(pi.x, pi.y, pi.z); int ti = slotType[32 * g + lane];
  float3 fi = {0, 0, 0};
  int end = rowStart[g + 1];
  for (int e0 = rowStart[g]; e0 < end; e0 += 32) {
    int e = e0 + lane;
    int js = e < end ? entSlot[e] : -1;
    unsigned m = e < end ? entMask[e] : 0u;
    float4 pj = js >= 0 ? slots[js] : make_float4(0, 0, 0, 0);
    int tj = js >= 0 ? slotType[js] : 0;
    float3 fj = {0, 0, 0};
    int src = (lane + 31) & 31;
    for (int step = 0; step < 32; ++step) {
      if (m >> lane & 1u) {
        float3 f = {0, 0, 0};
        pairForce(xi, pi.w, ti, make_float3(pj.x, pj.y, pj.z), pj.w, tj, L, iL, A, B, nt, f);
        fi.x += f.x; fi.y += f.y; fi.z += f.z;
        fj.x -= f.x; fj.y -= f.y; fj.z -= f.z;
      }
      pj.x = __shfl_sync(0xffffffff, pj.x, src); pj.y = __shfl_sync(0xffffffff, pj.y, src);
      pj.z = __shfl_sync(0xffffffff, pj.z, src); pj.w = __shfl_sync(0xffffffff, pj.w, src);
      tj = __shfl_sync(0xffffffff, tj, src); m = __shfl_sync(0xffffffff, m, src);
      fj.x = __shfl_sync(0xffffffff, fj.x, src); fj.y = __shfl_sync(0xffffffff, fj.y, src);
      fj.z = __shfl_sync(0xffffffff, fj.z, src);
    }
    int jb = __shfl_sync(0xffffffff, js, lane);
    if (jb >= 0) { float *a = force + 3 * jb; atomicAdd(a, fj.x); atomicAdd(a + 1, fj.y); atomicAdd(a + 2, fj.z); }
  }
  float *a = force + 3 * (32 * g + lane);
  atomicAdd(a, fi.x); atomicAdd(a + 1, fi.y); atomicAdd(a + 2, fi.z);
}

// Groups of 16, as pmemd.cuda arranges them (lanes 0-15 and 16-31 hold the
// same 16 particles i; each lane takes one of 32 particles j of a chunk,
// which turn within their half-warp, 16 steps a chunk): entries carry a
// mask of 16 bits over the particles i.
__global__ void group16Kernel(int Gn, const int *rowStart, const int *entSlot, const unsigned *entMask,
                              const float4 *slots, const int *slotType, const float *A, const float *B,
                              int nt, float3 L, float3 iL, float *force) {
  int g = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  if (g >= Gn) return;
  int lane = threadIdx.x % 32, u = lane & 15, half = lane & 16;
  float4 pi = slots[16 * g + u];
  float3 xi = make_float3(pi.x, pi.y, pi.z); int ti = slotType[16 * g + u];
  float3 fi = {0, 0, 0};
  int end = rowStart[g + 1];
  int src = half | ((u + 15) & 15);
  for (int e0 = rowStart[g]; e0 < end; e0 += 32) {
    int e = e0 + lane;
    int js = e < end ? entSlot[e] : -1;
    unsigned m = e < end ? entMask[e] : 0u;
    float4 pj = js >= 0 ? slots[js] : make_float4(0, 0, 0, 0);
    int tj = js >= 0 ? slotType[js] : 0;
    float3 fj = {0, 0, 0};
#pragma unroll 2
    for (int step = 0; step < 16; ++step) {
      if (m >> u & 1u) {
        float3 f = {0, 0, 0};
        pairForce(xi, pi.w, ti, make_float3(pj.x, pj.y, pj.z), pj.w, tj, L, iL, A, B, nt, f);
        fi.x += f.x; fi.y += f.y; fi.z += f.z;
        fj.x -= f.x; fj.y -= f.y; fj.z -= f.z;
      }
      pj.x = __shfl_sync(0xffffffff, pj.x, src); pj.y = __shfl_sync(0xffffffff, pj.y, src);
      pj.z = __shfl_sync(0xffffffff, pj.z, src); pj.w = __shfl_sync(0xffffffff, pj.w, src);
      tj = __shfl_sync(0xffffffff, tj, src); m = __shfl_sync(0xffffffff, m, src);
      fj.x = __shfl_sync(0xffffffff, fj.x, src); fj.y = __shfl_sync(0xffffffff, fj.y, src);
      fj.z = __shfl_sync(0xffffffff, fj.z, src);
    }
    if (js >= 0) { float *a = force + 3 * js; atomicAdd(a, fj.x); atomicAdd(a + 1, fj.y); atomicAdd(a + 2, fj.z); }
  }
  fi.x += __shfl_xor_sync(0xffffffff, fi.x, 16);
  fi.y += __shfl_xor_sync(0xffffffff, fi.y, 16);
  fi.z += __shfl_xor_sync(0xffffffff, fi.z, 16);
  if (!half) { float *a = force + 3 * (16 * g + u); atomicAdd(a, fi.x); atomicAdd(a + 1, fi.y); atomicAdd(a + 2, fi.z); }
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
  double R2 = (double)REACH * REACH;

  // The super-clusters: columns of about the width of 64 particles, sorted
  // by z and cut into 64, each halved in x, y, then z into 8 clusters of 8.
  double rho = n / (box[0] * box[1] * box[2]);
  double side = std::cbrt(64.0 / rho);
  int ncx = std::max(1, (int)(box[0] / side)), ncy = std::max(1, (int)(box[1] / side));
  std::vector<std::vector<int>> cols(ncx * ncy);
  for (int i = 0; i < n; ++i) {
    int cx = std::min(ncx - 1, (int)(x[3*i] / box[0] * ncx)), cy = std::min(ncy - 1, (int)(x[3*i+1] / box[1] * ncy));
    cols[cy * ncx + cx].push_back(i);
  }
  std::vector<int> members;  // by slot, -1 for padding
  for (auto &v : cols) {
    std::sort(v.begin(), v.end(), [&](int i, int j) { return x[3*i+2] < x[3*j+2]; });
    for (size_t s = 0; s < v.size(); s += 64) {
      std::vector<int> g(v.begin() + s, v.begin() + std::min(v.size(), s + 64));
      g.resize(64, -1);
      // Halve by coordinate k the range [a, a + len), padding last.
      auto halve = [&](int a, int len, int k) {
        std::sort(g.begin() + a, g.begin() + a + len, [&](int i, int j) {
          if (i < 0 || j < 0) return j < 0 && i >= 0;
          return x[3*i+k] < x[3*j+k]; });
      };
      halve(0, 64, 0);
      for (int a = 0; a < 64; a += 32) halve(a, 32, 1);
      for (int a = 0; a < 64; a += 16) halve(a, 16, 2);
      members.insert(members.end(), g.begin(), g.end());
    }
  }
  int C = members.size() / 8, S = C / 8;
  // Bounding boxes of the clusters.
  std::vector<double> lo(3 * C, 1e30), hi(3 * C, -1e30);
  for (int c = 0; c < C; ++c) for (int u = 0; u < 8; ++u) { int i = members[8*c+u]; if (i < 0) continue;
    for (int k = 0; k < 3; ++k) { lo[3*c+k] = std::min(lo[3*c+k], x[3*i+k]); hi[3*c+k] = std::max(hi[3*c+k], x[3*i+k]); } }
  auto bbDist2 = [&](int a, int b) {
    double s2 = 0;
    for (int k = 0; k < 3; ++k) {
      double ca = 0.5 * (lo[3*a+k] + hi[3*a+k]), cb = 0.5 * (lo[3*b+k] + hi[3*b+k]);
      double d = std::fabs(mi(ca - cb, box[k])) - 0.5 * (hi[3*a+k] - lo[3*a+k]) - 0.5 * (hi[3*b+k] - lo[3*b+k]);
      if (d > 0) s2 += d * d;
    }
    return s2;
  };
  // A grid of the clusters by center, of width the reach, for the search.
  int g[3]; for (int k = 0; k < 3; ++k) g[k] = std::max(1, (int)(box[k] / REACH));
  std::vector<std::vector<int>> grid(g[0] * g[1] * g[2]);
  auto gcell = [&](int c, int *gc) { for (int k = 0; k < 3; ++k) {
    double m = 0.5 * (lo[3*c+k] + hi[3*c+k]); m -= box[k] * std::floor(m / box[k]);
    gc[k] = std::min(g[k] - 1, (int)(m / box[k] * g[k])); } };
  for (int c = 0; c < C; ++c) { if (lo[3*c] > hi[3*c]) continue; int gc[3]; gcell(c, gc); grid[(gc[2] * g[1] + gc[1]) * g[0] + gc[0]].push_back(c); }
  // The list, and its statistics.
  std::vector<int> rowStart(S + 1, 0), entCluster, entMasks; std::vector<unsigned char> entImask;
  std::vector<unsigned long long> masks;
  size_t slotsUsed = 0, pairsWithin = 0, pairsReach = 0, clusterPairs = 0;
  int reachCells[3]; for (int k = 0; k < 3; ++k) reachCells[k] = std::min(g[k] / 2, 2);
  for (int s = 0; s < S; ++s) {
    std::vector<int> cand;
    for (int ci = 8 * s; ci < 8 * s + 8; ++ci) {
      if (lo[3*ci] > hi[3*ci]) continue;
      int gc[3]; gcell(ci, gc);
      for (int dz = -reachCells[2]; dz <= reachCells[2]; ++dz) for (int dy = -reachCells[1]; dy <= reachCells[1]; ++dy)
        for (int dx = -reachCells[0]; dx <= reachCells[0]; ++dx) {
          int cc = (((gc[2]+dz+g[2])%g[2]) * g[1] + (gc[1]+dy+g[1])%g[1]) * g[0] + (gc[0]+dx+g[0])%g[0];
          for (int cj : grid[cc]) if (cj >= ci) cand.push_back(cj);
        }
    }
    std::sort(cand.begin(), cand.end()); cand.erase(std::unique(cand.begin(), cand.end()), cand.end());
    for (int cj : cand) {
      unsigned im = 0; unsigned long long m[8]; bool special = false;
      for (int c = 0; c < 8; ++c) {
        int ci = 8 * s + c; m[c] = 0;
        if (ci > cj || lo[3*ci] > hi[3*ci] || bbDist2(ci, cj) > R2) continue;
        unsigned long long allowed = 0; bool any = false;
        for (int u = 0; u < 8; ++u) { int i = members[8*ci+u];
          for (int v = 0; v < 8; ++v) { int j = members[8*cj+v];
            bool ok = true;
            if (ci == cj && v <= u) ok = false;
            else if (i >= 0 && j >= 0 && excluded(i, j)) ok = false;
            if (ok) allowed |= 1ull << (u * 8 + v);
            if (ok && i >= 0 && j >= 0) { double d2 = dist2(i, j); if (d2 <= R2) { any = true; ++pairsReach; if (d2 < RC * RC) ++pairsWithin; } }
          } }
        if (!any) continue;
        im |= 1u << c; m[c] = allowed; ++clusterPairs;
        if (allowed != ~0ull) special = true;
      }
      if (!im) continue;
      entCluster.push_back(cj); entImask.push_back(im);
      if (special) { entMasks.push_back(masks.size() / 8); for (int c = 0; c < 8; ++c) masks.push_back(m[c] ? m[c] : ~0ull); }
      else entMasks.push_back(-1);
    }
    rowStart[s + 1] = entCluster.size();
  }
  slotsUsed = pairsWithin;
  printf("clusters %d, super-clusters %d (%.1f%% padding), entries %zu, cluster pairs %zu (%.1f per entry), masks %zu\n",
         C, S, 100.0 * (8.0 * C - n) / (8.0 * C), entCluster.size(), clusterPairs, (double)clusterPairs / entCluster.size(), masks.size() / 8);
  printf("slots %zu; pairs within the reach %.1f%%, within the cutoff %.1f%% of the slots\n",
         clusterPairs * 64, 100.0 * pairsReach / (clusterPairs * 64.0), 100.0 * slotsUsed / (clusterPairs * 64.0));


  // The groups of 32: four clusters of a super-cluster in a row. The list of
  // a group holds the particles j of the same or a later group within the
  // reach of one of its particles; within the group, only later lanes.
  int Gn = C / 4;
  std::vector<int> gStart(Gn + 1, 0), gEnt; std::vector<unsigned> gMask;
  size_t gSlotsWithin = 0;
  {
    // Candidate groups by the clusters that the super-cluster lists.
    for (int gi = 0; gi < Gn; ++gi) {
      int s = gi / 2;
      std::vector<int> cand;
      for (int e = rowStart[s]; e < rowStart[s + 1]; ++e) {
        unsigned im = entImask[e];
        unsigned mine = (gi % 2 == 0) ? 0x0Fu : 0xF0u;
        if (im & mine) cand.push_back(entCluster[e]);
      }
      // Clusters of the super-cluster's own other half pair too.
      std::vector<int> js;
      for (int cj : cand) for (int v = 0; v < 8; ++v) {
        int slot = cj * 8 + v, j = members[slot]; if (j < 0) continue;
        if (slot / 32 < gi) continue;
        unsigned mk = 0;
        for (int k = 0; k < 32; ++k) { int islot = 32 * gi + k, i = members[islot]; if (i < 0) continue;
          if (slot / 32 == gi && slot <= islot) continue;
          if (excluded(i, j)) continue;
          if (dist2(i, j) <= R2) mk |= 1u << k; }
        if (!mk) continue;
        // Lanes beyond the reach are allowed too: the kernel tests the cutoff.
        unsigned all = 0;
        for (int k = 0; k < 32; ++k) { int islot = 32 * gi + k, i = members[islot]; if (i < 0) continue;
          if (slot / 32 == gi && slot <= islot) continue;
          if (excluded(i, j)) continue;
          all |= 1u << k; if (dist2(i, j) < RC * RC) ++gSlotsWithin; }
        gEnt.push_back(slot); gMask.push_back(all);
      }
      gStart[gi + 1] = gEnt.size();
    }
  }
  size_t gChunks = 0; for (int gi = 0; gi < Gn; ++gi) gChunks += (gStart[gi+1] - gStart[gi] + 31) / 32;
  printf("groups %d, entries %zu (%.1f per group), slots %zu, within the cutoff %.1f%% of the slots\n",
         Gn, gEnt.size(), (double)gEnt.size() / Gn, gChunks * 1024, 100.0 * gSlotsWithin / (gChunks * 1024.0));

  // The groups of 16: two clusters in a row.
  int G16 = C / 2;
  std::vector<int> hStart(G16 + 1, 0), hEnt; std::vector<unsigned> hMask;
  size_t hWithin = 0, hChunks = 0;
  for (int gi = 0; gi < G16; ++gi) {
    int s = gi / 4;
    unsigned mine = 0x3u << (2 * (gi % 4));
    std::vector<int> cand;
    for (int e = rowStart[s]; e < rowStart[s + 1]; ++e)
      if (entImask[e] & mine) cand.push_back(entCluster[e]);
    for (int cj : cand) for (int v = 0; v < 8; ++v) {
      int slot = cj * 8 + v, j = members[slot]; if (j < 0) continue;
      if (slot / 16 < gi) continue;
      unsigned mk = 0, all = 0;
      for (int k = 0; k < 16; ++k) { int islot = 16 * gi + k, i = members[islot]; if (i < 0) continue;
        if (slot / 16 == gi && slot <= islot) continue;
        if (excluded(i, j)) continue;
        all |= 1u << k; double d2 = dist2(i, j);
        if (d2 <= R2) mk |= 1u << k;
        if (d2 < RC * RC) ++hWithin; }
      if (!mk) continue;
      hEnt.push_back(slot); hMask.push_back(all);
    }
    hStart[gi + 1] = hEnt.size();
    hChunks += (hStart[gi + 1] - hStart[gi] + 31) / 32;
  }
  printf("groups of 16 %d, entries %zu (%.1f per group), slots %zu, within the cutoff %.1f%% of the slots\n",
         G16, hEnt.size(), (double)hEnt.size() / G16, hChunks * 512, 100.0 * hWithin / (hChunks * 512.0));
  // The matrix, full, in the order of the slots, for the reference.
  std::vector<int> order; for (int i : members) if (i >= 0) order.push_back(i);
  std::vector<int> place(n); for (int p = 0; p < n; ++p) place[order[p]] = p;
  int nc[3]; for (int k = 0; k < 3; ++k) nc[k] = std::max(3, (int)(box[k] / (REACH / 2)));
  std::vector<std::vector<int>> cells(nc[0] * nc[1] * nc[2]);
  auto cellOf = [&](const double *p, int *c) { for (int k = 0; k < 3; ++k) c[k] = std::min(nc[k] - 1, (int)(p[k] / box[k] * nc[k])); };
  for (int i = 0; i < n; ++i) { int c[3]; cellOf(&x[3*i], c); cells[(c[2] * nc[1] + c[1]) * nc[0] + c[0]].push_back(i); }
  std::vector<std::vector<int>> rows(n);
  size_t listed = 0;
  for (int p = 0; p < n; ++p) {
    int i = order[p]; int c[3]; cellOf(&x[3*i], c);
    for (int dz = -2; dz <= 2; ++dz) for (int dy = -2; dy <= 2; ++dy) for (int dx = -2; dx <= 2; ++dx) {
      int cc = (((c[2]+dz+nc[2])%nc[2]) * nc[1] + (c[1]+dy+nc[1])%nc[1]) * nc[0] + (c[0]+dx+nc[0])%nc[0];
      for (int j : cells[cc]) if (j != i && dist2(i, j) <= R2) { rows[p].push_back(excluded(i, j) ? p : place[j]); ++listed; }
    }
    std::sort(rows[p].begin(), rows[p].end());
  }
  int W = 0; for (auto &r : rows) W = std::max(W, (int)r.size());
  std::vector<int> counts(n), index((size_t)n * W, 0);
  for (int p = 0; p < n; ++p) { counts[p] = rows[p].size(); std::copy(rows[p].begin(), rows[p].end(), index.begin() + (size_t)p * W); }
  printf("matrix: width %d, entries %zu\n", W, listed);

  std::vector<float> xs(3 * n), qs(n); std::vector<int> ts(n);
  for (int p = 0; p < n; ++p) { int i = order[p]; for (int k = 0; k < 3; ++k) xs[3*p+k] = x[3*i+k]; qs[p] = q[i]; ts[p] = type[i]; }
  std::vector<float4> slots(8 * C); std::vector<int> slotType(8 * C);
  for (int s = 0; s < 8 * C; ++s) { int i = members[s];
    if (i < 0) { slots[s] = make_float4(-1e4f, -1e4f, -1e4f, 0.f); slotType[s] = 0; }
    else { slots[s] = make_float4(x[3*i], x[3*i+1], x[3*i+2], q[i]); slotType[s] = type[i]; } }

  auto up = [](const auto &v) { using T = typename std::decay_t<decltype(v)>::value_type;
    T *d; CK(cudaMalloc(&d, sizeof(T) * std::max<size_t>(1, v.size()))); cudaMemcpy(d, v.data(), sizeof(T) * v.size(), cudaMemcpyHostToDevice); return d; };
  float *dx = up(xs), *dq = up(qs), *dA = up(ta), *dB = up(tb); int *dt = up(ts), *dcounts = up(counts), *dindex = up(index);
  int *drow = up(rowStart), *dent = up(entCluster), *dem = up(entMasks); unsigned char *dim = up(entImask);
  unsigned long long *dmasks = up(masks); float4 *dslots = up(slots); int *dstype = up(slotType);
  float *df1, *df2; CK(cudaMalloc(&df1, 12 * n)); CK(cudaMalloc(&df2, 12 * 8 * C));
  float3 L = make_float3(box[0], box[1], box[2]), iL = make_float3(1 / box[0], 1 / box[1], 1 / box[2]);

  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  const int reps = argc > 3 ? atoi(argv[3]) : 200;
  auto time = [&](auto launch) {
    launch(); CK(cudaDeviceSynchronize());
    cudaEventRecord(e0); for (int r = 0; r < reps; ++r) launch(); cudaEventRecord(e1); CK(cudaEventSynchronize(e1));
    float ms; cudaEventElapsedTime(&ms, e0, e1); return 1000.0 * ms / reps;
  };
  double tm = time([&] { matrixKernel<<<(16 * n + 127) / 128, 128>>>(n, W, dcounts, dindex, dx, dq, dt, dA, dB, ntypes, L, iL, df1); });
  for (int warps : {1, 2, 4}) {
    double ts2 = time([&] { cudaMemsetAsync(df2, 0, 12 * 8 * C);
      superKernel<<<(S + warps - 1) / warps, 32 * warps>>>(S, drow, dent, dim, dem, dmasks, dslots, dstype, dA, dB, ntypes, L, iL, df2); });
    printf("reach %.2f: super-clusters, %d warps a block (with the clearing): %.1f us\n", REACH, warps, ts2);
    printf("csv,pairs,super-clusters half %d warps,%.2f,,%.2f\n", warps, REACH, ts2);
  }
  {
    int *dgs = up(gStart), *dge = up(gEnt); unsigned *dgm = up(gMask);
    for (int warps : {1, 2, 4}) {
      double tg = time([&] { cudaMemsetAsync(df2, 0, 12 * 8 * C);
        groupKernel<<<(Gn + warps - 1) / warps, 32 * warps>>>(Gn, dgs, dge, dgm, dslots, dstype, dA, dB, ntypes, L, iL, df2); });
      printf("reach %.2f: groups of 32, %d warps a block (with the clearing): %.1f us\n", REACH, warps, tg);
      printf("csv,pairs,groups of 32 half %d warps,%.2f,,%.2f\n", warps, REACH, tg);
    }
    std::vector<float> f1(3 * n), f2(3 * 8 * C);
    cudaMemcpy(f1.data(), df1, 12 * n, cudaMemcpyDeviceToHost); cudaMemcpy(f2.data(), df2, 12 * 8 * C, cudaMemcpyDeviceToHost);
    double num = 0, den = 0;
    for (int s = 0; s < 8 * C; ++s) { int i = members[s]; if (i < 0) continue; int p = place[i];
      for (int k = 0; k < 3; ++k) { double d = f1[3*p+k] - f2[3*s+k]; num += d * d; den += (double)f1[3*p+k] * f1[3*p+k]; } }
    printf("groups: relative difference of the forces %.2e\n", std::sqrt(num / den));
    int *dhs = up(hStart), *dhe = up(hEnt); unsigned *dhm = up(hMask);
    for (int warps : {1, 4}) {
      double th = time([&] { cudaMemsetAsync(df2, 0, 12 * 8 * C);
        group16Kernel<<<(G16 + warps - 1) / warps, 32 * warps>>>(G16, dhs, dhe, dhm, dslots, dstype, dA, dB, ntypes, L, iL, df2); });
      printf("reach %.2f: groups of 16, %d warps a block (with the clearing): %.1f us\n", REACH, warps, th);
      printf("csv,pairs,groups of 16 half %d warps,%.2f,,%.2f\n", warps, REACH, th);
    }
    cudaMemcpy(f2.data(), df2, 12 * 8 * C, cudaMemcpyDeviceToHost);
    num = 0;
    for (int s = 0; s < 8 * C; ++s) { int i = members[s]; if (i < 0) continue; int p = place[i];
      for (int k = 0; k < 3; ++k) { double d = f1[3*p+k] - f2[3*s+k]; num += d * d; } }
    printf("groups of 16: relative difference of the forces %.2e\n", std::sqrt(num / den));
  }
  double ts4 = time([&] { cudaMemsetAsync(df2, 0, 12 * 8 * C);
    superKernel<<<(S + 3) / 4, 128>>>(S, drow, dent, dim, dem, dmasks, dslots, dstype, dA, dB, ntypes, L, iL, df2); });
  (void)ts4;
  std::vector<float> f1(3 * n), f2(3 * 8 * C);
  cudaMemcpy(f1.data(), df1, 12 * n, cudaMemcpyDeviceToHost); cudaMemcpy(f2.data(), df2, 12 * 8 * C, cudaMemcpyDeviceToHost);
  double num = 0, den = 0;
  for (int s = 0; s < 8 * C; ++s) { int i = members[s]; if (i < 0) continue; int p = place[i];
    for (int k = 0; k < 3; ++k) { double d = f1[3*p+k] - f2[3*s+k]; num += d * d; den += (double)f1[3*p+k] * f1[3*p+k]; } }
  printf("reach %.2f: matrix %.1f us; relative difference of the forces %.2e\n", REACH, tm, std::sqrt(num / den));
  printf("csv,pairs,matrix 16 lanes,%.2f,,%.2f\n", REACH, tm);
  return 0;
}
