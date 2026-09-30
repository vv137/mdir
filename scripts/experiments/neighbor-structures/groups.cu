// Prototype (2026-10-01, D89): the loop over pairs with groups of 16
// particles sharing a list of neighbors, each pair once, against the
// neighbor matrix, for an order of the particles and a threshold of
// trivial acceptance chosen on the command line. Lennard-Jones from type
// tables and the direct sum of PME in f32; the lists are built on the
// host, and only the loops are timed.
//
//   groups <system.bin> <reach> <order> <accept> [repeats]
//
// order:  cells   - cells of a third of the reach, x first, then by number
//                   (the order of a build of MDIR, D86)
//         compact - columns in x-y of the width of 64 particles, sorted by
//                   z, cut into 64 and halved in x, y, z into groups of 16
//         morton  - cells of a third of the reach, and within them the
//                   particles by the Morton code of their position in the
//                   cell at a resolution of 1/4 (as the Hilbert order of a
//                   cell refines a cell of [SalomonFerrer2013])
// accept: a particle j within accept * reach of the bounding box of a group
//         is taken for all its particles without testing them (0: every
//         pair tested)
// GROUPS_ALL_BITS set: an entry within the reach of one particle of the
//         group has the bits of all its particles (but excluded pairs and
//         empty places), not only of those within the reach
//
// Compiled with -DFAST_ERFC, the erfc of the direct sum is that of
// Abramowitz and Stegun 7.1.26, sharing the exponential of the force.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <cstring>
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
#ifdef FAST_ERFC
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

__global__ void matrixKernel(int n, int W, const int *counts, const int *index, const float4 *xq,
                             const int *ts, const float *A, const float *B, int nt,
                             float3 L, float3 iL, float *force) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  int i = t / 16, lane = t % 16;
  bool valid = i < n;
  float3 f = {0, 0, 0};
  if (valid) {
    float4 p = xq[i];
    float3 xi = {p.x, p.y, p.z};
    int ti = ts[i];
    int c = counts[i];
    for (int e = lane; e < c; e += 16) {
      int j = index[(size_t)i * W + e];
      if (j == i) continue;
      float4 pj = xq[j];
      pairForce(xi, p.w, ti, make_float3(pj.x, pj.y, pj.z), pj.w, ts[j], L, iL, A, B, nt, f);
    }
  }
  for (int o = 8; o; o >>= 1) {
    f.x += __shfl_xor_sync(0xffffffff, f.x, o);
    f.y += __shfl_xor_sync(0xffffffff, f.y, o);
    f.z += __shfl_xor_sync(0xffffffff, f.z, o);
  }
  if (valid && lane == 0) { force[3*i] = f.x; force[3*i+1] = f.y; force[3*i+2] = f.z; }
}

// A warp for each group: lanes u and u + 16 hold place 16 g + u; each lane
// takes one entry of 32, which turn within the half-warp for 16 steps.
__global__ void groupKernel(int G, const int *rowStart, const int *entPlace, const unsigned *entMask,
                            const float4 *xq, const int *ts, const float *A, const float *B,
                            int nt, float3 L, float3 iL, float *force) {
  int g = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  if (g >= G) return;
  int lane = threadIdx.x % 32, u = lane & 15, half = lane & 16;
  float4 pi = xq[16 * g + u];
  float3 xi = make_float3(pi.x, pi.y, pi.z); int ti = ts[16 * g + u];
  float3 fi = {0, 0, 0};
  int end = rowStart[g + 1];
  int src = half | ((u + 15) & 15);
  for (int e0 = rowStart[g]; e0 < end; e0 += 32) {
    int e = e0 + lane;
    int jp = e < end ? entPlace[e] : -1;
    unsigned m = e < end ? entMask[e] : 0u;
    float4 pj = jp >= 0 ? xq[jp] : make_float4(0, 0, 0, 0);
    int tj = jp >= 0 ? ts[jp] : 0;
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
    if (jp >= 0) { float *a = force + 3 * jp; atomicAdd(a, fj.x); atomicAdd(a + 1, fj.y); atomicAdd(a + 2, fj.z); }
  }
  fi.x += __shfl_xor_sync(0xffffffff, fi.x, 16);
  fi.y += __shfl_xor_sync(0xffffffff, fi.y, 16);
  fi.z += __shfl_xor_sync(0xffffffff, fi.z, 16);
  if (!half) { float *a = force + 3 * (16 * g + u); atomicAdd(a, fi.x); atomicAdd(a + 1, fi.y); atomicAdd(a + 2, fi.z); }
}

static uint32_t spread3(uint32_t v) {  // two bits of v into every third
  v &= 0x3; return (v & 1) | ((v & 2) << 2);
}

int main(int argc, char **argv) {
  if (argc < 5) { printf("groups <system.bin> <reach> <order> <accept> [repeats]\n"); return 1; }
  float REACH = atof(argv[2]);
  const char *order = argv[3];
  double accept = atof(argv[4]);
  int reps = argc > 5 ? atoi(argv[5]) : 50;
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

  // The order of the places: particle at each place, padded to groups of 16.
  std::vector<int> members;
  if (!strcmp(order, "compact")) {
    double rho = n / (box[0] * box[1] * box[2]);
    double side = std::cbrt(64.0 / rho);
    int ncx = std::max(1, (int)(box[0] / side)), ncy = std::max(1, (int)(box[1] / side));
    std::vector<std::vector<int>> cols(ncx * ncy);
    for (int i = 0; i < n; ++i) {
      int cx = std::min(ncx - 1, (int)(x[3*i] / box[0] * ncx)), cy = std::min(ncy - 1, (int)(x[3*i+1] / box[1] * ncy));
      cols[cy * ncx + cx].push_back(i);
    }
    for (auto &v : cols) {
      std::sort(v.begin(), v.end(), [&](int i, int j) { return x[3*i+2] < x[3*j+2]; });
      for (size_t s = 0; s < v.size(); s += 64) {
        std::vector<int> g(v.begin() + s, v.begin() + std::min(v.size(), s + 64));
        size_t len = g.size();
        auto halve = [&](size_t a, size_t l, int k) {
          std::sort(g.begin() + a, g.begin() + a + l, [&](int i, int j) { return x[3*i+k] < x[3*j+k]; });
        };
        // Halves by x, then y: four groups of up to 16, in z order inside.
        halve(0, len, 0);
        size_t h = (len + 1) / 2;
        halve(0, h, 1); halve(h, len - h, 1);
        std::vector<size_t> starts = {0, h / 2 + h % 2, h, h + (len - h + 1) / 2, len};
        for (int k = 0; k < 4; ++k) {
          std::vector<int> part(g.begin() + starts[k], g.begin() + starts[k + 1]);
          std::sort(part.begin(), part.end(), [&](int i, int j) { return x[3*i+2] < x[3*j+2]; });
          part.resize(16, -1);
          members.insert(members.end(), part.begin(), part.end());
        }
      }
    }
  } else {
    bool morton = !strcmp(order, "morton");
    double w = REACH / 3.0;
    int nc[3]; for (int k = 0; k < 3; ++k) nc[k] = std::max(1, (int)(box[k] / w));
    std::vector<std::pair<uint64_t, int>> key(n);
    for (int i = 0; i < n; ++i) {
      int c[3]; uint32_t sub[3];
      for (int k = 0; k < 3; ++k) {
        double s = x[3*i+k] / box[k] * nc[k];
        c[k] = std::min(nc[k] - 1, (int)s);
        sub[k] = std::min(3u, (uint32_t)((s - c[k]) * 4));
      }
      uint64_t cell = ((uint64_t)c[2] * nc[1] + c[1]) * nc[0] + c[0];
      uint64_t code = morton ? (spread3(sub[0]) | spread3(sub[1]) << 1 | spread3(sub[2]) << 2) : 0;
      key[i] = {(cell << 6 | code) << 20 | (uint64_t)i, i};
    }
    std::sort(key.begin(), key.end());
    for (auto &k : key) members.push_back(k.second);
    while (members.size() % 16) members.push_back(-1);
  }
  int P = members.size(), G = P / 16;
  std::vector<int> place(n);
  for (int p = 0; p < P; ++p) if (members[p] >= 0) place[members[p]] = p;

  // Bounding boxes of the groups (the particles are wrapped; a group that
  // spans the edge of the cell has a box as wide as the cell, which is
  // conservative).
  std::vector<double> lo(3 * G, 1e30), hi(3 * G, -1e30);
  for (int p = 0; p < P; ++p) { int i = members[p]; if (i < 0) continue; int g = p / 16;
    for (int k = 0; k < 3; ++k) { lo[3*g+k] = std::min(lo[3*g+k], x[3*i+k]); hi[3*g+k] = std::max(hi[3*g+k], x[3*i+k]); } }
  double extent = 0;
  for (int g = 0; g < G; ++g) for (int k = 0; k < 3; ++k) extent += hi[3*g+k] - lo[3*g+k];
  printf("order %s: %d groups, mean extent of a box %.2f A\n", order, G, extent / (3.0 * G));
  auto boxDist2 = [&](int g, int i) {
    double s = 0;
    for (int k = 0; k < 3; ++k) {
      double c = 0.5 * (lo[3*g+k] + hi[3*g+k]), h = 0.5 * (hi[3*g+k] - lo[3*g+k]);
      double d = std::fabs(mi(x[3*i+k] - c, box[k])) - h;
      if (d > 0) s += d * d;
    }
    return s;
  };
  // Candidates by a grid of cells of the reach.
  int gc[3]; for (int k = 0; k < 3; ++k) gc[k] = std::max(3, (int)(box[k] / REACH));
  std::vector<std::vector<int>> grid(gc[0] * gc[1] * gc[2]);
  for (int i = 0; i < n; ++i) { int c[3]; for (int k = 0; k < 3; ++k) c[k] = std::min(gc[k] - 1, (int)(x[3*i+k] / box[k] * gc[k]));
    grid[(c[2] * gc[1] + c[1]) * gc[0] + c[0]].push_back(i); }
  std::vector<int> rowStart(G + 1, 0), entPlace; std::vector<unsigned> entMask;
  size_t slots = 0, within = 0, reachPairs = 0;
  double A2 = accept * accept * R2;
  for (int g = 0; g < G; ++g) {
    // The cells that the box and the reach touch.
    int c0[3], c1[3];
    for (int k = 0; k < 3; ++k) {
      c0[k] = (int)std::floor((lo[3*g+k] - REACH) / box[k] * gc[k]);
      c1[k] = (int)std::floor((hi[3*g+k] + REACH) / box[k] * gc[k]);
      if (c1[k] - c0[k] + 1 > gc[k]) { c0[k] = 0; c1[k] = gc[k] - 1; }
    }
    std::vector<int> cand;
    for (int cz = c0[2]; cz <= c1[2]; ++cz) for (int cy = c0[1]; cy <= c1[1]; ++cy) for (int cx = c0[0]; cx <= c1[0]; ++cx) {
      int ix = ((cx % gc[0]) + gc[0]) % gc[0], iy = ((cy % gc[1]) + gc[1]) % gc[1], iz = ((cz % gc[2]) + gc[2]) % gc[2];
      for (int j : grid[(iz * gc[1] + iy) * gc[0] + ix]) cand.push_back(j);
    }
    std::sort(cand.begin(), cand.end(), [&](int a, int b) { return place[a] < place[b]; });
    cand.erase(std::unique(cand.begin(), cand.end()), cand.end());
    size_t first = entPlace.size();
    for (int j : cand) {
      int pj = place[j];
      if (pj / 16 < g) continue;
      double bd = boxDist2(g, j);
      if (bd > R2) continue;
      bool trivial = accept > 0 && bd <= A2;
      unsigned m = 0; bool any = false;
      for (int u = 0; u < 16; ++u) {
        int p = 16 * g + u, i = members[p];
        if (i < 0 || (pj / 16 == g && pj <= p) || excluded(i, j)) continue;
        double d2 = dist2(i, j);
        if (getenv("GROUPS_ALL_BITS")) { m |= 1u << u; if (d2 <= R2) any = true; }
        else if (trivial || d2 <= R2) { m |= 1u << u; any = true; }
        if (d2 <= R2) ++reachPairs;
      }
      if (!any) continue;
      entPlace.push_back(pj); entMask.push_back(m);
    }
    rowStart[g + 1] = entPlace.size();
    slots += (entPlace.size() - first + 31) / 32 * 512;
  }
  // Pairs within the cutoff among the slots.
  for (size_t e = 0, g = 0; g < (size_t)G; ++g)
    for (e = rowStart[g]; e < (size_t)rowStart[g + 1]; ++e)
      for (int u = 0; u < 16; ++u) if (entMask[e] >> u & 1u) {
        int i = members[16 * g + u], j = members[entPlace[e]];
        if (dist2(i, j) < RC * RC) ++within;
      }
  printf("reach %.2f, accept %.2f: entries %zu (%.1f a group), slots %zu, within the reach %.1f%%, within the cutoff %.1f%%\n",
         REACH, accept, entPlace.size(), (double)entPlace.size() / G, slots, 100.0 * reachPairs / slots, 100.0 * within / slots);

  // The matrix in the same places, for the reference.
  std::vector<std::vector<int>> rows(P);
  size_t listed = 0;
  for (int p = 0; p < P; ++p) {
    int i = members[p]; if (i < 0) continue;
    int c[3]; for (int k = 0; k < 3; ++k) c[k] = std::min(gc[k] - 1, (int)(x[3*i+k] / box[k] * gc[k]));
    for (int dz = -1; dz <= 1; ++dz) for (int dy = -1; dy <= 1; ++dy) for (int dx = -1; dx <= 1; ++dx) {
      int cc = (((c[2]+dz+gc[2])%gc[2]) * gc[1] + (c[1]+dy+gc[1])%gc[1]) * gc[0] + (c[0]+dx+gc[0])%gc[0];
      for (int j : grid[cc]) if (j != i && dist2(i, j) <= R2) { rows[p].push_back(excluded(i, j) ? p : place[j]); ++listed; }
    }
    std::sort(rows[p].begin(), rows[p].end());
    rows[p].erase(std::unique(rows[p].begin(), rows[p].end()), rows[p].end());
  }
  int W = 0; for (auto &r : rows) W = std::max(W, (int)r.size());
  std::vector<int> counts(P), index((size_t)P * W, 0);
  for (int p = 0; p < P; ++p) { counts[p] = rows[p].size(); std::copy(rows[p].begin(), rows[p].end(), index.begin() + (size_t)p * W); }

  std::vector<float4> xq(P); std::vector<int> ts(P);
  for (int p = 0; p < P; ++p) { int i = members[p];
    if (i < 0) { xq[p] = make_float4(-1e4f, -1e4f, -1e4f, 0.f); ts[p] = 0; }
    else { xq[p] = make_float4(x[3*i], x[3*i+1], x[3*i+2], q[i]); ts[p] = type[i]; } }
  auto up = [](const auto &v) { using T = typename std::decay_t<decltype(v)>::value_type;
    T *d; CK(cudaMalloc(&d, sizeof(T) * std::max<size_t>(1, v.size()))); cudaMemcpy(d, v.data(), sizeof(T) * v.size(), cudaMemcpyHostToDevice); return d; };
  float4 *dxq = up(xq); int *dts = up(ts); float *dA = up(ta), *dB = up(tb);
  int *dcounts = up(counts), *dindex = up(index), *drow = up(rowStart), *dent = up(entPlace); unsigned *dmask = up(entMask);
  float *df1, *df2; CK(cudaMalloc(&df1, 12 * P)); CK(cudaMalloc(&df2, 12 * P));
  float3 L = make_float3(box[0], box[1], box[2]), iL = make_float3(1 / box[0], 1 / box[1], 1 / box[2]);
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  auto time = [&](auto launch) {
    launch(); CK(cudaDeviceSynchronize());
    cudaEventRecord(e0); for (int r = 0; r < reps; ++r) launch(); cudaEventRecord(e1); CK(cudaEventSynchronize(e1));
    float ms; cudaEventElapsedTime(&ms, e0, e1); return 1000.0 * ms / reps;
  };
  double tm = time([&] { matrixKernel<<<(16 * P + 127) / 128, 128>>>(P, W, dcounts, dindex, dxq, dts, dA, dB, ntypes, L, iL, df1); });
  double tg = time([&] { cudaMemsetAsync(df2, 0, 12 * P);
    groupKernel<<<(G + 3) / 4, 128>>>(G, drow, dent, dmask, dxq, dts, dA, dB, ntypes, L, iL, df2); });
  std::vector<float> f1(3 * P), f2(3 * P);
  cudaMemcpy(f1.data(), df1, 12 * P, cudaMemcpyDeviceToHost); cudaMemcpy(f2.data(), df2, 12 * P, cudaMemcpyDeviceToHost);
  double num = 0, den = 0;
  for (int p = 0; p < P; ++p) if (members[p] >= 0) for (int k = 0; k < 3; ++k) {
    double d = f1[3*p+k] - f2[3*p+k]; num += d * d; den += (double)f1[3*p+k] * f1[3*p+k]; }
  printf("matrix %.1f us, groups %.1f us, relative difference of the forces %.2e\n", tm, tg, std::sqrt(num / den));
  printf("csv,groups,%s,%.2f,%.2f,%.1f,%.1f\n", order, REACH, accept, tm, tg);
  return 0;
}
