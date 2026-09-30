// Prototype (2026-10-01, D89, stage G0): the build of groups of 16 on a
// device, against the lists built on the host. Everything from the
// positions: the order of the places (a Z curve through cells of about 2
// particles, by a counting sort), the positions in that order, a grid of
// cells of half the reach over the places, the boxes of the groups, and the
// list of each group, a warp a group:
//
//   - the cells of the grid within the reach of the box of the group, a run
//     along x at a time, 32 candidates at once; a candidate before the
//     group (half list) or farther than the reach from the box is dropped;
//   - the survivors, two at once (one a half-warp), against the 16
//     particles of the group: a ballot gives the mask of those within the
//     reach; a pair of the group itself is kept once (the later place);
//   - the excluded pairs: the partners of the particles of the group are
//     gathered once, and a survivor between the least and the greatest of
//     them is compared with them;
//   - the entries (place, mask) are written into the room of the group, of
//     a fixed capacity here.
//
// Compared entry by entry with the lists built on the host.
//
//   build <system.bin> <reach> [repeats]
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

static const int CAP = 1024;        // entries a group can hold here
static const int EXCL = 256;        // exclusion partners a group can hold here

__host__ __device__ inline uint32_t spread3(uint32_t v) {
  v &= 0x3ff;
  v = (v | (v << 16)) & 0x030000ff;
  v = (v | (v << 8)) & 0x0300f00f;
  v = (v | (v << 4)) & 0x030c30c3;
  v = (v | (v << 2)) & 0x09249249;
  return v;
}

struct Params {
  int n, G;             // particles, groups
  float3 L, iL;         // cell
  int3 zc;              // cells of the order
  int3 gc;              // cells of the grid
  float reach2;
};

// 1. The key of each particle in the order, and its cell of the grid.
__global__ void keys(Params P, const float4 *x, uint32_t *zkey) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= P.n) return;
  float4 p = x[i];
  uint32_t cx = min(P.zc.x - 1, (int)(p.x * P.iL.x * P.zc.x));
  uint32_t cy = min(P.zc.y - 1, (int)(p.y * P.iL.y * P.zc.y));
  uint32_t cz = min(P.zc.z - 1, (int)(p.z * P.iL.z * P.zc.z));
  zkey[i] = spread3(cx) | spread3(cy) << 1 | spread3(cz) << 2;
}
// A counting sort: counts by key, then (after a scan on the host side of
// the prototype, by thrust in the real thing) the places.
__global__ void count(int n, const uint32_t *key, int *counts) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) atomicAdd(&counts[key[i]], 1);
}
__global__ void scatter(int n, const uint32_t *key, int *next, int *placeOf) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) placeOf[i] = atomicAdd(&next[key[i]], 1);
}
// 2. The positions in the order of the places, and the cell of the grid of
// each place.
__global__ void gather(Params P, const float4 *x, const int *placeOf, float4 *xp, uint32_t *gkey) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= P.n) return;
  int p = placeOf[i];
  float4 v = x[i];
  xp[p] = v;
  int cx = min(P.gc.x - 1, (int)(v.x * P.iL.x * P.gc.x));
  int cy = min(P.gc.y - 1, (int)(v.y * P.iL.y * P.gc.y));
  int cz = min(P.gc.z - 1, (int)(v.z * P.iL.z * P.gc.z));
  gkey[p] = (cz * P.gc.y + cy) * P.gc.x + cx;
}
__device__ __forceinline__ float image(float d, float l, float il) { return d - l * rintf(d * il); }

// 3. The box of each group, relative to its first particle in the minimum
// image, so that a group across the edge of the cell has its own size: a
// half-warp a group. lo is stored with the first particle in w of org.
__global__ void boxes(Params P, int nplaces, const float4 *xp, float4 *lo, float4 *hi, float4 *org) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  int g = t >> 4, u = t & 15;
  int p = 16 * g + u;
  bool valid = g < P.G && p < nplaces;
  float4 v = valid ? xp[p] : make_float4(0, 0, 0, 0);
  valid = valid && !isnan(v.w);
  float4 o = xp[min(16 * g, nplaces - 1)];
  float r[3] = {image(v.x - o.x, P.L.x, P.iL.x), image(v.y - o.y, P.L.y, P.iL.y), image(v.z - o.z, P.L.z, P.iL.z)};
  float a[3], b[3];
  for (int k = 0; k < 3; ++k) { a[k] = valid ? r[k] : 1e30f; b[k] = valid ? r[k] : -1e30f; }
  for (int o2 = 8; o2; o2 >>= 1)
    for (int k = 0; k < 3; ++k) {
      a[k] = fminf(a[k], __shfl_xor_sync(0xffffffff, a[k], o2));
      b[k] = fmaxf(b[k], __shfl_xor_sync(0xffffffff, b[k], o2));
    }
  if (g < P.G && u == 0) {
    lo[g] = make_float4(a[0], a[1], a[2], 0); hi[g] = make_float4(b[0], b[1], b[2], 0); org[g] = o;
  }
}

// 4. The lists: a warp a group; a lane takes a candidate and tests it
// against the 16 particles of the group.
__global__ void lists(Params P, int nplaces, const float4 *xp, const int *gstart, const int *gplace,
                      const float4 *glo, const float4 *ghi, const float4 *gorg, const int *exOff,
                      const int *exPlace, int *entCount, int *entPlace, unsigned *entMask) {
  __shared__ int partners[4][EXCL];
  __shared__ float4 group[4][16];
  int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int g = blockIdx.x * 4 + w;
  if (g >= P.G) return;
  int first = 16 * g;
  if (lane < 16) group[w][lane] = first + lane < nplaces ? xp[first + lane] : make_float4(1e30f, 1e30f, 1e30f, 0);
  float4 lo = glo[g], hi = ghi[g], o = gorg[g];
  float3 c = make_float3(o.x + 0.5f * (lo.x + hi.x), o.y + 0.5f * (lo.y + hi.y), o.z + 0.5f * (lo.z + hi.z));
  float3 h = make_float3(0.5f * (hi.x - lo.x), 0.5f * (hi.y - lo.y), 0.5f * (hi.z - lo.z));
  float reach = sqrtf(P.reach2);
  // The exclusion partners of the group (places), and their range.
  int np = 0, pmin = 0x7fffffff, pmax = -1;
  for (int k = 0; k < 16; ++k) {
    int p = first + k;
    if (p >= nplaces) break;
    int b = exOff[p], e = exOff[p + 1];
    for (int s2 = b + lane; s2 < e; s2 += 32) {
      int at = np + (s2 - b);
      if (at < EXCL) partners[w][at] = (exPlace[s2] << 4) | k;
    }
    np += e - b;
  }
  np = min(np, EXCL);
  __syncwarp();
  for (int s2 = lane; s2 < np; s2 += 32) { int q = partners[w][s2] >> 4; pmin = min(pmin, q); pmax = max(pmax, q); }
  for (int o2 = 16; o2; o2 >>= 1) { pmin = min(pmin, __shfl_xor_sync(0xffffffff, pmin, o2)); pmax = max(pmax, __shfl_xor_sync(0xffffffff, pmax, o2)); }

  // The cells of the grid that the box and the reach touch.
  float cv[3] = {c.x, c.y, c.z}, hv[3] = {h.x, h.y, h.z};
  float Lv[3] = {P.L.x, P.L.y, P.L.z}; int gcv[3] = {P.gc.x, P.gc.y, P.gc.z};
  int c0[3], c1[3];
  for (int k = 0; k < 3; ++k) {
    c0[k] = (int)floorf((cv[k] - hv[k] - reach) / Lv[k] * gcv[k]);
    c1[k] = (int)floorf((cv[k] + hv[k] + reach) / Lv[k] * gcv[k]);
    if (c1[k] - c0[k] + 1 > gcv[k]) { c0[k] = 0; c1[k] = gcv[k] - 1; }
  }
  int count = 0;
  int *outPlace = entPlace + (size_t)g * CAP;
  unsigned *outMask = entMask + (size_t)g * CAP;
  for (int cz = c0[2]; cz <= c1[2]; ++cz)
    for (int cy = c0[1]; cy <= c1[1]; ++cy) {
      int iz = (cz % P.gc.z + P.gc.z) % P.gc.z, iy = (cy % P.gc.y + P.gc.y) % P.gc.y;
      int row = (iz * P.gc.y + iy) * P.gc.x;
      for (int piece = 0; piece < 2; ++piece) {
        int a = c0[0], b = c1[0];
        int from, to;
        if (a >= 0 && b < P.gc.x) { if (piece) break; from = a; to = b; }
        else if (a < 0) { if (piece == 0) { from = a + P.gc.x; to = P.gc.x - 1; } else { from = 0; to = b; } }
        else { if (piece == 0) { from = a; to = P.gc.x - 1; } else { from = 0; to = b - P.gc.x; } }
        int begin = gstart[row + from], end = gstart[row + to + 1];
        for (int s0 = begin; s0 < end; s0 += 32) {
          int s2 = s0 + lane;
          unsigned bits = 0;
          int q = -1;
          if (s2 < end) {
            q = gplace[s2];
            if (q >= first) {
              float4 xq = xp[q];
              float dx = fmaxf(fabsf(image(xq.x - c.x, P.L.x, P.iL.x)) - h.x, 0.f);
              float dy = fmaxf(fabsf(image(xq.y - c.y, P.L.y, P.iL.y)) - h.y, 0.f);
              float dz = fmaxf(fabsf(image(xq.z - c.z, P.L.z, P.iL.z)) - h.z, 0.f);
              if (dx * dx + dy * dy + dz * dz <= P.reach2) {
                bool own = q < first + 16;
#pragma unroll
                for (int u = 0; u < 16; ++u) {
                  float4 xi = group[w][u];
                  float ex = image(xq.x - xi.x, P.L.x, P.iL.x), ey = image(xq.y - xi.y, P.L.y, P.iL.y),
                        ez = image(xq.z - xi.z, P.L.z, P.iL.z);
                  bool in = ex * ex + ey * ey + ez * ez <= P.reach2 && !(own && q <= first + u) && !isnan(xi.w);
                  bits |= (unsigned)in << u;
                }
                if (bits && q >= pmin && q <= pmax)
                  for (int e = 0; e < np; ++e) {
                    int pe = partners[w][e];
                    if ((pe >> 4) == q) bits &= ~(1u << (pe & 15));
                  }
              }
            }
          }
          unsigned kept = __ballot_sync(0xffffffff, bits != 0);
          if (bits) {
            int slot = count + __popc(kept & ((1u << lane) - 1));
            if (slot < CAP) { outPlace[slot] = q; outMask[slot] = bits; }
          }
          count += __popc(kept);
        }
      }
    }
  if (lane == 0) entCount[g] = count;
}

int main(int argc, char **argv) {
  float REACH = atof(argv[2]);
  int reps = argc > 3 ? atoi(argv[3]) : 20;
  FILE *fp = fopen(argv[1], "rb");
  int n, ntypes; double box[3];
  fread(&n, 4, 1, fp); fread(&ntypes, 4, 1, fp); fread(box, 8, 3, fp);
  std::vector<double> x(3 * n); std::vector<float> q(n); std::vector<int> type(n);
  std::vector<float> ta(ntypes * ntypes), tb(ntypes * ntypes);
  fread(x.data(), 8, 3 * n, fp); fread(q.data(), 4, n, fp); fread(type.data(), 4, n, fp);
  fread(ta.data(), 4, ntypes * ntypes, fp); fread(tb.data(), 4, ntypes * ntypes, fp);
  std::vector<int> exoff(n + 1); fread(exoff.data(), 4, n + 1, fp);
  std::vector<int> exl(exoff[n]); fread(exl.data(), 4, exoff[n], fp);
  fclose(fp);
  for (int i = 0; i < 3 * n; ++i) { int k = i % 3; x[i] -= box[k] * std::floor(x[i] / box[k]); }

  double rho = n / (box[0] * box[1] * box[2]);
  double wz = std::cbrt(2.0 / rho);
  Params P;
  P.n = n; P.L = make_float3(box[0], box[1], box[2]); P.iL = make_float3(1 / box[0], 1 / box[1], 1 / box[2]);
  P.zc = make_int3(std::max(1, (int)(box[0] / wz)), std::max(1, (int)(box[1] / wz)), std::max(1, (int)(box[2] / wz)));
  P.gc = make_int3(std::max(3, (int)(box[0] / (REACH / 2))), std::max(3, (int)(box[1] / (REACH / 2))), std::max(3, (int)(box[2] / (REACH / 2))));
  P.reach2 = REACH * REACH;
  int nplaces = n;
  P.G = (n + 15) / 16;
  uint32_t zmax = spread3(P.zc.x - 1) | spread3(P.zc.y - 1) << 1 | spread3(P.zc.z - 1) << 2;
  int nz = zmax + 1, ng = P.gc.x * P.gc.y * P.gc.z;
  printf("%d particles, %d groups, order cells %d x %d x %d (%d keys), grid %d x %d x %d\n",
         n, P.G, P.zc.x, P.zc.y, P.zc.z, nz, P.gc.x, P.gc.y, P.gc.z);

  std::vector<float4> xh(n);
  for (int i = 0; i < n; ++i) xh[i] = make_float4(x[3*i], x[3*i+1], x[3*i+2], q[i]);
  float4 *dx, *dxp, *dlo, *dhi, *dorg; uint32_t *dzkey, *dgkey; int *dcounts, *dnext, *dplace, *dgstart, *dgplace, *dgnext;
  CK(cudaMalloc(&dx, 16 * n)); CK(cudaMalloc(&dxp, 16 * n)); CK(cudaMalloc(&dlo, 16 * P.G)); CK(cudaMalloc(&dhi, 16 * P.G)); CK(cudaMalloc(&dorg, 16 * P.G));
  CK(cudaMalloc(&dzkey, 4 * n)); CK(cudaMalloc(&dgkey, 4 * n)); CK(cudaMalloc(&dcounts, 4 * (nz + 1)));
  CK(cudaMalloc(&dnext, 4 * (nz + 1))); CK(cudaMalloc(&dplace, 4 * n)); CK(cudaMalloc(&dgstart, 4 * (ng + 1)));
  CK(cudaMalloc(&dgplace, 4 * n)); CK(cudaMalloc(&dgnext, 4 * (ng + 1)));
  cudaMemcpy(dx, xh.data(), 16 * n, cudaMemcpyHostToDevice);
  int *dentCount, *dentPlace; unsigned *dentMask;
  CK(cudaMalloc(&dentCount, 4 * P.G)); CK(cudaMalloc(&dentPlace, 4 * (size_t)P.G * CAP)); CK(cudaMalloc(&dentMask, 4 * (size_t)P.G * CAP));

  // The order (on the device, with the scan on the host in this prototype).
  int T = 128; auto blocks = [&](long t) { return (unsigned)((t + T - 1) / T); };
  std::vector<int> hc(nz + 1), hg(ng + 1);
  auto scanOnHost = [&](int *d, int m, std::vector<int> &h) {
    cudaMemcpy(h.data(), d, 4 * m, cudaMemcpyDeviceToHost);
    int s = 0; for (int k = 0; k < m; ++k) { int c = h[k]; h[k] = s; s += c; } h[m] = s;
    cudaMemcpy(d, h.data(), 4 * (m + 1), cudaMemcpyHostToDevice);
  };
  auto order = [&]() {
    keys<<<blocks(n), T>>>(P, dx, dzkey);
    cudaMemset(dcounts, 0, 4 * (nz + 1));
    count<<<blocks(n), T>>>(n, dzkey, dcounts);
    scanOnHost(dcounts, nz, hc);
    cudaMemcpy(dnext, dcounts, 4 * (nz + 1), cudaMemcpyDeviceToDevice);
    scatter<<<blocks(n), T>>>(n, dzkey, dnext, dplace);
    gather<<<blocks(n), T>>>(P, dx, dplace, dxp, dgkey);
    cudaMemset(dgstart, 0, 4 * (ng + 1));
    count<<<blocks(n), T>>>(n, dgkey, dgstart);
    scanOnHost(dgstart, ng, hg);
    cudaMemcpy(dgnext, dgstart, 4 * (ng + 1), cudaMemcpyDeviceToDevice);
  };
  order();
  // With ORDER=compact, the places come instead from the compact order of
  // groups.cu, computed on the host (columns of the width of 64 particles,
  // sorted by z, cut into 64, halved in x and y into groups of 16, empty
  // places where a column or a group runs short): the lists and their
  // times for compact groups, before an order that a device computes.
  std::vector<int> hostPlace;
  if (getenv("ORDER") && std::string(getenv("ORDER")) == "compact") {
    double side = std::cbrt(64.0 / rho);
    int ncx = std::max(1, (int)(box[0] / side)), ncy = std::max(1, (int)(box[1] / side));
    std::vector<std::vector<int>> cols(ncx * ncy);
    for (int i = 0; i < n; ++i) {
      int cx = std::min(ncx - 1, (int)(x[3*i] / box[0] * ncx)), cy = std::min(ncy - 1, (int)(x[3*i+1] / box[1] * ncy));
      cols[cy * ncx + cx].push_back(i);
    }
    std::vector<int> members;
    for (auto &v : cols) {
      std::sort(v.begin(), v.end(), [&](int i, int j) { return x[3*i+2] < x[3*j+2]; });
      for (size_t s0 = 0; s0 < v.size(); s0 += 64) {
        std::vector<int> g(v.begin() + s0, v.begin() + std::min(v.size(), s0 + 64));
        size_t len = g.size();
        auto halve = [&](size_t a, size_t l, int k) {
          std::sort(g.begin() + a, g.begin() + a + l, [&](int i, int j) { return x[3*i+k] < x[3*j+k]; }); };
        halve(0, len, 0);
        size_t hh = (len + 1) / 2;
        halve(0, hh, 1); halve(hh, len - hh, 1);
        size_t st[5] = {0, hh / 2 + hh % 2, hh, hh + (len - hh + 1) / 2, len};
        for (int k = 0; k < 4; ++k) {
          std::vector<int> part(g.begin() + st[k], g.begin() + st[k + 1]);
          std::sort(part.begin(), part.end(), [&](int i, int j) { return x[3*i+2] < x[3*j+2]; });
          if (part.empty()) continue;
          part.resize(16, -1);
          members.insert(members.end(), part.begin(), part.end());
        }
      }
    }
    nplaces = members.size();
    P.G = nplaces / 16;
    hostPlace.assign(n, -1);
    for (int p = 0; p < nplaces; ++p) if (members[p] >= 0) hostPlace[members[p]] = p;
    printf("compact order from the host: %d places, %d groups\n", nplaces, P.G);
    std::vector<float4> xph(nplaces);
    for (int pl = 0; pl < nplaces; ++pl) {
      int i = members[pl];
      xph[pl] = i >= 0 ? xh[i] : make_float4(0, 0, 0, NAN);
    }
    cudaFree(dxp); CK(cudaMalloc(&dxp, 16 * nplaces));
    cudaMemcpy(dxp, xph.data(), 16 * nplaces, cudaMemcpyHostToDevice);
    cudaFree(dlo); cudaFree(dhi); cudaFree(dorg);
    CK(cudaMalloc(&dlo, 16 * P.G)); CK(cudaMalloc(&dhi, 16 * P.G)); CK(cudaMalloc(&dorg, 16 * P.G));
    cudaFree(dentCount); cudaFree(dentPlace); cudaFree(dentMask);
    CK(cudaMalloc(&dentCount, 4 * P.G)); CK(cudaMalloc(&dentPlace, 4 * (size_t)P.G * CAP)); CK(cudaMalloc(&dentMask, 4 * (size_t)P.G * CAP));
  }
  // The grid over the places: place p goes to its cell.
  std::vector<int> placeOf(n); cudaMemcpy(placeOf.data(), dplace, 4 * n, cudaMemcpyDeviceToHost);
  if (!hostPlace.empty()) placeOf = hostPlace;
  std::vector<int> member(nplaces, -1); for (int i = 0; i < n; ++i) member[placeOf[i]] = i;
  // The grid over the places (the real ones).
  std::vector<uint32_t> gk(nplaces, 0xffffffffu);
  for (int i = 0; i < n; ++i) {
    int cx = std::min(P.gc.x - 1, (int)(xh[i].x * P.iL.x * P.gc.x));
    int cy = std::min(P.gc.y - 1, (int)(xh[i].y * P.iL.y * P.gc.y));
    int cz = std::min(P.gc.z - 1, (int)(xh[i].z * P.iL.z * P.gc.z));
    gk[placeOf[i]] = (cz * P.gc.y + cy) * P.gc.x + cx;
  }
  std::vector<int> hs(ng + 1, 0);
  for (int pl = 0; pl < nplaces; ++pl) if (gk[pl] != 0xffffffffu) ++hs[gk[pl] + 1];
  for (int k = 0; k < ng; ++k) hs[k + 1] += hs[k];
  cudaMemcpy(dgstart, hs.data(), 4 * (ng + 1), cudaMemcpyHostToDevice);
  std::vector<int> gplace(n), next(hs.begin(), hs.end() - 1);
  for (int pl = 0; pl < nplaces; ++pl) if (gk[pl] != 0xffffffffu) gplace[next[gk[pl]]++] = pl;
  cudaMemcpy(dgplace, gplace.data(), 4 * n, cudaMemcpyHostToDevice);
  // The exclusion partners by place.
  std::vector<int> exOffP(nplaces + 1, 0), exPlace;
  for (int pl = 0; pl < nplaces; ++pl) { int i = member[pl];
    if (i >= 0) for (int k = exoff[i]; k < exoff[i+1]; ++k) exPlace.push_back(placeOf[exl[k]]);
    exOffP[pl + 1] = exPlace.size(); }
  int *dexOff, *dexPlace;
  CK(cudaMalloc(&dexOff, 4 * (nplaces + 1))); CK(cudaMalloc(&dexPlace, 4 * std::max<size_t>(1, exPlace.size())));
  cudaMemcpy(dexOff, exOffP.data(), 4 * (nplaces + 1), cudaMemcpyHostToDevice);
  cudaMemcpy(dexPlace, exPlace.data(), 4 * exPlace.size(), cudaMemcpyHostToDevice);

  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  auto time = [&](auto launch) {
    launch(); CK(cudaDeviceSynchronize());
    cudaEventRecord(e0); for (int r = 0; r < reps; ++r) launch(); cudaEventRecord(e1); CK(cudaEventSynchronize(e1));
    float ms; cudaEventElapsedTime(&ms, e0, e1); return 1000.0 * ms / reps;
  };
  double tk = time([&] { keys<<<blocks(n), T>>>(P, dx, dzkey); cudaMemsetAsync(dcounts, 0, 4 * (nz + 1)); count<<<blocks(n), T>>>(n, dzkey, dcounts); });
  double ts = time([&] { cudaMemcpyAsync(dnext, dcounts, 4 * (nz + 1), cudaMemcpyDeviceToDevice); scatter<<<blocks(n), T>>>(n, dzkey, dnext, dplace); });
  // Timed into buffers of its own, so that the positions of the places
  // that the lists read stay those of the order in use.
  float4 *dxp2; uint32_t *dgkey2; CK(cudaMalloc(&dxp2, 16 * n)); CK(cudaMalloc(&dgkey2, 4 * n));
  double tg = time([&] { gather<<<blocks(n), T>>>(P, dx, dplace, dxp2, dgkey2); });
  double tb2 = time([&] { boxes<<<blocks(16L * P.G), T>>>(P, nplaces, dxp, dlo, dhi, dorg); });
  double tl = time([&] { lists<<<(P.G + 3) / 4, 128>>>(P, nplaces, dxp, dgstart, dgplace, dlo, dhi, dorg, dexOff, dexPlace, dentCount, dentPlace, dentMask); });
  printf("keys and counts %.1f us, places %.1f us, positions and cells %.1f us, boxes %.1f us, lists %.1f us\n", tk, ts, tg, tb2, tl);
  printf("csv,build,groups,%.2f,%.1f,%.1f,%.1f,%.1f,%.1f\n", REACH, tk, ts, tg, tb2, tl);

  // Against the host: the list of each group, as a set of (place, mask).
  std::vector<int> cnt(P.G); cudaMemcpy(cnt.data(), dentCount, 4 * P.G, cudaMemcpyDeviceToHost);
  std::vector<int> ep((size_t)P.G * CAP); std::vector<unsigned> em((size_t)P.G * CAP);
  cudaMemcpy(ep.data(), dentPlace, 4 * ep.size(), cudaMemcpyDeviceToHost);
  cudaMemcpy(em.data(), dentMask, 4 * em.size(), cudaMemcpyDeviceToHost);
  std::vector<float4> xp(nplaces); cudaMemcpy(xp.data(), dxp, 16 * nplaces, cudaMemcpyDeviceToHost);
  auto d2f = [&](int a, int b) {
    float s = 0; float Lv[3] = {P.L.x, P.L.y, P.L.z}, iLv[3] = {P.iL.x, P.iL.y, P.iL.z};
    float av[3] = {xp[a].x, xp[a].y, xp[a].z}, bv[3] = {xp[b].x, xp[b].y, xp[b].z};
    for (int k = 0; k < 3; ++k) { float d = bv[k] - av[k]; d -= Lv[k] * rintf(d * iLv[k]); s += d * d; }
    return s;
  };
  long entries = 0, wrong = 0, overflow = 0, maxCount = 0;
  std::vector<std::vector<int>> grid(ng);
  for (int pl = 0; pl < nplaces; ++pl) if (gk[pl] != 0xffffffffu) grid[gk[pl]].push_back(pl);
  for (int g = 0; g < P.G; ++g) {
    entries += cnt[g]; maxCount = std::max<long>(maxCount, cnt[g]);
    if (cnt[g] > CAP) { ++overflow; continue; }
    std::vector<std::pair<int, unsigned>> dev;
    for (int k = 0; k < cnt[g]; ++k) dev.push_back({ep[(size_t)g * CAP + k], em[(size_t)g * CAP + k]});
    std::sort(dev.begin(), dev.end());
    std::vector<std::pair<int, unsigned>> ref;
    for (int qp = 16 * g; qp < n; ++qp) {
      (void)qp; break;
    }
    // The reference: every place q >= 16 g within the reach of a particle of g.
    std::vector<int> cands;
    for (int u = 0; u < 16; ++u) {
      int p = 16 * g + u; if (p >= nplaces || member[p] < 0) continue;
      int c[3] = {std::min(P.gc.x - 1, (int)(xp[p].x * P.iL.x * P.gc.x)), std::min(P.gc.y - 1, (int)(xp[p].y * P.iL.y * P.gc.y)),
                  std::min(P.gc.z - 1, (int)(xp[p].z * P.iL.z * P.gc.z))};
      for (int dz = -2; dz <= 2; ++dz) for (int dy = -2; dy <= 2; ++dy) for (int dxc = -2; dxc <= 2; ++dxc) {
        int cc = (((c[2]+dz+P.gc.z)%P.gc.z) * P.gc.y + (c[1]+dy+P.gc.y)%P.gc.y) * P.gc.x + (c[0]+dxc+P.gc.x)%P.gc.x;
        for (int qq : grid[cc]) if (qq >= 16 * g) cands.push_back(qq);
      }
    }
    std::sort(cands.begin(), cands.end()); cands.erase(std::unique(cands.begin(), cands.end()), cands.end());
    for (int qq : cands) {
      unsigned m = 0;
      for (int u = 0; u < 16; ++u) {
        int p = 16 * g + u; if (p >= nplaces || member[p] < 0) continue;
        if (qq / 16 == g && qq <= p) continue;
        bool ex = false; int i = member[p], j = member[qq];
        for (int k = exoff[i]; k < exoff[i+1]; ++k) if (exl[k] == j) ex = true;
        if (ex) continue;
        if (d2f(p, qq) <= P.reach2) m |= 1u << u;
      }
      if (m) ref.push_back({qq, m});
    }
    if (ref != dev) ++wrong;
  }
  {
    std::vector<float4> lo(P.G), hi(P.G);
    cudaMemcpy(lo.data(), dlo, 16 * P.G, cudaMemcpyDeviceToHost); cudaMemcpy(hi.data(), dhi, 16 * P.G, cudaMemcpyDeviceToHost);
    int bins[6] = {0, 0, 0, 0, 0, 0}; double edges[5] = {6, 9, 12, 20, 40};
    for (int g = 0; g < P.G; ++g) {
      float e = std::max({hi[g].x - lo[g].x, hi[g].y - lo[g].y, hi[g].z - lo[g].z});
      int b = 0; while (b < 5 && e > edges[b]) ++b; ++bins[b];
      if (b == 5 && bins[5] <= 3) printf("group %d of %d: box %g %g %g to %g %g %g\n", g, P.G, lo[g].x, lo[g].y, lo[g].z, hi[g].x, hi[g].y, hi[g].z);
    }
    printf("largest edge of a box: <=6 A %d, <=9 %d, <=12 %d, <=20 %d, <=40 %d, more %d\n", bins[0], bins[1], bins[2], bins[3], bins[4], bins[5]);
  }
  printf("entries %ld (%.1f a group, at most %ld), groups that differ from the host %ld, groups over the capacity %ld\n",
         entries, (double)entries / P.G, maxCount, wrong, overflow);
  return 0;
}
