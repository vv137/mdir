// Prototype (2026-09-30): the search of the neighbor matrix as MDIR's template does it
// (cells of half the reach, sorted by cell, rows of cells read as runs), with a
// thread per particle against a warp per particle compacting with ballots in the
// same order. Host binning; only the search kernel is timed.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

struct Grid { int nx, ny, nz, rx, ry, rz, fx, fy, fz, sx, sy, sz; float lx, ly, lz, ilx, ily, ilz, limit2; };

__device__ __forceinline__ float img(float d, float l, float il) { return d - l * rintf(d * il); }

__global__ void threadSearch(int n, Grid g, const float *sorted, const int *order, const int *key,
                             const int *start, int W, int *counts, int *index) {
  int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= n) return;
  int i = order[p];
  float xi = sorted[3*p], yi = sorted[3*p+1], zi = sorted[3*p+2];
  int k = key[i], cx = k % g.nx, cy = (k / g.nx) % g.ny, cz = k / (g.nx * g.ny);
  int count = 0;
  for (int r = 0; r < g.sy * g.sz; ++r) {
    int oz = r / g.sy, oy = r % g.sy;
    int zc = (cz + g.fz + oz) % g.nz, yc = (cy + g.fy + oy) % g.ny;
    int first = (zc * g.ny + yc) * g.nx;
    int xb = (cx + g.fx) % g.nx, xe = xb + g.sx;
    int stop = xe > g.nx ? g.nx : xe, rest = xe > g.nx ? xe - g.nx : 0;
    for (int part = 0; part < 2; ++part) {
      int from = part ? 0 : xb, to = part ? rest : stop;
      int b = start[first + from], e = start[first + to];
      for (int q = b; q < e; ++q) {
        float dx = img(xi - sorted[3*q], g.lx, g.ilx), dy = img(yi - sorted[3*q+1], g.ly, g.ily), dz = img(zi - sorted[3*q+2], g.lz, g.ilz);
        if (dx*dx + dy*dy + dz*dz < g.limit2 && q != p) { if (count < W) index[(size_t)i * W + count] = order[q]; ++count; }
      }
    }
  }
  counts[i] = count;
}

template <bool WRITE>
__global__ void warpSearch(int n, Grid g, const float *sorted, const int *order, const int *key,
                           const int *start, int W, int *counts, int *index) {
  int p = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x % 32;
  if (p >= n) return;
  int i = order[p];
  float xi = sorted[3*p], yi = sorted[3*p+1], zi = sorted[3*p+2];
  int k = key[i], cx = k % g.nx, cy = (k / g.nx) % g.ny, cz = k / (g.nx * g.ny);
  int count = 0;
  for (int r = 0; r < g.sy * g.sz; ++r) {
    int oz = r / g.sy, oy = r % g.sy;
    int zc = (cz + g.fz + oz) % g.nz, yc = (cy + g.fy + oy) % g.ny;
    int first = (zc * g.ny + yc) * g.nx;
    int xb = (cx + g.fx) % g.nx, xe = xb + g.sx;
    int stop = xe > g.nx ? g.nx : xe, rest = xe > g.nx ? xe - g.nx : 0;
    for (int part = 0; part < 2; ++part) {
      int from = part ? 0 : xb, to = part ? rest : stop;
      int b = start[first + from], e = start[first + to];
      for (int base = b; base < e; base += 32) {
        int q = base + lane;
        bool hit = false;
        if (q < e) {
          float dx = img(xi - sorted[3*q], g.lx, g.ilx), dy = img(yi - sorted[3*q+1], g.ly, g.ily), dz = img(zi - sorted[3*q+2], g.lz, g.ilz);
          hit = dx*dx + dy*dy + dz*dz < g.limit2 && q != p;
        }
        unsigned ballot = __ballot_sync(0xffffffff, hit);
        int slot = count + __popc(ballot & ((1u << lane) - 1));
        if (WRITE && hit && slot < W) index[(size_t)i * W + slot] = order[q];
        count += __popc(ballot);
      }
    }
  }
  if (lane == 0) counts[i] = count;
}

// A block of 4 warps per cell: the candidates of each run are loaded into
// shared memory once, in chunks of 128, and a warp per particle of the cell
// tests them, compacting with ballots in the order of the thread search.
__global__ void cellSearch(int n, Grid g, const float *sorted, const int *order,
                           const int *start, int W, int *counts, int *index) {
  __shared__ float sx[128], sy[128], sz[128];
  __shared__ int sj[128];
  __shared__ int cnt[64];
  int c = blockIdx.x, w = threadIdx.x / 32, lane = threadIdx.x % 32;
  int pb = start[c], pe = start[c + 1], m = pe - pb;
  int cx = c % g.nx, cy = (c / g.nx) % g.ny, cz = c / (g.nx * g.ny);
  for (int k = threadIdx.x; k < 64; k += blockDim.x) cnt[k] = 0;
  __syncthreads();
  for (int r = 0; r < g.sy * g.sz; ++r) {
    int oz = r / g.sy, oy = r % g.sy;
    int zc = (cz + g.fz + oz) % g.nz, yc = (cy + g.fy + oy) % g.ny;
    int first = (zc * g.ny + yc) * g.nx;
    int xb = (cx + g.fx) % g.nx, xe = xb + g.sx;
    int stop = xe > g.nx ? g.nx : xe, rest = xe > g.nx ? xe - g.nx : 0;
    for (int part = 0; part < 2; ++part) {
      int from = part ? 0 : xb, to = part ? rest : stop;
      int b = start[first + from], e = start[first + to];
      for (int base = b; base < e; base += 128) {
        __syncthreads();
        int q = base + threadIdx.x;
        if (q < e) { sx[threadIdx.x] = sorted[3*q]; sy[threadIdx.x] = sorted[3*q+1]; sz[threadIdx.x] = sorted[3*q+2]; sj[threadIdx.x] = q; }
        __syncthreads();
        int len = min(128, e - base);
        for (int a = w; a < m; a += 4) {
          int p = pb + a, i = order[p];
          float xi = sorted[3*p], yi = sorted[3*p+1], zi = sorted[3*p+2];
          int count = cnt[a];
          for (int o = 0; o < len; o += 32) {
            int t = o + lane;
            bool hit = false;
            if (t < len) {
              float dx = img(xi - sx[t], g.lx, g.ilx), dy = img(yi - sy[t], g.ly, g.ily), dz = img(zi - sz[t], g.lz, g.ilz);
              hit = dx*dx + dy*dy + dz*dz < g.limit2 && sj[t] != p;
            }
            unsigned ballot = __ballot_sync(0xffffffff, hit);
            int slot = count + __popc(ballot & ((1u << lane) - 1));
            if (hit && slot < W) index[(size_t)i * W + slot] = order[sj[t]];
            count += __popc(ballot);
          }
          if (lane == 0) cnt[a] = count;
          __syncwarp();
        }
      }
    }
  }
  __syncthreads();
  for (int a = threadIdx.x; a < m; a += blockDim.x) counts[order[pb + a]] = cnt[a];
}

// warpSearch with the excluded pairs of each particle looked up for every
// neighbor found (CSR of partners), entered as the particle itself.
__global__ void warpSearchExcl(int n, Grid g, const float *sorted, const int *order, const int *key,
                               const int *start, int W, int *counts, int *index,
                               const int *exoff, const int *exl) {
  int p = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x % 32;
  if (p >= n) return;
  int i = order[p];
  float xi = sorted[3*p], yi = sorted[3*p+1], zi = sorted[3*p+2];
  int k = key[i], cx = k % g.nx, cy = (k / g.nx) % g.ny, cz = k / (g.nx * g.ny);
  int eb = exoff[i], ee = exoff[i + 1];
  int count = 0;
  for (int r = 0; r < g.sy * g.sz; ++r) {
    int oz = r / g.sy, oy = r % g.sy;
    int zc = (cz + g.fz + oz) % g.nz, yc = (cy + g.fy + oy) % g.ny;
    int first = (zc * g.ny + yc) * g.nx;
    int xb = (cx + g.fx) % g.nx, xe = xb + g.sx;
    int stop = xe > g.nx ? g.nx : xe, rest = xe > g.nx ? xe - g.nx : 0;
    for (int part = 0; part < 2; ++part) {
      int from = part ? 0 : xb, to = part ? rest : stop;
      int b = start[first + from], e = start[first + to];
      for (int base = b; base < e; base += 32) {
        int q = base + lane;
        bool hit = false;
        if (q < e) {
          float dx = img(xi - sorted[3*q], g.lx, g.ilx), dy = img(yi - sorted[3*q+1], g.ly, g.ily), dz = img(zi - sorted[3*q+2], g.lz, g.ilz);
          hit = dx*dx + dy*dy + dz*dz < g.limit2 && q != p;
        }
        unsigned ballot = __ballot_sync(0xffffffff, hit);
        int slot = count + __popc(ballot & ((1u << lane) - 1));
        if (hit && slot < W) {
          int j = order[q];
          bool ex = false;
          for (int x = eb; x < ee; ++x) ex |= exl[x] == j;
          index[(size_t)i * W + slot] = ex ? i : j;
        }
        count += __popc(ballot);
      }
    }
  }
  if (lane == 0) counts[i] = count;
}

int main(int argc, char **argv) {
  float reach = argc > 2 ? atof(argv[2]) : 10.0f;
  FILE *fp = fopen(argv[1], "rb"); int n, nt; double box[3];
  fread(&n, 4, 1, fp); fread(&nt, 4, 1, fp); fread(box, 8, 3, fp);
  std::vector<double> x(3 * n); fread(x.data(), 8, 3 * n, fp);
  { std::vector<float> q(n); std::vector<int> t(n), ab(2 * nt * nt); fread(q.data(), 4, n, fp); fread(t.data(), 4, n, fp); fread(ab.data(), 4, 2 * nt * nt, fp); }
  std::vector<int> exoff(n + 1); fread(exoff.data(), 4, n + 1, fp);
  std::vector<int> exl(exoff[n]); fread(exl.data(), 4, exoff[n], fp); fclose(fp);
  Grid g; int nc[3], range[3], span[3], firstc[3];
  for (int k = 0; k < 3; ++k) {
    nc[k] = std::max(1, (int)(box[k] / (reach / (argc > 3 ? atof(argv[3]) : 2.0))));
    double w = box[k] / nc[k]; range[k] = (int)std::ceil(reach / w);
    int full = 2 * range[k] + 1; span[k] = std::min(nc[k], full); firstc[k] = span[k] == full ? nc[k] - range[k] : 0;
  }
  g.nx = nc[0]; g.ny = nc[1]; g.nz = nc[2]; g.sx = span[0]; g.sy = span[1]; g.sz = span[2];
  g.fx = firstc[0]; g.fy = firstc[1]; g.fz = firstc[2];
  g.lx = box[0]; g.ly = box[1]; g.lz = box[2]; g.ilx = 1 / g.lx; g.ily = 1 / g.ly; g.ilz = 1 / g.lz;
  g.limit2 = reach * reach;
  int cells = g.nx * g.ny * g.nz;
  std::vector<int> key(n), order(n), start(cells + 1, 0);
  std::vector<std::pair<int,int>> kk(n);
  for (int i = 0; i < n; ++i) {
    int c[3]; for (int k = 0; k < 3; ++k) { double w = x[3*i+k] - box[k] * std::floor(x[3*i+k] / box[k]); c[k] = std::min(nc[k] - 1, (int)(w / box[k] * nc[k])); }
    key[i] = (c[2] * g.ny + c[1]) * g.nx + c[0]; kk[i] = {key[i], i};
  }
  std::sort(kk.begin(), kk.end());
  std::vector<float> sorted(3 * n);
  for (int p = 0; p < n; ++p) { order[p] = kk[p].second; for (int k = 0; k < 3; ++k) sorted[3*p+k] = x[3*order[p]+k]; start[kk[p].first + 1]++; }
  for (int c = 0; c < cells; ++c) start[c + 1] += start[c];
  int W = 736;
  float *ds; int *dord, *dkey, *dst, *dc1, *di1, *dc2, *di2;
  CK(cudaMalloc(&ds, 12 * n)); CK(cudaMalloc(&dord, 4 * n)); CK(cudaMalloc(&dkey, 4 * n)); CK(cudaMalloc(&dst, 4 * (cells + 1)));
  CK(cudaMalloc(&dc1, 4 * n)); CK(cudaMalloc(&dc2, 4 * n)); CK(cudaMalloc(&di1, 4L * n * W)); CK(cudaMalloc(&di2, 4L * n * W));
  cudaMemcpy(ds, sorted.data(), 12 * n, cudaMemcpyHostToDevice); cudaMemcpy(dord, order.data(), 4 * n, cudaMemcpyHostToDevice);
  cudaMemcpy(dkey, key.data(), 4 * n, cudaMemcpyHostToDevice); cudaMemcpy(dst, start.data(), 4 * (cells + 1), cudaMemcpyHostToDevice);
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  auto time = [&](auto f) { f(); CK(cudaDeviceSynchronize()); cudaEventRecord(e0); for (int r = 0; r < 100; ++r) f(); cudaEventRecord(e1); cudaEventSynchronize(e1); float ms; cudaEventElapsedTime(&ms, e0, e1); return 10.0 * ms; };
  double t1 = time([&] { threadSearch<<<(n + 127) / 128, 128>>>(n, g, ds, dord, dkey, dst, W, dc1, di1); });
  double t2n = time([&] { warpSearch<false><<<(32 * n + 127) / 128, 128>>>(n, g, ds, dord, dkey, dst, W, dc2, di2); });
  printf("warp per particle without writing %.1f us\n", t2n);
  printf("csv,search,warp without writing,%.2f,%s,%.2f\n", reach, argc > 3 ? argv[3] : "2", t2n);
  int *deo, *del; CK(cudaMalloc(&deo, 4 * (n + 1))); CK(cudaMalloc(&del, 4 * exl.size()));
  cudaMemcpy(deo, exoff.data(), 4 * (n + 1), cudaMemcpyHostToDevice); cudaMemcpy(del, exl.data(), 4 * exl.size(), cudaMemcpyHostToDevice);
  double t2e = time([&] { warpSearchExcl<<<(32 * n + 127) / 128, 128>>>(n, g, ds, dord, dkey, dst, W, dc2, di2, deo, del); });
  printf("warp per particle with the excluded pairs %.1f us\n", t2e);
  printf("csv,search,warp with exclusions,%.2f,%s,%.2f\n", reach, argc > 3 ? argv[3] : "2", t2e);
  double t2 = time([&] { warpSearch<true><<<(32 * n + 127) / 128, 128>>>(n, g, ds, dord, dkey, dst, W, dc2, di2); });
  int mmax = 0; for (int c = 0; c < cells; ++c) mmax = std::max(mmax, start[c+1] - start[c]);
  // cnt[] of cellSearch holds 64 particles; a fuller cell skips it.
  bool fits = mmax <= 64;
  double t3 = fits ? time([&] { cellSearch<<<cells, 128>>>(n, g, ds, dord, dst, W, dc2, di2); }) : NAN;
  printf("block per cell %.1f us (at most %d particles in a cell)\n", t3, mmax);
  printf("csv,search,block per cell,%.2f,%s,%.2f\n", reach, argc > 3 ? argv[3] : "2", t3);
  std::vector<int> c1(n), c2(n), i1(1L * n * W), i2(1L * n * W);
  cudaMemcpy(c1.data(), dc1, 4 * n, cudaMemcpyDeviceToHost); cudaMemcpy(c2.data(), dc2, 4 * n, cudaMemcpyDeviceToHost);
  cudaMemcpy(i1.data(), di1, 4L * n * W, cudaMemcpyDeviceToHost); cudaMemcpy(i2.data(), di2, 4L * n * W, cudaMemcpyDeviceToHost);
  bool same = c1 == c2;
  for (int i = 0; same && i < n; ++i) for (int e = 0; e < std::min(c1[i], W); ++e) same &= i1[(size_t)i*W+e] == i2[(size_t)i*W+e];
  printf("reach %.1f, cells %dx%dx%d: thread per particle %.1f us, warp per particle %.1f us, same matrix: %s\n",
         reach, g.nx, g.ny, g.nz, t1, t2, same ? "yes" : "no");
  printf("csv,search,thread per particle,%.2f,%s,%.2f\ncsv,search,warp per particle,%.2f,%s,%.2f\n",
         reach, argc > 3 ? argv[3] : "2", t1, reach, argc > 3 ? argv[3] : "2", t2);
}
