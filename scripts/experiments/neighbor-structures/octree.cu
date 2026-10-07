// Prototype (2026-10-08, D[octane]): forces without a neighbor list, by a
// search in a region octree every step, as [Toutouni2026] describes it
// (Sections 3.2, 3.3, 3.6; Algorithms 1 and 2), against the same idea on a
// uniform grid of cells. A standalone program, not a template of MDIR.
//
//   octree <system.bin> <cutoff> [key=value ...]
//   octree gen <out.bin> lattice|random <m> <box> [jitter] [seed]
//
// T, the tree. A region octree over the periodic cell: a node has the
// bounds of its region, a pointer to its parent and one to its first child
// (the children of a node are contiguous); a leaf holds a block of MC
// indices of atoms and a count. A box that is not a cube halves only its
// longer axes at the upper levels, so that the leaves are near cubes (the
// paper has cubes only). The search is a thread per atom, in the order of
// an array of indices sorted along a Morton curve: from the leaf of the
// atom it climbs to the smallest ancestor whose region contains the query
// box of half-width r, then descends, pruning the nodes that the box does
// not meet, and at a leaf tests the atoms and sums the force (and the
// energy and virial) in registers: each pair is computed twice, and nothing
// is added atomically. The stack of the descent is the mask of the children
// yet to visit at each level, 8 bits a level in one 64-bit register, with
// the parent pointer to return (the paper says "a register-resident
// stack" and no more; an array indexed by a stack pointer lives in local
// memory, not in registers).
//
// The periodic boundary. The paper says "adjust query bounds for PBC"
// (Algorithm 1, line 11) and no more. Here a query box that leaves the
// cell is contained in no node but the root (which is the whole cell), so
// the climb ends at the root, and the descent tests a node against the box
// and against its image on the other side. The option prune=sphere tests
// the distance from the atom to the region instead (minimum image), which
// the paper does not do.
//
// The update. After the positions move: a thread per atom finds its leaf
// (climbing from the old one until a region contains the atom, then
// descending); a thread per leaf compacts its block, dropping the atoms
// that left; a thread per atom that moved appends itself to the new leaf
// with an atomic increment of the count. The paper removes and inserts in
// one kernel "with atomic operations" and does not say how a block stays
// dense; the three kernels here have no race and keep the counts exact.
//
// The capacity. MC = ceil(N * buffer / leaves) (Eq. 8 of the paper), raised
// to the fullest leaf for the timings, so that they have no overflow. The
// paper does not say what happens to an atom whose leaf is full. Here it
// goes to a list of overflow that every search reads in full, and it is
// tried again at each update: no atom is dropped, and the overflows are
// counted.
//
// G, the grid. The atoms sorted by cell (a counting sort on the device
// every step) and a thread or a warp per atom that reads the runs of cells
// within the cutoff, as search.cu does for the build of the neighbor
// matrix, and computes the forces instead of writing a list.
//
// The pair kernel is that of groups.cu (Lennard-Jones from the type tables
// and the direct sum of PME, f32; -DFAST_ERFC as there) when the system
// has charges, and plain Lennard-Jones of one type otherwise (argon).
//
// Keys: levels=3,4 (leaf levels along the longest axis; default: leaf side
// between 0.45 and 2.3 cutoffs), buffer=1.5, order=input|shuffle|morton
// (the order of the atoms in memory), check=1 (all pairs on the host, at
// most 60000 atoms), steps=N and equil=N (dynamics with the forces of T for
// a system without charges; displacements along fixed velocities
// otherwise), level=L (the level of the dynamics), reach=R (a list's reach,
// for its size and the steps between builds), divisors=1,2,3 (cells of G
// per cutoff), temp=, dt=, mass=, out=<file.bin> (the positions at the end).
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include <map>
#include <random>
#include <numeric>
#include <algorithm>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

static const float BETA = 0.3945f;
static const double KB = 0.0019872041, ACCEL = 4.184e-4;  // kcal/mol/K; (kcal/mol/A/u) to A/fs^2

// split: bit k set when the children halve axis k; 0 for a leaf, whose
// `first` is its block. The children of a node are first .. first + 2^popc(split) - 1.
struct Node { float lo[3], hi[3]; int parent, first, split; };
struct Par { float3 L, iL; float rq, rc2, a0, b0, eshift; int nt; const float *A, *B; const int *exoff, *exl; };
struct Acc { float fx, fy, fz, e, w; int cand, pairs, eb, ee, W; int *row; int xlo, xhi; };
struct Grid { int nx, ny, nz, fx, fy, fz, sx, sy, sz; float ilx, ily, ilz; };

// MODE 0: forces. 1: forces, energy, virial. 2: counts and the list of the pairs.
template <int MODE, bool PME, bool EXCL>
__device__ __forceinline__ void visit(const Par &P, Acc &s, int jref, const int *map, float4 pi, int ti, float4 pj, const int *ts) {
  float dx = pi.x - pj.x, dy = pi.y - pj.y, dz = pi.z - pj.z;
  dx -= P.L.x * rintf(dx * P.iL.x); dy -= P.L.y * rintf(dy * P.iL.y); dz -= P.L.z * rintf(dz * P.iL.z);
  float r2 = dx*dx + dy*dy + dz*dz;
  if (MODE == 2) ++s.cand;
  if (r2 >= P.rc2) return;
  if (EXCL || MODE == 2) {
    int j = map ? map[jref] : jref;
    // The partners of an atom are sorted: one between the least and the
    // greatest is compared with them.
    if (EXCL && j >= s.xlo && j <= s.xhi) { bool ex = false; for (int k = s.eb; k < s.ee; ++k) ex |= P.exl[k] == j; if (ex) return; }
    if (MODE == 2) { if (s.pairs < s.W) s.row[s.pairs] = j; ++s.pairs; return; }
  }
  float ri = rsqrtf(r2), r2i = ri * ri, r6i = r2i * r2i * r2i;
  float a = P.a0, b = P.b0;
  if (PME) { int tj = ts[jref]; a = P.A[ti * P.nt + tj]; b = P.B[ti * P.nt + tj]; }
  float fs = (12.f * a * r6i - 6.f * b) * r6i * r2i;
  if (MODE == 1) s.e += (a * r6i - b) * r6i - P.eshift;
  if (PME) {
    float r = r2 * ri, br = BETA * r, ex = __expf(-br * br), qq = pi.w * pj.w;
#ifdef FAST_ERFC
    float t = __frcp_rn(1.f + 0.3275911f * br);
    float erfcv = t * (0.254829592f + t * (-0.284496736f + t * (1.421413741f + t * (-1.453152027f + t * 1.061405429f)))) * ex;
#else
    float erfcv = erfcf(br);
#endif
    fs += qq * (erfcv * ri + 1.1283792f * BETA * ex) * r2i;
    if (MODE == 1) s.e += qq * erfcv * ri;
  }
  s.fx += fs * dx; s.fy += fs * dy; s.fz += fs * dz;
  if (MODE == 1) s.w += fs * r2;
}

__host__ __device__ inline int popc3(int m) { return (m & 1) + (m >> 1 & 1) + (m >> 2 & 1); }

// The leaf below node n that holds a point (the descent of FindLeafNode in
// Algorithm 2 of [Toutouni2026]).
__host__ __device__ inline int descend(const Node *nodes, int n, float x, float y, float z) {
  while (nodes[n].split) {
    int split = nodes[n].split, first = nodes[n].first, c = 0, pos = 0;
    const Node &f = nodes[first];
    if (split & 1) { c |= (x >= f.hi[0]) << pos; ++pos; }
    if (split & 2) { c |= (y >= f.hi[1]) << pos; ++pos; }
    if (split & 4) { c |= (z >= f.hi[2]) << pos; ++pos; }
    n = first + c;
  }
  return n;
}

template <int MODE, bool PME, bool EXCL>
__global__ void treeSearch(int n, Par P, const Node *nodes, const int *leafNode, const int *count, const int *slots, int MC,
                           const int *ovf, const int *novf, const int *ids, const float4 *xq, const int *ts, int sphere,
                           float *force, float *ev, int *stats, int *rows, int W) {
  int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= n) return;
  int i = ids[p];
  float4 pi = xq[i];
  int ti = PME ? ts[i] : 0;
  Acc s = {0, 0, 0, 0, 0, 0, 0, 0, 0, W, MODE == 2 ? rows + (size_t)i * W : nullptr, 1, 0};
  if (EXCL) { s.eb = P.exoff[i]; s.ee = P.exoff[i + 1]; if (s.ee > s.eb) { s.xlo = P.exl[s.eb]; s.xhi = P.exl[s.ee - 1]; } }
  float pc[3] = {pi.x, pi.y, pi.z}, L[3] = {P.L.x, P.L.y, P.L.z}, iL[3] = {P.iL.x, P.iL.y, P.iL.z};
  float qlo[3], qhi[3], qlo2[3], qhi2[3];
  for (int k = 0; k < 3; ++k) {
    qlo[k] = pc[k] - P.rq; qhi[k] = pc[k] + P.rq;
    float shift = qlo[k] < 0 ? L[k] : (qhi[k] > L[k] ? -L[k] : 0.f);
    qlo2[k] = qlo[k] + shift; qhi2[k] = qhi[k] + shift;
  }
  // The climb: to the smallest ancestor whose region contains the query box.
  int cur = leafNode[i], climbed = 0, leaves = 0;
  for (;;) {
    const Node &nd = nodes[cur];
    if (nd.parent < 0) break;
    if (nd.lo[0] <= qlo[0] && qhi[0] <= nd.hi[0] && nd.lo[1] <= qlo[1] && qhi[1] <= nd.hi[1] && nd.lo[2] <= qlo[2] && qhi[2] <= nd.hi[2]) break;
    cur = nd.parent; ++climbed;
  }
  auto leaf = [&](const Node &nd) {
    int b = nd.first, c = count[b];
    const int *sl = slots + (size_t)b * MC;
    for (int k = 0; k < c; ++k) { int j = sl[k]; if (j != i) visit<MODE, PME, EXCL>(P, s, j, nullptr, pi, ti, xq[j], ts); }
    ++leaves;
  };
  auto hit = [&](const Node &nd) -> bool {
    if (sphere) {
      float d2 = 0;
      for (int k = 0; k < 3; ++k) {
        float d = pc[k] - 0.5f * (nd.lo[k] + nd.hi[k]);
        d = fabsf(d - L[k] * rintf(d * iL[k])) - 0.5f * (nd.hi[k] - nd.lo[k]);
        if (d > 0) d2 += d * d;
      }
      return d2 <= P.rq * P.rq;
    }
    bool in = true;
    for (int k = 0; k < 3; ++k)
      in &= (nd.lo[k] <= qhi[k] && nd.hi[k] >= qlo[k]) || (nd.lo[k] <= qhi2[k] && nd.hi[k] >= qlo2[k]);
    return in;
  };
  auto expand = [&](const Node &nd) -> unsigned {
    unsigned m = 0; int nk = 1 << popc3(nd.split);
    for (int c = 0; c < nk; ++c) if (hit(nodes[nd.first + c])) m |= 1u << c;
    return m;
  };
  if (!nodes[cur].split) leaf(nodes[cur]);
  else {
    // The descent: the children yet to visit at depth d below the start are
    // 8 bits of `pend`.
    unsigned long long pend = 0; int d = 0;
    unsigned m = expand(nodes[cur]);
    for (;;) {
      if (!m) { if (!d) break; cur = nodes[cur].parent; --d; m = (unsigned)(pend >> (8 * d)) & 0xffu; continue; }
      int c = __ffs(m) - 1; m &= m - 1;
      int kid = nodes[cur].first + c;
      const Node &kn = nodes[kid];
      if (!kn.split) leaf(kn);
      else { pend = (pend & ~(0xffull << (8 * d))) | ((unsigned long long)m << (8 * d)); cur = kid; ++d; m = expand(kn); }
    }
  }
  int no = *novf;
  for (int k = 0; k < no; ++k) { int j = ovf[k]; if (j != i) visit<MODE, PME, EXCL>(P, s, j, nullptr, pi, ti, xq[j], ts); }
  if (MODE == 2) { stats[4*i] = s.cand; stats[4*i+1] = s.pairs; stats[4*i+2] = climbed; stats[4*i+3] = leaves; return; }
  force[3*i] = s.fx; force[3*i+1] = s.fy; force[3*i+2] = s.fz;
  if (MODE == 1) { ev[2*i] = 0.5f * s.e; ev[2*i+1] = 0.5f * s.w; }
}

// The update of the tree, in three kernels (see the head of the file).
__global__ void updFind(int n, const Node *nodes, const float4 *xq, const int *leafNode, int *newNode, int *hist) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float4 p = xq[i];
  int cur = leafNode[i], up = 0;
  for (;;) {
    const Node &nd = nodes[cur];
    if (nd.parent < 0 || (p.x >= nd.lo[0] && p.x < nd.hi[0] && p.y >= nd.lo[1] && p.y < nd.hi[1] && p.z >= nd.lo[2] && p.z < nd.hi[2])) break;
    cur = nd.parent; ++up;
  }
  if (up) { cur = descend(nodes, cur, p.x, p.y, p.z); if (hist) atomicAdd(hist + up, 1); }
  newNode[i] = cur;
}
__global__ void updCompact(int nleaves, int firstLeaf, int MC, const int *newNode, int *count, int *slots) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= nleaves) return;
  int c = count[b], w = 0, me = firstLeaf + b;
  int *sl = slots + (size_t)b * MC;
  for (int k = 0; k < c; ++k) { int j = sl[k]; if (newNode[j] == me) sl[w++] = j; }
  count[b] = w;
}
__global__ void updInsert(int n, int firstLeaf, int MC, const int *newNode, int *leafNode, int *count, int *slots,
                          int *inOvf, int *novf, int *ovf) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int nn = newNode[i];
  if (nn == leafNode[i] && !inOvf[i]) return;
  leafNode[i] = nn;
  int b = nn - firstLeaf;
  int s = atomicAdd(count + b, 1);
  if (s < MC) { slots[(size_t)b * MC + s] = i; inOvf[i] = 0; }
  else { atomicSub(count + b, 1); ovf[atomicAdd(novf, 1)] = i; inOvf[i] = 1; }
}

__device__ __forceinline__ uint32_t spread10(uint32_t v) {
  v &= 0x3ff; v = (v | (v << 16)) & 0x030000ff; v = (v | (v << 8)) & 0x0300f00f;
  v = (v | (v << 4)) & 0x030c30c3; v = (v | (v << 2)) & 0x09249249; return v;
}
// Morton codes of the positions, 10 bits an axis (Section 3.6.1 of [Toutouni2026]).
__global__ void mortonKeys(int n, const float4 *xq, float3 iL, uint32_t *keys) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float4 p = xq[i];
  uint32_t x = min(1023, (int)(p.x * iL.x * 1024.f)), y = min(1023, (int)(p.y * iL.y * 1024.f)), z = min(1023, (int)(p.z * iL.z * 1024.f));
  keys[i] = spread10(x) | spread10(y) << 1 | spread10(z) << 2;
}

__device__ __forceinline__ int cellOf(const Grid &g, float4 p, int &cx, int &cy, int &cz) {
  cx = min(g.nx - 1, (int)(p.x * g.ilx * g.nx)); cy = min(g.ny - 1, (int)(p.y * g.ily * g.ny)); cz = min(g.nz - 1, (int)(p.z * g.ilz * g.nz));
  return (cz * g.ny + cy) * g.nx + cx;
}
__global__ void gridCount(int n, Grid g, const float4 *xq, int *key, int *cnt) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int cx, cy, cz; int c = cellOf(g, xq[i], cx, cy, cz);
  key[i] = c; atomicAdd(cnt + c, 1);
}
__global__ void gridFill(int n, const float4 *xq, const int *ts, const int *key, int *cursor, int *order, float4 *sxq, int *sts) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int p = atomicAdd(cursor + key[i], 1);
  order[p] = i; sxq[p] = xq[i]; sts[p] = ts[i];
}

// A thread per atom over the runs of cells within the cutoff (threadSearch
// of search.cu), computing the forces.
template <int MODE, bool PME, bool EXCL>
__global__ void gridSearch(int n, Par P, Grid g, const float4 *sxq, const int *sts, const int *order, const int *start,
                           float *force, float *ev, int *stats, int *rows, int W) {
  int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= n) return;
  int i = order[p];
  float4 pi = sxq[p];
  int ti = PME ? sts[p] : 0;
  Acc s = {0, 0, 0, 0, 0, 0, 0, 0, 0, W, MODE == 2 ? rows + (size_t)i * W : nullptr, 1, 0};
  if (EXCL) { s.eb = P.exoff[i]; s.ee = P.exoff[i + 1]; if (s.ee > s.eb) { s.xlo = P.exl[s.eb]; s.xhi = P.exl[s.ee - 1]; } }
  int cx, cy, cz; cellOf(g, pi, cx, cy, cz);
  for (int r = 0; r < g.sy * g.sz; ++r) {
    int oz = r / g.sy, oy = r % g.sy;
    int zc = (cz + g.fz + oz) % g.nz, yc = (cy + g.fy + oy) % g.ny;
    int first = (zc * g.ny + yc) * g.nx;
    int xb = (cx + g.fx) % g.nx, xe = xb + g.sx;
    int stop = xe > g.nx ? g.nx : xe, rest = xe > g.nx ? xe - g.nx : 0;
    for (int part = 0; part < 2; ++part) {
      int from = part ? 0 : xb, to = part ? rest : stop;
      int b = start[first + from], e = start[first + to];
      for (int q = b; q < e; ++q) if (q != p) visit<MODE, PME, EXCL>(P, s, q, order, pi, ti, sxq[q], sts);
    }
  }
  if (MODE == 2) { stats[4*i] = s.cand; stats[4*i+1] = s.pairs; return; }
  force[3*i] = s.fx; force[3*i+1] = s.fy; force[3*i+2] = s.fz;
  if (MODE == 1) { ev[2*i] = 0.5f * s.e; ev[2*i+1] = 0.5f * s.w; }
}

// A warp per atom, a lane a candidate (warpSearch of search.cu).
template <bool PME, bool EXCL>
__global__ void gridWarp(int n, Par P, Grid g, const float4 *sxq, const int *sts, const int *order, const int *start, float *force) {
  int p = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x % 32;
  if (p >= n) return;
  int i = order[p];
  float4 pi = sxq[p];
  int ti = PME ? sts[p] : 0;
  Acc s = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, nullptr, 1, 0};
  if (EXCL) { s.eb = P.exoff[i]; s.ee = P.exoff[i + 1]; if (s.ee > s.eb) { s.xlo = P.exl[s.eb]; s.xhi = P.exl[s.ee - 1]; } }
  int cx, cy, cz; cellOf(g, pi, cx, cy, cz);
  for (int r = 0; r < g.sy * g.sz; ++r) {
    int oz = r / g.sy, oy = r % g.sy;
    int zc = (cz + g.fz + oz) % g.nz, yc = (cy + g.fy + oy) % g.ny;
    int first = (zc * g.ny + yc) * g.nx;
    int xb = (cx + g.fx) % g.nx, xe = xb + g.sx;
    int stop = xe > g.nx ? g.nx : xe, rest = xe > g.nx ? xe - g.nx : 0;
    for (int part = 0; part < 2; ++part) {
      int from = part ? 0 : xb, to = part ? rest : stop;
      int b = start[first + from], e = start[first + to];
      for (int q = b + lane; q < e; q += 32) if (q != p) visit<0, PME, EXCL>(P, s, q, order, pi, ti, sxq[q], sts);
    }
  }
  for (int o = 16; o; o >>= 1) {
    s.fx += __shfl_xor_sync(0xffffffff, s.fx, o); s.fy += __shfl_xor_sync(0xffffffff, s.fy, o); s.fz += __shfl_xor_sync(0xffffffff, s.fz, o);
  }
  if (lane == 0) { force[3*i] = s.fx; force[3*i+1] = s.fy; force[3*i+2] = s.fz; }
}

__device__ __forceinline__ float wrapf(float x, float l, float il) { x -= l * floorf(x * il); return x >= l ? 0.f : x; }
__global__ void drift(int n, float4 *xq, const float *v, float dt, float3 L, float3 iL) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float4 p = xq[i];
  p.x = wrapf(p.x + v[3*i] * dt, L.x, iL.x); p.y = wrapf(p.y + v[3*i+1] * dt, L.y, iL.y); p.z = wrapf(p.z + v[3*i+2] * dt, L.z, iL.z);
  xq[i] = p;
}
__global__ void kick(int n, float *v, const float *f, float c) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < 3 * n) v[i] += c * f[i];
}
__global__ void scale(int n, float *v, float c) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < 3 * n) v[i] *= c;
}
__global__ void maxDisp(int n, const float4 *xq, const float4 *ref, float3 L, float3 iL, unsigned *mx) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float dx = xq[i].x - ref[i].x, dy = xq[i].y - ref[i].y, dz = xq[i].z - ref[i].z;
  dx -= L.x * rintf(dx * iL.x); dy -= L.y * rintf(dy * iL.y); dz -= L.z * rintf(dz * iL.z);
  atomicMax(mx, __float_as_uint(sqrtf(dx*dx + dy*dy + dz*dz)));
}

template <class T> static T *dalloc(size_t count) { T *d; CK(cudaMalloc(&d, sizeof(T) * std::max<size_t>(1, count))); return d; }
template <class T> static T *up(const std::vector<T> &v) {
  T *d = dalloc<T>(v.size()); CK(cudaMemcpy(d, v.data(), sizeof(T) * v.size(), cudaMemcpyHostToDevice)); return d;
}
template <class T> static std::vector<T> down(const T *d, size_t count) {
  std::vector<T> v(count); CK(cudaMemcpy(v.data(), d, sizeof(T) * count, cudaMemcpyDeviceToHost)); return v;
}

struct System {
  int n, ntypes; double box[3];
  std::vector<double> x; std::vector<float> q, ta, tb; std::vector<int> type, exoff, exl;
};
static void writeSystem(const char *path, const System &s) {
  FILE *f = fopen(path, "wb");
  if (!f) { printf("cannot write %s\n", path); exit(1); }
  fwrite(&s.n, 4, 1, f); fwrite(&s.ntypes, 4, 1, f); fwrite(s.box, 8, 3, f);
  fwrite(s.x.data(), 8, 3 * s.n, f); fwrite(s.q.data(), 4, s.n, f); fwrite(s.type.data(), 4, s.n, f);
  fwrite(s.ta.data(), 4, s.ta.size(), f); fwrite(s.tb.data(), 4, s.tb.size(), f);
  fwrite(s.exoff.data(), 4, s.n + 1, f); fwrite(s.exl.data(), 4, s.exl.size(), f);
  fclose(f);
}
static System readSystem(const char *path) {
  System s; FILE *fp = fopen(path, "rb");
  if (!fp) { printf("cannot read %s\n", path); exit(1); }
  fread(&s.n, 4, 1, fp); fread(&s.ntypes, 4, 1, fp); fread(s.box, 8, 3, fp);
  int n = s.n, nt = s.ntypes;
  s.x.resize(3 * n); s.q.resize(n); s.type.resize(n); s.ta.resize(nt * nt); s.tb.resize(nt * nt);
  fread(s.x.data(), 8, 3 * n, fp); fread(s.q.data(), 4, n, fp); fread(s.type.data(), 4, n, fp);
  fread(s.ta.data(), 4, nt * nt, fp); fread(s.tb.data(), 4, nt * nt, fp);
  s.exoff.resize(n + 1); fread(s.exoff.data(), 4, n + 1, fp);
  s.exl.resize(s.exoff[n]); fread(s.exl.data(), 4, s.exoff[n], fp);
  fclose(fp);
  return s;
}

// Argon as [Toutouni2026] has it (Section 4.1: epsilon / k_B = 114.5 K,
// sigma = 3.374 A), in the layout of prep.py: a simple cubic lattice of
// m^3 sites with a uniform jitter on each coordinate, or uniform random
// positions (for the histograms of the levels only: atoms may overlap).
static int generate(int argc, char **argv) {
  if (argc < 6) { printf("octree gen <out.bin> lattice|random <m> <box> [jitter] [seed]\n"); return 1; }
  bool random = !strcmp(argv[3], "random");
  int m = atoi(argv[4]); double D = atof(argv[5]), jitter = argc > 6 ? atof(argv[6]) : 0.0;
  std::mt19937_64 rng(argc > 7 ? atoll(argv[7]) : 1);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  System s; s.n = m * m * m; s.ntypes = 1; s.box[0] = s.box[1] = s.box[2] = D;
  s.x.resize(3 * (size_t)s.n);
  for (int iz = 0; iz < m; ++iz) for (int iy = 0; iy < m; ++iy) for (int ix = 0; ix < m; ++ix) {
    size_t i = ((size_t)iz * m + iy) * m + ix; int c[3] = {ix, iy, iz};
    for (int k = 0; k < 3; ++k) s.x[3*i+k] = random ? D * u(rng) : D * (c[k] + 0.5) / m + jitter * (2 * u(rng) - 1);
  }
  double eps = 114.5 * KB, s6 = std::pow(3.374, 6);
  s.q.assign(s.n, 0.f); s.type.assign(s.n, 0); s.ta = {(float)(4 * eps * s6 * s6)}; s.tb = {(float)(4 * eps * s6)};
  s.exoff.assign(s.n + 1, 0);
  writeSystem(argv[2], s);
  printf("%d atoms, box %.3f A, %.4e atoms/A^3, %s\n", s.n, D, s.n / (D * D * D), random ? "uniform random" : "jittered lattice");
  return 0;
}

int main(int argc, char **argv) {
  if (argc > 1 && !strcmp(argv[1], "gen")) return generate(argc, argv);
  if (argc < 3) { printf("octree <system.bin> <cutoff> [key=value ...]\n"); return 1; }
  std::map<std::string, std::string> opt;
  for (int a = 3; a < argc; ++a) { std::string t = argv[a]; size_t e = t.find('='); if (e != std::string::npos) opt[t.substr(0, e)] = t.substr(e + 1); }
  auto num = [&](const char *k, double d) { return opt.count(k) ? atof(opt[k].c_str()) : d; };
  auto list = [&](const char *k) { std::vector<int> v; if (opt.count(k)) { std::string t = opt[k]; size_t p = 0; while (p < t.size()) { v.push_back(atoi(t.c_str() + p)); p = t.find(',', p); if (p == std::string::npos) break; ++p; } } return v; };
  std::string name = argv[1]; { size_t sl = name.rfind('/'); if (sl != std::string::npos) name = name.substr(sl + 1); size_t dot = name.rfind('.'); if (dot != std::string::npos) name = name.substr(0, dot); }
  const float RC = atof(argv[2]);
  const double buffer = num("buffer", 1.5);
  const bool check = num("check", 0) != 0;
  const std::string order = opt.count("order") ? opt["order"] : "input";

  System S = readSystem(argv[1]);
  const int n = S.n, nt = S.ntypes;
  double box[3] = {S.box[0], S.box[1], S.box[2]};
  for (int k = 0; k < 3; ++k) if (box[k] < 2.0 * RC * 1.001) { printf("the cell is narrower than two cutoffs\n"); return 1; }
  bool PME = false; for (float c : S.q) PME |= c != 0.f;
  const bool EXCL = S.exoff[n] > 0;
  const float3 L = make_float3(box[0], box[1], box[2]), iL = make_float3(1 / L.x, 1 / L.y, 1 / L.z);
  const float Lf[3] = {L.x, L.y, L.z};
  auto wrapHost = [&](double v, int k) { float w = (float)(v - box[k] * std::floor(v / box[k])); return (w >= Lf[k] || w < 0) ? 0.f : w; };

  // The order of the atoms in memory: that of the input, a random one, or
  // that of a Morton curve (30 bits).
  std::vector<int> perm(n); std::iota(perm.begin(), perm.end(), 0);
  if (order == "shuffle") { std::mt19937 rng(7); std::shuffle(perm.begin(), perm.end(), rng); }
  if (order == "morton") {
    std::vector<uint32_t> key(n);
    auto sp = [](uint32_t v) { uint32_t r = 0; for (int b = 0; b < 10; ++b) r |= ((v >> b) & 1u) << (3 * b); return r; };
    for (int i = 0; i < n; ++i) { uint32_t c[3]; for (int k = 0; k < 3; ++k) c[k] = std::min(1023, (int)(wrapHost(S.x[3*i+k], k) / Lf[k] * 1024.f)); key[i] = sp(c[0]) | sp(c[1]) << 1 | sp(c[2]) << 2; }
    std::stable_sort(perm.begin(), perm.end(), [&](int a, int b) { return key[a] < key[b]; });
  }
  std::vector<int> inv(n); for (int p = 0; p < n; ++p) inv[perm[p]] = p;
  std::vector<float4> xq(n); std::vector<int> ts(n), exoff(n + 1, 0), exl; exl.reserve(S.exl.size());
  for (int p = 0; p < n; ++p) {
    int i = perm[p];
    xq[p] = make_float4(wrapHost(S.x[3*i], 0), wrapHost(S.x[3*i+1], 1), wrapHost(S.x[3*i+2], 2), S.q[i]); ts[p] = S.type[i];
    for (int k = S.exoff[i]; k < S.exoff[i+1]; ++k) exl.push_back(inv[S.exl[k]]);
    std::sort(exl.begin() + exoff[p], exl.end()); exoff[p + 1] = exl.size();
  }
  const double rho = n / (box[0] * box[1] * box[2]);
  printf("%s: %d atoms, cell %.3f x %.3f x %.3f A, %.4e atoms/A^3, cutoff %.3f A, %s, %zu excluded pairs, order %s\n",
         name.c_str(), n, box[0], box[1], box[2], rho, RC, PME ? "Lennard-Jones and the direct sum of PME" : "Lennard-Jones", exl.size() / 2, order.c_str());

  Par P; P.L = L; P.iL = iL; P.rq = RC * 1.00001f; P.rc2 = RC * RC; P.nt = nt;
  P.a0 = S.ta[0]; P.b0 = S.tb[0];
  P.eshift = PME ? 0.f : (float)((S.ta[0] / std::pow((double)RC, 6) - S.tb[0]) / std::pow((double)RC, 6));
  float4 *dxq = up(xq), *dxq0 = up(xq); int *dts = up(ts);
  P.A = up(S.ta); P.B = up(S.tb); P.exoff = up(exoff); P.exl = up(exl);
  float *dforce = dalloc<float>(3 * (size_t)n), *dev = dalloc<float>(2 * (size_t)n);
  int *dstats = dalloc<int>(4 * (size_t)n);
  const int W = check ? 768 : 1;
  int *drows = dalloc<int>((size_t)n * W);
  int *dids = dalloc<int>(n), *didsId = dalloc<int>(n); uint32_t *dkeys = dalloc<uint32_t>(n);
  { std::vector<int> id(n); std::iota(id.begin(), id.end(), 0); CK(cudaMemcpy(didsId, id.data(), 4 * (size_t)n, cudaMemcpyHostToDevice)); }
  const int T = 128; auto blocks = [&](size_t c) { return (unsigned)((c + T - 1) / T); };
  cudaEvent_t e0, e1, e2, e3; cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventCreate(&e2); cudaEventCreate(&e3);
  const int repsMax = (int)num("reps", 200);
  // The time of a launch in microseconds: one launch to warm, one to choose
  // the number of repeats (about half a second in all), then the repeats
  // between two events, as the other prototypes do.
  auto timeK = [&](auto f) {
    f(); CK(cudaDeviceSynchronize());
    float ms; cudaEventRecord(e0); f(); cudaEventRecord(e1); CK(cudaEventSynchronize(e1)); cudaEventElapsedTime(&ms, e0, e1);
    int reps = std::max(5, std::min(repsMax, (int)(500.0 / std::max(ms, 1e-3f))));
    cudaEventRecord(e0); for (int r = 0; r < reps; ++r) f(); cudaEventRecord(e1); CK(cudaEventSynchronize(e1)); cudaEventElapsedTime(&ms, e0, e1);
    return 1000.0 * ms / reps;
  };
  auto csv = [&](const std::string &variant, const std::string &param, const char *quantity, double value) {
    printf("csv,%s,%s,%s,%s,%s,%.6g\n", name.c_str(), order.c_str(), variant.c_str(), param.c_str(), quantity, value);
  };
  auto sortIds = [&](int *ids) {
    mortonKeys<<<blocks(n), T>>>(n, dxq, iL, dkeys);
    thrust::device_ptr<int> ip(ids); thrust::device_ptr<uint32_t> kp(dkeys);
    thrust::sequence(ip, ip + n); thrust::sort_by_key(kp, kp + n, ip);
  };

  // The reference: all pairs on the host in f64, from the f32 positions.
  struct Ref { std::vector<std::vector<int>> rows; std::vector<double> f, fx; double e = 0, ex = 0; };
  auto pairRef = [&](const std::vector<float4> &pos, int i, int j, bool excl, double *f, double &e) {
    double d[3] = {(double)pos[i].x - pos[j].x, (double)pos[i].y - pos[j].y, (double)pos[i].z - pos[j].z}, r2 = 0;
    for (int k = 0; k < 3; ++k) { d[k] -= (double)Lf[k] * std::nearbyint(d[k] / (double)Lf[k]); r2 += d[k] * d[k]; }
    if (r2 >= (double)RC * RC) return false;
    if (excl) return true;
    double r = std::sqrt(r2), ri = 1 / r, r2i = ri * ri, r6i = r2i * r2i * r2i;
    double a = PME ? S.ta[ts[i] * nt + ts[j]] : S.ta[0], b = PME ? S.tb[ts[i] * nt + ts[j]] : S.tb[0];
    double fs = (12 * a * r6i - 6 * b) * r6i * r2i; e += (a * r6i - b) * r6i - P.eshift;
    if (PME) {
      double br = (double)BETA * r, ex = std::exp(-br * br), qq = (double)pos[i].w * pos[j].w;
#ifdef FAST_ERFC
      double t = 1 / (1 + 0.3275911 * br);
      double erfcv = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429)))) * ex;
#else
      double erfcv = std::erfc(br);
#endif
      fs += qq * (erfcv * ri + 1.1283792 * BETA * ex) * r2i; e += qq * erfcv * ri;
    }
    for (int k = 0; k < 3; ++k) { f[3*i+k] += fs * d[k]; f[3*j+k] -= fs * d[k]; }
    return true;
  };
  auto reference = [&](const std::vector<float4> &pos) {
    Ref R; R.rows.resize(n); R.f.assign(3 * (size_t)n, 0); R.fx.assign(3 * (size_t)n, 0);
    for (int i = 0; i < n; ++i) for (int j = i + 1; j < n; ++j) {
      if (!pairRef(pos, i, j, false, R.f.data(), R.e)) continue;
      R.rows[i].push_back(j); R.rows[j].push_back(i);
      bool ex = std::binary_search(exl.begin() + exoff[i], exl.begin() + exoff[i+1], j);
      if (!ex) pairRef(pos, i, j, false, R.fx.data(), R.ex);
    }
    return R;
  };
  // The pairs of a kernel (MODE 2) against the reference: pairs missing,
  // pairs added, and the largest distance of such a pair from the cutoff.
  auto comparePairs = [&](const Ref &R, const std::vector<float4> &pos, const char *what) {
    std::vector<int> st = down(dstats, 4 * (size_t)n), rows = down(drows, (size_t)n * W);
    size_t missing = 0, added = 0, total = 0; double far = 0; int wide = 0;
    auto dist = [&](int i, int j) { double r2 = 0; for (int k = 0; k < 3; ++k) { double d = (&pos[i].x)[k] - (double)(&pos[j].x)[k]; d -= (double)Lf[k] * std::nearbyint(d / (double)Lf[k]); r2 += d * d; } return std::sqrt(r2); };
    for (int i = 0; i < n; ++i) {
      int c = st[4*i+1]; if (c > W) { ++wide; c = W; }
      std::vector<int> row(rows.begin() + (size_t)i * W, rows.begin() + (size_t)i * W + c); std::sort(row.begin(), row.end());
      std::vector<int> a, b;
      std::set_difference(R.rows[i].begin(), R.rows[i].end(), row.begin(), row.end(), std::back_inserter(a));
      std::set_difference(row.begin(), row.end(), R.rows[i].begin(), R.rows[i].end(), std::back_inserter(b));
      missing += a.size(); added += b.size(); total += R.rows[i].size();
      for (int j : a) far = std::max(far, std::fabs(dist(i, j) - RC));
      for (int j : b) far = std::max(far, std::fabs(dist(i, j) - RC));
    }
    printf("  exactness, %s: %zu pairs of the reference (each twice), missing %zu, added %zu, farthest of those from the cutoff %.1e A%s\n",
           what, total, missing, added, far, wide ? ", ROWS OVER THE WIDTH" : "");
    return missing + added;
  };
  auto compareForces = [&](const std::vector<double> &ref, const char *what) {
    std::vector<float> f = down(dforce, 3 * (size_t)n); double num2 = 0, den = 0;
    for (size_t k = 0; k < 3 * (size_t)n; ++k) { double d = f[k] - ref[k]; num2 += d * d; den += ref[k] * ref[k]; }
    printf("  forces, %s: relative difference from the reference %.2e\n", what, std::sqrt(num2 / den));
  };
  Ref R0;
  const bool small = n <= 60000;
  if (check && small) R0 = reference(xq);
  if (check && !small) printf("  more than 60000 atoms: no reference over all pairs; T and G are compared with each other\n");

  // ---- T: the tree ----
  struct HostTree { std::vector<Node> nodes; int H, Lk[3], nl[3], firstLeaf, nleaves; double side[3]; };
  auto buildTree = [&](int Lmax) {
    HostTree t; double bmax = std::max(box[0], std::max(box[1], box[2]));
    for (int k = 0; k < 3; ++k) { t.Lk[k] = std::max(0, Lmax - (int)std::lround(std::log2(bmax / box[k]))); t.nl[k] = 1 << t.Lk[k]; t.side[k] = box[k] / t.nl[k]; }
    t.H = Lmax;
    struct Q { int h, c[3]; }; std::vector<Q> queue;
    Node root; for (int k = 0; k < 3; ++k) { root.lo[k] = 0; root.hi[k] = Lf[k]; } root.parent = -1; root.first = 0; root.split = 0;
    t.nodes.push_back(root); queue.push_back({t.H, {0, 0, 0}}); t.firstLeaf = -1;
    for (size_t qi = 0; qi < queue.size(); ++qi) {
      Q cur = queue[qi];
      if (cur.h == 0) { if (t.firstLeaf < 0) t.firstLeaf = qi; t.nodes[qi].first = qi - t.firstLeaf; continue; }
      int split = 0; for (int k = 0; k < 3; ++k) if (cur.h <= t.Lk[k]) split |= 1 << k;
      t.nodes[qi].split = split; t.nodes[qi].first = t.nodes.size();
      for (int c = 0; c < (1 << popc3(split)); ++c) {
        Q kid; kid.h = cur.h - 1; Node nd; nd.parent = qi; nd.first = 0; nd.split = 0; int pos = 0;
        for (int k = 0; k < 3; ++k) {
          bool sp = split >> k & 1; int bit = sp ? (c >> pos++ & 1) : 0;
          kid.c[k] = sp ? 2 * cur.c[k] + bit : cur.c[k];
          int d = std::max(1, t.nl[k] >> kid.h);
          nd.lo[k] = (float)(box[k] * kid.c[k] / d); nd.hi[k] = kid.c[k] + 1 == d ? Lf[k] : (float)(box[k] * (kid.c[k] + 1) / d);
        }
        t.nodes.push_back(nd); queue.push_back(kid);
      }
    }
    t.nleaves = t.nodes.size() - t.firstLeaf;
    return t;
  };
  struct DevTree { Node *nodes = nullptr; int *leafNode, *newNode, *count, *slots, *ovf, *novf, *inOvf; int MC, nleaves, firstLeaf, nnodes; size_t bytes; };
  // The atoms into the leaves on the host, as the insertions of the update
  // would do: an atom whose leaf is full goes to the list of overflow.
  auto fillTree = [&](const HostTree &t, const std::vector<float4> &pos, int MC, int &novf) {
    DevTree d; d.MC = MC; d.nleaves = t.nleaves; d.firstLeaf = t.firstLeaf; d.nnodes = t.nodes.size();
    std::vector<int> leafNode(n), count(t.nleaves, 0), slots((size_t)t.nleaves * MC, -1), ovf(n, -1), inOvf(n, 0);
    novf = 0;
    for (int i = 0; i < n; ++i) {
      int ln = descend(t.nodes.data(), 0, pos[i].x, pos[i].y, pos[i].z), b = ln - t.firstLeaf; leafNode[i] = ln;
      if (count[b] < MC) slots[(size_t)b * MC + count[b]++] = i; else { ovf[novf++] = i; inOvf[i] = 1; }
    }
    d.nodes = up(t.nodes); d.leafNode = up(leafNode); d.newNode = up(leafNode); d.count = up(count); d.slots = up(slots);
    d.ovf = up(ovf); d.inOvf = up(inOvf); d.novf = up(std::vector<int>{novf});
    // Bytes that the method needs on the device: nodes, counts, blocks, the
    // leaf of each atom (old and new), the sorted indices and their keys.
    d.bytes = sizeof(Node) * t.nodes.size() + 4 * (size_t)t.nleaves + 4 * (size_t)t.nleaves * MC + 4 * 4 * (size_t)n;
    return d;
  };
  auto freeTree = [&](DevTree &d) { cudaFree(d.nodes); cudaFree(d.leafNode); cudaFree(d.newNode); cudaFree(d.count); cudaFree(d.slots); cudaFree(d.ovf); cudaFree(d.inOvf); cudaFree(d.novf); };
  int *dhist = dalloc<int>(16);
  auto update = [&](DevTree &d, bool stats) {
    updFind<<<blocks(n), T>>>(n, d.nodes, dxq, d.leafNode, d.newNode, stats ? dhist : nullptr);
    cudaEventRecord(e1);
    updCompact<<<blocks(d.nleaves), T>>>(d.nleaves, d.firstLeaf, d.MC, d.newNode, d.count, d.slots);
    cudaEventRecord(e2);
    cudaMemsetAsync(d.novf, 0, 4);
    updInsert<<<blocks(n), T>>>(n, d.firstLeaf, d.MC, d.newNode, d.leafNode, d.count, d.slots, d.inOvf, d.novf, d.ovf);
  };
  auto search = [&](DevTree &d, int mode, bool excl, int sphere, const int *ids) {
#define TS(M, PM, EX) treeSearch<M, PM, EX><<<blocks(n), T>>>(n, P, d.nodes, d.leafNode, d.count, d.slots, d.MC, d.ovf, d.novf, ids, dxq, dts, sphere, dforce, dev, dstats, drows, W)
    if (PME) { if (mode == 2) TS(2, true, false); else if (mode == 1) { if (excl) TS(1, true, true); else TS(1, true, false); } else { if (excl) TS(0, true, true); else TS(0, true, false); } }
    else { if (mode == 2) TS(2, false, false); else if (mode == 1) TS(1, false, false); else TS(0, false, false); }
#undef TS
  };
  // Velocities from the Maxwell-Boltzmann distribution, in A/fs.
  const double temp = num("temp", PME ? 300.0 : 120.0), dt = num("dt", PME ? 2.0 : 1.0), mass = num("mass", PME ? 12.0 : 39.948);
  std::vector<float> vel(3 * (size_t)n);
  { std::mt19937_64 rng(11); std::normal_distribution<double> g(0.0, std::sqrt(KB * temp / mass * ACCEL)); double com[3] = {0, 0, 0};
    for (size_t k = 0; k < vel.size(); ++k) { vel[k] = g(rng); com[k % 3] += vel[k]; }
    for (size_t k = 0; k < vel.size(); ++k) vel[k] -= com[k % 3] / n; }
  float *dvel = up(vel);

  std::vector<int> levels = list("levels");
  if (levels.empty()) {
    double bmax = std::max(box[0], std::max(box[1], box[2]));
    for (int l = 1; l <= 8; ++l) { double side = bmax / (1 << l); if (side >= 0.45 * RC && side <= 2.3 * RC) levels.push_back(l); }
  }
  double bestT = 1e30; int bestLevel = -1, bestSphere = 0;
  std::vector<float> fT;  // the forces of T, for the comparison with G when there is no reference
  std::vector<int> pairsT;
  sortIds(dids);
  double tsort = timeK([&] { sortIds(dids); });
  double tkeys = timeK([&] { mortonKeys<<<blocks(n), T>>>(n, dxq, iL, dkeys); });
  printf("T: Morton keys and the sort of the indices (thrust::sort_by_key) %.1f us, of which the keys %.1f us\n", tsort, tkeys);
  csv("T", "", "sort_us", tsort); csv("T", "", "keys_us", tkeys);
  for (int lev : levels) {
    if (lev > 8) { printf("level %d: more than 8 levels do not fit the 64-bit stack\n", lev); continue; }
    HostTree t = buildTree(lev);
    std::vector<int> occ(t.nleaves, 0);
    for (int i = 0; i < n; ++i) ++occ[descend(t.nodes.data(), 0, xq[i].x, xq[i].y, xq[i].z) - t.firstLeaf];
    int maxocc = *std::max_element(occ.begin(), occ.end());
    double mean = (double)n / t.nleaves; int MCpaper = (int)std::ceil(n * buffer / t.nleaves);
    size_t over = 0, overLeaves = 0; for (int c : occ) if (c > MCpaper) { over += c - MCpaper; ++overLeaves; }
    int MC = std::max(MCpaper, maxocc);
    std::string par = "level=" + std::to_string(lev);
    printf("T level %d (%d x %d x %d leaves of %.2f x %.2f x %.2f A, %zu nodes): %.2f atoms a leaf, the fullest %d; Eq. 8 with buffer %.2f gives MC = %d, over which are %zu atoms in %zu leaves; buffer for no overflow %.2f; capacity used %d\n",
           lev, t.nl[0], t.nl[1], t.nl[2], t.side[0], t.side[1], t.side[2], t.nodes.size(), mean, maxocc, buffer, MCpaper, over, overLeaves, maxocc / mean, MC);
    csv("T", par, "leaf_side_A", t.side[0]); csv("T", par, "atoms_per_leaf", mean); csv("T", par, "fullest_leaf", maxocc);
    csv("T", par, "capacity_eq8", MCpaper); csv("T", par, "atoms_over_eq8", over); csv("T", par, "capacity_used", MC);
    if ((size_t)t.nleaves * MC * 4 > (size_t)8 << 30) { printf("  blocks over 8 GB: skipped\n"); continue; }
    int novf; DevTree d = fillTree(t, xq, MC, novf);
    printf("  memory of T on the device: %.2f MB (nodes %.2f, counts and blocks %.2f, leaf of each atom old and new, sorted indices, keys %.2f)\n",
           d.bytes / 1e6, sizeof(Node) * t.nodes.size() / 1e6, (4.0 * t.nleaves + 4.0 * t.nleaves * MC) / 1e6, 16.0 * n / 1e6);
    csv("T", par, "bytes", d.bytes);

    // Counts, and the levels of the climb.
    search(d, 2, false, 0, dids); CK(cudaDeviceSynchronize());
    std::vector<int> st = down(dstats, 4 * (size_t)n);
    double cand = 0, pairs = 0, leavesBox = 0; std::vector<size_t> hClimb(10, 0), hPaper(10, 0), hPbc(10, 0); size_t crossing = 0;
    for (int i = 0; i < n; ++i) {
      cand += st[4*i]; pairs += st[4*i+1]; leavesBox += st[4*i+3]; ++hClimb[st[4*i+2]];
      // The same by arithmetic on the leaf coordinates: the height of the
      // smallest ancestor that holds both ends of the box on each axis.
      // `paper`: the cell not periodic, the box cut at its faces (the model
      // of Eq. 1 to 7 of [Toutouni2026]). `pbc`: a box that leaves the cell
      // is held by the root alone.
      int hp = 0, hb = 0; bool cross = false;
      for (int k = 0; k < 3; ++k) {
        int lo = (int)std::floor(((double)(&xq[i].x)[k] - P.rq) / t.side[k]), hi = (int)std::floor(((double)(&xq[i].x)[k] + P.rq) / t.side[k]);
        if (lo < 0 || hi >= t.nl[k]) cross = true;
        lo = std::max(lo, 0); hi = std::min(hi, t.nl[k] - 1);
        int h = 0; while ((lo >> h) != (hi >> h)) ++h;
        hp = std::max(hp, h);
      }
      hb = cross ? t.H : hp; crossing += cross; ++hPaper[hp]; ++hPbc[hb];
    }
    if (pairsT.empty()) { pairsT.resize(n); for (int i = 0; i < n; ++i) pairsT[i] = st[4*i+1]; }
    search(d, 2, false, 1, dids); CK(cudaDeviceSynchronize());
    std::vector<int> st2 = down(dstats, 4 * (size_t)n); double candS = 0, leavesS = 0; for (int i = 0; i < n; ++i) { candS += st2[4*i]; leavesS += st2[4*i+3]; }
    printf("  candidates an atom %.1f (box), %.1f (sphere); leaves visited %.1f, %.1f; pairs within the cutoff an atom %.2f; atoms whose query box leaves the cell %.2f%%\n",
           cand / n, candS / n, leavesBox / n, leavesS / n, pairs / n, 100.0 * crossing / n);
    csv("T", par, "candidates_per_atom_box", cand / n); csv("T", par, "candidates_per_atom_sphere", candS / n); csv("T", par, "pairs_per_atom", pairs / n);
    csv("T", par, "query_box_leaves_cell_fraction", (double)crossing / n);
    auto hist = [&](const char *what, const char *key, const std::vector<size_t> &h, size_t total) {
      printf("  %s:", what); double m = 0;
      for (int k = 0; k <= t.H; ++k) { printf(" %d: %.4f", k, (double)h[k] / total); m += (double)k * h[k] / total; }
      printf(" (mean %.3f)\n", m);
      for (int k = 0; k <= t.H; ++k) csv("T", par + ";height=" + std::to_string(k), key, (double)h[k] / total);
    };
    printf("  levels climbed from the leaf (0: the leaf holds the box; %d: the root), fraction of the atoms\n", t.H);
    hist("measured in the kernel", "climb_measured", hClimb, n); hist("arithmetic, periodic", "climb_periodic", hPbc, n); hist("arithmetic, cell not periodic", "climb_not_periodic", hPaper, n);

    if (check && small) {
      search(d, 2, false, 0, dids); CK(cudaDeviceSynchronize()); comparePairs(R0, xq, "T, box");
      search(d, 2, false, 1, dids); CK(cudaDeviceSynchronize()); comparePairs(R0, xq, "T, sphere");
      search(d, 1, false, 0, dids); CK(cudaDeviceSynchronize()); compareForces(R0.f, "T");
      std::vector<float> ev = down(dev, 2 * (size_t)n); double e = 0; for (int i = 0; i < n; ++i) e += ev[2*i];
      printf("  energy of the pairs %.6f against %.6f kcal/mol (relative %.1e)\n", e, R0.e, std::fabs(e - R0.e) / std::fabs(R0.e));
      if (EXCL) { search(d, 0, true, 0, dids); CK(cudaDeviceSynchronize()); compareForces(R0.fx, "T with the excluded pairs"); }
    }

    // Times of the search with the forces.
    double tb = timeK([&] { search(d, 0, false, 0, dids); }), tsph = timeK([&] { search(d, 0, false, 1, dids); });
    int sph = tsph < tb; double tbest = std::min(tb, tsph);
    double tev = timeK([&] { search(d, 1, false, sph, dids); });
    double tid = timeK([&] { search(d, 0, false, sph, didsId); });
    double tx = EXCL ? timeK([&] { search(d, 0, true, sph, dids); }) : NAN;
    printf("  search and forces: %.1f us (box), %.1f us (sphere); with the energy and virial %.1f us; threads in the order of the atoms, not sorted, %.1f us; with the excluded pairs %.1f us\n",
           tb, tsph, tev, tid, tx);
    csv("T", par, "search_box_us", tb); csv("T", par, "search_sphere_us", tsph); csv("T", par, "search_energy_virial_us", tev);
    csv("T", par, "search_unsorted_threads_us", tid); if (EXCL) csv("T", par, "search_exclusions_us", tx);
    if (fT.empty() || tbest < bestT) { search(d, 0, false, sph, dids); CK(cudaDeviceSynchronize()); fT = down(dforce, 3 * (size_t)n); }
    if (tbest < bestT) { bestT = tbest; bestLevel = lev; bestSphere = sph; }

    // The update: steps along the velocities, each update timed alone.
    const int S_ = (int)num("usteps", 20);
    CK(cudaMemset(dhist, 0, 64));
    double uf = 0, uc = 0, ui = 0; int maxOvf = 0;
    for (int s = 0; s < S_; ++s) {
      drift<<<blocks(n), T>>>(n, dxq, dvel, (float)dt, L, iL); CK(cudaDeviceSynchronize());
      cudaEventRecord(e0); update(d, false); cudaEventRecord(e3); CK(cudaEventSynchronize(e3));
      float a, b, c; cudaEventElapsedTime(&a, e0, e1); cudaEventElapsedTime(&b, e1, e2); cudaEventElapsedTime(&c, e2, e3); uf += a; uc += b; ui += c;
      maxOvf = std::max(maxOvf, down(d.novf, 1)[0]);
    }
    // The statistics on steps of their own, so that the atomics of the
    // histogram are not in the times.
    for (int s = 0; s < S_; ++s) { drift<<<blocks(n), T>>>(n, dxq, dvel, (float)dt, L, iL); update(d, true); }
    CK(cudaDeviceSynchronize());
    std::vector<int> hu = down(dhist, 16); size_t moved = 0; for (int k = 1; k < 16; ++k) moved += hu[k];
    printf("  update, %d steps of %.1f fs at %.0f K, mass %.3f u: find %.1f us, compact %.1f us, insert %.1f us, in all %.1f us; overflow at most %d atoms\n",
           S_, dt, temp, mass, 1000 * uf / S_, 1000 * uc / S_, 1000 * ui / S_, 1000 * (uf + uc + ui) / S_, maxOvf);
    csv("T", par, "update_find_us", 1000 * uf / S_); csv("T", par, "update_compact_us", 1000 * uc / S_); csv("T", par, "update_insert_us", 1000 * ui / S_);
    csv("T", par, "update_us", 1000 * (uf + uc + ui) / S_);
    printf("  atoms that leave their leaf in a step: %.4f%%; levels climbed to the common ancestor of the old and new leaf, fraction of those:", 100.0 * moved / ((double)n * S_));
    { double m = 0; for (int k = 1; k <= t.H; ++k) { printf(" %d: %.4f", k, moved ? (double)hu[k] / moved : 0.0); m += moved ? (double)k * hu[k] / moved : 0.0; }
      printf(" (mean %.3f)\n", m);
      for (int k = 1; k <= t.H; ++k) csv("T", par + ";height=" + std::to_string(k), "update_climb", moved ? (double)hu[k] / moved : 0.0); }
    csv("T", par, "leave_leaf_fraction_per_step", moved / ((double)n * S_));
    if (check && small) {
      // Exactness after partial updates: larger steps, so that many atoms
      // change leaf and cross the faces of the cell.
      for (int s = 0; s < 25; ++s) { drift<<<blocks(n), T>>>(n, dxq, dvel, (float)(20 * dt), L, iL); update(d, true); }
      CK(cudaDeviceSynchronize());
      std::vector<int> h2 = down(dhist, 16); size_t mv = 0; for (int k = 1; k < 16; ++k) mv += h2[k] - hu[k];
      std::vector<float4> pos = down(dxq, n); Ref R = reference(pos);
      sortIds(dids); search(d, 2, false, 0, dids); CK(cudaDeviceSynchronize());
      printf("  after 25 more updates with steps 20 times as long (%zu changes of leaf, %d atoms in overflow):\n", mv, down(d.novf, 1)[0]);
      comparePairs(R, pos, "T after the updates");
      // The blocks against the positions: every atom once, in its leaf.
      std::vector<int> cnt = down(d.count, t.nleaves), sl = down(d.slots, (size_t)t.nleaves * MC), ov = down(d.ovf, n), seen(n, 0); int bad = 0, no = down(d.novf, 1)[0];
      for (int b = 0; b < t.nleaves; ++b) for (int k = 0; k < cnt[b]; ++k) { int j = sl[(size_t)b * MC + k]; ++seen[j]; bad += descend(t.nodes.data(), 0, pos[j].x, pos[j].y, pos[j].z) != t.firstLeaf + b; }
      for (int k = 0; k < no; ++k) ++seen[ov[k]];
      for (int i = 0; i < n; ++i) bad += seen[i] != 1;
      printf("  atoms not exactly once in the leaf of their position (or in the overflow): %d\n", bad);
    }
    freeTree(d);
    CK(cudaMemcpy(dxq, dxq0, 16 * (size_t)n, cudaMemcpyDeviceToDevice)); sortIds(dids);
    if (check && small) {
      // A leaf over its capacity: the capacity of the mean occupancy.
      int MCs = std::max(1, (int)std::ceil(mean)); DevTree d2 = fillTree(t, xq, MCs, novf);
      search(d2, 2, false, 0, dids); CK(cudaDeviceSynchronize());
      printf("  capacity %d (buffer 1.0): %d atoms in the list of overflow\n", MCs, novf);
      comparePairs(R0, xq, "T with overflow");
      double to = timeK([&] { search(d2, 0, false, 0, dids); });
      printf("  search and forces with that overflow: %.1f us\n", to); csv("T", par, "search_box_overflow_us", to); csv("T", par, "overflow_atoms_buffer_1", novf);
      freeTree(d2);
    }
  }
  csv("T", "", "best_level", bestLevel); csv("T", "", "best_search_us", bestT);

  // ---- G: the grid ----
  std::vector<int> divisors = list("divisors"); if (divisors.empty()) divisors = {1, 2, 3};
  float4 *dsxq = dalloc<float4>(n); int *dsts = dalloc<int>(n), *dorder = dalloc<int>(n), *dkey = dalloc<int>(n);
  auto makeGrid = [&](double reach, int divisor, int &cells) {
    Grid g; int nc[3], span[3], firstc[3];
    for (int k = 0; k < 3; ++k) {
      nc[k] = std::max(1, (int)(box[k] / (reach * 1.00001 / divisor)));
      double w = box[k] / nc[k]; int range = (int)std::ceil(reach * 1.00001 / w), full = 2 * range + 1;
      span[k] = std::min(nc[k], full); firstc[k] = span[k] == full ? nc[k] - range : 0;
    }
    g.nx = nc[0]; g.ny = nc[1]; g.nz = nc[2]; g.sx = span[0]; g.sy = span[1]; g.sz = span[2]; g.fx = firstc[0]; g.fy = firstc[1]; g.fz = firstc[2];
    g.ilx = iL.x; g.ily = iL.y; g.ilz = iL.z; cells = g.nx * g.ny * g.nz;
    return g;
  };
  for (int divisor : divisors) {
    int cells; Grid g = makeGrid(RC, divisor, cells);
    if ((size_t)cells > (size_t)200 << 20) { printf("G divisor %d: too many cells\n", divisor); continue; }
    int *dcnt = dalloc<int>(cells + 1), *dstart = dalloc<int>(cells + 1), *dcursor = dalloc<int>(cells + 1);
    auto gridBuild = [&] {
      cudaMemsetAsync(dcnt, 0, 4 * (size_t)(cells + 1));
      gridCount<<<blocks(n), T>>>(n, g, dxq, dkey, dcnt);
      thrust::device_ptr<int> c(dcnt), s(dstart); thrust::exclusive_scan(c, c + cells + 1, s);
      cudaMemcpyAsync(dcursor, dstart, 4 * (size_t)(cells + 1), cudaMemcpyDeviceToDevice);
      gridFill<<<blocks(n), T>>>(n, dxq, dts, dkey, dcursor, dorder, dsxq, dsts);
    };
    double tgb = timeK(gridBuild);
    std::string par = "divisor=" + std::to_string(divisor);
#define GS(M, PM, EX) gridSearch<M, PM, EX><<<blocks(n), T>>>(n, P, g, dsxq, dsts, dorder, dstart, dforce, dev, dstats, drows, W)
#define GW(PM, EX) gridWarp<PM, EX><<<blocks(32 * (size_t)n), T>>>(n, P, g, dsxq, dsts, dorder, dstart, dforce)
    auto gs = [&](int mode, bool excl) {
      if (PME) { if (mode == 2) GS(2, true, false); else if (mode == 1) GS(1, true, false); else if (excl) GS(0, true, true); else GS(0, true, false); }
      else { if (mode == 2) GS(2, false, false); else if (mode == 1) GS(1, false, false); else GS(0, false, false); }
    };
    auto gw = [&](bool excl) { if (PME) { if (excl) GW(true, true); else GW(true, false); } else GW(false, false); };
    gs(2, false); CK(cudaDeviceSynchronize());
    std::vector<int> st = down(dstats, 4 * (size_t)n); double cand = 0, pairs = 0; size_t differ = 0;
    for (int i = 0; i < n; ++i) { cand += st[4*i]; pairs += st[4*i+1]; if (!pairsT.empty()) differ += st[4*i+1] != pairsT[i]; }
    printf("G, cells of 1/%d of the cutoff (%d x %d x %d, %d x %d x %d read by an atom): the sort into cells %.1f us; candidates an atom %.1f, pairs within the cutoff %.2f; atoms whose count of pairs differs from that of T: %zu\n",
           divisor, g.nx, g.ny, g.nz, g.sx, g.sy, g.sz, tgb, cand / n, pairs / n, differ);
    csv("G", par, "grid_sort_us", tgb); csv("G", par, "candidates_per_atom", cand / n); csv("G", par, "pairs_per_atom", pairs / n);
    csv("G", par, "bytes", 4.0 * 3 * (cells + 1) + (16.0 + 4 + 4 + 4) * n);
    if (check && small) {
      comparePairs(R0, xq, "G");
      gs(1, false); CK(cudaDeviceSynchronize()); compareForces(R0.f, "G, a thread an atom");
      gw(false); CK(cudaDeviceSynchronize()); compareForces(R0.f, "G, a warp an atom");
      if (EXCL) { gs(0, true); CK(cudaDeviceSynchronize()); compareForces(R0.fx, "G with the excluded pairs, a thread an atom");
                  gw(true); CK(cudaDeviceSynchronize()); compareForces(R0.fx, "G with the excluded pairs, a warp an atom"); }
    }
    if (!fT.empty()) {
      gs(0, false); CK(cudaDeviceSynchronize());
      std::vector<float> f = down(dforce, 3 * (size_t)n); double num2 = 0, den = 0;
      for (size_t k = 0; k < f.size(); ++k) { double dd = (double)f[k] - fT[k]; num2 += dd * dd; den += (double)fT[k] * fT[k]; }
      printf("  forces of G against those of T: relative difference %.2e\n", std::sqrt(num2 / den));
    }
    double tt = timeK([&] { gs(0, false); }), tw = timeK([&] { gw(false); }), te = timeK([&] { gs(1, false); });
    double ttx = EXCL ? timeK([&] { gs(0, true); }) : NAN, twx = EXCL ? timeK([&] { gw(true); }) : NAN;
    printf("  search and forces: a thread an atom %.1f us, a warp an atom %.1f us; with the energy and virial (thread) %.1f us; with the excluded pairs %.1f and %.1f us\n", tt, tw, te, ttx, twx);
    csv("G", par, "search_thread_us", tt); csv("G", par, "search_warp_us", tw); csv("G", par, "search_energy_virial_us", te);
    if (EXCL) { csv("G", par, "search_thread_exclusions_us", ttx); csv("G", par, "search_warp_exclusions_us", twx); }
#undef GS
#undef GW
    cudaFree(dcnt); cudaFree(dstart); cudaFree(dcursor);
  }

  // The pairs within the reach of a list, for the size of a neighbor matrix.
  const double reach = num("reach", 0);
  if (reach > RC) {
    Par PR = P; PR.rc2 = reach * reach; int cells; Grid g = makeGrid(reach, 2, cells);
    int *dcnt = dalloc<int>(cells + 1), *dstart = dalloc<int>(cells + 1), *dcursor = dalloc<int>(cells + 1);
    cudaMemsetAsync(dcnt, 0, 4 * (size_t)(cells + 1));
    gridCount<<<blocks(n), T>>>(n, g, dxq, dkey, dcnt);
    thrust::device_ptr<int> c(dcnt), s(dstart); thrust::exclusive_scan(c, c + cells + 1, s);
    cudaMemcpyAsync(dcursor, dstart, 4 * (size_t)(cells + 1), cudaMemcpyDeviceToDevice);
    gridFill<<<blocks(n), T>>>(n, dxq, dts, dkey, dcursor, dorder, dsxq, dsts);
    if (PME) gridSearch<2, true, false><<<blocks(n), T>>>(n, PR, g, dsxq, dsts, dorder, dstart, dforce, dev, dstats, drows, 0);
    else gridSearch<2, false, false><<<blocks(n), T>>>(n, PR, g, dsxq, dsts, dorder, dstart, dforce, dev, dstats, drows, 0);
    CK(cudaDeviceSynchronize());
    std::vector<int> st = down(dstats, 4 * (size_t)n); double entries = 0; int widest = 0;
    for (int i = 0; i < n; ++i) { entries += st[4*i+1]; widest = std::max(widest, st[4*i+1]); }
    printf("a neighbor matrix of reach %.3f A: %.1f entries an atom (the excluded pairs among them), the widest row %d; %.2f MB as entries of 4 bytes and counts, %.2f MB as rows of the widest\n",
           reach, entries / n, widest, (4.0 * entries + 4.0 * n) / 1e6, (4.0 * n * widest + 4.0 * n) / 1e6);
    csv("M", "reach=" + std::to_string(reach), "entries_per_atom", entries / n); csv("M", "reach=" + std::to_string(reach), "widest_row", widest);
    csv("M", "reach=" + std::to_string(reach), "bytes_entries", 4.0 * entries + 4.0 * n); csv("M", "reach=" + std::to_string(reach), "bytes_rows", 4.0 * n * widest + 4.0 * n);
    cudaFree(dcnt); cudaFree(dstart); cudaFree(dcursor);
  }

  // ---- Steps: dynamics with the forces of T (no charges), or along the
  // fixed velocities (with charges: the pair kernel alone is not a force
  // field). The sorted indices age; the list of a reach would be rebuilt
  // when an atom has moved half the skin.
  const int steps = (int)num("steps", 0), equil = (int)num("equil", 0);
  if (steps > 0 && bestLevel >= 0) {
    int lev = (int)num("level", bestLevel); HostTree t = buildTree(lev);
    std::vector<int> occ(t.nleaves, 0);
    for (int i = 0; i < n; ++i) ++occ[descend(t.nodes.data(), 0, xq[i].x, xq[i].y, xq[i].z) - t.firstLeaf];
    int maxocc = *std::max_element(occ.begin(), occ.end());
    int MC = std::max((int)std::ceil(n * buffer / t.nleaves), (int)std::ceil(1.5 * maxocc) + 2), novf;
    DevTree d = fillTree(t, xq, MC, novf);
    const int sph = bestSphere; const float c = (float)(0.5 * dt * ACCEL / mass);
    printf("steps at level %d, capacity %d, %s: %d of equilibration, %d measured, %.1f fs\n", lev, MC, PME ? "along fixed velocities" : "velocity Verlet with the forces of T", equil, steps, dt);
    auto kinetic = [&] { std::vector<float> v = down(dvel, 3 * (size_t)n); double s = 0; for (float a : v) s += (double)a * a; return 0.5 * mass * s / ACCEL; };
    auto potential = [&] { std::vector<float> ev = down(dev, 2 * (size_t)n); double s = 0; for (int i = 0; i < n; ++i) s += ev[2*i]; return s; };
    int maxOvf = 0;
    auto step = [&](bool stats) {
      if (!PME) kick<<<blocks(3 * (size_t)n), T>>>(n, dvel, dforce, c);
      drift<<<blocks(n), T>>>(n, dxq, dvel, (float)dt, L, iL);
      update(d, stats);
      if (!PME) { search(d, 1, false, sph, dids); kick<<<blocks(3 * (size_t)n), T>>>(n, dvel, dforce, c); }
    };
    sortIds(dids);
    if (!PME) { search(d, 1, false, sph, dids); CK(cudaDeviceSynchronize()); }
    for (int s = 1; s <= equil && !PME; ++s) {
      step(false);
      if (s % 10 == 0) { double tk = 2 * kinetic() / (3.0 * n * KB); scale<<<blocks(3 * (size_t)n), T>>>(n, dvel, (float)std::sqrt(temp / tk)); sortIds(dids); }
      if (s % 500 == 0) { printf("  equilibration step %d: potential energy %.4f kcal/mol an atom, overflow %d\n", s, potential() / n, down(d.novf, 1)[0]); fflush(stdout); }
    }
    sortIds(dids);
    float4 *dref = dalloc<float4>(n); unsigned *dmx = dalloc<unsigned>(1); int *didsFresh = dalloc<int>(n);
    CK(cudaMemcpy(dref, dxq, 16 * (size_t)n, cudaMemcpyDeviceToDevice)); CK(cudaMemset(dmx, 0, 4)); CK(cudaMemset(dhist, 0, 64));
    const double skin = reach > RC ? reach - RC : 1.0; int builds = 0;
    double E0 = 0, Emin = 1e300, Emax = -1e300, Elast = 0, Tsum = 0; int Tn = 0;
    if (!PME) { E0 = kinetic() + potential(); Emin = Emax = Elast = E0; }
    std::vector<double> es, tsamp;
    for (int s = 1; s <= steps; ++s) {
      step(true);
      maxDisp<<<blocks(n), T>>>(n, dxq, dref, L, iL, dmx);
      float md; { unsigned u = down(dmx, 1)[0]; memcpy(&md, &u, 4); }
      if (2 * md > skin) { ++builds; CK(cudaMemcpy(dref, dxq, 16 * (size_t)n, cudaMemcpyDeviceToDevice)); CK(cudaMemset(dmx, 0, 4)); }
      maxOvf = std::max(maxOvf, down(d.novf, 1)[0]);
      if (!PME && s % 50 == 0) { double ke = kinetic(); Elast = ke + potential(); Emin = std::min(Emin, Elast); Emax = std::max(Emax, Elast); Tsum += 2 * ke / (3.0 * n * KB); ++Tn; es.push_back(Elast); tsamp.push_back(s * dt); }
      if (s == 1 || s == 10 || s == 100 || s == 1000 || s == 10000) {
        // The search with the indices sorted s steps ago, and sorted now.
        double tStale = timeK([&] { search(d, 0, false, sph, dids); });
        sortIds(didsFresh); double tFresh = timeK([&] { search(d, 0, false, sph, didsFresh); });
        if (!PME) { search(d, 1, false, sph, dids); CK(cudaDeviceSynchronize()); }
        printf("  %d steps after the sort: search and forces %.1f us; with the indices sorted now %.1f us\n", s, tStale, tFresh);
        csv("T", "level=" + std::to_string(lev) + ";steps_since_sort=" + std::to_string(s), "search_stale_us", tStale);
        csv("T", "level=" + std::to_string(lev) + ";steps_since_sort=" + std::to_string(s), "search_fresh_us", tFresh);
        fflush(stdout);
      }
    }
    std::vector<int> hu = down(dhist, 16); size_t moved = 0; for (int k = 1; k < 16; ++k) moved += hu[k];
    printf("  atoms that leave their leaf in a step: %.4f%%; levels to the common ancestor:", 100.0 * moved / ((double)n * steps));
    for (int k = 1; k <= t.H; ++k) printf(" %d: %.4f", k, moved ? (double)hu[k] / moved : 0.0);
    printf("; overflow at most %d atoms\n", maxOvf);
    for (int k = 1; k <= t.H; ++k) csv("T", "level=" + std::to_string(lev) + ";height=" + std::to_string(k), "dynamics_update_climb", moved ? (double)hu[k] / moved : 0.0);
    csv("T", "level=" + std::to_string(lev), "dynamics_leave_leaf_fraction_per_step", moved / ((double)n * steps));
    printf("  a list with a skin of %.2f A, rebuilt when an atom has moved half of it: %d builds in %d steps, %.1f steps between builds\n", skin, builds, steps, builds ? (double)steps / builds : (double)steps);
    csv("M", "skin=" + std::to_string(skin), "steps_between_builds", builds ? (double)steps / builds : (double)steps);
    if (!PME) {
      // The drift: the slope of a least-squares line through the total energy.
      double sx = 0, sy = 0, sxx = 0, sxy = 0; int m = es.size();
      for (int k = 0; k < m; ++k) { sx += tsamp[k]; sy += es[k]; sxx += tsamp[k] * tsamp[k]; sxy += tsamp[k] * es[k]; }
      double slope = m > 1 ? (m * sxy - sx * sy) / (m * sxx - sx * sx) : 0.0, kT = KB * temp;
      printf("  total energy (the potential shifted to zero at the cutoff): %.6f at the start, %.6f at the end, range %.6f kcal/mol an atom; drift %.3e kcal/mol an atom a ns, %.3e kT an atom a ns; mean temperature %.2f K\n",
             E0 / n, Elast / n, (Emax - Emin) / n, slope / n * 1e6, slope / n * 1e6 / kT, Tn ? Tsum / Tn : 0.0);
      csv("T", "level=" + std::to_string(lev), "energy_drift_kT_per_atom_per_ns", slope / n * 1e6 / kT); csv("T", "level=" + std::to_string(lev), "mean_temperature_K", Tn ? Tsum / Tn : 0.0);
    }
    if (opt.count("out")) {
      std::vector<float4> pos = down(dxq, n);
      for (int p = 0; p < n; ++p) { int i = perm[p]; S.x[3*i] = pos[p].x; S.x[3*i+1] = pos[p].y; S.x[3*i+2] = pos[p].z; }
      writeSystem(opt["out"].c_str(), S); printf("  positions written to %s\n", opt["out"].c_str());
    }
  }
  return 0;
}
