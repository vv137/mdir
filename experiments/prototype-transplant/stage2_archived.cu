// Stage 2 of the rigid water template (#774 line), measurement only.
//
// The whole short-range nonbonded path of a solvated system, driven by main's own trajectory
// (a DCD frame every step) or evaluated once on one f64 frame:
//   water clusters: 16 molecules (O, H1, H2 [, EP placed in registers]), anchored at the oxygen;
//   solute clusters: 16 atoms (protein and ions);
//   three cluster-pair lists, each a half or full list of entries (item, 16-bit mask | cells):
//     WW  water cluster x molecule   reach r_c + 2 r_mol  (+ skin), half list;
//     SW  solute cluster x molecule  reach r_c + r_mol    (+ skin), full list (the kinds differ);
//     SS  solute cluster x atom      reach r_c            (+ skin), half list, exclusions cleared;
//   the kernels: (c3) for WW (warp-cooperative molecule groups, j rotated by shuffles), and the
//   same skeleton for SW (lane = solute atom, j = molecule) and SS (lane = atom, j = atom).
// Refresh rule of main's group path (D80, D114): outer skin 3.0 A, inner 0.6 A; a list is valid
// while 2 max |x - x_ref| <= skin over the items it depends on. Two modes:
//   S2_LISTS=2: WW tested on the anchors alone; SW+SS on the anchors and the solute atoms;
//   S2_LISTS=1: all three lists tested, built and pruned together.
// In the two-list mode a WW build renumbers the water slots; the SW entries are then remapped.
//
// Mixed precision as main: f32 packed positions (moved by whole cells of the build, the sum in f64,
// then rounded), f32 displacement, charge products, the radial table of D94 for the Coulomb force,
// f32 forces and accumulators; energies (validation only) in f64 sums of f32 terms.
// The extra point: x_EP = x_O + a (u + v) with a constant a (rigid geometry), its force spread by
// the bisector Jacobian with the rigid constants (as the (c3) prototype).
// Units: Angstrom, kcal/mol, e.
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <nvtx3/nvToolsExt.h>
#include <thrust/iterator/counting_iterator.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(2);} } while (0)

struct Params {
  float L[3]; float rc2; float A12, B6, A, B; float k[4][4]; float kq[4]; float a, oma, invh, invs; float beta;
};

// ---------------------------------------------------------------------------
// The radial table of D94 (as the (c3) prototype): force/r of erfc(beta r)/r, cubic per cell.
constexpr int TB = 7;
struct Table { std::vector<float4> c; int base; float rmin2; int n; };
static double gforce(double r2, double beta) {
  double r = std::sqrt(r2);
  return (std::erfc(beta * r) / r + 2.0 * beta / std::sqrt(M_PI) * std::exp(-beta * beta * r2)) / r2;
}
static Table makeTable(double rc, double beta) {
  Table t;
  double lo = std::ldexp(1.0, (int)std::floor(std::log2(rc * rc / 1024.0)));
  int e0 = (int)std::lround(std::log2(lo)); int e1 = (int)std::floor(std::log2(rc * rc));
  t.rmin2 = (float)lo; uint32_t b0; float flo = (float)lo; memcpy(&b0, &flo, 4); t.base = (int)(b0 >> (23 - TB));
  t.n = (e1 - e0 + 1) << TB; t.c.resize(t.n);
  for (int i = 0; i < t.n; ++i) {
    uint32_t bits = (uint32_t)(t.base + i) << (23 - TB); float fs; memcpy(&fs, &bits, 4);
    uint32_t bitsE = (uint32_t)(t.base + i + 1) << (23 - TB); float fe; memcpy(&fe, &bitsE, 4);
    double s = fs, w = (double)fe - s, x[4], y[4];
    for (int k = 0; k < 4; ++k) { double c = std::cos(M_PI * (2 * k + 1) / 8.0); x[k] = 0.5 * w * (1 + c); y[k] = gforce(s + x[k], beta); }
    double co[4] = {0, 0, 0, 0};
    for (int k = 0; k < 4; ++k) {
      double p[4] = {1, 0, 0, 0}; double den = 1; int deg = 0;
      for (int m = 0; m < 4; ++m) if (m != k) {
        double q[4] = {0, 0, 0, 0};
        for (int d = 0; d <= deg; ++d) { q[d + 1] += p[d]; q[d] -= x[m] * p[d]; }
        ++deg; memcpy(p, q, sizeof p); den *= (x[k] - x[m]);
      }
      for (int d = 0; d < 4; ++d) co[d] += y[k] * p[d] / den;
    }
    t.c[i] = make_float4((float)co[0], (float)co[1], (float)co[2], (float)co[3]);
  }
  return t;
}
__device__ __forceinline__ float radial(const float4 *__restrict__ tab, int base, int n, float r2) {
  int bits = __float_as_int(r2);
  int idx = min(max((bits >> (23 - TB)) - base, 0), n - 1);
  float start = __int_as_float(bits & ~((1 << (23 - TB)) - 1));
  float u = r2 - start; float4 c = __ldg(tab + idx);
  return fmaf(fmaf(fmaf(c.w, u, c.z), u, c.y), u, c.x);
}
// The bisector Jacobian with the rigid geometry's constants (|u| = h, |u + v| = s): the force F on
// the extra point to O, H1, H2.
__device__ __forceinline__ void spreadRigid(const float *xo, const float *x1, const float *x2, const float *F, float invh,
                                            float invs, float c, float *fo, float *f1, float *f2) {
  float uh[3], vh[3], bh[3];
  for (int k = 0; k < 3; ++k) { float u = x1[k] - xo[k], v = x2[k] - xo[k]; uh[k] = u * invh; vh[k] = v * invh; bh[k] = (u + v) * invs; }
  float fb = F[0] * bh[0] + F[1] * bh[1] + F[2] * bh[2];
  float G[3]; for (int k = 0; k < 3; ++k) G[k] = F[k] - fb * bh[k];
  float gu = G[0] * uh[0] + G[1] * uh[1] + G[2] * uh[2], gv = G[0] * vh[0] + G[1] * vh[1] + G[2] * vh[2];
  for (int k = 0; k < 3; ++k) { float a1 = c * (G[k] - gu * uh[k]), a2 = c * (G[k] - gv * vh[k]); f1[k] += a1; f2[k] += a2; fo[k] += F[k] - a1 - a2; }
}
struct Counters { unsigned long long laneSteps, paired, sitePairsIn, pad; };
__device__ __forceinline__ float shfl(float v, int src) { return __shfl_sync(0xffffffffu, v, src, 32); }

// A list in the layout of main's group path: the entries of group g are at
// entries[blockStart[g] * 64 + k], k < count[g]; the active units (blocks of 64) of the pool.
struct ListView {
  const int *activeBlocks, *nActive, *unitGroup, *unitOrd, *blockStart, *innerCount; const int2 *entries;
};
__device__ __forceinline__ bool unitOf(const ListView &v, int w, int &g, int &begin, int &end, const int2 *&ent) {
  if (w >= *v.nActive) return false;
  int b = v.activeBlocks[w]; g = v.unitGroup[b]; int ord = v.unitOrd[b];
  begin = ord * 64; end = min(v.innerCount[g], begin + 64); ent = v.entries + (size_t)v.blockStart[g] * 64;
  return true;
}
__device__ __forceinline__ unsigned rotatedBits(unsigned mask, int u, int half) {
  unsigned own16 = 0;
#pragma unroll
  for (int q = 0; q < 16; ++q) { unsigned wq = __ballot_sync(0xffffffffu, (mask >> q) & 1u); own16 = (u == q) ? wq : own16; }
  unsigned halfBits = (own16 >> half) & 0xFFFFu;
  int turn = 15 - u;
  return ((halfBits << turn) | (halfBits >> (16 - turn))) & 0xFFFFu;
}
__device__ __forceinline__ void cellShift(unsigned code, const float *L, float *sh) {
  for (int ax = 0; ax < 3; ++ax) sh[ax] = (float)((int)((code >> (16 + 4 * ax)) & 15u) - 4) * L[ax];
}
template <int NS> struct Sites;   // the charged sites of a molecule: OPC H1, H2, EP; TIP3P O, H1, H2
template <> struct Sites<4> { static constexpr int lo = 1; };
template <> struct Sites<3> { static constexpr int lo = 0; };

// ---------------------------------------------------------------------------
// WW: (c3), a warp a unit; both half-warps hold the same 16 i molecules, each streams 16 of the 32
// entries of a round through 16 shuffle steps; the j forces written once an entry, the i forces once.
template <int NS, bool ENERGY>
__device__ __forceinline__ void ww_unit(int w, const float4 *__restrict__ pos, ListView lv, float *__restrict__ force, Params p,
                                                const float4 *__restrict__ tab, int tbase, int tn, double *energy, Counters *cnt) {
  const int lane = threadIdx.x & 31;
  int g, begin, end; const int2 *ent;
  if (!unitOf(lv, w, g, begin, end, ent)) return;
  const int u = lane & 15, half = lane & 16;
  const int mine = g * 16 + u;
  const float4 *pi = pos + 3 * (size_t)mine;
  float4 t0 = pi[0], t1 = pi[1], t2 = pi[2];
  float xi[NS][3] = {};
  xi[0][0] = t0.x; xi[0][1] = t0.y; xi[0][2] = t0.z; xi[1][0] = t1.x; xi[1][1] = t1.y; xi[1][2] = t1.z;
  xi[2][0] = t2.x; xi[2][1] = t2.y; xi[2][2] = t2.z;
  if (NS == 4) for (int c = 0; c < 3; ++c) xi[NS - 1][c] = fmaf(p.a, (xi[1][c] - xi[0][c]) + (xi[2][c] - xi[0][c]), xi[0][c]);
  float fi[NS][3] = {};
  double ei = 0.0; unsigned long long cLane = 0, cPaired = 0, cIn = 0;
  const int source = half | ((u + 15) & 15);
  for (int e0 = begin; e0 < end; e0 += 32) {
    const int e = e0 + lane; const bool has = e < end;
    int2 en = ent[has ? e : begin];
    int j = en.x; unsigned mask = has ? (unsigned)en.y : 0u;
    const float4 *pj = pos + 3 * (size_t)j;
    float4 s0 = pj[0], s1 = pj[1], s2 = pj[2];
    float sh[3]; cellShift(mask, p.L, sh);
    float xj[3][3] = {{s0.x + sh[0], s0.y + sh[1], s0.z + sh[2]}, {s1.x + sh[0], s1.y + sh[1], s1.z + sh[2]},
                      {s2.x + sh[0], s2.y + sh[1], s2.z + sh[2]}};
    unsigned bits = rotatedBits(mask, u, half);
    float fj[3][3] = {}; float fjE[3] = {0.f, 0.f, 0.f};
#pragma unroll 1
    for (int step = 0; step < 16; ++step) {
      const bool paired = bits & 0x8000u;
      if (ENERGY) { cLane += 1; cPaired += paired; }
      {
        float d0[3]; for (int c = 0; c < 3; ++c) d0[c] = xi[0][c] - xj[0][c];
        float rOO2 = fmaf(d0[0], d0[0], fmaf(d0[1], d0[1], d0[2] * d0[2]));
        float ir2 = __frcp_rn(rOO2); float ir6 = ir2 * ir2 * ir2;
        bool in = paired && rOO2 < p.rc2;
        float fs = in ? (p.A12 * ir6 - p.B6) * ir6 * ir2 : 0.0f;
        if (ENERGY && in) { ei += (double)((p.A * ir6 - p.B) * ir6); ++cIn; }
        for (int c = 0; c < 3; ++c) { float f = fs * d0[c]; fi[0][c] += f; fj[0][c] -= f; }
      }
      float xe[3], fe[3] = {0, 0, 0};
      if (NS == 4) for (int c = 0; c < 3; ++c) xe[c] = fmaf(p.a, (xj[1][c] - xj[0][c]) + (xj[2][c] - xj[0][c]), xj[0][c]);
#pragma unroll
      for (int a = Sites<NS>::lo; a < NS; ++a) {
#pragma unroll
        for (int bb = Sites<NS>::lo; bb < NS; ++bb) {
          const float *xb = (NS == 4 && bb == 3) ? xe : xj[bb < 3 ? bb : 0];
          float d[3]; for (int c = 0; c < 3; ++c) d[c] = xi[a][c] - xb[c];
          float r2 = fmaf(d[0], d[0], fmaf(d[1], d[1], d[2] * d[2]));
          bool in = paired && r2 < p.rc2;
          float gg = radial(tab, tbase, tn, r2);
          float fs = in ? p.k[a][bb] * gg : 0.0f;
          if (ENERGY && in) { float r = sqrtf(r2); ei += (double)(p.k[a][bb] * (erfcf(p.beta * r) / r)); ++cIn; }
          for (int c = 0; c < 3; ++c) { float f = fs * d[c]; fi[a][c] += f; if (NS == 4 && bb == 3) fe[c] -= f; else fj[bb < 3 ? bb : 0][c] -= f; }
        }
      }
      if (NS == 4) for (int c = 0; c < 3; ++c) fjE[c] += fe[c];
      for (int s = 0; s < 3; ++s)
        for (int c = 0; c < 3; ++c) { xj[s][c] = shfl(xj[s][c], source); fj[s][c] = shfl(fj[s][c], source); }
      if (NS == 4) for (int c = 0; c < 3; ++c) fjE[c] = shfl(fjE[c], source);
      bits <<= 1;
    }
    if (NS == 4) spreadRigid(xj[0], xj[1], xj[2], fjE, p.invh, p.invs, p.a, fj[0], fj[1], fj[2]);
    if (has) {
      float *fo = force + 9 * (size_t)j;
      for (int s = 0; s < 3; ++s) for (int c = 0; c < 3; ++c) atomicAdd(fo + 3 * s + c, fj[s][c]);
    }
  }
  if (NS == 4) spreadRigid(xi[0], xi[1], xi[2], fi[NS - 1], p.invh, p.invs, p.a, fi[0], fi[1], fi[2]);
  for (int s = 0; s < 3; ++s) for (int c = 0; c < 3; ++c) fi[s][c] += __shfl_xor_sync(0xffffffffu, fi[s][c], 16, 32);
  if (half == 0 && begin < end) {
    float *fo = force + 9 * (size_t)mine;
    for (int s = 0; s < 3; ++s) for (int c = 0; c < 3; ++c) atomicAdd(fo + 3 * s + c, fi[s][c]);
  }
  if (ENERGY) {
    for (int o = 16; o; o >>= 1) {
      ei += __shfl_xor_sync(0xffffffffu, ei, o, 32); cLane += __shfl_xor_sync(0xffffffffu, cLane, o, 32);
      cPaired += __shfl_xor_sync(0xffffffffu, cPaired, o, 32); cIn += __shfl_xor_sync(0xffffffffu, cIn, o, 32);
    }
    if (lane == 0) { atomicAdd(energy, ei); atomicAdd(&cnt->laneSteps, cLane); atomicAdd(&cnt->paired, cPaired); atomicAdd(&cnt->sitePairsIn, cIn); }
  }
}

// SW: lane = solute atom of cluster g (both half-warps the same 16), j = water molecule. Per pair:
// LJ with O (per-lane coefficients) and the Coulomb of the 3 charged sites.
template <int NS, bool ENERGY>
__device__ __forceinline__ void sw_unit(int w, const float4 *__restrict__ spos, const float4 *__restrict__ sLJ,
                                                const float4 *__restrict__ pos, ListView lv, float *__restrict__ fW,
                                                float *__restrict__ fS, Params p, const float4 *__restrict__ tab, int tbase, int tn,
                                                double *energy, Counters *cnt) {
  const int lane = threadIdx.x & 31;
  int g, begin, end; const int2 *ent;
  if (!unitOf(lv, w, g, begin, end, ent)) return;
  const int u = lane & 15, half = lane & 16;
  const int mine = g * 16 + u;
  float4 ai = spos[mine]; float4 lj = sLJ[mine];
  const float xi[3] = {ai.x, ai.y, ai.z};
  float kis[4]; for (int s = 0; s < 4; ++s) kis[s] = ai.w * p.kq[s];
  float fi[3] = {0.f, 0.f, 0.f};
  double ei = 0.0; unsigned long long cLane = 0, cPaired = 0, cIn = 0;
  const int source = half | ((u + 15) & 15);
  for (int e0 = begin; e0 < end; e0 += 32) {
    const int e = e0 + lane; const bool has = e < end;
    int2 en = ent[has ? e : begin];
    int j = en.x; unsigned mask = has ? (unsigned)en.y : 0u;
    const float4 *pj = pos + 3 * (size_t)j;
    float4 s0 = pj[0], s1 = pj[1], s2 = pj[2];
    float sh[3]; cellShift(mask, p.L, sh);
    float xj[3][3] = {{s0.x + sh[0], s0.y + sh[1], s0.z + sh[2]}, {s1.x + sh[0], s1.y + sh[1], s1.z + sh[2]},
                      {s2.x + sh[0], s2.y + sh[1], s2.z + sh[2]}};
    unsigned bits = rotatedBits(mask, u, half);
    float fj[3][3] = {}; float fjE[3] = {0.f, 0.f, 0.f};
#pragma unroll 1
    for (int step = 0; step < 16; ++step) {
      const bool paired = bits & 0x8000u;
      if (ENERGY) { cLane += 1; cPaired += paired; }
      {
        float d0[3]; for (int c = 0; c < 3; ++c) d0[c] = xi[c] - xj[0][c];
        float r2 = fmaf(d0[0], d0[0], fmaf(d0[1], d0[1], d0[2] * d0[2]));
        float ir2 = __frcp_rn(r2); float ir6 = ir2 * ir2 * ir2;
        bool in = paired && r2 < p.rc2;
        float fs = in ? (lj.x * ir6 - lj.y) * ir6 * ir2 : 0.0f;
        if (ENERGY && in) { ei += (double)((lj.z * ir6 - lj.w) * ir6); ++cIn; }
        for (int c = 0; c < 3; ++c) { float f = fs * d0[c]; fi[c] += f; fj[0][c] -= f; }
      }
      float xe[3], fe[3] = {0, 0, 0};
      if (NS == 4) for (int c = 0; c < 3; ++c) xe[c] = fmaf(p.a, (xj[1][c] - xj[0][c]) + (xj[2][c] - xj[0][c]), xj[0][c]);
#pragma unroll
      for (int bb = Sites<NS>::lo; bb < NS; ++bb) {
        const float *xb = (NS == 4 && bb == 3) ? xe : xj[bb < 3 ? bb : 0];
        float d[3]; for (int c = 0; c < 3; ++c) d[c] = xi[c] - xb[c];
        float r2 = fmaf(d[0], d[0], fmaf(d[1], d[1], d[2] * d[2]));
        bool in = paired && r2 < p.rc2;
        float gg = radial(tab, tbase, tn, r2);
        float fs = in ? kis[bb] * gg : 0.0f;
        if (ENERGY && in) { float r = sqrtf(r2); ei += (double)(kis[bb] * (erfcf(p.beta * r) / r)); ++cIn; }
        for (int c = 0; c < 3; ++c) { float f = fs * d[c]; fi[c] += f; if (NS == 4 && bb == 3) fe[c] -= f; else fj[bb < 3 ? bb : 0][c] -= f; }
      }
      if (NS == 4) for (int c = 0; c < 3; ++c) fjE[c] += fe[c];
      for (int s = 0; s < 3; ++s)
        for (int c = 0; c < 3; ++c) { xj[s][c] = shfl(xj[s][c], source); fj[s][c] = shfl(fj[s][c], source); }
      if (NS == 4) for (int c = 0; c < 3; ++c) fjE[c] = shfl(fjE[c], source);
      bits <<= 1;
    }
    if (NS == 4) spreadRigid(xj[0], xj[1], xj[2], fjE, p.invh, p.invs, p.a, fj[0], fj[1], fj[2]);
    if (has) {
      float *fo = fW + 9 * (size_t)j;
      for (int s = 0; s < 3; ++s) for (int c = 0; c < 3; ++c) atomicAdd(fo + 3 * s + c, fj[s][c]);
    }
  }
  for (int c = 0; c < 3; ++c) fi[c] += __shfl_xor_sync(0xffffffffu, fi[c], 16, 32);
  if (half == 0 && begin < end) for (int c = 0; c < 3; ++c) atomicAdd(fS + 3 * (size_t)mine + c, fi[c]);
  if (ENERGY) {
    for (int o = 16; o; o >>= 1) {
      ei += __shfl_xor_sync(0xffffffffu, ei, o, 32); cLane += __shfl_xor_sync(0xffffffffu, cLane, o, 32);
      cPaired += __shfl_xor_sync(0xffffffffu, cPaired, o, 32); cIn += __shfl_xor_sync(0xffffffffu, cIn, o, 32);
    }
    if (lane == 0) { atomicAdd(energy, ei); atomicAdd(&cnt->laneSteps, cLane); atomicAdd(&cnt->paired, cPaired); atomicAdd(&cnt->sitePairsIn, cIn); }
  }
}

// SS: lane = atom of cluster g, j = atom; LJ by the pair of types (a table in shared memory) and
// the Coulomb; the excluded pairs were cleared from the masks at the build.
template <bool ENERGY>
__device__ __forceinline__ void ss_unit(int w, const float4 *__restrict__ spos, const int *__restrict__ stype,
                                                const float4 *__restrict__ lt, int nt, ListView lv, float *__restrict__ fS,
                                                Params p, float ke, const float4 *__restrict__ tab, int tbase, int tn,
                                                double *energy, Counters *cnt) {
  const int lane = threadIdx.x & 31;
  int g, begin, end; const int2 *ent;
  if (!unitOf(lv, w, g, begin, end, ent)) return;
  const int u = lane & 15, half = lane & 16;
  const int mine = g * 16 + u;
  float4 ai = spos[mine]; const int ti = stype[mine] * nt;
  const float qi = ai.w * ke;
  float fi[3] = {0.f, 0.f, 0.f};
  double ei = 0.0; unsigned long long cLane = 0, cPaired = 0, cIn = 0;
  const int source = half | ((u + 15) & 15);
  for (int e0 = begin; e0 < end; e0 += 32) {
    const int e = e0 + lane; const bool has = e < end;
    int2 en = ent[has ? e : begin];
    int j = en.x; unsigned mask = has ? (unsigned)en.y : 0u;
    float4 aj = spos[j]; int tj = stype[j];
    float sh[3]; cellShift(mask, p.L, sh);
    float xj[4] = {aj.x + sh[0], aj.y + sh[1], aj.z + sh[2], aj.w};
    unsigned bits = rotatedBits(mask, u, half);
    float fj[3] = {0.f, 0.f, 0.f};
#pragma unroll 1
    for (int step = 0; step < 16; ++step) {
      const bool paired = bits & 0x8000u;
      if (ENERGY) { cLane += 1; cPaired += paired; }
      float d[3]; for (int c = 0; c < 3; ++c) d[c] = ai.x * (c == 0) + ai.y * (c == 1) + ai.z * (c == 2) - xj[c];
      float r2 = fmaf(d[0], d[0], fmaf(d[1], d[1], d[2] * d[2]));
      bool in = paired && r2 < p.rc2;
      float4 c4 = lt[ti + (tj & 0xff)];
      float ir2 = __frcp_rn(r2); float ir6 = ir2 * ir2 * ir2;
      float qq = qi * xj[3];
      float gg = radial(tab, tbase, tn, r2);
      float fs = in ? (c4.x * ir6 - c4.y) * ir6 * ir2 + qq * gg : 0.0f;
      if (ENERGY && in) { float r = sqrtf(r2); ei += (double)((c4.z * ir6 - c4.w) * ir6) + (double)(qq * (erfcf(p.beta * r) / r)); ++cIn; }
      for (int c = 0; c < 3; ++c) { float f = fs * d[c]; fi[c] += f; fj[c] -= f; }
      for (int c = 0; c < 4; ++c) xj[c] = shfl(xj[c], source);
      for (int c = 0; c < 3; ++c) fj[c] = shfl(fj[c], source);
      tj = __shfl_sync(0xffffffffu, tj, source, 32);
      bits <<= 1;
    }
    if (has) for (int c = 0; c < 3; ++c) atomicAdd(fS + 3 * (size_t)j + c, fj[c]);
  }
  for (int c = 0; c < 3; ++c) fi[c] += __shfl_xor_sync(0xffffffffu, fi[c], 16, 32);
  if (half == 0 && begin < end) for (int c = 0; c < 3; ++c) atomicAdd(fS + 3 * (size_t)mine + c, fi[c]);
  if (ENERGY) {
    for (int o = 16; o; o >>= 1) {
      ei += __shfl_xor_sync(0xffffffffu, ei, o, 32); cLane += __shfl_xor_sync(0xffffffffu, cLane, o, 32);
      cPaired += __shfl_xor_sync(0xffffffffu, cPaired, o, 32); cIn += __shfl_xor_sync(0xffffffffu, cIn, o, 32);
    }
    if (lane == 0) { atomicAdd(energy, ei); atomicAdd(&cnt->laneSteps, cLane); atomicAdd(&cnt->paired, cPaired); atomicAdd(&cnt->sitePairsIn, cIn); }
  }
}

// The kernels: one per class, or (S2_FUSE=1) one launch for the three classes, a class per warp
// (the SW and SS units first, then WW), so that the few long units of the solute overlap with
// the many of the water instead of running as two small launches of their own.
template <int NS, bool ENERGY>
__global__ void __launch_bounds__(128) ww_force(const float4 *__restrict__ pos, ListView lv, float *__restrict__ force, Params p,
                                                const float4 *__restrict__ tab, int tbase, int tn, double *energy, Counters *cnt) {
  ww_unit<NS, ENERGY>(blockIdx.x * 4 + (threadIdx.x >> 5), pos, lv, force, p, tab, tbase, tn, energy, cnt);
}
template <int NS, bool ENERGY>
__global__ void __launch_bounds__(128) sw_force(const float4 *__restrict__ spos, const float4 *__restrict__ sLJ,
                                                const float4 *__restrict__ pos, ListView lv, float *__restrict__ fW,
                                                float *__restrict__ fS, Params p, const float4 *__restrict__ tab, int tbase, int tn,
                                                double *energy, Counters *cnt) {
  sw_unit<NS, ENERGY>(blockIdx.x * 4 + (threadIdx.x >> 5), spos, sLJ, pos, lv, fW, fS, p, tab, tbase, tn, energy, cnt);
}
template <bool ENERGY>
__global__ void __launch_bounds__(128) ss_force(const float4 *__restrict__ spos, const int *__restrict__ stype,
                                                const float4 *__restrict__ ljTab, int nt, ListView lv, float *__restrict__ fS,
                                                Params p, float ke, const float4 *__restrict__ tab, int tbase, int tn,
                                                double *energy, Counters *cnt) {
  extern __shared__ float4 lt[];
  for (int k = threadIdx.x; k < nt * nt; k += blockDim.x) lt[k] = ljTab[k];
  __syncthreads();
  ss_unit<ENERGY>(blockIdx.x * 4 + (threadIdx.x >> 5), spos, stype, lt, nt, lv, fS, p, ke, tab, tbase, tn, energy, cnt);
}
template <int NS, bool ENERGY>
#ifndef FUSE_MINB
#define FUSE_MINB 6
#endif
__global__ void __launch_bounds__(128, FUSE_MINB) all_force(const float4 *__restrict__ pos, const float4 *__restrict__ spos, const float4 *__restrict__ sLJ,
                                                 const int *__restrict__ stype, const float4 *__restrict__ ljTab, int nt, ListView lw,
                                                 ListView ls, ListView lss, float *__restrict__ fW, float *__restrict__ fS, Params p, float ke,
                                                 const float4 *__restrict__ tab, int tbase, int tn, double *energy, Counters *cnt) {
  extern __shared__ float4 lt[];
  for (int k = threadIdx.x; k < nt * nt; k += blockDim.x) lt[k] = ljTab[k];
  __syncthreads();
  int w = blockIdx.x * 4 + (threadIdx.x >> 5);
  const int nsw = *ls.nActive, nss = *lss.nActive;
  if (w < nsw) { sw_unit<NS, ENERGY>(w, spos, sLJ, pos, ls, fW, fS, p, tab, tbase, tn, energy + 1, cnt + 1); return; }
  w -= nsw;
  if (w < nss) { ss_unit<ENERGY>(w, spos, stype, lt, nt, lss, fS, p, ke, tab, tbase, tn, energy + 2, cnt + 2); return; }
  w -= nss;
  ww_unit<NS, ENERGY>(w, pos, lw, fW, p, tab, tbase, tn, energy, cnt);
}

// ---------------------------------------------------------------------------
// The list maintenance.
struct Geo { float L[3]; int ncx, ncy; int nS, ns; };

// The test of D80 on the anchors and the solute atoms: the largest squared displacement against
// four references of the anchors (WW build/prune, solute-list build/prune) and two of the atoms.
template <class FT>
__global__ void s2_test(const FT *__restrict__ frame, Geo geo, int nW, const float4 *__restrict__ refWB, const float4 *__restrict__ refWP,
                        const float4 *__restrict__ refLB, const float4 *__restrict__ refLP, const float4 *__restrict__ refSB,
                        const float4 *__restrict__ refSP, float4 *__restrict__ ancNow, float4 *__restrict__ solNow,
                        unsigned *__restrict__ maxBits) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float d[4] = {0.f, 0.f, 0.f, 0.f};   // anchors vs WB, WP, LB, LP; atoms into LB, LP slots
  auto dd = [](float4 a, float4 b) { float x = a.x - b.x, y = a.y - b.y, z = a.z - b.z; return x * x + y * y + z * z; };
  float s0 = 0.f, s1 = 0.f;
  if (i < nW) {
    const FT *o = frame + 3 * ((size_t)geo.nS + (size_t)geo.ns * i);
    float4 a = make_float4((float)o[0], (float)o[1], (float)o[2], 0.f); ancNow[i] = a;
    d[0] = dd(a, refWB[i]); d[1] = dd(a, refWP[i]); d[2] = dd(a, refLB[i]); d[3] = dd(a, refLP[i]);
  } else if (i < nW + geo.nS) {
    int k = i - nW; const FT *o = frame + 3 * (size_t)k;
    float4 a = make_float4((float)o[0], (float)o[1], (float)o[2], 0.f); solNow[k] = a;
    s0 = dd(a, refSB[k]); s1 = dd(a, refSP[k]);
  }
  float v[6] = {d[0], d[1], d[2], d[3], s0, s1};
  for (int o = 16; o; o >>= 1) for (int k = 0; k < 6; ++k) v[k] = fmaxf(v[k], __shfl_xor_sync(0xffffffffu, v[k], o));
  if ((threadIdx.x & 31) == 0) for (int k = 0; k < 6; ++k) atomicMax(maxBits + k, __float_as_uint(v[k]));
}

// The refreshes of the references (after a build or a pruning of a list).
__global__ void s2_setref(const float4 *__restrict__ ancNow, const float4 *__restrict__ solNow, int nW, int nS, int flags,
                          float4 *refWB, float4 *refWP, float4 *refLB, float4 *refLP, float4 *refSB, float4 *refSP) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < nW) {
    float4 a = ancNow[i];
    if (flags & 1) refWB[i] = a; if (flags & 2) refWP[i] = a; if (flags & 4) refLB[i] = a; if (flags & 8) refLP[i] = a;
  } else if (i < nW + nS) {
    int k = i - nW; float4 a = solNow[k];
    if (flags & 4) refSB[k] = a; if (flags & 8) refSP[k] = a;
  }
}

// Keys: the wrapped point, the whole cells that wrap it (kept until the next build of its list), the
// column (x-y, serpentine) and z.
__global__ void s2_keys(const float4 *__restrict__ pts, int n, Geo geo, int ncx, int ncy, unsigned *__restrict__ keys,
                        int *__restrict__ vals, int4 *__restrict__ shift, int *__restrict__ colCount) {
  int m = blockIdx.x * blockDim.x + threadIdx.x;
  if (m >= n) return;
  float4 a = pts[m];
  float c[3] = {a.x, a.y, a.z}; int s[3]; float w[3];
  for (int ax = 0; ax < 3; ++ax) {
    s[ax] = -(int)floorf(c[ax] / geo.L[ax]);
    w[ax] = (float)((double)c[ax] + (double)s[ax] * (double)geo.L[ax]);
    w[ax] = fminf(fmaxf(w[ax], 0.f), geo.L[ax] * 0.9999999f);
  }
  int cx = min(ncx - 1, (int)(w[0] / geo.L[0] * ncx)), cy = min(ncy - 1, (int)(w[1] / geo.L[1] * ncy));
  int col = cx * ncy + ((cx & 1) ? ncy - 1 - cy : cy);
  unsigned zq = min((1u << 21) - 1, (unsigned)(w[2] / geo.L[2] * (float)(1u << 21)));
  keys[m] = ((unsigned)col << 21) | zq; vals[m] = m; shift[m] = make_int4(s[0], s[1], s[2], 0);
  if (colCount) atomicAdd(colCount + col, 1);
}

constexpr int SCANT = 1024, SCANI = 4;
__global__ void __launch_bounds__(SCANT) s2_columns(const int *__restrict__ colCount, int ncol, int *__restrict__ colStart,
                                                     int *__restrict__ padStart, int *__restrict__ colGroups, int *__restrict__ totals) {
  typedef cub::BlockScan<int, SCANT> Scan;
  __shared__ typename Scan::TempStorage tmp;
  int a[SCANI], b[SCANI];
  for (int i = 0; i < SCANI; ++i) { int c = threadIdx.x * SCANI + i; int n = c < ncol ? colCount[c] : 0; a[i] = n; b[i] = (n + 15) & ~15; }
  int ta, tb;
  Scan(tmp).ExclusiveSum(a, a, ta); __syncthreads();
  Scan(tmp).ExclusiveSum(b, b, tb);
  for (int i = 0; i < SCANI; ++i) {
    int c = threadIdx.x * SCANI + i;
    if (c < ncol) { colStart[c] = a[i]; padStart[c] = b[i]; colGroups[2 * c] = b[i] / 16; int n = colCount[c]; colGroups[2 * c + 1] = (b[i] + ((n + 15) & ~15)) / 16; }
  }
  if (threadIdx.x == 0) { totals[0] = ta; totals[1] = tb; }
}
// Water slots: padded per column.
__global__ void s2w_slots(const unsigned *__restrict__ keysSorted, const int *__restrict__ molSorted, int nm,
                          const int *__restrict__ colStart, const int *__restrict__ padStart, int *__restrict__ order,
                          int *__restrict__ placeOf) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= nm) return;
  int col = keysSorted[k] >> 21, m = molSorted[k];
  int slot = padStart[col] + (k - colStart[col]);
  order[slot] = m; placeOf[m] = slot;
}
// The point of each slot (w = live) and the box of each group of 16 (a half-warp a group).
__global__ void s2_boxes(const float4 *__restrict__ pts, const int4 *__restrict__ shift, const int *__restrict__ order, int npad,
                         Geo geo, float4 *__restrict__ ancSlot, float4 *__restrict__ boxLo, float4 *__restrict__ boxHi) {
  int slot = blockIdx.x * blockDim.x + threadIdx.x;
  bool live = false; float x[3] = {0, 0, 0};
  if (slot < npad) {
    int m = order[slot];
    if (m >= 0) {
      float4 a = pts[m]; int4 s = shift[m];
      x[0] = (float)((double)a.x + (double)s.x * geo.L[0]); x[1] = (float)((double)a.y + (double)s.y * geo.L[1]);
      x[2] = (float)((double)a.z + (double)s.z * geo.L[2]); live = true;
    }
    ancSlot[slot] = make_float4(x[0], x[1], x[2], live ? 1.f : 0.f);
  }
  float lo[3], hi[3];
  for (int ax = 0; ax < 3; ++ax) { lo[ax] = live ? x[ax] : 1e30f; hi[ax] = live ? x[ax] : -1e30f; }
  for (int o = 8; o; o >>= 1)
    for (int ax = 0; ax < 3; ++ax) { lo[ax] = fminf(lo[ax], __shfl_xor_sync(0xffffffffu, lo[ax], o)); hi[ax] = fmaxf(hi[ax], __shfl_xor_sync(0xffffffffu, hi[ax], o)); }
  if ((threadIdx.x & 15) == 0 && slot < npad) {
    int g = slot / 16;
    boxLo[g] = make_float4(lo[0], lo[1], lo[2], 0.f); boxHi[g] = make_float4(hi[0], hi[1], hi[2], 0.f);
  }
}
// Groups of the sorted order (column, then z): a segment is a run of one column without a gap in z
// larger than gapq (in the key's z units); a segment is cut into groups of 16; the last group of a
// segment padded. So the few items of a column far apart (ions; water above and below the protein)
// do not share a group, and every group's box stays small. Parallel: flags, scans, scatter.
__global__ void s2g_flags(const unsigned *__restrict__ keysS, int n, unsigned gapq, int *__restrict__ flag) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= n) return;
  unsigned a = keysS[k];
  int f = 1;
  if (k > 0) { unsigned b = keysS[k - 1]; f = ((a >> 21) != (b >> 21)) || ((a & 0x1FFFFF) - (b & 0x1FFFFF) > gapq); }
  flag[k] = f;
}
// seg = inclusive scan of the flags (1-based); the start and length of each segment.
__global__ void s2g_segments(const int *__restrict__ flag, const int *__restrict__ seg, int n, int *__restrict__ segStart) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= n) return;
  if (flag[k]) segStart[seg[k] - 1] = k;
}
__global__ void s2g_lengths(const int *__restrict__ flag, const int *__restrict__ seg, int n, const int *__restrict__ segStart,
                            int *__restrict__ segGroups, int *__restrict__ segLen) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= n) return;
  if (k == n - 1 || flag[k + 1]) { int s = seg[k] - 1; segLen[s] = k - segStart[s] + 1; segGroups[s] = (k - segStart[s] + 16) / 16; }
}
// A group of 16 whose z span exceeds spanq is cut at its largest gap (a new segment there); the
// grouping is then done again, until no group exceeds it.
__global__ void s2g_span(const unsigned *__restrict__ keysS, const int *__restrict__ seg, int n, const int *__restrict__ segStart,
                         const int *__restrict__ segLen, unsigned spanq, int *__restrict__ flag, int *__restrict__ changed) {
  int k0 = blockIdx.x * blockDim.x + threadIdx.x;
  if (k0 >= n) return;
  int s = seg[k0] - 1, pos = k0 - segStart[s];
  if (pos % 16) return;
  int k1 = min(k0 + 15, segStart[s] + segLen[s] - 1);
  unsigned z0 = keysS[k0] & 0x1FFFFF, z1 = keysS[k1] & 0x1FFFFF;
  if (z1 - z0 <= spanq) return;
  unsigned best = 0; int at = -1, prev = z0;
  for (int k = k0 + 1; k <= k1; ++k) { unsigned z = keysS[k] & 0x1FFFFF; if (z - prev >= best) { best = z - prev; at = k; } prev = z; }
  if (at > 0) { flag[at] = 1; atomicAdd(changed, 1); }
}
// segG0 = exclusive scan of segGroups; the slot of each item, the groups of each column.
__global__ void s2g_slots(const unsigned *__restrict__ keysS, const int *__restrict__ molS, const int *__restrict__ flag,
                          const int *__restrict__ seg, int n, const int *__restrict__ segStart, const int *__restrict__ segGroups,
                          const int *__restrict__ segG0, int *__restrict__ order, int *__restrict__ place, int *__restrict__ colGroups,
                          int *__restrict__ totals) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= n) return;
  int s = seg[k] - 1, pos = k - segStart[s];
  int slot = 16 * (segG0[s] + pos / 16) + pos % 16, m = molS[k];
  order[slot] = m; place[m] = slot;
  int col = keysS[k] >> 21;
  if (colGroups) {
    if (k == 0 || (keysS[k - 1] >> 21) != (unsigned)col) colGroups[2 * col] = segG0[s];
    if (k == n - 1 || (keysS[k + 1] >> 21) != (unsigned)col) colGroups[2 * col + 1] = segG0[s] + segGroups[s];
  }
  if (k == n - 1) { totals[0] = segG0[s] + segGroups[s]; totals[1] = seg[k]; }
}

// The charge, the type, and the LJ coefficients with the water oxygen of each solute slot.
__global__ void s2s_attr(const int *__restrict__ sorder, int nsPad, const float *__restrict__ q, const int *__restrict__ type,
                         const float4 *__restrict__ ljO, float *__restrict__ sq, int *__restrict__ stype, float4 *__restrict__ sLJ) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= nsPad) return;
  int m = sorder[k];
  sq[k] = m >= 0 ? q[m] : 0.f; stype[k] = m >= 0 ? type[m] : 0; sLJ[k] = m >= 0 ? ljO[m] : make_float4(0, 0, 0, 0);
}

// Box test and cells of the nearest image between two groups (the image of h nearest g).
__device__ __forceinline__ bool boxPass(float4 lo, float4 hi, float4 l2, float4 u2, const float *L, float R2, int &code) {
  float cg[3] = {0.5f * (lo.x + hi.x), 0.5f * (lo.y + hi.y), 0.5f * (lo.z + hi.z)};
  float hg[3] = {0.5f * (hi.x - lo.x), 0.5f * (hi.y - lo.y), 0.5f * (hi.z - lo.z)};
  float ch[3] = {0.5f * (l2.x + u2.x), 0.5f * (l2.y + u2.y), 0.5f * (l2.z + u2.z)};
  float hh[3] = {0.5f * (u2.x - l2.x), 0.5f * (u2.y - l2.y), 0.5f * (u2.z - l2.z)};
  float dd = 0.f; code = 0;
  for (int ax = 0; ax < 3; ++ax) {
    float dc = ch[ax] - cg[ax];
    int cell = -__float2int_rn(dc / L[ax]);
    float gap = fabsf(dc + cell * L[ax]) - hg[ax] - hh[ax];
    if (gap > 0.f) dd += gap * gap;
    code |= (cell + 4) << (16 + 4 * ax);
  }
  return dd < R2 && l2.x <= u2.x;   // an empty group has an inverted box
}
// The cells chosen from the centers are the only image within the reach R of every pair of the two
// groups if R + both half-edges < L/2 on each axis; otherwise another image could be within R.
__device__ __forceinline__ bool imageAmbiguous(float4 lo, float4 hi, float4 l2, float4 u2, const float *L, float R) {
  if (l2.x > u2.x || lo.x > hi.x) return false;
  return R + 0.5f * (hi.x - lo.x) + 0.5f * (u2.x - l2.x) >= 0.5f * L[0] || R + 0.5f * (hi.y - lo.y) + 0.5f * (u2.y - l2.y) >= 0.5f * L[1] ||
         R + 0.5f * (hi.z - lo.z) + 0.5f * (u2.z - l2.z) >= 0.5f * L[2];
}

// A list build, a warp a group g of the I side. Candidates h of the J side: (MODE 0, WW) the groups
// of the columns within reach, h >= g; (MODE 1, SW) every water group; (MODE 2, SS) every solute
// group h >= g. Entries (j, mask | cells) to a buffer of the warp, then to ceil(n/64) blocks of the
// pool taken with one atomic addition.
constexpr int BW = 2, ECAP = 2048;
struct BuildArgs {
  const float4 *ancI, *boxLoI, *boxHiI, *ancJ, *boxLoJ, *boxHiJ; int ngI, ngJ; const int *colGroups; Geo geo; int nx, ny; float R2;
  const int *sorder, *exOff, *exPart;   // SS: the atoms of the slots and the exclusions (solute numbering)
  int *pool; int poolCap; int2 *outer; int *outerCount, *blockStart, *unitGroup, *unitOrd, *flags;
};
template <int MODE>
__global__ void __launch_bounds__(32 * BW) s2_build(BuildArgs A) {
  __shared__ int2 buf[BW][ECAP];
  __shared__ float4 mine[BW][16];
  __shared__ int mineAtom[BW][16];
  const int lane = threadIdx.x & 31, wb = threadIdx.x >> 5;
  const int g = blockIdx.x * BW + wb;
  if (g >= A.ngI) return;
  if (lane < 16) { mine[wb][lane] = A.ancI[16 * g + lane]; if (MODE == 2) mineAtom[wb][lane] = A.sorder[16 * g + lane]; }
  __syncwarp();
  float4 lo = A.boxLoI[g], hi = A.boxHiI[g];
  int n = 0;
  auto scan = [&](int h0, int h1) {
    for (int hb = h0; hb < h1; hb += 32) {
      int h = hb + lane; bool pass = false; int code = 0;
      if (h < h1) {
        float4 l2 = A.boxLoJ[h], u2 = A.boxHiJ[h];
        pass = boxPass(lo, hi, l2, u2, A.geo.L, A.R2, code);
        if (imageAmbiguous(lo, hi, l2, u2, A.geo.L, sqrtf(A.R2))) atomicOr(A.flags, 16);
      }
      unsigned pm = __ballot_sync(0xffffffffu, pass);
      while (pm) {
        int s0 = __ffs(pm) - 1; pm &= pm - 1;
        int s1 = pm ? __ffs(pm) - 1 : -1; if (s1 >= 0) pm &= pm - 1;
        int src = lane < 16 ? s0 : s1;
        int hsel = __shfl_sync(0xffffffffu, h, src < 0 ? 0 : src);
        int csel = __shfl_sync(0xffffffffu, code, src < 0 ? 0 : src);
        unsigned mask = 0; int j = 16 * hsel + (lane & 15);
        if (src >= 0) {
          float4 aj = A.ancJ[j];
          if (aj.w != 0.f) {
            float sh[3]; cellShift((unsigned)csel, A.geo.L, sh);
            float xj = aj.x + sh[0], yj = aj.y + sh[1], zj = aj.z + sh[2];
#pragma unroll
            for (int q = 0; q < 16; ++q) {
              float4 ai = mine[wb][q];
              float x = ai.x - xj, y = ai.y - yj, z = ai.z - zj;
              bool in = ai.w != 0.f && x * x + y * y + z * z < A.R2 && (MODE == 1 || hsel != g || q < (lane & 15));
              mask |= (unsigned)in << q;
            }
            if (MODE == 2 && mask) {   // clear the excluded pairs
              int aj_ = A.sorder[j];
              for (int k = A.exOff[aj_]; k < A.exOff[aj_ + 1]; ++k) {
                int p = A.exPart[k];
#pragma unroll
                for (int q = 0; q < 16; ++q) if (mineAtom[wb][q] == p) mask &= ~(1u << q);
              }
            }
          }
        }
        unsigned keep = __ballot_sync(0xffffffffu, mask != 0);
        int at = n + __popc(keep & ((1u << lane) - 1));
        if (mask && at < ECAP) buf[wb][at] = make_int2(j, (int)(mask | (unsigned)csel));
        n += __popc(keep);
      }
    }
  };
  if (MODE == 0) {
    float cg[3] = {0.5f * (lo.x + hi.x), 0.5f * (lo.y + hi.y), 0.5f * (lo.z + hi.z)};
    int cx = min(A.geo.ncx - 1, max(0, (int)(cg[0] / A.geo.L[0] * A.geo.ncx))), cy = min(A.geo.ncy - 1, max(0, (int)(cg[1] / A.geo.L[1] * A.geo.ncy)));
    if (lo.x <= hi.x)
      for (int dx = -A.nx; dx <= A.nx; ++dx)
        for (int dy = -A.ny; dy <= A.ny; ++dy) {
          int qx = (cx + dx + 64 * A.geo.ncx) % A.geo.ncx, qy = (cy + dy + 64 * A.geo.ncy) % A.geo.ncy;
          int col = qx * A.geo.ncy + ((qx & 1) ? A.geo.ncy - 1 - qy : qy);
          scan(max(A.colGroups[2 * col], g), A.colGroups[2 * col + 1]);
        }
  } else if (lo.x <= hi.x) {
    scan(MODE == 2 ? g : 0, A.ngJ);
  }
  __syncwarp();
  if (n > ECAP) { if (lane == 0) atomicOr(A.flags, 1); return; }
  int nb = (n + 63) / 64, start = 0;
  if (lane == 0) start = atomicAdd(A.pool, nb);
  start = __shfl_sync(0xffffffffu, start, 0);
  if (start + nb > A.poolCap) { if (lane == 0) atomicOr(A.flags, 2); return; }
  for (int k = lane; k < nb * 64; k += 32) A.outer[(size_t)start * 64 + k] = k < n ? buf[wb][k] : make_int2(0, 0);
  for (int k = lane; k < nb; k += 32) { A.unitGroup[start + k] = g; A.unitOrd[start + k] = k; }
  if (lane == 0) { A.outerCount[g] = n; A.blockStart[g] = start; }
}

// Two-list mode: after a WW build the water slots are renumbered; the SW entries follow, their
// cells corrected by the change of the molecule's whole cells: p_j = x + (shift + cell) L kept.
__global__ void s2_remap(int2 *__restrict__ ent, const int *__restrict__ count, const int *__restrict__ blockStart, int ng,
                         const int *__restrict__ orderOld, const int4 *__restrict__ shiftOld, const int *__restrict__ placeOf,
                         const int4 *__restrict__ shiftNew, int *__restrict__ flags) {
  int g = blockIdx.x, lane = threadIdx.x;
  if (g >= ng) return;
  int n = count[g]; int2 *e = ent + (size_t)blockStart[g] * 64;
  for (int k = lane; k < n; k += blockDim.x) {
    int2 x = e[k]; int m = orderOld[x.x];
    if (m < 0) { atomicOr(flags, 4); continue; }
    int4 so = shiftOld[m], sn = shiftNew[m];
    int d[3] = {so.x - sn.x, so.y - sn.y, so.z - sn.z};
    unsigned c = (unsigned)x.y, out = c & 0xFFFFu;
    for (int ax = 0; ax < 3; ++ax) {
      int cell = (int)((c >> (16 + 4 * ax)) & 15u) + d[ax];
      if (cell < 0 || cell > 15) atomicOr(flags, 8);
      out |= (unsigned)(cell & 15) << (16 + 4 * ax);
    }
    e[k] = make_int2(placeOf[m], (int)out);
  }
}

// Each step: the f32 positions in the order of the slots, moved by the cells of the build (added in
// f64, then rounded); the water anchor of each slot; the solute point (w = charge) of each slot.
template <class FT>
__global__ void s2w_pack(const FT *__restrict__ frame, const int *__restrict__ order, const int4 *__restrict__ shift, int npad, Geo geo,
                         float4 *__restrict__ pos, float4 *__restrict__ ancFrame) {
  int slot = blockIdx.x * blockDim.x + threadIdx.x;
  if (slot >= npad) return;
  int m = order[slot];
  if (m < 0) { float4 far = make_float4(-1e4f, -1e4f, -1e4f, 0.f); pos[3 * slot] = pos[3 * slot + 1] = pos[3 * slot + 2] = far; ancFrame[slot] = far; return; }
  int4 s = shift[m];
  const FT *x = frame + 3 * ((size_t)geo.nS + (size_t)geo.ns * m);
  float4 p[3];
  for (int t = 0; t < 3; ++t)
    p[t] = make_float4((float)((double)x[3 * t] + (double)s.x * geo.L[0]), (float)((double)x[3 * t + 1] + (double)s.y * geo.L[1]),
                       (float)((double)x[3 * t + 2] + (double)s.z * geo.L[2]), 0.f);
  pos[3 * slot] = p[0]; pos[3 * slot + 1] = p[1]; pos[3 * slot + 2] = p[2];
  ancFrame[slot] = make_float4(p[0].x, p[0].y, p[0].z, 1.f);
}
template <class FT>
__global__ void s2s_pack(const FT *__restrict__ frame, const int *__restrict__ sorder, const int4 *__restrict__ sshift, const float *__restrict__ sq,
                         int nsPad, Geo geo, float4 *__restrict__ spos) {
  int slot = blockIdx.x * blockDim.x + threadIdx.x;
  if (slot >= nsPad) return;
  int m = sorder[slot];
  if (m < 0) { spos[slot] = make_float4(-2e4f, -2e4f, -2e4f, 0.f); return; }
  int4 s = sshift[m]; const FT *x = frame + 3 * (size_t)m;
  spos[slot] = make_float4((float)((double)x[0] + (double)s.x * geo.L[0]), (float)((double)x[1] + (double)s.y * geo.L[1]),
                           (float)((double)x[2] + (double)s.z * geo.L[2]), sq[slot]);
}

// The pruning, a warp a group: inner = outer & (r^2 < R_in^2) at the points of this step.
__global__ void __launch_bounds__(128) s2_prune(const float4 *__restrict__ posI, const float4 *__restrict__ posJ, int ng, Geo geo,
                                                float Rin2, const int2 *__restrict__ outer, const int *__restrict__ outerCount,
                                                const int *__restrict__ blockStart, int2 *__restrict__ inner, int *__restrict__ innerCount) {
  __shared__ float4 mine[4][16];
  const int lane = threadIdx.x & 31, wb = threadIdx.x >> 5;
  const int g = blockIdx.x * 4 + wb;
  if (g >= ng) return;
  if (lane < 16) mine[wb][lane] = posI[16 * g + lane];
  __syncwarp();
  const int no = outerCount[g];
  const int2 *src = outer + (size_t)blockStart[g] * 64;
  int2 *dst = inner + (size_t)blockStart[g] * 64;
  int k = 0;
  for (int e0 = 0; e0 < no; e0 += 32) {
    int e = e0 + lane;
    unsigned keepMask = 0; int2 en = make_int2(0, 0);
    if (e < no) {
      en = src[e];
      unsigned om = (unsigned)en.y & 0xFFFFu;
      float4 aj = posJ[en.x];
      float sh[3]; cellShift((unsigned)en.y, geo.L, sh);
      float xj = aj.x + sh[0], yj = aj.y + sh[1], zj = aj.z + sh[2];
#pragma unroll
      for (int q = 0; q < 16; ++q) {
        float4 ai = mine[wb][q];
        float x = ai.x - xj, y = ai.y - yj, z = ai.z - zj;
        keepMask |= (unsigned)(x * x + y * y + z * z < Rin2) << q;
      }
      keepMask &= om;
    }
    unsigned keep = __ballot_sync(0xffffffffu, keepMask != 0);
    int at = k + __popc(keep & ((1u << lane) - 1));
    if (keepMask) dst[at] = make_int2(en.x, (int)(((unsigned)en.y & 0xFFFF0000u) | keepMask));
    k += __popc(keep);
  }
  if (lane == 0) innerCount[g] = k;
}
struct ActivePred {
  const int *unitGroup, *unitOrd, *innerCount;
  __device__ bool operator()(int b) const { return unitOrd[b] * 64 < innerCount[unitGroup[b]]; }
};
// After the kernels, coalesced: a thread a float of a slot, to its atom; the slot cleared.
__global__ void s2w_unpack(float *__restrict__ fslot, const int *__restrict__ order, int npad, Geo geo, float *__restrict__ fatom) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= 9 * npad) return;
  int slot = i / 9, k = i - 9 * slot;
  int m = order[slot];
  float v = fslot[i];
  if (m >= 0) fatom[3 * ((size_t)geo.nS + (size_t)geo.ns * m) + k] = v;
  fslot[i] = 0.f;
}
__global__ void s2s_unpack(float *__restrict__ fS, const int *__restrict__ sorder, int nsPad, float *__restrict__ fatom) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= 3 * nsPad) return;
  int slot = i / 3, k = i - 3 * slot;
  int m = sorder[slot];
  float v = fS[i];
  if (m >= 0) fatom[3 * (size_t)m + k] = v;
  fS[i] = 0.f;
}

// ---------------------------------------------------------------------------
struct Input {
  int64_t natom, nW, ns, nS, ntS, nEx; double L[3], qw[4], AOO, BOO, beta, KE, rc, rOH, gs, a, rmol, dOE;
  std::vector<double> q, AOS, BOS, LA, LB; std::vector<int> type, exOff, exPart;
};
static Input readInput(const char *path) {
  Input in; FILE *f = fopen(path, "rb"); if (!f) { perror(path); exit(1); }
  int64_t h[8]; if (fread(h, 8, 8, f) != 8) exit(1);
  in.natom = h[0]; in.nW = h[1]; in.ns = h[2]; in.nS = h[3]; in.ntS = h[4]; in.nEx = h[5];
  double d[17]; if (fread(d, 8, 17, f) != 17) exit(1);
  memcpy(in.L, d, 24); memcpy(in.qw, d + 3, 32); in.AOO = d[7]; in.BOO = d[8]; in.beta = d[9]; in.KE = d[10]; in.rc = d[11];
  in.rOH = d[12]; in.gs = d[13]; in.a = d[14]; in.rmol = d[15]; in.dOE = d[16];
  auto rd = [&](auto &v, size_t n) { v.resize(n); if (n && fread(v.data(), sizeof(v[0]), n, f) != n) { fprintf(stderr, "short input\n"); exit(1); } };
  rd(in.q, in.nS); rd(in.type, in.nS); rd(in.AOS, in.nS); rd(in.BOS, in.nS); rd(in.LA, in.ntS * in.ntS); rd(in.LB, in.ntS * in.ntS);
  rd(in.exOff, in.nS + 1); rd(in.exPart, in.nEx);
  fclose(f); return in;
}
static int32_t rd32(FILE *f) { int32_t v; if (fread(&v, 4, 1, f) != 1) { fprintf(stderr, "short DCD\n"); exit(1); } return v; }
static std::vector<float> readDCD(const char *path, int natom, int &nframes, double box[3]) {
  FILE *f = fopen(path, "rb"); if (!f) { perror(path); exit(1); }
  int32_t n = rd32(f); std::vector<char> h(n); if (fread(h.data(), 1, n, f) != (size_t)n) exit(1); rd32(f);
  int32_t icntrl[20]; memcpy(icntrl, h.data() + 4, 80);
  int hascell = icntrl[10]; int nset = icntrl[0];
  n = rd32(f); fseek(f, n, SEEK_CUR); rd32(f);
  n = rd32(f); int32_t na = rd32(f); rd32(f);
  if (na != natom) { fprintf(stderr, "DCD has %d atoms, input %d\n", na, natom); exit(1); }
  if (nframes <= 0 || nframes > nset) nframes = nset;
  std::vector<float> X((size_t)nframes * natom * 3); std::vector<float> c(natom);
  for (int t = 0; t < nframes; ++t) {
    if (hascell) { n = rd32(f); double cell[6]; if (fread(cell, 8, 6, f) != 6) exit(1); rd32(f); box[0] = cell[0]; box[1] = cell[2]; box[2] = cell[5]; }
    for (int ax = 0; ax < 3; ++ax) {
      n = rd32(f); if (fread(c.data(), 4, natom, f) != (size_t)natom) exit(1); rd32(f);
      for (int i = 0; i < natom; ++i) X[((size_t)t * natom + i) * 3 + ax] = c[i];
    }
  }
  fclose(f); return X;
}
// NVTX ranges name the phases of a step for the attribution of the shared kernels (sort, scan,
// select, prune) to the WW and the solute lists.
struct Range { Range(const char *n) { nvtxRangePushA(n); } ~Range() { nvtxRangePop(); } };
template <class T> static T *dalloc(size_t n) { T *p; CK(cudaMalloc(&p, std::max<size_t>(1, n) * sizeof(T))); CK(cudaMemset(p, 0, std::max<size_t>(1, n) * sizeof(T))); return p; }
static int gridOf(long long n, int t) { return (int)std::max(1LL, (n + t - 1) / t); }

// A list's device state.
struct List {
  int ngCap = 0, poolCap = 0;
  int *pool, *outerCount, *innerCount, *blockStart, *unitGroup, *unitOrd, *activeBlocks, *nActive, *flags;
  int2 *outer, *inner;
  int nBlocks = 0, ng = 0; long long builds = 0, prunes = 0, sumOuter = 0, sumInner = 0;
  void alloc(int ngc, int pc) {
    ngCap = ngc; poolCap = pc;
    pool = dalloc<int>(1); flags = dalloc<int>(1); outerCount = dalloc<int>(ngc); innerCount = dalloc<int>(ngc);
    blockStart = dalloc<int>(ngc); unitGroup = dalloc<int>(pc); unitOrd = dalloc<int>(pc); activeBlocks = dalloc<int>(pc);
    nActive = dalloc<int>(1); outer = dalloc<int2>((size_t)pc * 64); inner = dalloc<int2>((size_t)pc * 64);
  }
  ListView view() const { return ListView{activeBlocks, nActive, unitGroup, unitOrd, blockStart, innerCount, inner}; }
};

int main(int argc, char **argv) {
  if (argc < 4) { fprintf(stderr, "usage: stage2 INPUT.s2in TRAJ.dcd|FRAME.f64 OUTPREFIX [nframes]\n"); return 1; }
  Input in = readInput(argv[1]);
  std::string trajPath = argv[2], out = argv[3];
  int nframes = argc > 4 ? atoi(argv[4]) : 0;
  const bool single = trajPath.size() > 4 && trajPath.substr(trajPath.size() - 4) == ".f64";
  const int nW = (int)in.nW, ns = (int)in.ns, nS = (int)in.nS, natom = (int)in.natom, ntS = (int)in.ntS;
  const int mode = getenv("S2_LISTS") ? atoi(getenv("S2_LISTS")) : 2;
  double box[3] = {in.L[0], in.L[1], in.L[2]};
  std::vector<float> X; std::vector<double> X64;
  if (single) {
    FILE *f = fopen(trajPath.c_str(), "rb"); if (!f) { perror(argv[2]); return 1; }
    X64.resize((size_t)natom * 3); double b[3];
    if (fread(b, 8, 3, f) != 3 || fread(X64.data(), 8, X64.size(), f) != X64.size()) { fprintf(stderr, "short frame\n"); return 1; }
    fclose(f); for (int ax = 0; ax < 3; ++ax) { box[ax] = b[ax]; in.L[ax] = b[ax]; } nframes = 1;
  } else {
    X = readDCD(argv[2], natom, nframes, box);
    for (int ax = 0; ax < 3; ++ax) if (std::fabs(box[ax] - in.L[ax]) > 1e-6 * in.L[ax]) { fprintf(stderr, "box differs (NVE expected)\n"); return 1; }
  }
  double maxStep = 0; long long unwrapped = 0;
  for (int t = 1; t < nframes; ++t)
    for (size_t i = 0; i < (size_t)natom * 3; ++i) {
      size_t k = (size_t)t * natom * 3 + i, kp = k - (size_t)natom * 3; int ax = i % 3;
      double d = (double)X[k] - X[kp], dm = d - in.L[ax] * std::round(d / in.L[ax]);
      if (dm != d) { X[k] = (float)(X[kp] + dm); ++unwrapped; }
      maxStep = std::max(maxStep, std::fabs(dm));
    }
  const double skinOut = 3.0, skinIn = 0.6, widen = 3e-6 * (in.L[0] + in.L[1] + in.L[2]);
  const double rc = in.rc, rm = in.rmol;
  const double RW = rc + 2 * rm + skinOut + widen, RWin = rc + 2 * rm + skinIn + widen;
  const double RSW = rc + rm + skinOut + widen, RSWin = rc + rm + skinIn + widen;
  const double RSS = rc + skinOut + widen, RSSin = rc + skinIn + widen;
  const float limB2 = (float)std::pow(0.5 * skinOut, 2), limP2 = (float)std::pow(0.5 * skinIn, 2);
  // Water columns: edge (16/rho)^(1/3) over the molecules (the prototype's); solute columns: S2_SEDGE (A).
  double vol = in.L[0] * in.L[1] * in.L[2], rho = nW / vol, edge = std::cbrt(16.0 / rho);
  Geo geo; for (int ax = 0; ax < 3; ++ax) geo.L[ax] = (float)in.L[ax];
  geo.ncx = std::max(1, (int)std::lround(in.L[0] / edge)); geo.ncy = std::max(1, (int)std::lround(in.L[1] / edge)); geo.nS = nS; geo.ns = ns;
  const int ncol = geo.ncx * geo.ncy;
  const int nx = (int)std::ceil(RW / (in.L[0] / geo.ncx)), ny = (int)std::ceil(RW / (in.L[1] / geo.ncy));
  if (2 * nx + 1 > geo.ncx || 2 * ny + 1 > geo.ncy || ncol > SCANT * SCANI) { fprintf(stderr, "columns: too few for the reach\n"); return 1; }
  const double sedge = getenv("S2_SEDGE") ? atof(getenv("S2_SEDGE")) : 6.0;
  const int sncx = std::max(1, (int)std::lround(in.L[0] / sedge)), sncy = std::max(1, (int)std::lround(in.L[1] / sedge));
  const int npadCap = 16 * std::max(1, nW), ngCapW = npadCap / 16;
  const int nsPadCap = 16 * std::max(1, nS), ngSCap = std::max(1, nS);
  int nsPad = 0, ngS = 0;
  const double wgap = getenv("S2_WGAP") ? atof(getenv("S2_WGAP")) : 4.0, sgap = getenv("S2_SGAP") ? atof(getenv("S2_SGAP")) : 3.0;
  const double wspan = getenv("S2_WSPAN") ? atof(getenv("S2_WSPAN")) : 9.0, sspan = getenv("S2_SSPAN") ? atof(getenv("S2_SSPAN")) : 6.0;
  const unsigned wgapq = (unsigned)(wgap / in.L[2] * (1u << 21)), sgapq = (unsigned)(sgap / in.L[2] * (1u << 21));
  const unsigned wspanq = (unsigned)(wspan / in.L[2] * (1u << 21)), sspanq = (unsigned)(sspan / in.L[2] * (1u << 21));
  const int sncol = sncx * sncy;
  if (sncol > 2047 || ncol > 2047) { fprintf(stderr, "too many columns for the keys\n"); return 1; }
  printf("input %s: %d atoms, %d waters x %d sites, %d solute atoms (%d types); box %.4f %.4f %.4f A; %s: %d frames, largest step %.4f A, %lld unwrapped\n",
         argv[1], natom, nW, ns, nS, ntS, in.L[0], in.L[1], in.L[2], argv[2], nframes, maxStep, unwrapped);
  printf("r_mol %.6f A (oxygen anchor); reaches WW %.4f/%.4f SW %.4f/%.4f SS %.4f/%.4f A; water columns %d x %d (edge %.3f, reach %d x %d); "
         "solute columns %d x %d (edge %.2f); groups cut at z gaps over %.1f / %.1f A and spans over %.1f / %.1f A (water / solute); lists: %s\n", rm, RW, RWin, RSW, RSWin, RSS, RSSin, geo.ncx, geo.ncy, edge, nx, ny,
         sncx, sncy, sedge, wgap, sgap, wspan, sspan, mode == 2 ? "two (WW on the anchors; SW+SS on anchors and solute)" : "one (all tested together)");

  // Device state.
  float *dF32 = nullptr; double *dF64 = nullptr;
  if (single) { dF64 = dalloc<double>(X64.size()); CK(cudaMemcpy(dF64, X64.data(), X64.size() * 8, cudaMemcpyHostToDevice)); }
  else { dF32 = dalloc<float>(X.size()); CK(cudaMemcpy(dF32, X.data(), X.size() * 4, cudaMemcpyHostToDevice)); std::vector<float>().swap(X); }
  float4 *ancNow = dalloc<float4>(nW), *solNow = dalloc<float4>(nS);
  float4 *refWB = dalloc<float4>(nW), *refWP = dalloc<float4>(nW), *refLB = dalloc<float4>(nW), *refLP = dalloc<float4>(nW);
  float4 *refSB = dalloc<float4>(nS), *refSP = dalloc<float4>(nS);
  unsigned *maxBits = dalloc<unsigned>(6);
  // water side
  unsigned *keys = dalloc<unsigned>(std::max(nW, nS)), *keysS = dalloc<unsigned>(std::max(nW, nS));
  int *vals = dalloc<int>(std::max(nW, nS)), *molS = dalloc<int>(std::max(nW, nS));
  int *colGroups = dalloc<int>(2 * ncol), *totals = dalloc<int>(2);
  const int nmax = std::max(nW, nS);
  int *gFlag = dalloc<int>(nmax), *gSeg = dalloc<int>(nmax), *gStart = dalloc<int>(nmax), *gGroups = dalloc<int>(nmax), *gG0 = dalloc<int>(nmax);
  int *gLen = dalloc<int>(nmax), *gChanged = dalloc<int>(1);
  int *order = dalloc<int>(npadCap), *placeOf = dalloc<int>(nW), *orderOld = dalloc<int>(npadCap);
  int4 *shift = dalloc<int4>(nW), *shiftOld = dalloc<int4>(nW);
  float4 *ancSlot = dalloc<float4>(npadCap), *boxLo = dalloc<float4>(ngCapW), *boxHi = dalloc<float4>(ngCapW);
  float4 *pos = dalloc<float4>(3 * (size_t)npadCap), *ancFrame = dalloc<float4>(npadCap);
  float *fW = dalloc<float>(9 * (size_t)npadCap), *fatom = dalloc<float>(3 * (size_t)natom);
  // solute side
  int4 *sshift = dalloc<int4>(nS); int *sorder = dalloc<int>(nsPadCap), *splace = dalloc<int>(nS), *stype = dalloc<int>(nsPadCap);
  float *sq = dalloc<float>(nsPadCap); float4 *sLJ = dalloc<float4>(nsPadCap), *spos = dalloc<float4>(nsPadCap), *sAnc = dalloc<float4>(nsPadCap);
  float4 *sboxLo = dalloc<float4>(ngSCap), *sboxHi = dalloc<float4>(ngSCap); float *fS = dalloc<float>(3 * (size_t)nsPadCap);
  int *totalsS = dalloc<int>(2);
  int *dType = dalloc<int>(nS), *dExOff = dalloc<int>(nS + 1), *dExPart = dalloc<int>(in.nEx);
  float *dQ = dalloc<float>(nS); float4 *dLjO = dalloc<float4>(nS), *dLjTab = dalloc<float4>(std::max<int64_t>(1, in.ntS * in.ntS));
  {
    std::vector<float> qf(nS); std::vector<float4> lo(nS), lt(std::max<int64_t>(1, in.ntS * in.ntS));
    for (int i = 0; i < nS; ++i) { qf[i] = (float)in.q[i]; lo[i] = make_float4((float)(12 * in.AOS[i]), (float)(6 * in.BOS[i]), (float)in.AOS[i], (float)in.BOS[i]); }
    for (int k = 0; k < ntS * ntS; ++k) lt[k] = make_float4((float)(12 * in.LA[k]), (float)(6 * in.LB[k]), (float)in.LA[k], (float)in.LB[k]);
    if (nS) {
      CK(cudaMemcpy(dQ, qf.data(), 4 * nS, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dLjO, lo.data(), 16 * nS, cudaMemcpyHostToDevice));
      CK(cudaMemcpy(dType, in.type.data(), 4 * nS, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dExOff, in.exOff.data(), 4 * (nS + 1), cudaMemcpyHostToDevice));
      if (in.nEx) CK(cudaMemcpy(dExPart, in.exPart.data(), 4 * in.nEx, cudaMemcpyHostToDevice));
      CK(cudaMemcpy(dLjTab, lt.data(), 16 * lt.size(), cudaMemcpyHostToDevice));
    }
  }
  if (ntS > 32) { fprintf(stderr, "more than 32 solute types\n"); return 1; }
  // Pools: from the densities, twice the expected entries.
  double rhoAll = natom / vol;
  int poolW = (int)(2.0 * ngCapW * (4.0 / 3.0 * M_PI * rho * std::pow(RW + 6.0, 3) / 2.0 / 64.0 + 1.0));
  const double ngEst = nS / 8.0 + 8;
  int poolSW = (int)(4.0 * ngEst * (4.0 / 3.0 * M_PI * rho * std::pow(RSW + 6.0, 3) / 64.0 + 1.0)) + 64;
  int poolSS = (int)(4.0 * ngEst * (4.0 / 3.0 * M_PI * rhoAll * std::pow(RSS + 6.0, 3) / 2.0 / 64.0 + 1.0)) + 64;
  List WW, SW, SS; WW.alloc(ngCapW, poolW); if (nS) { SW.alloc(ngSCap, poolSW); SS.alloc(ngSCap, poolSS); }
  double *dE = dalloc<double>(3); Counters *dC = dalloc<Counters>(3);
  // cub temporaries.
  size_t tmpSort = 0, tmpSel = 0;
  cub::DeviceRadixSort::SortPairs(nullptr, tmpSort, keys, keysS, vals, molS, std::max(nW, nS), 0, 32);
  {
    ActivePred pr{WW.unitGroup, WW.unitOrd, WW.innerCount};
    int pc = std::max(poolW, std::max(poolSW, poolSS));
    cub::DeviceSelect::If(nullptr, tmpSel, thrust::counting_iterator<int>(0), WW.activeBlocks, WW.nActive, pc, pr);
  }
  size_t tmpScan = 0; cub::DeviceScan::InclusiveSum(nullptr, tmpScan, gFlag, gSeg, nmax);
  { size_t t2 = 0; cub::DeviceScan::ExclusiveSum(nullptr, t2, gGroups, gG0, nmax); tmpScan = std::max(tmpScan, t2); }
  void *dTmp = dalloc<char>(std::max(std::max(tmpSort, tmpSel), tmpScan));
  // Pair parameters.
  Params p{};
  for (int c = 0; c < 3; ++c) p.L[c] = (float)in.L[c];
  p.rc2 = (float)(rc * rc); p.A = (float)in.AOO; p.B = (float)in.BOO; p.A12 = 12.0f * p.A; p.B6 = 6.0f * p.B;
  float fq[4]; for (int s = 0; s < 4; ++s) fq[s] = (float)in.qw[s];
  float fke = (float)in.KE;
  for (int s = 0; s < 4; ++s) { p.kq[s] = fke * fq[s]; for (int t = 0; t < 4; ++t) p.k[s][t] = fke * fq[s] * fq[t]; }
  p.a = (float)in.a; p.oma = 1.0f - 2.0f * p.a; p.invh = (float)(1.0 / in.rOH); p.invs = (float)(1.0 / in.gs); p.beta = (float)in.beta;
  Table tf = makeTable(rc, in.beta);
  float4 *dTab = dalloc<float4>(tf.c.size()); CK(cudaMemcpy(dTab, tf.c.data(), tf.c.size() * 16, cudaMemcpyHostToDevice));
  const int TPB = 128;
  int npad = 0, ngW = 0;
  FILE *fo = fopen((out + ".txt").c_str(), "w");
  auto F32 = [&](int t) { return dF32 + (size_t)t * natom * 3; };

  auto finishBuild = [&](List &Lst, int ng, const char *name, int t) {
    int pl, fl; CK(cudaMemcpy(&pl, Lst.pool, 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(&fl, Lst.flags, 4, cudaMemcpyDeviceToHost));
    if (fl) { fprintf(stderr, "%s build at frame %d: flags %d (1, 2: overflow; 16: an image not unique) (pool %d of %d)\n", name, t, fl, pl, Lst.poolCap); exit(3); }
    Lst.nBlocks = pl; Lst.ng = ng; ++Lst.builds;
    std::vector<int> oc(ng); CK(cudaMemcpy(oc.data(), Lst.outerCount, 4 * ng, cudaMemcpyDeviceToHost));
    long long s = 0; for (int x : oc) s += x; Lst.sumOuter += s;
  };
  // The groups of a sorted order; returns the number of groups.
  auto group = [&](int n, unsigned gapq, unsigned spanq, int *ord, int *plc, int *colG, int *tot) {
    s2g_flags<<<gridOf(n, TPB), TPB>>>(keysS, n, gapq, gFlag);
    int nseg = 0;
    for (int pass = 0; ; ++pass) {
      size_t ts = tmpScan; cub::DeviceScan::InclusiveSum(dTmp, ts, gFlag, gSeg, n);
      s2g_segments<<<gridOf(n, TPB), TPB>>>(gFlag, gSeg, n, gStart);
      s2g_lengths<<<gridOf(n, TPB), TPB>>>(gFlag, gSeg, n, gStart, gGroups, gLen);
      CK(cudaMemsetAsync(gChanged, 0, 4));
      s2g_span<<<gridOf(n, TPB), TPB>>>(keysS, gSeg, n, gStart, gLen, spanq, gFlag, gChanged);
      int ch; CK(cudaMemcpy(&ch, gChanged, 4, cudaMemcpyDeviceToHost));
      if (!ch) break;
      if (pass == 8) { fprintf(stderr, "grouping: spans still too long after 8 passes\n"); exit(3); }
    }
    CK(cudaMemcpy(&nseg, gSeg + n - 1, 4, cudaMemcpyDeviceToHost));
    size_t ts = tmpScan; cub::DeviceScan::ExclusiveSum(dTmp, ts, gGroups, gG0, nseg);
    s2g_slots<<<gridOf(n, TPB), TPB>>>(keysS, molS, gFlag, gSeg, n, gStart, gGroups, gG0, ord, plc, colG, tot);
    int h[2]; CK(cudaMemcpy(h, tot, 8, cudaMemcpyDeviceToHost));
    return h[0];
  };
  auto buildWW = [&](int t) {
    Range r_("build WW");
    CK(cudaMemsetAsync(colGroups, 0, ncol * 8)); CK(cudaMemsetAsync(order, 0xFF, npadCap * 4)); CK(cudaMemsetAsync(WW.pool, 0, 4));
    s2_keys<<<gridOf(nW, TPB), TPB>>>(ancNow, nW, geo, geo.ncx, geo.ncy, keys, vals, shift, nullptr);
    size_t ts = tmpSort; cub::DeviceRadixSort::SortPairs(dTmp, ts, keys, keysS, vals, molS, nW, 0, 32);
    ngW = group(nW, wgapq, wspanq, order, placeOf, colGroups, totals); npad = 16 * ngW;
    s2_boxes<<<gridOf(npad, TPB), TPB>>>(ancNow, shift, order, npad, geo, ancSlot, boxLo, boxHi);
    BuildArgs A{ancSlot, boxLo, boxHi, ancSlot, boxLo, boxHi, ngW, ngW, colGroups, geo, nx, ny, (float)(RW * RW), nullptr, nullptr, nullptr,
                WW.pool, WW.poolCap, WW.outer, WW.outerCount, WW.blockStart, WW.unitGroup, WW.unitOrd, WW.flags};
    s2_build<0><<<gridOf(ngW, BW), 32 * BW>>>(A); CK(cudaGetLastError());
    finishBuild(WW, ngW, "WW", t);
    std::vector<float4> wl(ngW), wh(ngW);
    CK(cudaMemcpy(wl.data(), boxLo, 16 * ngW, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(wh.data(), boxHi, 16 * ngW, cudaMemcpyDeviceToHost));
    float hw[3] = {0, 0, 0}, ew[3] = {0, 0, 0};
    for (int g = 0; g < ngW; ++g) for (int ax = 0; ax < 3; ++ax) { float e = (&wh[g].x)[ax] - (&wl[g].x)[ax]; hw[ax] = std::max(hw[ax], 0.5f * e); ew[ax] += e; }
    if (WW.builds <= 2)
      printf("WW build %lld (frame %d): %d water groups (%.2f molecules a group), mean box edges %.2f %.2f %.2f A, largest half-edges %.2f %.2f %.2f A; outer entries %lld\n",
             WW.builds, t, ngW, (double)nW / ngW, ew[0] / ngW, ew[1] / ngW, ew[2] / ngW, hw[0], hw[1], hw[2], WW.sumOuter);
  };
  auto buildSL = [&](int t, bool waterBoxesFresh) {
    Range r_("build SL");
    CK(cudaMemsetAsync(SW.pool, 0, 4)); CK(cudaMemsetAsync(SS.pool, 0, 4));
    CK(cudaMemsetAsync(sorder, 0xFF, 4 * (size_t)nsPadCap));
    s2_keys<<<gridOf(nS, TPB), TPB>>>(solNow, nS, geo, sncx, sncy, keys, vals, sshift, nullptr);
    size_t ts = tmpSort; cub::DeviceRadixSort::SortPairs(dTmp, ts, keys, keysS, vals, molS, nS, 0, 32);
    ngS = group(nS, sgapq, sspanq, sorder, splace, nullptr, totalsS); nsPad = 16 * ngS;
    s2s_attr<<<gridOf(nsPad, TPB), TPB>>>(sorder, nsPad, dQ, dType, dLjO, sq, stype, sLJ);
    s2_boxes<<<gridOf(nsPad, TPB), TPB>>>(solNow, sshift, sorder, nsPad, geo, sAnc, sboxLo, sboxHi);
    if (!waterBoxesFresh) s2_boxes<<<gridOf(npad, TPB), TPB>>>(ancNow, shift, order, npad, geo, ancSlot, boxLo, boxHi);
    BuildArgs A{sAnc, sboxLo, sboxHi, ancSlot, boxLo, boxHi, ngS, ngW, nullptr, geo, 0, 0, (float)(RSW * RSW), nullptr, nullptr, nullptr,
                SW.pool, SW.poolCap, SW.outer, SW.outerCount, SW.blockStart, SW.unitGroup, SW.unitOrd, SW.flags};
    s2_build<1><<<gridOf(ngS, BW), 32 * BW>>>(A);
    BuildArgs B{sAnc, sboxLo, sboxHi, sAnc, sboxLo, sboxHi, ngS, ngS, nullptr, geo, 0, 0, (float)(RSS * RSS), sorder, dExOff, dExPart,
                SS.pool, SS.poolCap, SS.outer, SS.outerCount, SS.blockStart, SS.unitGroup, SS.unitOrd, SS.flags};
    s2_build<2><<<gridOf(ngS, BW), 32 * BW>>>(B); CK(cudaGetLastError());
    finishBuild(SW, ngS, "SW", t); finishBuild(SS, ngS, "SS", t);
    // The single-image rule of the cells chosen per pair of groups: reach + both half-edges < L/2.
    std::vector<float4> lo(ngS), hi(ngS), wl(ngW), wh(ngW);
    CK(cudaMemcpy(lo.data(), sboxLo, 16 * ngS, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(hi.data(), sboxHi, 16 * ngS, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(wl.data(), boxLo, 16 * ngW, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(wh.data(), boxHi, 16 * ngW, cudaMemcpyDeviceToHost));
    float hs[3] = {0, 0, 0}, hw[3] = {0, 0, 0}, es[3] = {0, 0, 0}; int live = 0, filled = 0;
    std::vector<int> so(nsPad); CK(cudaMemcpy(so.data(), sorder, 4 * nsPad, cudaMemcpyDeviceToHost));
    for (int s : so) filled += s >= 0;
    for (int g = 0; g < ngS; ++g) if (lo[g].x <= hi[g].x) { ++live; float e[3] = {hi[g].x - lo[g].x, hi[g].y - lo[g].y, hi[g].z - lo[g].z};
      for (int ax = 0; ax < 3; ++ax) { hs[ax] = std::max(hs[ax], 0.5f * e[ax]); es[ax] += e[ax]; } }
    for (int g = 0; g < ngW; ++g) if (wl[g].x <= wh[g].x) { float e[3] = {wh[g].x - wl[g].x, wh[g].y - wl[g].y, wh[g].z - wl[g].z};
      for (int ax = 0; ax < 3; ++ax) hw[ax] = std::max(hw[ax], 0.5f * e[ax]); }
    if (SW.builds <= 2)
      printf("SL build %lld (frame %d): %d solute groups (%d live, %.1f atoms a group), mean box edges %.2f %.2f %.2f A, largest half-edges "
             "solute %.2f %.2f %.2f water %.2f %.2f %.2f A; SW outer %lld, SS outer %lld entries\n", SW.builds, t, ngS, live, (double)filled / ngS,
             es[0] / live, es[1] / live, es[2] / live, hs[0], hs[1], hs[2], hw[0], hw[1], hw[2], SW.sumOuter, SS.sumOuter);
  };
  auto select = [&](List &Lst) {
    size_t ts = tmpSel; ActivePred pr{Lst.unitGroup, Lst.unitOrd, Lst.innerCount};
    cub::DeviceSelect::If(dTmp, ts, thrust::counting_iterator<int>(0), Lst.activeBlocks, Lst.nActive, Lst.nBlocks, pr);
    ++Lst.prunes;
    std::vector<int> ic(Lst.ng); CK(cudaMemcpy(ic.data(), Lst.innerCount, 4 * Lst.ng, cudaMemcpyDeviceToHost));
    long long s = 0; for (int x : ic) s += x; Lst.sumInner += s;
  };
  auto pruneWW = [&]() {
    Range r_("prune WW");
    s2_prune<<<gridOf(ngW, 4), 128>>>(ancFrame, ancFrame, ngW, geo, (float)(RWin * RWin), WW.outer, WW.outerCount, WW.blockStart, WW.inner, WW.innerCount);
    select(WW);
  };
  auto pruneSL = [&]() {
    Range r_("prune SL");
    s2_prune<<<gridOf(ngS, 4), 128>>>(spos, ancFrame, ngS, geo, (float)(RSWin * RSWin), SW.outer, SW.outerCount, SW.blockStart, SW.inner, SW.innerCount);
    s2_prune<<<gridOf(ngS, 4), 128>>>(spos, spos, ngS, geo, (float)(RSSin * RSSin), SS.outer, SS.outerCount, SS.blockStart, SS.inner, SS.innerCount);
    select(SW); select(SS);
  };
  auto pack = [&](int t) {
    Range r_("pack");
    if (single) {
      s2w_pack<double><<<gridOf(npad, TPB), TPB>>>(dF64, order, shift, npad, geo, pos, ancFrame);
      if (nS) s2s_pack<double><<<gridOf(nsPad, TPB), TPB>>>(dF64, sorder, sshift, sq, nsPad, geo, spos);
    } else {
      s2w_pack<float><<<gridOf(npad, TPB), TPB>>>(F32(t), order, shift, npad, geo, pos, ancFrame);
      if (nS) s2s_pack<float><<<gridOf(nsPad, TPB), TPB>>>(F32(t), sorder, sshift, sq, nsPad, geo, spos);
    }
  };
  const bool fuse = getenv("S2_FUSE") && atoi(getenv("S2_FUSE"));
  auto kernels = [&](bool energy) {
    ListView vw = WW.view();
    if (fuse && nS) {
      ListView vs = SW.view(), vv = SS.view();
      size_t shm = 16 * (size_t)std::max(1, ntS * ntS);
      int g = gridOf((long long)WW.nBlocks + SW.nBlocks + SS.nBlocks, 4);
      if (energy) {
        if (ns == 4) all_force<4, true><<<g, 128, shm>>>(pos, spos, sLJ, stype, dLjTab, ntS, vw, vs, vv, fW, fS, p, fke, dTab, tf.base, tf.n, dE, dC);
        else all_force<3, true><<<g, 128, shm>>>(pos, spos, sLJ, stype, dLjTab, ntS, vw, vs, vv, fW, fS, p, fke, dTab, tf.base, tf.n, dE, dC);
      } else {
        if (ns == 4) all_force<4, false><<<g, 128, shm>>>(pos, spos, sLJ, stype, dLjTab, ntS, vw, vs, vv, fW, fS, p, fke, dTab, tf.base, tf.n, nullptr, nullptr);
        else all_force<3, false><<<g, 128, shm>>>(pos, spos, sLJ, stype, dLjTab, ntS, vw, vs, vv, fW, fS, p, fke, dTab, tf.base, tf.n, nullptr, nullptr);
      }
      CK(cudaGetLastError());
      return;
    }
    if (energy) {
      if (ns == 4) ww_force<4, true><<<gridOf(WW.nBlocks, 4), 128>>>(pos, vw, fW, p, dTab, tf.base, tf.n, dE, dC);
      else ww_force<3, true><<<gridOf(WW.nBlocks, 4), 128>>>(pos, vw, fW, p, dTab, tf.base, tf.n, dE, dC);
    } else {
      if (ns == 4) ww_force<4, false><<<gridOf(WW.nBlocks, 4), 128>>>(pos, vw, fW, p, dTab, tf.base, tf.n, nullptr, nullptr);
      else ww_force<3, false><<<gridOf(WW.nBlocks, 4), 128>>>(pos, vw, fW, p, dTab, tf.base, tf.n, nullptr, nullptr);
    }
    if (nS) {
      ListView vs = SW.view(), vv = SS.view();
      size_t shm = 16 * (size_t)std::max(1, ntS * ntS);
      if (energy) {
        if (ns == 4) sw_force<4, true><<<gridOf(SW.nBlocks, 4), 128>>>(spos, sLJ, pos, vs, fW, fS, p, dTab, tf.base, tf.n, dE + 1, dC + 1);
        else sw_force<3, true><<<gridOf(SW.nBlocks, 4), 128>>>(spos, sLJ, pos, vs, fW, fS, p, dTab, tf.base, tf.n, dE + 1, dC + 1);
        ss_force<true><<<gridOf(SS.nBlocks, 4), 128, shm>>>(spos, stype, dLjTab, ntS, vv, fS, p, fke, dTab, tf.base, tf.n, dE + 2, dC + 2);
      } else {
        if (ns == 4) sw_force<4, false><<<gridOf(SW.nBlocks, 4), 128>>>(spos, sLJ, pos, vs, fW, fS, p, dTab, tf.base, tf.n, nullptr, nullptr);
        else sw_force<3, false><<<gridOf(SW.nBlocks, 4), 128>>>(spos, sLJ, pos, vs, fW, fS, p, dTab, tf.base, tf.n, nullptr, nullptr);
        ss_force<false><<<gridOf(SS.nBlocks, 4), 128, shm>>>(spos, stype, dLjTab, ntS, vv, fS, p, fke, dTab, tf.base, tf.n, nullptr, nullptr);
      }
    }
    CK(cudaGetLastError());
  };
  auto unpack = [&]() {
    Range r_("unpack");
    s2w_unpack<<<gridOf(9LL * npad, TPB), TPB>>>(fW, order, npad, geo, fatom);
    if (nS) s2s_unpack<<<gridOf(3LL * nsPad, TPB), TPB>>>(fS, sorder, nsPad, fatom);
  };
  // The checks: the lists in use at a frame: the energy kernels' energies and counts, the forces,
  // and the pairs of each list (input numbering) for the CPU's brute force.
  int nchecks = 0;
  const bool noCheck = getenv("S2_NOCHECK") && atoi(getenv("S2_NOCHECK"));
  auto check = [&](int t, const char *why) {
    if (noCheck) return;
    Range r_("check");
    pack(t);
    CK(cudaMemset(dE, 0, 24)); CK(cudaMemset(dC, 0, 3 * sizeof(Counters))); CK(cudaMemset(fatom, 0, 12 * (size_t)natom));
    kernels(true); unpack(); CK(cudaDeviceSynchronize());
    double e[3]; Counters c[3]; CK(cudaMemcpy(e, dE, 24, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(c, dC, sizeof c, cudaMemcpyDeviceToHost));
    std::vector<float> fa((size_t)natom * 3); CK(cudaMemcpy(fa.data(), fatom, fa.size() * 4, cudaMemcpyDeviceToHost));
    char name[512]; snprintf(name, sizeof name, "%s.check%02d.forces", out.c_str(), nchecks);
    { FILE *f = fopen(name, "wb"); fwrite(fa.data(), 4, fa.size(), f); fclose(f); }
    std::vector<int> ord(npad), sord(nsPad); CK(cudaMemcpy(ord.data(), order, 4 * npad, cudaMemcpyDeviceToHost));
    if (nS) CK(cudaMemcpy(sord.data(), sorder, 4 * nsPad, cudaMemcpyDeviceToHost));
    auto dump = [&](const List &Lst, const std::vector<int> &oi, const std::vector<int> &oj, int ng) {
      std::vector<int> ic(ng), bs(ng); CK(cudaMemcpy(ic.data(), Lst.innerCount, 4 * ng, cudaMemcpyDeviceToHost));
      CK(cudaMemcpy(bs.data(), Lst.blockStart, 4 * ng, cudaMemcpyDeviceToHost));
      std::vector<int2> en((size_t)Lst.nBlocks * 64); CK(cudaMemcpy(en.data(), Lst.inner, en.size() * 8, cudaMemcpyDeviceToHost));
      std::vector<int32_t> pr;
      for (int g = 0; g < ng; ++g)
        for (int k = 0; k < ic[g]; ++k) {
          int2 x = en[(size_t)bs[g] * 64 + k]; unsigned msk = (unsigned)x.y & 0xFFFFu;
          for (int q = 0; q < 16; ++q) if (msk >> q & 1) {
            int a = oi[16 * g + q], b = oj[x.x];
            if (a < 0 || b < 0) { fprintf(stderr, "check: an empty slot in the list\n"); exit(4); }
            pr.push_back(a); pr.push_back(b);
          }
        }
      return pr;
    };
    std::vector<int32_t> pw = dump(WW, ord, ord, ngW), psw, pss;
    if (nS) { psw = dump(SW, sord, ord, ngS); pss = dump(SS, sord, sord, ngS); }
    snprintf(name, sizeof name, "%s.check%02d.pairs", out.c_str(), nchecks);
    { FILE *f = fopen(name, "wb"); int32_t h[3] = {(int32_t)pw.size() / 2, (int32_t)psw.size() / 2, (int32_t)pss.size() / 2}; fwrite(h, 4, 3, f);
      fwrite(pw.data(), 4, pw.size(), f); fwrite(psw.data(), 4, psw.size(), f); fwrite(pss.data(), 4, pss.size(), f); fclose(f); }
    printf("check %d (%s), frame %d: energy WW %.6f SW %.6f SS %.6f kJ/mol; site pairs within rc WW %llu SW %llu SS %llu; list pairs WW %zu SW %zu SS %zu; "
           "lane-steps WW %llu SW %llu SS %llu\n", nchecks, why, t, e[0] * 4.184, e[1] * 4.184, e[2] * 4.184, c[0].sitePairsIn, c[1].sitePairsIn, c[2].sitePairsIn,
           pw.size() / 2, psw.size() / 2, pss.size() / 2, c[0].laneSteps, c[1].laneSteps, c[2].laneSteps);
    fprintf(fo, "check %d %s %d %.9f %.9f %.9f %llu %llu %llu %zu %zu %zu %llu %llu %llu\n", nchecks, why, t, e[0] * 4.184, e[1] * 4.184, e[2] * 4.184,
            c[0].sitePairsIn, c[1].sitePairsIn, c[2].sitePairsIn, pw.size() / 2, psw.size() / 2, pss.size() / 2, c[0].laneSteps, c[1].laneSteps, c[2].laneSteps);
    ++nchecks;
  };
  int stale[2][2] = {{0, 0}, {0, 0}}, fresh[2] = {0, 0};   // [WW, SL][before build, before prune]
  std::vector<int> bW, pW, bL, pL;
  CK(cudaMemset(WW.flags, 0, 4)); if (nS) { CK(cudaMemset(SW.flags, 0, 4)); CK(cudaMemset(SS.flags, 0, 4)); }
  int remapFlags = 0; int *dRemap = dalloc<int>(1);
  for (int t = 0; t < nframes; ++t) {
    { Range r_("test");
    CK(cudaMemsetAsync(maxBits, 0, 24));
    if (single) s2_test<double><<<gridOf(nW + nS, TPB), TPB>>>(dF64, geo, nW, refWB, refWP, refLB, refLP, refSB, refSP, ancNow, solNow, maxBits);
    else s2_test<float><<<gridOf(nW + nS, TPB), TPB>>>(F32(t), geo, nW, refWB, refWP, refLB, refLP, refSB, refSP, ancNow, solNow, maxBits);
    }
    bool wb = t == 0, wp = false, lb = t == 0 && nS, lp = false;
    if (t > 0) {
      unsigned mb[6]; CK(cudaMemcpy(mb, maxBits, 24, cudaMemcpyDeviceToHost));
      float v[6]; memcpy(v, mb, 24);
      if (mode == 2) {
        wb = !(v[0] <= limB2); wp = !(v[1] <= limP2);
        if (nS) { lb = !(std::max(v[2], v[4]) <= limB2); lp = !(std::max(v[3], v[5]) <= limP2); }
      } else {
        bool b = !(std::max(v[0], v[4]) <= limB2), q = !(std::max(v[1], v[5]) <= limP2);
        wb = b; wp = q; if (nS) { lb = b; lp = q; }
      }
    }
    if (t > 0 && !single) {
      // The lists in use until now, on the last frame they were valid for.
      if (wb && stale[0][0] < 2) { check(t - 1, "WW-stale-before-build"); ++stale[0][0]; }
      else if (wp && !wb && stale[0][1] < 2) { check(t - 1, "WW-stale-before-prune"); ++stale[0][1]; }
      else if (lb && stale[1][0] < 2) { check(t - 1, "SL-stale-before-build"); ++stale[1][0]; }
      else if (lp && !lb && stale[1][1] < 2) { check(t - 1, "SL-stale-before-prune"); ++stale[1][1]; }
    }
    if (wb) {
      if (nS && !lb && t > 0) { CK(cudaMemcpyAsync(orderOld, order, 4 * (size_t)npadCap, cudaMemcpyDeviceToDevice)); CK(cudaMemcpyAsync(shiftOld, shift, 16 * (size_t)nW, cudaMemcpyDeviceToDevice)); }
      buildWW(t); bW.push_back(t);
      if (nS && !lb && t > 0) {
        Range r_("remap");
        s2_remap<<<ngS, 128>>>(SW.outer, SW.outerCount, SW.blockStart, ngS, orderOld, shiftOld, placeOf, shift, dRemap);
        s2_remap<<<ngS, 128>>>(SW.inner, SW.innerCount, SW.blockStart, ngS, orderOld, shiftOld, placeOf, shift, dRemap);
      }
    }
    if (lb) { buildSL(t, wb); bL.push_back(t); }
    pack(t);
    int fl = (wb ? 3 : 0) | (wp ? 2 : 0) | (lb ? 12 : 0) | (lp ? 8 : 0);
    if (fl) { Range r_("setref"); s2_setref<<<gridOf(nW + nS, TPB), TPB>>>(ancNow, solNow, nW, nS, fl, refWB, refWP, refLB, refLP, refSB, refSP); }
    if (wb || wp) { pruneWW(); pW.push_back(t); }
    if (lb || lp) { pruneSL(); pL.push_back(t); }
    if (single) { CK(cudaDeviceSynchronize()); check(t, "single-frame"); continue; }
    kernels(false); unpack(); CK(cudaGetLastError());
    if (t > 0 && wb && fresh[0] < 1) { CK(cudaDeviceSynchronize()); check(t, "WW-fresh-after-build"); ++fresh[0]; }
    else if (t > 0 && lb && fresh[1] < 1) { CK(cudaDeviceSynchronize()); check(t, "SL-fresh-after-build"); ++fresh[1]; }
  }
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(&remapFlags, dRemap, 4, cudaMemcpyDeviceToHost));
  if (remapFlags) { fprintf(stderr, "remap flags %d\n", remapFlags); return 5; }
  int T = nframes - 1;
  auto rep = [&](const char *nm, const List &Lst, const std::vector<int> &b, const std::vector<int> &q) {
    printf("%s: built %zu times incl. frame 0 (every %.2f steps after it), pruned %zu incl. builds (every %.2f); outer entries %.1f a build, inner %.1f a pruning\n",
           nm, b.size(), (double)T / std::max<size_t>(1, b.size() - 1), q.size(), (double)T / std::max<size_t>(1, q.size() - 1),
           (double)Lst.sumOuter / std::max(1LL, Lst.builds), (double)Lst.sumInner / std::max(1LL, Lst.prunes));
    fprintf(fo, "cadence %s %d %zu %zu %.3f %.3f %d\n", nm, T, b.size(), q.size(), (double)Lst.sumOuter / std::max(1LL, Lst.builds),
            (double)Lst.sumInner / std::max(1LL, Lst.prunes), Lst.nBlocks);
  };
  rep("WW", WW, bW, pW); if (nS) { rep("SW", SW, bL, pL); rep("SS", SS, bL, pL); }
  fprintf(fo, "buildsW"); for (int s : bW) fprintf(fo, " %d", s); fprintf(fo, "\nprunesW"); for (int s : pW) fprintf(fo, " %d", s);
  fprintf(fo, "\nbuildsL"); for (int s : bL) fprintf(fo, " %d", s); fprintf(fo, "\nprunesL"); for (int s : pL) fprintf(fo, " %d", s); fprintf(fo, "\n");
  fclose(fo);
  return 0;
}
