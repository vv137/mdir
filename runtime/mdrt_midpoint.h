#ifndef MDRT_MIDPOINT_H
#define MDRT_MIDPOINT_H

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

static int64_t division(double u, int64_t g) {
  int64_t a = (int64_t)floor(u * (double)g);
  if (a < 0)
    a = 0;
  if (a >= g)
    a = g - 1;
  return a;
}

/* Nonuniform cuts share the same primary-cell convention as division. */
static int64_t cutDivision(double u, int64_t grid, const double *cuts) {
  if (!cuts) return division(u,grid);
  int64_t lo = 0, hi = grid;
  while (hi-lo > 1) {
    int64_t mid = lo+(hi-lo)/2;
    if (u < cuts[mid]) hi = mid;
    else lo = mid;
  }
  return lo;
}

/* Classify the fractional midpoint relative to the cell of this explicit
   image, using the same primary-cell wrap and rounded division as selection. */
static int compareMidpoint(double u, int64_t image, int64_t grid, int64_t cell,
                           const double *cuts) {
  double whole = floor(u), target = -(double)image;
  if (whole != target) return whole < target ? -1 : 1;
  int64_t domain = cutDivision(u-whole,grid,cuts);
  return (domain > cell) - (domain < cell);
}

/* A domain face need not round to its geometric position. Bisect the
   ordered binary64 coordinates to find the first numerator whose rounded
   division enters (or passes) the cell. Both infinities bound the search.
   Keep the two signed zeroes adjacent in this ordering. */
struct MidpointRange { double lower, upper; };

static double orderedCoordinate(uint64_t key) {
  const uint64_t sign = UINT64_C(1) << 63;
  uint64_t bits = key & sign ? key ^ sign : ~key;
  double x;
  memcpy(&x,&bits,sizeof x);
  return x;
}

static double midpointFace(double length, int64_t image, int64_t grid,
                           int64_t cell, int past, const double *cuts) {
  uint64_t lo = UINT64_C(0x000fffffffffffff); /* -infinity */
  uint64_t hi = UINT64_C(0xfff0000000000000); /* +infinity */
  while (hi-lo > 1) {
    uint64_t mid = lo+(hi-lo)/2;
    int side = compareMidpoint(orderedCoordinate(mid)/length,image,grid,cell,cuts);
    if (side >= past) hi = mid;
    else lo = mid;
  }
  return orderedCoordinate(hi);
}

static struct MidpointRange midpointCutRange(double edge, int64_t image,
                                            int64_t grid, int64_t cell, const double *cuts) {
  return (struct MidpointRange){midpointFace(edge,image,grid,cell,0,cuts),
                                midpointFace(edge,image,grid,cell,1,cuts)};
}

struct MidpointBounds {
  double low[3], high[3], shift[3];
  int64_t image[3];
};

static double midpointPadding(double coordinateBound) {
  return coordinateBound <= DBL_MAX/16 ? 32*DBL_EPSILON*coordinateBound : INFINITY;
}

/* The exact affine midpoint is (xi + xj + (b-a))/2. Bound it
   at the coordinate endpoints and allow for rounding in both this
   expression and xi - .5*((xi-xj)-(b-a)), used by the pair test. */
static int classifyMidpoint(const struct MidpointBounds *i, const struct MidpointBounds *j,
                            const struct MidpointRange ranges[3], double padding) {
  int stable = 1;
  for (int c = 0; c < 3; ++c) {
    double shift = j->shift[c]-i->shift[c];
    double low = 0.5*((i->low[c]+j->low[c])+shift)-padding;
    double high = 0.5*((i->high[c]+j->high[c])+shift)+padding;
    if (!isfinite(low) || !isfinite(high)) return 0;
    if (low >= ranges[c].upper || high < ranges[c].lower) return -1;
    stable &= low >= ranges[c].lower && high < ranges[c].upper;
  }
  return stable;
}

#endif
