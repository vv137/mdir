#ifndef MDRT_MIDPOINT_H
#define MDRT_MIDPOINT_H

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

/* Classify the fractional midpoint relative to the cell of this explicit
   image, using the same primary-cell wrap and rounded division as selection. */
static int compareMidpoint(double u, int64_t image, int64_t grid, int64_t cell) {
  double whole = floor(u), target = -(double)image;
  if (whole != target) return whole < target ? -1 : 1;
  int64_t domain = division(u-whole, grid);
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
                           int64_t cell, int past) {
  uint64_t lo = UINT64_C(0x000fffffffffffff); /* -infinity */
  uint64_t hi = UINT64_C(0xfff0000000000000); /* +infinity */
  while (hi-lo > 1) {
    uint64_t mid = lo+(hi-lo)/2;
    int side = compareMidpoint(orderedCoordinate(mid)/length,image,grid,cell);
    if (side >= past) hi = mid;
    else lo = mid;
  }
  return orderedCoordinate(hi);
}

static struct MidpointRange midpointRange(double edge, int64_t image,
                                         int64_t grid, int64_t cell) {
  return (struct MidpointRange){midpointFace(edge,image,grid,cell,0),
                                midpointFace(edge,image,grid,cell,1)};
}

#endif
