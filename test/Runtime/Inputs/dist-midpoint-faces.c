#include "mdrt_midpoint.h"
#include <assert.h>
#include <float.h>

/* Check both adjacent binary64 values at each rounded entry and exit,
   including negative images and cells with no representable numerators. */
static void check(double x, double edge, int image, int grid, int cell,
                  struct MidpointRange range) {
  int reference = compareMidpoint(x/edge,image,grid,cell);
  assert((x >= range.lower) == (reference >= 0));
  assert((x >= range.upper) == (reference > 0));
}

/* Compare each definite classification with the actual midpoint over
   the corners and center of a box at neighboring representable faces. */
static void checkEnvelope(double scale, double radius, double shift) {
  struct MidpointBounds i = {0}, j = {0};
  double cx = 0.75*scale, cy = -0.5*scale;
  i.low[0] = cx-radius; i.high[0] = cx+radius;
  j.low[0] = cy-radius; j.high[0] = cy+radius;
  j.shift[0] = shift;
  double bound = fmax(1.0,fmax(fabs(shift),fmax(fabs(cx)+radius,fabs(cy)+radius)));
  double padding = midpointPadding(bound);
  double xs[] = {i.low[0],cx,i.high[0]}, ys[] = {j.low[0],cy,j.high[0]};
  double center = cx - 0.5*((cx-cy)-shift);
  double faces[] = {center, nextafter(center,-INFINITY), nextafter(center,INFINITY),
                    center-4*radius, center+4*radius};
  for (unsigned f = 0; f < sizeof faces/sizeof *faces; ++f) {
    for (int upper = 0; upper <= 1; ++upper) {
      struct MidpointRange ranges[3] = {
        {-INFINITY,INFINITY},{-INFINITY,INFINITY},{-INFINITY,INFINITY}};
      if (upper) ranges[0].upper = faces[f];
      else ranges[0].lower = faces[f];
      int status = classifyMidpoint(&i,&j,ranges,padding);
      for (unsigned a = 0; a < 3; ++a)
        for (unsigned b = 0; b < 3; ++b) {
          double midpoint = xs[a] - 0.5*((xs[a]-ys[b])-shift);
          int inside = midpoint >= ranges[0].lower && midpoint < ranges[0].upper;
          assert(status == 0 || (status > 0) == inside);
        }
    }
  }
}

int main(void) {
  double edges[] = {DBL_MIN, 0.1, 1.0, 19.750000000000004, DBL_MAX};
  int grids[] = {1,2,3,7,19};
  for (unsigned e = 0; e < sizeof edges/sizeof *edges; ++e)
    for (unsigned g = 0; g < sizeof grids/sizeof *grids; ++g)
      for (int cell = 0; cell < grids[g]; ++cell)
        for (int image = -3; image <= 3; ++image) {
          double edge = edges[e];
          struct MidpointRange r = midpointRange(edge,image,grids[g],cell);
          double probes[] = {r.lower, r.upper, -0.0, 0.0, edge,
                             edge*((double)cell/grids[g]-image)};
          for (unsigned p = 0; p < sizeof probes/sizeof *probes; ++p) {
            double x = probes[p];
            /* An infinite upper endpoint can represent an empty cell
               beyond all finite coordinates. The contract is finite x. */
            if (isfinite(x)) check(x,edge,image,grids[g],cell,r);
            double low = nextafter(x,-INFINITY), high = nextafter(x,INFINITY);
            if (isfinite(low)) check(low,edge,image,grids[g],cell,r);
            if (isfinite(high)) check(high,edge,image,grids[g],cell,r);
          }
        }
  double scales[] = {DBL_MIN, 1.0, 123.5, 0x1p500, DBL_MAX/8};
  for (unsigned i = 0; i < sizeof scales/sizeof *scales; ++i)
    for (int exponent = 3; exponent <= 60; exponent += 3)
      checkEnvelope(scales[i],ldexp(scales[i],-exponent),-0.25*scales[i]);
  return 0;
}
