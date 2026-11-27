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
  return 0;
}
