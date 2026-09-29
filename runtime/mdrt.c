/* The MDIR runtime: functions that compiled code calls. */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/* Called when a particle has more neighbors than a row of the neighbor
   matrix holds. The run cannot continue: pairs would be missed. */
void mdrtReportNeighborOverflow(int64_t needed, int64_t width) {
  fprintf(stderr,
          "mdrt: a particle has %lld neighbors, but the neighbor structure "
          "holds %lld per particle\n",
          (long long)needed, (long long)width);
  fflush(stderr);
  abort();
}
