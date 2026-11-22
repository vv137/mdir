// Untimed rectangular geometry selection for the mixed tower traversal.
#pragma once

#include <cmath>
#include <cstdint>

namespace mdir::tool {
inline int selectTowersReach(const double box[3], double cutoff) {
  const double nx = std::floor(box[0] / cutoff);
  const double ny = std::floor(box[1] / cutoff);
  const double nz = std::floor(box[2] / cutoff);
  const double halfZ = std::floor(2 * box[2] / cutoff);
  if (!(halfZ >= 5 && halfZ < INT32_MAX / 4))
    return 1;
  // A box between 2.5 and 3 cutoffs tall admits only HALF.
  if (nz < 3)
    return 2;
  // FULL-grid crossover fitted to the 4n rebuilt-force sweep. The nearest
  // observations on either side had 343 and 1530 towers. Use their geometric
  // midpoint (724.4), so neither side is favored on the log scale.
  return nx * ny * nz <= 724 ? 2 : 1;
}
} // namespace mdir::tool
