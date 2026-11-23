// System-only height selection for the mixed tower traversal.
#pragma once

#include <cmath>
#include <cstdint>

namespace mdir::tool {
inline int selectTowersReach(const double heights[3], double cutoff,
                             int64_t atoms) {
  const double nx = std::floor(heights[0] / cutoff);
  const double ny = std::floor(heights[1] / cutoff);
  const double nz = std::floor(heights[2] / cutoff);
  const double halfZ = std::floor(2 * heights[2] / cutoff);
  if (!(halfZ >= 5 && halfZ < INT32_MAX / 4))
    return 1;
  // A box between 2.5 and 3 cutoffs tall admits only HALF.
  if (nz < 3)
    return 2;
  // More independent towers favor FULL. More pairs per tower favor HALF.
  // The mean population squared estimates pair work on the FULL grid.
  // The 4n rectangular sweeps bracket towers/population² at 0.073 and
  // 0.433. Their arithmetic midpoint rounds to one quarter.
  const double towers = nx * ny * nz;
  if (towers <= 0)
    return 1;
  const double population = atoms / towers;
  return towers <= 0.25 * population * population ? 2 : 1;
}
} // namespace mdir::tool
