// System-only height selection for the mixed tower traversal.
#pragma once

#include <cmath>
#include <cstdint>
#include <string_view>
#include <vector>

namespace mdir::tool {
struct TowersCellChoice {
  int reach, split;
};
inline std::vector<TowersCellChoice> towersCells(const double heights[3],
                                                 double cutoff,
                                                 std::string_view height,
                                                 int pinnedSplit) {
  std::vector<TowersCellChoice> cells;
  for (int reach : {1, 2}) {
    if (!height.empty() && height != (reach == 1 ? "FULL" : "HALF"))
      continue;
    bool feasible = true;
    for (int a = 0; a < 3; ++a) {
      int radius = a == 2 ? reach : 1;
      double bins = std::floor(heights[a] * radius / cutoff);
      feasible &= bins >= 2 * radius + 1 && bins < INT32_MAX / 4;
    }
    if (feasible)
      for (int split : {1, 2})
        if (!pinnedSplit || split == pinnedSplit)
          cells.push_back({reach, split});
  }
  return cells;
}
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
