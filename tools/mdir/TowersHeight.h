// Untimed rectangular geometry selection for the mixed tower traversal.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace mdir::tool {
inline int selectTowersReach(uint64_t atoms, const double box[3],
                             double cutoff) {
  // Compare the two grids at the same nominal 4i launch (two blocks on
  // each of 128 SMs). Only geometry-dependent terms of the log fit remain.
  // At uniform density source and target lengths scale as (2k+1)n/C and
  // 5n/C. At fixed cutoff, accepted pairs and occupied records per tower
  // both scale as 1/C (constant mean survivors per record).
  auto cost = [&](int k) {
    double cells = 1;
    for (int a = 0; a < 3; ++a) {
      double bins = std::floor(box[a] * (a == 2 ? k : 1) / cutoff);
      if (!(bins >= (a == 2 ? 2 * k + 1 : 3) && bins < INT32_MAX / 4))
        return std::numeric_limits<double>::infinity();
      cells *= bins;
    }
    double mean = atoms / cells;
    double active = cells * -std::expm1(-5 * mean);
    double wave = std::ceil(active / 256) * 256 / active;
    return (1 + 0.00564036 - 0.02597647) * std::log(active) +
           0.13572139 * std::log((2 * k + 1) * mean) +
           0.13627289 * std::log(5 * mean) +
           (0.44278671 + 0.39735107) * std::log(mean) +
           0.60165672 * std::log(wave) -
           0.02357148 * std::max(5 * mean / 384 - 1, 0.0);
  };
  return cost(2) < cost(1) ? 2 : 1;
}
} // namespace mdir::tool
