// Launch geometry shared by the moving dispatch and its setup trial.
#pragma once
#include <cstdint>

namespace mdir::kernels {
inline int towersExclusionRows(int exclusions, int workers) {
  // Smaller lists need the parallelism of one thread per pair (D431).
  return exclusions >= 128 * 4 * workers ? 4 : 1;
}

inline int64_t towersForceBlocks(int cells, int split, int exclusions,
                                 int workers) {
  int64_t width = workers * towersExclusionRows(exclusions, workers);
  return int64_t(cells) * split + (exclusions + width - 1) / width;
}
inline int64_t towersFinishBlocks(int64_t atoms, int workers) {
  return (atoms + workers - 1) / workers;
}
} // namespace mdir::kernels
