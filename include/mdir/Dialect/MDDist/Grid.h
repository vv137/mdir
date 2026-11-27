#ifndef MDIR_DIALECT_MDDIST_GRID_H
#define MDIR_DIALECT_MDDIST_GRID_H

// The grid of domains that a count of domains chooses for a cell
// (D447).

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <cstdint>
#include <limits>

namespace mdir::md_dist {
/// Choose the factor grid with the smallest surface area of an
/// orthorhombic domain box. For fixed N and volume, the varying factor is
/// gx/Lx + gy/Ly + gz/Lz. The order of iteration breaks equal-score ties
/// lexicographically. This geometric model is a compile-time choice, not a
/// measurement of transport cost (Plimpton 1995; Saunders et al. 2018).
inline llvm::SmallVector<int64_t, 3>
selectGrid(int64_t count, llvm::ArrayRef<double> edges,
           llvm::function_ref<void(int64_t, int64_t, int64_t, long double)>
               visit = {}) {
  llvm::SmallVector<int64_t> factors;
  for (int64_t candidate = 1; candidate <= count / candidate; ++candidate)
    if (count % candidate == 0) {
      factors.push_back(candidate);
      if (candidate != count / candidate)
        factors.push_back(count / candidate);
    }
  llvm::sort(factors);
  llvm::SmallVector<int64_t, 3> best;
  long double bestScore = std::numeric_limits<long double>::infinity();
  for (int64_t gx : factors) {
    int64_t rest = count / gx;
    for (int64_t gy : factors) {
      if (rest % gy)
        continue;
      int64_t gz = rest / gy;
      long double score = static_cast<long double>(gx) / edges[0] +
                          static_cast<long double>(gy) / edges[1] +
                          static_cast<long double>(gz) / edges[2];
      if (visit)
        visit(gx, gy, gz, score);
      // A permutation of an exactly tied grid can round its three adds in
      // another order. Keep the lexicographically first one within that
      // rounding, as the score is only a geometric estimate.
      long double tie = 8 * std::numeric_limits<long double>::epsilon() *
                        std::max(score, bestScore);
      if (best.empty() || score < bestScore - tie) {
        bestScore = score;
        best = {gx, gy, gz};
      }
    }
  }
  return best;
}

} // namespace mdir::md_dist
#endif
