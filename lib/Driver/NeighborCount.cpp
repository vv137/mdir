//===- NeighborCount.cpp - The most neighbors of a particle ---------------===//
//
// A search with cells as wide as the reach (cell lists,
// [AllenTildesley2017]), which visits the candidates of a cell once for all its
// members, from positions gathered in the order of the cells, and spreads
// the cells over threads (D316).
//
// Its result is that of the serial search over particles that it replaces,
// to the integer: the same cells (the same wrapping of a position and the
// same rounding of its cell), the same distinct cells among the 27 around
// a cell, where fewer than 3 cells along an axis make some coincide, and
// the same test of a pair, the minimum image of the difference of the
// positions as they are given, whatever cell they are in. The count of a
// particle is a sum of integers and the result their maximum, so neither
// depends on the order of the candidates or of the cells.
//
//===----------------------------------------------------------------------===//

#include "mdir/Driver/NeighborCount.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Threading.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <thread>
#include <vector>

using namespace mdir;

namespace {

/// The cells, and the positions gathered in their order.
struct Cells {
  int counts[3];
  /// Cell c holds the places offsets[c] to offsets[c + 1] of x, y, z.
  std::vector<size_t> offsets;
  std::vector<double> x, y, z;
};

/// The threads of the search: those that the process may run on (its
/// affinity), bounded by `given` or else by MDIR_COMPILE_THREADS, which
/// bounds the threads of a process among those that run side by side, as
/// the tests of a suite do.
unsigned getThreads(unsigned given) {
  if (given != 0)
    return given;
  unsigned threads = llvm::hardware_concurrency().compute_thread_count();
  if (const char *bound = std::getenv("MDIR_COMPILE_THREADS")) {
    unsigned value = 0;
    if (!llvm::StringRef(bound).getAsInteger(10, value) && value != 0)
      threads = std::min(threads, value);
  }
  return std::max(threads, 1u);
}

/// std::round (halfway cases away from zero) in operations that a vector
/// unit has, without branches: below 2^52 in magnitude, x - trunc(x) is
/// the fraction of x exactly (the two are within a factor of 2, or
/// trunc(x) is a zero of the sign of x), and the step to the next integer
/// is exact; from 2^52 x is an integer and its fraction 0; an infinity or
/// not a number is returned as it is, since its comparisons are false. It
/// differs from std::round only in the sign of a zero (+0 for x in
/// (-0.5, -0]), which a pair cannot see: d - L t is d itself for d other
/// than zero whatever the sign of t, and its square is +0 for d = ±0.
inline double roundHalfAway(double x) {
  double t = std::trunc(x);
  double fraction = x - t;
  double step = static_cast<double>(fraction >= 0.5) -
                static_cast<double>(fraction <= -0.5);
  return t + step;
}

// The sweep is compiled also for AVX2 and chosen at run time where the
// processor has it. A vector of the same IEEE operations gives the same
// bits as one at a time; AVX2 does not bring FMA, so no product and sum
// are fused, as in the default build.
#if defined(__x86_64__) && defined(__GNUC__) && !defined(__AVX2__) &&         \
    !defined(__FMA__)
#define MDIR_SWEEP_CLONES __attribute__((target_clones("avx2", "default")))
#else
#define MDIR_SWEEP_CLONES
#endif

/// The particles q from `begin` to `end` within the reach of (xi, yi, zi):
/// the test of a pair of the search it replaces, in the same operations,
/// the difference of the positions as given, its minimum image along each
/// axis in turn, and the sum of the squares in the order x, y, z.
MDIR_SWEEP_CLONES
int64_t countWithin(const double *xs, const double *ys, const double *zs,
                    size_t begin, size_t end, double xi, double yi, double zi,
                    double lx, double ly, double lz, double reach2) {
  int64_t found = 0;
  for (size_t q = begin; q != end; ++q) {
    double r2 = 0.0;
    double d = xi - xs[q];
    d -= lx * roundHalfAway(d / lx);
    r2 += d * d;
    d = yi - ys[q];
    d -= ly * roundHalfAway(d / ly);
    r2 += d * d;
    d = zi - zs[q];
    d -= lz * roundHalfAway(d / lz);
    r2 += d * d;
    found += r2 < reach2;
  }
  return found;
}

} // namespace

int64_t driver::countMostNeighbors(llvm::ArrayRef<double> positions,
                                   const double box[3], double reach,
                                   unsigned threads) {
  size_t count = positions.size() / 3;
  if (count == 0)
    return 0;
  Cells cells;
  for (int k = 0; k != 3; ++k)
    cells.counts[k] = std::max(1, static_cast<int>(box[k] / reach));
  const int *counts = cells.counts;
  // The cell of a particle from its position wrapped into the box.
  auto wrap = [&](double x, int k) {
    double length = box[k];
    return x - length * std::floor(x / length);
  };
  auto cellOf = [&](size_t i, int k) {
    int c = static_cast<int>(wrap(positions[3 * i + k], k) / box[k] *
                             counts[k]);
    return std::min(c, counts[k] - 1);
  };
  size_t numCells = static_cast<size_t>(counts[0]) * counts[1] * counts[2];
  std::vector<size_t> cellOfParticle(count);
  cells.offsets.assign(numCells + 1, 0);
  for (size_t i = 0; i != count; ++i) {
    size_t cell =
        (static_cast<size_t>(cellOf(i, 0)) * counts[1] + cellOf(i, 1)) *
            counts[2] +
        cellOf(i, 2);
    cellOfParticle[i] = cell;
    ++cells.offsets[cell + 1];
  }
  for (size_t c = 0; c != numCells; ++c)
    cells.offsets[c + 1] += cells.offsets[c];
  // A counting sort of the positions as given (not wrapped), so that the
  // members of a cell are contiguous.
  cells.x.resize(count);
  cells.y.resize(count);
  cells.z.resize(count);
  {
    std::vector<size_t> cursor(cells.offsets.begin(), cells.offsets.end() - 1);
    for (size_t i = 0; i != count; ++i) {
      size_t place = cursor[cellOfParticle[i]]++;
      cells.x[place] = positions[3 * i];
      cells.y[place] = positions[3 * i + 1];
      cells.z[place] = positions[3 * i + 2];
    }
  }
  cellOfParticle = {};

  const double lx = box[0], ly = box[1], lz = box[2];
  const double reach2 = reach * reach;
  const double *xs = cells.x.data(), *ys = cells.y.data(),
               *zs = cells.z.data();
  const size_t *offsets = cells.offsets.data();
  // The most neighbors of the members of the cells from `first` to `last`.
  auto search = [&](size_t first, size_t last) -> int64_t {
    int64_t most = 0;
    size_t ranges[27][2];
    for (size_t cell = first; cell != last; ++cell) {
      size_t begin = offsets[cell], end = offsets[cell + 1];
      if (begin == end)
        continue;
      int home[3] = {static_cast<int>(cell / counts[2] / counts[1]),
                     static_cast<int>(cell / counts[2] % counts[1]),
                     static_cast<int>(cell % counts[2])};
      // Each cell once, where the cells on the two sides coincide.
      size_t seen[27];
      int numRanges = 0;
      for (int dx = -1; dx <= 1; ++dx)
        for (int dy = -1; dy <= 1; ++dy)
          for (int dz = -1; dz <= 1; ++dz) {
            int c[3] = {home[0] + dx, home[1] + dy, home[2] + dz};
            for (int k = 0; k != 3; ++k)
              c[k] = (c[k] % counts[k] + counts[k]) % counts[k];
            size_t near =
                (static_cast<size_t>(c[0]) * counts[1] + c[1]) * counts[2] +
                c[2];
            if (std::find(seen, seen + numRanges, near) != seen + numRanges)
              continue;
            seen[numRanges] = near;
            ranges[numRanges][0] = offsets[near];
            ranges[numRanges][1] = offsets[near + 1];
            ++numRanges;
          }
      for (size_t p = begin; p != end; ++p) {
        double xi = xs[p], yi = ys[p], zi = zs[p];
        int64_t found = 0;
        for (int r = 0; r != numRanges; ++r)
          found += countWithin(xs, ys, zs, ranges[r][0], ranges[r][1], xi, yi,
                               zi, lx, ly, lz, reach2);
        // The particle itself is among the candidates of its own cell, once;
        // the search it replaces skipped it, so its own test is taken back.
        found -= countWithin(xs, ys, zs, p, p + 1, xi, yi, zi, lx, ly, lz,
                             reach2);
        most = std::max(most, found);
      }
    }
    return most;
  };

  // Small systems in one thread; the result is the same.
  unsigned numThreads = getThreads(threads);
  if (count < 16384 || numCells < 2)
    numThreads = 1;
  numThreads = static_cast<unsigned>(
      std::min<size_t>(numThreads, numCells));
  if (numThreads == 1)
    return search(0, numCells);

  // Chunks of cells taken in turn, so that dense and empty regions spread.
  size_t chunk = std::max<size_t>(1, numCells / (64 * numThreads));
  std::atomic<size_t> next{0};
  std::vector<int64_t> most(numThreads, 0);
  std::vector<std::thread> workers;
  workers.reserve(numThreads);
  for (unsigned t = 0; t != numThreads; ++t)
    workers.emplace_back([&, t] {
      int64_t local = 0;
      for (;;) {
        size_t first = next.fetch_add(chunk);
        if (first >= numCells)
          break;
        local = std::max(local, search(first, std::min(first + chunk, numCells)));
      }
      most[t] = local;
    });
  for (std::thread &worker : workers)
    worker.join();
  return *std::max_element(most.begin(), most.end());
}
