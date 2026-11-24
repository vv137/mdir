// Perpendicular heights of rectangular and triclinic frames, in nm.
#include "TowersHeight.h"
#include "TowersTrialRecord.h"
#include "mdir/Conversion/TowersLaunch.h"
#include "mdir/Driver/Checkpoint.h"
#include "llvm/Support/Error.h"
#include <algorithm>
#include <cstdio>

// Two otherwise identical writer states isolate whether the new traversal
// consumes a force computed under a different tower geometry.
static int writeCheckpointVariants(const char *input, const std::string &dir) {
  llvm::ExitOnError take;
  auto state = take(mdir::driver::readCheckpoint(input));
  state.towersCellHeight = "FULL";
  take(mdir::driver::writeCheckpoint(dir + "/full.h5", state));
  std::fill(state.forces.begin(), state.forces.end(), 0.0);
  take(mdir::driver::writeCheckpoint(dir + "/zero.h5", state));
  return 0;
}

int main(int argc, char **argv) {
  if (argc == 3)
    return writeCheckpointVariants(argv[1], argv[2]);

  using namespace mdir::tool;
  const double cube[] = {4, 4, 4}, flat[] = {4, 4, 2.75};
  for (const auto &height : {"", "FULL", "HALF"})
    for (int pin : {0, 1, 2}) {
      auto cells = towersCells(cube, 1, height, pin);
      if (cells.size() != size_t((*height ? 1 : 2) * (pin ? 1 : 2)))
        return 1;
      for (auto cell : cells)
        if ((pin && pin != cell.split) ||
            (*height && cell.reach != (std::string(height) == "FULL" ? 1 : 2)))
          return 1;
    }
  auto cells = towersCells(flat, 1, "", 0);
  if (cells.size() != 2 || cells[0].reach != 2 || cells[1].reach != 2 ||
      !towersCells(flat, 1, "FULL", 0).empty())
    return 1;

  auto record = [](float offset) {
    return towersTrialRecord(
        llvm::json::Array{llvm::json::Object{{"shape", "a"}},
                          llvm::json::Object{{"shape", "b"}}},
        {{5, 1, 4, 2, 3},
         {5 + offset, 1 + offset, 4 + offset, 2 + offset, 3 + offset}});
  };
  for (float offset : {-8.f, 1.f}) {
    auto comparison = record(offset);
    if (comparison.getNumber("margin_us") != std::abs(offset) ||
        comparison.getBoolean("below_noise") != (offset == 1) ||
        comparison.getString("winner") != (offset == 1 ? "a" : "b"))
      return 1;
  }
  if (mdir::kernels::towersForceBlocks(13, 2, 65535, 128) != 26 + 512 ||
      mdir::kernels::towersForceBlocks(13, 2, 65536, 128) != 26 + 128)
    return 1;
  struct Frame {
    const char *name;
    double heights[3], cutoff;
    int64_t atoms;
  };
  const Frame frames[] = {
      {"t10", {4.67210933, 4.67463534, 4.63455783}, 0.9, 9989},
      {"t20", {5.86840328, 5.85005704, 5.86608633}, 0.9, 19880},
      {"jac", {6.223, 6.223, 6.223}, 0.8, 23558},
      {"factorix", {14.20855468, 8.33368905, 7.86783548}, 0.8, 90906},
      {"cellulose", {25.92299548, 12.45580494, 12.35021394}, 0.8, 408609},
      {"stmv", {22.11723142, 22.31988809, 22.44925841}, 0.8, 1067095},
      {"ubiquitin-octahedron", {13.063949, 13.063943, 13.063950}, 0.9, 311971},
  };
  for (const auto &f : frames)
    std::printf("%s %s\n", f.name,
                mdir::tool::selectTowersReach(f.heights, f.cutoff, f.atoms) == 1
                    ? "FULL"
                    : "HALF");
}
