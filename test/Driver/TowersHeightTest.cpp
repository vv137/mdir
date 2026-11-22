// Perpendicular heights of rectangular and triclinic frames, in nm.
#include "TowersHeight.h"
#include <cstdio>

int main() {
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
