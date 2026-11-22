// Fixed rectangular frames of the 4n geometry comparison (lengths in nm).
#include "TowersHeight.h"
#include <cstdio>

int main() {
  struct Frame {
    const char *name;
    double box[3], cutoff;
  };
  const Frame frames[] = {
      {"t10", {4.67210933, 4.67463534, 4.63455783}, 0.9},
      {"t20", {5.86840328, 5.85005704, 5.86608633}, 0.9},
      {"jac", {6.223, 6.223, 6.223}, 0.8},
      {"factorix", {14.20855468, 8.33368905, 7.86783548}, 0.8},
      {"cellulose", {25.92299548, 12.45580494, 12.35021394}, 0.8},
      {"stmv", {22.11723142, 22.31988809, 22.44925841}, 0.8},
  };
  for (const auto &f : frames)
    std::printf("%s %s\n", f.name,
                mdir::tool::selectTowersReach(f.box, f.cutoff) == 1
                    ? "FULL"
                    : "HALF");
}
