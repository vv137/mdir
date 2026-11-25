// Host representation shared by the offline partitioner and its device check.
#ifndef MDIR_EXPERIMENTS_BONDED_PARTITION_H
#define MDIR_EXPERIMENTS_BONDED_PARTITION_H

#include <array>
#include <cstdint>
#include <vector>

namespace bonded {
struct Term {
  std::array<unsigned, 5> atoms{};
  unsigned arity;
};

struct Unit {
  std::vector<unsigned> terms;
  std::vector<unsigned> atoms;
  std::vector<uint32_t> instructions;
};

void checkOnDevice(const std::vector<Term> &terms,
                   const std::vector<Unit> &units,
                   const std::vector<double> &positions, unsigned width);
} // namespace bonded
#endif
