// Compare the five independent whole-step samples of each feasible cell.
#pragma once
#include "llvm/Support/JSON.h"
#include <algorithm>
#include <array>
#include <numeric>
#include <vector>

namespace mdir::tool {
inline llvm::json::Object
towersTrialRecord(llvm::json::Array cells,
                  std::vector<std::array<float, 5>> times) {
  std::vector<size_t> order(times.size());
  std::iota(order.begin(), order.end(), 0);
  for (size_t i : order) {
    auto &t = times[i];
    std::sort(t.begin(), t.end());
    auto &cell = *cells[i].getAsObject();
    cell["median_us"] = double(t[2]);
    cell["spread_us"] = double(t[3] - t[1]);
  }
  std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
    return times[a][2] < times[b][2];
  });
  size_t best = order.front();
  auto &winner = *cells[best].getAsObject();
  llvm::json::Object record{{"winner_cell", int64_t(best)},
                            {"winner", winner["shape"]},
                            {"margin_us", nullptr},
                            {"below_noise", nullptr}};
  if (order.size() > 1) {
    auto &runner = times[order[1]];
    record["margin_us"] = double(runner[2] - times[best][2]);
    record["below_noise"] = times[best][3] >= runner[1];
  }
  record["cells"] = std::move(cells);
  return record;
}
} // namespace mdir::tool
