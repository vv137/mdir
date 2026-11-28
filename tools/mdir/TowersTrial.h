// Private rebuilt frame for timing compiled launch variants on one CUDA grid.
#pragma once
#include "TowersTrialInputs.h"

namespace mdir::tool {
struct TowersTrial;
struct DropTowersTrial {
  void operator()(TowersTrial *) const;
};
using TrialFrame = std::unique_ptr<TowersTrial, DropTowersTrial>;
llvm::Expected<TrialFrame> createTowersTrial(int device,
                                             const driver::System &system,
                                             int cells, int reach, int capacity,
                                             const TowersBuffers &parameters,
                                             int retrySplit = 0);
struct TowersPopulation {
  int maximum;
  double mean;
};
llvm::Expected<TowersPopulation> readTowersPopulation(TowersTrial &frame,
                                                      const std::string &image);
struct TowersTiming {
  float pairUs = 0, rebuildUs = 0;
  int64_t stripeRetries = 0;
  double total() const { return double(pairUs) + rebuildUs; }
};
llvm::Expected<TowersTiming>
measureTowersTrial(TowersTrial &frame, const std::string &image,
                   const driver::TowersShape &shape, bool pressure);
} // namespace mdir::tool
