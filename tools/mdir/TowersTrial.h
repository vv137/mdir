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
                                             const TowersBuffers &parameters);
llvm::Expected<float> measureTowersTrial(TowersTrial &frame,
                                         const std::string &image,
                                         const driver::TowersShape &shape);
} // namespace mdir::tool
