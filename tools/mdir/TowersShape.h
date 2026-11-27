// Preflight of the periodic TOWERS launch on the selected device.
#pragma once
#include "mdir/Driver/System.h"
#include "llvm/Support/JSON.h"

namespace mdir::tool {
/// The rule of `neighbor_structure = "AUTO"` for TOWERS in `mdir run` and
/// `mdir check` (D[best-defaults]): `take` where the input is eligible and
/// the device's entry of the traversal table (D420) admits its size;
/// otherwise the rule of D261 stands and `reason` says why. `record` holds
/// the table lookup (architecture, threshold, basis), empty where no
/// device was queried.
struct AutoTowers {
  bool take = false;
  std::string reason;
  llvm::json::Object record;
};
AutoTowers resolveAutoTowers(const driver::Control &control,
                             const driver::System &system);
/// Resolve the choice available before a timing trial. AUTO as the value
/// means an eligible input still needs the paired measurements.
llvm::json::Object initialStepGraphChoice(llvm::StringRef request,
                                         llvm::StringRef refusal, size_t particles);
llvm::Expected<llvm::json::Object>
checkTowersShape(const driver::Control &control, const driver::System &system,
                 bool setup, llvm::StringRef checkpointHeight = {},
                 driver::TowersSetup *resolved = nullptr,
                 bool untimed = false);
}
