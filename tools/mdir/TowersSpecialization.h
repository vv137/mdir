// Prepare the mixed pair region once and specialize its rectangular launch
// variants.
#pragma once
#include "TowersTrialInputs.h"
#include "mdir/Driver/System.h"
#include <functional>

namespace mdir::tool {
using TowersEmitter = std::function<llvm::Expected<std::string>(
    const driver::TowersShape &, TowersBuffers *)>;
llvm::Expected<TowersEmitter>
prepareTowersSpecialization(const driver::Control &, const driver::System &,
                            int reach, int capacity, int device);
} // namespace mdir::tool
