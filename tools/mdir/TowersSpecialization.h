// Prepare the mixed pair region once and specialize its rectangular launch
// variants. An empty emitter means the program has no force-only region to time.
// In that case setup requests the rebuild-only image to measure populations.
#pragma once
#include "TowersTrialInputs.h"
#include "mdir/Driver/System.h"
#include <functional>

namespace mdir::tool {
using TowersEmitter = std::function<llvm::Expected<std::string>(
    const driver::TowersShape &, TowersBuffers *)>;
llvm::Expected<TowersEmitter>
prepareTowersSpecialization(const driver::Control &, const driver::System &,
                            int reach, int capacity, int device,
                            std::string *rebuildOnly);
} // namespace mdir::tool
