// Preflight of the rectangular TOWERS launch on the selected device.
#pragma once
#include "mdir/Driver/System.h"
#include "llvm/Support/JSON.h"

namespace mdir::tool {
llvm::Expected<llvm::json::Object>
checkTowersShape(const driver::Control &control, const driver::System &system);
}
