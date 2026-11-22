// Compile the mixed direct pair region for occupancy qualification.
#pragma once
#include "mdir/Driver/System.h"

namespace mdir::tool {
llvm::Expected<std::string>
compileTowersSpecialization(const driver::Control &, const driver::System &,
                            const driver::TowersShape &);
}
