// Immutable arguments of a compiled pair specialization, in bare-pointer order.
#pragma once

#include "mdir/Driver/Builder.h"
#include "mlir/IR/ValueRange.h"

namespace mdir::tool {
struct TowersBuffers {
  std::vector<std::vector<char>> values;
  size_t fields = 0;
};
llvm::Expected<TowersBuffers>
bindTowersTrialInputs(const driver::Program &program, mlir::ValueRange fields,
                      llvm::ArrayRef<mlir::Value> captures);
} // namespace mdir::tool
