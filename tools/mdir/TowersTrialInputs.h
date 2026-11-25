// Immutable arguments of a compiled pair specialization, in bare-pointer order.
#pragma once

#include "mdir/Driver/Builder.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ValueRange.h"

namespace mdir::tool {
/// Resolve the dimensions used by storage assignment for a pair table.
mlir::MemRefType getTowersTableType(const driver::Program &program,
                                    mlir::Value table);

struct TowersBuffers {
  std::vector<std::vector<char>> values;
  size_t fields = 0;
};
llvm::Expected<TowersBuffers>
bindTowersTrialInputs(const driver::Program &program, mlir::ValueRange fields,
                      llvm::ArrayRef<mlir::Value> captures);
} // namespace mdir::tool
