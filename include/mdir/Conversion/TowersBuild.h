// Rectangular TOWERS rebuild entries, specialized for one setup geometry.
#pragma once

#include "mlir/IR/BuiltinOps.h"

namespace mdir::kernels {
struct TowersLayout {
  int atoms, nx, ny, nz, reach, capacity;
  bool host, compact;
  int cells() const { return nx * ny * nz; }
};

/// One source and one target span in the coordinate frame of the owner.
/// The fields identify their periodic offsets. In a triangular span only
/// source slots strictly above the target slot participate.
struct TowersSpan {
  mlir::Value field, first, count;
};
void emitOwnedTowersSpans(
    mlir::OpBuilder &builder, mlir::Location loc, int reach, mlir::Value owner,
    mlir::Value descriptors,
    llvm::function_ref<void(TowersSpan source, TowersSpan target,
                            mlir::Value triangular)>
        body);

/// Emit the serial functions or the device module. The device pack entry
/// requires whole warps. An overflow reports the largest required capacity;
/// grow the layout and repeat setup and insertion on the same coordinates.
mlir::LogicalResult emitTowersBuild(mlir::ModuleOp module, TowersLayout layout,
                                    bool audit = false);
} // namespace mdir::kernels
