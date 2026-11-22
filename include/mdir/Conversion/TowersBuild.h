// Rectangular TOWERS rebuild entries, specialized for one setup geometry.
#pragma once

#include "mlir/IR/BuiltinOps.h"

namespace mdir::kernels {
struct TowersLayout {
  int atoms, nx, ny, nz, reach, capacity;
  bool host, compact;
  int cells() const { return nx * ny * nz; }
};

/// Emit the serial functions or the device module. The device pack entry
/// requires whole warps. An overflow reports the largest required capacity;
/// grow the layout and repeat setup and insertion on the same coordinates.
mlir::LogicalResult emitTowersBuild(mlir::ModuleOp module, TowersLayout layout,
                                    bool audit = false);
} // namespace mdir::kernels
