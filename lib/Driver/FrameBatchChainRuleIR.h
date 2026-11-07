// Ordered type sums and the host entry's raw correction phases on device.
#ifndef MDIR_LIB_DRIVER_FRAMEBATCHCHAINRULEIR_H
#define MDIR_LIB_DRIVER_FRAMEBATCHCHAINRULEIR_H
#include "mlir/IR/Value.h"
namespace mlir { class OpBuilder; }
namespace mdir::driver::builder {
struct FrameBatchParameterPlan;
void buildFrameBatchRawCorrections(mlir::OpBuilder &, const FrameBatchParameterPlan &,
    mlir::Value raw, mlir::Value kinds, mlir::Value volumeConstants,
    mlir::Value fixedConstants, mlir::Value typeOffsets,
    mlir::Value particlesByType, mlir::Value volumeScale);
}
#endif
