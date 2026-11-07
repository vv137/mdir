// Charge-dependent self/background corrections in the host arithmetic order.
#ifndef MDIR_LIB_DRIVER_FRAMEBATCHPMEIR_H
#define MDIR_LIB_DRIVER_FRAMEBATCHPMEIR_H
#include "mlir/IR/Value.h"
#include <cstdint>
namespace mlir { class OpBuilder; }
namespace mdir::driver::builder {
struct FrameBatchPMEConstants {
  int64_t rawBase;
  double selfMultiplier, backgroundMultiplier, backgroundFactor;
  double gradientSelfMultiplier, gradientBackgroundMultiplier, volume;
};
void buildFrameBatchPMEConstants(mlir::OpBuilder &, mlir::Value charges,
    mlir::Value output, mlir::Value rawVolume, mlir::Value rawFixed,
    const FrameBatchPMEConstants &);
}
#endif
