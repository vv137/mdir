// Derivatives of the independent LJ dispersion/shift correction.
#ifndef MDIR_LIB_DRIVER_FRAMEBATCHLJSLOPEIR_H
#define MDIR_LIB_DRIVER_FRAMEBATCHLJSLOPEIR_H
#include "mlir/IR/Value.h"
#include <cstdint>
namespace mlir { class OpBuilder; }
namespace mdir::driver::builder {
struct FrameBatchLJSlopeIR {
  unsigned types;
  int64_t rawBase;
  bool sigma;
  double factor, fixedFactor;
};
void buildFrameBatchLJSlopes(mlir::OpBuilder &, mlir::Value sigma,
    mlir::Value epsilon, mlir::Value orderedTypePairs, mlir::Value weights,
    mlir::Value volumeConstants, mlir::Value fixedConstants,
    const FrameBatchLJSlopeIR &);
}
#endif
