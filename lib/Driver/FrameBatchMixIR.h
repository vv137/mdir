// Per-type combining rules and their seed weights, entirely on device.
#ifndef MDIR_LIB_DRIVER_FRAMEBATCHMIXIR_H
#define MDIR_LIB_DRIVER_FRAMEBATCHMIXIR_H
#include "mlir/IR/Value.h"
#include "llvm/Support/Error.h"
namespace mlir { class OpBuilder; }
namespace mdir::driver::builder {
struct FrameBatchMixIR {
  unsigned types;
  bool geometric, derivative;
};
/// Pair map is a theta index or -1 for a table; for derivative weights it
/// is an enabled mask (0 or -1). Type values retain the model's logical
/// f64 precision. The optional logical destination receives the same value
/// before truncation to the physical destination's f32/f64 precision.
llvm::Error buildFrameBatchMixing(mlir::OpBuilder &, mlir::Value theta,
    mlir::Value typeMap, mlir::Value typeValues, mlir::Value pairMap,
    mlir::Value fixedPairs, mlir::Value destination,
    mlir::Value logicalDestination, const FrameBatchMixIR &, mlir::Value status = {});
}
#endif
