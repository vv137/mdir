// The independent device implementation of the tunable LJ tail and shift.
#ifndef MDIR_LIB_DRIVER_FRAMEBATCHLJIR_H
#define MDIR_LIB_DRIVER_FRAMEBATCHLJIR_H
#include "mlir/IR/Value.h"
#include "llvm/Support/Error.h"
namespace mlir { class OpBuilder; }
namespace mdir::driver::builder {
struct FrameBatchLJConstants {
  unsigned types;
  double particles, orderedPairs, cutoff, volume;
  bool shifted;
};
/// Logical f64 sigma/epsilon tables are flat row-major arrays. Counts has
/// one f64 per type; excluded has the two i64 types of each excluded pair.
/// The five f64 outputs at the reference volume are dispersion energy,
/// fixed energy, gradient shift energy, fixed shift, and dispersion virial.
/// Caller-owned immutable topology buffers are uploaded once per activation.
llvm::Error buildFrameBatchLJConstants(mlir::OpBuilder &,
    mlir::Value sigma, mlir::Value epsilon, mlir::Value counts,
    mlir::Value excluded, mlir::Value output, const FrameBatchLJConstants &);
}
#endif
