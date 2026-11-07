// Device-only parameter maps and ordered contractions for frame batches.
#ifndef MDIR_LIB_DRIVER_FRAMEBATCHTUNABLEIR_H
#define MDIR_LIB_DRIVER_FRAMEBATCHTUNABLEIR_H
#include "mlir/IR/Value.h"
#include "llvm/Support/Error.h"
#include "llvm/ADT/ArrayRef.h"
#include <cstdint>
namespace mlir { class OpBuilder; }
namespace mdir::driver::builder {
/// Upload immutable indices once at activation start. The activation's
/// device allocation record owns the buffer until the activation ends.
llvm::Expected<mlir::Value> buildFrameBatchParameterMap(mlir::OpBuilder &,
    llvm::ArrayRef<int64_t> map, int64_t parameters);
llvm::Expected<mlir::Value> buildFrameBatchInitialValues(mlir::OpBuilder &,
    llvm::ArrayRef<double> values);
/// All buffers are contiguous device memrefs. A negative map preserves a
/// site's existing value; other sites take theta[map[i]] in physical precision.
/// Shapes and map bounds are checked when preparing the immutable map.
llvm::Error buildFrameBatchMappedParameters(mlir::OpBuilder &,
    mlir::Value theta, mlir::Value map, mlir::Value destination);
/// One independent thread per output. CSR contributions accumulate in their
/// given order, without atomics or a reassociated parallel reduction. The
/// caller's immutable plan gives offsets, source indices and f64 weights.
/// This also contracts rows with tangent/cotangent vectors on the device.
llvm::Error buildFrameBatchOrderedContraction(mlir::OpBuilder &,
    mlir::Value source, mlir::Value offsets, mlir::Value indices,
    mlir::Value weights, mlir::Value destination);
}
#endif
