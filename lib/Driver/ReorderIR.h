// Typed physical state and semantic parameter slots for particle reordering.
#ifndef MDIR_LIB_DRIVER_REORDERIR_H
#define MDIR_LIB_DRIVER_REORDERIR_H
#include "mlir/IR/ValueRange.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>
namespace mlir { class OpBuilder; }
namespace mdir::driver::builder {
struct ReorderFieldIR {
  mlir::Value value;
  std::optional<unsigned> regenerateFrom;
};
struct ReorderIRInputs {
  mlir::Value positions, velocities, forces, masses, ids, cell;
  llvm::ArrayRef<ReorderFieldIR> fields;
  mlir::ValueRange originalRelations;
};
struct ReorderIRConfig {
  double width;
  std::optional<unsigned> anchorRelation;
};
struct ReorderIRResult {
  mlir::Value positions, velocities, forces, masses, ids;
  llvm::SmallVector<mlir::Value> fields, relations;
};
ReorderIRResult buildReorder(mlir::OpBuilder &, const ReorderIRInputs &,
                              const ReorderIRConfig &);
mlir::Value buildRenumber(mlir::OpBuilder &, mlir::Value relation, mlir::Value ids);
} // namespace mdir::driver::builder
#endif
