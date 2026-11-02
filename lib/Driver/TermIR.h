// Owned native program and typed term fragments (#568).
#ifndef MDIR_LIB_DRIVER_TERMIR_H
#define MDIR_LIB_DRIVER_TERMIR_H

#include "mlir/IR/ValueRange.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include <memory>

namespace mlir {
class MLIRContext;
class ModuleOp;
class OpBuilder;
class Operation;
} // namespace mlir
namespace mdir::driver {
struct Program;
namespace builder {

using TermIRValues = llvm::SmallVector<mlir::Value, 4>;
/// One build owns one native module and context. Every factory inserts
/// operations directly; only the final consumer handoff is serialized.
class TermIRPlan {
public:
  TermIRPlan();
  ~TermIRPlan();
  mlir::MLIRContext &getContext();
  mlir::ModuleOp getModule();

private:
  // Adapters are destroyed before the module and context. Borrowed input IR
  // shares the full build lifetime.
  struct OwnedProgram;
  std::unique_ptr<OwnedProgram> program;
};

/// A frame iteration must contain an actual potential evaluation.
llvm::Error checkFrameEvaluation(mlir::Operation &scope);

/// Serializes the owned native program at the consumer handoff.
llvm::Error serializeNativeProgram(Program &, TermIRPlan &);

} // namespace builder
} // namespace mdir::driver
#endif
