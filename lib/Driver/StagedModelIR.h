// Owned user model IR at its input boundary (#568).
#ifndef MDIR_LIB_DRIVER_STAGEDMODELIR_H
#define MDIR_LIB_DRIVER_STAGEDMODELIR_H

#include "mlir/IR/ValueRange.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include <memory>

namespace mlir {
class MLIRContext;
class OpBuilder;
} // namespace mlir

namespace mdir::driver::builder {

/// A build's context outlives this input module and all of its clone recipes.
class StagedModelIR {
public:
  ~StagedModelIR();
  static llvm::Expected<std::shared_ptr<StagedModelIR>>
  parse(mlir::MLIRContext &, llvm::StringRef text, llvm::StringRef sourceName,
        llvm::StringRef termName);
  llvm::Error cloneDeclarations(mlir::OpBuilder &) const;
  llvm::Expected<mlir::Value> cloneBody(mlir::OpBuilder &,
                                       mlir::ValueRange inputs) const;

private:
  struct Impl;
  explicit StagedModelIR(std::unique_ptr<Impl>);
  std::unique_ptr<Impl> impl;
};

} // namespace mdir::driver::builder
#endif
