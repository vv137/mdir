// Owned native program and typed term fragments (#568).
#ifndef MDIR_LIB_DRIVER_TERMIR_H
#define MDIR_LIB_DRIVER_TERMIR_H

#include "mlir/IR/ValueRange.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include <functional>
#include <memory>
#include <vector>

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
using TermIRRecipe =
    std::function<llvm::Expected<TermIRValues>(mlir::OpBuilder &,
                                               mlir::ValueRange)>;

/// Hooks borrow this synchronous capability; the plan owns each submitted
/// recipe.
using TermIRWriter = llvm::function_ref<void(
    llvm::StringRef, llvm::ArrayRef<llvm::StringRef>,
    llvm::ArrayRef<llvm::StringRef>, llvm::ArrayRef<llvm::StringRef>,
    llvm::ArrayRef<llvm::StringRef>, TermIRRecipe)>;

/// A build owns its recipes and every capture. Names/types are borrowed only
/// while their connection to the remaining assembly is written; recipes see
/// typed operands at the original insertion point, never the whole builder.
class TermIRPlan {
public:
  TermIRPlan();
  ~TermIRPlan();
  mlir::MLIRContext &getContext();
  mlir::ModuleOp getModule();
  bool empty() const { return recipes.empty(); }
  void emit(llvm::raw_ostream &, llvm::StringRef indent,
            llvm::ArrayRef<llvm::StringRef> results,
            llvm::ArrayRef<llvm::StringRef> resultTypes,
            llvm::ArrayRef<llvm::StringRef> operands,
            llvm::ArrayRef<llvm::StringRef> operandTypes, TermIRRecipe);
  llvm::Error materialize(mlir::ModuleOp);

private:
  // The build declares its plan before its adapters, and recipes are destroyed
  // before the owned module and context. Input IR shares the full build life.
  struct OwnedProgram;
  std::unique_ptr<OwnedProgram> program;
  std::vector<TermIRRecipe> recipes;
};

/// A frame iteration must contain an actual potential evaluation.
llvm::Error checkFrameEvaluation(mlir::Operation &scope);

/// Joins the remaining assembly and typed fragments before printing/lowering.
/// No recipe or internal connection survives into a valid generated program.
llvm::Error finishTermIRProgram(Program &, TermIRPlan &);

} // namespace builder
} // namespace mdir::driver
#endif
