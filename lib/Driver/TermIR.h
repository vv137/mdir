// Owned typed term fragments at the existing assembly sites (#568).
#ifndef MDIR_LIB_DRIVER_TERMIR_H
#define MDIR_LIB_DRIVER_TERMIR_H

#include "mlir/IR/ValueRange.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include <functional>
#include <vector>

namespace mlir {
class ModuleOp;
class OpBuilder;
} // namespace mlir
namespace mdir::driver {
struct Program;
namespace builder {

using TermIRValues = llvm::SmallVector<mlir::Value, 4>;
using TermIRRecipe =
    std::function<TermIRValues(mlir::OpBuilder &, mlir::ValueRange)>;

/// A build owns its recipes and every capture. Names/types are borrowed only
/// while their connection to the remaining assembly is written; recipes see
/// typed operands at the original insertion point, never the whole builder.
class TermIRPlan {
public:
  bool empty() const { return recipes.empty(); }
  void emit(llvm::raw_ostream &, llvm::StringRef indent,
            llvm::ArrayRef<llvm::StringRef> results,
            llvm::ArrayRef<llvm::StringRef> resultTypes,
            llvm::ArrayRef<llvm::StringRef> operands,
            llvm::ArrayRef<llvm::StringRef> operandTypes, TermIRRecipe);
  llvm::Error materialize(mlir::ModuleOp);

private:
  std::vector<TermIRRecipe> recipes;
};

/// Joins the remaining assembly and typed fragments before printing/lowering.
/// No recipe or internal connection survives into a valid generated program.
llvm::Error finishTermIRProgram(Program &, TermIRPlan &);

} // namespace builder
} // namespace mdir::driver
#endif
