// Typed fragments become operations in their original shared regions (#568).
#include "TermIR.h"
#include "mdir/Dialect/MD/MDOps.h"
#include "mdir/Dialect/ProgramDialects.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/STLExtras.h"

using namespace mdir::driver::builder;

struct TermIRPlan::OwnedProgram {
  std::unique_ptr<mlir::MLIRContext> context;
  mlir::OwningOpRef<mlir::ModuleOp> module;
};

TermIRPlan::TermIRPlan() = default;
TermIRPlan::~TermIRPlan() = default;

mlir::MLIRContext &TermIRPlan::getContext() {
  if (!program) {
    program = std::make_unique<OwnedProgram>();
    mlir::DialectRegistry registry;
    mdir::insertProgramDialects(registry);
    program->context = std::make_unique<mlir::MLIRContext>(
        registry, mlir::MLIRContext::Threading::DISABLED);
    program->context->loadAllAvailableDialects();
  }
  return *program->context;
}

mlir::ModuleOp TermIRPlan::getModule() {
  mlir::MLIRContext &context = getContext();
  if (!program->module)
    program->module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
  return *program->module;
}

llvm::Error mdir::driver::builder::checkFrameEvaluation(mlir::Operation &scope) {
  bool evaluates = false;
  scope.walk([&](mdir::md::EvaluateOp) { evaluates = true; });
  if (!evaluates)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "the program of frames: the start of this program evaluates "
        "nothing, so its loop over frames would give nothing "
        "(Builder::emitEntry)");
  return llvm::Error::success();
}
