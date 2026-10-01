// Moves the allocas of a constant size outside the regions of OpenMP to the
// entry block of their function (D117).

#include "mdir/Conversion/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Matchers.h"

using namespace mlir;

namespace mdir {

#define GEN_PASS_DEF_HOISTSTATICALLOCAS
#include "mdir/Conversion/Passes.h.inc"

namespace {
/// Whether `op` is inside an op of OpenMP below `function`.
bool isInOpenMP(Operation *op, Operation *function) {
  for (Operation *parent = op->getParentOp(); parent && parent != function;
       parent = parent->getParentOp())
    if (parent->getDialect() &&
        parent->getDialect()->getNamespace() == "omp")
      return true;
  return false;
}

class HoistStaticAllocas
    : public impl::HoistStaticAllocasBase<HoistStaticAllocas> {
public:
  void runOnOperation() final {
    getOperation().walk([&](FunctionOpInterface function) {
      Region &body = function.getFunctionBody();
      if (body.empty())
        return;
      Block &entry = body.front();
      SmallVector<LLVM::AllocaOp> moved;
      function.walk([&](LLVM::AllocaOp alloca) {
        if (alloca->getBlock() == &entry ||
            isInOpenMP(alloca, function.getOperation()))
          return;
        if (!matchPattern(alloca.getArraySize(), m_Constant()))
          return;
        moved.push_back(alloca);
      });
      for (LLVM::AllocaOp alloca : moved) {
        Operation *size = alloca.getArraySize().getDefiningOp();
        OpBuilder builder(&entry, entry.begin());
        Operation *copy = builder.clone(*size);
        alloca->moveAfter(copy);
        alloca.getArraySizeMutable().assign(copy->getResult(0));
      }
    });
  }
};
} // namespace

} // namespace mdir
