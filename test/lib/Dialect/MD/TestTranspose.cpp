// A test pass that asks the declared transpose of an op that is linear in
// some operands (AdjointOpInterface, D276, D295) for the
// adjoint of one operand, with the op's own result as the adjoint of the
// result, and remarks on what it gives. md-differentiate takes the
// transposes of md and md_dist itself; this reaches those of other
// dialects, such as md_exec.scatter_add, which no derivative takes yet.

#include "mdir/Dialect/MD/Transforms/AdjointInterface.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mdir::md;

namespace {

/// For each op with the attribute `test.transpose = N`, emits the adjoint
/// of operand N and remarks "transpose in operand N: ..." with the source
/// of the value, or lets the op's diagnostic stand.
struct TestTranspose
    : public PassWrapper<TestTranspose, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestTranspose)

  StringRef getArgument() const final { return "test-md-transpose"; }
  StringRef getDescription() const final {
    return "Asks the declared transpose of an op for an operand's adjoint";
  }

  void runOnOperation() final {
    SmallVector<Operation *> ops;
    getOperation()->walk([&](Operation *op) {
      if (op->hasAttr("test.transpose"))
        ops.push_back(op);
    });
    for (Operation *op : ops) {
      auto index = static_cast<unsigned>(
          op->getAttrOfType<IntegerAttr>("test.transpose").getInt());
      auto declared = dyn_cast<AdjointOpInterface>(op);
      if (!declared) {
        op->emitError() << "has no declared transpose";
        continue;
      }
      if (!declared.isLinearOperand(index)) {
        op->emitRemark() << "not linear in operand " << index;
        continue;
      }
      OpBuilder builder(op->getContext());
      builder.setInsertionPointAfter(op);
      Value adjoint =
          declared.emitTranspose(builder, index, op->getResult(0), Value());
      if (!adjoint)
        continue;
      if (adjoint == op->getResult(0))
        op->emitRemark() << "transpose in operand " << index
                         << ": the adjoint of the result";
      else
        op->emitRemark() << "transpose in operand " << index << ": '"
                         << adjoint.getDefiningOp()->getName() << "'";
    }
  }
};

} // namespace

namespace mdir {
namespace test {
void registerTestTranspose() { PassRegistration<TestTranspose>(); }
} // namespace test
} // namespace mdir
