// Proof of exchange contracts.
//
// See docs/ops-m0.md, Section 4.7.

#include "mdir/Dialect/MD/Transforms/Passes.h"

#include "mdir/Dialect/MD/MDOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/DenseMap.h"

using namespace mlir;
using namespace mdir::md;

namespace {

/// Compares the values of a pair kernel with the values of the same kernel
/// after the two particles have been swapped.
///
/// Swapping leaves the distance unchanged, negates the displacement, and
/// swaps the two values of every gathered field. Values defined outside the
/// kernel are unchanged.
class ExchangeProof {
public:
  explicit ExchangeProof(Block &kernel) : kernel(kernel) {}

  /// Returns +1 if `a` equals `b` after the swap, −1 if `a` equals the
  /// negative of `b` after the swap, and 0 if neither could be shown.
  int compare(Value a, Value b);

private:
  int compute(Value a, Value b);

  /// Compares the operands of `a` with the operands of `b`. If `swapped`,
  /// the two operands of `b` are taken in reverse order. Sets `signs` and
  /// returns true if every operand could be compared.
  bool compareOperands(Operation *a, Operation *b, bool swapped,
                       SmallVectorImpl<int> &signs);

  /// The sign of the result of `op`, given the signs of its operands, or 0.
  int combine(Operation *op, ArrayRef<int> signs);

  bool isInKernel(Operation *op) { return op->getBlock() == &kernel; }

  Block &kernel;
  llvm::DenseMap<std::pair<Value, Value>, int> known;
};

} // namespace

int ExchangeProof::compare(Value a, Value b) {
  auto key = std::make_pair(a, b);
  auto found = known.find(key);
  if (found != known.end())
    return found->second;
  int sign = compute(a, b);
  known[key] = sign;
  return sign;
}

int ExchangeProof::compute(Value a, Value b) {
  // An argument of the kernel.
  if (auto argument = dyn_cast<BlockArgument>(b);
      argument && argument.getOwner() == &kernel) {
    unsigned index = argument.getArgNumber();
    if (index == 0)
      return a == b ? 1 : 0;
    if (index == 1)
      return a == b ? -1 : 0;
    unsigned partner = 2 + ((index - 2) ^ 1u);
    return a == kernel.getArgument(partner) ? 1 : 0;
  }

  // A value from outside the kernel does not change.
  Operation *opB = b.getDefiningOp();
  if (!opB || !isInKernel(opB))
    return a == b ? 1 : 0;

  Operation *opA = a.getDefiningOp();
  if (!opA || !isInKernel(opA))
    return 0;

  if (opA->getName() != opB->getName() ||
      opA->getAttrDictionary() != opB->getAttrDictionary() ||
      opA->getNumOperands() != opB->getNumOperands() ||
      a.getType() != b.getType() ||
      cast<OpResult>(a).getResultNumber() !=
          cast<OpResult>(b).getResultNumber())
    return 0;

  // A lookup in a symmetric table of pairs gives the same value with its
  // two indices swapped.
  if (auto lookup = dyn_cast<LookupOp>(opA)) {
    auto table = dyn_cast<TableType>(lookup.getTable().getType());
    if (table && table.getSymmetric() &&
        compare(opA->getOperand(0), opB->getOperand(0)) == 1 &&
        compare(opA->getOperand(1), opB->getOperand(2)) == 1 &&
        compare(opA->getOperand(2), opB->getOperand(1)) == 1)
      return 1;
  }

  SmallVector<int> signs;
  if (compareOperands(opA, opB, /*swapped=*/false, signs))
    if (int sign = combine(opA, signs))
      return sign;

  if (opA->getNumOperands() == 2 &&
      opA->hasTrait<OpTrait::IsCommutative>() &&
      compareOperands(opA, opB, /*swapped=*/true, signs))
    return combine(opA, signs);
  return 0;
}

bool ExchangeProof::compareOperands(Operation *a, Operation *b, bool swapped,
                                    SmallVectorImpl<int> &signs) {
  signs.clear();
  unsigned count = a->getNumOperands();
  for (unsigned i = 0; i != count; ++i) {
    unsigned j = swapped ? count - 1 - i : i;
    int sign = compare(a->getOperand(i), b->getOperand(j));
    if (sign == 0)
      return false;
    signs.push_back(sign);
  }
  return true;
}

int ExchangeProof::combine(Operation *op, ArrayRef<int> signs) {
  auto allPositive = [&]() {
    return llvm::all_of(signs, [](int sign) { return sign == 1; });
  };
  auto allEqual = [&]() {
    return llvm::all_of(signs, [&](int sign) { return sign == signs[0]; });
  };

  if (signs.empty())
    return 1;

  // The sign of a product or a quotient is the product of the signs.
  if (isa<arith::MulFOp, arith::DivFOp>(op))
    return signs[0] * signs[1];

  // A sum keeps a sign only if all its terms have it.
  if (isa<arith::AddFOp, arith::SubFOp>(op))
    return allEqual() ? signs[0] : 0;

  if (isa<arith::NegFOp, vector::BroadcastOp>(op))
    return signs[0];

  // The condition must be unchanged; the branches must agree.
  if (isa<arith::SelectOp>(op))
    return (signs[0] == 1 && signs[1] == signs[2]) ? signs[1] : 0;

  // Odd functions keep the sign of their argument.
  if (isa<math::SinOp, math::TanOp, math::AsinOp, math::AtanOp, math::SinhOp,
          math::TanhOp, math::ErfOp>(op))
    return signs[0];

  // Even functions lose it.
  if (isa<math::AbsFOp, math::CosOp, math::CoshOp>(op))
    return 1;

  // An integer power is even or odd with its exponent.
  if (isa<math::FPowIOp>(op)) {
    if (signs[1] != 1)
      return 0;
    if (signs[0] == 1)
      return 1;
    APInt exponent;
    if (!matchPattern(op->getOperand(1), m_ConstantInt(&exponent)))
      return 0;
    return exponent[0] ? -1 : 1;
  }

  // Any other op gives the same result for the same operands.
  return allPositive() ? 1 : 0;
}

/// Proves the exchange contract of `op`, or emits a diagnostic.
template <typename OpTy>
static LogicalResult checkContract(OpTy op) {
  if (op.getExchangeBasis() != ExchangeBasis::Proof ||
      op.getExchange() == Exchange::None)
    return success();

  Block &kernel = op.getKernel().front();
  Value value = cast<YieldOp>(kernel.getTerminator()).getOperand(0);

  ExchangeProof proof(kernel);
  int expected = op.getExchange() == Exchange::Symmetric ? 1 : -1;
  if (proof.compare(value, value) == expected)
    return success();

  InFlightDiagnostic diagnostic = op.emitOpError()
                                  << "cannot prove that the kernel is "
                                  << stringifyExchange(op.getExchange())
                                  << " under exchange of the two particles";
  diagnostic.attachNote()
      << "if the contract holds, state it as 'exchange("
      << stringifyExchange(op.getExchange()) << ", asserted)'";
  return diagnostic;
}

namespace mdir {
namespace md {

#define GEN_PASS_DEF_CHECKEXCHANGE
#include "mdir/Dialect/MD/Transforms/Passes.h.inc"

namespace {
class CheckExchange : public impl::CheckExchangeBase<CheckExchange> {
public:
  using impl::CheckExchangeBase<CheckExchange>::CheckExchangeBase;

  void runOnOperation() final {
    bool failed = false;
    getOperation()->walk([&](Operation *op) {
      if (auto sum = dyn_cast<SumRelationOp>(op))
        failed |= mlir::failed(checkContract(sum));
      else if (auto gather = dyn_cast<GatherRelationOp>(op))
        failed |= mlir::failed(checkContract(gather));
    });
    if (failed)
      signalPassFailure();
  }
};
} // namespace

} // namespace md
} // namespace mdir
