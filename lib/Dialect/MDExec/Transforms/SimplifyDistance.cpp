// Rewriting of pair kernels in powers of the squared distance.

#include "mdir/Dialect/MDExec/Transforms/Passes.h"

#include "mdir/Dialect/MDExec/MDExecDialect.h"
#include "mdir/Dialect/MDExec/MDExecOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/DenseMap.h"

using namespace mlir;
using namespace mdir;
using namespace mdir::md_exec;

namespace {

/// scale · Π factor^exponent · r^power
struct Term {
  double scale = 1.0;
  SmallVector<std::pair<Value, int>, 4> factors;
  int power = 0;
};

/// A sum of terms.
using Form = SmallVector<Term, 4>;

/// Rewrites one kernel.
class Rewriter {
public:
  Rewriter(PairForOp op) : op(op), builder(op.getContext()) {}

  void run();

private:
  /// The form of a value of the old kernel, if the value is an f64.
  std::optional<Form> getForm(Value old);

  /// The value of the new kernel that holds a value of the old one.
  Value materialize(Value old);

  /// Emits code that computes `form`.
  Value emit(const Form &form, Location loc);
  Value emitTerm(const Term &term, Location loc, bool magnitude);
  Value emitPowerOfDistance(int power, Location loc);

  /// Computes the form of the result of `oldOp`, if `oldOp` is one of the
  /// ops that a form can be carried through. Returns false otherwise.
  bool carry(Operation *oldOp);

  static void normalize(Form &form);
  static Form multiply(const Form &lhs, const Form &rhs);
  static Form negate(Form form);

  /// A form with a single term that stands for the value `value`.
  static Form opaque(Value value) {
    Term term;
    term.factors.push_back({value, 1});
    return Form{term};
  }
  static Form constant(double value) {
    Term term;
    term.scale = value;
    return Form{term};
  }

  /// The form of `old`, as one term. Emits code if the form has more.
  Term asSingleTerm(Value old);

  Value createConstant(double value, Location loc) {
    return arith::ConstantOp::create(builder, loc, builder.getF64Type(),
                                     builder.getF64FloatAttr(value));
  }
  Value createPower(Value base, int exponent, Location loc) {
    assert(exponent > 0 && "negative powers are divisions");
    if (exponent == 1)
      return base;
    Value count = arith::ConstantOp::create(
        builder, loc, builder.getI32Type(),
        builder.getI32IntegerAttr(exponent));
    return math::FPowIOp::create(builder, loc, base, count);
  }
  Value multiplyValues(Value lhs, Value rhs, Location loc) {
    if (!lhs)
      return rhs;
    if (!rhs)
      return lhs;
    return arith::MulFOp::create(builder, loc, lhs, rhs);
  }

  PairForOp op;
  OpBuilder builder;
  Block *fresh = nullptr;

  /// Old values to new values, for values that have no form.
  IRMapping mapping;
  llvm::DenseMap<Value, Form> forms;
  llvm::DenseMap<Value, Value> emitted;

  /// The squared distance, and what is derived from it on demand.
  Value r2;
  Value inverse;
  Value distance;
};

} // namespace

//===----------------------------------------------------------------------===//
// Forms
//===----------------------------------------------------------------------===//

void Rewriter::normalize(Form &form) {
  auto order = [](const std::pair<Value, int> &lhs,
                  const std::pair<Value, int> &rhs) {
    return lhs.first.getAsOpaquePointer() < rhs.first.getAsOpaquePointer();
  };

  for (Term &term : form) {
    llvm::sort(term.factors, order);
    SmallVector<std::pair<Value, int>, 4> merged;
    for (auto &factor : term.factors) {
      if (!merged.empty() && merged.back().first == factor.first)
        merged.back().second += factor.second;
      else
        merged.push_back(factor);
    }
    llvm::erase_if(merged, [](auto &factor) { return factor.second == 0; });
    term.factors = std::move(merged);
  }

  // Add up the terms that differ only in their scale.
  Form result;
  for (Term &term : form) {
    auto same = llvm::find_if(result, [&](const Term &other) {
      return other.power == term.power && other.factors == term.factors;
    });
    if (same == result.end())
      result.push_back(term);
    else
      same->scale += term.scale;
  }
  llvm::erase_if(result, [](const Term &term) { return term.scale == 0.0; });
  form = std::move(result);
}

Form Rewriter::multiply(const Form &lhs, const Form &rhs) {
  Form product;
  for (const Term &a : lhs) {
    for (const Term &b : rhs) {
      Term term;
      term.scale = a.scale * b.scale;
      term.power = a.power + b.power;
      term.factors = a.factors;
      term.factors.append(b.factors.begin(), b.factors.end());
      product.push_back(std::move(term));
    }
  }
  normalize(product);
  return product;
}

Form Rewriter::negate(Form form) {
  for (Term &term : form)
    term.scale = -term.scale;
  return form;
}

std::optional<Form> Rewriter::getForm(Value old) {
  auto found = forms.find(old);
  if (found != forms.end())
    return found->second;
  if (!old.getType().isF64())
    return std::nullopt;

  // A constant from outside the kernel.
  APFloat value(0.0);
  if (matchPattern(old, m_ConstantFloat(&value)))
    return constant(value.convertToDouble());

  // A value that was computed by an op that forms are not carried through,
  // an argument of the kernel, or a value from outside.
  return opaque(mapping.lookupOrDefault(old));
}

Term Rewriter::asSingleTerm(Value old) {
  Form form = *getForm(old);
  if (form.size() == 1)
    return form.front();
  if (form.empty())
    return constant(0.0).front();
  return opaque(materialize(old)).front();
}

//===----------------------------------------------------------------------===//
// Emission
//===----------------------------------------------------------------------===//

Value Rewriter::emitPowerOfDistance(int power, Location loc) {
  if (power == 0)
    return Value();

  // r^(2k + 1) = r^(2k) · r, with k rounded toward minus infinity.
  bool odd = power % 2 != 0;
  int half = (power - (odd ? 1 : 0)) / 2;

  Value result;
  if (half > 0) {
    result = createPower(r2, half, loc);
  } else if (half < 0) {
    if (!inverse) {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(fresh);
      inverse = arith::DivFOp::create(builder, loc,
                                      createConstant(1.0, loc), r2);
    }
    result = createPower(inverse, -half, loc);
  }
  if (odd) {
    if (!distance) {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(fresh);
      distance = math::SqrtOp::create(builder, loc, r2);
    }
    result = multiplyValues(result, distance, loc);
  }
  return result;
}

Value Rewriter::emitTerm(const Term &term, Location loc, bool magnitude) {
  double scale = magnitude ? std::abs(term.scale) : term.scale;

  Value numerator, denominator;
  for (auto [value, exponent] : term.factors) {
    if (exponent > 0)
      numerator = multiplyValues(numerator,
                                 createPower(value, exponent, loc), loc);
    else
      denominator = multiplyValues(
          denominator, createPower(value, -exponent, loc), loc);
  }
  numerator =
      multiplyValues(numerator, emitPowerOfDistance(term.power, loc), loc);

  if (scale != 1.0 || !numerator)
    numerator = multiplyValues(createConstant(scale, loc), numerator, loc);
  if (denominator)
    numerator = arith::DivFOp::create(builder, loc, numerator, denominator);
  return numerator;
}

Value Rewriter::emit(const Form &form, Location loc) {
  if (form.empty())
    return createConstant(0.0, loc);

  Value total;
  for (const Term &term : form) {
    if (!total) {
      total = emitTerm(term, loc, /*magnitude=*/false);
      continue;
    }
    Value value = emitTerm(term, loc, /*magnitude=*/true);
    if (term.scale < 0.0)
      total = arith::SubFOp::create(builder, loc, total, value);
    else
      total = arith::AddFOp::create(builder, loc, total, value);
  }
  return total;
}

Value Rewriter::materialize(Value old) {
  auto form = forms.find(old);
  if (form == forms.end())
    return mapping.lookupOrDefault(old);

  auto found = emitted.find(old);
  if (found != emitted.end())
    return found->second;
  Value value = emit(form->second, old.getLoc());
  emitted[old] = value;
  return value;
}

//===----------------------------------------------------------------------===//
// Ops that forms are carried through
//===----------------------------------------------------------------------===//

bool Rewriter::carry(Operation *oldOp) {
  if (oldOp->getNumResults() != 1 || !oldOp->getResult(0).getType().isF64())
    return false;
  Value result = oldOp->getResult(0);

  APFloat value(0.0);
  if (matchPattern(oldOp, m_ConstantFloat(&value))) {
    forms[result] = constant(value.convertToDouble());
    return true;
  }

  // The square root of the squared distance is the distance.
  if (isa<math::SqrtOp>(oldOp) &&
      oldOp->getOperand(0) == op.getKernel().front().getArgument(0)) {
    Term term;
    term.power = 1;
    forms[result] = Form{term};
    return true;
  }

  if (isa<arith::NegFOp>(oldOp)) {
    forms[result] = negate(*getForm(oldOp->getOperand(0)));
    return true;
  }

  if (isa<arith::AddFOp, arith::SubFOp>(oldOp)) {
    Form sum = *getForm(oldOp->getOperand(0));
    Form rhs = *getForm(oldOp->getOperand(1));
    if (isa<arith::SubFOp>(oldOp))
      rhs = negate(std::move(rhs));
    sum.append(rhs.begin(), rhs.end());
    normalize(sum);
    forms[result] = std::move(sum);
    return true;
  }

  if (isa<arith::MulFOp>(oldOp)) {
    Form lhs = *getForm(oldOp->getOperand(0));
    Form rhs = *getForm(oldOp->getOperand(1));
    // A product of two sums is not multiplied out. The terms of the
    // expanded product can be much larger than the product itself, as in a
    // polynomial in `r - a` that is written in powers of `r`, and their sum
    // then loses the digits that the product had.
    if (lhs.size() > 1 && rhs.size() > 1) {
      lhs = Form{asSingleTerm(oldOp->getOperand(0))};
      rhs = Form{asSingleTerm(oldOp->getOperand(1))};
    }
    // A sum that has been computed is used as it is.
    if (lhs.size() > 1 && emitted.count(oldOp->getOperand(0)))
      lhs = Form{asSingleTerm(oldOp->getOperand(0))};
    if (rhs.size() > 1 && emitted.count(oldOp->getOperand(1)))
      rhs = Form{asSingleTerm(oldOp->getOperand(1))};
    forms[result] = multiply(lhs, rhs);
    return true;
  }

  if (isa<arith::DivFOp>(oldOp)) {
    Form numerator = *getForm(oldOp->getOperand(0));
    Term denominator = asSingleTerm(oldOp->getOperand(1));
    // A division by a constant zero stays a division.
    if (denominator.scale == 0.0)
      return false;

    Term inverted;
    inverted.scale = 1.0 / denominator.scale;
    inverted.power = -denominator.power;
    for (auto [value, exponent] : denominator.factors)
      inverted.factors.push_back({value, -exponent});
    forms[result] = multiply(numerator, Form{inverted});
    return true;
  }

  if (isa<math::FPowIOp>(oldOp)) {
    APInt exponent;
    if (!matchPattern(oldOp->getOperand(1), m_ConstantInt(&exponent)))
      return false;
    int n = exponent.getSExtValue();
    Term base = asSingleTerm(oldOp->getOperand(0));
    if (base.scale == 0.0 && n <= 0)
      return false;

    Term power;
    power.scale = std::pow(base.scale, n);
    power.power = base.power * n;
    for (auto [value, e] : base.factors)
      power.factors.push_back({value, e * n});
    Form form{power};
    normalize(form);
    forms[result] = std::move(form);
    return true;
  }

  return false;
}

//===----------------------------------------------------------------------===//
// The kernel
//===----------------------------------------------------------------------===//

void Rewriter::run() {
  Region &region = op.getKernel();
  Block &old = region.front();

  fresh = new Block();
  region.push_back(fresh);
  for (BlockArgument argument : old.getArguments())
    mapping.map(argument,
                fresh->addArgument(argument.getType(), argument.getLoc()));
  builder.setInsertionPointToEnd(fresh);

  r2 = fresh->getArgument(0);
  Term squared;
  squared.power = 2;
  forms[old.getArgument(0)] = Form{squared};

  for (Operation &oldOp : old) {
    if (carry(&oldOp))
      continue;
    // Any other op needs its operands as values.
    for (Value operand : oldOp.getOperands())
      if (forms.count(operand))
        mapping.map(operand, materialize(operand));
    builder.clone(oldOp, mapping);
  }
  old.erase();
}

namespace mdir {
namespace md_exec {

#define GEN_PASS_DEF_SIMPLIFYDISTANCE
#include "mdir/Dialect/MDExec/Transforms/Passes.h.inc"

namespace {
class SimplifyDistance
    : public impl::SimplifyDistanceBase<SimplifyDistance> {
public:
  using impl::SimplifyDistanceBase<SimplifyDistance>::SimplifyDistanceBase;

  void runOnOperation() final {
    SmallVector<PairForOp> loops;
    getOperation()->walk([&](PairForOp loop) { loops.push_back(loop); });
    for (PairForOp loop : loops)
      Rewriter(loop).run();
  }
};
} // namespace

} // namespace md_exec
} // namespace mdir
