// Proof of the site geometry and covering support of a template window.
#include "mdir/Dialect/MD/MDTemplateWindow.h"
#include "mdir/Dialect/MD/MDOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include <cmath>

using namespace mlir;
using namespace mdir::md;

llvm::SmallVector<std::array<unsigned, 2>, 5>
mdir::md::getTemplateWindowMembers(unsigned arity) {
  if (arity == 6)
    return {{0, 3}, {1, 0}, {2, 0}, {4, 3}, {5, 3}};
  if (arity == 4)
    return {{0, 1}, {2, 1}, {3, 1}};
  assert(arity == 2);
  return {{0, 1}};
}

namespace {
// Symbolic affine proof, not a tolerance-based numerical classification.
// Only literal scalar weights and zero vector constants may enter it.
static bool coefficients(Value value, Block &block, unsigned count,
                         SmallVectorImpl<double> &out, unsigned depth = 0) {
  if (depth > 64) return false;
  out.assign(count, 0.0);
  if (auto arg = dyn_cast<BlockArgument>(value)) {
    if (arg.getOwner() != &block || arg.getArgNumber() >= count) return false;
    out[arg.getArgNumber()] = 1.0;
    return true;
  }
  DenseFPElementsAttr zero;
  if (matchPattern(value, m_Constant(&zero)))
    return zero.isSplat() && zero.getSplatValue<APFloat>().isZero();
  Operation *op = value.getDefiningOp();
  if (!op) return false;
  if (isa<arith::AddFOp, arith::SubFOp>(op)) {
    SmallVector<double> a, b;
    if (!coefficients(op->getOperand(0), block, count, a, depth + 1) ||
        !coefficients(op->getOperand(1), block, count, b, depth + 1)) return false;
    double sign = isa<arith::SubFOp>(op) ? -1.0 : 1.0;
    for (unsigned i = 0; i != count; ++i) out[i] = a[i] + sign * b[i];
    return true;
  }
  if (auto mul = dyn_cast<arith::MulFOp>(op)) {
    for (unsigned i = 0; i != 2; ++i) {
      auto splat = mul->getOperand(i).getDefiningOp<vector::BroadcastOp>();
      FloatAttr scalar;
      DenseFPElementsAttr folded;
      double factor;
      if (splat && matchPattern(splat.getSource(), m_Constant(&scalar)))
        factor = scalar.getValueAsDouble();
      else if (matchPattern(mul->getOperand(i), m_Constant(&folded)) && folded.isSplat())
        factor = folded.getSplatValue<APFloat>().convertToDouble();
      else continue;
      if (!std::isfinite(factor) ||
          !coefficients(mul->getOperand(1 - i), block, count, out, depth + 1))
        return false;
      for (double &coefficient : out) coefficient *= factor;
      return true;
    }
  }
  return false;
}
static bool site(double a, double b, double alpha) {
  return (a == 0.0 && b == 0.0) || (a == 1.0 && b == 0.0) ||
         (a == 0.0 && b == 1.0) || (a == alpha && b == alpha);
}

template <typename Op> static LogicalResult verifyOperands(Op op, bool gather) {
  auto source = op.getRelation().template getDefiningOp<TemplatePairsOp>();
  if (!source)
    return op.emitOpError("expected a proved class domain from 'md.template_pairs'");
  auto relation = cast<RelationType>(op.getRelation().getType());
  auto positions = cast<FieldType>(op.getPositions().getType());
  if (positions.getParticleSet() != relation.getParticleSet() ||
      positions.getNumComponents() != 3 || !positions.getElementType().isF64())
    return op.emitOpError("expected positions of the class domain, with 3 components of f64");
  for (Value gathered : op.getGathered())
    if (cast<FieldType>(gathered.getType()).getParticleSet() != relation.getParticleSet())
      return op.emitOpError("expected gathered fields of the class domain");
  for (double value : {op.getCutoff().convertToDouble(), op.getExtent().convertToDouble()})
    if (!std::isfinite(value) || value <= 0.0)
      return op.emitOpError("expected positive finite cutoff and extent");
  double alpha = op.getAlpha().convertToDouble();
  if (!std::isfinite(alpha) || alpha < 0.0 || alpha > 0.5)
    return op.emitOpError("expected a finite template alpha between zero and one half");
  Type result = op.getResult().getType();
  if (gather) {
    if (result != op.getPositions().getType())
      return op.emitOpError("expected a force field with the positions' type");
  } else if (!result.isF64() && result != VectorType::get({9}, Float64Type::get(op.getContext()))) {
    return op.emitOpError("expected f64 energy or vector<9xf64> strain");
  }
  return success();
}

template <typename Op> static LogicalResult verifyBody(Op op, bool gather) {
  auto relation = cast<RelationType>(op.getRelation().getType());
  unsigned arity = relation.getArity();
  if (arity != 6 && arity != 4 && arity != 2)
    return op.emitOpError("expected WW, SW or SS class arity");
  unsigned count = getTemplateWindowMembers(arity).size();
  Block &block = op.getKernel().front();
  Builder builder(op.getContext());
  Type vector = VectorType::get({3}, builder.getF64Type());
  SmallVector<Type> types(count, vector);
  for (Value field : op.getGathered())
    types.append(arity, cast<FieldType>(field.getType()).getKernelValueType());
  if (!llvm::equal(block.getArgumentTypes(), types))
    return op.emitOpError("expected image displacement, whole-molecule offsets, then gathered member values");
  for (Operation &inner : block.without_terminator())
    if (inner.getNumRegions() || !isMemoryEffectFree(&inner))
      return op.emitOpError("expected a straight-line, effect-free window body");
  auto yield = dyn_cast<YieldOp>(block.getTerminator());
  unsigned stride = gather ? arity + 1 : 2;
  if (!yield || yield.getNumOperands() == 0 || yield.getNumOperands() % stride)
    return op.emitOpError("expected site contributions followed by their site displacements");
  unsigned pairs = yield.getNumOperands() / stride;
  Type contribution = gather ? vector : op.getResult().getType();
  unsigned contributions = pairs * (stride - 1);
  for (Value value : yield.getOperands().take_front(contributions))
    if (value.getType() != contribution)
      return op.emitOpError("site contribution type disagrees with the result");
  for (Value displacement : yield.getOperands().drop_front(contributions)) {
    SmallVector<double> c;
    double alpha = op.getAlpha().convertToDouble();
    if (displacement.getType() != vector ||
        !coefficients(displacement, block, count, c) || c[0] != 1.0 ||
        (arity == 6 && (!site(c[1], c[2], alpha) || !site(-c[3], -c[4], alpha))) ||
        (arity == 4 && !site(-c[1], -c[2], alpha)))
      return op.emitOpError("cannot prove template-window site geometry");
  }
  return success();
}
} // namespace

LogicalResult SumTemplateWindowOp::verify() { return verifyOperands(*this, false); }
LogicalResult SumTemplateWindowOp::verifyRegions() { return verifyBody(*this, false); }
LogicalResult GatherTemplateWindowOp::verify() { return verifyOperands(*this, true); }
LogicalResult GatherTemplateWindowOp::verifyRegions() { return verifyBody(*this, true); }
