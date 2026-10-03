// Functions of the squared distance as tables in f32, or as themselves in
// f64.
//
// See the description of the pass in Passes.td, and D94 in
// docs/decisions.md.

#include "mdir/Dialect/MDExec/Transforms/Passes.h"
#include "mdir/Dialect/MDExec/Transforms/Radial.h"

#include "mdir/Dialect/MDExec/MDExecOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"

#include <cmath>
#include <map>
#include <cstring>

using namespace mlir;
using namespace mdir;
using namespace mdir::md_exec;

namespace mdir {
namespace md_exec {

#define GEN_PASS_DEF_EXPANDRADIAL
#include "mdir/Dialect/MDExec/Transforms/Passes.h.inc"

std::optional<double> evaluateRadialOp(Operation *op,
                                       llvm::ArrayRef<double> a) {
  if (auto constant = dyn_cast<arith::ConstantOp>(op)) {
    if (auto real = dyn_cast<FloatAttr>(constant.getValue()))
      return real.getValueAsDouble();
    if (auto integer = dyn_cast<IntegerAttr>(constant.getValue()))
      return static_cast<double>(integer.getInt());
    return std::nullopt;
  }
  if (isa<arith::AddFOp>(op))
    return a[0] + a[1];
  if (isa<arith::SubFOp>(op))
    return a[0] - a[1];
  if (isa<arith::MulFOp>(op))
    return a[0] * a[1];
  if (isa<arith::DivFOp>(op))
    return a[0] / a[1];
  if (isa<arith::NegFOp>(op))
    return -a[0];
  if (isa<math::PowFOp, math::FPowIOp>(op))
    return std::pow(a[0], a[1]);
  if (isa<math::Atan2Op>(op))
    return std::atan2(a[0], a[1]);
  double x = a.empty() ? 0.0 : a[0];
  if (isa<math::SqrtOp>(op))
    return std::sqrt(x);
  if (isa<math::ExpOp>(op))
    return std::exp(x);
  if (isa<math::ErfcOp>(op))
    return std::erfc(x);
  if (isa<math::ErfOp>(op))
    return std::erf(x);
  if (isa<math::LogOp>(op))
    return std::log(x);
  if (isa<math::AbsFOp>(op))
    return std::fabs(x);
  if (isa<math::TanhOp>(op))
    return std::tanh(x);
  if (isa<math::SinOp>(op))
    return std::sin(x);
  if (isa<math::CosOp>(op))
    return std::cos(x);
  if (isa<math::TanOp>(op))
    return std::tan(x);
  if (isa<math::AsinOp>(op))
    return std::asin(x);
  if (isa<math::AcosOp>(op))
    return std::acos(x);
  if (isa<math::AtanOp>(op))
    return std::atan(x);
  if (isa<math::SinhOp>(op))
    return std::sinh(x);
  if (isa<math::CoshOp>(op))
    return std::cosh(x);
  return std::nullopt;
}

bool isRadialOp(Operation *op) {
  if (op->getNumResults() != 1 || op->getNumRegions() != 0)
    return false;
  // The operands do not matter for whether the op is known.
  SmallVector<double, 2> zeros(op->getNumOperands(), 1.0);
  return evaluateRadialOp(op, zeros).has_value();
}

namespace {
/// Evaluates the function `function`, (f64) -> f64, at `s`, op by op.
double evaluate(func::FuncOp function, double s) {
  llvm::DenseMap<Value, double> values;
  Block &body = function.getBody().front();
  values[body.getArgument(0)] = s;
  for (Operation &op : body) {
    if (auto ret = dyn_cast<func::ReturnOp>(op))
      return values.lookup(ret.getOperand(0));
    SmallVector<double, 2> operands;
    for (Value operand : op.getOperands())
      operands.push_back(values.lookup(operand));
    std::optional<double> result = evaluateRadialOp(&op, operands);
    if (op.getNumResults() == 1)
      values[op.getResult(0)] = result.value_or(NAN);
  }
  return NAN;
}

float bitsToFloat(uint32_t bits) {
  float value;
  std::memcpy(&value, &bits, sizeof value);
  return value;
}
uint32_t floatToBits(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof bits);
  return bits;
}

/// A table of a function: its first key, its bits, and four coefficients
/// for each interval.
struct Table {
  int bits = 0;
  uint32_t base = 0;
  std::vector<float> coefficients;
  /// The largest error of the polynomials in f64, and in f32 as a kernel
  /// evaluates them, relative to the largest value on each interval.
  double fitError = 0.0;
  double error = 0.0;
};

/// Fits the polynomials of `function` on [low, high] with `bits` bits.
Table fit(func::FuncOp function, double low, double high, int bits) {
  Table table;
  table.bits = bits;
  int shift = 23 - bits;
  table.base = floatToBits(static_cast<float>(low)) >> shift;
  uint32_t last = floatToBits(static_cast<float>(high)) >> shift;
  for (uint32_t key = table.base; key <= last; ++key) {
    double start = bitsToFloat(key << shift);
    double end = bitsToFloat((key + 1) << shift);
    double width = end - start;
    // Chebyshev nodes of the interval, and the cubic through them, in
    // powers of u = s - start (Newton's divided differences).
    double u[4], g[4];
    for (int i = 0; i != 4; ++i) {
      double t = std::cos(M_PI * (2 * i + 1) / 8.0);
      u[i] = 0.5 * width * (1.0 + t);
      g[i] = evaluate(function, start + u[i]);
    }
    double d[4] = {g[0], g[1], g[2], g[3]};
    for (int j = 1; j != 4; ++j)
      for (int i = 3; i >= j; --i)
        d[i] = (d[i] - d[i - 1]) / (u[i] - u[i - j]);
    // d0 + d1 (u - u0) + d2 (u - u0)(u - u1) + d3 (u - u0)(u - u1)(u - u2),
    // by Horner on the Newton form, to powers of u.
    double poly[4] = {d[3], 0, 0, 0};
    int degree = 0;
    for (int i = 2; i >= 0; --i) {
      // poly = poly * (u - u[i]) + d[i]
      double next[4] = {0, 0, 0, 0};
      for (int k = 0; k <= degree; ++k) {
        next[k + 1] += poly[k];
        next[k] -= poly[k] * u[i];
      }
      next[0] += d[i];
      for (int k = 0; k != 4; ++k)
        poly[k] = next[k];
      ++degree;
    }
    // The error at 16 points, relative to the largest value.
    double largest = 0.0, worst = 0.0, worstFit = 0.0;
    for (int i = 0; i <= 16; ++i) {
      double x = width * i / 16.0;
      double exact = evaluate(function, start + x);
      float xf = static_cast<float>(x);
      float p = static_cast<float>(poly[3]);
      for (int k = 2; k >= 0; --k)
        p = std::fma(p, xf, static_cast<float>(poly[k]));
      double q = poly[3];
      for (int k = 2; k >= 0; --k)
        q = q * x + poly[k];
      largest = std::max(largest, std::abs(exact));
      worst = std::max(worst, std::abs(static_cast<double>(p) - exact));
      worstFit = std::max(worstFit, std::abs(q - exact));
    }
    if (largest > 0.0) {
      table.error = std::max(table.error, worst / largest);
      table.fitError = std::max(table.fitError, worstFit / largest);
    }
    for (int k = 0; k != 4; ++k)
      table.coefficients.push_back(static_cast<float>(poly[k]));
  }
  return table;
}

} // namespace

bool canTabulate(Operation *op, double cutoff, double tolerance) {
  auto function = cast<func::FuncOp>(op);
  double high = cutoff * cutoff;
  double low = high * std::ldexp(1.0, -10);
  Table table;
  for (int bits = 4; bits <= 12; ++bits) {
    table = fit(function, low, high, bits);
    if (table.fitError <= 0.1 * tolerance)
      break;
  }
  return llvm::all_of(table.coefficients,
                      [](float c) { return std::isfinite(c); }) &&
         table.error <= tolerance;
}

namespace {

class ExpandRadial : public impl::ExpandRadialBase<ExpandRadial> {
public:
  using impl::ExpandRadialBase<ExpandRadial>::ExpandRadialBase;

  void runOnOperation() final {
    ModuleOp module = getOperation();
    SymbolTable symbols(module);
    SmallVector<RadialOp> radials;
    module.walk([&](RadialOp op) { radials.push_back(op); });
    std::map<std::pair<std::string, double>, memref::GlobalOp> globals;
    std::map<std::pair<std::string, double>, Table> tables;
    for (RadialOp op : radials) {
      auto function = symbols.lookup<func::FuncOp>(op.getFunction());
      if (!function) {
        op.emitOpError() << "has no function " << op.getFunction();
        return signalPassFailure();
      }
      OpBuilder builder(op);
      Location loc = op.getLoc();
      // A function that a table cannot hold within the tolerance, such as
      // one with a pole at the cutoff or that vanishes there with all its
      // derivatives (the factor exp(sigma / (r - a sigma)) of the
      // Stillinger-Weber form), is evaluated as itself in f64 (D159).
      bool asItself = !op.getType().isF32();
      if (!asItself)
        if (auto loop = op->getParentOfType<PairForOp>())
          asItself = !canTabulate(function, loop.getCutoff().convertToDouble(),
                                  tolerance);
      if (asItself) {
        // The function itself, in f64 or the type of the kernel.
        IRMapping values;
        Block &body = function.getBody().front();
        Value s = op.getR2();
        if (s.getType() != body.getArgument(0).getType())
          s = arith::ExtFOp::create(builder, loc, builder.getF64Type(), s);
        values.map(body.getArgument(0), s);
        for (Operation &nested : body.without_terminator())
          builder.clone(nested, values);
        Value result = values.lookup(body.getTerminator()->getOperand(0));
        if (result.getType() != op.getType())
          result = arith::TruncFOp::create(builder, loc, op.getType(), result);
        op.replaceAllUsesWith(result);
        op.erase();
        continue;
      }
      auto loop = op->getParentOfType<PairForOp>();
      if (!loop) {
        op.emitOpError() << "is not in the kernel of a loop over pairs";
        return signalPassFailure();
      }
      double cutoff = loop.getCutoff().convertToDouble();
      double high = cutoff * cutoff;
      double low = high * std::ldexp(1.0, -10);
      auto key = std::make_pair(op.getFunction().str(), high);
      memref::GlobalOp global = globals[key];
      if (!global) {
        // The fewest bits whose polynomials are within a tenth of the
        // tolerance in f64, so that the rounding of f32 dominates.
        Table table;
        for (int bits = 4; bits <= 12; ++bits) {
          table = fit(function, low, high, bits);
          if (table.fitError <= 0.1 * tolerance)
            break;
        }
        // A value that the evaluation does not know is NaN, and would
        // pass the test of the error below unseen.
        if (!llvm::all_of(table.coefficients,
                          [](float c) { return std::isfinite(c); })) {
          op.emitOpError() << "cannot tabulate " << op.getFunction()
                           << ": it is not finite on (" << low << ", "
                           << high << "]";
          return signalPassFailure();
        }
        if (!(table.error <= tolerance)) {
          op.emitOpError() << "cannot tabulate " << op.getFunction()
                           << " within " << tolerance << ": "
                           << table.error << " with " << table.bits
                           << " bits";
          return signalPassFailure();
        }
        // A table that another function gave already serves this one.
        for (auto &[otherKey, other] : tables)
          if (other.bits == table.bits && other.base == table.base &&
              other.coefficients == table.coefficients) {
            global = globals[otherKey];
            break;
          }
        if (!global) {
          int64_t count = table.coefficients.size() / 4;
          auto type = MemRefType::get({count, 4}, builder.getF32Type());
          auto data = DenseElementsAttr::get(
              RankedTensorType::get({count, 4}, builder.getF32Type()),
              ArrayRef<float>(table.coefficients));
          OpBuilder top(module.getBodyRegion());
          global = memref::GlobalOp::create(
              top, loc, (op.getFunction() + "_table").str(),
              top.getStringAttr("private"), type, data, /*constant=*/true,
              top.getI64IntegerAttr(16));
          symbols.insert(global);
        }
        globals[key] = global;
        tables[key] = table;
      }
      const Table &table = tables[key];
      int shift = 23 - table.bits;
      int64_t count = table.coefficients.size() / 4;
      Type i32 = builder.getI32Type();
      auto constant = [&](int64_t v) -> Value {
        return arith::ConstantOp::create(builder, loc, i32,
                                         builder.getI32IntegerAttr(v));
      };
      Value s = op.getR2();
      Value bits = arith::BitcastOp::create(builder, loc, i32, s);
      Value key32 = arith::ShRUIOp::create(builder, loc, bits, constant(shift));
      Value k0 = arith::SubIOp::create(builder, loc, key32,
                                       constant(table.base));
      Value k = arith::MinSIOp::create(
          builder, loc, arith::MaxSIOp::create(builder, loc, k0, constant(0)),
          constant(count - 1));
      Value startBits = arith::ShLIOp::create(
          builder, loc,
          arith::AddIOp::create(builder, loc, k, constant(table.base)),
          constant(shift));
      Value start = arith::BitcastOp::create(builder, loc,
                                             builder.getF32Type(), startBits);
      Value u = arith::SubFOp::create(builder, loc, s, start);
      Value index =
          arith::IndexCastOp::create(builder, loc, builder.getIndexType(), k);
      Value memory = memref::GetGlobalOp::create(
          builder, loc, global.getType(), global.getSymName());
      Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
      Value c = vector::LoadOp::create(
          builder, loc, VectorType::get({4}, builder.getF32Type()), memory,
          ValueRange{index, zero});
      Value p = vector::ExtractOp::create(builder, loc, c, 3);
      for (int64_t j = 2; j >= 0; --j)
        p = math::FmaOp::create(builder, loc, p, u,
                                vector::ExtractOp::create(builder, loc, c, j));
      op.replaceAllUsesWith(p);
      op.erase();
    }
    // The functions that no op uses any more.
    for (func::FuncOp function :
         llvm::make_early_inc_range(module.getOps<func::FuncOp>()))
      if (function.getSymName().starts_with("md_radial") &&
          !function.getSymName().ends_with("_table") &&
          SymbolTable::symbolKnownUseEmpty(function, module))
        function.erase();
  }
};
} // namespace

} // namespace md_exec
} // namespace mdir
