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
#include <cstring>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

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
/// The function `function`, (f64) -> f64, as a list of steps over numbered
/// values, which gives its value at a point without a look at its ops. Each
/// step is the one operation that evaluateRadialOp has for its op, on the
/// same values, so the value has the same bits as an evaluation op by op.
class Evaluator {
public:
  explicit Evaluator(func::FuncOp function) {
    // Value 0 is the zero that an operand without a value reads; value 1
    // is the argument.
    llvm::DenseMap<Value, uint32_t> numbers;
    Block &body = function.getBody().front();
    numbers[body.getArgument(0)] = 1;
    auto numberOf = [&](Value value) { return numbers.lookup(value); };
    for (Operation &op : body) {
      if (auto ret = dyn_cast<func::ReturnOp>(op)) {
        result = numberOf(ret.getOperand(0));
        returns = true;
        break;
      }
      // The steps have no effects: one whose value is not kept is left out.
      if (op.getNumResults() != 1)
        continue;
      Step step;
      step.kind = getKind(&op, step.constant);
      if (op.getNumOperands() > 0)
        step.first = numberOf(op.getOperand(0));
      if (op.getNumOperands() > 1)
        step.second = numberOf(op.getOperand(1));
      step.target = numValues++;
      numbers[op.getResult(0)] = step.target;
      steps.push_back(step);
    }
  }

  /// The value of the function at `s`.
  double operator()(double s) const {
    if (!returns)
      return NAN;
    SmallVector<double, 64> values(numValues, 0.0);
    values[1] = s;
    for (const Step &step : steps) {
      double a = values[step.first], b = values[step.second];
      double value = NAN;
      switch (step.kind) {
      case Kind::Constant: value = step.constant; break;
      case Kind::Add: value = a + b; break;
      case Kind::Sub: value = a - b; break;
      case Kind::Mul: value = a * b; break;
      case Kind::Div: value = a / b; break;
      case Kind::Neg: value = -a; break;
      case Kind::Pow: value = std::pow(a, b); break;
      case Kind::Atan2: value = std::atan2(a, b); break;
      case Kind::Sqrt: value = std::sqrt(a); break;
      case Kind::Exp: value = std::exp(a); break;
      case Kind::Erfc: value = std::erfc(a); break;
      case Kind::Erf: value = std::erf(a); break;
      case Kind::Log: value = std::log(a); break;
      case Kind::Abs: value = std::fabs(a); break;
      case Kind::Tanh: value = std::tanh(a); break;
      case Kind::Sin: value = std::sin(a); break;
      case Kind::Cos: value = std::cos(a); break;
      case Kind::Tan: value = std::tan(a); break;
      case Kind::Asin: value = std::asin(a); break;
      case Kind::Acos: value = std::acos(a); break;
      case Kind::Atan: value = std::atan(a); break;
      case Kind::Sinh: value = std::sinh(a); break;
      case Kind::Cosh: value = std::cosh(a); break;
      case Kind::Unknown: break;
      }
      values[step.target] = value;
    }
    return values[result];
  }

  /// The steps as bytes: functions with the same bytes have the same value
  /// at every point.
  void appendTo(std::string &key) const {
    auto append = [&](const void *data, size_t size) {
      key.append(static_cast<const char *>(data), size);
    };
    uint32_t header[3] = {static_cast<uint32_t>(steps.size()), result,
                          returns ? 1u : 0u};
    append(header, sizeof header);
    for (const Step &step : steps) {
      uint32_t fields[4] = {static_cast<uint32_t>(step.kind), step.first,
                            step.second, step.target};
      append(fields, sizeof fields);
      append(&step.constant, sizeof step.constant);
    }
  }

private:
  enum class Kind : uint8_t {
    Constant, Add, Sub, Mul, Div, Neg, Pow, Atan2, Sqrt, Exp, Erfc, Erf, Log,
    Abs, Tanh, Sin, Cos, Tan, Asin, Acos, Atan, Sinh, Cosh, Unknown
  };
  struct Step {
    Kind kind = Kind::Unknown;
    uint32_t first = 0, second = 0, target = 0;
    double constant = 0.0;
  };

  /// The kind of `op`, as evaluateRadialOp tells the ops apart, and the
  /// value of a constant.
  static Kind getKind(Operation *op, double &value) {
    if (auto constant = dyn_cast<arith::ConstantOp>(op)) {
      if (auto real = dyn_cast<FloatAttr>(constant.getValue())) {
        value = real.getValueAsDouble();
        return Kind::Constant;
      }
      if (auto integer = dyn_cast<IntegerAttr>(constant.getValue())) {
        value = static_cast<double>(integer.getInt());
        return Kind::Constant;
      }
      return Kind::Unknown;
    }
    if (isa<arith::AddFOp>(op)) return Kind::Add;
    if (isa<arith::SubFOp>(op)) return Kind::Sub;
    if (isa<arith::MulFOp>(op)) return Kind::Mul;
    if (isa<arith::DivFOp>(op)) return Kind::Div;
    if (isa<arith::NegFOp>(op)) return Kind::Neg;
    if (isa<math::PowFOp, math::FPowIOp>(op)) return Kind::Pow;
    if (isa<math::Atan2Op>(op)) return Kind::Atan2;
    if (isa<math::SqrtOp>(op)) return Kind::Sqrt;
    if (isa<math::ExpOp>(op)) return Kind::Exp;
    if (isa<math::ErfcOp>(op)) return Kind::Erfc;
    if (isa<math::ErfOp>(op)) return Kind::Erf;
    if (isa<math::LogOp>(op)) return Kind::Log;
    if (isa<math::AbsFOp>(op)) return Kind::Abs;
    if (isa<math::TanhOp>(op)) return Kind::Tanh;
    if (isa<math::SinOp>(op)) return Kind::Sin;
    if (isa<math::CosOp>(op)) return Kind::Cos;
    if (isa<math::TanOp>(op)) return Kind::Tan;
    if (isa<math::AsinOp>(op)) return Kind::Asin;
    if (isa<math::AcosOp>(op)) return Kind::Acos;
    if (isa<math::AtanOp>(op)) return Kind::Atan;
    if (isa<math::SinhOp>(op)) return Kind::Sinh;
    if (isa<math::CoshOp>(op)) return Kind::Cosh;
    return Kind::Unknown;
  }

  std::vector<Step> steps;
  uint32_t numValues = 2, result = 0;
  bool returns = false;
};

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
  /// Whether every coefficient is finite, and whether the table holds the
  /// function within the tolerance that it was searched with.
  bool finite = false;
  bool fits = false;
};

/// Fits the polynomials of the function of `evaluate` on [low, high] with
/// `bits` bits.
Table fit(const Evaluator &evaluate, double low, double high, int bits) {
  Table table;
  table.bits = bits;
  int shift = 23 - bits;
  table.base = floatToBits(static_cast<float>(low)) >> shift;
  uint32_t last = floatToBits(static_cast<float>(high)) >> shift;
  table.coefficients.reserve(4 * (size_t(last) - table.base + 1));
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
      g[i] = evaluate(start + u[i]);
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
      double exact = evaluate(start + x);
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

/// The tables that the process has searched for, by the steps of the
/// function, the cutoff squared, and the tolerance. md-exec-simplify-distance
/// asks whether a function fits a table, md-exec-expand-radial asks again
/// and then takes the table, and a process that lowers a program again asks
/// for the same tables each time; a search takes tenths of a second. A
/// table follows from its key alone, so the lowerings of several threads
/// share the tables under a mutex. The oldest tables leave when the
/// coefficients that are kept pass `bound` bytes.
class Tables {
public:
  std::shared_ptr<const Table> find(const std::string &key) {
    std::lock_guard<std::mutex> lock(mutex);
    auto found = tables.find(key);
    return found == tables.end() ? nullptr : found->second;
  }

  /// Keeps `table` under `key`, or returns the table that another thread
  /// kept under it meanwhile, which is the same.
  std::shared_ptr<const Table> insert(const std::string &key, Table table) {
    std::lock_guard<std::mutex> lock(mutex);
    auto [found, inserted] = tables.try_emplace(key);
    if (!inserted)
      return found->second;
    auto kept = std::make_shared<const Table>(std::move(table));
    found->second = kept;
    order.push_back(key);
    bytes += getBytes(key, *kept);
    // The table just kept stays, whatever its size.
    while (bytes > bound && order.size() > 1) {
      auto oldest = tables.find(order.front());
      bytes -= getBytes(oldest->first, *oldest->second);
      tables.erase(oldest);
      order.pop_front();
    }
    return kept;
  }

private:
  static size_t getBytes(const std::string &key, const Table &table) {
    return 2 * key.size() + sizeof(Table) +
           table.coefficients.size() * sizeof(float);
  }

  /// A table of 12 bits is 0.6 MiB; those of a program are a few.
  static constexpr size_t bound = size_t(16) << 20;
  std::mutex mutex;
  std::unordered_map<std::string, std::shared_ptr<const Table>> tables;
  std::deque<std::string> order;
  size_t bytes = 0;
};

Tables &getTables() {
  static Tables tables;
  return tables;
}

/// The table of `function` up to `cutoff`: that of the fewest bits, from 4
/// to 12, whose polynomials are within a tenth of `tolerance` in f64, so
/// that the rounding of f32 dominates, or else that of 12 bits. A table
/// that does not hold the function (`fits`) keeps no coefficients.
std::shared_ptr<const Table> search(func::FuncOp function, double cutoff,
                                    double tolerance) {
  Evaluator evaluate(function);
  double high = cutoff * cutoff;
  double low = high * std::ldexp(1.0, -10);
  std::string key;
  key.append(reinterpret_cast<const char *>(&high), sizeof high);
  key.append(reinterpret_cast<const char *>(&tolerance), sizeof tolerance);
  evaluate.appendTo(key);
  if (std::shared_ptr<const Table> known = getTables().find(key))
    return known;
  Table table;
  for (int bits = 4; bits <= 12; ++bits) {
    table = fit(evaluate, low, high, bits);
    if (table.fitError <= 0.1 * tolerance)
      break;
  }
  // A value that the evaluation does not know is NaN, and would pass the
  // test of the error unseen.
  table.finite = llvm::all_of(table.coefficients,
                              [](float c) { return std::isfinite(c); });
  table.fits = table.finite && table.error <= tolerance;
  if (!table.fits)
    std::vector<float>().swap(table.coefficients);
  return getTables().insert(key, std::move(table));
}

} // namespace

bool canTabulate(Operation *op, double cutoff, double tolerance) {
  return search(cast<func::FuncOp>(op), cutoff, tolerance)->fits;
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
    std::map<std::pair<std::string, double>, std::shared_ptr<const Table>>
        tables;
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
        // The table that the test above searched for.
        std::shared_ptr<const Table> found =
            search(function, cutoff, tolerance);
        const Table &table = *found;
        if (!table.finite) {
          op.emitOpError() << "cannot tabulate " << op.getFunction()
                           << ": it is not finite on (" << low << ", "
                           << high << "]";
          return signalPassFailure();
        }
        if (!table.fits) {
          op.emitOpError() << "cannot tabulate " << op.getFunction()
                           << " within " << tolerance << ": "
                           << table.error << " with " << table.bits
                           << " bits";
          return signalPassFailure();
        }
        // A table that another function gave already serves this one.
        for (auto &[otherKey, other] : tables)
          if (other == found ||
              (other->bits == table.bits && other->base == table.base &&
               other->coefficients == table.coefficients)) {
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
        tables[key] = found;
      }
      const Table &table = *tables[key];
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
