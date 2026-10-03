// Energy expressions.

#include "mdir/Driver/Expression.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/StringSwitch.h"

#include <functional>

#include <cmath>
#include <cstdio>
#include <cstdlib>

using namespace mdir::driver;
using llvm::StringRef;

struct Expression::Node {
  enum Kind { Number, Name, Negate, Add, Subtract, Multiply, Divide, Power,
              Call };
  Kind kind;
  double number = 0.0;
  /// The name of a value, or of a function.
  std::string name;
  std::unique_ptr<Node> lhs;
  std::unique_ptr<Node> rhs;
  /// The arguments of a function.
  std::vector<std::unique_ptr<Node>> arguments;
};

Expression::Expression() = default;
Expression::Expression(Expression &&) = default;
Expression &Expression::operator=(Expression &&) = default;
Expression::~Expression() = default;

std::string mdir::driver::formatReal(double value) {
  char buffer[64];
  std::snprintf(buffer, sizeof(buffer), "%.17e", value);
  return buffer;
}

/// The op that computes the function `name` of one argument, or an empty
/// string.
static StringRef getFunctionOp(StringRef name) {
  return llvm::StringSwitch<StringRef>(name)
      .Case("sqrt", "math.sqrt")
      .Case("exp", "math.exp")
      .Case("log", "math.log")
      .Case("sin", "math.sin")
      .Case("cos", "math.cos")
      .Case("tan", "math.tan")
      .Case("asin", "math.asin")
      .Case("acos", "math.acos")
      .Case("atan", "math.atan")
      .Case("sinh", "math.sinh")
      .Case("cosh", "math.cosh")
      .Case("tanh", "math.tanh")
      .Case("abs", "math.absf")
      .Case("floor", "math.floor")
      .Case("ceil", "math.ceil")
      .Case("erf", "math.erf")
      .Case("erfc", "math.erfc")
      .Default("");
}

/// The number of arguments of the function `name`, or 0 if there is no
/// such function. Those of the custom forces of OpenMM [Eastman2017] (D22).
static unsigned getArity(StringRef name) {
  if (!getFunctionOp(name).empty())
    return 1;
  return llvm::StringSwitch<unsigned>(name)
      .Cases({"step", "delta", "square", "cube", "recip", "sec", "csc", "cot"},
             1)
      .Cases({"min", "max", "atan2"}, 2)
      .Case("select", 3)
      .Default(0);
}

bool Expression::isFunction(StringRef name) { return getArity(name) != 0; }

/// Solves m[i-1] + 4 m[i] + m[i+1] = right[i] for i in [0, n), with
/// m[-1] = m[n] = 0, by elimination.
static std::vector<double> solveTridiagonal(std::vector<double> right) {
  size_t n = right.size();
  std::vector<double> diagonal(n, 4.0), m(n, 0.0);
  for (size_t i = 1; i < n; ++i) {
    double factor = 1.0 / diagonal[i - 1];
    diagonal[i] -= factor;
    right[i] -= factor * right[i - 1];
  }
  for (size_t i = n; i-- > 0;)
    m[i] = (right[i] - (i + 1 < n ? m[i + 1] : 0.0)) / diagonal[i];
  return m;
}

/// The second derivatives m of the cubic spline through `values` at the
/// spacing h, from
///   m[i-1] + 4 m[i] + m[i+1] = 6 (y[i-1] - 2 y[i] + y[i+1]) / h²:
/// natural, m = 0 at both ends, or periodic, the indices of the n - 1
/// intervals taken modulo n - 1, a cyclic system solved by the formula of
/// Sherman and Morrison from two tridiagonal ones.
static std::vector<double> getCurvatures(llvm::ArrayRef<double> values,
                                         double h, bool periodic) {
  size_t n = values.size();
  auto curvature = [&](size_t before, size_t i, size_t after) {
    return 6.0 * (values[before] - 2.0 * values[i] + values[after]) / (h * h);
  };
  std::vector<double> m(n, 0.0);
  if (!periodic) {
    std::vector<double> right;
    for (size_t i = 1; i + 1 < n; ++i)
      right.push_back(curvature(i - 1, i, i + 1));
    std::vector<double> inner = solveTridiagonal(right);
    std::copy(inner.begin(), inner.end(), m.begin() + 1);
    return m;
  }
  // A = T + u vᵀ with T tridiagonal, u = (γ, 0, ..., 0, 1), and
  // v = (1, 0, ..., 0, 1/γ), γ = -4, so that the corners of A are 1 and
  // the first and last diagonal of T are 4 - γ and 4 - 1/γ.
  size_t count = n - 1;
  std::vector<double> right(count);
  for (size_t i = 0; i != count; ++i)
    right[i] = curvature((i + count - 1) % count, i, i + 1);
  double gamma = -4.0;
  auto solve = [&](std::vector<double> b) {
    // T x = b, with the first and last diagonal changed.
    std::vector<double> diagonal(count, 4.0), x(count, 0.0);
    diagonal.front() -= gamma;
    diagonal.back() -= 1.0 / gamma;
    for (size_t i = 1; i < count; ++i) {
      double factor = 1.0 / diagonal[i - 1];
      diagonal[i] -= factor;
      b[i] -= factor * b[i - 1];
    }
    for (size_t i = count; i-- > 0;)
      x[i] = (b[i] - (i + 1 < count ? x[i + 1] : 0.0)) / diagonal[i];
    return x;
  };
  std::vector<double> u(count, 0.0);
  u.front() = gamma;
  u.back() = 1.0;
  std::vector<double> x = solve(right), z = solve(u);
  double factor = (x.front() + x.back() / gamma) /
                  (1.0 + z.front() + z.back() / gamma);
  for (size_t i = 0; i != count; ++i)
    m[i] = x[i] - factor * z[i];
  m[count] = m[0];
  return m;
}

/// The derivative of the spline of getCurvatures at each of its points:
/// that of the interval that begins there, and of the last interval at the
/// last point.
static std::vector<double> getSlopes(llvm::ArrayRef<double> values, double h,
                                     bool periodic) {
  std::vector<double> m = getCurvatures(values, h, periodic);
  size_t n = values.size();
  std::vector<double> slopes(n);
  for (size_t i = 0; i + 1 < n; ++i)
    slopes[i] = (values[i + 1] - values[i]) / h - h * (2.0 * m[i] + m[i + 1]) / 6.0;
  slopes[n - 1] = (values[n - 1] - values[n - 2]) / h +
                  h * (m[n - 2] + 2.0 * m[n - 1]) / 6.0;
  return slopes;
}

std::vector<double> TabulatedFunction::getCoefficients() const {
  if (discrete)
    return values;
  unsigned dimensions = getNumArguments();
  std::vector<double> h(dimensions);
  for (unsigned k = 0; k != dimensions; ++k)
    h[k] = (maxs[k] - mins[k]) / static_cast<double>(sizes[k] - 1);
  if (dimensions == 1) {
    // On [x_i, x_i + h], with t = (x - x_i) / h,
    //   y = (1 - t) y_i + t y_i+1 + h²/6 (((1 - t)³ - (1 - t)) m_i + (t³ - t) m_i+1).
    std::vector<double> m = getCurvatures(values, h[0], periodic);
    std::vector<double> coefficients;
    for (size_t i = 0; i + 1 < values.size(); ++i) {
      double a = h[0] * h[0] / 6.0;
      coefficients.push_back(values[i]);
      coefficients.push_back(values[i + 1] - values[i] -
                             a * (2.0 * m[i] + m[i + 1]));
      coefficients.push_back(3.0 * a * m[i]);
      coefficients.push_back(a * (m[i + 1] - m[i]));
    }
    return coefficients;
  }

  // The derivatives at the points of the grid along each set of axes, a
  // bit of `mask` for each, as the continuous functions of two and three
  // arguments of OpenMM take them [Eastman2017]: along one axis from the
  // spline through the values on each line of the grid, and along more
  // than one from the spline through those along fewer, in the order
  // below, which the result depends on at the level of the error of the
  // splines.
  std::vector<unsigned> stride(dimensions, 1);
  for (unsigned k = 1; k != dimensions; ++k)
    stride[k] = stride[k - 1] * sizes[k - 1];
  size_t total = values.size();
  std::vector<std::vector<double>> derivatives(1u << dimensions);
  derivatives[0] = values;
  auto differentiate = [&](unsigned mask, unsigned axis, unsigned from) {
    std::vector<double> &result = derivatives[mask];
    result.assign(total, 0.0);
    const std::vector<double> &source = derivatives[from];
    for (size_t start = 0; start != total; ++start) {
      if ((start / stride[axis]) % sizes[axis] != 0)
        continue;
      std::vector<double> line(sizes[axis]);
      for (unsigned i = 0; i != sizes[axis]; ++i)
        line[i] = source[start + i * stride[axis]];
      std::vector<double> slopes = getSlopes(line, h[axis], periodic);
      for (unsigned i = 0; i != sizes[axis]; ++i)
        result[start + i * stride[axis]] = slopes[i];
    }
  };
  // (mask, axis, from): ∂x, ∂y, ∂z, ∂x∂y = x of ∂y, ∂x∂z = z of ∂x,
  // ∂y∂z = y of ∂z, ∂x∂y∂z = x of ∂y∂z.
  static const unsigned order[][3] = {{1, 0, 0}, {2, 1, 0}, {4, 2, 0},
                                      {3, 0, 2}, {5, 2, 1}, {6, 1, 4},
                                      {7, 0, 6}};
  for (const auto &[mask, axis, from] : order)
    if (mask < derivatives.size())
      differentiate(mask, axis, from);

  // The patch of each cell: the product of the cubics of Hermite along the
  // axes, which match the values, the derivatives along each axis, and the
  // mixed ones at the corners, the bicubic of Press et al. and the
  // tricubic of Lekien and Marsden [Press2007, Lekien2005]. Along an axis,
  // the cubic of p0, p1 and of the derivatives d0, d1 times the spacing
  // has the coefficients (p0, d0, 3 (p1 - p0) - 2 d0 - d1,
  // 2 (p0 - p1) + d0 + d1); `hermite[a][q]` is the factor of the q-th of
  // (p0, p1, d0, d1) in the coefficient of the power a.
  static const double hermite[4][4] = {
      {1, 0, 0, 0}, {0, 0, 1, 0}, {-3, 3, -2, -1}, {2, -2, 1, 1}};
  unsigned columns = getColumns();
  std::vector<unsigned> cells(dimensions);
  size_t numCells = 1;
  for (unsigned k = 0; k != dimensions; ++k) {
    cells[k] = sizes[k] - 1;
    numCells *= cells[k];
  }
  std::vector<double> coefficients(numCells * columns, 0.0);
  for (size_t cell = 0; cell != numCells; ++cell) {
    std::vector<unsigned> corner(dimensions);
    for (size_t rest = cell, k = 0; k != dimensions; ++k) {
      corner[k] = rest % cells[k];
      rest /= cells[k];
    }
    // q[k] in 0..3 picks along axis k p0, p1, d0, or d1.
    for (unsigned column = 0; column != columns; ++column) {
      double sum = 0.0;
      for (unsigned qs = 0; qs != columns; ++qs) {
        double factor = 1.0;
        unsigned mask = 0;
        size_t point = 0;
        for (unsigned k = 0; k != dimensions; ++k) {
          unsigned a = (column >> (2 * k)) & 3, q = (qs >> (2 * k)) & 3;
          factor *= hermite[a][q];
          if (q >= 2) {
            mask |= 1u << k;
            factor *= h[k];
          }
          point += (corner[k] + (q & 1)) * stride[k];
        }
        if (factor != 0.0)
          sum += factor * derivatives[mask][point];
      }
      coefficients[cell * columns + column] = sum;
    }
  }
  return coefficients;
}

double Expression::Spline::evaluate(llvm::ArrayRef<double> arguments) const {
  unsigned dimensions = getNumArguments();
  std::vector<double> t(dimensions);
  size_t row = 0, stride = 1;
  for (unsigned k = 0; k != dimensions; ++k) {
    double x = arguments[k];
    double place;
    if (discrete) {
      place = std::min(std::max(std::floor(x + 0.5), 0.0),
                       static_cast<double>(counts[k] - 1));
    } else {
      if (periodic)
        x -= (maxs[k] - mins[k]) * std::floor((x - mins[k]) / (maxs[k] - mins[k]));
      else if (x < mins[k] || x > maxs[k])
        return 0.0;
      double u = (x - mins[k]) * scales[k];
      place = std::min(std::max(std::floor(u), 0.0),
                       static_cast<double>(counts[k] - 1));
      t[k] = u - place;
    }
    row += static_cast<size_t>(place) * stride;
    stride *= counts[k];
  }
  unsigned columns = getColumns();
  const double *c = &coefficients[row * columns];
  if (discrete)
    return c[0];
  // Horner's rule along the last argument first.
  std::function<double(unsigned, unsigned)> sum = [&](unsigned k,
                                                      unsigned offset) {
    double p = 0.0;
    for (int a = 3; a >= 0; --a)
      p = p * t[k] + (k == 0 ? c[offset + a]
                             : sum(k - 1, offset + (a << (2 * k))));
    return p;
  };
  return sum(dimensions - 1, 0);
}

namespace {

class Parser {
public:
  Parser(StringRef text, std::vector<std::string> &names,
         llvm::ArrayRef<TabulatedFunction> functions,
         std::vector<std::string> &called)
      : text(text), rest(text), names(names), functions(functions),
        called(called) {}

  llvm::Expected<std::unique_ptr<Expression::Node>> parse() {
    auto node = parseSum();
    if (!node)
      return node;
    skipSpace();
    if (!rest.empty())
      return fail("unexpected '" + rest.take_front(1) + "'");
    return node;
  }

private:
  using Result = llvm::Expected<std::unique_ptr<Expression::Node>>;

  llvm::Error fail(const llvm::Twine &message) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(), "in the expression '%s', at %u: %s",
        text.str().c_str(),
        static_cast<unsigned>(text.size() - rest.size() + 1),
        message.str().c_str());
  }

  void skipSpace() { rest = rest.ltrim(); }

  bool consume(char c) {
    skipSpace();
    if (rest.empty() || rest.front() != c)
      return false;
    rest = rest.drop_front();
    return true;
  }

  static std::unique_ptr<Expression::Node>
  create(Expression::Node::Kind kind, std::unique_ptr<Expression::Node> lhs,
         std::unique_ptr<Expression::Node> rhs = nullptr) {
    auto node = std::make_unique<Expression::Node>();
    node->kind = kind;
    node->lhs = std::move(lhs);
    node->rhs = std::move(rhs);
    return node;
  }

  Result parseSum() {
    auto first = parseProduct();
    if (!first)
      return first;
    std::unique_ptr<Expression::Node> lhs = std::move(*first);
    while (true) {
      Expression::Node::Kind kind;
      if (consume('+'))
        kind = Expression::Node::Add;
      else if (consume('-'))
        kind = Expression::Node::Subtract;
      else
        return std::move(lhs);
      auto rhs = parseProduct();
      if (!rhs)
        return rhs;
      lhs = create(kind, std::move(lhs), std::move(*rhs));
    }
  }

  Result parseProduct() {
    auto first = parseUnary();
    if (!first)
      return first;
    std::unique_ptr<Expression::Node> lhs = std::move(*first);
    while (true) {
      Expression::Node::Kind kind;
      if (consume('*'))
        kind = Expression::Node::Multiply;
      else if (consume('/'))
        kind = Expression::Node::Divide;
      else
        return std::move(lhs);
      auto rhs = parseUnary();
      if (!rhs)
        return rhs;
      lhs = create(kind, std::move(lhs), std::move(*rhs));
    }
  }

  Result parseUnary() {
    if (consume('-')) {
      auto operand = parseUnary();
      if (!operand)
        return operand;
      return create(Expression::Node::Negate, std::move(*operand));
    }
    if (consume('+'))
      return parseUnary();
    return parsePower();
  }

  // A power binds more tightly than a sign before it and takes a sign after
  // it: -a^-b is -(a^(-b)).
  Result parsePower() {
    auto base = parsePrimary();
    if (!base)
      return base;
    if (!consume('^'))
      return base;
    auto exponent = parseUnary();
    if (!exponent)
      return exponent;
    return create(Expression::Node::Power, std::move(*base),
                  std::move(*exponent));
  }

  Result parsePrimary() {
    skipSpace();
    if (rest.empty())
      return fail("expected a number, a name, or '('");

    if (consume('(')) {
      auto inner = parseSum();
      if (!inner)
        return inner;
      if (!consume(')'))
        return fail("expected ')'");
      return inner;
    }

    char first = rest.front();
    if (llvm::isDigit(first) || first == '.') {
      std::string digits = rest.str();
      char *end = nullptr;
      double value = std::strtod(digits.c_str(), &end);
      if (end == digits.c_str())
        return fail("expected a number");
      rest = rest.drop_front(end - digits.c_str());
      auto node = std::make_unique<Expression::Node>();
      node->kind = Expression::Node::Number;
      node->number = value;
      return std::move(node);
    }

    if (llvm::isAlpha(first) || first == '_') {
      StringRef name = rest.take_while(
          [](char c) { return llvm::isAlnum(c) || c == '_'; });
      rest = rest.drop_front(name.size());
      if (consume('(')) {
        unsigned arity = getArity(name);
        const TabulatedFunction *tabulated = nullptr;
        for (const TabulatedFunction &function : functions)
          if (function.name == name)
            tabulated = &function;
        if (arity == 0 && tabulated) {
          arity = tabulated->getNumArguments();
          if (!llvm::is_contained(called, name))
            called.push_back(name.str());
        }
        if (arity == 0)
          return fail("unknown function '" + name + "'");
        auto node = std::make_unique<Expression::Node>();
        node->kind = Expression::Node::Call;
        node->name = name.str();
        do {
          auto argument = parseSum();
          if (!argument)
            return argument;
          node->arguments.push_back(std::move(*argument));
        } while (consume(','));
        if (!consume(')'))
          return fail("expected ')' after the arguments of '" + name + "'");
        if (node->arguments.size() != arity)
          return fail("'" + name + "' takes " + llvm::Twine(arity) +
                      " argument" + (arity == 1 ? "" : "s") + ", not " +
                      llvm::Twine(node->arguments.size()));
        return std::move(node);
      }
      auto node = std::make_unique<Expression::Node>();
      node->kind = Expression::Node::Name;
      node->name = name.str();
      if (!llvm::is_contained(names, node->name))
        names.push_back(node->name);
      return std::move(node);
    }
    return fail("unexpected '" + rest.take_front(1) + "'");
  }

  StringRef text;
  StringRef rest;
  std::vector<std::string> &names;
  llvm::ArrayRef<TabulatedFunction> functions;
  /// The tabulated functions that the text calls.
  std::vector<std::string> &called;
};

class Emitter {
public:
  Emitter(llvm::raw_ostream &os, const llvm::StringMap<std::string> &values,
          const llvm::StringMap<const Expression::Node *> &definitions,
          llvm::ArrayRef<Expression::Spline> splines, StringRef prefix,
          StringRef indent)
      : os(os), values(values), definitions(definitions), splines(splines),
        prefix(prefix), indent(indent) {}

  std::string emit(const Expression::Node &node) {
    using Node = Expression::Node;
    switch (node.kind) {
    case Node::Number: {
      std::string result = next();
      os << indent << result << " = arith.constant "
         << formatReal(node.number) << " : f64\n";
      return result;
    }
    case Node::Name: {
      if (values.count(node.name))
        return values.lookup(node.name);
      // A name that the expression defines after a semicolon, computed once.
      auto known = defined.find(node.name);
      if (known != defined.end())
        return known->second;
      std::string value = emit(*definitions.lookup(node.name));
      defined[node.name] = value;
      return value;
    }
    case Node::Negate:
      return emitOp("arith.negf", {emit(*node.lhs)});
    case Node::Add:
      return emitBinary("arith.addf", node);
    case Node::Subtract:
      return emitBinary("arith.subf", node);
    case Node::Multiply:
      return emitBinary("arith.mulf", node);
    case Node::Divide:
      return emitBinary("arith.divf", node);
    case Node::Call:
      return emitCall(node);
    case Node::Power: {
      std::string base = emit(*node.lhs);
      // A power with a whole number is a product.
      double exponent;
      if (isWholeNumber(*node.rhs, exponent)) {
        std::string count = next();
        os << indent << count << " = arith.constant "
           << static_cast<long long>(exponent) << " : i32\n";
        std::string result = next();
        os << indent << result << " = math.fpowi " << base << ", " << count
           << " : f64, i32\n";
        return result;
      }
      return emitOp("math.powf", {base, emit(*node.rhs)});
    }
    }
    return "";
  }

private:
  std::string next() { return (prefix + llvm::Twine(counter++)).str(); }

  std::string constant(double value) {
    std::string result = next();
    os << indent << result << " = arith.constant " << formatReal(value)
       << " : f64\n";
    return result;
  }

  /// `select` of two values by a comparison of `x` with 0.
  std::string compareWithZero(StringRef predicate, const std::string &x,
                              const std::string &onTrue,
                              const std::string &onFalse) {
    std::string zero = constant(0.0), condition = next();
    os << indent << condition << " = arith.cmpf " << predicate << ", " << x
       << ", " << zero << " : f64\n";
    std::string result = next();
    os << indent << result << " = arith.select " << condition << ", "
       << onTrue << ", " << onFalse << " : f64\n";
    return result;
  }

  /// A tabulated function at the arguments `x`: the polynomial of the cell
  /// of x, whose index along each argument is clamped so that the lookup
  /// stays in the table, and zero outside the range; or the value at the
  /// nearest point. The places in the cell carry the derivative; the
  /// indices, the lookups, and the comparisons do not.
  std::string emitSpline(const Expression::Spline &spline,
                         std::vector<std::string> x) {
    unsigned dimensions = spline.getNumArguments();
    std::string zero = constant(0.0);
    std::vector<std::string> t(dimensions);
    std::string row, outside;
    double stride = 1.0;
    for (unsigned k = 0; k != dimensions; ++k) {
      std::string place;
      if (spline.discrete) {
        // The nearest whole number: floor(x + 1/2), which differs from
        // rounding half away from zero only below 0, where it is clamped.
        place = emitOp(
            "arith.minimumf",
            {emitOp("arith.maximumf",
                    {emitOp("math.floor",
                            {emitOp("arith.addf", {x[k], constant(0.5)})}),
                     zero}),
             constant(spline.counts[k] - 1)});
      } else {
        std::string low = constant(spline.mins[k]);
        // A periodic function takes x less the periods from min to it;
        // the floor of that number has no derivative.
        if (spline.periodic) {
          std::string period = constant(spline.maxs[k] - spline.mins[k]);
          std::string turns = emitOp(
              "math.floor",
              {emitOp("arith.divf", {emitOp("arith.subf", {x[k], low}),
                                     period})});
          x[k] = emitOp("arith.subf",
                        {x[k], emitOp("arith.mulf", {turns, period})});
        } else {
          // Outside [min, max] along this argument.
          std::string below = next(), above = next(), either = next();
          os << indent << below << " = arith.cmpf olt, " << x[k] << ", "
             << low << " : f64\n";
          std::string high = constant(spline.maxs[k]);
          os << indent << above << " = arith.cmpf ogt, " << x[k] << ", "
             << high << " : f64\n"
             << indent << either << " = arith.ori " << below << ", "
             << above << " : i1\n";
          if (outside.empty()) {
            outside = either;
          } else {
            std::string any = next();
            os << indent << any << " = arith.ori " << outside << ", "
               << either << " : i1\n";
            outside = any;
          }
        }
        std::string u = emitOp(
            "arith.mulf",
            {emitOp("arith.subf", {x[k], low}), constant(spline.scales[k])});
        place = emitOp(
            "arith.minimumf",
            {emitOp("arith.maximumf", {emitOp("math.floor", {u}), zero}),
             constant(spline.counts[k] - 1)});
        t[k] = emitOp("arith.subf", {u, place});
      }
      std::string term =
          k == 0 ? place : emitOp("arith.mulf", {place, constant(stride)});
      row = k == 0 ? term : emitOp("arith.addf", {row, term});
      stride *= spline.counts[k];
    }
    std::string index = next();
    os << indent << index << " = arith.fptosi " << row << " : f64 to i32\n";
    auto lookup = [&](unsigned k) {
      std::string column = next(), value = next();
      os << indent << column << " = arith.constant " << k << " : i32\n"
         << indent << value << " = md.lookup %t_" << spline.table << "["
         << index << ", " << column << "] : !grid, i32, i32 -> f64\n";
      return value;
    };
    if (spline.discrete)
      return lookup(0);
    // Horner's rule along the last argument first.
    std::function<std::string(unsigned, unsigned)> sum =
        [&](unsigned k, unsigned offset) {
          std::string p;
          for (int a = 3; a >= 0; --a) {
            std::string c = k == 0 ? lookup(offset + a)
                                   : sum(k - 1, offset + (a << (2 * k)));
            p = p.empty() ? c
                          : emitOp("arith.addf",
                                   {emitOp("arith.mulf", {p, t[k]}), c});
          }
          return p;
        };
    std::string p = sum(dimensions - 1, 0);
    if (outside.empty())
      return p;
    std::string result = next();
    os << indent << result << " = arith.select " << outside << ", " << zero
       << ", " << p << " : f64\n";
    return result;
  }

  std::string emitCall(const Expression::Node &node) {
    StringRef name = node.name;
    std::vector<std::string> a;
    for (const auto &argument : node.arguments)
      a.push_back(emit(*argument));
    for (const Expression::Spline &spline : splines)
      if (spline.name == name)
        return emitSpline(spline, a);
    StringRef op = getFunctionOp(name);
    if (!op.empty())
      return emitOp(op, {a[0]});
    if (name == "min")
      return emitOp("arith.minimumf", {a[0], a[1]});
    if (name == "max")
      return emitOp("arith.maximumf", {a[0], a[1]});
    if (name == "atan2")
      return emitOp("math.atan2", {a[0], a[1]});
    if (name == "select")
      return compareWithZero("une", a[0], a[1], a[2]);
    if (name == "step")
      return compareWithZero("oge", a[0], constant(1.0), constant(0.0));
    if (name == "delta")
      return compareWithZero("oeq", a[0], constant(1.0), constant(0.0));
    if (name == "square")
      return emitOp("arith.mulf", {a[0], a[0]});
    if (name == "cube")
      return emitOp("arith.mulf", {emitOp("arith.mulf", {a[0], a[0]}), a[0]});
    std::string one = constant(1.0);
    if (name == "recip")
      return emitOp("arith.divf", {one, a[0]});
    if (name == "sec")
      return emitOp("arith.divf", {one, emitOp("math.cos", {a[0]})});
    if (name == "csc")
      return emitOp("arith.divf", {one, emitOp("math.sin", {a[0]})});
    return emitOp("arith.divf", {one, emitOp("math.tan", {a[0]})});
  }

  /// Returns true if `node` is a whole number, with or without a sign.
  static bool isWholeNumber(const Expression::Node &node, double &value) {
    const Expression::Node *inner = &node;
    double sign = 1.0;
    while (inner->kind == Expression::Node::Negate) {
      sign = -sign;
      inner = inner->lhs.get();
    }
    if (inner->kind != Expression::Node::Number)
      return false;
    value = sign * inner->number;
    return value == std::floor(value) && std::fabs(value) < 1.0e6;
  }

  std::string emitBinary(StringRef op, const Expression::Node &node) {
    std::string lhs = emit(*node.lhs);
    std::string rhs = emit(*node.rhs);
    return emitOp(op, {lhs, rhs});
  }

  std::string emitOp(StringRef op,
                     std::initializer_list<std::string> operands) {
    std::string result = next();
    os << indent << result << " = " << op << " "
       << llvm::join(operands, ", ") << " : f64\n";
    return result;
  }

  llvm::raw_ostream &os;
  const llvm::StringMap<std::string> &values;
  const llvm::StringMap<const Expression::Node *> &definitions;
  llvm::ArrayRef<Expression::Spline> splines;
  llvm::StringMap<std::string> defined;
  StringRef prefix;
  StringRef indent;
  unsigned counter = 0;
};

} // namespace

/// Adds the names that `node` uses to `used`.
static void collectNames(const Expression::Node &node,
                         llvm::StringSet<> &used) {
  if (node.kind == Expression::Node::Name)
    used.insert(node.name);
  for (const auto *child : {node.lhs.get(), node.rhs.get()})
    if (child)
      collectNames(*child, used);
  for (const auto &argument : node.arguments)
    collectNames(*argument, used);
}

llvm::Expected<Expression>
Expression::parse(StringRef text, llvm::ArrayRef<TabulatedFunction> functions) {
  // The expression, then definitions of names after semicolons, as the
  // custom forces of OpenMM write them: "k*d^2; d = r - r0".
  llvm::SmallVector<StringRef> parts;
  text.split(parts, ';');
  Expression expression;
  std::vector<std::string> used, called;
  Parser parser(parts.front(), used, functions, called);
  auto root = parser.parse();
  if (!root)
    return root.takeError();
  expression.root = std::move(*root);
  llvm::StringMap<const Node *> byName;
  for (StringRef part : llvm::drop_begin(parts)) {
    part = part.trim();
    if (part.empty())
      continue;
    auto [name, body] = part.split('=');
    name = name.trim();
    auto fail = [&](const llvm::Twine &message) {
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "in the expression '%s': %s",
                                     text.str().c_str(),
                                     message.str().c_str());
    };
    if (name.empty() || body.data() == nullptr ||
        !(llvm::isAlpha(name.front()) || name.front() == '_') ||
        !llvm::all_of(name, [](char c) { return llvm::isAlnum(c) || c == '_'; }))
      return fail("expected 'name = expression' after ';', not '" + part +
                  "'");
    if (getArity(name) != 0 || byName.count(name) ||
        llvm::any_of(functions,
                     [&](const auto &function) { return function.name == name; }))
      return fail("'" + name + "' is defined twice or is a function");
    Parser definition(body, used, functions, called);
    auto node = definition.parse();
    if (!node)
      return node.takeError();
    byName[name] = node->get();
    expression.definitions.push_back({name.str(), std::move(*node)});
  }
  // A definition may use names defined after it, but not itself: a search
  // in depth marks a name 1 while it is on the path and 2 when it is done.
  llvm::StringMap<int> mark;
  std::function<bool(StringRef)> cyclic = [&](StringRef name) {
    if (mark.lookup(name) == 1)
      return true;
    if (mark.lookup(name) == 2)
      return false;
    mark[name] = 1;
    llvm::StringSet<> inner;
    collectNames(*byName.lookup(name), inner);
    for (const auto &entry : inner)
      if (byName.count(entry.getKey()) && cyclic(entry.getKey()))
        return true;
    mark[name] = 2;
    return false;
  };
  for (const auto &[name, node] : expression.definitions)
    if (cyclic(name))
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "in the expression '%s': the definition of '%s' refers to itself",
          text.str().c_str(), name.c_str());
  for (const std::string &name : used)
    if (!byName.count(name))
      expression.names.push_back(name);
  for (const std::string &name : called) {
    const TabulatedFunction &function = *llvm::find_if(
        functions, [&](const auto &f) { return f.name == name; });
    Spline spline;
    spline.name = name;
    spline.table = function.getTableName();
    spline.mins = function.mins;
    spline.maxs = function.maxs;
    spline.periodic = function.periodic;
    spline.discrete = function.discrete;
    for (unsigned k = 0; k != function.getNumArguments(); ++k) {
      unsigned intervals = function.sizes[k] - 1;
      spline.counts.push_back(function.discrete ? function.sizes[k]
                                                : intervals);
      spline.scales.push_back(
          function.discrete ? 1.0
                            : static_cast<double>(intervals) /
                                  (function.maxs[k] - function.mins[k]));
    }
    spline.coefficients = function.getCoefficients();
    expression.splines.push_back(std::move(spline));
  }
  return std::move(expression);
}

static double
evaluateNode(const Expression::Node &node,
             const llvm::StringMap<double> &values,
             const llvm::StringMap<const Expression::Node *> &definitions,
             llvm::ArrayRef<Expression::Spline> splines) {
  using Node = Expression::Node;
  switch (node.kind) {
  case Node::Number:
    return node.number;
  case Node::Name:
    if (!values.count(node.name) && definitions.count(node.name))
      return evaluateNode(*definitions.lookup(node.name), values, definitions, splines);
    return values.lookup(node.name);
  case Node::Negate:
    return -evaluateNode(*node.lhs, values, definitions, splines);
  case Node::Add:
    return evaluateNode(*node.lhs, values, definitions, splines) + evaluateNode(*node.rhs, values, definitions, splines);
  case Node::Subtract:
    return evaluateNode(*node.lhs, values, definitions, splines) - evaluateNode(*node.rhs, values, definitions, splines);
  case Node::Multiply:
    return evaluateNode(*node.lhs, values, definitions, splines) * evaluateNode(*node.rhs, values, definitions, splines);
  case Node::Divide:
    return evaluateNode(*node.lhs, values, definitions, splines) / evaluateNode(*node.rhs, values, definitions, splines);
  case Node::Power:
    return std::pow(evaluateNode(*node.lhs, values, definitions, splines),
                    evaluateNode(*node.rhs, values, definitions, splines));
  case Node::Call: {
    std::vector<double> a;
    for (const auto &argument : node.arguments)
      a.push_back(evaluateNode(*argument, values, definitions, splines));
    double x = a[0];
    for (const Expression::Spline &spline : splines)
      if (spline.name == node.name)
        return spline.evaluate(a);
    double y = a.size() > 1 ? a[1] : 0.0, z = a.size() > 2 ? a[2] : 0.0;
    return llvm::StringSwitch<double>(node.name)
        .Case("sqrt", std::sqrt(x))
        .Case("exp", std::exp(x))
        .Case("log", std::log(x))
        .Case("sin", std::sin(x))
        .Case("cos", std::cos(x))
        .Case("tan", std::tan(x))
        .Case("asin", std::asin(x))
        .Case("acos", std::acos(x))
        .Case("atan", std::atan(x))
        .Case("sinh", std::sinh(x))
        .Case("cosh", std::cosh(x))
        .Case("tanh", std::tanh(x))
        .Case("abs", std::fabs(x))
        .Case("floor", std::floor(x))
        .Case("ceil", std::ceil(x))
        .Case("erf", std::erf(x))
        .Case("erfc", std::erfc(x))
        .Case("step", x >= 0.0 ? 1.0 : 0.0)
        .Case("delta", x == 0.0 ? 1.0 : 0.0)
        .Case("square", x * x)
        .Case("cube", x * x * x)
        .Case("recip", 1.0 / x)
        .Case("sec", 1.0 / std::cos(x))
        .Case("csc", 1.0 / std::sin(x))
        .Case("cot", 1.0 / std::tan(x))
        .Case("min", std::fmin(x, y))
        .Case("max", std::fmax(x, y))
        .Case("atan2", std::atan2(x, y))
        .Case("select", x != 0.0 ? y : z)
        .Default(std::nan(""));
  }
  }
  return std::nan("");
}

/// Replaces in `node` every call of `function` of the name `argument` by
/// the name `replacement`.
static void replaceCalls(Expression::Node &node, StringRef function,
                         StringRef argument, StringRef replacement) {
  using Node = Expression::Node;
  if (node.kind == Node::Call && node.name == function &&
      node.arguments.size() == 1 &&
      node.arguments.front()->kind == Node::Name &&
      node.arguments.front()->name == argument) {
    node.kind = Node::Name;
    node.name = replacement.str();
    node.arguments.clear();
    return;
  }
  for (auto *child : {node.lhs.get(), node.rhs.get()})
    if (child)
      replaceCalls(*child, function, argument, replacement);
  for (auto &child : node.arguments)
    replaceCalls(*child, function, argument, replacement);
}

/// Appends to `names` the names that `node` uses, in the order of their
/// first use, once each.
static void appendNames(const Expression::Node &node,
                        std::vector<std::string> &names) {
  if (node.kind == Expression::Node::Name &&
      !llvm::is_contained(names, node.name))
    names.push_back(node.name);
  for (const auto *child : {node.lhs.get(), node.rhs.get()})
    if (child)
      appendNames(*child, names);
  for (const auto &child : node.arguments)
    appendNames(*child, names);
}

void Expression::replaceCall(StringRef function, StringRef argument,
                             StringRef replacement) {
  replaceCalls(*root, function, argument, replacement);
  for (auto &[name, node] : definitions)
    replaceCalls(*node, function, argument, replacement);
  // The names that the expression uses and does not define, again.
  std::vector<std::string> used;
  appendNames(*root, used);
  for (const auto &[name, node] : definitions)
    appendNames(*node, used);
  names.clear();
  for (const std::string &name : used)
    if (llvm::none_of(definitions,
                      [&](const auto &entry) { return entry.first == name; }))
      names.push_back(name);
}

llvm::StringMap<const Expression::Node *> Expression::getDefinitions() const {
  llvm::StringMap<const Node *> byName;
  for (const auto &[name, node] : definitions)
    byName[name] = node.get();
  return byName;
}

double Expression::evaluate(const llvm::StringMap<double> &values) const {
  return evaluateNode(*root, values, getDefinitions(), splines);
}

std::string Expression::emit(llvm::raw_ostream &os,
                             const llvm::StringMap<std::string> &values,
                             StringRef prefix, StringRef indent) const {
  auto byName = getDefinitions();
  return Emitter(os, values, byName, splines, prefix, indent).emit(*root);
}
