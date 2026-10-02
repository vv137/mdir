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

namespace {

class Parser {
public:
  Parser(StringRef text, std::vector<std::string> &names)
      : text(text), rest(text), names(names) {}

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
};

class Emitter {
public:
  Emitter(llvm::raw_ostream &os, const llvm::StringMap<std::string> &values,
          const llvm::StringMap<const Expression::Node *> &definitions,
          StringRef prefix, StringRef indent)
      : os(os), values(values), definitions(definitions), prefix(prefix),
        indent(indent) {}

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

  std::string emitCall(const Expression::Node &node) {
    StringRef name = node.name;
    std::vector<std::string> a;
    for (const auto &argument : node.arguments)
      a.push_back(emit(*argument));
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

llvm::Expected<Expression> Expression::parse(StringRef text) {
  // The expression, then definitions of names after semicolons, as the
  // custom forces of OpenMM write them: "k*d^2; d = r - r0".
  llvm::SmallVector<StringRef> parts;
  text.split(parts, ';');
  Expression expression;
  std::vector<std::string> used;
  Parser parser(parts.front(), used);
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
    if (getArity(name) != 0 || byName.count(name))
      return fail("'" + name + "' is defined twice or is a function");
    Parser definition(body, used);
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
  return std::move(expression);
}

static double
evaluateNode(const Expression::Node &node,
             const llvm::StringMap<double> &values,
             const llvm::StringMap<const Expression::Node *> &definitions) {
  using Node = Expression::Node;
  switch (node.kind) {
  case Node::Number:
    return node.number;
  case Node::Name:
    if (!values.count(node.name) && definitions.count(node.name))
      return evaluateNode(*definitions.lookup(node.name), values, definitions);
    return values.lookup(node.name);
  case Node::Negate:
    return -evaluateNode(*node.lhs, values, definitions);
  case Node::Add:
    return evaluateNode(*node.lhs, values, definitions) + evaluateNode(*node.rhs, values, definitions);
  case Node::Subtract:
    return evaluateNode(*node.lhs, values, definitions) - evaluateNode(*node.rhs, values, definitions);
  case Node::Multiply:
    return evaluateNode(*node.lhs, values, definitions) * evaluateNode(*node.rhs, values, definitions);
  case Node::Divide:
    return evaluateNode(*node.lhs, values, definitions) / evaluateNode(*node.rhs, values, definitions);
  case Node::Power:
    return std::pow(evaluateNode(*node.lhs, values, definitions),
                    evaluateNode(*node.rhs, values, definitions));
  case Node::Call: {
    std::vector<double> a;
    for (const auto &argument : node.arguments)
      a.push_back(evaluateNode(*argument, values, definitions));
    double x = a[0];
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

llvm::StringMap<const Expression::Node *> Expression::getDefinitions() const {
  llvm::StringMap<const Node *> byName;
  for (const auto &[name, node] : definitions)
    byName[name] = node.get();
  return byName;
}

double Expression::evaluate(const llvm::StringMap<double> &values) const {
  return evaluateNode(*root, values, getDefinitions());
}

std::string Expression::emit(llvm::raw_ostream &os,
                             const llvm::StringMap<std::string> &values,
                             StringRef prefix, StringRef indent) const {
  auto byName = getDefinitions();
  return Emitter(os, values, byName, prefix, indent).emit(*root);
}
