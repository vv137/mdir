// Energy expressions.

#include "mdir/Driver/Expression.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"

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

/// The op that computes the function `name`, or an empty string.
static StringRef getFunctionOp(StringRef name) {
  return llvm::StringSwitch<StringRef>(name)
      .Case("sqrt", "math.sqrt")
      .Case("exp", "math.exp")
      .Case("log", "math.log")
      .Case("sin", "math.sin")
      .Case("cos", "math.cos")
      .Case("tan", "math.tan")
      .Case("tanh", "math.tanh")
      .Case("abs", "math.absf")
      .Case("erf", "math.erf")
      .Case("erfc", "math.erfc")
      .Default("");
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
        if (getFunctionOp(name).empty())
          return fail("unknown function '" + name + "'");
        auto argument = parseSum();
        if (!argument)
          return argument;
        if (!consume(')'))
          return fail("expected ')' after the argument of '" + name + "'");
        auto node = create(Expression::Node::Call, std::move(*argument));
        node->name = name.str();
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
          StringRef prefix, StringRef indent)
      : os(os), values(values), prefix(prefix), indent(indent) {}

  std::string emit(const Expression::Node &node) {
    using Node = Expression::Node;
    switch (node.kind) {
    case Node::Number: {
      std::string result = next();
      os << indent << result << " = arith.constant "
         << formatReal(node.number) << " : f64\n";
      return result;
    }
    case Node::Name:
      return values.lookup(node.name);
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
      return emitOp(getFunctionOp(node.name), {emit(*node.lhs)});
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
  StringRef prefix;
  StringRef indent;
  unsigned counter = 0;
};

} // namespace

llvm::Expected<Expression> Expression::parse(StringRef text) {
  Expression expression;
  Parser parser(text, expression.names);
  auto root = parser.parse();
  if (!root)
    return root.takeError();
  expression.root = std::move(*root);
  return std::move(expression);
}

static double evaluateNode(const Expression::Node &node,
                           const llvm::StringMap<double> &values) {
  using Node = Expression::Node;
  switch (node.kind) {
  case Node::Number:
    return node.number;
  case Node::Name:
    return values.lookup(node.name);
  case Node::Negate:
    return -evaluateNode(*node.lhs, values);
  case Node::Add:
    return evaluateNode(*node.lhs, values) + evaluateNode(*node.rhs, values);
  case Node::Subtract:
    return evaluateNode(*node.lhs, values) - evaluateNode(*node.rhs, values);
  case Node::Multiply:
    return evaluateNode(*node.lhs, values) * evaluateNode(*node.rhs, values);
  case Node::Divide:
    return evaluateNode(*node.lhs, values) / evaluateNode(*node.rhs, values);
  case Node::Power:
    return std::pow(evaluateNode(*node.lhs, values),
                    evaluateNode(*node.rhs, values));
  case Node::Call: {
    double x = evaluateNode(*node.lhs, values);
    return llvm::StringSwitch<double>(node.name)
        .Case("sqrt", std::sqrt(x))
        .Case("exp", std::exp(x))
        .Case("log", std::log(x))
        .Case("sin", std::sin(x))
        .Case("cos", std::cos(x))
        .Case("tan", std::tan(x))
        .Case("tanh", std::tanh(x))
        .Case("abs", std::fabs(x))
        .Case("erf", std::erf(x))
        .Case("erfc", std::erfc(x))
        .Default(std::nan(""));
  }
  }
  return std::nan("");
}

double Expression::evaluate(const llvm::StringMap<double> &values) const {
  return evaluateNode(*root, values);
}

std::string Expression::emit(llvm::raw_ostream &os,
                             const llvm::StringMap<std::string> &values,
                             StringRef prefix, StringRef indent) const {
  return Emitter(os, values, prefix, indent).emit(*root);
}
