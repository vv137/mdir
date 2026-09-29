// Energy expressions.
//
// An expression is written in the syntax of D22: numbers, names, the
// operators + - * / ^, parentheses, and functions of one argument.

#ifndef MDIR_DRIVER_EXPRESSION_H
#define MDIR_DRIVER_EXPRESSION_H

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <string>
#include <vector>

namespace mdir {
namespace driver {

class Expression {
public:
  struct Node;

  Expression();
  Expression(Expression &&);
  Expression &operator=(Expression &&);
  ~Expression();

  static llvm::Expected<Expression> parse(llvm::StringRef text);

  /// The names that the expression uses, in the order of their first use.
  const std::vector<std::string> &getNames() const { return names; }

  /// Writes ops that compute the expression in `f64`, one on a line, each
  /// after `indent`. `values` gives the SSA value of every name. The values
  /// that the ops define begin with `prefix`. Returns the value of the
  /// expression.
  std::string emit(llvm::raw_ostream &os,
                   const llvm::StringMap<std::string> &values,
                   llvm::StringRef prefix, llvm::StringRef indent) const;

  /// The value of the expression, with `values` giving the value of every
  /// name.
  double evaluate(const llvm::StringMap<double> &values) const;

private:
  std::unique_ptr<Node> root;
  std::vector<std::string> names;
};

/// A number as MLIR reads a constant of a floating-point type.
std::string formatReal(double value);

} // namespace driver
} // namespace mdir

#endif // MDIR_DRIVER_EXPRESSION_H
