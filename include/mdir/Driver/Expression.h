// Energy expressions.
//
// An expression is written in the syntax of D22: numbers, names, the
// operators + - * / ^, parentheses, the functions of the custom forces of
// OpenMM, tabulated functions of the control file, and definitions of
// names after semicolons.

#ifndef MDIR_DRIVER_EXPRESSION_H
#define MDIR_DRIVER_EXPRESSION_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <string>
#include <vector>

namespace mdir {
namespace driver {

/// A function of one argument given by its values at evenly spaced points
/// from `min` to `max`, a natural cubic spline between them, and zero
/// outside; or, if `periodic`, a periodic spline, with the first and the
/// last value equal, and the argument taken modulo `max - min`: the
/// continuous tabulated function of OpenMM (D138).
struct TabulatedFunction {
  std::string name;
  std::vector<double> values;
  double min = 0.0;
  double max = 0.0;
  bool periodic = false;

  /// The cubic of each interval in the place t in it, from 0 to 1: four
  /// numbers c0, c1, c2, c3 of c0 + c1 t + c2 t² + c3 t³ for each.
  std::vector<double> getCoefficients() const;
  /// The name of the table of the coefficients in the IR.
  std::string getTableName() const { return "fn_" + name; }
};

class Expression {
public:
  struct Node;

  Expression();
  Expression(Expression &&);
  Expression &operator=(Expression &&);
  ~Expression();

  /// Parses `text`, in which a call of the name of one of `functions`
  /// calls that tabulated function.
  static llvm::Expected<Expression>
  parse(llvm::StringRef text, llvm::ArrayRef<TabulatedFunction> functions = {});

  /// Whether `name` is a function of the syntax, which no name of the
  /// control file may take.
  static bool isFunction(llvm::StringRef name);

  /// The names that the expression uses and does not define, in the order
  /// of their first use.
  const std::vector<std::string> &getNames() const { return names; }

  /// Replaces every call of the function `function` whose argument is the
  /// name `argument` by the name `replacement`, which may be one that no
  /// expression can write, such as "cos(theta)": a coordinate that the
  /// kernel then takes in place of the call.
  void replaceCall(llvm::StringRef function, llvm::StringRef argument,
                   llvm::StringRef replacement);

  /// Whether the expression calls a tabulated function.
  bool callsTabulated() const { return !splines.empty(); }

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

  /// A tabulated function that the expression calls: its table in the IR,
  /// and the coefficients of TabulatedFunction::getCoefficients.
  struct Spline {
    std::string name;
    std::string table;
    double min = 0.0;
    /// The intervals per unit of the argument.
    double scale = 0.0;
    double max = 0.0;
    bool periodic = false;
    std::vector<double> coefficients;

    unsigned getNumIntervals() const { return coefficients.size() / 4; }
    double evaluate(double x) const;
  };

private:
  llvm::StringMap<const Node *> getDefinitions() const;

  std::vector<Spline> splines;
  std::unique_ptr<Node> root;
  std::vector<std::string> names;
  /// The names defined after semicolons, and their expressions.
  std::vector<std::pair<std::string, std::unique_ptr<Node>>> definitions;
};

/// A number as MLIR reads a constant of a floating-point type.
std::string formatReal(double value);

} // namespace driver
} // namespace mdir

#endif // MDIR_DRIVER_EXPRESSION_H
