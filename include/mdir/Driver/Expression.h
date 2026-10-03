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

/// A tabulated function of one, two, or three arguments. A continuous one
/// is given by its values at evenly spaced points from `mins` to `maxs`
/// along each argument: of one argument a natural cubic spline between them
/// (D138), of two or three the bicubic or tricubic patch of each cell that
/// matches the values and the derivatives at its corners, the derivatives
/// taken from natural splines along the axes (D165); zero outside, or, if
/// `periodic`, with periodic splines, the first and the last value along
/// each axis equal, and each argument taken modulo `maxs - mins`: the
/// continuous tabulated functions of OpenMM [Eastman2017]. A `discrete`
/// one is the value at the nearest point, the arguments rounded to whole
/// numbers from 0 and clamped to the table: the discrete functions of
/// OpenMM.
struct TabulatedFunction {
  std::string name;
  /// The values, the first argument varying fastest.
  std::vector<double> values;
  /// The number of points along each argument.
  std::vector<unsigned> sizes;
  std::vector<double> mins;
  std::vector<double> maxs;
  bool periodic = false;
  bool discrete = false;

  unsigned getNumArguments() const { return sizes.size(); }
  /// The polynomial of each cell, in the places t, u, v in it, each from 0
  /// to 1: 4ⁿ numbers c[a + 4b + 16c] of Σ c tᵃ uᵇ vᶜ for each cell, the
  /// first argument varying fastest; or, if `discrete`, the values.
  std::vector<double> getCoefficients() const;
  /// The numbers of each row of the table: 4ⁿ, or 1 if `discrete`.
  unsigned getColumns() const {
    return discrete ? 1u : 1u << (2 * getNumArguments());
  }
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
  /// and the rows of TabulatedFunction::getCoefficients.
  struct Spline {
    std::string name;
    std::string table;
    std::vector<double> mins;
    std::vector<double> maxs;
    /// Along each argument, the intervals per unit of it, and the number of
    /// intervals, or, if `discrete`, of points.
    std::vector<double> scales;
    std::vector<unsigned> counts;
    bool periodic = false;
    bool discrete = false;
    std::vector<double> coefficients;

    unsigned getNumArguments() const { return counts.size(); }
    unsigned getColumns() const {
      return discrete ? 1u : 1u << (2 * getNumArguments());
    }
    double evaluate(llvm::ArrayRef<double> x) const;
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
