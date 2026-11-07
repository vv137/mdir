#ifndef MDIR_LIB_DRIVER_FRAMEBATCHTAILIR_H
#define MDIR_LIB_DRIVER_FRAMEBATCHTAILIR_H
#include "mdir/Driver/Expression.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Error.h"
#include "mlir/IR/Value.h"
#include <array>
#include <vector>
namespace mlir { class OpBuilder; }
namespace mdir::model { struct TunableSet; }
namespace mdir::driver { struct Program; }
namespace mdir::driver::builder {
class TermInputs;
struct FrameBatchTailPlan {
  struct Term {
    explicit Term(ExpressionIR expression) : expression(std::move(expression)) {}
    ExpressionIR expression;
    struct Pair { llvm::StringMap<double> values; double count; };
    std::vector<Pair> pairs;
    struct Parameter { std::string name; int64_t theta, raw; };
    std::vector<Parameter> parameters;
    struct Observed { unsigned column; std::string derivative; };
    std::vector<Observed> observed;
    std::array<double, 4> initial{};
  };
  std::vector<Term> terms;
  double cutoff = 1, factor = 0, fixedFactor = 0;
  bool shifted = false, resetCorrections = false;
  std::array<double, 5> initialCorrections{};
};
llvm::Expected<FrameBatchTailPlan> makeFrameBatchTailPlan(const TermInputs &,
    const Program &, const model::TunableSet &, llvm::ArrayRef<int64_t> offsets);
// Workspace is a device f64 (4001, 5) array reused by serial quadrature.
llvm::Error buildFrameBatchTails(mlir::OpBuilder &, const FrameBatchTailPlan &,
    mlir::Value theta, mlir::Value workspace, mlir::Value corrections,
    mlir::Value rawVolume, mlir::Value rawFixed, mlir::Value observableVolume,
    mlir::Value observableFixed, const llvm::StringMap<mlir::Value> &tables);
}
#endif
