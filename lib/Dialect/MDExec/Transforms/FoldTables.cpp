// Values of kernels that depend only on tables, computed once per entry of
// the tables.
//
// See the description of the pass in Passes.td.

#include "mdir/Dialect/MDExec/Transforms/Passes.h"

#include "mdir/Dialect/MD/MDOps.h"
#include "mdir/Dialect/MDExec/MDExecOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SetVector.h"

using namespace mlir;
using namespace mdir;
using namespace mdir::md_exec;

namespace mdir {
namespace md_exec {

#define GEN_PASS_DEF_FOLDTABLES
#include "mdir/Dialect/MDExec/Transforms/Passes.h.inc"

namespace {
/// What a value of a kernel depends on: constants only, tables looked up at
/// `indices` and constants, or something else.
struct Dependence {
  enum Kind { Constant, Tables, Other } kind = Other;
  SmallVector<Value, 2> indices;
};

/// Returns true if `op` computes on numbers and nothing else: an op of
/// arith or math without memory effects or regions, whose results are
/// floating-point numbers.
bool isArithmetic(Operation *op) {
  if (!isa<arith::ArithDialect, math::MathDialect>(op->getDialect()) ||
      op->getNumRegions() != 0 || !isMemoryEffectFree(op))
    return false;
  return llvm::all_of(op->getResultTypes(),
                      [](Type type) { return isa<FloatType>(type); });
}

class Folder {
public:
  explicit Folder(Operation *loop) : loop(loop), kernel(loop->getRegion(0).front()) {}

  /// Folds the values of the kernel of the loop. `made` holds the tables
  /// made so far, which are used again where they fit.
  void run(SmallVectorImpl<TabulateOp> &made);

private:
  Dependence getDependence(Value value);
  TabulateOp makeTable(Value value, SmallVectorImpl<TabulateOp> &made);

  Operation *loop;
  Block &kernel;
  DenseMap<Value, Dependence> dependence;
};

Dependence Folder::getDependence(Value value) {
  auto found = dependence.find(value);
  if (found != dependence.end())
    return found->second;
  Dependence result;
  Operation *op = value.getDefiningOp();
  if (op && isa<arith::ConstantOp>(op)) {
    result.kind = Dependence::Constant;
  } else if (op && op->getBlock() == &kernel) {
    if (auto lookup = dyn_cast<md::LookupOp>(op)) {
      if (lookup.getTable().getParentBlock() != &kernel &&
          isa<md::TableType>(lookup.getTable().getType())) {
        result.kind = Dependence::Tables;
        result.indices.assign(lookup.getIndices().begin(),
                              lookup.getIndices().end());
      }
    } else if (isArithmetic(op)) {
      result.kind = Dependence::Constant;
      for (Value operand : op->getOperands()) {
        Dependence of = getDependence(operand);
        if (of.kind == Dependence::Other) {
          result.kind = Dependence::Other;
          break;
        }
        if (of.kind != Dependence::Tables)
          continue;
        if (result.kind == Dependence::Tables &&
            result.indices != of.indices) {
          result.kind = Dependence::Other;
          break;
        }
        result = of;
      }
    }
  }
  dependence[value] = result;
  return result;
}

TabulateOp Folder::makeTable(Value value, SmallVectorImpl<TabulateOp> &made) {
  // The ops that compute the value, in the order of the kernel, and the
  // tables they look up.
  llvm::SetVector<Operation *> cone;
  SmallVector<Value> pending = {value};
  while (!pending.empty()) {
    Operation *op = pending.pop_back_val().getDefiningOp();
    if (!cone.insert(op))
      continue;
    if (!isa<md::LookupOp>(op))
      llvm::append_range(pending, op->getOperands());
  }
  // Constants from outside the kernel first, then the ops of the kernel in
  // its order.
  SmallVector<Operation *> ordered;
  for (Operation *op : cone)
    if (op->getBlock() != &kernel)
      ordered.push_back(op);
  for (Operation &op : kernel)
    if (cone.contains(&op))
      ordered.push_back(&op);
  llvm::SetVector<Value> tables;
  for (Operation *op : ordered)
    if (auto lookup = dyn_cast<md::LookupOp>(op))
      tables.insert(lookup.getTable());

  // The tables must be defined where the new one can be: in the entry block
  // of the function, before the loop.
  auto function = loop->getParentOfType<FunctionOpInterface>();
  if (!function)
    return TabulateOp();
  Block &entry = function.getFunctionBody().front();
  Operation *last = nullptr;
  for (Value table : tables) {
    if (auto argument = dyn_cast<BlockArgument>(table)) {
      if (argument.getOwner() != &entry)
        return TabulateOp();
      continue;
    }
    Operation *def = table.getDefiningOp();
    if (def->getBlock() != &entry)
      return TabulateOp();
    if (!last || last->isBeforeInBlock(def))
      last = def;
  }
  OpBuilder builder(loop->getContext());
  if (last)
    builder.setInsertionPointAfter(last);
  else
    builder.setInsertionPointToStart(&entry);

  bool symmetric = llvm::all_of(tables, [](Value table) {
    return cast<md::TableType>(table.getType()).getSymmetric();
  });
  Type f64 = builder.getF64Type();
  auto type = md::TableType::get(
      builder.getContext(),
      cast<md::TableType>(tables.front().getType()).getRank(), f64,
      symmetric);
  Location loc = value.getLoc();
  auto table = TabulateOp::create(builder, loc, type, tables.getArrayRef());
  Block *body = builder.createBlock(&table.getKernel());
  for (Value unused : tables) {
    (void)unused;
    body->addArgument(f64, loc);
  }
  IRMapping mapping;
  for (Operation *op : ordered) {
    if (auto lookup = dyn_cast<md::LookupOp>(op)) {
      unsigned place = llvm::find(tables, lookup.getTable()) - tables.begin();
      mapping.map(lookup.getResult(), body->getArgument(place));
      continue;
    }
    builder.clone(*op, mapping);
  }
  YieldOp::create(builder, loc, ValueRange{mapping.lookup(value)});

  // A table that another loop made already serves this one.
  for (TabulateOp other : made)
    if (other.getTables() == table.getTables() &&
        OperationEquivalence::isRegionEquivalentTo(
            &other.getKernel(), &table.getKernel(),
            OperationEquivalence::IgnoreLocations)) {
      table.erase();
      return other;
    }
  made.push_back(table);
  return table;
}

void Folder::run(SmallVectorImpl<TabulateOp> &made) {
  // The values that the rest of the kernel takes from tables.
  SmallVector<Value> frontier;
  for (Operation &op : kernel.without_terminator()) {
    if (isa<md::LookupOp>(op) || op.getNumResults() != 1)
      continue;
    Value value = op.getResult(0);
    if (!value.getType().isF64() ||
        getDependence(value).kind != Dependence::Tables)
      continue;
    bool used = llvm::any_of(value.getUsers(), [&](Operation *user) {
      return user->getNumResults() != 1 || user->getBlock() != &kernel ||
             getDependence(user->getResult(0)).kind != Dependence::Tables;
    });
    if (used)
      frontier.push_back(value);
  }
  for (Value value : frontier) {
    TabulateOp table = makeTable(value, made);
    if (!table)
      continue;
    OpBuilder builder(value.getDefiningOp());
    Value looked = md::LookupOp::create(
        builder, value.getLoc(), value.getType(), table.getResult(),
        getDependence(value).indices);
    value.replaceAllUsesWith(looked);
  }
  // What the kernel no longer uses, last first.
  SmallVector<Operation *> ops;
  for (Operation &op : kernel.without_terminator())
    ops.push_back(&op);
  for (Operation *op : llvm::reverse(ops))
    if (op->use_empty() && isMemoryEffectFree(op))
      op->erase();
}

class FoldTables : public impl::FoldTablesBase<FoldTables> {
public:
  void runOnOperation() final {
    SmallVector<Operation *> loops;
    getOperation()->walk([&](Operation *op) {
      if (isa<PairForOp, TupleForOp, ParticleForOp>(op))
        loops.push_back(op);
    });
    SmallVector<TabulateOp> made;
    for (Operation *loop : loops)
      Folder(loop).run(made);
  }
};
} // namespace

} // namespace md_exec
} // namespace mdir
