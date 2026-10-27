// A read-only proof of a pair loop's construction coverage.

#include "mdir/Dialect/MDExec/NeighborCoverage.h"

#include "mdir/Dialect/MDExec/MDExecOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "llvm/ADT/DenseSet.h"
#include <optional>

using namespace mlir;
using namespace mdir::md_exec;

namespace {

/// Exact forwarding of an unchanged center subject, not numeric equality.
static Value invariantCenter(Value value) {
  llvm::DenseSet<Value> seen;
  while (value && seen.insert(value).second) {
    scf::ForOp loop;
    unsigned index;
    if (auto argument = dyn_cast<BlockArgument>(value)) {
      loop = dyn_cast<scf::ForOp>(argument.getOwner()->getParentOp());
      if (!loop || argument.getOwner() != loop.getBody() ||
          argument.getArgNumber() == 0)
        break;
      index = argument.getArgNumber() - 1;
    } else if (auto result = dyn_cast<OpResult>(value)) {
      loop = dyn_cast<scf::ForOp>(result.getOwner());
      if (!loop)
        break;
      index = result.getResultNumber();
    } else {
      break;
    }
    auto yield = dyn_cast<scf::YieldOp>(loop.getBody()->getTerminator());
    if (!yield || index >= yield.getNumOperands() ||
        index >= loop.getInitArgs().size() ||
        index >= loop.getRegionIterArgs().size() ||
        yield.getOperand(index) != loop.getRegionIterArgs()[index])
      break;
    value = loop.getInitArgs()[index];
  }
  return value;
}

struct Obligation {
  Value neighbors, centers;
  std::string boundary;
};

class Coverage {
public:
  explicit Coverage(PairForOp loop) : loop(loop) {}
  LogicalResult prove();

private:
  LogicalResult fact(const Obligation &request, double cutoff, Value centers,
                      Images images);
  LogicalResult unsupported(Operation *boundary, StringRef description);
  LogicalResult crossLoop(const Obligation &request, scf::ForOp forOp,
                           unsigned index);
  LogicalResult crossFunction(const Obligation &request, BlockArgument argument,
                               func::FuncOp function);
  bool recursive(func::FuncOp function);
  void append(Value neighbors, Value centers, StringRef boundary) {
    work.push_back({neighbors, invariantCenter(centers), boundary.str()});
  }

  PairForOp loop;
  SmallVector<Obligation> work;
  llvm::DenseSet<std::pair<Value, Value>> seen;
  unsigned facts = 0;
};

LogicalResult Coverage::unsupported(Operation *boundary,
                                     StringRef description) {
  auto diagnostic = loop.emitOpError()
                    << "cannot prove neighbor coverage across " << description
                    << " boundary";
  if (auto name = boundary->getAttrOfType<StringAttr>(
          SymbolTable::getSymbolAttrName()))
    diagnostic << " @" << name.getValue();
  diagnostic.attachNote(boundary->getLoc()) << "unresolved coverage boundary";
  return failure();
}

LogicalResult Coverage::fact(const Obligation &request, double cutoff,
                              Value centers, Images images) {
  std::string error;
  llvm::raw_string_ostream message(error);
  double required = loop.getCutoff().convertToDouble();
  if (required > cutoff)
    message << "the cutoff, " << required
            << ", exceeds the cutoff that the neighbor structure was built "
               "with, " << cutoff;
  else if (invariantCenter(centers) != request.centers)
    message << "the centers of the loop are not those of the neighbor structure";
  else if (images != loop.getImages())
    message << "the loop takes images(" << stringifyImages(loop.getImages())
            << "), but the neighbor structure holds images("
            << stringifyImages(images) << ")";
  else {
    ++facts;
    return success();
  }
  auto diagnostic = loop.emitOpError() << error;
  if (!request.boundary.empty())
    diagnostic << " (through " << request.boundary << " boundary)";
  return failure();
}

LogicalResult Coverage::crossLoop(const Obligation &request, scf::ForOp forOp,
                                   unsigned index) {
  auto yield = dyn_cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  if (!yield || index >= forOp.getInitArgs().size() ||
      index >= yield.getNumOperands())
    return unsupported(forOp, "unrecognized scf.for");

  // Centers that themselves cross the loop are substituted for each
  // path, just as the neighbor value is. A body expression or induction
  // variable cannot be equated with an external subject by assumption.
  auto substitute = [&](Value center, ValueRange path) -> std::optional<Value> {
    if (!center)
      return Value();
    if (auto argument = dyn_cast<BlockArgument>(center)) {
      if (argument.getOwner() == forOp.getBody()) {
        if (argument.getArgNumber() == 0)
          return std::nullopt;
        unsigned place = argument.getArgNumber() - 1;
        return place < path.size() ? std::optional<Value>(path[place])
                                   : std::nullopt;
      }
    } else if (auto result = dyn_cast<OpResult>(center)) {
      if (result.getOwner() == forOp)
        return result.getResultNumber() < path.size()
                   ? std::optional<Value>(path[result.getResultNumber()])
                   : std::nullopt;
    }
    if (forOp->isAncestor(center.getParentBlock()->getParentOp()))
      return std::nullopt;
    return center;
  };
  auto initial = substitute(request.centers, forOp.getInitArgs());
  auto carried = substitute(request.centers, yield.getOperands());
  if (!initial || !carried)
    return unsupported(forOp, "scf.for center subject");
  append(forOp.getInitArgs()[index], *initial, "scf.for initial value");
  append(yield.getOperand(index), *carried, "scf.for yielded value");
  return success();
}

bool Coverage::recursive(func::FuncOp function) {
  SmallVector<func::FuncOp> pending{function};
  llvm::DenseSet<Operation *> visited;
  while (!pending.empty()) {
    func::FuncOp current = pending.pop_back_val();
    if (!visited.insert(current).second)
      continue;
    bool returns = false;
    current.walk([&](func::CallOp call) {
      auto callee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
          call, call.getCalleeAttr());
      if (callee == function)
        returns = true;
      else if (callee && !callee.isExternal())
        pending.push_back(callee);
    });
    if (returns)
      return true;
  }
  return false;
}

LogicalResult Coverage::crossFunction(const Obligation &request,
                                       BlockArgument argument,
                                       func::FuncOp function) {
  if (argument.getOwner() != &function.getBody().front())
    return unsupported(function, "unrecognized function block argument");
  if (!function.isPrivate())
    return unsupported(function, "public function");
  if (recursive(function))
    return unsupported(function, "recursive private function");
  auto module = function->getParentOfType<ModuleOp>();
  auto uses = module ? SymbolTable::getSymbolUses(function, module)
                     : std::nullopt;
  if (!uses || uses->empty())
    return unsupported(function, "private function without known call sites");

  for (const SymbolTable::SymbolUse &use : *uses) {
    auto call = dyn_cast<func::CallOp>(use.getUser());
    if (!call || SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
                     call, call.getCalleeAttr()) != function)
      return unsupported(function, "indirect or address-taken private function");
    if (argument.getArgNumber() >= call.getNumOperands())
      return unsupported(function, "unrecognized private function signature");
    Value centers = request.centers;
    if (centers) {
      if (auto centerArgument = dyn_cast<BlockArgument>(centers)) {
        if (centerArgument.getOwner() == &function.getBody().front())
          centers = call.getOperand(centerArgument.getArgNumber());
        else if (function->isAncestor(centers.getParentBlock()->getParentOp()))
          return unsupported(function, "private function center subject");
      } else if (function->isAncestor(centers.getDefiningOp())) {
        return unsupported(function, "private function center subject");
      }
    }
    append(call.getOperand(argument.getArgNumber()), centers,
           ("private function @" + function.getSymName()).str());
  }
  return success();
}

LogicalResult Coverage::prove() {
  append(loop.getNeighbors(), loop.getCenters(), "");
  for (unsigned cursor = 0; cursor != work.size(); ++cursor) {
    // Appending obligations may reallocate work, so do not retain a reference.
    Obligation request = work[cursor];
    if (!seen.insert({request.neighbors, request.centers}).second)
      continue;
    if (auto build = request.neighbors.getDefiningOp<BuildNeighborsOp>()) {
      if (failed(fact(request, build.getCutoff().convertToDouble(),
                      build.getCenters(), build.getImages())))
        return failure();
      continue;
    }
    if (auto refresh = request.neighbors.getDefiningOp<RefreshNeighborsOp>()) {
      if (failed(fact(request, refresh.getCutoff().convertToDouble(),
                      refresh.getCenters(), refresh.getImages())))
        return failure();
      continue;
    }
    if (auto argument = dyn_cast<BlockArgument>(request.neighbors)) {
      Operation *owner = argument.getOwner()->getParentOp();
      if (auto forOp = dyn_cast<scf::ForOp>(owner)) {
        if (argument.getOwner() != forOp.getBody() ||
            argument.getArgNumber() == 0)
          return unsupported(owner, "unrecognized scf.for argument");
        if (failed(crossLoop(request, forOp, argument.getArgNumber() - 1)))
          return failure();
      } else if (auto function = dyn_cast<func::FuncOp>(owner)) {
        if (failed(crossFunction(request, argument, function)))
          return failure();
      } else {
        return unsupported(owner, owner->getName().getStringRef());
      }
      continue;
    }
    Operation *owner = request.neighbors.getDefiningOp();
    if (auto forOp = dyn_cast<scf::ForOp>(owner)) {
      if (failed(crossLoop(request, forOp,
                           cast<OpResult>(request.neighbors).getResultNumber())))
        return failure();
      continue;
    }
    return unsupported(owner ? owner : loop.getOperation(),
                       owner ? owner->getName().getStringRef()
                             : StringRef("unrecognized SSA"));
  }
  // Every loop edge contributed an initial obligation as well as a yield.
  // Revisiting an obligation closes only that anchored induction; no
  // unknown producer or recursive function is accepted as a cycle seed.
  if (facts == 0)
    return unsupported(loop, "unanchored structured loop");
  return success();
}

} // namespace

LogicalResult mdir::md_exec::verifyNeighborCoverage(PairForOp loop) {
  return Coverage(loop).prove();
}
