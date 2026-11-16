// A read-only proof of a pair loop's construction coverage.

#include "mdir/Dialect/MDExec/NeighborCoverage.h"

#include "mdir/Dialect/MDExec/MDExecOps.h"
#include "mdir/Dialect/MDRT/MDRTTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Format.h"
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
  Coverage(Operation *op, Value neighbors, Value centers, double cutoff,
           Images images)
      : op(op), neighbors(neighbors), centers(centers), cutoff(cutoff),
        images(images) {}
  /// A list of units, of which the consumer visits `segments`.
  Coverage(Operation *op, Value list, double cutoff,
           ArrayRef<std::pair<unsigned, unsigned>> segments)
      : op(op), neighbors(list), cutoff(cutoff), images(Images::Minimum),
        segments(segments.begin(), segments.end()), segmented(true) {}
  LogicalResult prove();

private:
  LogicalResult fact(const Obligation &request, double cutoff, Value centers,
                      Images images);
  LogicalResult segmentFact(const Obligation &request, double built,
                            Value origin);
  LogicalResult unsupported(Operation *boundary, StringRef description);
  LogicalResult crossLoop(const Obligation &request, scf::ForOp forOp,
                           unsigned index);
  LogicalResult crossFunction(const Obligation &request, BlockArgument argument,
                               func::FuncOp function);
  bool recursive(func::FuncOp function);
  void append(Value neighbors, Value centers, StringRef boundary) {
    work.push_back({neighbors, invariantCenter(centers), boundary.str()});
  }

  Operation *op;
  Value neighbors, centers;
  double cutoff;
  Images images;
  SmallVector<Obligation> work;
  llvm::DenseSet<std::pair<Value, Value>> seen;
  unsigned facts = 0;
  SmallVector<std::pair<unsigned, unsigned>> segments;
  bool segmented = false;
};

/// The op that a list of units began at, through its refreshes and the
/// values that loops carry: its empty list or its build, which hold the
/// rest's radius. Null when the walk ends elsewhere; `last` is then the
/// value it ended at (a function argument, the result of another op, or a
/// value it has seen).
static Operation *getSegmentOrigin(Value list, Value &last) {
  llvm::DenseSet<Value> seen;
  while (list && seen.insert(list).second) {
    last = list;
    if (auto refresh = list.getDefiningOp<RefreshSegmentedPairsOp>()) {
      list = refresh.getList();
      continue;
    }
    if (Operation *op = list.getDefiningOp();
        op && isa<EmptySegmentedPairsOp, BuildSegmentedPairsOp>(op))
      return op;
    if (auto argument = dyn_cast<BlockArgument>(list)) {
      auto loop = dyn_cast<scf::ForOp>(argument.getOwner()->getParentOp());
      if (!loop || argument.getArgNumber() == 0)
        return nullptr;
      list = loop.getInitArgs()[argument.getArgNumber() - 1];
      continue;
    }
    if (auto loop = list.getDefiningOp<scf::ForOp>()) {
      list = loop.getInitArgs()[cast<OpResult>(list).getResultNumber()];
      continue;
    }
    return nullptr;
  }
  return nullptr;
}

LogicalResult Coverage::segmentFact(const Obligation &request, double built,
                                    Value origin) {
  // A build is its own origin; a refresh's chain ends at its empty list or
  // at a build. This is a public query (#898): a chain that ends at a
  // function argument or at the result of another op is refused here, not
  // only by convert-md-exec-to-gpu.
  Value last;
  Operation *start = getSegmentOrigin(origin, last);
  if (!start) {
    std::string where;
    llvm::raw_string_ostream text(where);
    Location at = op->getLoc();
    if (auto argument = dyn_cast_or_null<BlockArgument>(last)) {
      Operation *owner = argument.getOwner()->getParentOp();
      text << "an argument of '" << owner->getName() << "'";
      at = owner->getLoc();
    } else if (Operation *producer = last ? last.getDefiningOp() : nullptr) {
      text << "a result of '" << producer->getName() << "'";
      at = producer->getLoc();
    } else {
      text << "a cycle of refreshes";
    }
    auto diagnostic =
        op->emitOpError()
        << "takes a list of units whose empty list or build is not found: "
           "its refreshes begin at "
        << where << ", which does not give the radius of the rest";
    diagnostic.attachNote(at) << "the list of units begins here";
    return failure();
  }
  std::optional<APFloat> radius;
  if (auto empty = dyn_cast<EmptySegmentedPairsOp>(start))
    radius = empty.getRadius();
  else
    radius = cast<BuildSegmentedPairsOp>(start).getRadius();
  double rest = radius ? radius->convertToDouble() : 0.0;
  Type type = origin.getType();
  for (auto [f, g] : segments) {
    double ef = getSegmentRadius(type, f, rest);
    double eg = getSegmentRadius(type, g, rest);
    // The requirement, formed apart from the reach it is checked against.
    double required = cutoff + ef + eg;
    double reach = getSegmentCoveringReach(built, ef, eg);
    if (required > reach) {
      std::string text;
      llvm::raw_string_ostream message(text);
      message << "segment (" << f << ", " << g
              << ") needs pairs of reference points within "
              << llvm::format("%.6g", required)
              << ", beyond the cutoff of the list plus the radii, "
              << llvm::format("%.6g", reach);
      auto diagnostic = op->emitOpError() << text;
      if (!request.boundary.empty())
        diagnostic << " (through " << request.boundary << " boundary)";
      return failure();
    }
  }
  ++facts;
  return success();
}

LogicalResult Coverage::unsupported(Operation *boundary,
                                     StringRef description) {
  auto diagnostic = op->emitOpError()
                    << "cannot prove neighbor coverage across " << description
                    << " boundary";
  if (auto name = boundary->getAttrOfType<StringAttr>(
          SymbolTable::getSymbolAttrName()))
    diagnostic << " @" << name.getValue();
  diagnostic.attachNote(boundary->getLoc()) << "unresolved coverage boundary";
  return failure();
}

LogicalResult Coverage::fact(const Obligation &request, double built,
                              Value builtCenters, Images builtImages) {
  std::string error;
  llvm::raw_string_ostream message(error);
  double required = cutoff;
  if (required > built)
    message << "the cutoff, " << required
            << ", exceeds the cutoff that the neighbor structure was built "
               "with, " << built;
  else if (invariantCenter(builtCenters) != request.centers)
    message << "the centers of the loop are not those of the neighbor structure";
  else if (builtImages != images)
    message << "the loop takes images(" << stringifyImages(images)
            << "), but the neighbor structure holds images("
            << stringifyImages(builtImages) << ")";
  else {
    ++facts;
    return success();
  }
  auto diagnostic = op->emitOpError() << error;
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
  append(neighbors, centers, "");
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
    if (segmented) {
      // A list of units: its build, or the refresh that renews it, with
      // the cutoff its segments are valid for.
      if (auto build =
              request.neighbors.getDefiningOp<BuildSegmentedPairsOp>()) {
        if (failed(segmentFact(request, build.getCutoff().convertToDouble(),
                               request.neighbors)))
          return failure();
        continue;
      }
      if (auto refresh =
              request.neighbors.getDefiningOp<RefreshSegmentedPairsOp>()) {
        if (failed(segmentFact(request, refresh.getCutoff().convertToDouble(),
                               request.neighbors)))
          return failure();
        continue;
      }
      if (auto empty =
              request.neighbors.getDefiningOp<EmptySegmentedPairsOp>()) {
        // An empty list holds no pair: a consumer of it before any refresh
        // takes nothing that a build would have given.
        return unsupported(empty, "md_exec.empty_segmented_pairs");
      }
    }
    // A cast of the type is the same structure (D413).
    if (auto weaken = request.neighbors.getDefiningOp<WeakenNeighborsOp>()) {
      append(weaken.getNeighbors(), request.centers, request.boundary);
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
    return unsupported(owner ? owner : op,
                       owner ? owner->getName().getStringRef()
                             : StringRef("unrecognized SSA"));
  }
  // Every loop edge contributed an initial obligation as well as a yield.
  // Revisiting an obligation closes only that anchored induction; no
  // unknown producer or recursive function is accepted as a cycle seed.
  if (facts == 0)
    return unsupported(op, "unanchored structured loop");
  return success();
}

} // namespace

LogicalResult mdir::md_exec::verifyNeighborCoverage(PairForOp loop) {
  return Coverage(loop, loop.getNeighbors(), loop.getCenters(),
                  loop.getCutoff().convertToDouble(), loop.getImages())
      .prove();
}

double mdir::md_exec::getSegmentCoveringReach(double reach, double ef,
                                               double eg) {
  return reach + ef + eg;
}

double mdir::md_exec::getSegmentRadius(Type type, unsigned kind,
                                       double restRadius) {
  auto list = cast<mdrt::SegmentedPairsType>(type);
  if (kind >= list.getNumKinds())
    return restRadius;
  return list.getKind(kind).getExtent().getValueAsDouble();
}

LogicalResult mdir::md_exec::verifySegmentCoverage(
    Operation *op, Value list, double cutoff,
    ArrayRef<std::pair<unsigned, unsigned>> segments) {
  return Coverage(op, list, cutoff, segments).prove();
}

LogicalResult mdir::md_exec::verifyNeighborCoverage(Operation *op,
                                                    Value neighbors,
                                                    double cutoff) {
  return Coverage(op, neighbors, Value(), cutoff, Images::Minimum).prove();
}
