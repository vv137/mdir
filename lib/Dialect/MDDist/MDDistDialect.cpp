// The md_dist dialect, its types, and its ops.
//
// See docs/distributed-ml2.md for the design (D275). The checks
// V1 to V9 are those of Section 4.9.

#include "mdir/Dialect/MDDist/MDDistDialect.h"
#include "mdir/Dialect/MDDist/MDDistOps.h"
#include "mdir/Dialect/MDDist/MDDistTypes.h"

#include "mdir/Dialect/MD/MDTypes.h"
#include "mdir/Dialect/MD/MDExchange.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include <cstdio>
#include <cstdlib>
#include <limits>

using namespace mlir;
using namespace mdir::md_dist;
namespace md = mdir::md;

#include "mdir/Dialect/MDDist/MDDistDialect.cpp.inc"
#include "mdir/Dialect/MDDist/MDDistEnums.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "mdir/Dialect/MDDist/MDDistTypes.cpp.inc"

#define GET_OP_CLASSES
#include "mdir/Dialect/MDDist/MDDistOps.cpp.inc"

void MDDistDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "mdir/Dialect/MDDist/MDDistTypes.cpp.inc"
      >();
}

void MDDistDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mdir/Dialect/MDDist/MDDistOps.cpp.inc"
      >();
  registerTypes();
}

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

/// Prints `value` in the fewest digits that read back to it, with a point.
static void printShortest(AsmPrinter &printer, double value) {
  char text[32];
  for (int digits = 1; digits <= 17; ++digits) {
    std::snprintf(text, sizeof text, "%.*g", digits, value);
    if (std::strtod(text, nullptr) == value)
      break;
  }
  std::string result = text;
  if (result.find_first_of(".eEn") == std::string::npos)
    result += ".0";
  printer << result;
}

Type HaloType::parse(AsmParser &parser) {
  FlatSymbolRefAttr decomposition;
  double reach;
  if (parser.parseLess() || parser.parseAttribute(decomposition) ||
      parser.parseComma() || parser.parseFloat(reach) ||
      parser.parseGreater())
    return {};
  return getChecked([&] { return parser.emitError(parser.getNameLoc()); },
                    parser.getContext(), decomposition, llvm::APFloat(reach));
}

void HaloType::print(AsmPrinter &printer) const {
  printer << "<" << getDecomposition() << ", ";
  printShortest(printer, getReach().convertToDouble());
  printer << ">";
}

LogicalResult HaloType::verify(function_ref<InFlightDiagnostic()> emitError,
                               FlatSymbolRefAttr decomposition,
                               llvm::APFloat reach) {
  if (!decomposition)
    return emitError() << "expected a decomposition";
  if (!(reach.convertToDouble() > 0.0))
    return emitError() << "expected a positive reach";
  return success();
}

Type FieldType::parse(AsmParser &parser) {
  FlatSymbolRefAttr decomposition;
  StringRef roleName;
  if (parser.parseLess() || parser.parseAttribute(decomposition) ||
      parser.parseComma() || parser.parseKeyword(&roleName) ||
      parser.parseComma())
    return {};
  std::optional<Role> role = symbolizeRole(roleName);
  if (!role) {
    parser.emitError(parser.getNameLoc())
        << "expected 'owned' or 'ghost', got '" << roleName << "'";
    return {};
  }
  bool partial = false;
  if (succeeded(parser.parseOptionalKeyword("partial"))) {
    partial = true;
    if (parser.parseComma())
      return {};
  }
  unsigned numComponents = 1;
  OptionalParseResult count = parser.parseOptionalInteger(numComponents);
  if (count.has_value() && (count.value() || parser.parseKeyword("x")))
    return {};
  Type elementType;
  if (parser.parseType(elementType) || parser.parseGreater())
    return {};
  return getChecked([&] { return parser.emitError(parser.getNameLoc()); },
                    parser.getContext(), decomposition, *role, partial,
                    numComponents, elementType);
}

void FieldType::print(AsmPrinter &printer) const {
  printer << "<" << getDecomposition() << ", " << stringifyRole(getRole());
  if (getPartial())
    printer << ", partial";
  printer << ", ";
  if (getNumComponents() != 1)
    printer << getNumComponents() << " x ";
  printer << getElementType() << ">";
}

LogicalResult FieldType::verify(function_ref<InFlightDiagnostic()> emitError,
                                FlatSymbolRefAttr decomposition, Role role,
                                bool partial, unsigned numComponents,
                                Type elementType) {
  if (!decomposition)
    return emitError() << "expected a decomposition";
  if (numComponents != 1 && numComponents != 3)
    return emitError() << "expected 1 or 3 components, got " << numComponents;
  bool isReal = elementType.isF64() || elementType.isF32();
  bool isInteger =
      elementType.isSignlessInteger(32) || elementType.isSignlessInteger(64);
  if (!isReal && !isInteger)
    return emitError() << "expected element type f64, f32, i32, or i64, got "
                       << elementType;
  if (numComponents == 3 && !isReal)
    return emitError() << "a field with 3 components must have element type "
                          "f64 or f32, got "
                       << elementType;
  if (partial && !isReal)
    return emitError() << "a partial field holds contributions to a sum and "
                          "must be of f64 or f32";
  return success();
}

Type FieldType::getKernelValueType() const {
  if (getNumComponents() == 1)
    return getElementType();
  return VectorType::get({static_cast<int64_t>(getNumComponents())},
                         getElementType());
}

FieldType FieldType::with(Role role, bool partial) const {
  return FieldType::get(getContext(), getDecomposition(), role, partial,
                        getNumComponents(), getElementType());
}

//===----------------------------------------------------------------------===//
// Provenance
//===----------------------------------------------------------------------===//

static FieldType getFieldType(Value value) {
  return dyn_cast<FieldType>(value.getType());
}

Value mdir::md_dist::haloOf(Value value) {
  if (isa<HaloType>(value.getType()))
    return value;
  Operation *def = value.getDefiningOp();
  if (!def)
    return {};
  if (auto op = dyn_cast<NeighborhoodOp>(def))
    return op.getHalo();
  if (auto op = dyn_cast<HaloCopyOp>(def))
    return op.getHalo();
  if (auto op = dyn_cast<ZerosOp>(def))
    return isa<HaloType>(op.getRows().getType()) ? op.getRows() : Value();
  if (auto op = dyn_cast<GatherPairsOp>(def)) {
    FieldType type = getFieldType(value);
    return type && type.isGhost() ? haloOf(op.getRelation()) : Value();
  }
  if (auto op = dyn_cast<MapParticlesOp>(def)) {
    FieldType type = getFieldType(value);
    return type && type.isGhost() && !op.getGathered().empty()
               ? haloOf(op.getGathered()[0])
               : Value();
  }
  return {};
}

Value mdir::md_dist::layoutOf(Value value) {
  Type type = value.getType();
  if (isa<LayoutType>(type))
    return value;
  Operation *def = value.getDefiningOp();
  if (!def)
    return {};
  if (auto op = dyn_cast<HaloOp>(def))
    return op.getLayout();
  FieldType field = getFieldType(value);
  if (isa<RelationType>(type) || (field && field.isGhost())) {
    Value halo = haloOf(value);
    return halo ? layoutOf(halo) : Value();
  }
  if (!field)
    return {};
  if (auto op = dyn_cast<DistributeOp>(def))
    return op.getLayout();
  if (auto op = dyn_cast<RedistributeOp>(def))
    return op.getNewLayout();
  if (auto op = dyn_cast<HaloSumOp>(def))
    return layoutOf(op.getHalo());
  if (auto op = dyn_cast<PartitionCentersOp>(def))
    return op.getLayout();
  if (auto op = dyn_cast<ZerosOp>(def))
    return layoutOf(op.getRows());
  if (auto op = dyn_cast<GatherPairsOp>(def))
    return layoutOf(op.getRelation());
  if (auto op = dyn_cast<MapParticlesOp>(def))
    return op.getGathered().empty() ? Value() : layoutOf(op.getGathered()[0]);
  return {};
}

Value mdir::md_dist::copySourceOf(Value ghosts) {
  auto copy = ghosts.getDefiningOp<HaloCopyOp>();
  return copy ? copy.getField() : Value();
}

/// Checks that `value` is of `layout`, the epoch of ownership the op works
/// in (V5).
static LogicalResult verifySameLayout(Operation *op, StringRef what,
                                      Value value, Value layout) {
  Value own = layoutOf(value);
  if (!own)
    return op->emitOpError()
           << "cannot show the layout of " << what
           << ": it does not come from an op of md_dist that states it";
  if (!layout)
    return op->emitOpError() << "cannot show the layout that the op works in";
  if (own != layout) {
    InFlightDiagnostic diagnostic =
        op->emitOpError()
        << what << " is of another layout (epoch of ownership) than the op: "
        << "a field must be redistributed to a new layout before it is used "
           "with it";
    diagnostic.attachNote(own.getLoc()) << "the layout of " << what;
    diagnostic.attachNote(layout.getLoc()) << "the layout of the op";
    return diagnostic;
  }
  return success();
}

/// Checks that `value` is a complete field of the owned rows (V6).
static LogicalResult verifyOwnedComplete(Operation *op, StringRef what,
                                         Value value) {
  FieldType type = getFieldType(value);
  if (!type || !type.isOwned())
    return op->emitOpError() << "expected " << what
                             << " to be a field of owned rows";
  if (type.getPartial())
    return op->emitOpError()
           << what << " is partial: contributions that are not summed at "
           << "their owners yet (md_dist.halo_sum) cannot be read";
  return success();
}

LogicalResult mdir::md_dist::verifyGhostCopy(Operation *op, StringRef what,
                                             Value owned, Value ghosts,
                                             Value halo, bool positions) {
  if (failed(verifyOwnedComplete(op, what, owned)))
    return failure();
  FieldType ghostType = getFieldType(ghosts);
  // V1: the field is read at neighbors, so its values at the ghosts must be
  // there.
  if (!ghostType || !ghostType.isGhost())
    return op->emitOpError()
           << "missing transfer: " << what
           << " is read at neighbors, which are ghosts as well, but its "
              "ghost values are not given; copy it with md_dist.halo_copy";
  if (ghostType.getPartial())
    return op->emitOpError() << "the ghost values of " << what
                             << " are partial and cannot be read";
  if (ghostType.getNumComponents() !=
          getFieldType(owned).getNumComponents() ||
      ghostType.getElementType() != getFieldType(owned).getElementType())
    return op->emitOpError() << "the ghost values of " << what
                             << " are not of the type of its owned values";
  auto copy = ghosts.getDefiningOp<HaloCopyOp>();
  if (!copy)
    return op->emitOpError()
           << "cannot show that the ghost values of " << what
           << " are a copy of its owned values: they do not come from "
              "md_dist.halo_copy";
  // V2: the ghosts hold the version that the op reads at its centers.
  if (copy.getField() != owned) {
    InFlightDiagnostic diagnostic =
        op->emitOpError()
        << "stale halo: the ghost values of " << what
        << " are a copy of another version of the field than the owned "
           "values that the op reads";
    diagnostic.attachNote(copy.getLoc()) << "the copy to the ghosts";
    diagnostic.attachNote(copy.getField().getLoc())
        << "the version that was copied";
    diagnostic.attachNote(owned.getLoc()) << "the version that the op reads";
    return diagnostic;
  }
  // V3: through the relation's halo; positions with their images.
  if (copy.getHalo() != halo) {
    InFlightDiagnostic diagnostic =
        op->emitOpError() << "the ghost values of " << what
                          << " were copied through another halo than the "
                             "one the op traverses";
    diagnostic.attachNote(copy.getLoc()) << "the copy to the ghosts";
    return diagnostic;
  }
  if (positions && !copy.getImage())
    return op->emitOpError()
           << "the ghost positions must be copied with their images "
              "(md_dist.halo_copy ... image(%cell))";
  return success();
}

LogicalResult mdir::md_dist::verifyPartialUses(Operation *op) {
  for (OpResult result : op->getResults()) {
    FieldType field = getFieldType(result);
    bool isField = field && field.getPartial();
    bool isNumber = isa<PartialType>(result.getType());
    if (!isField && !isNumber)
      continue;
    unsigned uses = std::distance(result.use_begin(), result.use_end());
    if (uses != 1)
      return op->emitOpError()
             << "result #" << result.getResultNumber()
             << " is partial and must have exactly one use, which completes "
                "it; it has "
             << uses << ": contributions would be "
             << (uses == 0 ? "lost" : "counted more than once");
    Operation *user = *result.user_begin();
    if (isField && !isa<HaloSumOp>(user))
      return op->emitOpError()
             << "result #" << result.getResultNumber()
             << " is a partial field, which only md_dist.halo_sum may "
                "consume, not '"
             << user->getName() << "'";
    if (isNumber && !isa<AllReduceOp>(user))
      return op->emitOpError()
             << "result #" << result.getResultNumber()
             << " is a partial sum, which only md_dist.all_reduce may "
                "consume, not '"
             << user->getName() << "'";
  }
  return success();
}

/// Checks that `type` is of the decomposition `decomposition`.
static LogicalResult verifyDecomposition(Operation *op, Type type,
                                         FlatSymbolRefAttr decomposition) {
  FlatSymbolRefAttr own =
      TypeSwitch<Type, FlatSymbolRefAttr>(type)
          .Case<LayoutType, HaloType, FieldType, RelationType>(
              [](auto t) { return t.getDecomposition(); })
          .Default([](Type) { return FlatSymbolRefAttr(); });
  if (own && own != decomposition)
    return op->emitOpError() << "mixes the decompositions " << own << " and "
                             << decomposition;
  return success();
}

//===----------------------------------------------------------------------===//
// DecompositionOp, PartitionOp, DistributeOp, CollectOp
//===----------------------------------------------------------------------===//

LogicalResult DecompositionOp::verify() {
  if (getGrid().size() != 3)
    return emitOpError() << "expected a grid of 3 divisions, got "
                         << getGrid().size();
  int64_t count = 1;
  for (int64_t n : getGrid()) {
    if (n < 1)
      return emitOpError() << "expected divisions of at least 1, got " << n;
    if (count > std::numeric_limits<int64_t>::max() / n)
      return emitOpError() << "the product of grid divisions overflows i64";
    count *= n;
  }
  return success();
}

int64_t DecompositionOp::getNumDomains() {
  int64_t count = 1;
  for (int64_t n : getGrid())
    count *= n;
  return count;
}

LogicalResult
PartitionOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  FlatSymbolRefAttr name = getLayout().getType().getDecomposition();
  auto decomposition =
      symbolTable.lookupNearestSymbolFrom<DecompositionOp>(*this, name);
  if (!decomposition)
    return emitOpError() << "'" << name.getValue()
                         << "' does not name a decomposition";
  auto positions = cast<md::FieldType>(getPositions().getType());
  if (positions.getParticleSet() != decomposition.getParticleSetAttr())
    return emitOpError() << "the positions are of "
                         << positions.getParticleSet()
                         << ", the decomposition of "
                         << decomposition.getParticleSetAttr();
  if (positions.getNumComponents() != 3)
    return emitOpError() << "expected positions with 3 components";
  return success();
}

LogicalResult MigrateOp::verify() {
  if (getOldLayout().getType() != getLayout().getType())
    return emitOpError() << "the new layout must have the old decomposition";
  if (failed(verifyOwnedComplete(*this, "the positions", getPositions())) ||
      failed(verifySameLayout(*this, "the positions", getPositions(),
                              getOldLayout())))
    return failure();
  if (getPositions().getType().getNumComponents() != 3)
    return emitOpError() << "expected positions with 3 components";
  return success();
}

LogicalResult DistributeOp::verify() {
  auto field = cast<md::FieldType>(getField().getType());
  FieldType result = getResult().getType();
  if (!result.isOwned() || result.getPartial())
    return emitOpError() << "expected a complete field of owned rows";
  if (failed(verifyDecomposition(*this, result,
                                 getLayout().getType().getDecomposition())))
    return failure();
  if (field.getNumComponents() != result.getNumComponents() ||
      field.getElementType() != result.getElementType())
    return emitOpError() << "the result is not of the type of the field";
  if (getImage() && field.getNumComponents() != 3)
    return emitOpError() << "'image' wraps positions, of 3 components";
  return success();
}

LogicalResult CollectOp::verify() {
  if (failed(verifyOwnedComplete(*this, "the field", getField())) ||
      failed(verifySameLayout(*this, "the field", getField(), getLayout())))
    return failure();
  auto result = cast<md::FieldType>(getResult().getType());
  FieldType field = getField().getType();
  if (field.getNumComponents() != result.getNumComponents() ||
      field.getElementType() != result.getElementType())
    return emitOpError() << "the result is not of the type of the field";
  return success();
}

LogicalResult RedistributeOp::verify() {
  if (failed(verifyOwnedComplete(*this, "the field", getField())) ||
      failed(verifySameLayout(*this, "the field", getField(),
                              getOldLayout())))
    return failure();
  if (getOldLayout().getType() != getNewLayout().getType())
    return emitOpError() << "the layouts must have one decomposition";
  auto migrate = getNewLayout().getDefiningOp<MigrateOp>();
  if (!migrate || migrate.getOldLayout() != getOldLayout())
    return emitOpError()
           << "the new layout must come from migration of the field's old "
              "layout";
  if (getResult().getType() != getField().getType())
    return emitOpError() << "the result must have the field's type";
  if (getImage() && getField().getType().getNumComponents() != 3)
    return emitOpError() << "'image' wraps positions, of 3 components";
  if (getImage() && getImage() != migrate.getCell())
    return emitOpError() << "'image' must use the migration's cell";
  return success();
}

//===----------------------------------------------------------------------===//
// HaloOp, HaloCopyOp, HaloSumOp, ZerosOp
//===----------------------------------------------------------------------===//

LogicalResult HaloOp::verify() {
  FlatSymbolRefAttr decomposition = getLayout().getType().getDecomposition();
  if (failed(verifyDecomposition(*this, getResult().getType(),
                                 decomposition)) ||
      failed(verifyOwnedComplete(*this, "the positions", getPositions())) ||
      failed(verifySameLayout(*this, "the positions", getPositions(),
                              getLayout())))
    return failure();
  if (getPositions().getType().getNumComponents() != 3)
    return emitOpError() << "expected positions with 3 components";
  return success();
}

LogicalResult HaloCopyOp::verify() {
  if (failed(verifyOwnedComplete(*this, "the field", getField())) ||
      failed(verifySameLayout(*this, "the field", getField(),
                              layoutOf(getHalo()))))
    return failure();
  FieldType result = getResult().getType();
  if (result != getField().getType().with(Role::Ghost, false))
    return emitOpError() << "expected the result "
                         << getField().getType().with(Role::Ghost, false);
  if (failed(verifyDecomposition(*this, result,
                                 getHalo().getType().getDecomposition())))
    return failure();
  if (getImage() && result.getNumComponents() != 3)
    return emitOpError() << "'image' copies positions, of 3 components";
  return success();
}

LogicalResult HaloSumOp::verify() {
  FieldType owned = getOwned().getType();
  FieldType ghosts = getGhosts().getType();
  if (!owned.isOwned())
    return emitOpError() << "expected the first operand to be owned rows";
  if (!ghosts.isGhost())
    return emitOpError() << "expected the second operand to be ghost rows";
  if (getResult().getType() != owned.with(Role::Owned, false) ||
      ghosts.with(Role::Owned, false) != owned.with(Role::Owned, false))
    return emitOpError() << "expected operands and result of one field type";
  if (failed(verifySameLayout(*this, "the owned values", getOwned(),
                              layoutOf(getHalo()))))
    return failure();
  Value halo = haloOf(getGhosts());
  if (!halo)
    return emitOpError() << "cannot show the halo of the ghost values";
  if (halo != getHalo())
    return emitOpError() << "the ghost values are of another halo than the "
                            "one the op sums through";
  // V7: partial operands are the two results of one traversal, so that each
  // contribution is counted once.
  if (owned.getPartial() != ghosts.getPartial())
    return emitOpError()
           << "expected both operands complete, or both partial (the "
              "contributions of one traversal at owned and at ghost rows)";
  if (owned.getPartial()) {
    auto producer = getOwned().getDefiningOp<GatherPairsOp>();
    if (!producer || getGhosts().getDefiningOp() != producer ||
        cast<OpResult>(getOwned()).getResultNumber() != 0 ||
        cast<OpResult>(getGhosts()).getResultNumber() != 1)
      return emitOpError()
             << "the partial operands are not the two results of one "
                "traversal: contributions would be lost or counted twice";
  }
  return success();
}

LogicalResult ZerosOp::verify() {
  FieldType result = getResult().getType();
  if (result.getPartial())
    return emitOpError() << "expected a complete field";
  bool overGhosts = isa<HaloType>(getRows().getType());
  if (overGhosts != result.isGhost())
    return emitOpError() << "expected a field of "
                         << (overGhosts ? "ghost" : "owned")
                         << " rows over a " << (overGhosts ? "halo" : "layout");
  FlatSymbolRefAttr decomposition =
      overGhosts ? cast<HaloType>(getRows().getType()).getDecomposition()
                 : cast<LayoutType>(getRows().getType()).getDecomposition();
  return verifyDecomposition(*this, result, decomposition);
}

//===----------------------------------------------------------------------===//
// NeighborhoodOp
//===----------------------------------------------------------------------===//

LogicalResult NeighborhoodOp::verify() {
  HaloType halo = getHalo().getType();
  if (failed(verifyDecomposition(*this, getResult().getType(),
                                 halo.getDecomposition())) ||
      failed(verifyGhostCopy(*this, "the positions", getPositions(),
                             getGhosts(), getHalo(), /*positions=*/true)) ||
      failed(verifySameLayout(*this, "the positions", getPositions(),
                              layoutOf(getHalo()))))
    return failure();
  if (getPositions().getType().getNumComponents() != 3)
    return emitOpError() << "expected positions with 3 components";
  return verifyCoverage(*this, *this);
}

LogicalResult PartitionCentersOp::verify() {
  return succeeded(proveCenterPartition(*this)) ? success() : failure();
}

LogicalResult mdir::md_dist::verifyCoverage(Operation *op,
                                            NeighborhoodOp neighborhood) {
  // V4: every pair within the cutoff is within the rows of its owners
  // (docs/distributed-ml2.md, Section 4.3).
  double cutoff = neighborhood.getCutoff().convertToDouble();
  double reach =
      neighborhood.getHalo().getType().getReach().convertToDouble();
  if (!(cutoff > 0.0))
    return op->emitOpError() << "expected a positive cutoff";
  if (reach < cutoff)
    return op->emitOpError() << "the reach of the halo, " << reach
                             << ", is less than the cutoff, " << cutoff
                             << ": the ghosts do not cover the pairs";
  return success();
}

//===----------------------------------------------------------------------===//
// Pair ops
//===----------------------------------------------------------------------===//

/// The checks of a pair op that do not depend on its kernel.
template <typename OpTy>
static LogicalResult verifyPairOperands(OpTy op) {
  auto neighborhood = op.getRelation().template getDefiningOp<NeighborhoodOp>();
  if (!neighborhood)
    return op.emitOpError() << "cannot show the configuration of the "
                               "relation: it does not come from "
                               "md_dist.neighborhood";
  Value halo = neighborhood.getHalo();
  if (Value selected = op.getCenterSet()) {
    auto partition = selected.template getDefiningOp<PartitionCentersOp>();
    if (!partition ||
        (selected != partition.getInterior() &&
         selected != partition.getBoundary()))
      return op.emitOpError()
             << "center_set must be a direct interior or boundary result "
                "of md_dist.partition_centers";
    if (partition.getRelation() != op.getRelation())
      return op.emitOpError()
             << "center_set must partition the exact traversed relation";
    if (partition->getBlock() != op->getBlock() ||
        !partition->isBeforeInBlock(op))
      return op.emitOpError()
             << "center_set must be proved earlier in the same block";
    if (failed(proveCenterPartition(partition)))
      return failure();
  }
  // V9: the relation is of the positions the op reads.
  if (neighborhood.getPositions() != op.getPositions() ||
      neighborhood.getGhosts() != op.getGhosts()) {
    InFlightDiagnostic diagnostic =
        op.emitOpError()
        << "the relation was found at other positions than those the op "
           "reads, and nothing shows that it covers them";
    diagnostic.attachNote(neighborhood.getLoc()) << "the relation";
    return diagnostic;
  }
  if (op.getGathered().size() % 2 != 0)
    return op.emitOpError()
           << "missing transfer: 'gather' takes each field as its owned "
              "values and their copy to the ghosts, in pairs";
  for (unsigned i = 0; i < op.getGathered().size(); i += 2) {
    std::string what = "gathered field #" + std::to_string(i / 2);
    if (failed(verifyGhostCopy(op, what, op.getGathered()[i],
                               op.getGathered()[i + 1], halo,
                               /*positions=*/false)))
      return failure();
  }
  return success();
}

/// The checks of the kernel of a pair op: its arguments, and the exchange
/// contract (V8), proven by the code of md-check-exchange.
template <typename OpTy>
static LogicalResult verifyPairKernel(OpTy op, Type &yielded) {
  Block &kernel = op.getKernel().front();
  MLIRContext *context = op.getContext();
  Type f64 = Float64Type::get(context);
  SmallVector<Type> expected = {f64, VectorType::get({3}, f64)};
  for (unsigned i = 0; i < op.getGathered().size(); i += 2) {
    Type value =
        cast<FieldType>(op.getGathered()[i].getType()).getKernelValueType();
    expected.push_back(value);
    expected.push_back(value);
  }
  if (kernel.getArgumentTypes() != ArrayRef<Type>(expected))
    return op.emitOpError()
           << "expected the kernel to take the distance, the displacement, "
              "and two values of each gathered field";
  auto yield = cast<YieldOp>(kernel.getTerminator());
  if (yield.getValues().size() != 1)
    return op.emitOpError() << "expected the kernel to yield one value";
  yielded = yield.getValues()[0].getType();
  return success();
}

LogicalResult mdir::md_dist::proveContract(Operation *op, Block &kernel,
                                           md::Exchange exchange) {
  if (exchange == md::Exchange::None)
    return success();
  Value value = cast<YieldOp>(kernel.getTerminator()).getValues()[0];
  int expected = exchange == md::Exchange::Symmetric ? 1 : -1;
  if (md::proveExchange(kernel, value) == expected)
    return success();
  return op->emitOpError() << "cannot prove that the kernel is "
                           << md::stringifyExchange(exchange)
                           << " under exchange of the two particles";
}

LogicalResult SumPairsOp::verify() {
  if (failed(verifyPairOperands(*this)) ||
      failed(verifyPartialUses(*this)))
    return failure();
  // Over a directed relation every pair is summed twice with the weight
  // one half, and over a unique one once: both are the sum of the global
  // relation only for a symmetric kernel.
  if (getExchange() != md::Exchange::Symmetric)
    return emitOpError() << "a sum over pairs needs a symmetric kernel: "
                            "expected 'exchange(symmetric)'";
  return success();
}

LogicalResult SumPairsOp::verifyRegions() {
  Type yielded;
  if (failed(verifyPairKernel(*this, yielded)))
    return failure();
  if (yielded != getResult().getType().getValueType())
    return emitOpError() << "the kernel yields " << yielded
                         << ", the sum is of "
                         << getResult().getType().getValueType();
  return proveContract(*this, getKernel().front(), getExchange());
}

LogicalResult GatherPairsOp::verify() {
  if (failed(verifyPairOperands(*this)) ||
      failed(verifyPartialUses(*this)))
    return failure();
  Traversal traversal = getRelation().getType().getTraversal();
  FieldType first =
      getResults().empty() ? FieldType() : dyn_cast<FieldType>(
                                               getResults()[0].getType());
  if (!first)
    return emitOpError() << "expected a field as the result";
  if (traversal == Traversal::Directed && getResults().size() == 1) {
    if (!first.isOwned() || first.getPartial())
      return emitOpError()
             << "over a directed relation every pair of a particle is at "
                "its owner: expected one complete field of owned rows, or "
                "two partial fields of owned and of ghost rows";
    return success();
  }
  if (getResults().size() != 2 ||
      first != first.with(Role::Owned, true) ||
      getResults()[1].getType() != first.with(Role::Ghost, true))
    return emitOpError()
           << (traversal == Traversal::Unique
                   ? "over a unique relation the contributions are at owned "
                     "and at ghost rows: expected two partial fields, of "
                     "owned and of ghost rows"
                   : "over a directed relation expected one complete field "
                     "of owned rows, or two partial fields of owned and of "
                     "ghost rows");
  if (getExchange() == md::Exchange::None)
    return emitOpError()
           << "the contributions at the second particle of a pair come "
              "from the exchange contract: expected 'exchange(symmetric)' or "
              "'exchange(antisymmetric)'";
  return success();
}

LogicalResult GatherPairsOp::verifyRegions() {
  Type yielded;
  if (failed(verifyPairKernel(*this, yielded)))
    return failure();
  auto first = cast<FieldType>(getResults()[0].getType());
  if (yielded != first.getKernelValueType())
    return emitOpError() << "the kernel yields " << yielded
                         << ", a value of the field is "
                         << first.getKernelValueType();
  return proveContract(*this, getKernel().front(), getExchange());
}

//===----------------------------------------------------------------------===//
// SumParticlesOp, AllReduceOp
//===----------------------------------------------------------------------===//

static LogicalResult verifyParticleFields(Operation *op, ValueRange fields);

LogicalResult SumParticlesOp::verify() {
  if (failed(verifyParticleFields(*this, getGathered())))
    return failure();
  return verifyPartialUses(*this);
}

LogicalResult SumParticlesOp::verifyRegions() {
  Block &kernel = getKernel().front();
  SmallVector<Type> expected;
  for (Value field : getGathered())
    expected.push_back(cast<FieldType>(field.getType()).getKernelValueType());
  if (kernel.getArgumentTypes() != ArrayRef<Type>(expected))
    return emitOpError() << "expected the kernel to take one value of each "
                            "field";
  auto yield = cast<YieldOp>(kernel.getTerminator());
  if (yield.getValues().size() != 1 ||
      yield.getValues()[0].getType() != getResult().getType().getValueType())
    return emitOpError() << "expected the kernel to yield one value of "
                         << getResult().getType().getValueType();
  return success();
}

/// The checks of md_dist.sum_particles and md_dist.map_particles on their
/// fields: complete, of one role, and of one layout or halo.
static LogicalResult verifyParticleFields(Operation *op, ValueRange fields) {
  if (fields.empty())
    return op->emitOpError() << "expected at least one field";
  auto first = cast<FieldType>(fields[0].getType());
  Value rows = first.isOwned() ? layoutOf(fields[0]) : haloOf(fields[0]);
  for (Value field : fields) {
    auto type = cast<FieldType>(field.getType());
    if (type.getRole() != first.getRole())
      return op->emitOpError() << "expected fields of one role";
    if (type.getPartial())
      return op->emitOpError() << "a partial field cannot be read";
    Value own = first.isOwned() ? layoutOf(field) : haloOf(field);
    if (!own || own != rows)
      return op->emitOpError()
             << "expected fields of one "
             << (first.isOwned() ? "layout" : "halo")
             << ", shown by the ops that give them";
  }
  return success();
}

LogicalResult MapParticlesOp::verify() {
  if (failed(verifyParticleFields(*this, getGathered())))
    return failure();
  FieldType result = getResult().getType();
  auto first = cast<FieldType>(getGathered()[0].getType());
  if (result.getPartial() || result.getRole() != first.getRole() ||
      result.getDecomposition() != first.getDecomposition())
    return emitOpError() << "expected a complete field of the rows of the "
                            "fields it reads";
  return success();
}

LogicalResult MapParticlesOp::verifyRegions() {
  Block &kernel = getKernel().front();
  SmallVector<Type> expected;
  for (Value field : getGathered())
    expected.push_back(cast<FieldType>(field.getType()).getKernelValueType());
  if (kernel.getArgumentTypes() != ArrayRef<Type>(expected))
    return emitOpError() << "expected the kernel to take one value of each "
                            "field";
  auto yield = cast<YieldOp>(kernel.getTerminator());
  Type value = getResult().getType().getKernelValueType();
  if (yield.getValues().size() != 1 || yield.getValues()[0].getType() != value)
    return emitOpError() << "expected the kernel to yield one value of "
                         << value;
  return success();
}

LogicalResult AllReduceOp::verify() {
  if (getResult().getType() != getPartial().getType().getValueType())
    return emitOpError() << "expected a result of "
                         << getPartial().getType().getValueType();
  Type type = getResult().getType();
  auto vector = dyn_cast<VectorType>(type);
  if (!type.isF64() && !(vector && vector.getRank() == 1 &&
                         vector.getElementType().isF64()))
    return emitOpError() << "expected a sum of f64 or of a vector of f64";
  return success();
}
