// The rigid kinds of D403 and the segments of their pairs (D[kind-pairs]).
#include "mdir/Dialect/MD/MDDialect.h"
#include "mdir/Dialect/MD/MDOps.h"
#include "mdir/Dialect/MD/MDSegmentWindow.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/TypeSwitch.h"
#include <cfloat>
#include <cmath>

using namespace mlir;
using namespace mdir::md;

#define GET_ATTRDEF_CLASSES
#include "mdir/Dialect/MD/MDAttrs.cpp.inc"

void MDDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mdir/Dialect/MD/MDAttrs.cpp.inc"
      >();
}

LogicalResult RigidKindAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                    DenseI64ArrayAttr frame, DenseI64ArrayAttr sites,
                                    DenseF64ArrayAttr weights, FloatAttr extent) {
  if (!frame || !sites || !weights || !extent)
    return emitError() << "expected frame, sites, weights and extent";
  if (frame.empty())
    return emitError() << "expected a nonempty frame";
  size_t size = frame.size() + sites.size();
  SmallVector<bool> seen(size, false);
  for (int64_t place : frame.asArrayRef()) {
    if (place < 0 || static_cast<size_t>(place) >= size || seen[place])
      return emitError() << "expected the frame and the sites to hold each place 0 to "
                         << size - 1 << " once";
    seen[place] = true;
  }
  if (!llvm::is_sorted(frame.asArrayRef()))
    return emitError() << "expected the frame's places in ascending order";
  for (int64_t place : sites.asArrayRef()) {
    if (place < 0 || static_cast<size_t>(place) >= size || seen[place])
      return emitError() << "expected the frame and the sites to hold each place 0 to "
                         << size - 1 << " once";
    seen[place] = true;
  }
  if (weights.size() != static_cast<int64_t>(sites.size() * frame.size()))
    return emitError() << "expected " << sites.size() * frame.size()
                       << " weights, one for each site and frame atom";
  ArrayRef<double> values = weights.asArrayRef();
  for (size_t site = 0; site != sites.size(); ++site) {
    // R2 of D403: the weights of a site sum to 1. Each is a rounded value
    // of an exact one, and their sum is rounded once per term.
    double sum = 0.0, magnitude = 0.0;
    for (double weight : values.slice(site * frame.size(), frame.size())) {
      if (!std::isfinite(weight))
        return emitError() << "expected finite weights";
      sum += weight;
      magnitude += std::fabs(weight);
    }
    if (std::fabs(sum - 1.0) > 2.0 * frame.size() * DBL_EPSILON * magnitude)
      return emitError() << "expected the weights of each site to sum to 1";
  }
  double radius = extent.getValueAsDouble();
  if (!extent.getType().isF64() || !std::isfinite(radius) || radius <= 0.0)
    return emitError() << "expected a positive finite f64 extent";
  return success();
}

SmallVector<double> RigidKindAttr::getPlaceCoefficients(unsigned place) const {
  unsigned m = getFrame().size();
  SmallVector<double> coefficients(m - 1, 0.0);
  if (std::optional<unsigned> index = getFrameIndex(place)) {
    if (*index) coefficients[*index - 1] = 1.0;
    return coefficients;
  }
  ArrayRef<double> weights = getSiteWeights(place);
  // x_s = sum_k w_k x_k = x_0 + sum_{k >= 1} w_k (x_k - x_0): the weights
  // sum to 1, so the anchor's weight does not appear.
  for (unsigned k = 1; k < m; ++k) coefficients[k - 1] = weights[k];
  return coefficients;
}

std::optional<unsigned> RigidKindAttr::getFrameIndex(unsigned place) const {
  for (auto [index, value] : llvm::enumerate(getFrame().asArrayRef()))
    if (value == place) return index;
  return std::nullopt;
}

ArrayRef<double> RigidKindAttr::getSiteWeights(unsigned place) const {
  unsigned m = getFrame().size();
  for (auto [index, value] : llvm::enumerate(getSites().asArrayRef()))
    if (value == place) return getWeights().asArrayRef().slice(index * m, m);
  return {};
}

std::pair<unsigned, unsigned> mdir::md::getSegmentKinds(unsigned s) {
  unsigned f = 0;
  while (getSegmentNumber(f + 1, 0) <= s) ++f;
  return {f, s - getSegmentNumber(f, 0)};
}

SmallVector<std::array<unsigned, 2>> SegmentShape::getCoordinateMembers() const {
  unsigned ma = a.getFrameSize(), mb = b.getFrameSize();
  SmallVector<std::array<unsigned, 2>> members = {{0, ma}};
  for (unsigned k = 1; k < ma; ++k) members.push_back({k, 0});
  for (unsigned k = 1; k < mb; ++k) members.push_back({ma + k, ma});
  return members;
}

SegmentShape mdir::md::getSegmentShape(Operation *op, unsigned f, unsigned g) {
  auto pairs = cast<KindPairsOp>(op);
  unsigned taken = pairs.getNumTakenKinds();
  auto side = [&](unsigned kind) {
    return SegmentSide{kind < taken ? pairs.getDescriptor(kind) : RigidKindAttr()};
  };
  return {side(f), side(g)};
}

SegmentShape mdir::md::getSegmentShape(Operation *window) {
  Value relation = window->getOperand(0);
  auto f = window->getAttrOfType<IntegerAttr>("f").getInt();
  auto g = window->getAttrOfType<IntegerAttr>("g").getInt();
  return getSegmentShape(relation.getDefiningOp(), f, g);
}

static LogicalResult verifyUniverseRelation(Operation *op, RelationType type,
                                            Attribute universe, StringRef what) {
  if (type.getParticleSet() != universe || !type.getTupleSet())
    return op->emitOpError("expected ") << what << " to be a named relation of the universe";
  return success();
}

LogicalResult KindPairsOp::verify() {
  auto universe = cast<FieldType>(getUniverse().getType());
  if (universe.getNumComponents() != 1 || !universe.getElementType().isInteger(32))
    return emitOpError("expected a scalar i32 universe field; only its particle set is used");
  Attribute set = universe.getParticleSet();
  unsigned taken = getNumTakenKinds();
  if (getDescriptors().size() != taken)
    return emitOpError("expected one #md.rigid_kind for each kind, ")
           << taken << ", not " << getDescriptors().size();
  llvm::SmallPtrSet<Attribute, 8> tupleSets;
  for (unsigned f = 0; f != taken; ++f) {
    auto type = cast<RelationType>(getKinds()[f].getType());
    if (failed(verifyUniverseRelation(*this, type, set, "each kind")))
      return failure();
    if (type.getOrientation() != Orientation::Ordered ||
        type.getArity() != getDescriptor(f).getSize())
      return emitOpError("expected kind ")
             << f << " to be ordered tuples of its " << getDescriptor(f).getSize()
             << " places, as its #md.rigid_kind holds";
    if (!tupleSets.insert(type.getTupleSet()).second)
      return emitOpError("expected the kinds to be distinct tuple sets");
  }
  unsigned count = getNumSegments(taken);
  if (getSegments().size() != count)
    return emitOpError("expected (K + 1)(K + 2)/2 = ")
           << count << " segments for K = " << taken << " kinds, not "
           << getSegments().size();
  Attribute excludedSet;
  if (Value excluded = getExcluded()) {
    auto type = cast<RelationType>(excluded.getType());
    if (type.getParticleSet() != set || type.getArity() != 2 ||
        type.getOrientation() != Orientation::Unordered)
      return emitOpError("expected unordered excluded pairs of the universe");
    excludedSet = type.getTupleSet();
  }
  for (unsigned s = 0; s != count; ++s) {
    auto [f, g] = getSegmentKinds(s);
    SegmentShape shape = getSegmentShape(*this, f, g);
    auto type = cast<RelationType>(getSegments()[s].getType());
    auto orientation = f == g && shape.a.getFrameSize() == 1
                           ? Orientation::Unordered : Orientation::Ordered;
    if (failed(verifyUniverseRelation(*this, type, set, "each segment")))
      return failure();
    if (type.getArity() != shape.getArity() || type.getOrientation() != orientation)
      return emitOpError("expected segment ")
             << s << " = s(" << f << ", " << g << ") of arity " << shape.getArity()
             << ", " << stringifyOrientation(orientation);
    if (tupleSets.contains(type.getTupleSet()) || type.getTupleSet() == excludedSet)
      return emitOpError("a segment cannot alias a kind or the excluded pairs");
  }
  if (bool(getAnchored()) != bool(getRadius()))
    return emitOpError("expected the anchors of the rest together with their radius");
  if (Value anchored = getAnchored()) {
    auto type = cast<RelationType>(anchored.getType());
    if (type.getParticleSet() != set || type.getArity() != 2 ||
        type.getOrientation() != Orientation::Ordered)
      return emitOpError("expected ordered (member, anchor) pairs of the universe");
    double radius = getRadius()->convertToDouble();
    if (!(radius > 0.0) || !std::isfinite(radius))
      return emitOpError("expected a positive finite anchor radius");
  }
  return success();
}

LogicalResult KindPairsOp::verifySymbolUses(SymbolTableCollection &symbols) {
  for (Value kind : getKinds()) {
    auto type = cast<RelationType>(kind.getType());
    auto set = symbols.lookupNearestSymbolFrom<TupleSetOp>(*this, type.getTupleSet());
    if (!set || !set.getDisjoint())
      return emitOpError("expected each kind to be a declared disjoint tuple set");
    if (set.getParticleSetAttr() != type.getParticleSet() ||
        set.getArity() != type.getArity() ||
        set.getOrientation() != type.getOrientation())
      return emitOpError("a kind's tuple declaration and operand type must agree");
  }
  for (Value segment : getSegments()) {
    auto type = cast<RelationType>(segment.getType());
    auto domain = symbols.lookupNearestSymbolFrom<TupleSetOp>(*this, type.getTupleSet());
    if (!domain || domain.getParticleSetAttr() != type.getParticleSet() ||
        domain.getArity() != type.getArity() ||
        domain.getOrientation() != type.getOrientation())
      return emitOpError("a segment's tuple declaration and result type must agree");
  }
  return success();
}

// #md.rigid_kind<frame = [...], sites = [...], weights = [...], extent = x>
Attribute RigidKindAttr::parse(AsmParser &parser, Type) {
  SmallVector<int64_t> frame, sites;
  SmallVector<double> weights;
  double extent = 0.0;
  auto integers = [&](StringRef key, SmallVectorImpl<int64_t> &out) {
    return failure(parser.parseKeyword(key) || parser.parseEqual() ||
                   parser.parseCommaSeparatedList(AsmParser::Delimiter::Square, [&] {
                     return parser.parseInteger(out.emplace_back());
                   }));
  };
  if (parser.parseLess() || failed(integers("frame", frame)) || parser.parseComma() ||
      failed(integers("sites", sites)) || parser.parseComma() ||
      parser.parseKeyword("weights") || parser.parseEqual() ||
      parser.parseCommaSeparatedList(AsmParser::Delimiter::Square, [&] {
        return parser.parseFloat(weights.emplace_back());
      }) ||
      parser.parseComma() || parser.parseKeyword("extent") || parser.parseEqual() ||
      parser.parseFloat(extent) || parser.parseGreater())
    return {};
  MLIRContext *context = parser.getContext();
  return parser.getChecked<RigidKindAttr>(
      context, DenseI64ArrayAttr::get(context, frame),
      DenseI64ArrayAttr::get(context, sites), DenseF64ArrayAttr::get(context, weights),
      FloatAttr::get(Float64Type::get(context), extent));
}

void RigidKindAttr::print(AsmPrinter &printer) const {
  auto integers = [&](ArrayRef<int64_t> values) {
    printer << '[';
    llvm::interleaveComma(values, printer);
    printer << ']';
  };
  printer << "<frame = ";
  integers(getFrame().asArrayRef());
  printer << ", sites = ";
  integers(getSites().asArrayRef());
  printer << ", weights = [";
  llvm::interleaveComma(getWeights().asArrayRef(), printer,
                        [&](double weight) { printer.printFloat(APFloat(weight)); });
  printer << "], extent = ";
  printer.printFloat(getExtent().getValue());
  printer << '>';
}
