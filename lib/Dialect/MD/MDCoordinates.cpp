// Internal coordinates of tuples.

#include "mdir/Dialect/MD/MDCoordinates.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;
using namespace mdir::md;

/// The number of places that the attribute has for each coordinate.
static const unsigned numPlaces = 4;

unsigned mdir::md::getNumMembers(CoordinateKind kind) {
  switch (kind) {
  case CoordinateKind::Distance:
  case CoordinateKind::Displacement:
    return 2;
  case CoordinateKind::Angle:
  case CoordinateKind::Cosine:
    return 3;
  case CoordinateKind::Dihedral:
    return 4;
  }
  llvm_unreachable("unknown kind of coordinate");
}

Type mdir::md::getCoordinateType(CoordinateKind kind, Type real) {
  if (kind == CoordinateKind::Displacement)
    return VectorType::get({3}, real);
  return real;
}

SmallVector<Coordinate, 2>
mdir::md::getCoordinates(ArrayRef<int32_t> kinds, ArrayRef<int64_t> members) {
  SmallVector<Coordinate, 2> coordinates;
  for (auto [index, kind] : llvm::enumerate(kinds)) {
    Coordinate coordinate;
    coordinate.kind = static_cast<CoordinateKind>(kind);
    for (unsigned i = 0, e = getNumMembers(coordinate.kind); i != e; ++i)
      coordinate.members.push_back(members[numPlaces * index + i]);
    coordinates.push_back(coordinate);
  }
  return coordinates;
}

void mdir::md::getCoordinateAttrs(Builder &builder,
                                  ArrayRef<Coordinate> coordinates,
                                  DenseI32ArrayAttr &kinds,
                                  DenseI64ArrayAttr &members) {
  SmallVector<int32_t> allKinds;
  SmallVector<int64_t> allMembers;
  for (const Coordinate &coordinate : coordinates) {
    allKinds.push_back(static_cast<int32_t>(coordinate.kind));
    for (unsigned i = 0; i != numPlaces; ++i)
      allMembers.push_back(i < coordinate.members.size()
                               ? coordinate.members[i]
                               : -1);
  }
  kinds = builder.getDenseI32ArrayAttr(allKinds);
  members = builder.getDenseI64ArrayAttr(allMembers);
}

ParseResult mdir::md::parseCoordinateList(OpAsmParser &parser,
                                          DenseI32ArrayAttr &kinds,
                                          DenseI64ArrayAttr &members) {
  SmallVector<Coordinate, 2> coordinates;
  auto parseCoordinate = [&]() -> ParseResult {
    StringRef keyword;
    SMLoc loc;
    if (parser.getCurrentLocation(&loc) || parser.parseKeyword(&keyword))
      return failure();
    std::optional<CoordinateKind> kind = symbolizeCoordinateKind(keyword);
    if (!kind)
      return parser.emitError(loc)
             << "expected 'distance', 'displacement', 'angle', 'cosine', or "
                "'dihedral', got '"
             << keyword << "'";

    Coordinate coordinate;
    coordinate.kind = *kind;
    auto parseMember = [&]() -> ParseResult {
      int64_t member;
      if (parser.parseInteger(member))
        return failure();
      coordinate.members.push_back(member);
      return success();
    };
    if (parser.parseCommaSeparatedList(AsmParser::Delimiter::Paren,
                                       parseMember))
      return failure();
    if (coordinate.members.size() != getNumMembers(*kind))
      return parser.emitError(loc)
             << "expected " << getNumMembers(*kind) << " members for '"
             << keyword << "', got " << coordinate.members.size();
    coordinates.push_back(coordinate);
    return success();
  };

  if (parser.parseKeyword("coordinates") ||
      parser.parseCommaSeparatedList(AsmParser::Delimiter::Paren,
                                     parseCoordinate))
    return failure();
  getCoordinateAttrs(parser.getBuilder(), coordinates, kinds, members);
  return success();
}

void mdir::md::printCoordinateList(OpAsmPrinter &printer,
                                   DenseI32ArrayAttr kinds,
                                   DenseI64ArrayAttr members) {
  printer << "coordinates(";
  llvm::interleaveComma(
      getCoordinates(kinds.asArrayRef(), members.asArrayRef()), printer,
      [&](const Coordinate &coordinate) {
        printer << stringifyCoordinateKind(coordinate.kind) << "(";
        llvm::interleaveComma(coordinate.members, printer);
        printer << ")";
      });
  printer << ")";
}

LogicalResult mdir::md::verifyCoordinates(Operation *op,
                                          ArrayRef<int32_t> kinds,
                                          ArrayRef<int64_t> members,
                                          unsigned arity) {
  if (kinds.empty())
    return op->emitOpError() << "expected at least 1 coordinate";
  if (members.size() != numPlaces * kinds.size())
    return op->emitOpError()
           << "expected " << numPlaces << " places for the members of each "
           << "coordinate";
  for (int32_t kind : kinds)
    if (kind < 0 || kind > static_cast<int32_t>(CoordinateKind::Dihedral))
      return op->emitOpError() << "unknown kind of coordinate: " << kind;

  for (const Coordinate &coordinate : getCoordinates(kinds, members)) {
    StringRef name = stringifyCoordinateKind(coordinate.kind);
    for (auto [index, member] : llvm::enumerate(coordinate.members)) {
      if (member < 0 || member >= static_cast<int64_t>(arity))
        return op->emitOpError()
               << "'" << name << "' names member " << member
               << ", but a tuple has the members 0 to " << arity - 1;
      for (int64_t other : ArrayRef(coordinate.members).take_front(index))
        if (other == member)
          return op->emitOpError()
                 << "'" << name << "' names member " << member << " twice";
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Values and derivatives
//===----------------------------------------------------------------------===//

SmallVector<Coordinate, 3>
mdir::md::getDisplacements(const Coordinate &coordinate) {
  ArrayRef<int64_t> members = coordinate.members;
  auto displacement = [&](unsigned from, unsigned to) {
    return Coordinate{CoordinateKind::Displacement,
                      {members[from], members[to]}};
  };
  switch (coordinate.kind) {
  case CoordinateKind::Distance:
  case CoordinateKind::Displacement:
    return {displacement(0, 1)};
  case CoordinateKind::Angle:
  case CoordinateKind::Cosine:
    return {displacement(0, 1), displacement(2, 1)};
  case CoordinateKind::Dihedral:
    return {displacement(0, 1), displacement(1, 2), displacement(3, 2)};
  }
  llvm_unreachable("unknown kind of coordinate");
}

namespace {

/// Emits arithmetic on numbers and on vectors of three numbers.
///
/// The operands of a call are evaluated in an order that the compiler
/// chooses. Where the operands emit ops, they are therefore named one by
/// one, so that the order of the ops is the same with every compiler.
struct Emitter {
  Emitter(OpBuilder &builder, Location loc) : builder(builder), loc(loc) {}

  Value constant(double value, Type type) {
    APFloat number(value);
    bool losesInfo = false;
    number.convert(cast<FloatType>(type).getFloatSemantics(),
                   APFloat::rmNearestTiesToEven, &losesInfo);
    return arith::ConstantOp::create(builder, loc, type,
                                     builder.getFloatAttr(type, number));
  }

  Value add(Value lhs, Value rhs) {
    return arith::AddFOp::create(builder, loc, lhs, rhs);
  }
  Value sub(Value lhs, Value rhs) {
    return arith::SubFOp::create(builder, loc, lhs, rhs);
  }
  Value mul(Value lhs, Value rhs) {
    return arith::MulFOp::create(builder, loc, lhs, rhs);
  }
  Value div(Value lhs, Value rhs) {
    return arith::DivFOp::create(builder, loc, lhs, rhs);
  }
  Value neg(Value value) { return arith::NegFOp::create(builder, loc, value); }
  Value sqrt(Value value) { return math::SqrtOp::create(builder, loc, value); }

  /// The components of a vector. They are extracted once for each vector.
  ArrayRef<Value> components(Value vector) {
    SmallVector<Value, 3> &found = extracted[vector];
    if (found.empty())
      for (int64_t k = 0; k < 3; ++k)
        found.push_back(vector::ExtractOp::create(builder, loc, vector, k));
    return found;
  }

  Value dot(Value lhs, Value rhs) {
    SmallVector<Value, 3> a(components(lhs));
    SmallVector<Value, 3> b(components(rhs));
    Value x = mul(a[0], b[0]);
    Value y = mul(a[1], b[1]);
    Value z = mul(a[2], b[2]);
    return add(add(x, y), z);
  }

  /// `lhs * rhs − mhs * nhs`
  Value difference(Value lhs, Value rhs, Value mhs, Value nhs) {
    Value first = mul(lhs, rhs);
    Value second = mul(mhs, nhs);
    return sub(first, second);
  }

  Value cross(Value lhs, Value rhs) {
    SmallVector<Value, 3> a(components(lhs));
    SmallVector<Value, 3> b(components(rhs));
    Value elements[3];
    elements[0] = difference(a[1], b[2], a[2], b[1]);
    elements[1] = difference(a[2], b[0], a[0], b[2]);
    elements[2] = difference(a[0], b[1], a[1], b[0]);
    return vector::FromElementsOp::create(builder, loc, lhs.getType(),
                                          elements);
  }

  Value norm(Value vector) { return sqrt(dot(vector, vector)); }

  /// The product of a number and a vector.
  Value scale(Value factor, Value vector) {
    Value broadcast =
        vector::BroadcastOp::create(builder, loc, vector.getType(), factor);
    return mul(broadcast, vector);
  }

  OpBuilder &builder;
  Location loc;
  llvm::DenseMap<Value, SmallVector<Value, 3>> extracted;
};

/// What the value and the derivative of an angle have in common.
struct AngleParts {
  Value inverseU, inverseV, cosine;
};

/// What the value and the derivative of a dihedral have in common.
struct DihedralParts {
  Value p, q, normG, pp, qq;
};

} // namespace

static Type getElementType(Value displacement) {
  return cast<VectorType>(displacement.getType()).getElementType();
}

static AngleParts emitAngleParts(Emitter &emit, Value u, Value v) {
  Value one = emit.constant(1.0, getElementType(u));
  AngleParts parts;
  parts.inverseU = emit.div(one, emit.norm(u));
  parts.inverseV = emit.div(one, emit.norm(v));
  Value product = emit.dot(u, v);
  parts.cosine =
      emit.mul(product, emit.mul(parts.inverseU, parts.inverseV));
  return parts;
}

static DihedralParts emitDihedralParts(Emitter &emit, Value f, Value g,
                                       Value h) {
  DihedralParts parts;
  parts.p = emit.cross(f, g);
  parts.q = emit.cross(h, g);
  parts.normG = emit.norm(g);
  parts.pp = emit.dot(parts.p, parts.p);
  parts.qq = emit.dot(parts.q, parts.q);
  return parts;
}

Value mdir::md::emitCoordinate(OpBuilder &builder, Location loc,
                               CoordinateKind kind,
                               ArrayRef<Value> displacements) {
  Emitter emit(builder, loc);
  switch (kind) {
  case CoordinateKind::Displacement:
    return displacements[0];
  case CoordinateKind::Distance:
    return emit.norm(displacements[0]);
  case CoordinateKind::Cosine:
    return emitAngleParts(emit, displacements[0], displacements[1]).cosine;
  case CoordinateKind::Angle: {
    // Rounding can take the cosine out of the domain of the arc cosine.
    Value cosine =
        emitAngleParts(emit, displacements[0], displacements[1]).cosine;
    Type type = cosine.getType();
    Value most = emit.constant(1.0, type);
    Value least = emit.constant(-1.0, type);
    Value clamped = arith::MinimumFOp::create(builder, loc, most, cosine);
    clamped = arith::MaximumFOp::create(builder, loc, least, clamped);
    return math::AcosOp::create(builder, loc, clamped);
  }
  case CoordinateKind::Dihedral: {
    Value g = displacements[1];
    DihedralParts parts =
        emitDihedralParts(emit, displacements[0], g, displacements[2]);
    Value triple = emit.dot(emit.cross(parts.q, parts.p), g);
    Value sine = emit.div(triple, parts.normG);
    Value cosine = emit.dot(parts.p, parts.q);
    return math::Atan2Op::create(builder, loc, sine, cosine);
  }
  }
  llvm_unreachable("unknown kind of coordinate");
}

CoordinateGradient
mdir::md::emitCoordinateGradient(OpBuilder &builder, Location loc,
                                 CoordinateKind kind,
                                 ArrayRef<Value> displacements) {
  Emitter emit(builder, loc);
  CoordinateGradient gradient;
  switch (kind) {
  case CoordinateKind::Displacement:
    llvm_unreachable("a displacement has a derivative for each component");

  case CoordinateKind::Distance: {
    // ∂r/∂x_a = d / r, and the negative of it for b.
    Value d = displacements[0];
    Value one = emit.constant(1.0, getElementType(d));
    Value unit = emit.scale(emit.div(one, emit.norm(d)), d);
    Value opposite = emit.neg(unit);
    gradient.members = {unit, opposite};
    gradient.arms = {d, Value()};
    return gradient;
  }

  case CoordinateKind::Angle:
  case CoordinateKind::Cosine: {
    // With the unit vectors e_u and e_v:
    //   ∂c/∂x_a = (e_v − c e_u) / |u|
    //   ∂c/∂x_c = (e_u − c e_v) / |v|
    // and the negative of their sum for b.
    Value u = displacements[0];
    Value v = displacements[1];
    AngleParts parts = emitAngleParts(emit, u, v);
    Value unitU = emit.scale(parts.inverseU, u);
    Value unitV = emit.scale(parts.inverseV, v);
    Value factorA = parts.inverseU;
    Value factorC = parts.inverseV;
    if (kind == CoordinateKind::Angle) {
      // ∂θ/∂c = −1 / sin θ
      Value one = emit.constant(1.0, parts.cosine.getType());
      Value sine =
          emit.sqrt(emit.sub(one, emit.mul(parts.cosine, parts.cosine)));
      Value slope = emit.neg(emit.div(one, sine));
      factorA = emit.mul(slope, factorA);
      factorC = emit.mul(slope, factorC);
    }
    Value a = emit.sub(unitV, emit.scale(parts.cosine, unitU));
    a = emit.scale(factorA, a);
    Value c = emit.sub(unitU, emit.scale(parts.cosine, unitV));
    c = emit.scale(factorC, c);
    Value b = emit.neg(emit.add(a, c));
    gradient.members = {a, b, c};
    gradient.arms = {u, Value(), v};
    return gradient;
  }

  case CoordinateKind::Dihedral: {
    // The form of Blondel and Karplus, J. Comput. Chem. 17, 1132 (1996).
    // With f = d_ab, g = d_bc, h = d_dc, p = f × g, and q = h × g:
    //   ∂φ/∂x_a = −|g| p / p²
    //   ∂φ/∂x_d = +|g| q / q²
    //   ∂φ/∂x_b = +|g| p / p² + (f·g) p / (p² |g|) − (h·g) q / (q² |g|)
    //   ∂φ/∂x_c = −|g| q / q² + (h·g) q / (q² |g|) − (f·g) p / (p² |g|)
    Value f = displacements[0];
    Value g = displacements[1];
    Value h = displacements[2];
    DihedralParts parts = emitDihedralParts(emit, f, g, h);
    Value overP = emit.div(parts.normG, parts.pp);
    Value overQ = emit.div(parts.normG, parts.qq);
    Value gg = emit.mul(parts.normG, parts.normG);
    // (f·g) / (p² |g|) = (f·g) / g² · |g| / p²
    Value alongP = emit.mul(emit.div(emit.dot(f, g), gg), overP);
    Value alongQ = emit.mul(emit.div(emit.dot(h, g), gg), overQ);

    Value a = emit.neg(emit.scale(overP, parts.p));
    Value d = emit.scale(overQ, parts.q);
    Value twistP = emit.scale(alongP, parts.p);
    Value twistQ = emit.scale(alongQ, parts.q);
    Value twist = emit.sub(twistP, twistQ);
    Value b = emit.sub(twist, a);
    Value c = emit.neg(emit.add(twist, d));
    gradient.members = {a, b, c, d};
    gradient.arms = {emit.add(f, g), g, Value(), h};
    return gradient;
  }
  }
  llvm_unreachable("unknown kind of coordinate");
}
