// Internal coordinates of tuples: what a kernel over the tuples of a
// topology takes. See docs/design-m1.md, Section 3.

#ifndef MDIR_DIALECT_MD_MDCOORDINATES_H
#define MDIR_DIALECT_MD_MDCOORDINATES_H

#include "mdir/Dialect/MD/MDEnums.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallVector.h"

namespace mdir {
namespace md {

/// An internal coordinate of a tuple, and the places in the tuple of the
/// members that it is a coordinate of.
struct Coordinate {
  CoordinateKind kind;
  llvm::SmallVector<int64_t, 4> members;
};

/// The number of members that a coordinate of the kind `kind` takes.
unsigned getNumMembers(CoordinateKind kind);

/// The type of a coordinate of the kind `kind` in a kernel that computes in
/// `real`: a number, or a vector of three for a displacement.
mlir::Type getCoordinateType(CoordinateKind kind, mlir::Type real);

/// The coordinates that the two attributes hold. `members` has four places
/// for each coordinate; those that the coordinate does not take hold -1.
llvm::SmallVector<Coordinate, 2> getCoordinates(llvm::ArrayRef<int32_t> kinds,
                                                llvm::ArrayRef<int64_t> members);

/// The two attributes that hold `coordinates`.
void getCoordinateAttrs(mlir::Builder &builder,
                        llvm::ArrayRef<Coordinate> coordinates,
                        mlir::DenseI32ArrayAttr &kinds,
                        mlir::DenseI64ArrayAttr &members);

/// Parses and prints `coordinates(cosine(0, 1, 2), distance(0, 2))`.
mlir::ParseResult parseCoordinateList(mlir::OpAsmParser &parser,
                                      mlir::DenseI32ArrayAttr &kinds,
                                      mlir::DenseI64ArrayAttr &members);
void printCoordinateList(mlir::OpAsmPrinter &printer,
                         mlir::DenseI32ArrayAttr kinds,
                         mlir::DenseI64ArrayAttr members);

/// Verifies the coordinates of `op`, an op over tuples of `arity` members.
llvm::LogicalResult verifyCoordinates(mlir::Operation *op,
                                      llvm::ArrayRef<int32_t> kinds,
                                      llvm::ArrayRef<int64_t> members,
                                      unsigned arity);

//===----------------------------------------------------------------------===//
// Values and derivatives
//===----------------------------------------------------------------------===//

/// The displacements that the value of `coordinate` is computed from, each
/// a coordinate of the kind `displacement`.
///
/// | Coordinate | Displacements |
/// |---|---|
/// | `distance(a, b)`, `displacement(a, b)` | `d_ab` |
/// | `angle(a, b, c)`, `cosine(a, b, c)` | `d_ab`, `d_cb` |
/// | `dihedral(a, b, c, d)` | `d_ab`, `d_bc`, `d_dc` |
llvm::SmallVector<Coordinate, 3> getDisplacements(const Coordinate &coordinate);

/// Emits the value of a coordinate of the kind `kind` from `displacements`,
/// the values of the coordinates that getDisplacements returns.
mlir::Value emitCoordinate(mlir::OpBuilder &builder, mlir::Location loc,
                           CoordinateKind kind,
                           llvm::ArrayRef<mlir::Value> displacements);

/// The derivative of a coordinate with respect to the positions of its
/// members.
struct CoordinateGradient {
  /// For each member of the coordinate, the derivative with respect to its
  /// position.
  llvm::SmallVector<mlir::Value, 4> members;
  /// For each member of the coordinate, its displacement from one of the
  /// members, which is the same for all. Null for that member.
  llvm::SmallVector<mlir::Value, 4> arms;
};

/// Emits the derivative of a coordinate of the kind `kind`, which is not
/// `displacement`. What the derivative has in common with the value is
/// emitted as emitCoordinate emits it, so that the elimination of common
/// subexpressions leaves one of each.
CoordinateGradient emitCoordinateGradient(
    mlir::OpBuilder &builder, mlir::Location loc, CoordinateKind kind,
    llvm::ArrayRef<mlir::Value> displacements);

} // namespace md
} // namespace mdir

#endif // MDIR_DIALECT_MD_MDCOORDINATES_H
