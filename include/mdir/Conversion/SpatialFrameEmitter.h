// Periodic fractional frames for spatial keys and conservative boxes.
#ifndef MDIR_CONVERSION_SPATIALFRAMEEMITTER_H
#define MDIR_CONVERSION_SPATIALFRAMEEMITTER_H
#include "mlir/IR/Builders.h"
namespace mdir::kernels {
struct SpatialReferenceFrame {
  mlir::Value wrapped, lattice;
};
/// A finite vector<3xf64> reference is wrapped componentwise by subtracting
/// floor, in f64 before any key narrowing. `lattice` retains those integer
/// translations. A query image in the wrapped frame must add the first
/// reference's lattice translation before forming the original displacement.
SpatialReferenceFrame emitSpatialReferenceFrame(mlir::OpBuilder &,
    mlir::Location, mlir::Value point);
struct SpatialBoundingBox {
  mlir::Value minimum, maximum;
};
/// Conservative extrema over valid lanes of a warp. All 32 lanes reach the
/// shared reduction; valid points are finite vector<3xf64>. An empty warp
/// yields (+infinity,-infinity), which the caller must not search.
SpatialBoundingBox emitSpatialWarpBox(mlir::OpBuilder &, mlir::Location,
    mlir::Value point, mlir::Value valid);
/// Conservative Cartesian box prefilter. `shift` translates the second
/// box, and `roundoff` bounds coordinate/conversion error per component.
/// This admits uncertain cases; the original f64 pair test remains required.
mlir::Value emitSpatialBoxesWithinReach(mlir::OpBuilder &, mlir::Location,
    SpatialBoundingBox first, SpatialBoundingBox second, mlir::Value shift,
    mlir::Value reach, mlir::Value roundoff);
}
#endif
