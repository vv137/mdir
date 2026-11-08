// One class-prefixed spatial order for water anchors and atoms.
#ifndef MDIR_CONVERSION_TEMPLATESPATIALKERNELS_H
#define MDIR_CONVERSION_TEMPLATESPATIALKERNELS_H
#include "mlir/IR/Builders.h"
namespace mdir::kernels {
struct TemplateSpatialBuffers {
  /// Canonical fractional reference points, fixed unit IDs [0,W+A).
  /// The caller proves finite components in [0,1) before calling.
  mlir::Value points, keys, zKeys, counts, starts, cursors, items, scanScratch;
  mlir::Value order, places;
  /// Index sizes, positive column dimensions and enough scan blocks for
  /// 2*columnsX*columnsY bins. Each unit belongs to exactly one bin.
  mlir::Value waters, atoms, columnsX, columnsY, scanBlocks;
  /// Rebuild only these groups' padded order/inverse maps. The bins always
  /// contain both groups' hybrid references for the shared SW search.
  mlir::Value waterActive, atomActive;
};
/// Sort once with class-prefixed x-y serpentine columns, then z keys and
/// fixed-ID ties. Key arithmetic is f32 after narrowing canonical reference
/// coordinates. Padding occurs only between the two classes; each class
/// is cut into consecutive clusters of sixteen real units.
void emitTemplateSpatialOrder(mlir::OpBuilder &, mlir::Location,
                               const TemplateSpatialBuffers &);
}
#endif
