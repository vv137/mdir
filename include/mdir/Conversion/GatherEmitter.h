// A wrapped, contiguous-axis grid read shared by the PME gathers
// (D325, docs/device-primitives.md). Each lane reads one grid
// point; the caller keeps its particle and lane mapping and its arithmetic.
// Portability: docs/device-primitives.md, Section 6 (D[device-portability-inventory]).

#ifndef MDIR_CONVERSION_GATHEREMITTER_H
#define MDIR_CONVERSION_GATHEREMITTER_H

#include "mlir/IR/Builders.h"

namespace mdir::kernels {

/// (zStart + lane) mod zSize with one subtraction. All operands have the
/// same type, i32 or index; the caller proves both coordinates are in
/// [0, zSize). `signedCompare` retains a caller's signed comparison when
/// its coordinates are i32. This is computed once per lane where possible.
mlir::Value emitWrappedGridAxis(mlir::OpBuilder &builder, mlir::Location loc,
                                 mlir::Value zStart, mlir::Value lane,
                                 mlir::Value zSize,
                                 bool signedCompare = false);

/// Read grid[rowBase + z], where rowBase is row * zSize and z is the
/// wrapped axis. The coordinate type is i32 or index. The i32 form casts
/// only the final address to index, preserving the stored-weight gather's
/// address arithmetic. When `active` is given, an inactive lane reads
/// rowBase itself so its address stays in bounds, as the prolate gather
/// does for points beyond its window.
mlir::Value emitGridGather(mlir::OpBuilder &builder, mlir::Location loc,
                           mlir::Value grid, mlir::Value rowBase,
                           mlir::Value z, mlir::Value active = {});

} // namespace mdir::kernels

#endif // MDIR_CONVERSION_GATHEREMITTER_H
