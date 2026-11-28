// Periodic TOWERS rebuild entries, specialized for one setup geometry.
#pragma once

#include "mlir/IR/BuiltinOps.h"

namespace mdir::kernels {
struct TowersLayout {
  int atoms, nx, ny, nz, reach, capacity;
  bool host, compact;
  bool triclinic = false;
  // The moving layout allocates dense cells from a pool of at most one
  // additional slot per atom. Inverse entries also carry the owning cell.
  bool growCells = false;
  bool widePack = false;
  int64_t slots() const {
    return int64_t(cells()) * capacity + (growCells ? atoms : 0);
  }
  int64_t keyWords() const { return slots() + (growCells ? atoms : 0); }
  int64_t countWords() const { return growCells ? 2LL * cells() + 1 : cells(); }
  int inverseWords() const { return growCells ? 2 * atoms : atoms; }
  // Triclinic entries take fractional coordinates and the row-major lattice.
  // Setup stores fractional bin geometry, the lattice, Cartesian bin vectors,
  // the unit face normals and perpendicular bin widths in each axis row.
  static constexpr int lattice = 3, step = 6, normal = 9, slab = 12;
  int rawColumns() const { return triclinic ? 13 : 3; }
  int boxSize() const { return triclinic ? 9 : 3; }
  int packedGeometrySize() const {
    return triclinic ? 22 + 3 * (5 + 2 * reach) : 7;
  }
  int cells() const { return nx * ny * nz; }
};

/// Per-atom preparation shared by the moving schedule and its setup trial.
struct TowersMovingRow {
  TowersLayout g;
  bool cartesianPositions;
  mlir::Value positions, cell, raw, box, geometry, status, charge, types,
      counts, keys, local, initialWords, initialState, initialFailed, arrivals;
  // The caller initializes `box` to NaN once and keeps `box` and `geometry`
  // to this layout: thread 0 then recomputes the geometry only when the
  // lattice words change (D[towers-input-geometry]).
  bool cachedGeometry = false;
  void emit(mlir::OpBuilder &b, mlir::Location loc, mlir::Value atom,
            mlir::ValueRange binsInfo, mlir::ValueRange fields) const;
};
llvm::SmallVector<mlir::Value> towersMovingBinGeometry(mlir::OpBuilder &b,
                                                       mlir::Location loc,
                                                       TowersLayout g,
                                                       mlir::Value cell);

/// The moving producer initializes membership and the fixed stencil once.
void emitTowersInitRow(mlir::OpBuilder &builder, mlir::Location loc,
                       TowersLayout layout, mlir::Value cell, mlir::Value map,
                       mlir::Value counts);
/// Insert one current position and charge. Each axis supplies its length,
/// bin width and reciprocal quarter-bin width.
void emitTowersInsertRow(mlir::OpBuilder &builder, mlir::Location loc,
                         TowersLayout layout, mlir::Value atom,
                         mlir::ValueRange xyzq, mlir::ValueRange binGeometry,
                         mlir::ValueRange lattice, mlir::Value counts,
                         mlir::Value keys, mlir::Value local);
void emitTowersGeometry(mlir::OpBuilder &builder, mlir::Location loc,
                        TowersLayout layout, mlir::Value box,
                        mlir::Value status, mlir::Value info);

/// Emit a serial loop or one GPU work item per element of a TOWERS phase.
void emitTowersEntry(
    mlir::OpBuilder &builder, mlir::Location loc, bool host,
    llvm::StringRef name, mlir::TypeRange types, int64_t items,
    llvm::function_ref<void(mlir::Value, mlir::ValueRange)> body);

/// One source and one target span in the coordinate frame of the owner.
/// The fields identify their periodic offsets. In a triangular span only
/// source slots strictly above the target slot participate.
struct TowersSpan {
  mlir::Value field, first, count;
};
void emitOwnedTowersSpans(
    mlir::OpBuilder &builder, mlir::Location loc, int reach, mlir::Value owner,
    mlir::Value descriptors,
    llvm::function_ref<void(TowersSpan source, TowersSpan target,
                            mlir::Value triangular)>
        body);

/// Emit the serial functions or the device module. The device pack entry
/// requires whole warps. A growing layout rebuilds dense spans on device.
/// Fixed layouts report the largest required capacity to their caller.
mlir::LogicalResult emitTowersBuild(mlir::ModuleOp module, TowersLayout layout,
                                    bool audit = false);
} // namespace mdir::kernels
