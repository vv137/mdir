// Explicit site geometry of the class-segmented pair list's window.
#ifndef MDIR_LIB_DRIVER_WATERTEMPLATEIR_H
#define MDIR_LIB_DRIVER_WATERTEMPLATEIR_H

#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLFunctionalExtras.h"

namespace mlir { class OpBuilder; class Block; }
namespace mdir::driver {
struct WaterTemplateModel;
namespace builder {

/// One structurally nonzero site pair. The displacement is x_i - x_j in
/// the anchor image supplied by the caller. Each pair still needs its own
/// site cutoff. A covering bound on the anchors only selects candidates.
struct WaterTemplateSitePair {
  mlir::Value displacement, typeI, typeJ, chargeI, chargeJ;
  bool lennardJones = false, coulomb = false;
  mlir::Value alchemicalI, alchemicalJ;
};

/// O-relative H1 and H2 displacements in one whole molecule. Geometry is
/// ordinary floating-point arithmetic, visible to scalar differentiation.
/// No stored site is loaded: a four-site model forms alpha * (h1 + h2),
/// whose transpose is rule A of D[rigid-water-template].
struct WaterTemplateFrame { mlir::Value h1, h2; };

/// anchorDisplacement is x_Oa - image(x_Ob). The list lowering supplies
/// every candidate anchor image within the covering reach. This routine
/// neither takes a new minimum image nor substitutes an oxygen cutoff.
llvm::SmallVector<WaterTemplateSitePair, 10> buildWaterWaterWindow(
    mlir::OpBuilder &, const WaterTemplateModel &, mlir::Value anchorDisplacement,
    const WaterTemplateFrame &, const WaterTemplateFrame &);

/// Solute - image(O) and that water's whole-molecule frame. The solute
/// retains its dynamic charge/type; only template parameters are constants.
llvm::SmallVector<WaterTemplateSitePair, 4> buildSoluteWaterWindow(
    mlir::OpBuilder &, const WaterTemplateModel &, mlir::Value anchorDisplacement,
    mlir::Value soluteType, mlir::Value soluteCharge,
    const WaterTemplateFrame &);

struct WaterTemplateReductionInputs {
  mlir::Value relation, positions, cell, types, charges;
  double cutoff = 0.0;
  bool lennardJones = true, coulomb = true;
  mlir::Value alchemical;
};
/// Constructs one class's complete structured reduction. The pair callback
/// clones the selected scalar pair expression; it does not replace the
/// placement, cutoff, force or strain rules with an opaque kernel.
mlir::Value buildWaterTemplateReduction(
    mlir::OpBuilder &, const WaterTemplateModel &,
    const WaterTemplateReductionInputs &,
    llvm::function_ref<mlir::Value(mlir::OpBuilder &,
                                  const WaterTemplateSitePair &)> pairEnergy);
/// Reuses the existing scalar primary-pair body. Fixed template arguments
/// replace its gathered values; captured mixing tables and parameters keep
/// their actual SSA values. No alternate LJ/Coulomb expression is written.
mlir::Value cloneWaterTemplatePairEnergy(
    mlir::OpBuilder &, mlir::Block &, const WaterTemplateSitePair &);
} // namespace builder
} // namespace mdir::driver
#endif
