// Scalar pieces shared by packing and the class-segmented fused kernel.
#ifndef MDIR_CONVERSION_TEMPLATEWINDOWKERNELS_H
#define MDIR_CONVERSION_TEMPLATEWINDOWKERNELS_H
#include "mlir/IR/Builders.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include <array>
namespace mlir { class IRMapping; }
namespace mdir::md_exec { class TemplateForOp; }
namespace mdir::kernels {
enum class TemplatePairClass { WaterWater, SoluteWater, SoluteSolute };

/// Whole-molecule O/H/H positions and the group's lattice shift have the
/// state's precision. Add that shift before narrowing, as in D95: narrowing
/// an unwrapped position first can erase its local geometry. No stored
/// extra point is loaded or packed.
std::array<mlir::Value, 3> emitPackTemplateFrame(
    mlir::OpBuilder &, mlir::Location, llvm::ArrayRef<mlir::Value> whole,
    mlir::Value latticeShift, mlir::Type packedReal);

struct PreparedTemplateFrame {
  std::array<mlir::Value, 3> whole, packed;
  mlir::Value latticeShift;
};
/// Recover both hydrogen images relative to the oxygen in state precision,
/// then shift the whole molecule into the anchor's central cell before
/// narrowing. The returned shift is also in state precision. This uses
/// the proved small O-H extent, never a minimum-image shortcut for a
/// molecule-pair window. Both admitted cell representations are supported.
PreparedTemplateFrame emitPrepareTemplateFrame(
    mlir::OpBuilder &, mlir::Location, llvm::ArrayRef<mlir::Value> positions,
    mlir::Value cell, mlir::Value inverse, mlir::Type packedReal);

/// Image-aware anchor displacement followed by whole-molecule H-O offsets,
/// in the order of md.sum_template_window. A water frame has O/H/H values;
/// an atom frame has one. The partner image shifts its anchor, not its
/// reusable local geometry. All arithmetic has the packed positions' type.
llvm::SmallVector<mlir::Value, 5> emitTemplateWindowCoordinates(
    mlir::OpBuilder &, mlir::Location, TemplatePairClass,
    llvm::ArrayRef<mlir::Value> first, llvm::ArrayRef<mlir::Value> second,
    mlir::Value partnerShift);

struct TemplatePartner {
  llvm::SmallVector<mlir::Value, 3> frame;
  llvm::SmallVector<mlir::Value, 3> parameters;
  mlir::Value shift, mask;
};
struct TemplateContribution {
  llvm::SmallVector<mlir::Value, 3> firstForces, secondForces;
  llvm::SmallVector<mlir::Value, 2> sums;
};
/// Emit one class's actual scalar/derived body with an independent guard
/// for every yielded site displacement. Arguments and captures are mapped
/// by the caller. At most one force destination remains after accumulation;
/// energy and strain reductions retain their declared types.
TemplateContribution emitTemplateWindowContributions(
    mlir::OpBuilder &, mlir::Location, mdir::md_exec::TemplateForOp,
    unsigned tag, mlir::IRMapping &);
/// Bind only exact typed alpha*(H1+H2) expressions to the prepared site
/// offsets. This is common-expression reuse, not another placement rule.
unsigned mapTemplateSiteOffsets(mdir::md_exec::TemplateForOp, unsigned tag,
    mlir::ValueRange sites, mlir::IRMapping &);
struct TemplateUnit {
  TemplatePairClass kind;
  /// i32 lane and entry bounds. The caller assigns one class uniformly
  /// to each warp, supplies uniform begin/end bounds, and keeps all 32
  /// lanes active through its shuffles.
  mlir::Value lane, begin, end, firstValid;
  llvm::SmallVector<mlir::Value, 3> first;
  llvm::SmallVector<mlir::Value, 3> parameters;
  bool forces = true;
  llvm::SmallVector<mlir::Type, 2> sumTypes;
  bool hasSite = false;
  double alpha = 0.0;
};
/// Runs a block of entries of one segment. A partner's frame, mask, scalar
/// parameters and accumulated forces rotate through its half warp together.
/// A lane holds one of 16 I units, repeated in the other half; those halves
/// are combined before the single I write. Each valid J entry is written
/// once, after the rotation. Callbacks supply the actual scalar body and
/// atomic destinations; no pair expression is embedded in this scheduler.
/// The evaluation callback may run divergently and must not use warp
/// collectives; the scheduler owns all warp communication.
/// The fourth evaluation range holds cached formed-site offsets: first
/// then second for WW, second for SW, none for SS or three-site models.
llvm::SmallVector<mlir::Value, 2> emitTemplateClassUnit(
    mlir::OpBuilder &, mlir::Location, const TemplateUnit &,
    llvm::function_ref<TemplatePartner(mlir::OpBuilder &, mlir::Value)> loadPartner,
    llvm::function_ref<TemplateContribution(mlir::OpBuilder &,
        mlir::ValueRange, mlir::ValueRange, mlir::ValueRange,
        mlir::ValueRange)> evaluate,
    llvm::function_ref<void(mlir::OpBuilder &, mlir::ValueRange)> writeFirst,
    llvm::function_ref<void(mlir::OpBuilder &, mlir::Value,
                            mlir::ValueRange)> writeSecond);
}
#endif
