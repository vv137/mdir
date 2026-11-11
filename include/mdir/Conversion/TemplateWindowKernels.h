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

/// Fractional coordinates in the reduced cell, in state precision. This
/// does not wrap: reference epochs and covering images retain their frame.
mlir::Value emitTemplateFractional(mlir::OpBuilder &, mlir::Location,
    mlir::Value position, mlir::Value cell, mlir::Value inverse);
/// Integer-valued fractional shift chosen at an outer-group build. It
/// reproduces the sequential c/b/a central-image convention in D95, and
/// remains fixed until that group's next outer build. Under a barostat,
/// apply it to the current cell, never reuse its old Cartesian value.
mlir::Value emitTemplatePackingCells(mlir::OpBuilder &, mlir::Location,
    mlir::Value anchor, mlir::Value cell, mlir::Value inverse);
/// Reconstruct whole O/H/H but retain the outer build's packing cells.
/// In particular a face crossing does not change an existing entry image.
PreparedTemplateFrame emitRetainedTemplateFrame(
    mlir::OpBuilder &, mlir::Location, llvm::ArrayRef<mlir::Value> positions,
    mlir::Value cell, mlir::Value inverse, mlir::Value packingCells,
    mlir::Type packedReal);

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
/// Whether one class's source body may run on every lane and have its
/// results selected, instead of running under a divergent guard. It holds
/// when every operation of the body is (a) pure, trap-free arithmetic of
/// bounded cost: arith (not remf), vector value operations, and math
/// fma/sqrt/rsqrt/absf/copysign/fpowi with a constant exponent, never a
/// transcendental function or an operation with a region or a call; (b) a
/// table lookup whose indices do not depend on the coordinates, so the
/// guard could not change them; or (c) a load of a constant global whose
/// every index is proved in bounds by its constant/min/max interval (the
/// radial tables clamp their bin, md-exec-expand-radial). A non-finite
/// value of a lane outside the cutoff is then discarded by the select,
/// never multiplied by zero. Any other body keeps the guard.
bool isTemplateBodySpeculatable(mdir::md_exec::TemplateForOp, unsigned tag);
/// Emit one class's actual scalar/derived body with an independent cutoff
/// predicate for every yielded site displacement: a select of the computed
/// value or +0 when isTemplateBodySpeculatable holds, otherwise a guard.
/// Arguments and captures are mapped by the caller. At most one force
/// destination remains after accumulation; energy and strain reductions
/// retain their declared types.
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
  /// Retain O/H1/H2/E forces until the write boundary, then apply rule A.
  bool siteForces = false;
  /// Evaluate every turn on all lanes and select +0 for unpaired lanes
  /// (isTemplateBodySpeculatable), instead of a divergent guard.
  bool speculate = false;
};
/// Borrowed views of the one pair arena and the fixed-size unit state.
/// `work` indexes four-word work records at `workOffset`; entries are
/// four-word records at `entryOffset`. `order` holds padded water places
/// followed by padded atom places, with -1 in empty lanes. Its nonempty
/// values name fixed water units [0,W) or atom units [W,W+A). `members`
/// maps packed real-site rows to the original particle indices. `frames`
/// holds those rows' f32 coordinates, already whole and shifted in f64.
struct TemplateWorkBuffers {
  mlir::Value pairs, workOffset, entryOffset, order, members, frames, waters, cell;
  mlir::Value groupLengths;
  bool hasSite = false;
};
/// Execute an arena work record through its uniformly tagged class and
/// actual source-derived body. Every lane participates, including padding.
/// Returns each lane's global-sum contributions; the caller owns their
/// warp/block reduction. Forces accumulate directly at original particles.
llvm::SmallVector<mlir::Value, 2> emitTemplateWorkUnit(
    mlir::OpBuilder &, mlir::Location, mdir::md_exec::TemplateForOp,
    const TemplateWorkBuffers &, mlir::Value work, mlir::Value lane,
    const mlir::IRMapping &captures);
/// Runs a block of entries of one segment. A partner's frame, mask, scalar
/// parameters and accumulated forces rotate through its half warp together.
/// A lane holds one of 16 I units, repeated in the other half; those halves
/// are combined before the single I write. Each valid J entry is written
/// once, after the rotation. Callbacks supply the actual scalar body and
/// atomic destinations; no pair expression is embedded in this scheduler.
/// The evaluation callback may run divergently (or, with `speculate`, on
/// every lane, its results selected) and must not use warp collectives;
/// the scheduler owns all warp communication.
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
