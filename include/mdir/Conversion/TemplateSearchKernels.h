// One combined periodic search over the two mobility groups.
#ifndef MDIR_CONVERSION_TEMPLATESEARCHKERNELS_H
#define MDIR_CONVERSION_TEMPLATESEARCHKERNELS_H
#include "mdir/Conversion/TemplateBuildKernels.h"
#include "mdir/Conversion/TemplateSpatialKernels.h"
namespace mdir::kernels {
struct TemplateSearchBuffers {
  TemplateSpatialBuffers spatial;
  TemplateBuildBuffers build;
  mlir::Value counters, excludedRows, excludedNeighbors;
  /// Current reduced cell, base reach (cutoff + skin), and covering extent,
  /// all f64. The caller proves baseReach+2*extent < every cell diagonal.
  mlir::Value cell, baseReach, extent;
  /// Optional actual WW/SW/SS reaches in vector<3xf64>, already formed in
  /// the same addition order as their validity margins. Empty uses the
  /// base reach plus covering extent, as in the isolated search fixtures.
  mlir::Value reaches;
  mlir::Value classCells, classReaches;
};
/// Search each active class row once, with explicit periodic images and
/// per-entry masks. Inactive rows copy retained outer records instead.
/// References are fractions in nearby unit-local lattice frames; inactive
/// groups retain their order and references. Keys wrap, but the I bounding
/// box keeps its frame. A partner's entry image compensates its own frame.
/// The spatial bins contain both groups. Clears chunk/flag/entry counters,
/// retaining the extent control.
/// Compaction must wait for the caller to handle overflow flags and an
/// entry total exceeding the final arena capacity (staging includes slack).
void emitSearchTemplatePairs(mlir::OpBuilder &, mlir::Location,
                              const TemplateSearchBuffers &);
}
#endif
