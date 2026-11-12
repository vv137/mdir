// Auxiliary search caches; the original f64 reference epoch is retained.
#ifndef MDIR_CONVERSION_TEMPLATESEARCHPREPARATION_H
#define MDIR_CONVERSION_TEMPLATESEARCHPREPARATION_H
#include "mdir/Conversion/TemplateSearchKernels.h"
namespace mdir::kernels {
/// Run after group publication. sortScratch is four i32 words per group
/// place (the unit's f32 Cartesian anchor and its packed lattice frame),
/// groupBoxes is [2*(W+A),3] f32, and fractionalBoxes remains f64.
/// Direct descriptor-column endpoints replace z binary searches. Rebuilds
/// active and retained groups from hybrid references.
void emitPrepareTemplateSearch(mlir::OpBuilder &, mlir::Location,
                               const TemplateSearchBuffers &);
}
#endif
