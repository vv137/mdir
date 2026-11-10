// Refresh-time image-policy selection for template covering search.
#ifndef MDIR_CONVERSION_TEMPLATEIMAGEPOLICY_H
#define MDIR_CONVERSION_TEMPLATEIMAGEPOLICY_H
#include "mlir/IR/Builders.h"

namespace mdir::kernels {
struct TemplateImagePolicy {
  /// i64: 0 = pass image, 1 = pass plus conditional c/b/a neighbors,
  /// 2 = original general ordered iterator. Dispatch before kernel entry.
  mlir::Value path;
  /// Positive outward f64 padding. An ordinary f64 classReach+margin
  /// encloses the old predicate's rounding shell and the image decisions.
  mlir::Value margin;
};

/// cell is vector<3xf64> or vector<6xf64> in the native reduced convention.
/// maxReach bounds all class anchor reaches used by this dispatch.
/// maxAbsFrame bounds abs(floor(reference)) in every component of ALL hybrid
/// references used by this refresh; it is not an arbitrary origin constant.
/// A stricter bound derived from the entry image ABI ensures every
/// provisional nI+s-nJ fits before ordered publication; larger frames use
/// the unchanged general path rather than adding a new overflow refusal.
/// The new image generator uses f64 wrapped-reference Cartesian precompute,
/// f64 c/b/a prefix arithmetic, roundEven of a correctly rounded division
/// (or correctly rounded reciprocal followed by multiplication), and f64
/// other-face comparisons. No FTZ/DAZ/approximate arithmetic is permitted.
/// It re-centers b/a for each higher-axis branch. Conditional enumeration
/// takes at most two choices per axis, retains all accepted images, and
/// separately restores original mask aggregation and emission order.
/// Unsupported/uncertain/nonfinite cases select general, never refusal.
TemplateImagePolicy emitTemplateImagePolicy(
    mlir::OpBuilder &, mlir::Location, mlir::Value cell,
    mlir::Value maxReach, mlir::Value maxAbsFrame);
} // namespace mdir::kernels
#endif
