// Certified Cartesian geometry for the template covering search.
#ifndef MDIR_CONVERSION_TEMPLATESEARCHGEOMETRY_H
#define MDIR_CONVERSION_TEMPLATESEARCHGEOMETRY_H
#include "mlir/IR/Builders.h"

namespace mdir::kernels {
struct TemplateSearchCertificate {
  /// Outward f32 squared-norm thresholds. Fast acceptance requires
  /// enabled && acceptEnabled && Q32 <= lower; rejection requires
  /// enabled && Q32 > upper. Other cases use the literal original f64 graph.
  mlir::Value lower, upper;
  /// A finite f32 Cartesian box norm may reject only if enabled and
  /// Qbox32 > boxUpper. Otherwise the box must remain a candidate.
  mlir::Value boxUpper;
  mlir::Value enabled, acceptEnabled;
};

/// Frame-wide setup for c32 = truncf(H64 * wrapped64), image shift
/// truncf(H64) * exact32(s), and d32 = cI32 - (cJ32 + shift32).
/// cell is vector<3xf64> or vector<6xf64>, frame is the integer-valued
/// vector<3xf64> first-reference lattice frame, imageEnvelope is a
/// vector<3xf64> outward bound on the absolute query-image components, and
/// reach2 is the LITERAL original
/// rounded f64 reach*reach. Cache and boxes share the same f64 Cartesian
/// precompute. All f32 operations must preserve gradual underflow (no DAZ
/// or FTZ); emitted arithmetic must be audited before enabling this path.
/// The caller retains original references/images/ownership and invokes
/// the unchanged f64 predicate in the uncertainty shell or if !enabled.
/// Nonfinite Q32 falls back; nonfinite box geometry means MAY REACH.
/// No bound is additional physical skin or a certified class reach.
TemplateSearchCertificate emitTemplateSearchCertificate(
    mlir::OpBuilder &, mlir::Location, mlir::Value cell,
    mlir::Value frame, mlir::Value imageEnvelope, mlir::Value reach2);
} // namespace mdir::kernels
#endif
