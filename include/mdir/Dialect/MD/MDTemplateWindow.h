// The independent, image-aware displacements of a template window.
#ifndef MDIR_DIALECT_MD_MDTEMPLATEWINDOW_H
#define MDIR_DIALECT_MD_MDTEMPLATEWINDOW_H
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include <array>
#include <optional>
namespace mlir { class OpBuilder; class Location; class Operation; class Block; }
namespace mdir::md {
/// Native-builder proof that this operation applies the matched waters'
/// position constraints. Carried through execution lowering so a fresh
/// pair list cannot reset the run's post-constraint extent provenance.
inline constexpr llvm::StringLiteral kTemplateConstraintEpoch =
    "md.template_constraint_epoch";
/// (positive member, negative member), with the anchor image in the first
/// coordinate. Remaining coordinates are whole-molecule H-O offsets.
llvm::SmallVector<std::array<unsigned, 2>, 5>
getTemplateWindowMembers(unsigned arity);
/// Optional per-output site-force layout. Each class row has one flag per
/// force destination; a water has O/H1/H2/E slots before rule A is applied.
inline constexpr llvm::StringLiteral kTemplateSiteForces = "md.template_site_forces";
unsigned getTemplateForceArity(mlir::Operation *, unsigned arity,
                              unsigned column = 0, unsigned outs = 1);
/// Linear rule-A transpose of accumulated O/H1/H2/E forces. A one- or
/// three-site range is unchanged; no normalized-bisector rule is used.
llvm::SmallVector<mlir::Value> emitTemplateForceTranspose(
    mlir::OpBuilder &, mlir::Location, mlir::ValueRange, double alpha);
/// Proved interaction-site indices in the expanded O/H1/H2/E layout.
std::optional<std::array<unsigned, 2>> getTemplateWindowSites(
    mlir::Value displacement, mlir::Block &, unsigned arity, double alpha);
/// Shared by native construction and physical packing/validity lowering.
/// The formed site requires a third squared distance only where the O-H
/// bound cannot prove coverage. All arithmetic uses the frame's real type.
mlir::Value emitTemplateExtentSquared(mlir::OpBuilder &, mlir::Location,
    double alpha, bool hasSite, mlir::Value h1, mlir::Value h2);
}
#endif
