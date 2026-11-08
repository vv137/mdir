// The independent, image-aware displacements of a template window.
#ifndef MDIR_DIALECT_MD_MDTEMPLATEWINDOW_H
#define MDIR_DIALECT_MD_MDTEMPLATEWINDOW_H
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Value.h"
#include <array>
namespace mlir { class OpBuilder; class Location; }
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
/// Shared by native construction and physical packing/validity lowering.
/// The formed site requires a third squared distance only where the O-H
/// bound cannot prove coverage. All arithmetic uses the frame's real type.
mlir::Value emitTemplateExtentSquared(mlir::OpBuilder &, mlir::Location,
    double alpha, bool hasSite, mlir::Value h1, mlir::Value h2);
}
#endif
