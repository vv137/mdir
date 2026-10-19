// The declared transposes of the ops that md-differentiate carries
// adjoints through (D276, D295).

#ifndef MDIR_DIALECT_MD_TRANSFORMS_ADJOINTINTERFACE_H
#define MDIR_DIALECT_MD_TRANSFORMS_ADJOINTINTERFACE_H

#include "mdir/Dialect/MD/MDEnums.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir { class DialectRegistry; }
namespace mdir { namespace md {

/// The kernel of the op of a transposed structure, as `createTranspose`
/// leaves it for the caller to complete.
///
/// - An op over pairs or tuples whose kernel the transpose evaluates as the
///   source does (the sums over a relation and over tuples): `block` holds
///   a copy of the source kernel, with its terminator, and `cloned` is set.
///   The cotangent is a number that the caller's kernel captures.
/// - A gather over a relation: `block` has the arguments of the source
///   kernel (the distance, the displacement, the values of each gathered
///   field at both ends), then the cotangent at the center and at the
///   neighbor, `cotangentI` and `cotangentJ`; `forward` maps the source
///   kernel's arguments to evaluate it with the center at the center, and
///   `backward` with the ends exchanged and the displacement negated
///   (each pair's kernel from both of its ends).
/// - A map or a sum over particles: `block` has the arguments of the source
///   kernel, mapped by `forward`, then, for the adjoint of a map, the
///   cotangent at the particle, `cotangentI`; for a sum the cotangent is
///   the number itself, which the kernel captures.
///
/// No op but the copy of the source kernel is in `block`, except the
/// negated displacement that `backward` maps to, which the caller's
/// `eraseDeadOps` removes where unused.
struct TransposeKernel {
  mlir::Block *block = nullptr;
  bool cloned = false;
  mlir::IRMapping forward;
  mlir::IRMapping backward;
  mlir::Value cotangentI;
  mlir::Value cotangentJ;
};

/// Attaches the declared transposes: those of the transfers of md_dist (G
/// and G^T, the distribution of a field and its collection, and the sum of
/// the partial sums of the domains), and those of the structure of the ops
/// of md with a kernel.
void registerAdjointInterfaces(mlir::DialectRegistry &registry);
} }

#include "mdir/Dialect/MD/Transforms/AdjointInterface.h.inc"
#endif
