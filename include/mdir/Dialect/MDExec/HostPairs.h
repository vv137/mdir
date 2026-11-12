// The lanes of a row of a loop over pairs on the host and their
// precondition (D402, docs/device-primitives.md).
//
// On the CPU target a loop over the pairs of the neighbor matrix may run
// each row `W` candidates at a time in registers of the `vector` dialect,
// `host(lanes(W))` on `md_exec.pair_for`. `md-exec-choose-host-pairs` sets
// it where `checkLanes` holds and the host's policy selects it; the host
// lowering (`convert-md-exec-to-loops`) checks it again with the same
// function and fails on an attribute that it refuses
// (docs/principles.md, principle 2). The emitter of the lanes
// (`kernels::emitPairRowLanes`) maps each op of the kernel by
// `getLaneMapping`, the same classification that `checkLanes` reads.

#ifndef MDIR_DIALECT_MDEXEC_HOSTPAIRS_H
#define MDIR_DIALECT_MDEXEC_HOSTPAIRS_H

#include "mdir/Dialect/MDExec/MDExecOps.h"

#include "llvm/ADT/StringRef.h"

#include <optional>
#include <string>

namespace mdir {
namespace md_exec {
namespace hostpairs {

/// Why the lanes do not apply, or nothing where they do.
using Refusal = std::optional<std::string>;

/// The most lanes of a row: 64, a register of 512 bits in bytes.
constexpr int64_t kMaxLanes = 64;

/// How the lanes evaluate an op of the kernel.
enum class LaneMapping {
  /// Not evaluated in lanes: the precondition refuses the kernel.
  None,
  /// `arith.constant`, `memref.get_global`: one value for every lane.
  Uniform,
  /// An op of `arith` or `math` that maps elementwise: the same op on
  /// vectors of the lanes, one a component.
  Elementwise,
  /// `math.fpowi` with a constant exponent: `llvm.intr.powi` on the
  /// vector of the lanes with the scalar exponent, which LLVM's code
  /// generator expands as it expands the scalar one.
  PowI,
  /// `vector.extract` at a constant place of a vector of one dimension.
  Extract,
  /// `vector.broadcast` of a number to a vector of one dimension.
  Broadcast,
  /// `vector.from_elements` of a vector of one dimension.
  FromElements,
  /// A read at an index of a lane (`md.lookup`, `vector.load`,
  /// `memref.load`): one read a lane, gathered into the components.
  Gather,
  /// `md_exec.yield`.
  Yield,
};

/// The mapping of `op`, an op of the kernel of a loop over pairs.
LaneMapping getLaneMapping(mlir::Operation *op);

/// Why `lanes` is not a width of the lanes: a power of two from 1 to
/// kMaxLanes.
Refusal checkWidth(int64_t lanes);

/// The precondition of `host(lanes(lanes))` on `op`, in the value or the
/// storage form:
/// 1. the policy `(directed, owner_only)`, without `centers`,
///    `center_set`, `ghost_free`, or `images(explicit)`;
/// 2. a kernel each of whose ops has a lane mapping, without regions,
///    with the exponent of each `math.fpowi` a constant;
/// 3. the kernel computes in f32 or f64, and its values are numbers,
///    vectors of one dimension, or the globals of its tables;
/// 4. the op states the algebra of what it reduces (D301): the lanes
///    order a row's sum by `lanes` alone, which every algebra allows, as
///    the order of the scalar row is a function of the list;
/// 5. `checkWidth(lanes)`.
Refusal checkLanes(PairForOp op, int64_t lanes);

/// The type that the kernel of `op` computes in: that of its squared
/// distance.
mlir::Type getComputedType(PairForOp op);

/// The lanes of a register of `vectorBits` in `element`.
int64_t getLanes(int64_t vectorBits, mlir::Type element);

/// The lanes of the deterministic mode, whatever the host: 8 in f32 and 4
/// in f64 (the maintainer's ruling on PR #823). A narrower host holds them
/// in several registers, in the same order, and a wider one does not widen,
/// so that the bits do not depend on the host's vector width.
int64_t getDeterministicLanes(mlir::Type element);

/// The measured policy of a host (#433's host column), keyed by the CPU
/// that LLVM names for it (`znver3`): a host whose measurement made the
/// lanes its default, and that measurement.
struct Policy {
  llvm::StringRef cpu;
  llvm::StringRef measured;
};

/// The policy of `cpu`, or nothing for a host whose default is the scalar
/// row (not measured, or measured without a gain).
std::optional<Policy> findPolicy(llvm::StringRef cpu);

} // namespace hostpairs
} // namespace md_exec
} // namespace mdir

#endif // MDIR_DIALECT_MDEXEC_HOSTPAIRS_H
