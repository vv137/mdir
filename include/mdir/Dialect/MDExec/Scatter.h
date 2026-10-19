// The strategies of a scatter-accumulate and their preconditions
// (D285, docs/device-primitives.md).
//
// A scatter-accumulate adds values into a field at indices that the
// threads of a device share: the charges of PME into its grid, the values
// of the tuples into the forces of their members. `#md_exec.scatter` on
// the op states how. `md-exec-choose-scatter` chooses the strategy where
// its preconditions hold, and the lowerings that apply it check them with
// these same functions (docs/principles.md, principle 2).

#ifndef MDIR_DIALECT_MDEXEC_SCATTER_H
#define MDIR_DIALECT_MDEXEC_SCATTER_H

#include "mdir/Dialect/MDExec/MDExecOps.h"

#include "llvm/ADT/ArrayRef.h"

#include <optional>
#include <string>

namespace mdir {
namespace md_exec {
namespace scatter {

/// Why a strategy does not apply, or nothing where it does.
using Refusal = std::optional<std::string>;

/// The block memory that a kernel may take without asking for more: the
/// static limit of a block on every device of compute capability 7.0 or
/// later. Neither strategy asks for more, so this bound is also what keeps
/// a block's occupancy where the measurements of D282 and D283 put it.
constexpr int64_t kBlockMemoryBytes = 48 * 1024;

/// Whether the values may be added in another order than the atomics of
/// the device add them. A privatized sum adds the values of a block first
/// and the block's sums afterwards: exact for integers in any order, but
/// for floating-point numbers another rounding, which the default mode
/// allows (the atomics already add in an order the threads decide) and the
/// deterministic mode does not. Fixed point is not privatized yet (#416,
/// step 2), so every element type other than f32 and f64 is refused.
Refusal checkReorder(mlir::Type element, bool deterministic);

//===----------------------------------------------------------------------===//
// tile(E): the dense tile of a bin (D282)
//===----------------------------------------------------------------------===//

/// Whether the template of PME has the kernels of bins of `edge` points
/// (TILE_EDGES of scripts/generate-pme-gpu-template.py).
bool isTileEdge(int64_t edge);
/// The threads of a block of the tiles of `edge` (TILE_THREADS of the
/// script).
int64_t getTileThreads(int64_t edge);
/// The points of a side of a tile: the bin and the stencil of order
/// `order` past it, E + P - 1.
int64_t getTileSide(int64_t edge, int64_t order);
/// The block memory of a tile in `element`, with the charges, fractions,
/// and first points of a round of one particle a thread (tile_kernel of
/// the script).
int64_t getTileBytes(int64_t edge, int64_t order, mlir::Type element);
/// The preconditions of `tile(edge)` on a reciprocal sum of order `order`
/// on `grid` with its grid in `element`: the reordering, an edge of the
/// template, the block memory, and a footprint that wraps around the grid
/// once at most (every edge of the grid at least the side of a tile).
Refusal checkTile(int64_t edge, int64_t order, llvm::ArrayRef<int64_t> grid,
                  mlir::Type element, bool deterministic);
/// The same for `op`, its grid in the type of its forces (D271), in the
/// value or the storage form.
Refusal checkTile(ReciprocalOp op, int64_t edge, bool deterministic);

//===----------------------------------------------------------------------===//
// hash(S, P, L): the table of a block (D283)
//===----------------------------------------------------------------------===//

/// The slots for a block of `blockSize` threads that each add to `arity`
/// members: room for every member, rounded up to a power of two, so that a
/// slot is a mask of the hash.
int64_t getHashSlots(int64_t blockSize, int64_t arity);
/// The particles that a thread of a loop over groups adds to for a unit of
/// work, in place of the arity of a tuple: the particle of its group and
/// those of two entries, a unit holding 64 entries for 32 lanes
/// (D[scatter-op]).
constexpr int64_t kPairHashReach = 3;

/// A destination of a table: a number or a row of `components` numbers of
/// `element` for each slot.
struct Destination {
  mlir::Type element;
  int64_t components = 1;
};
/// The destination that a field or a buffer of one takes in a table, or
/// why it takes none: the fallback is an atomic of the device into it,
/// which the lowering has for numbers and rows of a static length of f32
/// and f64.
Refusal getHashDestination(mlir::Type fieldOrBuffer, Destination &out);
/// The block memory of a table of `slots` slots with these destinations:
/// a key of 32 bits and the sums of each destination for each slot.
int64_t getHashBytes(int64_t slots, llvm::ArrayRef<Destination> outs);
/// The preconditions of `hash` with `slots` on destinations `outs`: the
/// reordering, the fallback of each destination, and the block memory.
/// `destinations` takes the destinations where they are given.
Refusal checkHash(int64_t slots, mlir::TypeRange outs, bool deterministic,
                  llvm::SmallVectorImpl<Destination> *destinations = nullptr);

} // namespace scatter
} // namespace md_exec
} // namespace mdir

#endif // MDIR_DIALECT_MDEXEC_SCATTER_H
