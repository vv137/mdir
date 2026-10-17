// How each reciprocal sum of PME on a device spreads its charges.
//
// See the description of the pass in Passes.td, and D[pme-tiles] in
// docs/decisions.md.

#include "mdir/Dialect/MDExec/Transforms/Passes.h"

#include "mdir/Dialect/MD/MDTypes.h"
#include "mdir/Dialect/MDExec/MDExecOps.h"
#include "mlir/IR/Builders.h"

using namespace mlir;
using namespace mdir;
using namespace mdir::md_exec;

namespace mdir {
namespace md_exec {

#define GEN_PASS_DEF_CHOOSESPREADING
#include "mdir/Dialect/MDExec/Transforms/Passes.h.inc"

namespace {
/// The bytes of the memory of a block that a tile may take: the static
/// limit of a block on every device of compute capability 7.0 or later.
constexpr int64_t kTileBytes = 48 * 1024;

class ChooseSpreading : public impl::ChooseSpreadingBase<ChooseSpreading> {
public:
  using impl::ChooseSpreadingBase<ChooseSpreading>::ChooseSpreadingBase;

  void runOnOperation() final {
    if (layout != "auto" && layout != "global" && layout != "tiles") {
      getOperation()->emitError()
          << "expected the layout 'auto', 'global', or 'tiles', got '"
          << static_cast<std::string>(layout) << "'";
      return signalPassFailure();
    }
    // The edges for which the template has kernels
    // (TILE_EDGES of scripts/generate-pme-gpu-template.py).
    if (edge != 4 && edge != 8 && edge != 16) {
      getOperation()->emitError()
          << "expected an edge of the tiles of 4, 8, or 16 points, got "
          << static_cast<int64_t>(edge);
      return signalPassFailure();
    }
    if (layout == "global" || deterministic)
      return;
    Builder builder(&getContext());
    getOperation()->walk([&](ReciprocalOp op) {
      if (op.isStorageForm())
        return;
      // The type of the grid: that of the forces (D271).
      auto forces = dyn_cast_or_null<md::FieldType>(
          op.getForces() ? op.getForces().getType() : Type());
      if (!forces)
        return;
      Type element = forces.getElementType();
      if (!element.isF32() && !element.isF64())
        return;
      // The tile, and the charges, fractions, and first points of a round
      // of one particle a thread (tile_kernel and TILE_THREADS of
      // scripts/generate-pme-gpu-template.py).
      int64_t threads = edge == 16 ? 512 : 128;
      int64_t side = edge + op.getOrder() - 1;
      int64_t real = element.isF32() ? 4 : 8;
      int64_t bytes =
          (side * side * side + 4 * threads) * real + 3 * threads * 4;
      if (bytes > kTileBytes)
        return;
      int64_t points = 1;
      for (int64_t length : op.getGrid()) {
        if (length < side)
          return;
        points *= length;
      }
      // With `auto`, a grid in f64 keeps the spreading of before, which is
      // faster there on the devices measured (their f64 is slow, and the
      // atomics of a tile are a loop of compare-and-swap of 64 bits); so
      // does a grid of fewer than `min-points` points, whose bins are too
      // few to occupy the device (on JAC, 512 bins of 8 points, the tiles
      // were slower on an RTX PRO 6000 at every order). Every other
      // spreading takes the tiles that fit.
      if (layout == "auto" && (!element.isF32() || points < minPoints))
        return;
      op.setTilesAttr(builder.getI64IntegerAttr(edge));
    });
  }
};
} // namespace

} // namespace md_exec
} // namespace mdir
