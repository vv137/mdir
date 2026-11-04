// Native runtime prototypes retain their original declaration phases.
#ifndef MDIR_LIB_DRIVER_ENTRYRUNTIMEIR_H
#define MDIR_LIB_DRIVER_ENTRYRUNTIMEIR_H
#include <cstdint>
#include "mlir/IR/Value.h"
namespace mlir { class OpBuilder; }
namespace mdir::driver::builder {
struct RuntimeDeclarationConfig {
  bool state64, force64, tunableGradient, frames, segments;
  int64_t partCounts;
  bool minimize, stopsMinimization, solvent, constraintStops;
  bool rescales, noseHoover, bath, barostat, barostatState, checkpoints;
  bool writesForces;
};

// The part boundary resorts all three physical buffers in place.
void buildPartBoundary(mlir::OpBuilder &, mlir::Value positions, mlir::Value velocities,
                       mlir::Value forces, mlir::Value identities,
                       mlir::Value invalid, mlir::Value counts);

void emitRuntimeDeclarations(mlir::OpBuilder &, const RuntimeDeclarationConfig &, bool afterModels);
}
#endif
