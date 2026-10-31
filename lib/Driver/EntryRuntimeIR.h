// Native runtime prototypes retain their original declaration phases.
#ifndef MDIR_LIB_DRIVER_ENTRYRUNTIMEIR_H
#define MDIR_LIB_DRIVER_ENTRYRUNTIMEIR_H
#include <cstdint>
namespace mlir { class OpBuilder; }
namespace mdir::driver::builder {
struct RuntimeDeclarationConfig {
  bool state64, force64, tunableGradient, frames, segments;
  int64_t partCounts;
  bool minimize, stopsMinimization, solvent, constraintStops;
  bool rescales, noseHoover, bath, barostat, barostatState, checkpoints;
  bool writesForces;
};

void emitRuntimeDeclarations(mlir::OpBuilder &, const RuntimeDeclarationConfig &, bool afterModels);
}
#endif
