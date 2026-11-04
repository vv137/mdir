// Owned noise constants and typed operands at the original particle loop.
#ifndef MDIR_LIB_DRIVER_NOISEIR_H
#define MDIR_LIB_DRIVER_NOISEIR_H

#include "mlir/IR/Value.h"
#include <cstdint>

namespace mlir {
class OpBuilder;
}
namespace mdir::driver::builder {
struct NoiseIRInputs {
  mlir::Value velocities, masses, ids, step;
};
struct NoiseIRConfig {
  uint64_t seed;
  double damping, spread;
};
mlir::Value buildNoise(mlir::OpBuilder &, const NoiseIRInputs &,
                        const NoiseIRConfig &);
mlir::Value buildBrownianDrift(mlir::OpBuilder &, mlir::Value forces,
                                mlir::Value masses, double gamma);
} // namespace mdir::driver::builder
#endif
