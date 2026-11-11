// The final coordinates of one unit, as the template writer takes them.
#ifndef MDIR_CONVERSION_TEMPLATEPARTICLEKERNELS_H
#define MDIR_CONVERSION_TEMPLATEPARTICLEKERNELS_H
#include "mlir/IR/Builders.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLFunctionalExtras.h"
namespace mlir { class IRMapping; }
namespace mdir::md_exec { class ParticleForOp; }
namespace mdir::kernels {
struct TemplateParticleResult {
  mlir::Value particle;
  llvm::SmallVector<mlir::Value> outputs, reductions;
};
}
#endif
