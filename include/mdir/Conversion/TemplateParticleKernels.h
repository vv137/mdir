// Position-writer ownership for complete rigid-water frames.
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
/// One thread per original particle, with O threads owning complete OHH
/// updates and H threads idle. All inputs of all three members are read
/// before any output is stored, including in-place position updates.
/// Unmatched atoms and stored extra points retain their original updates.
/// `slots` maps original particles to packed real-site rows (-1 for a
/// template extra point); `members` is the inverse real-site map. `waters`
/// is index typed. The callback runs once per unit after its stores, with
/// three results for a water or one otherwise, and receives the fixed i32
/// unit ID (-1 for a template extra point). It can pack a complete frame
/// and publish validity in the writer's kernel, without a cross-thread
/// read of positions. No particle expression is embedded in this helper.
void emitTemplateParticleKernel(mlir::OpBuilder &, mlir::Location,
    mdir::md_exec::ParticleForOp, mlir::Value particle,
    mlir::Value slots, mlir::Value members, mlir::Value waters,
    const mlir::IRMapping &captures,
    llvm::function_ref<void(mlir::OpBuilder &, mlir::Value,
                            llvm::ArrayRef<TemplateParticleResult>)> finish);
}
#endif
