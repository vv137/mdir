// What the lowerings of md_exec have in common: the code that a loop runs
// for one particle. The lowerings differ in how they run it for all.

#ifndef MDIR_CONVERSION_MDEXECKERNELS_H
#define MDIR_CONVERSION_MDEXECKERNELS_H

#include "mdir/Dialect/MDExec/MDExecOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"

#include <functional>

namespace mdir {
namespace kernels {

mlir::Value createIndex(mlir::OpBuilder &builder, mlir::Location loc,
                        int64_t value);
mlir::Value createZero(mlir::OpBuilder &builder, mlir::Location loc,
                       mlir::Type type);

/// The constant `value` of the floating-point type `real`.
mlir::Value createReal(mlir::OpBuilder &builder, mlir::Location loc,
                       mlir::Type real, double value);

/// `value`, a floating-point value or a vector of them, converted to the
/// floating-point type `real`.
mlir::Value convertReal(mlir::OpBuilder &builder, mlir::Location loc,
                        mlir::Value value, mlir::Type real);

/// One over each edge length of `box`. A kernel finds the minimum image
/// with a multiplication in place of a division.
///
/// The number of images is a whole number, so the displacement is the same
/// as with the division unless the two round to different whole numbers.
/// They do so only where the displacement along an edge is within rounding
/// of half the edge length, which is beyond every cutoff.
mlir::Value createInverse(mlir::OpBuilder &builder, mlir::Location loc,
                          mlir::Value box);

/// The value that `buffer` holds for the particle `particle`.
mlir::Value loadElement(mlir::OpBuilder &builder, mlir::Location loc,
                        mlir::Value buffer, mlir::Value particle);
void storeElement(mlir::OpBuilder &builder, mlir::Location loc,
                  mlir::Value value, mlir::Value buffer,
                  mlir::Value particle);

/// Emits what a loop over particles does for the particle `particle`: it
/// loads the values of the fields, runs the kernel, and stores what the
/// kernel yields for the destinations. Returns the contributions to the
/// global sums.
///
/// `local` maps values from outside the kernel to the values that stand for
/// them where the code is emitted.
llvm::SmallVector<mlir::Value>
emitParticleKernel(mlir::OpBuilder &builder, md_exec::ParticleForOp op,
                   mlir::Value particle, mlir::IRMapping &local);

/// How the threads of a group share the row of one particle, of neighbors
/// or of tuples: the thread `lane` of `lanes` takes every `lanes`th entry
/// of the row from
/// `lane` on, `combine` sums a value over the group and gives the sum to
/// every thread of it, and only the first thread of a group whose particle
/// is `valid` writes. The threads of a group whose particle is not valid
/// take no entry but still combine, as a combination may need every thread.
struct RowLanes {
  mlir::Value lane;
  int64_t lanes;
  mlir::Value valid;
  std::function<mlir::Value(mlir::OpBuilder &, mlir::Location, mlir::Value)>
      combine;
};

/// Emits what a loop over pairs does for the particle `central`: the loop
/// over its neighbors, with the minimum image, the cutoff, and the kernel,
/// and the update of the destinations. Returns the contributions to the
/// global sums, with their weights applied.
///
/// `counts` and `index` are the neighbor matrix. `box` holds the edge
/// lengths of the cell, as a vector of the type of the positions, and
/// `inverse` what `createInverse` returns for it. With `lanes`, a group of
/// threads shares the row (RowLanes).
llvm::SmallVector<mlir::Value>
emitPairKernel(mlir::OpBuilder &builder, md_exec::PairForOp op,
               mlir::Value counts, mlir::Value index, mlir::Value box,
               mlir::Value inverse, mlir::Value central,
               mlir::IRMapping &local, const RowLanes *lanes = nullptr);

/// Emits what a loop over tuples does for the particle `particle`: the loop
/// over the tuples in its row of `incidence`, with the displacements in the
/// minimum image, the kernel, and the update of the destinations with the
/// values that the kernel yields for the place of the particle. Returns the
/// contributions to the global sums, which only the tuples where the
/// particle is at place 0 make.
///
/// `box`, `inverse`, and `lanes` are as for `emitPairKernel`.
llvm::SmallVector<mlir::Value>
emitTupleKernel(mlir::OpBuilder &builder, md_exec::TupleForOp op,
                mlir::Value incidence, mlir::Value box, mlir::Value inverse,
                mlir::Value particle, mlir::IRMapping &local,
                const RowLanes *lanes = nullptr);

/// Emits, on the host, the build of the incidence structure of the tuples
/// that `members` holds, for `size` particles, and returns it in a new
/// buffer of the host. A row holds the number of tuples of the particle,
/// then for each tuple its number, the place of the particle in it, and
/// its members. The tuples of a row are in the order of their numbers, and
/// a row is as wide as the particle with the most tuples needs.
mlir::Value emitBuildIncidence(mlir::OpBuilder &builder, mlir::Location loc,
                               mlir::Value members, mlir::Value size);

/// Emits what the particle `particle` does to leave the pairs of the
/// incidence structure `excluded`, a structure of pairs, out of its row of
/// the neighbor matrix `counts`, `index`: it keeps the other neighbors, in
/// their order, and stores their number. A row whose count exceeds its
/// width has been reported as too narrow and is left alone.
void emitExclusionFilter(mlir::OpBuilder &builder, mlir::Location loc,
                         mlir::Value counts, mlir::Value index,
                         mlir::Value excluded, mlir::Value particle);

/// Replaces every `md.lookup` in `root` by a load from the buffer that holds
/// the table, converted to the type of the result.
void lowerLookups(mlir::Operation *root);

/// Emits, on the host, the members `members`, buffers of the host in the
/// order of the files, at the places that `ids`, a buffer of the host, gives
/// the particles: a member `m` becomes the place `p` with `ids[p] = m`.
/// Returns them in a new buffer of the host.
mlir::Value emitRenumber(mlir::OpBuilder &builder, mlir::Location loc,
                         mlir::Value members, mlir::Value ids);

/// Frees `buffer`, a buffer of the host, where the block of `op` ends.
void freeAtEndOfBlock(mlir::Operation *op, mlir::Value buffer);

/// The text of the templates `text` for positions of the type `real`. The
/// templates are written for `f64`; the functions of the instance for `f32`
/// have names that end in `_f32`.
std::string instantiateTemplates(llvm::StringRef text, mlir::Type real);
std::string getInstanceName(llvm::StringRef name, mlir::Type real);

/// The templates of particle mesh Ewald for positions, charges, and forces
/// stored as `position`, `charge`, and `force`: the aliases `!pme_pos`,
/// `!pme_chg`, and `!pme_frc` take those types, the conversions from and to
/// f64 become the ops that they need, and the functions take the suffix of
/// `getPMESuffix`. The alias `!pme_real` of the template for devices, the
/// type of the splines and the grid, takes the type of the forces.
std::string instantiatePMETemplates(llvm::StringRef text, mlir::Type position,
                                    mlir::Type charge, mlir::Type force);
std::string getPMEInstanceName(llvm::StringRef name, mlir::Type position,
                               mlir::Type charge, mlir::Type force);

} // namespace kernels
} // namespace mdir

#endif // MDIR_CONVERSION_MDEXECKERNELS_H
