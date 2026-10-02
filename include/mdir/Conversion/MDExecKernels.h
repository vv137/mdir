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

/// Whether `box`, a cell lowered to a vector, is triclinic: the vector of six,
/// a_x, b_y, c_z, b_x, c_x, c_y (docs/triclinic-m2.md), rather than the three
/// edges of an orthorhombic cell.
bool isTriclinic(mlir::Value box);

/// The edges of `box`, or its diagonal a_x, b_y, c_z if it is triclinic.
mlir::Value getEdges(mlir::OpBuilder &builder, mlir::Location loc,
                     mlir::Value box);

/// The minimum image of the displacement `raw`, with `inverse` from
/// `createInverse`: per edge for an orthorhombic cell, and for a triclinic
/// one in one pass along c, b, and a, which is exact for displacements
/// whose nearest image is within half of the least of a_x, b_y, c_z
/// (docs/triclinic-m2.md, Section 2).
mlir::Value emitMinimumImage(mlir::OpBuilder &builder, mlir::Location loc,
                             mlir::Value raw, mlir::Value box,
                             mlir::Value inverse);

/// The widths of the triclinic cell `box` between its faces, the distances
/// between the planes of constant fractional coordinates: V / |b × c|,
/// b_y c_z / |(c_y, c_z)|, and c_z, in the element type of `box`. The
/// neighbor matrix bins its fractional coordinates with them.
mlir::Value emitFaceWidths(mlir::OpBuilder &builder, mlir::Location loc,
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

/// Where a loop over pairs finds its particles: the positions and the
/// fields of `ins` gathered into the order of the cells of the neighbor
/// structure, whose place `p` holds the particle `order[p]` (D86). The
/// destinations stay in the order of the particles.
struct PairLayout {
  mlir::Value positions;
  llvm::SmallVector<mlir::Value> ins;
  mlir::Value order;
};

/// Emits what a loop over pairs does for the particle `central`: the loop
/// over its neighbors, with the minimum image, the cutoff, and the kernel,
/// and the update of the destinations. Returns the contributions to the
/// global sums, with their weights applied.
///
/// `counts` and `index` are the neighbor matrix. `box` holds the edge
/// lengths of the cell, as a vector of the type of the positions, and
/// `inverse` what `createInverse` returns for it. With `lanes`, a group of
/// threads shares the row (RowLanes). With `outTotals`, the sums of the
/// particle for its destinations are returned there, in every thread of
/// the group, and not stored: the caller stores them. With `layout`,
/// `central` and the entries of the matrix are places in its order.
llvm::SmallVector<mlir::Value>
emitPairKernel(mlir::OpBuilder &builder, md_exec::PairForOp op,
               mlir::Value counts, mlir::Value index, mlir::Value box,
               mlir::Value inverse, mlir::Value central,
               mlir::IRMapping &local, const RowLanes *lanes = nullptr,
               llvm::SmallVectorImpl<mlir::Value> *outTotals = nullptr,
               const PairLayout *layout = nullptr);

/// Where a loop over pairs over groups of 16 (D89, docs/groups-m1.md) finds
/// its work: the lists of the groups in blocks of 64 entries (`entries` and
/// `masks`; block b is entries 64 b to 64 b + 63), the number of entries
/// of each group (`counts`), what each block is (block `ordinals[b]` of the
/// list of group `units[b]`: a block is a unit of work), and the particle
/// at each place (`order`, -1 at an empty place). The positions and the fields of `ins` of `layout` are in
/// the order of the places.
struct GroupLists {
  mlir::Value entries;
  mlir::Value masks;
  mlir::Value counts;
  mlir::Value units;
  mlir::Value ordinals;
  mlir::Value order;
  /// For the inner list of a dual list (D114): the entries that each block
  /// keeps, at its front; `counts` is then not read.
  mlir::Value blockCounts = {};
};

/// Emits what a warp does for the unit of work `unit` of a loop over pairs
/// over groups, each pair once: lanes u and u + 16 hold the particle at
/// place 16 g + u; the entries of the unit, 32 at a time, turn within each
/// half-warp for 16 steps, so that each particle of the group meets each
/// entry once. The value of the kernel goes to the particle of the group
/// and, times the sign of its exchange contract, to the particle of the
/// entry, with atomic additions to the destinations (in the order of the
/// particles); `lane` is the lane in the warp. Returns the contributions of
/// the lane to the global sums, each pair once (no weight): the caller sums
/// them over the warp.
llvm::SmallVector<mlir::Value>
emitGroupPairKernel(mlir::OpBuilder &builder, md_exec::PairForOp op,
                    const GroupLists &lists, const PairLayout &layout,
                    mlir::Value box, mlir::Value inverse, mlir::Value unit,
                    mlir::Value lane, mlir::IRMapping &local);

/// Emits what a loop over tuples does for the particle `particle`: the loop
/// over the tuples in its row of `incidence`, with the displacements in the
/// minimum image, the kernel, and the update of the destinations with the
/// values that the kernel yields for the place of the particle. Returns the
/// contributions to the global sums, which only the tuples where the
/// particle is at place 0 make.
///
/// `box`, `inverse`, `lanes`, and `outTotals` are as for `emitPairKernel`;
/// `outTotals` is for a set that is not disjoint.
llvm::SmallVector<mlir::Value>
emitTupleKernel(mlir::OpBuilder &builder, md_exec::TupleForOp op,
                mlir::Value incidence, mlir::Value box, mlir::Value inverse,
                mlir::Value particle, mlir::IRMapping &local,
                const RowLanes *lanes = nullptr,
                llvm::SmallVectorImpl<mlir::Value> *outTotals = nullptr);

/// A loop over particles, a run of loops over disjoint tuples that read
/// what it wrote, and a loop over particles that reads what they all wrote:
/// the kick and drift of a step, the corrections of the constraints, and
/// the update of the positions and velocities, which one kernel does
/// (D110). The loops over tuples write their destinations as before; the
/// loop before writes its destinations through the loop after, which
/// writes them in place.
struct IntegrationRun {
  md_exec::ParticleForOp before;
  llvm::SmallVector<md_exec::TupleForOp, 4> loops;
  md_exec::ParticleForOp after;
  /// For each destination of each loop over tuples, whether anything reads
  /// it after the loop after; the loop after takes the values from
  /// registers, so a destination that nothing else reads is not written.
  llvm::SmallVector<llvm::SmallVector<bool, 2>, 4> keepOuts;
};

/// The particles that a warp of the first kernel of an integration run
/// takes at most past the boundaries of the tuples: 32 less the widest
/// arity less one.
int64_t getIntegrationStride(const IntegrationRun &run);

/// Emits what one thread of a kernel of an integration run (D110) does
/// for `thread`: the loop before, the tuple of the particle if any (at
/// most one over the loops, the sets being disjoint), and the loop after,
/// whose values of the loop before and of the tuple stay in registers.
/// `boxes` and `inverses` are those of the loops over tuples, in the type
/// of their kernels. `storeAfter` is
/// called for each particle whose values the thread writes, with the
/// contributions of the loop after to its reductions, in their order.
///
/// The run takes two kernels. The first (`across` false, 32 threads for
/// every `getIntegrationStride` of the `span` particles) handles the
/// particles in no tuple and the tuples whose members lie in one warp,
/// where the member at place 0 gathers the values of the loop before from
/// the lanes of the members and the members take their values of the
/// tuple from its lane; a warp takes whole tuples where their members
/// follow one another. It lists in `acrossList` (counted in
/// `acrossCount`, zero before) the members at place 0 of the other
/// tuples that lie across warps, whose members write their values of the
/// loop before. The second (`across` true, one block of `span` threads)
/// handles those tuples from those values and clears the count. Spread
/// over the warps of the first kernel, such tuples would hold each warp
/// for the latency of a tuple.
void emitIntegrationThread(
    mlir::OpBuilder &builder, const IntegrationRun &run,
    llvm::ArrayRef<mlir::Value> boxes, llvm::ArrayRef<mlir::Value> inverses,
    mlir::Value thread, mlir::Value span, mlir::Value acrossList,
    mlir::Value acrossCount,
    bool across,
    llvm::function_ref<void(mlir::OpBuilder &, mlir::Value,
                            llvm::ArrayRef<mlir::Value>)>
        storeAfter);

/// Emits what a loop over tuples does for the tuple `tuple` alone, on a
/// device, where each tuple is evaluated once: the members from `members`,
/// a buffer of a row of members for each tuple, the kernel, and the values
/// for the destinations added to the members with atomics, whose order the
/// threads decide (not in the deterministic mode, D84). The loop has no
/// global sums, and adds to its destinations.
void emitTupleOnce(mlir::OpBuilder &builder, md_exec::TupleForOp op,
                   mlir::Value members, mlir::Value tuple, mlir::Value box,
                   mlir::Value inverse, mlir::IRMapping &local);

/// Emits what a thread of a loop over tuples with global sums does where
/// each tuple is evaluated once (D103): the tuples `first`, `first +
/// stride`, ... below `count`, each as emitTupleOnce does it, and returns
/// the sums of their contributions to the global sums of the loop, in the
/// order of the tuples, each summed in the type the kernel computes it in
/// and widened once.
llvm::SmallVector<mlir::Value>
emitTuplesOnceWithSums(mlir::OpBuilder &builder, md_exec::TupleForOp op,
                       mlir::Value members, mlir::Value first,
                       mlir::Value stride, mlir::Value count, mlir::Value box,
                       mlir::Value inverse);

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
/// type of the splines and the grid, takes the type of the forces, and
/// `PME_ORDER` the order of the splines, a constant in its kernels.
std::string instantiatePMETemplates(llvm::StringRef text, mlir::Type position,
                                    mlir::Type charge, mlir::Type force,
                                    int64_t order);
std::string getPMEInstanceName(llvm::StringRef name, mlir::Type position,
                               mlir::Type charge, mlir::Type force,
                               int64_t order);

} // namespace kernels
} // namespace mdir

#endif // MDIR_CONVERSION_MDEXECKERNELS_H
