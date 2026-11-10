// The profile map (D398, #840): what each parallel region of the
// host computes, recorded by the lowering that makes it, so that a profile
// is read without guessing from the code of a region.
//
// A record is a dictionary on the location of an op, the metadata of a
// `FusedLoc` named `mdir.profile`. Locations survive every pass that
// rewrites an op into another (`scf.parallel` into `omp.parallel`, the
// LLVM dialect) and are dropped by the translation to LLVM IR, which emits
// no debug information: a record changes neither the code nor the keys of
// the compile caches. The lowerings attach records with these functions;
// the driver reads them from the lowered module (lib/Compiler/ProfileMap.cpp)
// and pairs each with the function that the translation outlines.
//
// The record of an op of md_exec has the keys:
//   source      the op, without its dialect: `pair_for`, `tuple_for`,
//               `particle_for`, `scatter_add`, `refresh_neighbors`, ..., or
//               `template` for a loop of a function of the runtime's
//               templates (`mdrt.pme_spread_...`, `mdrt.build_neighbors_...`)
//   stage       the part of the op, where its lowering makes several loops
//               (`exclusion_filter`, `reference_positions`, ...)
//   set         the set of the loop's rows: a tuple set (`bonds`,
//               `excluded`, `settles`, ...) or the particle set
//   template    the function of a template the loop is in
//   potentials  the potentials whose bodies the kernel holds (`energy`,
//               `term_lj`, ...), from the locations that `md-inline` gives
//               the ops it inlines
//   terms       the terms the kernel holds: those of the potentials of one
//               term (`term_lj` is `lj`) and the custom terms of the
//               control file, from the locations of their expressions; the
//               potential of the sum of the terms (`energy`) names none
//   class       the class of the pairs (WW, SW, SS), where the lowering
//               has one; none on the host
// The precision is read where the map is made, from the arithmetic of the
// region as it was lowered (`getPrecision`).
//
// The lowering of a device (MDExecToGPU) is to attach the same records to
// its kernels, with `describe` on the op a kernel comes from.

#ifndef MDIR_DIALECT_MDEXEC_PROFILEMAP_H
#define MDIR_DIALECT_MDEXEC_PROFILEMAP_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/StringRef.h"

#include <string>

namespace mdir {
namespace profile {

/// The name of the metadata of a record.
constexpr llvm::StringLiteral kRecordName = "mdir.profile";
/// The prefix of the name of a location in the body of a potential.
constexpr llvm::StringLiteral kPotentialPrefix = "potential: ";
/// The prefix of the name of a location of a custom term's expression
/// (lib/Driver/Expression.cpp).
constexpr llvm::StringLiteral kTermPrefix = "control-file term: ";

/// `loc`, inside the body of `potential` (md-inline).
mlir::Location inPotential(mlir::Location loc, llvm::StringRef potential);

/// The record of `op`, an op of md_exec, before its lowering.
mlir::DictionaryAttr describe(mlir::Operation *op);

/// `loc` with `record`, replacing a record it has.
mlir::Location attach(mlir::Location loc, mlir::DictionaryAttr record);

/// The record of `loc`, or null.
mlir::DictionaryAttr find(mlir::Location loc);

/// `loc`, whose record is that of the part `stage` of its op.
mlir::Location refine(mlir::Location loc, llvm::StringRef stage);

/// Gives every op of md_exec under `root` its record, which every op that
/// its lowering creates at its location inherits.
void tagOps(mlir::Operation *root);

/// Gives every `scf.parallel` in `function`, a function of a template of
/// the runtime, the record of a loop of that template.
void tagTemplate(mlir::Operation *function);

/// The floating-point types in which the arithmetic of `region` computes:
/// "f32", "f64", "f32+f64", or "" when it computes in none.
std::string getPrecision(mlir::Region &region);

} // namespace profile
} // namespace mdir

#endif // MDIR_DIALECT_MDEXEC_PROFILEMAP_H
