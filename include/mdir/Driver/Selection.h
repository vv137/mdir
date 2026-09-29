// Selections of particles by the names and numbers of their atoms and
// residues.

#ifndef MDIR_DRIVER_SELECTION_H
#define MDIR_DRIVER_SELECTION_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <vector>

namespace mdir {
namespace driver {

struct Topology;

/// The particles of `topology` that the mask `text` selects, one flag for
/// each in the order of the file. The masks are those of Amber, in part:
///
/// | Form | Selects |
/// |---|---|
/// | `:1-10,15` | Residues by their numbers, from 1 |
/// | `:ALA,GLY` | Residues by their names |
/// | `@1-100` | Atoms by their numbers, from 1 |
/// | `@CA,C,N` | Atoms by their names |
/// | `:1-10@CA` | Atoms of those names in those residues |
/// | `*` | Every particle |
/// | `!a`, `a & b`, `a \| b`, `(a)` | Not, and, or, and grouping, in that order of precedence |
///
/// A name may hold `*`, any run of characters, and `?`, any one character:
/// `@H*` is every atom whose name begins with H.
llvm::Expected<std::vector<bool>> selectParticles(llvm::StringRef text,
                                                  const Topology &topology);

} // namespace driver
} // namespace mdir

#endif // MDIR_DRIVER_SELECTION_H
