// The automatic choice of the skins of the neighbor lists (D281).
//
// See docs/neighbors.md, "The choice of the skins".
#ifndef MDIR_COMPILER_SKINS_H
#define MDIR_COMPILER_SKINS_H

#include "mdir/Compiler/CompileCache.h"
#include "mdir/Driver/Control.h"
#include "mdir/Driver/System.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"

#include <optional>
#include <string>
#include <vector>

namespace mdir {
namespace compiler {

/// One candidate pair of reaches and what its measurement gave.
struct SkinTrial {
  /// The reaches, in Å; `inner` 0 for one list.
  double outer = 0.0, inner = 0.0;
  /// What the cost model predicted for it, relative to the reference.
  double predicted = 1.0;
  /// The time of a step in µs, the median of its blocks, and the relative
  /// standard error of their mean.
  double microseconds = 0.0, error = 0.0;
  /// The steps between builds and between prunings in the blocks.
  double buildInterval = 0.0, pruneInterval = 0.0;
  /// The seconds of its compilation and of its steps; of the compilation,
  /// those of the MLIR pipeline, of the serialization of the kernels, and
  /// of the host code.
  double compileSeconds = 0.0, runSeconds = 0.0;
  double pipelineSeconds = 0.0, kernelSeconds = 0.0, hostSeconds = 0.0;
  /// What its compilation took from the compile cache and stored there.
  CompileStats cache;
  /// Empty, or why it could not be measured.
  std::string failure;
};

/// What a choice of the skins did.
struct SkinChoice {
  /// The candidates measured, the reference first, if there was a search.
  std::vector<SkinTrial> trials;
  /// The file of the cache that was read or written, if any.
  std::string cachePath;
  /// The seconds the search took, compilations included.
  double seconds = 0.0;
  /// Whether this run searched (measured candidates and chose among them).
  bool searched = false;
  /// The candidates the run afforded (D[skin-budget-record]), once the
  /// reference was measured; and what it was computed from, as JSON.
  std::optional<int64_t> budget;
  llvm::json::Value budgetBasis = nullptr;
  /// Why nothing was cached although the reference was measured, or empty.
  std::string notCached;
  /// Why the entry of the cache for the key was refused, or empty, and its
  /// file.
  std::string refused, refusedPath;
  /// The search that an entry of the cache records, where the choice was
  /// reused from it: its budget, candidates, reason, and seconds.
  std::optional<llvm::json::Object> recorded;
  /// The candidates and the result, for the manifest.
  llvm::json::Value toJSON(const driver::Control &control) const;
};

/// Where a search would apply: whether `control`, with its neighbor
/// structure resolved (resolveNeighborStructure), is one whose reaches may
/// be measured, and why not otherwise. The length of the run is not part of
/// it (see `chooseSkins`).
bool canSearchSkins(const driver::Control &control, std::string &why);

/// Resolves the reaches of `control` that are "AUTO" (D281): from
/// the defaults that resolveNeighborStructure set; then, where
/// canSearchSkins holds, from the cache of earlier choices; then, where
/// `search` is true and the run is long enough (or MDIR_SKIN_SEARCH says
/// so), by measuring candidates on `system`, whose positions and velocities
/// are those the run begins with. Sets `pairlistDistance`,
/// `prunedDistance`, `skinSource`, and `skinReason` of `control`. A
/// candidate that fails is dropped; a failure of the search as a whole
/// leaves the defaults, with the reason. `log`, if given, receives a line
/// for each candidate as it is measured.
SkinChoice chooseSkins(driver::Control &control, const driver::System &system,
                       bool search,
                       llvm::raw_ostream *log = nullptr);

/// The key of the cache of a choice for `control` and `system`: the text
/// that names everything the choice depends on, the build of MDIR included
/// (the salt of the compile cache, D397).
std::string getSkinCacheKey(const driver::Control &control,
                            const driver::System &system);

} // namespace compiler
} // namespace mdir

#endif
