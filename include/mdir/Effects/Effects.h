// Shared declared effects and resource identities.
#ifndef MDIR_EFFECTS_EFFECTS_H
#define MDIR_EFFECTS_EFFECTS_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include <string>

namespace mdir {
namespace effects {
/// Parses the D355 resource summary shared by programs and declared calls.
/// This query does not assert that an arbitrary function body has the
/// declared effects; callers that inspect a definition must check its body.
mlir::LogicalResult parseEffectSummary(
    mlir::DictionaryAttr summary,
    llvm::SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects,
    std::string &error);

} // namespace effects
namespace dyn {
/// The energy of the bath, which the conserved energy of the log and the
/// checkpoint's `bath` take (D129; docs/dynamics.md, Section 2.2.3).
struct BathResource : mlir::SideEffects::Resource::Base<BathResource> {
  static BathResource *get();
  llvm::StringRef getName() const override { return "dyn.bath"; }
};

/// The state of a Nose-Hoover chain on the host, which the checkpoint
/// keeps (D163a).
struct ChainResource : mlir::SideEffects::Resource::Base<ChainResource> {
  static ChainResource *get();
  llvm::StringRef getName() const override { return "dyn.chain"; }
};

/// The stops of the run that the host records, such as that of a chain
/// whose factorization no longer follows it (D206).
struct StopResource : mlir::SideEffects::Resource::Base<StopResource> {
  static StopResource *get();
  llvm::StringRef getName() const override { return "dyn.stop"; }
};

/// What the log, the trajectory, and the checkpoint keep of the cell and of
/// the state of the pressure.
struct RecordResource : mlir::SideEffects::Resource::Base<RecordResource> {
  static RecordResource *get();
  llvm::StringRef getName() const override { return "dyn.record"; }
};

/// The statistics of the pressures of the axes that the log reports at the
/// end of a run of semi-isotropic or anisotropic coupling.
struct StatisticsResource
    : mlir::SideEffects::Resource::Base<StatisticsResource> {
  static StatisticsResource *get();
  llvm::StringRef getName() const override { return "dyn.statistics"; }
};

} // namespace dyn
} // namespace mdir

#endif // MDIR_EFFECTS_EFFECTS_H
