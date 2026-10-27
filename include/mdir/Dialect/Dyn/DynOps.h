// Ops of the dyn dialect.

#ifndef MDIR_DIALECT_DYN_DYNOPS_H
#define MDIR_DIALECT_DYN_DYNOPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "mdir/Dialect/MD/MDTypes.h"

namespace mdir {
namespace dyn {

/// Parses the D355 resource summary shared by programs and declared calls.
/// This query does not assert that an arbitrary function body has the
/// declared effects; callers that inspect a definition must check its body.
mlir::LogicalResult parseEffectSummary(
    mlir::DictionaryAttr summary,
    llvm::SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects,
    std::string &error);

/// The energy of the bath, which the conserved energy of the log and the
/// checkpoint's `bath` take (D129; docs/dynamics.md, Section 2.2.3).
struct BathResource : mlir::SideEffects::Resource::Base<BathResource> {
  llvm::StringRef getName() const override { return "dyn.bath"; }
};

/// The state of a Nose-Hoover chain on the host, which the checkpoint
/// keeps (D163a).
struct ChainResource : mlir::SideEffects::Resource::Base<ChainResource> {
  llvm::StringRef getName() const override { return "dyn.chain"; }
};

/// The stops of the run that the host records, such as that of a chain
/// whose factorization no longer follows it (D206).
struct StopResource : mlir::SideEffects::Resource::Base<StopResource> {
  llvm::StringRef getName() const override { return "dyn.stop"; }
};

/// What the log, the trajectory, and the checkpoint keep of the cell and of
/// the state of the pressure.
struct RecordResource : mlir::SideEffects::Resource::Base<RecordResource> {
  llvm::StringRef getName() const override { return "dyn.record"; }
};

/// The statistics of the pressures of the axes that the log reports at the
/// end of a run of semi-isotropic or anisotropic coupling.
struct StatisticsResource
    : mlir::SideEffects::Resource::Base<StatisticsResource> {
  llvm::StringRef getName() const override { return "dyn.statistics"; }
};

} // namespace dyn
} // namespace mdir

#define GET_OP_CLASSES
#include "mdir/Dialect/Dyn/DynOps.h.inc"

#endif // MDIR_DIALECT_DYN_DYNOPS_H
