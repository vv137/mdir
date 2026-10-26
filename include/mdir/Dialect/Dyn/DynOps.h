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

} // namespace dyn
} // namespace mdir

#define GET_OP_CLASSES
#include "mdir/Dialect/Dyn/DynOps.h.inc"

#endif // MDIR_DIALECT_DYN_DYNOPS_H
