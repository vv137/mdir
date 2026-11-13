// Ops of the md_alg dialect (D406).

#ifndef MDIR_DIALECT_MDALG_MDALGOPS_H
#define MDIR_DIALECT_MDALG_MDALGOPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "mdir/Dialect/MDAlg/MDAlgDialect.h"
#include "mdir/Dialect/MDAlg/MDAlgEnums.h.inc"

namespace mdir::md_alg {
/// Refuses a determinism requirement above the op's bound (D[md-alg-ir]).
mlir::LogicalResult verifyDeterminism(mlir::Operation *op);
/// The name of a tier, as the IR writes it.
llvm::StringRef getDeterminismName(Determinism determinism);
} // namespace mdir::md_alg

#include "mdir/Dialect/MDAlg/MDAlgInterfaces.h.inc"

#define GET_OP_CLASSES
#include "mdir/Dialect/MDAlg/MDAlgOps.h.inc"

#endif // MDIR_DIALECT_MDALG_MDALGOPS_H
