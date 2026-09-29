// Passes of the md dialect.

#ifndef MDIR_DIALECT_MD_TRANSFORMS_PASSES_TD
#define MDIR_DIALECT_MD_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def ExpandTruncation : Pass<"md-expand-truncation", "::mlir::ModuleOp"> {
  let summary = "Expands truncation attributes into kernels";
  let description = [{
    Rewrites the kernel of every `md.sum_relation` that carries a truncation,
    so that the kernel itself computes the truncated energy. Afterward every
    `md.sum_relation` has no truncation.

    | Truncation | Kernel after the pass |
    |---|---|
    | `shift` | `u(r) − u(r_c)` |
    | `force_shift` | `u(r) − u(r_c) − (r − r_c) · u'(r_c)` |
    | `switch`, from `r_s` | `u(r) · S(r)` |
  }];
  let dependentDialects = ["::mlir::arith::ArithDialect",
                           "::mlir::math::MathDialect",
                           "::mlir::vector::VectorDialect"];
}

def Differentiate : Pass<"md-differentiate", "::mlir::ModuleOp"> {
  let summary = "Replaces md.evaluate with calls to derivative functions";
  let description = [{
    For every `md.evaluate`, generates an `md.function` that computes the
    requested quantities from the potential, and replaces the op with an
    `md.call`. Truncation is expanded in the generated function.

    Forces become an `md.gather_relation` whose kernel is
    `K(i, j) = −u'(r) · d / r`.
  }];
  let dependentDialects = ["::mlir::arith::ArithDialect",
                           "::mlir::math::MathDialect",
                           "::mlir::vector::VectorDialect"];
}

def CheckExchange : Pass<"md-check-exchange", "::mlir::ModuleOp"> {
  let summary = "Proves the exchange contracts that rest on a proof";
  let description = [{
    For every pair kernel whose exchange contract has the basis `proof`,
    attempts to prove the contract by swapping the two particles and
    comparing the kernel with itself structurally. Commutative ops are
    compared in either operand order. The pass fails if a proof fails.

    Contracts with the basis `asserted` or `derived` are not checked.
  }];
}

#endif // MDIR_DIALECT_MD_TRANSFORMS_PASSES_TD
