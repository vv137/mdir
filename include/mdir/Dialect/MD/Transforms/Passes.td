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
    | `force_switch`, from `r_s` | `u(r) − P(r) − C` |
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
  let options = [
    Option<"remarks", "remarks", "bool", /*default=*/"false",
           "Remark on every op that a derivative with respect to a "
           "parameter takes as independent of it">
  ];
  let dependentDialects = ["::mlir::arith::ArithDialect",
                           "::mlir::math::MathDialect",
                           "::mlir::vector::VectorDialect"];
}

def AnalyzeActivity : Pass<"md-analyze-activity", "::mlir::ModuleOp"> {
  let summary = "Report active, proven inactive, or unknown results for a potential argument";
  let options = [Option<"argument", "argument", "int", "2", "Potential argument to analyze">];
}

def CheckDerivativeCoverage : Pass<"md-check-derivative-coverage", "::mlir::ModuleOp"> {
  let summary = "Require a derivative rule or declared zero for every scalar op in potentials";
}

def CheckDerivatives : Pass<"md-check-derivatives", "::mlir::ModuleOp"> {
  let summary = "Check requested derivatives against f64 central differences at evaluation inputs";
  let options = [
    Option<"step", "step", "double", "1e-5", "Relative finite difference step">,
    Option<"atol", "atol", "double", "1e-6", "Absolute tolerance">,
    Option<"rtol", "rtol", "double", "1e-4", "Relative tolerance">
  ];
  let dependentDialects = ["::mlir::arith::ArithDialect", "::mlir::math::MathDialect",
    "::mlir::vector::VectorDialect", "::mlir::memref::MemRefDialect",
    "::mlir::scf::SCFDialect", "::mlir::func::FuncDialect",
    "::mdir::mdrt::MDRTDialect"];
}

def Inline : Pass<"md-inline", "::mlir::ModuleOp"> {
  let summary = "Inlines potentials, functions, and programs";
  let description = [{
    Replaces every `md.call` and `dyn.step` with the body of what it calls,
    then removes the potentials, functions, and programs that are no longer
    referenced. What remains is code in ordinary functions.
  }];
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

def BypassUpdates : Pass<"md-bypass-updates", "::mlir::ModuleOp"> {
  let summary = "Lets the loops over the sets of a disjoint union read the "
                "fields before the updates of the other sets";
  let description = [{
    A loop over the tuples of a set (`md.gather_tuples`) that is the next
    link of a chain of updates, whose result a map adds to a field it reads,
    reads its positions and gathered fields at the members of its tuples
    only. When the set is in an `md.disjoint_union`, what a loop over the
    tuples of an other set of the union gathers is zero at those members,
    and a field that is another field plus that gathered value agrees with
    the other field there. The loop reads the other field instead:

    ```mlir
    %c1 = md.gather_tuples %settles, %x, %cell ...
    %y1 = md.map_particles gather(%x, %c1 : !vec, !vec) {
    ^bb0(%a: vector<3xf64>, %b: vector<3xf64>):
      %s = arith.addf %a, %b : vector<3xf64>
      md.yield %s : vector<3xf64>
    } : !vec
    %c2 = md.gather_tuples %shake1, %y1, %cell ...   // reads %x instead
    ```

    The loops of the sets then depend on no loop of the union, and a
    later pass may run them at once; the updates become one loop over the
    particles (`md-exec-fuse-loops`). The values read are the same, up to
    the sign of a zero (x + 0 is +0 where x is −0).
  }];
  let dependentDialects = ["::mlir::arith::ArithDialect"];
}

#endif // MDIR_DIALECT_MD_TRANSFORMS_PASSES_TD
