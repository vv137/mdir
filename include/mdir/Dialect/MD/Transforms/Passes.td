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
           "parameter takes as independent of it">,
    Option<"separateStages", "separate-stages", "bool", /*default=*/"false",
           "Take the adjoints of the fields, the forces, and the virial of "
           "a stage kernel over pairs as three ops, the reference form, "
           "instead of one gather of all three (D288)">
  ];
  let dependentDialects = ["::mlir::arith::ArithDialect",
                           "::mlir::math::MathDialect",
                           "::mlir::vector::VectorDialect"];
}

def Distribute : Pass<"md-distribute", "::mlir::ModuleOp"> {
  let summary = "Rewrites global potentials to run on a grid of domains";
  let description = [{
    Rewrites the body of every `md.potential` (or of those that
    `potentials` names) into `md_dist` (D276,
    docs/distributed-ml2.md, Section 14): a decomposition of the particle
    set into a grid of domains, a layout from the positions, the owned
    rows of every field, one halo of ghosts with their images, a copy to
    the ghosts of every field read at neighbors, and the pair ops on the
    chosen traversal. The potential keeps its signature: it takes and
    returns the values of the particle set.

    | `md` | `md_dist` |
    |---|---|
    | the positions, a field argument | `distribute` (positions with `image`) |
    | `md.neighborhood` | `neighborhood` over owned and ghost rows |
    | `md.sum_relation` | `sum_pairs` and `all_reduce` |
    | `md.gather_relation` | `gather_pairs`; on `unique`, its two partial results and `halo_sum` |
    | `md.sum_particles` | `sum_particles` over owned rows and `all_reduce` |
    | `md.map_particles` | `map_particles` over owned rows |
    | a field read at neighbors | `halo_copy` of its owned rows |

    It runs before `md-differentiate`, so that the derivative carries
    the transfers' adjoints. It refuses, naming the reason, what it cannot
    distribute: exclusions, tuples, reciprocal sums, terms of the absolute
    positions, a cell that is not orthorhombic at run time (checked by
    the executor), a contract it cannot prove, and ops it does not know.
  }];
  let options = [
    ListOption<"grid", "grid", "int64_t",
               "Explicit numbers of domains along the three cell axes; "
               "alternative to domains with box">,
    Option<"domains", "domains", "int64_t", /*default=*/"0",
           "Total domains to factor into a Cartesian grid; requires box">,
    ListOption<"box", "box", "double",
               "Compile-time orthorhombic edge lengths for selecting a "
               "factor grid from domains (three positive finite numbers)">,
    Option<"traversal", "traversal", "std::string", /*default=*/"\"directed\"",
           "The traversal of the pairs: 'directed' (every pair at both of "
           "its owners) or 'unique' (every pair once, the ghosts' "
           "contributions returned)">,
    Option<"reach", "reach", "double", /*default=*/"0.0",
           "The reach of the halo; 0 takes the largest cutoff">,
    ListOption<"potentials", "potentials", "std::string",
               "The potentials to distribute; all if none is named">
  ];
  let dependentDialects = ["::mdir::md_dist::MDDistDialect",
                           "::mlir::arith::ArithDialect",
                           "::mlir::math::MathDialect",
                           "::mlir::vector::VectorDialect"];
}

def DistOwnerComputes
    : Pass<"md-dist-owner-computes", "::mlir::ModuleOp"> {
  let summary = "Computes the forces of a directed traversal at the owners";
  let description = [{
    The transpose of a directed `md_dist.sum_pairs` is a `gather_pairs`
    over the same relation with two partial results, completed by
    `halo_sum`: every pair adds half its kernel at each of its two
    particles, and the contributions of the ghosts are returned to their
    owners. Every pair of a directed relation is at the owner of each of
    its particles, so the same sum is the gather with one complete result,
    every pair at its center with the weight one, and nothing to return
    (D276, docs/distributed-ml2.md, Section 14):

    ```text
    p_i + (G^T q)_i = ½ Σ_k k(i, k) + ½ σ Σ_{(j, k) : s(k) = i} k(j, k)
                    = Σ_k k(i, k)
    ```

    The precondition, checked with the code of the verifier: the relation
    is directed, its halo covers its cutoff (V4), and the kernel has the
    exchange contract that gives the value at the second particle (V8).
    The pass is the switch: without it, the derivative's form runs.
  }];
}

def AnalyzeActivity : Pass<"md-analyze-activity", "::mlir::ModuleOp"> {
  let summary = "Report active, proven inactive, or unknown results for a potential argument";
  let options = [Option<"argument", "argument", "int", "2", "Potential argument to analyze">];
}

def CheckDerivativeCoverage : Pass<"md-check-derivative-coverage", "::mlir::ModuleOp"> {
  let summary = "Require a derivative rule or declared zero for every scalar op in potentials";
}

def CheckTransposes : Pass<"md-check-transposes", "::mlir::ModuleOp"> {
  let summary = "Require a declared transpose for every op of the structure "
                "in potentials";
  let description = [{
    Refuses a sum, gather, or map of `md`, or a transfer or loop of
    `md_dist`, in a potential, that does not declare its transpose through
    `AdjointOpInterface` (D295): the ops through which
    `md-differentiate` carries adjoints. The scalar ops of kernels are
    checked by `--md-check-derivative-coverage` (D183).
  }];
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
