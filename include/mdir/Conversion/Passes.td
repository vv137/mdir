// Passes that convert one dialect to another.

#ifndef MDIR_CONVERSION_PASSES_TD
#define MDIR_CONVERSION_PASSES_TD

include "mlir/Pass/PassBase.td"

def ConvertMDToMDExec : Pass<"convert-md-to-md-exec", "::mlir::ModuleOp"> {
  let summary = "Converts md and dyn ops to md_exec loops";
  let description = [{
    | Op | Becomes |
    |---|---|
    | `md.neighborhood` | `md_exec.build_cells` and `md_exec.build_neighbors` |
    | `md.sum_relation` | `md_exec.pair_for` with a `reduce` clause |
    | `md.gather_relation` | `md_exec.pair_for` with an `outs` clause |
    | `md.sum_particles` | `md_exec.particle_for` with a `reduce` clause |
    | `md.map_particles` | `md_exec.particle_for` with an `outs` clause |
    | `dyn.kick`, `dyn.drift` | `md_exec.particle_for` with an `outs` clause |

    A sum over an unordered relation becomes a loop over directed pairs with
    the weight one half.

    The neighbor structure is built where the neighborhood was, so it is
    rebuilt at every evaluation. Reusing it across steps is the task of a
    later pass.

    The pass expects that truncation has been expanded and that
    `md.evaluate` has been replaced.
  }];
  let options = [
    Option<"skin", "skin", "double", /*default=*/"0.0",
           "Distance that is added to the cutoff of neighbor structures">,
    Option<"width", "width", "int64_t", /*default=*/"64",
           "Number of neighbors that a neighbor structure holds per "
           "particle">
  ];
  let dependentDialects = ["::mdir::md_exec::MDExecDialect",
                           "::mdir::mdrt::MDRTDialect",
                           "::mlir::arith::ArithDialect",
                           "::mlir::math::MathDialect",
                           "::mlir::vector::VectorDialect"];
}

#endif // MDIR_CONVERSION_PASSES_TD
