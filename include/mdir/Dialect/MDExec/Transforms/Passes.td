// Passes of the md_exec dialect.

#ifndef MDIR_DIALECT_MDEXEC_TRANSFORMS_PASSES_TD
#define MDIR_DIALECT_MDEXEC_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def ReuseNeighbors : Pass<"md-exec-reuse-neighbors", "::mlir::ModuleOp"> {
  let summary = "Reuses neighbor structures across the iterations of a loop";
  let description = [{
    A neighbor structure that is built in the body of a loop is built in
    every iteration. The pass makes the structure a value that the loop
    carries, and replaces the build with a refresh, which builds only when
    the structure is no longer valid.

    Before:

    ```mlir
    scf.for ... {
      %cells = md_exec.build_cells %x, %cell ...
      %nl    = md_exec.build_neighbors %cells, %x, %cell ...
      ...
    }
    ```

    After:

    ```mlir
    %nl0 = md_exec.empty_neighbors ...
    scf.for ... iter_args(%nl = %nl0) {
      %nl1 = md_exec.refresh_neighbors %nl, %x, %cell ... policy(check)
      ...
      scf.yield %nl1
    }
    ```

    The structure is empty when the loop begins, so the first iteration
    builds it.
  }];
  let dependentDialects = ["::mdir::md_exec::MDExecDialect",
                           "::mdir::mdrt::MDRTDialect"];
}

#endif // MDIR_DIALECT_MDEXEC_TRANSFORMS_PASSES_TD
