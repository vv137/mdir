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

def FuseLoops : Pass<"md-exec-fuse-loops", "::mlir::ModuleOp"> {
  let summary = "Fuses loops over the pairs of one neighbor structure";
  let description = [{
    Two `md_exec.pair_for` ops are fused into one when they run over the
    same neighbor structure, positions, and cell, with the same cutoff and
    policy, and the second does not use a result of the first.

    The fused loop has the fields, destinations, and global sums of both,
    and its kernel is the first kernel followed by the second. Common
    subexpression elimination, run afterward, removes what the two kernels
    compute twice.

    The results are the same as before, bit for bit: every sum receives the
    same contributions in the same order.
  }];
  let dependentDialects = ["::mdir::md_exec::MDExecDialect"];
}

def SimplifyDistance
    : Pass<"md-exec-simplify-distance", "::mlir::ModuleOp"> {
  let summary = "Rewrites pair kernels in powers of the squared distance";
  let description = [{
    A pair kernel receives the squared distance `r2`. A kernel that was
    written in terms of the distance starts by taking the square root. The
    pass writes every value of the kernel as a sum of terms

    ```text
    c · r^p
    ```

    with `c` free of explicit powers of `r`, carries that form through
    sums, products, quotients, and integer powers, and emits code only
    where a value is needed. An even power of `r` is computed from `r2`
    and from `1 / r2`; only an odd power needs the square root.

    A kernel whose results contain only even powers of the distance, such
    as the Lennard-Jones energy and force, ends up with no square root.

    The pass reassociates floating-point arithmetic, so results change in
    their last bits. It must not be run in the deterministic mode.
  }];
  let dependentDialects = ["::mlir::arith::ArithDialect",
                           "::mlir::math::MathDialect"];
}

#endif // MDIR_DIALECT_MDEXEC_TRANSFORMS_PASSES_TD
