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
           "particle">,
    Option<"cells", "cells", "int64_t", /*default=*/"3",
           "Number of cells that the cutoff plus the skin may span: the "
           "cells of a neighbor build are that fraction of it wide or "
           "wider">
  ];
  let dependentDialects = ["::mdir::md_exec::MDExecDialect",
                           "::mdir::mdrt::MDRTDialect",
                           "::mlir::arith::ArithDialect",
                           "::mlir::math::MathDialect",
                           "::mlir::vector::VectorDialect"];
}

def ConvertMDExecToLoops
    : Pass<"convert-md-exec-to-loops", "::mlir::ModuleOp"> {
  let summary = "Converts md_exec loops in the storage form to loops";
  let description = [{
    Replaces every op of `md_exec` in the storage form, where it is, with
    `scf` loops that load from and store to `memref`s.

    | Before | After |
    |---|---|
    | `md_exec.particle_for` | `scf.parallel` over the particles |
    | `md_exec.pair_for` | `scf.parallel` over the particles, with an `scf.for` over the neighbors of each |
    | `md_exec.empty_neighbors` | The buffers of a neighbor matrix |
    | `md_exec.refresh_neighbors` | The test of validity and a call to the neighbor build template, which the pass adds to the module |
    | `md_exec.rebuild_count` | A load |
    | A cell | `vector<3xf64>`, the edge lengths of an orthorhombic cell |

    The pass decides nothing about buffers; `md-exec-assign-storage` has
    done that. It rejects ops in the value form.
  }];
  let dependentDialects = ["::mlir::arith::ArithDialect",
                           "::mlir::func::FuncDialect",
                           "::mlir::math::MathDialect",
                           "::mlir::memref::MemRefDialect",
                           "::mlir::scf::SCFDialect",
                           "::mlir::vector::VectorDialect"];
}

def ConvertMDExecToGPU
    : Pass<"convert-md-exec-to-gpu", "::mlir::ModuleOp"> {
  let summary = "Converts md_exec loops in the storage form to GPU kernels";
  let description = [{
    Replaces every op of `md_exec` in the storage form, where it is, with
    ops of the upstream `gpu` dialect. The buffers are on the device, as
    `md-exec-assign-storage` with `memory=device` leaves them.

    | Before | After |
    |---|---|
    | `md_exec.particle_for`, `md_exec.pair_for` | One `gpu.launch` with one thread per particle. A thread writes only to its own particle. |
    | The global sums of a loop | The kernel writes the contributions of each particle to buffers in `scratch`. A second kernel adds up chunks of particles, a third adds up the chunks, and the host reads the results with one copy. The order is fixed. |
    | `md_exec.empty_neighbors` | The buffers of a neighbor matrix, on the device |
    | `md_exec.refresh_neighbors` | The test of validity, as a global maximum, and a call to the neighbor build template for devices, which the pass adds to the module |
    | A cell | `vector<3xf64>`, the edge lengths of an orthorhombic cell |

    A constant that a kernel uses is a constant of the kernel, not an
    argument.

    A loop of the host that launches kernels releases, at the end of every
    iteration, the stack that the iteration has taken: the lowering of a
    launch puts the arguments on the stack.

    The result is lowered further by the upstream pipeline
    `gpu-lower-to-nvvm-pipeline`.
  }];
  let options = [
    Option<"blockSize", "block-size", "int64_t", /*default=*/"128",
           "The number of threads in a block">,
    Option<"rowLanes", "row-lanes", "int64_t", /*default=*/"16",
           "The threads that share the neighbors or tuples of a particle, "
           "a power of two up to 32; 1 gives a thread to each particle">,
    Option<"fuseRows", "fuse-rows", "bool", /*default=*/"true",
           "Whether consecutive loops over pairs and tuples of the same "
           "positions become one kernel">,
    Option<"contract", "contract", "bool", /*default=*/"true",
           "Whether the arithmetic of the kernels may contract a product "
           "and a sum into a fused multiply-add, which rounds once "
           "(the control keyword fast_math)">,
    Option<"tuplesOnce", "tuples-once", "bool", /*default=*/"true",
           "Whether a loop over tuples that may share particles evaluates "
           "each tuple once and adds to its members with atomics, outside "
           "the deterministic mode (JAC: 526 against 505 ns/day for the "
           "rows of the particles)">,
    Option<"deterministic", "deterministic", "bool", /*default=*/"false",
           "Whether sums are added in an order that does not depend on the "
           "threads (D84): the charges of PME in fixed point, not with "
           "floating-point atomics">
  ];
  let dependentDialects = ["::mlir::arith::ArithDialect",
                           "::mlir::func::FuncDialect",
                           "::mlir::gpu::GPUDialect",
                           "::mlir::LLVM::LLVMDialect",
                           "::mlir::math::MathDialect",
                           "::mlir::memref::MemRefDialect",
                           "::mlir::scf::SCFDialect",
                           "::mlir::vector::VectorDialect"];
}

#endif // MDIR_CONVERSION_PASSES_TD
