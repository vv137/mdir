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
    | `md_exec.tuple_for` | `scf.parallel` over the particles, with an `scf.for` over the tuples that each belongs to (the incidence of the set) |
    | The global sums of a loop | `scf.reduce` of the `scf.parallel`, which `convert-scf-to-openmp` makes a reduction of OpenMP when the run has more than one thread |
    | `md_exec.empty_neighbors` | The buffers of a neighbor matrix |
    | `md_exec.refresh_neighbors` | The test of validity and a call to the neighbor build template, which the pass adds to the module |
    | `md_exec.reciprocal` | The templates of particle mesh Ewald for the host, in f64, and the transforms of pocketfft (`mdrtFFTForward3D`, `mdrtFFTBackward3D`) |
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

def FixedOrderReductions
    : Pass<"fixed-order-reductions", "::mlir::ModuleOp"> {
  let summary = "Sums the reductions of parallel loops in an order that "
                "the threads do not decide";
  let description = [{
    A reduction of `scf.parallel` that `convert-scf-to-openmp` lowers is
    combined by the OpenMP runtime in the order in which the threads
    finish, so a sum of floating-point numbers can differ in its last bits
    from one run to the next. In a run with a thermostat or a barostat such
    sums (the kinetic energy, the virial) set the coupling, and the
    difference grows into the trajectory (D171).

    The pass rewrites each one-dimensional `scf.parallel` with results
    into a parallel loop over a fixed number of chunks, `chunks` or the
    number of iterations if smaller. The chunks split the iterations into
    contiguous ranges that depend only on that number; each chunk folds
    its range in order and stores its partial result, and a serial loop
    then folds the initial value with the partial results in the order of
    the chunks. The result depends neither on the schedule of the threads
    nor on their number: a run with one thread gives the same bits as a
    run with many. The partial results live in an alloca of a constant
    size at the entry of the function, so a loop of steps does not grow
    the stack (D117). A reduction over more than one dimension, or inside
    another parallel loop, whose iterations would share that buffer, is
    refused.
  }];
  let options = [
    Option<"chunks", "chunks", "int64_t", /*default=*/"256",
           "The number of chunks of the iterations">,
  ];
  let dependentDialects = ["::mlir::arith::ArithDialect",
                           "::mlir::memref::MemRefDialect",
                           "::mlir::scf::SCFDialect"];
}

def HoistStaticAllocas : Pass<"hoist-static-allocas", "::mlir::ModuleOp"> {
  let summary = "Moves the allocas of a constant size outside the regions "
                "of OpenMP to the entry of their function";
  let description = [{
    `convert-scf-to-openmp` makes each variable of a reduction an
    `llvm.alloca` where the parallel loop is, inside a `memref.alloca_scope`
    that `canonicalize` inlines, since `convert-scf-to-cf` would otherwise
    leave the scope with more than one block. Inside the loop of the steps
    the stack then grows at every reduction until it ends: a run on the CPU
    grew it by 22 bytes a step and overflowed its 8 MB after some 300,000
    steps (D117). The pass moves every `llvm.alloca` whose size is a
    constant, and that is not inside an op of OpenMP (whose regions become
    functions of their own, with an alloca for each thread), to the start
    of the entry block of its function, where it is allocated once. The
    variable of a reduction is live within one iteration of the loops
    around it, so one place serves every iteration.
  }];
  let dependentDialects = ["::mlir::LLVM::LLVMDialect"];
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
    | `md_exec.particle_for` | One `gpu.launch` with a thread for each particle, which writes only to its own particle. |
    | `md_exec.pair_for` over a neighbor matrix | `row-lanes` threads share the row of a particle and add their values by shuffles; the first writes the particle. No atomics. |
    | `md_exec.pair_for` over groups | A warp for each block of 64 entries of the lists of groups of 16; each pair once, added to the particle of the entry with an atomic addition and to the group once for the block (D89). |
    | `md_exec.tuple_for`, not disjoint | A thread evaluates each tuple once and adds to its members with atomics (`tuples-once`, D103); in the deterministic mode, the rows of the particles take the tuples they belong to. |
    | `md_exec.tuple_for`, disjoint | A thread for each particle; the member at place 0 evaluates the tuple and writes every member, which no other thread writes (D83). |
    | Consecutive loops of the same positions | One kernel, where no loop reads what another writes (`fuse-rows`); a loop over particles, the loops over the tuples of a disjoint union that read it, and the loop over particles that reads them become one integration kernel, in which a warp takes whole groups of constraints (`fuse-integration`, D110, D111). |
    | The global sums of a loop | Each thread adds up the particles that it takes, and a block adds up its threads by a tree (`gpu.all_reduce`) into partial sums in `scratch`; one block adds up the partial sums. The order depends only on the number of particles. The host reads the results with one copy for each type of number. Sums over tuples evaluated once that reach only a later loop over tuples evaluated once, through arithmetic, are computed with one part by a block of the kernel of that loop, which then does the arithmetic and the later loop; several parts keep their kernels, launched only for several, which leave the totals in a buffer for that block (`fuse-centers`, #66). |
    | An `i1` reduction (any) | A flag on the device, set by a plain store; the host reads it through pinned memory and an event, without waiting for the stream, just before the first op that uses it (`mdrtFlagStart`, `mdrtFlagFinish`, D113). |
    | `md_exec.empty_neighbors` | The buffers of a neighbor matrix or of the lists of groups, which the runtime grows when a build needs more |
    | `md_exec.refresh_neighbors` | A branch on the flag of the test of validity to the build of its kind, a template that the pass adds to the module; with `prune_skin`, on the flag of the inner list to a pruning of the inner list from the outer one (D114) |
    | `md_exec.reciprocal` | The kernels of the template of particle mesh Ewald on a device and the transforms of cuFFT; with `md_exec.side`, on the second stream |
    | A cell | `vector<3xf64>`, the edge lengths of an orthorhombic cell |

    A constant that a kernel uses is a constant of the kernel, not an
    argument.

    A loop of the host that launches kernels releases, at the end of every
    iteration, the stack that the iteration has taken: the lowering of a
    launch puts the arguments on the stack.

    A division in f32 that may be approximated (`afn`, from
    `md-exec-approximate`) becomes a product with `nvvm.rcp.approx.ftz.f`,
    which flushes subnormal numbers: the division of the backend,
    `div.approx.f32`, scales its operands where they are subnormal or very
    large, five instructions more (D98).

    The result is lowered further by `mdir-gpu-lower-to-nvvm`, the
    upstream pipeline `gpu-lower-to-nvvm-pipeline` with its serialization
    of the GPU modules replaced by `mdir-gpu-module-to-binary`.
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
    Option<"fuseIntegration", "fuse-integration", "bool", /*default=*/"true",
           "Whether a loop over particles, the loops over the tuples of a "
           "disjoint union that read it, and the loop over particles that "
           "reads them become one kernel (D110)">,
    Option<"contract", "contract", "bool", /*default=*/"true",
           "Whether the arithmetic of the kernels may contract a product "
           "and a sum into a fused multiply-add, which rounds once "
           "(the control keyword fast_math)">,
    Option<"explicitFma", "explicit-fma", "bool", /*default=*/"false",
           "Whether every sum or difference in a kernel with a product as "
           "an operand becomes a math.fma, whatever else uses the product, "
           "so that whether a product is fused does not depend on its "
           "other uses (the deterministic mode, #102)">,
    Option<"tuplesOnce", "tuples-once", "bool", /*default=*/"true",
           "Whether a loop over tuples that may share particles evaluates "
           "each tuple once and adds to its members with atomics, outside "
           "the deterministic mode (JAC: 526 against 505 ns/day for the "
           "rows of the particles)">,
    Option<"fuseCenters", "fuse-centers", "bool", /*default=*/"true",
           "Whether the sums of a loop over few tuples evaluated once, which "
           "reach only a later such loop through arithmetic, are computed by "
           "one block of the kernel of that loop, as for the centers of a "
           "term over the centers of groups (#66)">,
    Option<"deterministic", "deterministic", "bool", /*default=*/"false",
           "Whether sums are added in an order that does not depend on the "
           "threads (D84): the charges of PME in fixed point, not with "
           "floating-point atomics">
  ];
  let dependentDialects = ["::mlir::arith::ArithDialect",
                           "::mlir::func::FuncDialect",
                           "::mlir::gpu::GPUDialect",
                           "::mlir::LLVM::LLVMDialect",
                           "::mlir::NVVM::NVVMDialect",
                           "::mlir::math::MathDialect",
                           "::mlir::memref::MemRefDialect",
                           "::mlir::scf::SCFDialect",
                           "::mlir::vector::VectorDialect"];
}

def GpuModuleToBinary
    : Pass<"mdir-gpu-module-to-binary", "::mlir::ModuleOp"> {
  let summary = "Serialize the GPU modules to binaries in parallel";
  let description = [{
    Replaces each `gpu.module` with a `gpu.binary` of its PTX (`format=isa`)
    or of its cubin (`format=bin`), as upstream `gpu-module-to-binary`
    does, but serializes the modules on the threads of the context, at
    most `MDIR_COMPILE_THREADS` (D211), and puts the binaries in the order
    of the modules, so that the result does not depend on the threads
    (D[gpu-module-compile]). The time the serialization took is left out of
    the objects' properties, which the upstream pass records, so that the
    result is the same from run to run.
  }];
  let options = [
    Option<"format", "format", "std::string", /*default=*/"\"isa\"",
           "isa (PTX, which the driver compiles at load) or bin (a cubin)">
  ];
}

#endif // MDIR_CONVERSION_PASSES_TD
