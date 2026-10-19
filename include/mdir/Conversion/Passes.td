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
    | `md_exec.reciprocal` | The templates of particle mesh Ewald for the host, in parallel loops, the grid in the type of the forces, and the transforms of pocketfft for blocks of lines (`mdrtFFT64RowsForward` and the others of `runtime/mdrt_fft.c`); `pme_scale` where the energy and the virial have no use (D271) |
    | `md_exec.rebuild_count` | A load |
    | A cell | `vector<3xf64>`, the edge lengths of an orthorhombic cell |

    The pass decides nothing about buffers; `md-exec-assign-storage` has
    done that. It rejects ops in the value form.

    With `kept-tuples`, the members that `md_exec.renumber` puts at their
    places, and the incidence structure that `md_exec.build_incidence`
    builds from them or from the members that the entry was given, are
    taken from the runtime (`mdrtTuplesRenumber`, `mdrtTuplesIncidence`)
    and given back to it where they would be freed (`mdrtTuplesRelease`):
    a program that an embedding front end runs in activations keeps them
    from one activation to the next (D253), and `mdir run` from one
    segment to the next (D255). The pipelines of both set
    it; without it they are built by loops in the module.
  }];
  let options = [
    Option<"keptTuples", "kept-tuples", "bool", /*default=*/"false",
           "Whether the renumbered members of the tuples and their incidence "
           "structures are taken from the runtime, which keeps them across "
           "the activations of an embedding program and the segments of "
           "mdir run">,
    Option<"atomicGrid", "atomic-grid", "bool", /*default=*/"true",
           "Whether the spreading of particle mesh Ewald adds to the "
           "fixed-point grid with atomics, for a program whose parallel "
           "loops run on several threads; on one thread a plain addition "
           "gives the same integers (D271)">,
  ];
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

    The pass runs a second time where the lowering for the CPU ends, for
    every program, on what `finalize-memref-to-llvm` has made of the
    `memref.alloca` of the module by then (#305): the rows that the builder
    allocates where it writes them (the observed columns, the coordinates
    and the terms of pulling, dH/dλ and the energies of the states) stand in
    the loops over steps, 16 bytes of the stack for every two numbers at
    every row, kept until the entry returns, which the entry of a
    simulation never does. A row is written anew before the host reads it,
    in one iteration, so one place serves every iteration here too. The
    lowering for a device does not need the pass: it releases the stack of
    an iteration of a loop of the host where the iteration ends.

    With `refuse-rest` the pass fails where it would leave an `llvm.alloca`
    outside the first block of its function and outside every op of OpenMP:
    one whose size is not a constant, which it cannot take once. The
    lowering for the CPU runs it so where it ends, for every program
    (#309, #312): the entry of a program of frames is a loop over frames
    that never returns, that of a program of segments never returns either,
    and the loops over steps of `mdir run` keep what they take until the run
    ends. A module that the pass accepts has no allocation of the stack that
    a step or a frame could take again, and one that it cannot prove so is
    an error that names the allocation, not a stack that grows.
  }];
  let options = [
    Option<"refuseRest", "refuse-rest", "bool", /*default=*/"false",
           "Fail on an alloca outside the first block of its function that "
           "the pass does not move">,
  ];
  let dependentDialects = ["::mlir::LLVM::LLVMDialect"];
}

def CheckReadOnlyArguments
    : Pass<"check-read-only-arguments", "::mlir::ModuleOp"> {
  let summary = "Proves that a function writes none of the buffers that it "
                "declares read-only";
  let description = [{
    The entry of a program declares the buffers of the model that it takes
    (the masses, the fields of the particles, the tables, and the members
    and fields of the tuple sets) read-only with the attribute
    `mdrt.read_only` on its arguments, so that the caller may hand the same
    buffers to every call without a copy (D258).
    The pass runs where the lowering to loops or to a device ends and
    follows every use of each such argument, through the views of it
    (casts, subviews, and other ops of `ViewLikeOpInterface`) and into the
    functions of the module that it is passed to. It accepts:

    - an op whose memory effects on the buffer are reads alone
      (`memref.load`, the source of `memref.copy` or of `gpu.memcpy`, and
      the like) or that has none and returns no buffer (`memref.dim`);
    - a call of a function of the module, whose argument it then follows;
    - a call of a declared function (of the runtime) whose parameter has
      `mdrt.read_only` too: what the runtime does with it is the runtime's
      declaration, which the lowering that declares the function makes.

    Any other use fails the pass, with an error that names the argument and
    the op: a write, a release, a buffer carried by a loop or returned, a
    pointer taken out of it, a call of a function that does not declare the
    parameter read-only, and an op whose effects are not known. So a module
    that the pass accepts is one whose code provably writes none of these
    buffers, and one that it cannot prove so does not compile; there is no
    third outcome.
  }];
  let dependentDialects = ["::mlir::func::FuncDialect",
                           "::mlir::memref::MemRefDialect"];
}

def SeparateSinCos : Pass<"separate-sin-cos", "::mlir::ModuleOp"> {
  let summary = "Makes every sine and cosine a call of its own to the C "
                "library";
  let description = [{
    The code generator of LLVM computes a sine and a cosine of one argument
    in one basic block by one call of `sincos` of the C library, and a sine
    without such a cosine by `sin`. The two functions of glibc do not round
    alike: in 2.35, 7 sines in 10,000 of arguments between -8 and 8 differ
    in their last bit. The torsion of a dihedral has a cosine in its energy
    and a sine in its force, so the forces of a step that writes energies
    were not those of a step that does not (#243).

    The pass replaces every `math.sin` and `math.cos` of f32 or f64, or of
    a vector of them of a fixed length, element by element, with a call of
    `sin`, `cos`, `sinf`, or `cosf`, which it declares. The code generator
    leaves the call of a function that is declared without attributes as
    it is, so the rounding of a sine does not depend on what else its
    kernel computes; a declaration that said that the function reads no
    memory (`memory(none)`) would make the calls the nodes that it merges.
    The CPU pipeline runs the pass in the deterministic mode, before
    `convert-math-to-llvm`; the default mode keeps `sincos`.
  }];
  let dependentDialects = ["::mlir::func::FuncDialect",
                           "::mlir::vector::VectorDialect"];
}

def LowerStageBodies : Pass<"md-exec-lower-stage-bodies", "::mlir::ModuleOp"> {
  let summary = "Lowers the batched bodies of neural stages to buffers";
  let description = [{
    The batched bodies of `md.stage` (D293,
    docs/ml4-ir.md, Section 8.3), functions of `linalg` on tensors with a
    static number of rows, become functions of buffers that the tiles of a
    loop call: elementwise ops are fused (with `fuse`, D[ml4-fused-bodies]:
    unit dimensions folded, pads as gathers, reductions started from the
    constants of their fills, producers of several uses evaluated again in
    each consumer that reads them once per element, nothing fused into a
    contraction, and the results written in the caller's buffers), the
    bodies are bufferized with
    their results as buffers that the caller passes after the operands, and
    the `linalg` ops become parallel loops. With `device`, each parallel
    loop becomes one kernel of a thread for each of its iterations, the
    constants of the body go into the kernels that read them, and the
    body's intermediate buffers are allocated on the device for the call.
    The bodies are marked `md.stage_body`, which the lowerings of the loops
    do not lower further. Run it before `convert-md-exec-to-gpu`.
  }];
  let options = [
    Option<"device", "device", "bool", /*default=*/"true",
           "Run the bodies' loops as kernels of the device">,
    Option<"blockSize", "block-size", "int64_t", /*default=*/"256",
           "The threads of a block of a body's kernel">,
    Option<"fuse", "fuse", "bool", /*default=*/"true",
           "Fold unit dimensions and fuse elementwise producers of several "
           "uses into their consumers; false fuses producers of one use only "
           "(D293)">,
  ];
  let dependentDialects = [
    "::mlir::arith::ArithDialect", "::mlir::bufferization::BufferizationDialect",
    "::mlir::gpu::GPUDialect", "::mlir::linalg::LinalgDialect",
    "::mlir::memref::MemRefDialect", "::mlir::scf::SCFDialect",
    "::mlir::tensor::TensorDialect"
  ];
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
    | `md_exec.tuple_for`, not disjoint | A thread evaluates each tuple once and adds to its members with atomics (`tuples-once`, D103), or, with `scatter(hash(S, P, L))`, through a table of the particles of its block in block memory (D283, D285); in the deterministic mode, the rows of the particles take the tuples they belong to. |
    | `md_exec.tuple_for`, disjoint | A thread for each particle; the member at place 0 evaluates the tuple and writes every member, which no other thread writes (D83). |
    | Consecutive loops of the same positions | One kernel, where no loop reads what another writes (`fuse-rows`); a loop over particles, the loops over the tuples of a disjoint union that read it, and the loop over particles that reads them become one integration kernel, in which a warp takes whole groups of constraints (`fuse-integration`, D110, D111). |
    | The global sums of a loop | Each thread adds up the particles that it takes, and a block adds up its threads by a tree (`gpu.all_reduce`) into partial sums in `scratch`; one block adds up the partial sums. The order depends only on the number of particles. The host reads the results with one copy for each type of number. Sums over tuples evaluated once that reach only a later loop over tuples evaluated once, through arithmetic, are computed with one part by a block of the kernel of that loop, which then does the arithmetic and the later loop; several parts keep their kernels, launched only for several, which leave the totals in a buffer for that block (`fuse-centers`, #66). |
    | An `i1` reduction (any) | A flag on the device, set by a plain store; the host reads it through pinned memory and an event, without waiting for the stream, just before the first op that uses it (`mdrtFlagStart`, `mdrtFlagFinish`, D113). |
    | `md_exec.empty_neighbors` | The buffers of a neighbor matrix or of the lists of groups, which the runtime grows when a build needs more |
    | `md_exec.refresh_neighbors` | A branch on the flag of the test of validity to the build of its kind, a template that the pass adds to the module; with `prune_skin`, on the flag of the inner list to a pruning of the inner list from the outer one (D114) |
    | `md_exec.reciprocal` | The kernels of the template of particle mesh Ewald on a device and the transforms of cuFFT, with `scatter(tile(E))` those of the tiles (D282); with `md_exec.side`, on the second stream |
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
           "floating-point atomics">,
    Option<"keptTuples", "kept-tuples", "bool", /*default=*/"false",
           "Whether the renumbered members of the tuples, their incidence "
           "structures, and the copies of the members on the device are "
           "taken from the runtime (mdrtDeviceTuplesRenumber, "
           "mdrtDeviceTuplesIncidence, mdrtDeviceTuplesMembers), which keeps "
           "them across the activations of an embedding program "
           "(D253) and the segments of mdir run (D255)">,
    Option<"deviceTuples", "device-tuples", "bool", /*default=*/"false",
           "With kept-tuples: whether the renumbered members and the "
           "incidence structures are built on the device by kernels of the "
           "module, in memory of the runtime's store, where it does not "
           "take them as kept (D257)">,
    Option<"checkPruning", "check-pruning", "bool", /*default=*/"false",
           "Whether each refresh of a dual list of groups counts the pairs "
           "within the cutoff in its outer list and in its inner list, "
           "through the active units, at the positions of the step, and "
           "the runtime stops the run if the counts differ "
           "(MDIR_CHECK_PRUNING, D268)">,
    Option<"fusePruning", "fuse-pruning", "bool", /*default=*/"false",
           "Whether the first loop over pairs after a refresh of a dual "
           "list of groups, at the same positions and in the same cell, "
           "prunes the inner list in its own kernel, a warp a group, on the "
           "steps where the refresh prunes (MDIR_FUSED_PRUNE, "
           "D284)">,
    Option<"fusePruningGroups", "fuse-pruning-groups", "int64_t",
           /*default=*/"0",
           "With fuse-pruning: the least number of groups of the last build "
           "for which the loop prunes; below it the refresh prunes "
           "(D284)">
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

def ExpandDevicePrimitives
    : Pass<"expand-device-primitives", "::mlir::ModuleOp"> {
  let summary = "Replaces the calls of the device primitives with the code "
                "of their emitters";
  let description = [{
    A template of the device lowering calls a primitive of
    `CompactionEmitter.h` by name, `func.call @mdir.primitive.<name>(...)`,
    with a declaration of its own (D296,
    docs/device-primitives.md, Section 4). `convert-md-exec-to-gpu`
    replaces each call with the code of the primitive's emitter where it
    stands when it parses a template, so that every site of a primitive,
    in a template or in the lowering, runs the code of one emitter. This
    pass does the same on a module, for the tests of the emitters. A call
    of a primitive that does not exist, or with operands or results that
    the primitive does not have, is an error.
  }];
  let dependentDialects = ["::mlir::arith::ArithDialect",
                           "::mlir::func::FuncDialect",
                           "::mlir::gpu::GPUDialect",
                           "::mlir::LLVM::LLVMDialect",
                           "::mlir::math::MathDialect",
                           "::mlir::memref::MemRefDialect",
                           "::mlir::scf::SCFDialect"];
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
    (D214). The time the serialization took is left out of
    the objects' properties, which the upstream pass records, so that the
    result is the same from run to run.

    With `format=bin`, the toolkit's `ptxas` compiles the PTX of each module
    to a cubin. With `binary=auto`, a module stays PTX when there is no
    `ptxas` or it fails; with `binary=cubin`, that is an error. With
    `MDIR_COMPILE_CACHE_DIR` set, the PTX and the cubin of each module are
    read from and written to the compile cache (D212), unless `cache` is
    false (D217).
  }];
  let options = [
    Option<"format", "format", "std::string", /*default=*/"\"isa\"",
           "isa (PTX, which the driver compiles at load) or bin (a cubin)">,
    Option<"binary", "binary", "std::string", /*default=*/"\"auto\"",
           "MDIR_GPU_BINARY: auto (a cubin, or PTX when ptxas cannot make "
           "one), cubin (a cubin, or an error), or ptx; part of the keys of "
           "the cache">,
    Option<"cache", "cache", "bool", /*default=*/"true",
           "use the compile cache that the environment names; false "
           "bypasses it: no entry is read or written">
  ];
}

def ConvertMDDistToLoops
    : Pass<"convert-md-dist-to-loops", "::mlir::ModuleOp"> {
  let summary = "Runs every domain of md_dist in one process";
  let description = [{
    The reference executor of `md_dist` (D275,
    docs/distributed-ml2.md, Section 6): every op runs for all domains in
    turn, in the order of the domains, which is a valid bulk-synchronous
    schedule of the global view. It is the oracle of every later
    transport.

    | Op | Becomes |
    |---|---|
    | A distributed field | `memref<D x rows x k x f64>`: domain, row, component |
    | `md_dist.partition`, `distribute`, `collect` | Calls of the in-process transport (`runtime/mdrt_dist.c`) |
    | `md_dist.halo`, `halo_copy` (G), `halo_sum` (G^T) | Calls of the transport |
    | `md_dist.sum_pairs`, `gather_pairs` | `scf.for` over the domains, their owned centers, and all their local rows, with the cutoff test and the kernel inlined: a search over all pairs, which shares no code with the neighbor structures of `md_exec` |
    | `md_dist.sum_particles` | `scf.for` over the domains and their rows |
    | `md_dist.all_reduce` | A sum over the domains in their order |

    The values of `md_dist` must be in one block, with no use across
    blocks or functions; the cell must come from `md.orthorhombic_cell`;
    fields are of `f64`. The pass rejects what it cannot run, naming the
    reason. It runs before the passes of `md`, which then compile what
    `md_dist.collect` and `md.evaluate` leave.
  }];
  let dependentDialects = ["::mdir::mdrt::MDRTDialect",
                           "::mlir::arith::ArithDialect",
                           "::mlir::func::FuncDialect",
                           "::mlir::math::MathDialect",
                           "::mlir::memref::MemRefDialect",
                           "::mlir::scf::SCFDialect",
                           "::mlir::vector::VectorDialect"];
}

def ConvertMDDistToMDExec
    : Pass<"convert-md-dist-to-md-exec", "::mlir::ModuleOp"> {
  let summary = "Runs every domain of md_dist through md_exec";
  let description = [{
    The second executor of `md_dist` (D290,
    docs/distributed-ml2.md, Section 7.2): a domain's work is ordinary
    `md_exec` over its local rows, and the passes of `md_exec` that follow
    are the existing ones, on the CPU and on a device. The transfers are
    those of the reference executor (`--convert-md-dist-to-loops`), which
    is the oracle of this one.

    | Op | Becomes, for each domain in turn |
    |---|---|
    | `md_dist.partition`, `distribute`, `collect`, `halo`, `halo_copy`, `halo_sum`, `all_reduce` | As in the reference executor |
    | `md_dist.neighborhood` | `md_exec.build_cells` and `md_exec.build_neighbors ... centers(n) images(explicit)` over the local rows of the domain, owned then ghost |
    | `md_dist.sum_pairs` | `md_exec.pair_for` with a sum, of weight 1/2 on a directed relation |
    | `md_dist.gather_pairs`, one result | `md_exec.pair_for` with the policy `(directed, owner_only)`: its owned rows |
    | `md_dist.gather_pairs`, two results | `md_exec.pair_for` with the policy `(directed, atomic)`: its owned rows and its ghost rows; the kernel halved on a directed relation |
    | `md_dist.sum_particles`, `map_particles` | `md_exec.particle_for` over the owned rows |

    Each domain has two particle sets, declared by the pass,
    `@<decomposition>.d<d>.owned` and `@<decomposition>.d<d>.local`. Its
    rows are copied from the buffers of all domains into buffers of the
    domain, handed to `md_exec` by `mdrt.from_buffer`, and back by
    `mdrt.to_buffer`. The unique traversal takes a pair where the identity
    of its center is the lesser, from a field of the identities of the
    local rows. Besides what the reference executor refuses, the pass
    refuses loops over ghost rows. It runs before
    `--convert-md-to-md-exec`.
  }];
  let options = [
    Option<"width", "width", "int64_t", /*default=*/"64",
           "the number of neighbors that a row of a structure holds at first">,
    Option<"cells", "cells", "int64_t", /*default=*/"3",
           "the number of cells that the reach of a structure spans">,
    ListOption<"functions", "functions", "std::string",
               "the functions to run; all if empty. The ops of md_dist of "
               "the others, and the decompositions they need, are left to "
               "--convert-md-dist-to-loops, which must follow, so that a "
               "test compares the two">
  ];
  let dependentDialects = ["::mdir::md::MDDialect",
                           "::mdir::md_exec::MDExecDialect",
                           "::mdir::mdrt::MDRTDialect",
                           "::mlir::arith::ArithDialect",
                           "::mlir::func::FuncDialect",
                           "::mlir::math::MathDialect",
                           "::mlir::memref::MemRefDialect",
                           "::mlir::scf::SCFDialect",
                           "::mlir::vector::VectorDialect"];
}

#endif // MDIR_CONVERSION_PASSES_TD
