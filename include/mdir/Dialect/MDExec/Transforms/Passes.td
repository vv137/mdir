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

def ExposeValidity
    : Pass<"md-exec-expose-validity", "::mlir::ModuleOp"> {
  let summary = "Makes the test of validity of a neighbor structure a loop";
  let description = [{
    A refresh with the policy `check` tests whether a particle has moved
    more than half the skin since the structure was built. The pass makes
    the test a loop over particles and hands its result to the refresh:

    ```mlir
    %nl1 = md_exec.refresh_neighbors %nl, %x, %cell ... policy(check)
    ```

    ```mlir
    %ref   = md_exec.reference_positions %nl
    %moved = md_exec.particle_for ins(%x, %ref) reduce(%false : i1) {
    ^bb0(%x_i: vector<3xf64>, %ref_i: vector<3xf64>):
      ...   // |x_i − ref_i|² > (skin / 2)²
    }
    %nl1 = md_exec.refresh_neighbors %nl, %x, %cell moved(%moved) ...
             policy(check)
    ```

    `md-exec-fuse-loops`, run afterward, fuses the loop with the loop that
    writes the positions: the thread that moves a particle tests it as
    well. The test is not a global maximum any more, which on a device
    takes three kernels. No thread computes more than the displacement of
    its own particle, and a thread writes only if its particle has moved
    too far.

    The structure is rebuilt in the same steps as before.

    A refresh of an empty structure builds it whatever has moved. The pass
    gives it the policy `always`.
  }];
  let dependentDialects = ["::mlir::arith::ArithDialect",
                           "::mlir::vector::VectorDialect"];
}

def FuseLoops : Pass<"md-exec-fuse-loops", "::mlir::ModuleOp"> {
  let summary = "Fuses loops over pairs, and loops over particles";
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

    Two `md_exec.particle_for` ops on the particles of one set are fused as
    well. The second may read a field that the first writes: a loop over
    particles computes the value of a particle from the values of that
    particle, so the second kernel reads what the first has yielded. A kick
    and the drift that follows it become one loop:

    ```mlir
    %v1 = md_exec.particle_for ins(%v, %f, %m) outs(%e0) { ... }
    %x1 = md_exec.particle_for ins(%x, %v1) outs(%e1) { ... }
    ```

    ```mlir
    %v1, %x1 = md_exec.particle_for ins(%v, %f, %m, %x) outs(%e0, %e1) {
      ...   // the kick, then the drift with the velocity of the kick
    }
    ```

    A field that only the second loop reads is not stored at all.
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

    A product of two sums is not multiplied out: the terms of the expanded
    product can be much larger than the product, and their sum then loses
    digits.

    The pass reassociates floating-point arithmetic, so results change in
    their last bits. It must not be run in the deterministic mode.
  }];
  let dependentDialects = ["::mlir::arith::ArithDialect",
                           "::mlir::math::MathDialect"];
}

def AssignPrecision
    : Pass<"md-exec-assign-precision", "::mlir::ModuleOp"> {
  let summary = "Assigns floating-point types to fields and kernels";
  let description = [{
    Before the pass, every field and every kernel has the reference
    precision, `f64`. The pass gives each field the type it is stored in
    and each kernel the type it computes in, according to the precision
    policy.

    A mode assigns a type to each role:

    | Mode | `position` | `force` | `kernel` | `integrator` | `accumulator` |
    |---|---|---|---|---|---|
    | `single` | `f32` | `f32` | `f32` | `f32` | `f64` |
    | `mixed` | `f64` | `f32` | `f32` | `f64` | `f64` |
    | `double` | `f64` | `f64` | `f64` | `f64` | `f64` |

    An option that is named after a role replaces the type of that role.

    Fields that must have one type are stored together: a loop-carried
    field with the fields it is initialized and updated with, and the field
    that a loop accumulates into with the result of the loop. The type that
    fields are stored in is, in this order:

    1. the element type of a buffer that holds one of them, as
       `mdrt.from_buffer` and `mdrt.to_buffer` state it;
    2. the type of `position`, if one of them is used as positions;
    3. the type of `force`, if a loop over pairs writes one of them;
    4. the type of `integrator`, if a loop over particles writes one of
       them.

    A loop over pairs computes in the type of `kernel` and a loop over
    particles in the type of `integrator`. Where `kernel` is narrower than
    the positions, a loop over particles converts the positions to it once,
    and the loops over pairs and over tuples that are not disjoint (the
    terms of the potential, not the constraints) take that field
    (`kernel-positions`; D79). Values are converted where they
    enter and leave a kernel. Global sums are accumulated in the type of
    `accumulator`.

    Scalars outside the loops keep the type `f64`.

    The results agree with those of the reference precision within a
    tolerance only.
  }];
  let options = [
    Option<"mode", "mode", "std::string", /*default=*/"\"double\"",
           "The precision mode: single, mixed, or double">,
    Option<"position", "position", "std::string", /*default=*/"\"\"",
           "The type of positions that no buffer holds: f32 or f64">,
    Option<"force", "force", "std::string", /*default=*/"\"\"",
           "The type of fields that loops over pairs write: f32 or f64">,
    Option<"kernel", "kernel", "std::string", /*default=*/"\"\"",
           "The type that loops over pairs compute in: f32 or f64">,
    Option<"integrator", "integrator", "std::string", /*default=*/"\"\"",
           "The type that loops over particles compute in: f32 or f64">,
    Option<"accumulator", "accumulator", "std::string",
           /*default=*/"\"\"",
           "The type that global sums are accumulated in: f32 or f64">,
    Option<"kernelPositions", "kernel-positions", "bool",
           /*default=*/"true",
           "Whether loops over pairs, and over tuples that are not "
           "disjoint, take the positions converted once to the type of "
           "the kernel, where that is narrower than the positions">
  ];
  let dependentDialects = ["::mdir::md_exec::MDExecDialect",
                           "::mlir::arith::ArithDialect"];
}

def AssignStorage : Pass<"md-exec-assign-storage", "::mlir::ModuleOp"> {
  let summary = "Gives every field a buffer";
  let description = [{
    Converts the ops of `md_exec` from the value form to the storage form.
    Before the pass, a loop takes fields and returns fields. After it, a
    loop takes the buffers that hold the fields and updates them where they
    are.

    Before:

    ```mlir
    %v0 = md_exec.empty : !md.field<@atoms, 3 x f64>
    %v1 = md_exec.particle_for ins(%v, %f, %m : ...) outs(%v0 : ...) { ... }
            -> !md.field<@atoms, 3 x f64>
    ```

    After, if `%v` is not used afterward:

    ```mlir
    md_exec.particle_for
        ins(%v, %f, %m : memref<?x3xf64>, memref<?x3xf64>, memref<?xf64>)
        outs(%v : memref<?x3xf64>) { ... }
    ```

    The pass decides which buffer holds which field, which loops write to
    the buffer of a field that they read, and which buffers a loop carries.
    It never copies a field: where a field needs a buffer of its own and
    none is free, the pass fails and names the reason.

    A neighbor structure gets storage, which is refreshed where it is.

    With `memory=device` the buffers are on a device, and their types have
    a memory space. `mdrt.from_buffer` copies a field to the device and
    `mdrt.to_buffer` copies it back. Loops with global sums, and refreshes
    that test validity, get the buffers they need for themselves from the
    pool, as `scratch`; the lowering to a device does not allocate.

    Everything else that depends on where a field is stored, such as its
    layout, is decided on the result of this pass.
  }];
  let options = [
    Option<"memory", "memory", "std::string", /*default=*/"\"host\"",
           "Where the buffers are: host or device">,
    Option<"tables", "tables", "std::string", /*default=*/"\"f64\"",
           "The type in which tables are copied to a device: f64, or f32 "
           "where the kernels compute in f32 and would convert every "
           "value they look up">
  ];
  let dependentDialects = ["::mdir::md_exec::MDExecDialect",
                           "::mlir::arith::ArithDialect",
                           "::mlir::gpu::GPUDialect",
                           "::mlir::memref::MemRefDialect"];
}

#endif // MDIR_DIALECT_MDEXEC_TRANSFORMS_PASSES_TD
