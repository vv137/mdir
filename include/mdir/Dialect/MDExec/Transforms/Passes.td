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

    With `prune-skin` above zero, each refresh with the policy `check`
    keeps a dual list (D114): the inner list, which the loops over pairs
    take, is pruned from the outer one with the reach `cutoff +
    prune-skin` (`prune_skin` of `md_exec.refresh_neighbors`).
  }];
  let options = [
    Option<"pruneSkin", "prune-skin", "double", /*default=*/"0.0",
           "The skin of the inner list of a dual list, in the units of "
           "the cutoff; 0 keeps one list">
  ];
  let dependentDialects = ["::mdir::md_exec::MDExecDialect",
                           "::mdir::mdrt::MDRTDialect"];
}

def RebuildAtInterval
    : Pass<"md-exec-rebuild-at-interval", "::mlir::ModuleOp"> {
  let summary = "Rebuilds neighbor structures at a fixed interval, whether "
                "they are valid or not (opt-in; may miss pairs)";
  let description = [{
    NOT A DEFAULT. A run takes this pass only when its control file sets
    `rebuild_interval`, and the driver then warns at the start of the run.

    Gives every refresh with the policy `check` the policy `interval`: the
    structure is built at the first refresh and at every `interval`-th
    refresh since the last build, and is not tested for validity at the
    refreshes between. A particle that moves more than half the skin in
    fewer refreshes than that leaves pairs out of the structure while they
    are within the cutoff, and the forces between them are not computed.
    The run saves the test at each step, and on a device the wait of the
    host for its result.

    The structure is still tested at each build, before it, and a build
    that finds it no longer valid is counted: the driver reports how many
    were late. A particle that went too far and came back within the
    interval is not seen. See D88.

    ```mlir
    %nl1 = md_exec.refresh_neighbors %nl, %x, %cell ... policy(check)
    ```

    ```mlir
    %nl1 = md_exec.refresh_neighbors %nl, %x, %cell ... interval(10)
             policy(interval)
    ```

    Runs before `md-exec-expose-validity`, which then leaves the refresh
    alone.
  }];
  let options = [
    Option<"interval", "interval", "int64_t", /*default=*/"0",
           "The number of refreshes after a build at which the structure "
           "is built again">
  ];
}

def ChooseNeighbors
    : Pass<"md-exec-choose-neighbors", "::mlir::ModuleOp"> {
  let summary = "Chooses the kind of each neighbor structure and the policy "
                "of the loops over it";
  let description = [{
    With `kind=groups`, a neighbor structure becomes a structure of groups
    of 16 particles that share a list of neighbors (D89,
    docs/groups-m1.md), and every loop over it takes the policy
    `(unique, atomic)`: each pair once, the value of the kernel added to
    both particles. It does so only when every loop over pairs over the
    structure has an exchange contract for each destination, symmetric or
    antisymmetric, and for each sum, symmetric: the value of the kernel for
    (j, i) follows from that for (i, j). Otherwise the structure stays a
    matrix. The weight of a sum stays that of the sum over both orders of
    each pair; the lowering doubles it.

    A structure is found through the refreshes and the loops that carry it
    to where it begins (`md_exec.build_neighbors` or
    `md_exec.empty_neighbors`); a loop over a structure whose beginnings
    cannot be found keeps the matrix, and so do those beginnings.

    Structures of groups are built on a device only: the pass runs on
    programs for a device.

    ```mlir
    %nl = md_exec.empty_neighbors kind(matrix) width(96) ...
    %f, %e = md_exec.pair_for %nl1, ... exchange [antisymmetric, symmetric]
               policy(directed, owner_only) { ... }
    ```

    ```mlir
    %nl = md_exec.empty_neighbors kind(groups) width(96) ...
    %f, %e = md_exec.pair_for %nl1, ... exchange [antisymmetric, symmetric]
               policy(unique, atomic) { ... }
    ```
  }];
  let options = [
    Option<"kind", "kind", "std::string", /*default=*/"\"groups\"",
           "The kind of structure to choose where the loops allow it: "
           "groups or matrix">
  ];
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

    With `radial`, the terms of a value that share their factors that do
    not depend on the distance, and whose part that does includes a
    function other than a root or a power (an exponential, erfc), become
    one term: those factors times `md_exec.radial` of a function of the
    squared distance, which the module gains, `(f64) -> f64`. The screened
    Coulomb force of Ewald summation becomes q_i q_j times one function of
    r², which `md-exec-expand-radial` tabulates (D94).
  }];
  let options = [
    Option<"radial", "radial", "bool", /*default=*/"false",
           "Separate the functions of the distance alone into "
           "md_exec.radial">
  ];
  let dependentDialects = ["::mlir::arith::ArithDialect",
                           "::mlir::func::FuncDialect",
                           "::mlir::math::MathDialect"];
}

def NarrowSums
    : Pass<"md-exec-narrow-sums", "::mlir::ModuleOp"> {
  let summary = "Sums only the elements of a vector sum that are used";
  let description = [{
    A loop that sums a vector over particles, pairs, or tuples sums every
    element, but its users may take only some: the pressure of an isotropic
    barostat and of the log takes the trace of the virial, three of its
    nine elements. The pass follows each vector back from its users,
    through elementwise arithmetic and `vector.extract`, to the sums of the
    loops, and makes each loop sum only the elements used: its initial
    value, the value its kernel yields, and its result shrink to them, and
    a vector of the old type with zeros elsewhere replaces the result. A
    user that the pass does not know uses every element.

    The elements kept are summed as before, to the bit; the kernel no
    longer computes the others, which canonicalization removes.

    ```mlir
    %w = md_exec.pair_for ... reduce(%w0 : vector<9xf64>) { ... }
    %t0 = vector.extract %w[0] : f64 from vector<9xf64>
    %t4 = vector.extract %w[4] : f64 from vector<9xf64>
    %t8 = vector.extract %w[8] : f64 from vector<9xf64>
    ```

    ```mlir
    %d = md_exec.pair_for ... reduce(%d0 : vector<3xf64>) { ... }
    %w = vector.from_elements %d[0], %z, %z, %z, %d[1], %z, %z, %z, %d[2]
    ```
  }];
  let dependentDialects = ["::mlir::vector::VectorDialect"];
}

def FoldTables
    : Pass<"md-exec-fold-tables", "::mlir::ModuleOp"> {
  let summary = "Computes what a kernel derives from tables once per entry "
                "of the tables";
  let description = [{
    A value of the kernel of a loop that depends only on tables, looked up
    at the same indices, and on constants, such as 4 ε σ¹² of Lennard-Jones
    from the tables of σ and ε of the pair of types, is the same for every
    pair of particles of those types. The pass computes it once per entry,
    into a new table (`md_exec.tabulate`), and the kernel looks it up
    there: a load for the arithmetic of each pair of particles. The new
    table is computed in f64, before `md-exec-assign-precision`, so a kernel
    in f32 reads the value rounded once.

    Only the values that the rest of the kernel uses are tabulated; a table
    that serves two loops is made once. The tables that a kernel reads at
    the same indices become one table of vectors, read with one lookup and
    taken apart with `vector.extract`: one load of 8 bytes for the two
    coefficients of Lennard-Jones, not two of 4. The pass runs on kernels in
    f64, before the precision is assigned.

    ```mlir
    %s = md.lookup %sigma[%ti, %tj] : ... -> f64
    %e = md.lookup %epsilon[%ti, %tj] : ... -> f64
    %s2 = arith.mulf %s, %s : f64
    ...
    %a = arith.mulf %e, %s12 : f64
    %force = arith.mulf %a, %r14 : f64
    ```

    ```mlir
    %t = md_exec.tabulate (%sigma, %epsilon : ...) { ... } -> ...
    ...
    %a = md.lookup %t[%ti, %tj] : ... -> f64
    %force = arith.mulf %a, %r14 : f64
    ```
  }];
}

def ExpandRadial
    : Pass<"md-exec-expand-radial", "::mlir::ModuleOp"> {
  let summary = "Makes each md_exec.radial a table in f32, or its function "
                "in f64";
  let description = [{
    In a kernel that computes in f32, `md_exec.radial` becomes a lookup in a
    table of its function of the squared distance s: the intervals of s
    that the bits of s as an f32 number name, with `bits` bits of its
    significand (2^bits intervals for each power of two), from the cutoff
    squared down to 2^-10 of it, and in each a polynomial of degree 3 in
    u = s − (the start of the interval), fitted at four Chebyshev nodes in
    f64. The index of an interval is the bits of s shifted, clamped to the
    table; the start of the interval is exact in f32, and so is u. The pass
    takes the fewest bits, from 4 to 12, whose polynomials are within a
    tenth of `tolerance` of the function in f64, relative to the largest
    value of the function on each interval, at 17 points of each, and fails
    if those in f32, as a kernel evaluates them, are not within
    `tolerance`.
    The table, four f32 numbers for each interval, is a constant of the
    module (`memref.global`), which a device reads with one load of 16
    bytes.

    In a kernel that computes in f64 the function is inlined.
  }];
  let options = [
    Option<"tolerance", "tolerance", "double", /*default=*/"3.0e-7",
           "The largest relative error of a table">
  ];
  let dependentDialects = ["::mlir::arith::ArithDialect",
                           "::mlir::math::MathDialect",
                           "::mlir::memref::MemRefDialect",
                           "::mlir::vector::VectorDialect"];
}

def Approximate
    : Pass<"md-exec-approximate", "::mlir::ModuleOp"> {
  let summary = "Approximates divisions and erfc in the f32 kernels of "
                "loops (fast_math)";
  let description = [{
    The control keyword `fast_math` runs the pass, after
    `md-exec-assign-precision`. In the kernels of `md_exec.pair_for` and
    of `md_exec.tuple_for` over tuples that are not disjoint, the terms of
    the potential, in f32:

    - A division may be approximate (`afn`): on a device it becomes
      `div.approx.f32` or `rcp.approx.f32`, within 2 ulp, instead of the
      correctly rounded sequence.
    - An exponential may be approximate (`afn`): on a device `ex2.approx`
      of the argument times log2(e), within 2 + 1.2 |x| ulp (about 1.5e-6
      at the cutoff of the direct sum of Ewald, x = -(β rc)² ≈ -10),
      instead of the reduction of the argument that `expf` makes.
    - `math.erfc(sqrt(y) * c)`, with `c` a positive constant, becomes

      ```text
      exp(-c² y) · P(t),   t = 1 / (1 + x / 2),   x = sqrt(y) · c,
      ```

      with P a polynomial of degree 9 fitted to erfc(x) exp(x²) on
      [0, 6] (`scripts/fit-erfc.py`): the relative error of P in f32 is
      below 3.3e-7 there, that of `erfcf` a few ulp. The argument of the
      exponential rounds to f32 as it does in the exponential of the
      derivative of erfc, which the kernel of a force has already:
      `exp(y * k)` with `k` within 4 ulp of `-c²` is taken as it is, so the
      two share it. Beyond 6, erfc(x) < 2.2e-17 and the term does not
      count. A negative argument cannot match, so the rewrite never sees
      one.

    The kernels of loops over particles and of loops over disjoint tuples,
    the integrators and the constraints, keep their arithmetic: the
    constraints divide by the masses and solve for their multipliers, and
    an inverse mass within 2 ulp weighs the members of a group otherwise
    than the masses that the kicks divide by, which heated JAC by 7 K in
    2 ns (D112).

    In a force of the direct sum of Ewald, the erfc term is a share
    1 / (1 + 2 x²) of the force factor, so the relative error of the force
    that the approximation adds is at most that of erfc over 1 + 2 x².

    ```mlir
    %s = math.sqrt %r2 : f32
    %x = arith.mulf %s, %beta : f32
    %e = math.erfc %x : f32
    ```

    ```mlir
    %k = arith.mulf %r2, %minus_beta2 : f32
    %g = math.exp %k : f32
    %t = arith.divf %one, %d fastmath<afn> : f32   // %d = 1 + x / 2
    %p = math.fma ...                               // P(t), Horner
    %e = arith.mulf %g, %p : f32
    ```
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
    (`kernel-positions`; D79); a loop over pairs that takes each pair once
    (over groups) keeps the positions as they are stored, which its
    lowering moves to the frames of the groups before converting them
    (D101). Values are converted where they
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

def AccumulateDestinations
    : Pass<"md-exec-accumulate-destinations", "::mlir::ModuleOp"> {
  let summary = "Lets loops that a loop over particles adds up accumulate "
                "into one destination";
  let description = [{
    A loop over particles that adds the fields of several loops over pairs
    or tuples, as the forces of the terms of a potential are added, reads
    them all and each is stored:

    ```mlir
    %f1 = md_exec.pair_for ... outs(%z1 : !vec) { ... }
    %f2 = md_exec.tuple_for ... outs(%z2 : !vec) { ... }
    %v = md_exec.particle_for ins(%f1, %f2, ...) {
    ^bb0(%a: vector<3xf64>, %b: vector<3xf64>, ...):
      %s = arith.addf %a, %b : vector<3xf64>
      ...
    }
    ```

    Where each field is the destination of a loop that accumulates from
    zeros (`md_exec.zeros`), used by the loop over particles alone, and the
    loops come in the order of the sum, each loop accumulates onto what the
    loop before it gave, and the loop over particles reads the last:

    ```mlir
    %f1 = md_exec.pair_for ... outs(%z1 : !vec) { ... }
    %f2 = md_exec.tuple_for ... outs(%f1 : !vec) { ... }
    %v = md_exec.particle_for ins(%f2, ...) { ... }
    ```

    A loop computes `a0 + Σ`, so the sums are those of the loop over
    particles, in the same order, in the type of the destination: to the
    bit where that is the type in which the loop over particles added
    (double precision); in the mixed mode the forces are added in `f32`
    where they were added in `f64` after being stored in `f32`, as GROMACS
    adds them. A run of such loops in one kernel adds to one buffer and
    stores it once (`convert-md-exec-to-gpu`).
  }];
  let dependentDialects = ["::mlir::arith::ArithDialect"];
}

def AssignStreams : Pass<"md-exec-assign-streams", "::mlir::ModuleOp"> {
  let summary = "Runs ops on a second stream beside those they are "
                "independent of";
  let description = [{
    Belongs to the storage form. Marks the ops that the options name
    `md_exec.side`, so that they run on a second stream, and puts an
    `md_exec.join` after each where the first stream waits for it. Each
    marked op first moves up its block past the ops that it is independent
    of and whose results it does not take; the join goes before the first
    op after it that is not independent of it (or that is marked too), or
    before the terminator of the block.

    The ops between a marked op and its join run in another order than the
    program gives, or at once. That leaves memory and every value as the
    program does because each is independent of the marked op: see D87 in
    docs/decisions.md for the argument, and `Independence.h` for what
    independent means. `convert-md-exec-to-gpu` checks it again, so that
    it holds of any program that it lowers.

    Nothing runs beside by default: measured, the sums gained nothing
    beside the refresh of the neighbor structure, and slowed the loops over
    pairs beside them (D81).
  }];
  let options = [
    Option<"reciprocal", "reciprocal", "bool", /*default=*/"false",
           "Run the reciprocal sums of particle mesh Ewald on a second "
           "stream">
  ];
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
