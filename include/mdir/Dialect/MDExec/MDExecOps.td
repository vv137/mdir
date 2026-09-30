// Ops of the md_exec dialect.
//
// The ops that touch fields have two forms. In the value form they take
// fields and return fields. In the storage form they take the buffers that
// hold the fields and update them where they are. See docs/ops-m0.md,
// Section 8, for the specification.

#ifndef MDIR_DIALECT_MDEXEC_MDEXECOPS_TD
#define MDIR_DIALECT_MDEXEC_MDEXECOPS_TD

include "mlir/IR/BuiltinAttributes.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mdir/Dialect/MD/MDEnums.td"
include "mdir/Dialect/MD/MDTypes.td"
include "mdir/Dialect/MDExec/MDExecDialect.td"
include "mdir/Dialect/MDExec/MDExecEnums.td"
include "mdir/Dialect/MDRT/MDRTTypes.td"

def MDExec_Buffer : Type<
    CPred<"::mdir::md_exec::isBufferType($_self)">,
    "buffer of a field", "::mlir::MemRefType">;

def MDExec_FieldOrBuffer : AnyTypeOf<[MD_FieldType, MDExec_Buffer]>;

// A buffer that an op may use as it likes: one number per particle, or a
// fixed number of them.
def MDExec_Scratch : Type<
    CPred<"::mdir::md_exec::isScratchType($_self)">,
    "buffer with one value per particle", "::mlir::MemRefType">;

// The members of the tuples of a tuple set: `memref<?xkxi32>`.
def MDExec_Members : Type<
    CPred<"::mdir::md_exec::isMembersType($_self)">,
    "buffer of the members of tuples", "::mlir::MemRefType">;

// The tuples by particle: `memref<?x?xi32>`.
def MDExec_IncidenceBuffer : Type<
    CPred<"::mdir::md_exec::isIncidenceBufferType($_self)">,
    "buffer of the tuples of each particle", "::mlir::MemRefType">;

def MDExec_RelationOrMembers : AnyTypeOf<[MD_RelationType, MDExec_Members]>;
def MDExec_Incidence
    : AnyTypeOf<[MDRT_IncidenceType, MDExec_IncidenceBuffer]>;

//===----------------------------------------------------------------------===//
// Destinations
//===----------------------------------------------------------------------===//

def MDExec_ZerosOp : MDExec_Op<"zeros", [Pure]> {
  let summary = "A field of zeros";
  let description = [{
    The destination of a loop that accumulates from zero.

    ```mlir
    %f0 = md_exec.zeros : !md.field<@atoms, 3 x f64>
    ```
  }];
  let results = (outs MD_FieldType:$result);
  let assemblyFormat = "attr-dict `:` qualified(type($result))";
}

def MDExec_EmptyOp : MDExec_Op<"empty", [Pure]> {
  let summary = "A field with unspecified values";
  let description = [{
    The destination of a loop that writes every value. Reading the field
    before it is written is an error.

    ```mlir
    %v0 = md_exec.empty : !md.field<@atoms, 3 x f64>
    ```
  }];
  let results = (outs MD_FieldType:$result);
  let assemblyFormat = "attr-dict `:` qualified(type($result))";
}

//===----------------------------------------------------------------------===//
// Spatial structures
//===----------------------------------------------------------------------===//

def MDExec_BuildCellsOp : MDExec_Op<"build_cells", [Pure]> {
  let summary = "Bins particles into cells";
  let description = [{
    Divides the simulation cell into cells that are at least `width` wide in
    every direction and assigns each particle to one.

    ```mlir
    %cells = md_exec.build_cells %x, %cell width(2.8)
               : !md.field<@atoms, 3 x f64> -> !mdrt.cells<@atoms>
    ```
  }];
  let arguments = (ins
    MD_FieldType:$positions,
    MD_CellType:$cell,
    F64Attr:$width
  );
  let results = (outs MDRT_CellsType:$result);
  let assemblyFormat = [{
    $positions `,` $cell `width` `(` $width `)` attr-dict
    `:` qualified(type($positions)) `->` qualified(type($result))
  }];
  let hasVerifier = 1;
}

def MDExec_Order : AnyTypeOf<[MDRT_PermutationType, MDExec_Buffer]>;

def MDExec_SpatialOrderOp : MDExec_Op<"spatial_order", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let summary = "Computes the order of the particles by cell";
  let description = [{
    Returns the order in which the particles follow one another in space:
    by the cell that they are in, in cells that are `width` wide or wider,
    and within a cell by `ids`. Entry `k` of the order is the particle that
    comes to place `k`.

    The cells are numbered along x first, then along y and z. Particles
    that are near one another in space are then near one another in
    memory, and a loop over pairs reads the neighbors of a particle from
    few places.

    `ids` holds a number for each particle that no other particle has. The
    order depends on the positions and on these numbers only, not on the
    order that the particles are in. A run that continues from a
    checkpoint therefore arrives at the order of the run that was not
    interrupted.

    ```mlir
    %order = md_exec.spatial_order %x, %cell, %ids width(1.4)
               : !md.field<@atoms, 3 x f64>, !md.field<@atoms, i32>
               -> !mdrt.permutation<@atoms>
    ```

    In the storage form the fields are buffers, and the order is written
    to the buffer in `outs`, which holds one `i32` per particle.

    ```mlir
    md_exec.spatial_order %x, %cell, %ids outs(%order : memref<?xi32>)
        width(1.4) : memref<?x3xf64>, memref<?xi32>
    ```
  }];
  let arguments = (ins
    MDExec_FieldOrBuffer:$positions,
    MD_CellType:$cell,
    MDExec_FieldOrBuffer:$ids,
    Optional<MDExec_Buffer>:$order,
    F64Attr:$width
  );
  let results = (outs Optional<MDRT_PermutationType>:$result);
  let assemblyFormat = [{
    $positions `,` $cell `,` $ids
    (`outs` `(` $order^ `:` type($order) `)`)?
    `width` `(` $width `)` attr-dict
    `:` qualified(type($positions)) `,` qualified(type($ids))
    (`->` qualified(type($result))^)?
  }];
  let extraClassDeclaration = [{
    bool isStorageForm() {
      return ::llvm::isa<::mlir::MemRefType>(getPositions().getType());
    }
  }];
  let hasVerifier = 1;
}

def MDExec_ReciprocalOp : MDExec_Op<"reciprocal", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let summary = "The reciprocal sum of particle mesh Ewald";
  let description = [{
    What `md.reciprocal` computes: the energy, the virial, and the forces
    of the reciprocal sum of smooth particle mesh Ewald (docs/pme-m1.md).

    ```mlir
    %u, %w, %f = md_exec.reciprocal %x, %q, %cell, %moduli
                   grid([32, 32, 32]) order(4) beta(3.12) coulomb(138.935)
                   : !md.field<@atoms, 3 x f64>, !md.field<@atoms, f64>,
                     !md.table<2, f64>
                   -> f64, vector<9xf64>, !md.field<@atoms, 3 x f64>
    ```

    In the storage form the fields and the table are buffers, the forces
    are written to the buffer in `outs`, and `scratch` holds the grid: the
    charges in fixed point, the grid in f64, and the half-complex grid of
    its transform, then on a device the sums of the rows of the grid.
  }];
  let arguments = (ins
    MDExec_FieldOrBuffer:$positions,
    MDExec_FieldOrBuffer:$charges,
    MD_CellType:$cell,
    AnyType:$moduli,
    Optional<MDExec_Buffer>:$out,
    Variadic<MDExec_Buffer>:$scratch,
    DenseI64ArrayAttr:$grid,
    I64Attr:$order,
    F64Attr:$beta,
    F64Attr:$coulomb
  );
  let results = (outs
    F64:$energy,
    VectorOfLengthAndType<[9], [F64]>:$virial,
    Optional<MD_FieldType>:$forces
  );
  let assemblyFormat = [{
    $positions `,` $charges `,` $cell `,` $moduli
    (`outs` `(` $out^ `:` type($out) `)`)?
    (`scratch` `(` $scratch^ `:` type($scratch) `)`)?
    `grid` `(` $grid `)` `order` `(` $order `)` `beta` `(` $beta `)`
    `coulomb` `(` $coulomb `)`
    attr-dict `:` qualified(type($positions)) `,` qualified(type($charges))
    `,` qualified(type($moduli)) `->` type($energy) `,` type($virial)
    (`,` qualified(type($forces))^)?
  }];
  let extraClassDeclaration = [{
    bool isStorageForm() {
      return ::llvm::isa<::mlir::MemRefType>(getPositions().getType());
    }
  }];
}

def MDExec_PermuteOp : MDExec_Op<"permute", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let summary = "Applies an order to a field";
  let description = [{
    Returns the field with the value of particle `order[k]` at place `k`.

    ```mlir
    %xs = md_exec.permute %x, %order
            : !md.field<@atoms, 3 x f64>, !mdrt.permutation<@atoms>
            -> !md.field<@atoms, 3 x f64>
    ```

    In the storage form the field and the order are buffers, and the
    result is written to the buffer in `outs`, which is not the buffer of
    the field.

    ```mlir
    md_exec.permute %x, %order outs(%xs : memref<?x3xf64>)
        : memref<?x3xf64>, memref<?xi32>
    ```
  }];
  let arguments = (ins
    MDExec_FieldOrBuffer:$field,
    MDExec_Order:$order,
    Optional<MDExec_Buffer>:$out
  );
  let results = (outs Optional<MD_FieldType>:$result);
  let assemblyFormat = [{
    $field `,` $order (`outs` `(` $out^ `:` type($out) `)`)? attr-dict
    `:` qualified(type($field)) `,` qualified(type($order))
    (`->` qualified(type($result))^)?
  }];
  let extraClassDeclaration = [{
    bool isStorageForm() {
      return ::llvm::isa<::mlir::MemRefType>(getField().getType());
    }
  }];
  let hasVerifier = 1;
}

def MDExec_BuildNeighborsOp : MDExec_Op<"build_neighbors", [Pure]> {
  let summary = "Builds a physical neighbor structure";
  let description = [{
    The structure is built at the configuration `positions` with the
    extended cutoff `cutoff + skin`. It holds the directed list

    ```text
    L = { (i, j) : i ≠ j, r_ij < cutoff + skin }
    ```

    and is valid for any configuration in which no particle has moved more
    than `skin / 2` and the cell is unchanged.

    `width` is the number of neighbors that the structure can hold for one
    particle.

    ```mlir
    %nl = md_exec.build_neighbors %cells, %x, %cell
            cutoff(2.5) skin(0.3) kind(matrix) width(96)
            : !mdrt.cells<@atoms>, !md.field<@atoms, 3 x f64>
            -> !mdrt.neighbors<@atoms>
    ```

    With `exclude`, the structure leaves out the pairs of an incidence
    structure of a tuple set of arity 2. A row must be wide enough for the
    neighbors of a particle before those are left out.
  }];
  let arguments = (ins
    MDRT_CellsType:$cells,
    MD_FieldType:$positions,
    MD_CellType:$cell,
    Optional<MDExec_Incidence>:$excluded,
    F64Attr:$cutoff,
    F64Attr:$skin,
    MDExec_NeighborKind:$kind,
    I64Attr:$width
  );
  let results = (outs MDRT_NeighborsType:$result);
  let assemblyFormat = [{
    $cells `,` $positions `,` $cell
    `cutoff` `(` $cutoff `)` `skin` `(` $skin `)`
    `kind` `(` $kind `)` `width` `(` $width `)`
    (`exclude` `(` $excluded^ `:` qualified(type($excluded)) `)`)? attr-dict
    `:` qualified(type($cells)) `,` qualified(type($positions))
    `->` qualified(type($result))
  }];
  let hasVerifier = 1;
}

def MDExec_EmptyNeighborsOp : MDExec_Op<"empty_neighbors", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let summary = "A neighbor structure that is valid for no configuration";
  let description = [{
    The first refresh of the structure builds it.

    ```mlir
    %nl0 = md_exec.empty_neighbors kind(matrix) width(96)
             : !mdrt.neighbors<@atoms>
    ```

    In the storage form the op allocates the storage of a structure for
    `size` particles. `positions` is the type of the buffers that hold the
    positions the structure is built at. The storage is where those buffers
    are: on the host, or on a device if the type has a memory space.

    ```mlir
    %nl = md_exec.empty_neighbors size(%n) positions(memref<?x3xf32>)
            kind(matrix) width(96) : !mdrt.neighbors<@atoms>
    ```

    `exclude` is as for `md_exec.build_neighbors`: every build of the
    structure leaves out those pairs.
  }];
  let arguments = (ins
    Optional<Index>:$size,
    Optional<MDExec_Incidence>:$excluded,
    OptionalAttr<TypeAttr>:$positions,
    MDExec_NeighborKind:$kind,
    I64Attr:$width
  );
  let results = (outs MDRT_NeighborsType:$result);
  let assemblyFormat = [{
    (`size` `(` $size^ `)` `positions` `(` $positions `)`)?
    `kind` `(` $kind `)` `width` `(` $width `)`
    (`exclude` `(` $excluded^ `:` qualified(type($excluded)) `)`)? attr-dict
    `:` qualified(type($result))
  }];
  let extraClassDeclaration = [{
    bool isStorageForm() { return static_cast<bool>(getSize()); }
  }];
  let hasVerifier = 1;
}

def MDExec_RefreshNeighborsOp : MDExec_Op<"refresh_neighbors", [
    AllTypesMatch<["neighbors", "result"]>, AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let summary = "A neighbor structure that is valid for a configuration";
  let description = [{
    Returns `neighbors` if it is valid for the configuration `positions` in
    the cell `cell`, and a structure built at that configuration otherwise.

    A structure that was built at the configuration `x_ref` in the cell of
    edges `L_ref` is valid in the cell of edges `L` when, with the scale
    `m = L / L_ref` of each axis, `R = cutoff + skin`, and `r_c = cutoff`,

    ```text
    2 max_i |x_i − m ⊙ x_ref,i| ≤ min(m) R − r_c
    ```

    Under that condition it contains every pair within the cutoff: a pair
    that it leaves out was farther than `R` apart at the build, so it is
    farther than `min(m) R − 2 max_i |x_i − m ⊙ x_ref,i|` apart now. In
    the cell of the build the condition is `max_i |x_i − x_ref,i| ≤ skin /
    2`; a barostat that scales the positions with the cell costs only the
    margin `(1 − min(m)) R` (D80). With the
    policy `always` the structure is built whether it is valid or not.

    `cell_width` is the least width of the cells that a rebuild bins the
    particles into. The build chooses the width from the density of the
    particles: the cutoff plus the skin, half of it, or a third.

    ```mlir
    %nl1 = md_exec.refresh_neighbors %nl, %x, %cell
             cutoff(2.5) skin(0.3) cell_width(2.8) policy(check)
             : !mdrt.neighbors<@atoms>, !md.field<@atoms, 3 x f64>
    ```

    In the storage form `positions` is a buffer. The structure is rebuilt
    where it is, and the result is `neighbors` itself.

    ```mlir
    %nl1 = md_exec.refresh_neighbors %nl, %x, %cell
             cutoff(2.5) skin(0.3) cell_width(2.8) policy(check)
             : !mdrt.neighbors<@atoms>, memref<?x3xf64>
    ```

    With the policy `interval`, the structure is built at the first
    refresh and at every `interval`-th refresh since the last build,
    whether it is valid or not; the displacements are tested only at those
    refreshes, before the build, and a build that finds the structure no
    longer valid is counted (`mdrtCountLateBuild`): pairs within the cutoff
    may have been missed since. The test does not see a particle that went
    too far and came back within the interval. Not a default of any run
    (`md-exec-rebuild-at-interval`, D88).

    ```mlir
    %nl1 = md_exec.refresh_neighbors %nl, %x, %cell
             cutoff(2.5) skin(0.3) cell_width(2.8) interval(10)
             policy(interval)
             : !mdrt.neighbors<@atoms>, !md.field<@atoms, 3 x f64>
    ```

    `scratch` holds buffers that the op may use as it likes. What they hold
    before and after is unspecified. On a device the test of validity needs
    two, with one value of the type of the positions per particle.

    With `moved`, the op does not test the displacements: `moved` tells
    whether the condition above fails for a particle. A loop over particles
    computes it from the positions, `md_exec.reference_positions`, and the
    cells (`md_exec.reference_cell`). The op then needs nothing in
    `scratch`.

    ```mlir
    %nl1 = md_exec.refresh_neighbors %nl, %x, %cell moved(%moved)
             cutoff(2.5) skin(0.3) cell_width(2.8) policy(check)
             : !mdrt.neighbors<@atoms>, !md.field<@atoms, 3 x f64>
    ```
  }];
  let arguments = (ins
    MDRT_NeighborsType:$neighbors,
    MDExec_FieldOrBuffer:$positions,
    MD_CellType:$cell,
    Variadic<MDExec_Scratch>:$scratch,
    Optional<I1>:$moved,
    F64Attr:$cutoff,
    F64Attr:$skin,
    F64Attr:$cell_width,
    DefaultValuedAttr<MDExec_RebuildPolicy,
                      "::mdir::md_exec::RebuildPolicy::Check">:$policy,
    OptionalAttr<I64Attr>:$interval
  );
  let results = (outs MDRT_NeighborsType:$result);
  let assemblyFormat = [{
    $neighbors `,` $positions `,` $cell
    (`scratch` `(` $scratch^ `:` type($scratch) `)`)?
    (`moved` `(` $moved^ `)`)?
    `cutoff` `(` $cutoff `)` `skin` `(` $skin `)`
    `cell_width` `(` $cell_width `)` (`interval` `(` $interval^ `)`)?
    `policy` `(` $policy `)` attr-dict
    `:` qualified(type($neighbors)) `,` qualified(type($positions))
  }];
  let extraClassDeclaration = [{
    bool isStorageForm() {
      return ::llvm::isa<::mlir::MemRefType>(getPositions().getType());
    }
  }];
  let hasVerifier = 1;
}

def MDExec_ReferencePositionsOp : MDExec_Op<"reference_positions", [Pure]> {
  let summary = "The configuration that a neighbor structure was built at";
  let description = [{
    Returns the positions `x_ref` that `neighbors` was built at. For a
    structure that has not been built, the values are unspecified.

    ```mlir
    %ref = md_exec.reference_positions %nl
             : !mdrt.neighbors<@atoms> -> !md.field<@atoms, 3 x f64>
    ```

    A loop over particles tests with them whether the structure is still
    valid, and hands the result to `md_exec.refresh_neighbors` as `moved`:

    ```mlir
    %moved = md_exec.particle_for ins(%x, %ref : ...) reduce(%false : i1) {
    ^bb0(%x_i: vector<3xf64>, %ref_i: vector<3xf64>):
      ...   // |x_i − ref_i|² > (skin / 2)²
      md_exec.yield %far : i1
    } -> i1
    ```

    Such a loop can be fused with the loop that writes the positions. The
    test then costs no loop of its own.

    In the storage form the result is the buffer of the structure that
    holds the positions. A refresh that builds the structure writes to it.

    ```mlir
    %ref = md_exec.reference_positions %nl
             : !mdrt.neighbors<@atoms> -> memref<?x3xf64>
    ```
  }];
  let arguments = (ins MDRT_NeighborsType:$neighbors);
  let results = (outs MDExec_FieldOrBuffer:$result);
  let assemblyFormat = [{
    $neighbors attr-dict
    `:` qualified(type($neighbors)) `->` qualified(type($result))
  }];
  let extraClassDeclaration = [{
    bool isStorageForm() {
      return ::llvm::isa<::mlir::MemRefType>(getResult().getType());
    }
  }];
  let hasVerifier = 1;
}

def MDExec_ReferenceCellOp : MDExec_Op<"reference_cell", [Pure]> {
  let summary = "The cell that a neighbor structure was built in";
  let description = [{
    Returns the cell that `neighbors` was built in. For a structure that
    has not been built, it is unspecified. With the cell of the step, it
    tells how far a barostat has scaled the configuration since the build,
    which the test of validity allows for (D80).

    ```mlir
    %cell0 = md_exec.reference_cell %nl : !mdrt.neighbors<@atoms>
    ```
  }];
  let arguments = (ins MDRT_NeighborsType:$neighbors);
  let results = (outs MD_CellType:$result);
  let assemblyFormat = [{
    $neighbors attr-dict `:` qualified(type($neighbors))
  }];
}

def MDExec_CellEdgesOp : MDExec_Op<"cell_edges", [Pure]> {
  let summary = "The lengths of the edges of an orthorhombic cell";
  let description = [{
    ```mlir
    %edges = md_exec.cell_edges %cell : vector<3xf64>
    ```
  }];
  let arguments = (ins MD_CellType:$cell);
  let results = (outs VectorOfLengthAndType<[3], [F64]>:$result);
  let assemblyFormat = "$cell attr-dict `:` type($result)";
}

def MDExec_ResetNeighborsOp : MDExec_Op<"reset_neighbors"> {
  let summary = "Makes a neighbor structure valid for no configuration";
  let description = [{
    Belongs to the storage form. The storage of a structure is allocated in
    the body of the function. Where the value form has an empty structure
    inside a loop, the storage form resets the structure, so that every
    iteration begins with a structure that the first refresh builds.

    The count of builds starts again from zero.

    ```mlir
    md_exec.reset_neighbors %nl : !mdrt.neighbors<@atoms>
    ```

    With `exclude`, the builds from here on leave out the pairs of that
    incidence structure, in place of those that the structure had. A loop
    whose iterations renumber the particles gives each iteration its own.
  }];
  let arguments = (ins
    Arg<MDRT_NeighborsType, "the structure", [MemWrite]>:$neighbors,
    Optional<MDExec_IncidenceBuffer>:$excluded
  );
  let assemblyFormat = [{
    $neighbors (`exclude` `(` $excluded^ `:` type($excluded) `)`)? attr-dict
    `:` qualified(type($neighbors))
  }];
}

def MDExec_RebuildCountOp : MDExec_Op<"rebuild_count"> {
  let summary = "The number of times a neighbor structure has been built";
  let description = [{
    Counts the builds of the structure since it was created, the first one
    included.

    The type of a neighbor structure is the same in both forms, so the op
    cannot tell them apart. It reads the structure, which in the storage
    form changes where it is.

    ```mlir
    %builds = md_exec.rebuild_count %nl : !mdrt.neighbors<@atoms>
    ```
  }];
  let arguments = (ins
    Arg<MDRT_NeighborsType, "the structure", [MemRead]>:$neighbors
  );
  let results = (outs I64:$result);
  let assemblyFormat = [{
    $neighbors attr-dict `:` qualified(type($neighbors))
  }];
}

//===----------------------------------------------------------------------===//
// Streams
//===----------------------------------------------------------------------===//

def MDExec_JoinOp : MDExec_Op<"join"> {
  let summary = "Waits for the op that runs beside the ops before it";
  let description = [{
    Belongs to the storage form. An op marked `md_exec.side` runs on a
    second stream, beside the ops that follow it, up to the next
    `md_exec.join` in its block: the ops that follow the join run after it.
    Every op between the two must be independent of the marked op
    (`md-exec-assign-streams`, D87).

    ```mlir
    %e, %w = md_exec.reciprocal ... {md_exec.side}
    md_exec.refresh_neighbors ...
    md_exec.join
    md_exec.particle_for ins(%forces, ...) ...
    ```

    The op declares no effects, so that no pass moves an op across it, or
    moves it.
  }];
  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// Loops
//===----------------------------------------------------------------------===//

def MDExec_YieldOp : MDExec_Op<"yield", [
    Pure, Terminator, ReturnLike,
    ParentOneOf<["PairForOp", "ParticleForOp", "TupleForOp"]>
  ]> {
  let summary = "Yields the contributions of a kernel";
  let arguments = (ins Variadic<AnyType>:$values);
  let assemblyFormat = "attr-dict ($values^ `:` type($values))?";
}

def MDExec_PairForOp : MDExec_Op<"pair_for", [
    AttrSizedOperandSegments, RecursiveMemoryEffects, SingleBlock,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let summary = "Runs a kernel over the pairs of a neighbor structure";
  let description = [{
    With the directed traversal, for every particle `i` and every entry `j`
    of its list with `r_ij < cutoff`:

    ```text
    a_i = a0_i + Σ_j k_a(i, j)          for every field in `outs`
    S   = S0   + w · Σ_ij k_S(i, j)     for every value in `reduce`
    ```

    The loop computes the displacement `d` and the squared distance `r2`
    from the positions and the cell, and evaluates the cutoff predicate. The
    kernel receives `r2`, `d`, and two values for each field in `ins`. It
    yields one contribution for each field in `outs`, then one for each
    value in `reduce`.

    ```mlir
    %f, %u = md_exec.pair_for %nl, %x, %cell
               outs(%f0 : !md.field<@atoms, 3 x f64>)
               reduce(%u0 : f64) cutoff(2.5) weights [0.5]
               policy(directed, owner_only) {
    ^bb0(%r2: f64, %d: vector<3xf64>):
      ...
      md_exec.yield %k_f, %k_u : vector<3xf64>, f64
    } : !mdrt.neighbors<@atoms>, !md.field<@atoms, 3 x f64>
        -> !md.field<@atoms, 3 x f64>, f64
    ```

    In the storage form the positions and the fields are buffers. A buffer
    in `outs` is updated where it is, and the op has results for the values
    in `reduce` only. `overwrite` has one flag per buffer in `outs`: if it
    is set, the loop ignores what the buffer holds, as if it held zeros. If
    it is absent, no flag is set.

    ```mlir
    %u = md_exec.pair_for %nl, %x, %cell
           outs(%f : memref<?x3xf64>) reduce(%u0 : f64)
           cutoff(2.5) weights [0.5] overwrite [true]
           policy(directed, owner_only) {
    ^bb0(%r2: f64, %d: vector<3xf64>):
      ...
      md_exec.yield %k_f, %k_u : vector<3xf64>, f64
    } : !mdrt.neighbors<@atoms>, memref<?x3xf64> -> f64
    ```

    `scratch` holds buffers that the loop may use as it likes. What they
    hold before and after is unspecified. On a device a loop needs two for
    each value in `reduce`, with one value of the type of the sum per
    particle.

    `exchange` has one contract for each field in `outs` and then each
    value in `reduce`, in the order of the yield: how the kernel of that
    value behaves when the two particles are swapped, from the
    `md.gather_relation` or `md.sum_relation` it came from. A loop may
    compute a pair once for its two particles only if every value is
    `symmetric` or `antisymmetric` (docs/groups-m1.md, D89). Without it,
    every value is `none`.

    ```mlir
    %f, %u = md_exec.pair_for %nl, %x, %cell
               outs(%f0 : !md.field<@atoms, 3 x f64>)
               reduce(%u0 : f64) cutoff(2.5) weights [0.5]
               exchange [antisymmetric, symmetric]
               policy(directed, owner_only) { ... }
    ```
  }];
  let arguments = (ins
    MDRT_NeighborsType:$neighbors,
    MDExec_FieldOrBuffer:$positions,
    MD_CellType:$cell,
    Variadic<MDExec_FieldOrBuffer>:$ins,
    Variadic<MDExec_FieldOrBuffer>:$outs,
    Variadic<AnyType>:$reduce,
    Variadic<MDExec_Scratch>:$scratch,
    F64Attr:$cutoff,
    OptionalAttr<DenseF64ArrayAttr>:$weights,
    OptionalAttr<DenseBoolArrayAttr>:$overwrite,
    DefaultValuedAttr<MDExec_Traversal,
                      "::mdir::md_exec::Traversal::Directed">:$traversal,
    DefaultValuedAttr<MDExec_Conflict,
                      "::mdir::md_exec::Conflict::OwnerOnly">:$conflict,
    OptionalAttr<TypedArrayAttrBase<MD_Exchange,
                 "exchange contracts">>:$exchange
  );
  let regions = (region SizedRegion<1>:$kernel);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = [{
    $neighbors `,` $positions `,` $cell
    (`ins` `(` $ins^ `:` qualified(type($ins)) `)`)?
    (`outs` `(` $outs^ `:` qualified(type($outs)) `)`)?
    (`reduce` `(` $reduce^ `:` type($reduce) `)`)?
    (`scratch` `(` $scratch^ `:` type($scratch) `)`)?
    `cutoff` `(` $cutoff `)`
    (`weights` $weights^)?
    (`overwrite` $overwrite^)?
    custom<ExchangeList>($exchange)
    `policy` `(` $traversal `,` $conflict `)`
    $kernel attr-dict
    `:` qualified(type($neighbors)) `,` qualified(type($positions))
    (`->` qualified(type($results))^)?
  }];
  let extraClassDeclaration = [{
    bool isStorageForm() {
      return ::llvm::isa<::mlir::MemRefType>(getPositions().getType());
    }
    /// Returns true if the loop ignores what destination `index` holds.
    bool overwrites(unsigned index) {
      auto flags = getOverwrite();
      return flags && (*flags)[index];
    }
    /// The exchange contract of value `index`: a destination in `outs`,
    /// then a value in `reduce`. `none` without the attribute.
    ::mdir::md::Exchange getExchange(unsigned index);
  }];
  let hasVerifier = 1;
  let hasRegionVerifier = 1;
}

def MDExec_ParticleForOp : MDExec_Op<"particle_for", [
    AttrSizedOperandSegments, RecursiveMemoryEffects, SingleBlock,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let summary = "Runs a kernel over particles";
  let description = [{
    For every particle `i`:

    ```text
    b_i = k_b(i)                 for every field in `outs`
    S   = S0 + Σ_i k_S(i)        for every number or vector in `reduce`
    A   = A0 or any_i k_A(i)     for every value of the type i1 in `reduce`
    ```

    A field in `outs` is written, not accumulated into. Its old values are
    not read.

    A value of the type `i1` in `reduce` tells whether the kernel yields
    true for any particle. It needs nothing in `scratch`: on a device, a
    thread that yields true sets a flag, and the others do nothing.

    The kernel receives one value for each field in `ins`. It yields one
    value for each field in `outs`, then one contribution for each value in
    `reduce`.

    ```mlir
    %v1 = md_exec.particle_for
            ins(%v, %f, %m : !md.field<@atoms, 3 x f64>,
                             !md.field<@atoms, 3 x f64>,
                             !md.field<@atoms, f64>)
            outs(%v0 : !md.field<@atoms, 3 x f64>) {
    ^bb0(%v_i: vector<3xf64>, %f_i: vector<3xf64>, %m_i: f64):
      ...
      md_exec.yield %v_new : vector<3xf64>
    } -> !md.field<@atoms, 3 x f64>
    ```

    In the storage form the fields are buffers. A buffer in `outs` is
    written where it is, and the op has results for the values in `reduce`
    only. A buffer may be in `ins` and in `outs`: the kernel of a particle
    reads and writes the values of that particle only.

    ```mlir
    md_exec.particle_for
        ins(%v, %f, %m : memref<?x3xf64>, memref<?x3xf64>, memref<?xf64>)
        outs(%v : memref<?x3xf64>) {
    ^bb0(%v_i: vector<3xf64>, %f_i: vector<3xf64>, %m_i: f64):
      ...
      md_exec.yield %v_new : vector<3xf64>
    }
    ```

    `scratch` is as for `md_exec.pair_for`.
  }];
  let arguments = (ins
    Variadic<MDExec_FieldOrBuffer>:$ins,
    Variadic<MDExec_FieldOrBuffer>:$outs,
    Variadic<AnyType>:$reduce,
    Variadic<MDExec_Scratch>:$scratch
  );
  let regions = (region SizedRegion<1>:$kernel);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = [{
    (`ins` `(` $ins^ `:` qualified(type($ins)) `)`)?
    (`outs` `(` $outs^ `:` qualified(type($outs)) `)`)?
    (`reduce` `(` $reduce^ `:` type($reduce) `)`)?
    (`scratch` `(` $scratch^ `:` type($scratch) `)`)?
    $kernel attr-dict
    (`->` qualified(type($results))^)?
  }];
  let extraClassDeclaration = [{
    bool isStorageForm() {
      ::mlir::Value field =
          getIns().empty() ? getOuts().front() : getIns().front();
      return ::llvm::isa<::mlir::MemRefType>(field.getType());
    }
  }];
  let hasVerifier = 1;
  let hasRegionVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Tuples of a topology
//===----------------------------------------------------------------------===//

def MDExec_BuildIncidenceOp : MDExec_Op<"build_incidence", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let summary = "Finds, for every particle, the tuples it is a member of";
  let description = [{
    A loop over tuples visits, for every particle, the tuples that the
    particle is a member of. This op finds them.

    ```mlir
    %incidence = md_exec.build_incidence %angles
        : !md.relation<@atoms, 3, reversal, @angles>
          -> !mdrt.incidence<@atoms, @angles, 3>
    ```

    The members of a tuple are places in the fields of the particles. The
    result holds as long as the particles keep their places.

    In the storage form the op takes the buffer of the members and the
    number of particles, and it returns a new buffer. For every particle
    the buffer has a row: the number of tuples of the particle, then for
    each tuple its number, the place of the particle in it, and its
    members. A row holds as many tuples as the particle with the most
    tuples has.

    ```mlir
    %incidence = md_exec.build_incidence %members size(%n)
        : memref<?x3xi32> -> memref<?x?xi32>
    ```

    The buffer of the members is on the host. The result is where its type
    says: on the host, or on a device.
  }];
  let arguments = (ins
    MDExec_RelationOrMembers:$relation,
    Optional<Index>:$size
  );
  let results = (outs MDExec_Incidence:$result);
  let assemblyFormat = [{
    $relation (`size` `(` $size^ `)`)? attr-dict
    `:` qualified(type($relation)) `->` qualified(type($result))
  }];
  let extraClassDeclaration = [{
    bool isStorageForm() {
      return ::llvm::isa<::mlir::MemRefType>(getRelation().getType());
    }
  }];
  let hasVerifier = 1;
}

def MDExec_RenumberOp : MDExec_Op<"renumber", [
    AllTypesMatch<["members", "result"]>,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let summary = "The members of tuples in the current order of the particles";
  let description = [{
    The members of the tuples of a topology are places of particles. Where
    the particles are put in another order, their places change. This op
    takes the members as they are in the order of the files, and the
    numbers of the particles in the current order, `ids`, which say for
    each place the particle of the files that is there, and returns the
    members at their current places.

    ```text
    member' = the place p with ids[p] = member
    ```

    ```mlir
    %angles1 = md_exec.renumber %angles, %ids
        : !md.relation<@atoms, 3, reversal, @angles>, !md.field<@atoms, i32>
    ```

    In the storage form the op takes buffers and returns a new buffer of
    the host, which lasts to the end of the block that it is in.
  }];
  let arguments = (ins
    MDExec_RelationOrMembers:$members,
    MDExec_FieldOrBuffer:$ids
  );
  let results = (outs MDExec_RelationOrMembers:$result);
  let assemblyFormat = [{
    $members `,` $ids attr-dict `:` qualified(type($members)) `,`
    qualified(type($ids))
  }];
  let extraClassDeclaration = [{
    bool isStorageForm() {
      return ::llvm::isa<::mlir::MemRefType>(getMembers().getType());
    }
  }];
  let hasVerifier = 1;
}

def MDExec_TupleForOp : MDExec_Op<"tuple_for", [
    AttrSizedOperandSegments, RecursiveMemoryEffects, SingleBlock,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let summary = "Runs a kernel over the tuples of each particle";
  let description = [{
    For every particle `i` and every tuple `t` that it is a member of, at
    the place `s`:

    ```text
    a_i = a0_i + Σ_t k_a(t)[s]          for every field in `outs`
    S   = S0   + Σ_{t : s = 0} k_S(t)   for every value in `reduce`
    ```

    A tuple of `k` members is evaluated `k` times, once by each of its
    members, and every member takes its own part. A sum takes the
    contribution of a tuple from the member at place 0, so that a tuple
    counts once.

    The loop computes the displacements that `coordinates` names, each in
    the minimum image, from the positions and the cell. The kernel receives
    them, then one value for each member for each field in `ins`, then one
    value for each field in `tuple`. It yields one value for each member
    for each field in `outs`, then one contribution for each value in
    `reduce`.

    ```mlir
    %f, %u = md_exec.tuple_for %incidence, %x, %cell
               coordinates(displacement(0, 1), displacement(2, 1))
               tuple(%k : !md.field<@angles, f64>)
               outs(%f0 : !md.field<@atoms, 3 x f64>) reduce(%u0 : f64)
               arity(3) {
    ^bb0(%u: vector<3xf64>, %v: vector<3xf64>, %k_t: f64):
      ...
      md_exec.yield %f_0, %f_1, %f_2, %e
          : vector<3xf64>, vector<3xf64>, vector<3xf64>, f64
    } : !mdrt.incidence<@atoms, @angles, 3>, !md.field<@atoms, 3 x f64>
        -> !md.field<@atoms, 3 x f64>, f64
    ```

    The storage form is as that of `md_exec.pair_for`, with `overwrite` and
    `scratch`.

    `disjoint` states that no particle is a member of two tuples, as the
    tuple set declares (`md.tuple_set`). The results are the same; a
    lowering may evaluate each tuple once, by its member at place 0, and
    write to all of its members.
  }];
  let arguments = (ins
    MDExec_Incidence:$incidence,
    MDExec_FieldOrBuffer:$positions,
    MD_CellType:$cell,
    Variadic<MDExec_FieldOrBuffer>:$ins,
    Variadic<MDExec_FieldOrBuffer>:$parameters,
    Variadic<MDExec_FieldOrBuffer>:$outs,
    Variadic<AnyType>:$reduce,
    Variadic<MDExec_Scratch>:$scratch,
    DenseI32ArrayAttr:$coordinate_kinds,
    DenseI64ArrayAttr:$coordinate_members,
    I64Attr:$arity,
    OptionalAttr<DenseBoolArrayAttr>:$overwrite,
    UnitAttr:$disjoint
  );
  let regions = (region SizedRegion<1>:$kernel);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = [{
    $incidence `,` $positions `,` $cell
    custom<Coordinates>($coordinate_kinds, $coordinate_members)
    (`ins` `(` $ins^ `:` qualified(type($ins)) `)`)?
    (`tuple` `(` $parameters^ `:` qualified(type($parameters)) `)`)?
    (`outs` `(` $outs^ `:` qualified(type($outs)) `)`)?
    (`reduce` `(` $reduce^ `:` type($reduce) `)`)?
    (`scratch` `(` $scratch^ `:` type($scratch) `)`)?
    `arity` `(` $arity `)`
    (`overwrite` $overwrite^)?
    (`disjoint` $disjoint^)?
    $kernel attr-dict
    `:` qualified(type($incidence)) `,` qualified(type($positions))
    (`->` qualified(type($results))^)?
  }];
  let extraClassDeclaration = [{
    bool isStorageForm() {
      return ::llvm::isa<::mlir::MemRefType>(getPositions().getType());
    }
    /// Returns true if the loop ignores what destination `index` holds.
    bool overwrites(unsigned index) {
      auto flags = getOverwrite();
      return flags && (*flags)[index];
    }
    /// The displacements that the kernel takes.
    ::llvm::SmallVector<::mdir::md::Coordinate, 2> getCoordinates() {
      return ::mdir::md::getCoordinates(getCoordinateKinds(),
                                        getCoordinateMembers());
    }
  }];
  let hasVerifier = 1;
  let hasRegionVerifier = 1;
}

#endif // MDIR_DIALECT_MDEXEC_MDEXECOPS_TD
