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
  }];
  let arguments = (ins
    MDRT_CellsType:$cells,
    MD_FieldType:$positions,
    MD_CellType:$cell,
    F64Attr:$cutoff,
    F64Attr:$skin,
    MDExec_NeighborKind:$kind,
    I64Attr:$width
  );
  let results = (outs MDRT_NeighborsType:$result);
  let assemblyFormat = [{
    $cells `,` $positions `,` $cell
    `cutoff` `(` $cutoff `)` `skin` `(` $skin `)`
    `kind` `(` $kind `)` `width` `(` $width `)` attr-dict
    `:` qualified(type($cells)) `,` qualified(type($positions))
    `->` qualified(type($result))
  }];
  let hasVerifier = 1;
}

def MDExec_EmptyNeighborsOp : MDExec_Op<"empty_neighbors", [
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
  }];
  let arguments = (ins
    Optional<Index>:$size,
    OptionalAttr<TypeAttr>:$positions,
    MDExec_NeighborKind:$kind,
    I64Attr:$width
  );
  let results = (outs MDRT_NeighborsType:$result);
  let assemblyFormat = [{
    (`size` `(` $size^ `)` `positions` `(` $positions `)`)?
    `kind` `(` $kind `)` `width` `(` $width `)` attr-dict
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

    A structure that was built at the configuration `x_ref` is valid when
    the cell is the one it was built in and

    ```text
    max_i |x_i − x_ref,i| ≤ skin / 2
    ```

    Under that condition it contains every pair within the cutoff. With the
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

    `scratch` holds buffers that the op may use as it likes. What they hold
    before and after is unspecified. On a device the test of validity needs
    two, with one value of the type of the positions per particle.

    With `moved`, the op does not test the displacements: `moved` tells
    whether a particle has moved more than half the skin since the
    structure was built. A loop over particles computes it from the
    positions and from `md_exec.reference_positions`. The op then needs
    nothing in `scratch`.

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
                      "::mdir::md_exec::RebuildPolicy::Check">:$policy
  );
  let results = (outs MDRT_NeighborsType:$result);
  let assemblyFormat = [{
    $neighbors `,` $positions `,` $cell
    (`scratch` `(` $scratch^ `:` type($scratch) `)`)?
    (`moved` `(` $moved^ `)`)?
    `cutoff` `(` $cutoff `)` `skin` `(` $skin `)`
    `cell_width` `(` $cell_width `)` `policy` `(` $policy `)` attr-dict
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
  }];
  let arguments = (ins
    Arg<MDRT_NeighborsType, "the structure", [MemWrite]>:$neighbors
  );
  let assemblyFormat = [{
    $neighbors attr-dict `:` qualified(type($neighbors))
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
// Loops
//===----------------------------------------------------------------------===//

def MDExec_YieldOp : MDExec_Op<"yield", [
    Pure, Terminator, ReturnLike,
    ParentOneOf<["PairForOp", "ParticleForOp"]>
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
                      "::mdir::md_exec::Conflict::OwnerOnly">:$conflict
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

#endif // MDIR_DIALECT_MDEXEC_MDEXECOPS_TD
