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

def MDExec_SpatialOrderOp : MDExec_Op<"spatial_order", [Pure]> {
  let summary = "Computes the order of the particles by cell";
  let description = [{
    ```mlir
    %order = md_exec.spatial_order %cells
               : !mdrt.cells<@atoms> -> !mdrt.permutation<@atoms>
    ```
  }];
  let arguments = (ins MDRT_CellsType:$cells);
  let results = (outs MDRT_PermutationType:$result);
  let assemblyFormat = [{
    $cells attr-dict
    `:` qualified(type($cells)) `->` qualified(type($result))
  }];
  let hasVerifier = 1;
}

def MDExec_PermuteOp : MDExec_Op<"permute", [
    Pure, AllTypesMatch<["field", "result"]>
  ]> {
  let summary = "Applies an order to a field";
  let description = [{
    ```mlir
    %xs = md_exec.permute %x, %order
            : !md.field<@atoms, 3 x f64>, !mdrt.permutation<@atoms>
    ```
  }];
  let arguments = (ins
    MD_FieldType:$field,
    MDRT_PermutationType:$order
  );
  let results = (outs MD_FieldType:$result);
  let assemblyFormat = [{
    $field `,` $order attr-dict
    `:` qualified(type($field)) `,` qualified(type($order))
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
    `size` particles whose positions have the type `element`.

    ```mlir
    %nl = md_exec.empty_neighbors size(%n) element(f32)
            kind(matrix) width(96) : !mdrt.neighbors<@atoms>
    ```
  }];
  let arguments = (ins
    Optional<Index>:$size,
    OptionalAttr<TypeAttr>:$element,
    MDExec_NeighborKind:$kind,
    I64Attr:$width
  );
  let results = (outs MDRT_NeighborsType:$result);
  let assemblyFormat = [{
    (`size` `(` $size^ `)` `element` `(` $element `)`)?
    `kind` `(` $kind `)` `width` `(` $width `)` attr-dict
    `:` qualified(type($result))
  }];
  let extraClassDeclaration = [{
    bool isStorageForm() { return static_cast<bool>(getSize()); }
  }];
  let hasVerifier = 1;
}

def MDExec_RefreshNeighborsOp : MDExec_Op<"refresh_neighbors", [
    AllTypesMatch<["neighbors", "result"]>,
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

    `cell_width` is the width of the cells that a rebuild bins the particles
    into.

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
  }];
  let arguments = (ins
    MDRT_NeighborsType:$neighbors,
    MDExec_FieldOrBuffer:$positions,
    MD_CellType:$cell,
    F64Attr:$cutoff,
    F64Attr:$skin,
    F64Attr:$cell_width,
    DefaultValuedAttr<MDExec_RebuildPolicy,
                      "::mdir::md_exec::RebuildPolicy::Check">:$policy
  );
  let results = (outs MDRT_NeighborsType:$result);
  let assemblyFormat = [{
    $neighbors `,` $positions `,` $cell
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
  }];
  let arguments = (ins
    MDRT_NeighborsType:$neighbors,
    MDExec_FieldOrBuffer:$positions,
    MD_CellType:$cell,
    Variadic<MDExec_FieldOrBuffer>:$ins,
    Variadic<MDExec_FieldOrBuffer>:$outs,
    Variadic<AnyType>:$reduce,
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
    S   = S0 + Σ_i k_S(i)        for every value in `reduce`
    ```

    A field in `outs` is written, not accumulated into. Its old values are
    not read.

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
  }];
  let arguments = (ins
    Variadic<MDExec_FieldOrBuffer>:$ins,
    Variadic<MDExec_FieldOrBuffer>:$outs,
    Variadic<AnyType>:$reduce
  );
  let regions = (region SizedRegion<1>:$kernel);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = [{
    (`ins` `(` $ins^ `:` qualified(type($ins)) `)`)?
    (`outs` `(` $outs^ `:` qualified(type($outs)) `)`)?
    (`reduce` `(` $reduce^ `:` type($reduce) `)`)?
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
