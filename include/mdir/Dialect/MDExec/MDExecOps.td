// Ops of the md_exec dialect, in the value form.
//
// See docs/ops-m0.md, Section 8, for the specification.

#ifndef MDIR_DIALECT_MDEXEC_MDEXECOPS_TD
#define MDIR_DIALECT_MDEXEC_MDEXECOPS_TD

include "mlir/IR/BuiltinAttributes.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mdir/Dialect/MD/MDTypes.td"
include "mdir/Dialect/MDExec/MDExecDialect.td"
include "mdir/Dialect/MDExec/MDExecEnums.td"
include "mdir/Dialect/MDRT/MDRTTypes.td"

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
    AttrSizedOperandSegments, RecursiveMemoryEffects, SingleBlock
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
  }];
  let arguments = (ins
    MDRT_NeighborsType:$neighbors,
    MD_FieldType:$positions,
    MD_CellType:$cell,
    Variadic<MD_FieldType>:$ins,
    Variadic<MD_FieldType>:$outs,
    Variadic<AnyType>:$reduce,
    F64Attr:$cutoff,
    OptionalAttr<DenseF64ArrayAttr>:$weights,
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
    `policy` `(` $traversal `,` $conflict `)`
    $kernel attr-dict
    `:` qualified(type($neighbors)) `,` qualified(type($positions))
    (`->` qualified(type($results))^)?
  }];
  let hasVerifier = 1;
  let hasRegionVerifier = 1;
}

def MDExec_ParticleForOp : MDExec_Op<"particle_for", [
    AttrSizedOperandSegments, RecursiveMemoryEffects, SingleBlock
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
  }];
  let arguments = (ins
    Variadic<MD_FieldType>:$ins,
    Variadic<MD_FieldType>:$outs,
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
  let hasVerifier = 1;
  let hasRegionVerifier = 1;
}

#endif // MDIR_DIALECT_MDEXEC_MDEXECOPS_TD
