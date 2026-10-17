// Ops of the md_dist dialect.
//
// See docs/distributed-ml2.md, Section 4, for the design (D275).

#ifndef MDIR_DIALECT_MDDIST_MDDISTOPS_TD
#define MDIR_DIALECT_MDDIST_MDDISTOPS_TD

include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mdir/Dialect/MD/MDTypes.td"
include "mdir/Dialect/MDDist/MDDistTypes.td"

//===----------------------------------------------------------------------===//
// Decomposition and ownership
//===----------------------------------------------------------------------===//

def MDDist_DecompositionOp : MDDist_Op<"decomposition", [Symbol]> {
  let summary = "Declares a Cartesian grid of domains over a particle set";
  let description = [{
    The domains are the boxes of a grid of `grid` divisions of each axis of
    the cell, in its fractional coordinates: domain `(a, b, c)` is
    `[a/ga, (a+1)/ga) x [b/gb, (b+1)/gb) x [c/gc, (c+1)/gc)`, numbered
    `a + ga * (b + gb * c)`.

    ```mlir
    md_dist.decomposition @dd on(@atoms) grid([2, 1, 1])
    ```
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    FlatSymbolRefAttr:$particle_set,
    DenseI64ArrayAttr:$grid
  );
  let assemblyFormat = [{
    $sym_name `on` `(` $particle_set `)` `grid` `(` $grid `)` attr-dict
  }];
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    /// The number of domains.
    int64_t getNumDomains();
  }];
}

def MDDist_PartitionOp : MDDist_Op<"partition", [
    Pure, DeclareOpInterfaceMethods<SymbolUserOpInterface>
  ]> {
  let summary = "The layout that gives each particle to the domain it is in";
  let description = [{
    Each particle is owned by the domain whose region holds its position,
    wrapped into the cell. Within a domain the owned rows are in the order
    of the particle set.

    ```mlir
    %l = md_dist.partition %x, %cell
           : !md.field<@atoms, 3 x f64> -> !md_dist.layout<@dd>
    ```
  }];
  let arguments = (ins MD_ReferenceFieldType:$positions, MD_CellType:$cell);
  let results = (outs MDDist_LayoutType:$layout);
  let assemblyFormat = [{
    $positions `,` $cell attr-dict `:` qualified(type($positions)) `->`
    qualified(type($layout))
  }];
}

// The layout operand of an op whose type the assembly does not print: of
// the decomposition of `source`.
class MDDist_LayoutOf<string source> : TypesMatchWith<
    "the layout is of the decomposition of " # source, source, "layout",
    "::mdir::md_dist::LayoutType::get($_self.getContext(), "
    "::llvm::cast<::mdir::md_dist::FieldType>($_self).getDecomposition())">;

def MDDist_DistributeOp : MDDist_Op<"distribute", [
    Pure, MDDist_LayoutOf<"result">
  ]> {
  let summary = "The values of a field at the owned rows of each domain";
  let description = [{
    With `image`, the values are positions, and each is wrapped into the
    cell, so that it lies in the region of its domain.

    ```mlir
    %xo = md_dist.distribute %x, %l image(%cell)
            : !md.field<@atoms, 3 x f64> -> !md_dist.field<@dd, owned, 3 x f64>
    ```
  }];
  let arguments = (ins
    MD_ReferenceFieldType:$field,
    MDDist_LayoutType:$layout,
    Optional<MD_CellType>:$image
  );
  let results = (outs MDDist_FieldType:$result);
  let assemblyFormat = [{
    $field `,` $layout (`image` `(` $image^ `)`)? attr-dict
    `:` qualified(type($field)) `->` qualified(type($result))
  }];
  let hasVerifier = 1;
}

def MDDist_CollectOp : MDDist_Op<"collect", [
    Pure, MDDist_LayoutOf<"field">
  ]> {
  let summary = "A field of the particle set from its owned rows";
  let description = [{
    The inverse of `md_dist.distribute`: the values in the order of the
    particle set, whatever the number of domains. The field must be
    complete and of the layout.

    ```mlir
    %f = md_dist.collect %fo, %l
           : !md_dist.field<@dd, owned, 3 x f64> -> !md.field<@atoms, 3 x f64>
    ```
  }];
  let arguments = (ins MDDist_FieldType:$field, MDDist_LayoutType:$layout);
  let results = (outs MD_ReferenceFieldType:$result);
  let assemblyFormat = [{
    $field `,` $layout attr-dict `:` qualified(type($field)) `->`
    qualified(type($result))
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Halos and transfers
//===----------------------------------------------------------------------===//

def MDDist_HaloOp : MDDist_Op<"halo", [
    Pure, TypesMatchWith<"the layout is of the decomposition of the halo",
                         "result", "layout",
                         "::mdir::md_dist::LayoutType::get($_self.getContext(), "
                         "::llvm::cast<::mdir::md_dist::HaloType>($_self)"
                         ".getDecomposition())">
  ]> {
  let summary = "The ghosts of each domain at a configuration";
  let description = [{
    For each domain, every image `x_j + n H` of every particle, other than
    the domain's own particles in the cell (`n = 0`), whose distance from
    the region of the domain is less than the reach of the result type.
    `positions` are the owned positions of `layout`.

    ```mlir
    %h = md_dist.halo %l, %xo, %cell
           : !md_dist.field<@dd, owned, 3 x f64> -> !md_dist.halo<@dd, 2.0>
    ```
  }];
  let arguments = (ins
    MDDist_LayoutType:$layout,
    MDDist_FieldType:$positions,
    MD_CellType:$cell
  );
  let results = (outs MDDist_HaloType:$result);
  let assemblyFormat = [{
    $layout `,` $positions `,` $cell attr-dict `:` qualified(type($positions))
    `->` qualified(type($result))
  }];
  let hasVerifier = 1;
}

def MDDist_HaloCopyOp : MDDist_Op<"halo_copy", [Pure]> {
  let summary = "The owner-to-ghost copy G";
  let description = [{
    ```text
    (G a)_g = a_{s(g)}
    ```

    Each ghost takes the value of its source. With `image`, the values are
    positions and each ghost takes its image, `x_{s(g)} + n_g H`.

    ```mlir
    %xg = md_dist.halo_copy %xo, %h image(%cell)
            : !md_dist.field<@dd, owned, 3 x f64>, !md_dist.halo<@dd, 2.0>
            -> !md_dist.field<@dd, ghost, 3 x f64>
    ```

    The adjoint of the op is `md_dist.halo_sum` (G^T).
  }];
  let arguments = (ins
    MDDist_FieldType:$field,
    MDDist_HaloType:$halo,
    Optional<MD_CellType>:$image
  );
  let results = (outs MDDist_FieldType:$result);
  let assemblyFormat = [{
    $field `,` $halo (`image` `(` $image^ `)`)? attr-dict
    `:` qualified(type($field)) `,` qualified(type($halo)) `->`
    qualified(type($result))
  }];
  let hasVerifier = 1;
}

def MDDist_HaloSumOp : MDDist_Op<"halo_sum", [Pure]> {
  let summary = "An owned value plus the contributions of its ghosts, G^T";
  let description = [{
    ```text
    r = a + G^T b,    (G^T b)_i = Σ_{g : s(g) = i} b_g
    ```

    The values that the ghosts of every domain hold are summed at the
    owners of their sources and added to `owned`. The result is complete.
    G^T alone is the op with `owned` of `md_dist.zeros`.

    The two operands are complete, or the two partial results of one
    traversal: then the result is the traversal's sum, each contribution
    counted once.

    ```mlir
    %f = md_dist.halo_sum %fo, %fg, %h
           : !md_dist.field<@dd, owned, partial, 3 x f64>,
             !md_dist.field<@dd, ghost, partial, 3 x f64>,
             !md_dist.halo<@dd, 2.0>
           -> !md_dist.field<@dd, owned, 3 x f64>
    ```

    The adjoint of the op is `md_dist.halo_copy` for `ghosts` and the
    identity for `owned`.
  }];
  let arguments = (ins
    MDDist_FieldType:$owned,
    MDDist_FieldType:$ghosts,
    MDDist_HaloType:$halo
  );
  let results = (outs MDDist_FieldType:$result);
  let assemblyFormat = [{
    $owned `,` $ghosts `,` $halo attr-dict
    `:` qualified(type($owned)) `,` qualified(type($ghosts)) `,`
    qualified(type($halo)) `->` qualified(type($result))
  }];
  let hasVerifier = 1;
}

def MDDist_ZerosOp : MDDist_Op<"zeros", [Pure]> {
  let summary = "A field of zeros over the owned rows or the ghosts";
  let description = [{
    Over the owned rows of a layout, or over the ghosts of a halo.

    ```mlir
    %z = md_dist.zeros %l : !md_dist.layout<@dd>
           -> !md_dist.field<@dd, owned, 3 x f64>
    ```
  }];
  let arguments = (ins AnyTypeOf<[MDDist_LayoutType, MDDist_HaloType]>:$rows);
  let results = (outs MDDist_FieldType:$result);
  let assemblyFormat = [{
    $rows attr-dict `:` qualified(type($rows)) `->` qualified(type($result))
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Pairs
//===----------------------------------------------------------------------===//

def MDDist_NeighborhoodOp : MDDist_Op<"neighborhood", [Pure]> {
  let summary = "The pairs within a cutoff, over owned and ghost rows";
  let description = [{
    The relation of `md.neighborhood` over the local rows of each domain.
    `positions` are owned positions and `ghosts` their copy through `halo`,
    with images. With the traversal `directed`, a domain holds the pairs
    `(i, k)` of an owned `i` and any other local row `k` within `cutoff`;
    with `unique`, those where the identity of `i` is less than that of
    the source of `k`.

    ```mlir
    %n = md_dist.neighborhood %xo, %xg, %h cutoff(2.0)
           : !md_dist.field<@dd, owned, 3 x f64>,
             !md_dist.field<@dd, ghost, 3 x f64>, !md_dist.halo<@dd, 2.0>
           -> !md_dist.relation<@dd, directed>
    ```
  }];
  let arguments = (ins
    MDDist_FieldType:$positions,
    MDDist_FieldType:$ghosts,
    MDDist_HaloType:$halo,
    F64Attr:$cutoff
  );
  let results = (outs MDDist_RelationType:$result);
  let assemblyFormat = [{
    $positions `,` $ghosts `,` $halo `cutoff` `(` $cutoff `)` attr-dict
    `:` qualified(type($positions)) `,` qualified(type($ghosts)) `,`
    qualified(type($halo)) `->` qualified(type($result))
  }];
  let hasVerifier = 1;
}

def MDDist_YieldOp : MDDist_Op<"yield", [
    Pure, Terminator, ReturnLike,
    ParentOneOf<["SumPairsOp", "GatherPairsOp", "SumParticlesOp"]>
  ]> {
  let summary = "Yields the value of a kernel";
  let arguments = (ins Variadic<AnyType>:$values);
  let assemblyFormat = "attr-dict ($values^ `:` type($values))?";
}

class MDDist_PairOp<string mnemonic> : MDDist_Op<mnemonic, [
    RecursiveMemoryEffects, SingleBlock
  ]> {
  let arguments = (ins
    MDDist_RelationType:$relation,
    MDDist_FieldType:$positions,
    MDDist_FieldType:$ghosts,
    Variadic<MDDist_FieldType>:$gathered,
    MD_Exchange:$exchange
  );
  let regions = (region SizedRegion<1>:$kernel);
  let hasVerifier = 1;
  let hasRegionVerifier = 1;
}

def MDDist_SumPairsOp : MDDist_PairOp<"sum_pairs"> {
  let summary = "Sums a kernel over the pairs of a relation";
  let description = [{
    ```text
    S = Σ_d S_d,  S_d = ½ Σ_{(i,k) ∈ D_d} k(i, k)   (directed)
                  S_d =   Σ_{(i,k) ∈ U_d} k(i, k)   (unique)
    ```

    The sum of `md.sum_relation` over the global relation, each pair once;
    the result holds the part of each domain. The kernel receives the
    distance, the displacement `x_i − x_k`, and two values of each gathered
    field, of `i` and of `k`; it must be symmetric under exchange of the
    two, which the verifier proves.

    `gather` takes each field as its owned value and its ghost copy through
    the relation's halo.

    ```mlir
    %u = md_dist.sum_pairs %n, %xo, %xg exchange(symmetric) {
    ^bb0(%r: f64, %d: vector<3xf64>):
      ...
      md_dist.yield %k : f64
    } : !md_dist.relation<@dd, directed>, !md_dist.field<@dd, owned, 3 x f64>,
        !md_dist.field<@dd, ghost, 3 x f64> -> !md_dist.partial<f64>
    ```
  }];
  let results = (outs MDDist_PartialType:$result);
  let assemblyFormat = [{
    $relation `,` $positions `,` $ghosts
    (`gather` `(` $gathered^ `:` qualified(type($gathered)) `)`)?
    `exchange` `(` $exchange `)` $kernel attr-dict
    `:` qualified(type($relation)) `,` qualified(type($positions)) `,`
    qualified(type($ghosts)) `->` qualified(type($result))
  }];
}

def MDDist_GatherPairsOp : MDDist_PairOp<"gather_pairs"> {
  let summary = "Sums a kernel, per owned particle, over its pairs";
  let description = [{
    ```text
    a_i = Σ_{k : (i, k) ∈ R} k(i, k)
    ```

    The field of `md.gather_relation` at the owned rows. Over a `directed`
    relation, every pair of `i` is at its owner, and the one result is
    complete. Over a `unique` relation, the op yields the contributions at
    owned rows and at ghost rows, both partial, with `a = p + G^T q`: the
    kernel is evaluated once per pair, and the value at `k` is the kernel's
    value under exchange, which `exchange` states and the verifier proves.

    ```mlir
    %fo, %fg = md_dist.gather_pairs %n, %xo, %xg exchange(antisymmetric) {
      ...
    } : !md_dist.relation<@dd, unique>, !md_dist.field<@dd, owned, 3 x f64>,
        !md_dist.field<@dd, ghost, 3 x f64>
        -> !md_dist.field<@dd, owned, partial, 3 x f64>,
           !md_dist.field<@dd, ghost, partial, 3 x f64>
    ```
  }];
  let results = (outs Variadic<MDDist_FieldType>:$results);
  let assemblyFormat = [{
    $relation `,` $positions `,` $ghosts
    (`gather` `(` $gathered^ `:` qualified(type($gathered)) `)`)?
    `exchange` `(` $exchange `)` $kernel attr-dict
    `:` qualified(type($relation)) `,` qualified(type($positions)) `,`
    qualified(type($ghosts)) `->` qualified(type($results))
  }];
}

//===----------------------------------------------------------------------===//
// Particles and reductions
//===----------------------------------------------------------------------===//

def MDDist_SumParticlesOp : MDDist_Op<"sum_particles", [
    RecursiveMemoryEffects, SingleBlock
  ]> {
  let summary = "Sums a kernel over the rows of fields";
  let description = [{
    Over owned rows, the sum of `md.sum_particles`, each particle once.
    Over ghost rows, the inner product of the space of the ghosts, which
    no physical sum is, and which the test of the adjoint of `halo_copy`
    uses. The fields are of one role and of one layout or halo.

    ```mlir
    %k = md_dist.sum_particles gather(%v : !md_dist.field<@dd, owned, 3 x f64>) {
    ^bb0(%v_i: vector<3xf64>):
      ...
      md_dist.yield %e : f64
    } : !md_dist.partial<f64>
    ```
  }];
  let arguments = (ins Variadic<MDDist_FieldType>:$gathered);
  let regions = (region SizedRegion<1>:$kernel);
  let results = (outs MDDist_PartialType:$result);
  let assemblyFormat = [{
    `gather` `(` $gathered `:` qualified(type($gathered)) `)`
    $kernel attr-dict `:` qualified(type($result))
  }];
  let hasVerifier = 1;
  let hasRegionVerifier = 1;
}

def MDDist_AllReduceOp : MDDist_Op<"all_reduce", [Pure]> {
  let summary = "Sums the partial sums of the domains";
  let description = [{
    The result is one value, the same on every domain. The partial sums
    are added in the order of the domains.

    ```mlir
    %u = md_dist.all_reduce %up : !md_dist.partial<f64> -> f64
    ```
  }];
  let arguments = (ins MDDist_PartialType:$partial);
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $partial attr-dict `:` qualified(type($partial)) `->` type($result)
  }];
  let hasVerifier = 1;
}

#endif // MDIR_DIALECT_MDDIST_MDDISTOPS_TD
