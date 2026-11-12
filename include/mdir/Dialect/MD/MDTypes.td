// Types of the md dialect.

#ifndef MDIR_DIALECT_MD_MDTYPES_TD
#define MDIR_DIALECT_MD_MDTYPES_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mdir/Dialect/MD/MDDialect.td"
include "mdir/Dialect/MD/MDEnums.td"

class MD_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<MD_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def MD_FieldType : MD_Type<"Field", "field"> {
  let summary = "A per-particle field";
  let description = [{
    A field assigns a value to every particle of a particle set. The value
    has one component, or several of a floating-point type: three for a
    vector, any number for the features of a learned potential.

    ```mlir
    !md.field<@atoms, f64>
    !md.field<@atoms, 3 x f64>
    !md.field<@atoms, 16 x f64>
    !md.field<@atoms, i32>
    ```

    The features may state the representation of the rotation group O(3)
    that their components carry (D289,
    docs/ml4-ir.md, Section 3): a sum of blocks `M x l p`, M copies of the
    irreducible representation of degree l (2l + 1 components) and parity
    `e` or `o`, in e3nn's notation [Geiger2022]. The components of a
    particle are the blocks in order; within a block, `mul_ir` (the
    default) places the 2l + 1 components of each copy together, and
    `ir_mul` the M copies of each component together. Within a copy the
    components are those of the real spherical harmonics of e3nn 0.4, in
    its order: for l = 1, (x, y, z).

    ```mlir
    !md.field<@atoms, irreps<4x0e + 4x1o>, f64>           // 16 components
    !md.field<@atoms, irreps<128x0e + 128x1o, ir_mul>, f64>
    ```

    At the semantic level `f64` is the only floating-point element type. It
    is the reference precision. The element type `f32` appears at the
    execution level, where the precision policy assigns it.

    ```mlir
    !md.field<@atoms, 3 x f32>
    ```
  }];
  let parameters = (ins
    "::mlir::FlatSymbolRefAttr":$particleSet,
    "unsigned":$numComponents,
    "::mlir::Type":$elementType,
    // The blocks of the irreducible representations, three numbers each:
    // the multiplicity M, the degree l, and the parity (0 even, 1 odd).
    OptionalArrayRefParameter<"int64_t">:$irreps,
    // ir_mul: the copies of each component together.
    DefaultValuedParameter<"bool", "false">:$componentMajor
  );
  let builders = [
    TypeBuilder<(ins "::mlir::FlatSymbolRefAttr":$particleSet,
                     "unsigned":$numComponents,
                     "::mlir::Type":$elementType), [{
      return $_get($_ctxt, particleSet, numComponents, elementType,
                   ::llvm::ArrayRef<int64_t>(), false);
    }]>
  ];
  let assemblyFormat = [{
    `<` $particleSet `,` custom<FieldElement>($numComponents, $elementType,
                                              $irreps, $componentMajor) `>`
  }];
  let genVerifyDecl = 1;
  let extraClassDeclaration = [{
    /// The type of one value of this field inside a kernel: the element type
    /// for a one-component field, a fixed-size vector otherwise.
    ::mlir::Type getKernelValueType() const;
    /// This type with the element type `element`, its components and their
    /// representation kept.
    FieldType withElementType(::mlir::Type element) const;
  }];
}

def MD_ReferenceFieldType : Type<
    And<[MD_FieldType.predicate,
         CPred<"!::llvm::cast<::mdir::md::FieldType>($_self)"
               ".getElementType().isF32()">]>,
    "field at the reference precision, with elements of type f64, i32, or "
    "i64",
    "::mdir::md::FieldType">;

def MD_CellType : MD_Type<"Cell", "cell"> {
  let summary = "A simulation cell";
  let description = [{
    Three lattice vectors and a periodicity flag per direction.
  }];
}

def MD_RelationType : MD_Type<"Relation", "relation"> {
  let summary = "A set of tuples of distinct particles";
  let description = [{
    A relation is a set: a tuple appears at most once. The type describes the
    logical relation only. How the relation is traversed is an execution
    decision and is never part of the type.

    ```mlir
    !md.relation<@atoms, 2, unordered>
    ```

    The relation of a tuple set names the set. Its tuples are entities with
    fields of their own, which are fields of that set.

    ```mlir
    !md.relation<@atoms, 3, reversal, @angles>
    ```
  }];
  let parameters = (ins
    "::mlir::FlatSymbolRefAttr":$particleSet,
    "unsigned":$arity,
    EnumParameter<MD_Orientation>:$orientation,
    OptionalParameter<"::mlir::FlatSymbolRefAttr">:$tupleSet
  );
  let assemblyFormat = [{
    `<` $particleSet `,` $arity `,` $orientation (`,` $tupleSet^)? `>`
  }];
  let genVerifyDecl = 1;
}

def MD_GridStencilType : MD_Type<"GridStencil", "grid_stencil"> {
  let summary = "A structural particle-grid stencil relation";
  let description = [{
    A relation from a named particle set to a static three-dimensional
    periodic grid. The window and support order are structural. The first
    window is bspline, with order 2 through 8 and each grid extent at least
    the order, so a particle's wrapped support indices are distinct.

    The relation carries indices only. Positions, cell, local coordinates,
    and window values remain explicit operands and arithmetic of consumers;
    an index relation never proves their derivatives inactive. Part of #452.
  }];
  let parameters = (ins
    "::mlir::FlatSymbolRefAttr":$particleSet,
    "::mlir::DenseI64ArrayAttr":$shape,
    "unsigned":$order,
    "::mlir::StringAttr":$window
  );
  let assemblyFormat = [{
    `<` $particleSet `,` $shape `,` `order` `(` $order `)`
    `,` `window` `(` $window `)` `>`
  }];
  let genVerifyDecl = 1;
}

def MD_EdgesType : MD_Type<"Edges", "edges"> {
  let summary = "The directed pairs of images within a cutoff";
  let description = [{
    The pairs that a local learned potential takes (D277,
    docs/metatomic-ml1.md, Section 12): for each particle `i`, every other
    particle `j` and every image of it, and every image of `i` itself,
    within the cutoff,

    ```text
    E = { (i, j, s) : (j, s) ≠ (i, 0), |x_j − x_i + s H| < r_c },
    ```

    `s` the shift in cells (integers) and `H` the cell, its rows the edges.
    Each pair is in it in both directions. Unlike a relation, the members
    need not be distinct and a pair may be in it several times, once for
    each image: the cell may be narrower than the cutoff. The pairs are
    grouped by `i`, in the order of the particles.

    ```mlir
    !md.edges<@atoms>
    ```
  }];
  let parameters = (ins "::mlir::FlatSymbolRefAttr":$particleSet);
  let assemblyFormat = "`<` $particleSet `>`";
}

def MD_TableType : MD_Type<"Table", "table"> {
  let summary = "A table of parameters, indexed by types";
  let description = [{
    A table of rank 1 has a value for each type of particle, and a table of
    rank 2 one for each pair of types: σ and ε of Lennard-Jones, for
    example, with the pairs that a force field sets apart from its mixing
    rule. The number of types is a run-time value.

    ```mlir
    !md.table<2, f64, symmetric>
    !md.table<1, f64>
    ```

    `symmetric` states that a table of rank 2 has the same value for
    `(a, b)` and `(b, a)`, which the proof of exchange uses.
  }];
  let parameters = (ins
    "unsigned":$rank,
    "::mlir::Type":$elementType,
    "bool":$symmetric
  );
  let assemblyFormat = [{
    `<` $rank `,` $elementType ``custom<Symmetry>($symmetric) `>`
  }];
  let genVerifyDecl = 1;
}


//===----------------------------------------------------------------------===//
// Attributes
//===----------------------------------------------------------------------===//

def MD_RigidKindAttr : AttrDef<MD_Dialect, "RigidKind"> {
  let mnemonic = "rigid_kind";
  let summary = "A rigid kind of D403 as the pair segments take it";
  let description = [{
    One kind of rigid group (D403) that `md.kind_pairs` takes. Its
    particles are numbered by their place in a molecule, from 0; the size
    of the kind is the arity of its relation.

    ```mlir
    #md.rigid_kind<frame = [0, 1, 2], sites = [3],
                   weights = [0.70455878105857583, 0.14772060947071208,
                              0.14772060947071208],
                   extent = 0.087343313000000005>
    ```

    `frame` holds the places of the massive atoms, ascending; the first is
    the anchor. `sites` holds the places of the massless sites; with
    `frame` it holds every place once. `weights` holds, for each site in
    the order of `sites`, its weights over the atoms of `frame` in frame
    order (rule R2 of D403: they sum to 1). `extent` is the radius about
    the anchor that the consumer checks to cover every place of a copy.
    Charges and types are not part of the kind here: the segment bodies
    hold them as constants (D[kind-pairs]).
  }];
  let parameters = (ins "::mlir::DenseI64ArrayAttr":$frame,
                        "::mlir::DenseI64ArrayAttr":$sites,
                        "::mlir::DenseF64ArrayAttr":$weights,
                        "::mlir::FloatAttr":$extent);
  // Weights and the extent are printed so that they read back as the same
  // f64 numbers.
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
  let extraClassDeclaration = [{
    /// The particles of one copy: frame atoms and sites.
    unsigned getSize() const { return getFrame().size() + getSites().size(); }
    /// The coefficients of a place over the frame offsets x_k - x_anchor,
    /// k = 1, ..., m - 1: zero for the anchor, a unit vector for a frame
    /// atom, and the weights w_1, ..., w_{m-1} for a site.
    ::llvm::SmallVector<double> getPlaceCoefficients(unsigned place) const;
    /// The index in `frame` of a frame atom's place, or none for a site.
    std::optional<unsigned> getFrameIndex(unsigned place) const;
    /// The weights over the frame of the site at a place, or empty.
    ::llvm::ArrayRef<double> getSiteWeights(unsigned place) const;
  }];
}

#endif // MDIR_DIALECT_MD_MDTYPES_TD
