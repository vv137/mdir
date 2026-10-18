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
    that their components carry (D[ml4-feature-types],
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

#endif // MDIR_DIALECT_MD_MDTYPES_TD
