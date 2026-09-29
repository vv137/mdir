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
    has one component or three.

    ```mlir
    !md.field<@atoms, f64>
    !md.field<@atoms, 3 x f64>
    !md.field<@atoms, i32>
    ```

    At the semantic level `f64` is the only floating-point element type. It
    is the reference precision; storage precision is assigned during
    lowering.
  }];
  let parameters = (ins
    "::mlir::FlatSymbolRefAttr":$particleSet,
    "unsigned":$numComponents,
    "::mlir::Type":$elementType
  );
  let assemblyFormat = [{
    `<` $particleSet `,` custom<FieldElement>($numComponents, $elementType) `>`
  }];
  let genVerifyDecl = 1;
  let extraClassDeclaration = [{
    /// The type of one value of this field inside a kernel: the element type
    /// for a one-component field, a fixed-size vector otherwise.
    ::mlir::Type getKernelValueType() const;
  }];
}

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
  }];
  let parameters = (ins
    "::mlir::FlatSymbolRefAttr":$particleSet,
    "unsigned":$arity,
    EnumParameter<MD_Orientation>:$orientation
  );
  let assemblyFormat = [{
    `<` $particleSet `,` $arity `,` $orientation `>`
  }];
  let genVerifyDecl = 1;
}

#endif // MDIR_DIALECT_MD_MDTYPES_TD
