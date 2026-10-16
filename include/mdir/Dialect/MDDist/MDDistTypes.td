// Types of the md_dist dialect.

#ifndef MDIR_DIALECT_MDDIST_MDDISTTYPES_TD
#define MDIR_DIALECT_MDDIST_MDDISTTYPES_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mdir/Dialect/MDDist/MDDistDialect.td"
include "mdir/Dialect/MDDist/MDDistEnums.td"

class MDDist_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<MDDist_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def MDDist_LayoutType : MDDist_Type<"Layout", "layout"> {
  let summary = "The ownership of the particles by the domains";
  let description = [{
    For each particle of the set of the decomposition, the domain that owns
    it, its row among the owned rows of that domain, and its global
    identity, its place in the particle set. A layout never changes; a
    migration gives a new layout, so that a layout value is an epoch of
    ownership.

    ```mlir
    !md_dist.layout<@dd>
    ```
  }];
  let parameters = (ins "::mlir::FlatSymbolRefAttr":$decomposition);
  let assemblyFormat = "`<` $decomposition `>`";
}

def MDDist_HaloType : MDDist_Type<"Halo", "halo"> {
  let summary = "The ghosts of each domain";
  let description = [{
    For each domain, its ghosts: every image of every particle, other than
    the domain's own particles in the cell, within `reach` of the domain's
    region. Each ghost records its source (owner and owned row), its global
    identity, and its image. The reach is in the type, so that coverage is
    checked where the halo is used.

    ```mlir
    !md_dist.halo<@dd, 2.0>
    ```
  }];
  let parameters = (ins
    "::mlir::FlatSymbolRefAttr":$decomposition,
    APFloatParameter<"the reach">:$reach
  );
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

def MDDist_FieldType : MDDist_Type<"Field", "field"> {
  let summary = "A field over the owned or the ghost rows of every domain";
  let description = [{
    The values of a per-particle field at the owned rows of every domain,
    or at the ghost rows of a halo.

    ```mlir
    !md_dist.field<@dd, owned, 3 x f64>
    !md_dist.field<@dd, ghost, 3 x f64>
    ```

    `partial` marks contributions to a sum that is not complete: those a
    traversal that computes each pair once leaves at owned rows and at
    ghost rows. Only `md_dist.halo_sum` consumes a partial field, and a
    partial value has exactly one use.

    ```mlir
    !md_dist.field<@dd, ghost, partial, 3 x f64>
    ```
  }];
  let parameters = (ins
    "::mlir::FlatSymbolRefAttr":$decomposition,
    EnumParameter<MDDist_Role>:$role,
    "bool":$partial,
    "unsigned":$numComponents,
    "::mlir::Type":$elementType
  );
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
  let extraClassDeclaration = [{
    bool isOwned() const { return getRole() == ::mdir::md_dist::Role::Owned; }
    bool isGhost() const { return getRole() == ::mdir::md_dist::Role::Ghost; }
    /// The type of one value inside a kernel: the element type for one
    /// component, a vector otherwise.
    ::mlir::Type getKernelValueType() const;
    /// The same field with other rows or another completeness.
    FieldType with(::mdir::md_dist::Role role, bool partial) const;
  }];
}

def MDDist_PartialType : MDDist_Type<"Partial", "partial"> {
  let summary = "One partial sum per domain";
  let description = [{
    The contribution of each domain to a sum over all of them: the energy
    of the pairs a domain holds, for one. Only `md_dist.all_reduce`
    consumes it, and a partial value has exactly one use.

    ```mlir
    !md_dist.partial<f64>
    !md_dist.partial<vector<9xf64>>
    ```
  }];
  let parameters = (ins "::mlir::Type":$valueType);
  let assemblyFormat = "`<` $valueType `>`";
}

def MDDist_RelationType : MDDist_Type<"Relation", "relation"> {
  let summary = "The pairs within a cutoff, over the rows of each domain";
  let description = [{
    The relation of `md.neighborhood`, realized over the owned and ghost
    rows of each domain: with `directed`, every pair at the owner of each
    of its two particles; with `unique`, every pair once.

    ```mlir
    !md_dist.relation<@dd, directed>
    ```
  }];
  let parameters = (ins
    "::mlir::FlatSymbolRefAttr":$decomposition,
    EnumParameter<MDDist_Traversal>:$traversal
  );
  let assemblyFormat = "`<` $decomposition `,` $traversal `>`";
}

#endif // MDIR_DIALECT_MDDIST_MDDISTTYPES_TD
