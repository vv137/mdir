// Enumerations of the md_dist dialect.

#ifndef MDIR_DIALECT_MDDIST_MDDISTENUMS_TD
#define MDIR_DIALECT_MDDIST_MDDISTENUMS_TD

include "mlir/IR/EnumAttr.td"

// Which rows of each domain a distributed field has.
def MDDist_Role : I32EnumAttr<"Role", "rows of a distributed field", [
    // The particles that the domain owns.
    I32EnumAttrCase<"Owned", 0, "owned">,
    // The ghosts of a halo: images of particles near the domain.
    I32EnumAttrCase<"Ghost", 1, "ghost">
  ]> {
  let cppNamespace = "::mdir::md_dist";
}

// How the pairs of a relation are spread over the domains.
def MDDist_Traversal : I32EnumAttr<"Traversal", "traversal of a relation", [
    // Every pair at the owner of each of its particles: twice in all.
    I32EnumAttrCase<"Directed", 0, "directed">,
    // Every pair once, at the owner of the particle of the lesser identity.
    I32EnumAttrCase<"Unique", 1, "unique">
  ]> {
  let cppNamespace = "::mdir::md_dist";
}

#endif // MDIR_DIALECT_MDDIST_MDDISTENUMS_TD
