// Enumerations of the md_exec dialect.

#ifndef MDIR_DIALECT_MDEXEC_MDEXECENUMS_TD
#define MDIR_DIALECT_MDEXEC_MDEXECENUMS_TD

include "mlir/IR/EnumAttr.td"

// The physical form of a neighbor structure.
def MDExec_NeighborKind : I32EnumAttr<"NeighborKind",
    "physical form of a neighbor structure", [
    // A row of fixed width per particle.
    I32EnumAttrCase<"Matrix", 0, "matrix">
  ]> {
  let cppNamespace = "::mdir::md_exec";
}

// Which pairs a loop visits.
def MDExec_Traversal : I32EnumAttr<"Traversal", "pairs that a loop visits", [
    // Both (i, j) and (j, i).
    I32EnumAttrCase<"Directed", 0, "directed">,
    // Each pair once.
    I32EnumAttrCase<"Unique", 1, "unique">
  ]> {
  let cppNamespace = "::mdir::md_exec";
}

// How a loop avoids conflicting writes.
def MDExec_Conflict : I32EnumAttr<"Conflict",
    "strategy against conflicting writes", [
    // A kernel writes only to its central particle.
    I32EnumAttrCase<"OwnerOnly", 0, "owner_only">,
    I32EnumAttrCase<"Atomic", 1, "atomic">,
    I32EnumAttrCase<"Privatized", 2, "privatized">,
    I32EnumAttrCase<"Colored", 3, "colored">
  ]> {
  let cppNamespace = "::mdir::md_exec";
}

// When a neighbor structure is rebuilt.
def MDExec_RebuildPolicy : I32EnumAttr<"RebuildPolicy",
    "when a neighbor structure is rebuilt", [
    // Whenever the validity condition fails. The condition is evaluated at
    // every refresh.
    I32EnumAttrCase<"Check", 0, "check">
  ]> {
  let cppNamespace = "::mdir::md_exec";
}

#endif // MDIR_DIALECT_MDEXEC_MDEXECENUMS_TD
