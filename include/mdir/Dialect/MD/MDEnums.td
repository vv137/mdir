// Enumerations of the md dialect.

#ifndef MDIR_DIALECT_MD_MDENUMS_TD
#define MDIR_DIALECT_MD_MDENUMS_TD

include "mlir/IR/EnumAttr.td"

// Whether the tuples of a relation have a first member.
def MD_Orientation : I32EnumAttr<"Orientation", "orientation of a relation", [
    I32EnumAttrCase<"Unordered", 0, "unordered">,
    I32EnumAttrCase<"Ordered", 1, "ordered">
  ]> {
  let cppNamespace = "::mdir::md";
}

// How a pair kernel behaves when the two particles are swapped.
def MD_Exchange : I32EnumAttr<"Exchange", "exchange behavior of a kernel", [
    I32EnumAttrCase<"None", 0, "none">,
    I32EnumAttrCase<"Symmetric", 1, "symmetric">,
    I32EnumAttrCase<"Antisymmetric", 2, "antisymmetric">
  ]> {
  let cppNamespace = "::mdir::md";
}

// What an exchange contract rests on.
def MD_ExchangeBasis : I32EnumAttr<"ExchangeBasis",
    "basis of an exchange contract", [
    // The compiler must prove the contract from the kernel.
    I32EnumAttrCase<"Proof", 0, "proof">,
    // The front end asserts the contract.
    I32EnumAttrCase<"Asserted", 1, "asserted">,
    // A compiler pass produced the kernel; the contract holds by construction.
    I32EnumAttrCase<"Derived", 2, "derived">
  ]> {
  let cppNamespace = "::mdir::md";
}

// How a kernel is modified near the cutoff.
def MD_Truncation : I32EnumAttr<"Truncation", "truncation of a kernel", [
    I32EnumAttrCase<"None", 0, "none">,
    I32EnumAttrCase<"Shift", 1, "shift">,
    I32EnumAttrCase<"ForceShift", 2, "force_shift">,
    I32EnumAttrCase<"Switch", 3, "switch">
  ]> {
  let cppNamespace = "::mdir::md";
}

// What `md.evaluate` can be asked for.
def MD_Request : I32EnumAttr<"Request", "quantity requested from a potential", [
    I32EnumAttrCase<"Energy", 0, "energy">,
    I32EnumAttrCase<"Forces", 1, "forces">,
    I32EnumAttrCase<"Virial", 2, "virial">,
    I32EnumAttrCase<"Derivative", 3, "derivative">
  ]> {
  let cppNamespace = "::mdir::md";
}

#endif // MDIR_DIALECT_MD_MDENUMS_TD
