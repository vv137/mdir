// Types of the mdrt dialect.

#ifndef MDIR_DIALECT_MDRT_MDRTTYPES_TD
#define MDIR_DIALECT_MDRT_MDRTTYPES_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mdir/Dialect/MDRT/MDRTDialect.td"

class MDRT_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<MDRT_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

// A type that describes a structure over one particle set.
class MDRT_ParticleSetType<string name, string typeMnemonic>
    : MDRT_Type<name, typeMnemonic> {
  let parameters = (ins "::mlir::FlatSymbolRefAttr":$particleSet);
  let assemblyFormat = "`<` $particleSet `>`";
}

def MDRT_CellsType : MDRT_ParticleSetType<"Cells", "cells"> {
  let summary = "Particles binned into cells";
  let description = [{
    For every cell, the particles that lie in it.

    ```mlir
    !mdrt.cells<@atoms>
    ```
  }];
}

def MDRT_PermutationType : MDRT_ParticleSetType<"Permutation", "permutation"> {
  let summary = "An order of the particles of a set";
  let description = [{
    ```mlir
    !mdrt.permutation<@atoms>
    ```
  }];
}

def MDRT_NeighborsType : MDRT_ParticleSetType<"Neighbors", "neighbors"> {
  let summary = "A physical neighbor structure";
  let description = [{
    A list of pairs that was built at a reference configuration. It contains
    every pair within the cutoff for as long as no particle has moved more
    than half the skin.

    ```mlir
    !mdrt.neighbors<@atoms>
    ```
  }];
}

def MDRT_EventType : MDRT_Type<"Event", "event"> {
  let summary = "An opaque runtime completion object";
  let description = [{
    Stands for the completion of work that the runtime carries out
    asynchronously. It is not tied to one transport primitive.
  }];
}

#endif // MDIR_DIALECT_MDRT_MDRTTYPES_TD
