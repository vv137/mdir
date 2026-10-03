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

def MDRT_IncidenceType : MDRT_Type<"Incidence", "incidence"> {
  let summary = "For every particle, the tuples that it is a member of";
  let description = [{
    The tuples of a tuple set, by particle. For every particle it holds the
    tuples that the particle is a member of, with the place of the particle
    in each and the other members.

    ```mlir
    !mdrt.incidence<@atoms, 3, @angles>
    ```

    A relation without a tuple set, such as the triplets of
    `md.triplets`, has an incidence structure without one:

    ```mlir
    !mdrt.incidence<@atoms, 3>
    ```
  }];
  let parameters = (ins
    "::mlir::FlatSymbolRefAttr":$particleSet,
    "unsigned":$arity,
    OptionalParameter<"::mlir::FlatSymbolRefAttr">:$tupleSet
  );
  let assemblyFormat =
      "`<` $particleSet `,` $arity (`,` $tupleSet^)? `>`";
}

def MDRT_EventType : MDRT_Type<"Event", "event"> {
  let summary = "An opaque runtime completion object";
  let description = [{
    Stands for the completion of work that the runtime carries out
    asynchronously. It is not tied to one transport primitive.
  }];
}

#endif // MDIR_DIALECT_MDRT_MDRTTYPES_TD
