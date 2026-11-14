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

def MDRT_NeighborsType : MDRT_Type<"Neighbors", "neighbors"> {
  let summary = "A physical neighbor structure";
  let description = [{
    A list of pairs that was built at a reference configuration. It contains
    every pair within the cutoff for as long as no particle has moved more
    than half the skin.

    The type states what the build guarantees about its lists
    (D413, `NeighborsContract.h`), in four words:

    - the traversal: `directed`, both orders of each pair, or `unique`,
      each pair once;
    - the images: `minimum`, the minimum image in the cell, or `explicit`,
      the plain difference of positions that carry their images (D290);
    - the exclusions: `masked`, an entry may stand for an excluded pair
      (the particle itself), which a consumer skips, or `removed`, no entry
      stands for one;
    - the organization (the values of D409): `none`, `grouped_by(G)`,
      `sorted_within(G, K)`, or `stable(G, K)`, with the group `G` the row
      of one center (`owner`) or a group of a build of groups (`group`),
      and the key `K` the partner (`item`) or the order of the build's
      search (`search`). Both keys are total, so `sorted_within(G, K)` is
      read as `stable(G, K)`.

    A consumer states what it requires in the same values, and its verifier
    refuses a structure whose guarantee does not meet it: `masked ⊑
    removed`, `none ⊑ grouped_by(G) ⊑ sorted_within(G, K) ⊑ stable(G, K)`
    for the same keys, and the traversal and the images equal. Cost
    choices, such as the kind of the structure, its width, or the order of
    the places of a build of groups, are attributes of the build, not part
    of the type.

    The four words are always spelled. A matrix:

    ```mlir
    !mdrt.neighbors<@atoms, directed, minimum, masked, stable(owner, search)>
    ```

    A structure of groups, and its dual list:

    ```mlir
    !mdrt.neighbors<@atoms, unique, minimum, removed, stable(group, search)>
    ```
  }];
  let parameters = (ins
    "::mlir::FlatSymbolRefAttr":$particleSet,
    "::mdir::mdrt::NeighborsContract":$contract
  );
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
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
