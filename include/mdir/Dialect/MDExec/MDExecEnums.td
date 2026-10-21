// Enumerations of the md_exec dialect.

#ifndef MDIR_DIALECT_MDEXEC_MDEXECENUMS_TD
#define MDIR_DIALECT_MDEXEC_MDEXECENUMS_TD

include "mlir/IR/EnumAttr.td"

// The physical form of a neighbor structure.
def MDExec_NeighborKind : I32EnumAttr<"NeighborKind",
    "physical form of a neighbor structure", [
    // A row of fixed width per particle.
    I32EnumAttrCase<"Matrix", 0, "matrix">,
    // Groups of 16 particles that share a list of neighbors, each pair once
    // (D89, docs/neighbor-groups.md). On a device only.
    I32EnumAttrCase<"Groups", 1, "groups">
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

// How a loop over pairs takes the displacement of a pair, and the pairs
// that a neighbor structure holds.
def MDExec_Images : I32EnumAttr<"Images", "images of a displacement", [
    // The minimum image in the cell.
    I32EnumAttrCase<"Minimum", 0, "minimum">,
    // The plain difference of the positions: the rows carry their images,
    // as the ghost rows of a domain do (D290,
    // docs/distributed-ml2.md, Section 7.2).
    I32EnumAttrCase<"Explicit", 1, "explicit">
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

// How a loop or a sum adds values into a field at indices that its threads
// share: the strategy of a scatter-accumulate (D285,
// docs/device-primitives.md), which `md-exec-choose-scatter` sets.
def MDExec_ScatterStrategy : I32EnumAttr<"ScatterStrategy",
    "strategy of a scatter-accumulate", [
    // An atomic addition of the device for each value.
    I32EnumAttrCase<"Atomic", 0, "atomic">,
    // A dense tile in block memory for each bin of the indices, whose
    // footprint the IR bounds; one atomic for each point of the tile.
    I32EnumAttrCase<"Tile", 1, "tile">,
    // A hash table in block memory, keyed by the index, with bounded
    // probing and a direct atomic where the probing fails; one atomic for
    // each index of the table.
    I32EnumAttrCase<"Hash", 2, "hash">,
    // Sorted by index and the runs reduced, without atomics: reserved for
    // #417, refused by the verifier.
    I32EnumAttrCase<"Sorted", 3, "sorted">
  ]> {
  let cppNamespace = "::mdir::md_exec";
}

// The algebra of a reduced value: a licence to the lowering about the
// order of evaluation of its sum (D301, ruled on #470;
// docs/device-primitives.md, Section 5). It is not an algebraic claim:
// floating-point addition is not associative. `fixed_order` may be met
// through an `any_order_exact` form: the fixed-point spreading of PME in the
// deterministic mode is the case. The licence is about order only:
// `any_order` does not permit changing the accumulator's precision (an f64
// sum taken in f32) or the order of growth of its rounding error (a tree
// replaced by a serial chain of atomics); such a change is a separate
// decision, with its own measurement (the maintainer's guards on #470).
// Only md-exec-assign-algebra states an algebra; every other pass carries
// it unchanged.
def MDExec_Algebra : I32EnumAttr<"Algebra", "algebra of a reduced value", [
    // any_order_exact: the lowering may evaluate in any order and grouping,
    // and every order gives the same bits (integers, `or`, fixed point).
    I32EnumAttrCase<"AnyOrderExact", 0, "any_order_exact">,
    // any_order: the lowering may evaluate in any order and grouping. For
    // floating-point values the result's rounding then depends on that
    // order. This is the floating-point sums of the default mode.
    I32EnumAttrCase<"AnyOrder", 1, "any_order">,
    // fixed_order: the lowering must use the order the IR states, or an
    // exact form of the sum, so that the result is the same on every run of
    // the same build. For md_dist this holds for a given decomposition
    // (#409).
    I32EnumAttrCase<"FixedOrder", 2, "fixed_order">
  ]> {
  let cppNamespace = "::mdir::md_exec";
  let genSpecializedAttr = 0;
}

// When a neighbor structure is rebuilt.
def MDExec_RebuildPolicy : I32EnumAttr<"RebuildPolicy",
    "when a neighbor structure is rebuilt", [
    // Whenever the validity condition fails. The condition is evaluated at
    // every refresh.
    I32EnumAttrCase<"Check", 0, "check">,
    // At every refresh.
    I32EnumAttrCase<"Always", 1, "always">,
    // At the first refresh, and then at every `interval`-th refresh since
    // the last build, whether the structure is still valid or not. It may
    // miss pairs within the cutoff; opt-in only (D88).
    I32EnumAttrCase<"Interval", 2, "interval">
  ]> {
  let cppNamespace = "::mdir::md_exec";
}

#endif // MDIR_DIALECT_MDEXEC_MDEXECENUMS_TD
