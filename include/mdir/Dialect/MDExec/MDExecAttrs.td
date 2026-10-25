// Attributes of the md_exec dialect.

#ifndef MDIR_DIALECT_MDEXEC_MDEXECATTRS_TD
#define MDIR_DIALECT_MDEXEC_MDEXECATTRS_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mdir/Dialect/MDExec/MDExecDialect.td"

// A physical identity layout that also names the particle or tuple set
// whose elements the rows of a storage-form buffer hold (D337).
def MDExec_SetLayoutAttr : AttrDef<MDExec_Dialect, "SetLayout",
    [DeclareAttrInterfaceMethods<MemRefLayoutAttrInterface, ["getAffineMap"]>]> {
  let mnemonic = "set_layout";
  let summary = "Identity memref layout carrying the set of its rows";
  let parameters = (ins "::mlir::FlatSymbolRefAttr":$set,
                        "int64_t":$rank);
  let assemblyFormat = "`<` $set `,` $rank `>`";
  let genVerifyDecl = 1;
}

def MDExec_ScatterAttr : AttrDef<MDExec_Dialect, "Scatter"> {
  let mnemonic = "scatter";
  let summary = "The strategy of a scatter-accumulate";
  let description = [{
    How an op adds values into a field at indices that its threads share
    (D285, docs/device-primitives.md):

    | Form | Strategy |
    |---|---|
    | `atomic` | an atomic addition of the device for each value |
    | `tile(E)` | a dense tile in block memory for each bin of $E^3$ points |
    | `hash(S, P, L)` | a hash table of `S` slots in block memory, at most `P` slots tried before a direct atomic, taken from `L` items on |
    | `sorted` | reserved for #417 |

    `md-exec-choose-scatter` sets it where the strategy's preconditions
    hold (include/mdir/Dialect/MDExec/Scatter.h); the lowering checks them
    with the same functions.

    ```mlir
    #md_exec.scatter<tile(8)>
    #md_exec.scatter<hash(512, 8, 100000)>
    ```
  }];
  let parameters = (ins
    "::mdir::md_exec::ScatterStrategy":$strategy,
    // tile: the points of an edge of a bin.
    "int64_t":$edge,
    // hash: the slots of the table, a power of two.
    "int64_t":$slots,
    // hash: the slots that a search tries before the direct atomic.
    "int64_t":$probes,
    // hash: the fewest items of a launch that take the table; fewer keep
    // the atomics, chosen on the host by the count.
    "int64_t":$least
  );
  let builders = [
    AttrBuilder<(ins "::mdir::md_exec::ScatterStrategy":$strategy), [{
      return $_get($_ctxt, strategy, 0, 0, 0, 0);
    }]>
  ];
  let extraClassDeclaration = [{
    /// `tile(edge)`.
    static ScatterAttr getTile(::mlir::MLIRContext *context, int64_t edge);
    /// `hash(slots, probes, least)`.
    static ScatterAttr getHash(::mlir::MLIRContext *context, int64_t slots,
                               int64_t probes, int64_t least);
    /// Prints the strategy and its parameters, `tile(8)`.
    void printBody(::llvm::raw_ostream &os) const;
    /// Parses what printBody prints.
    static ScatterAttr parseBody(::mlir::AsmParser &parser);
    bool isTile() const {
      return getStrategy() == ::mdir::md_exec::ScatterStrategy::Tile;
    }
    bool isHash() const {
      return getStrategy() == ::mdir::md_exec::ScatterStrategy::Hash;
    }
  }];
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

def MDExec_AlgebraListAttr : AttrDef<MDExec_Dialect, "AlgebraList"> {
  let mnemonic = "algebra";
  let summary = "The algebra of each value that an op reduces";
  let description = [{
    One algebra for each value that an op reduces, in the order of
    `md_exec::algebra::getReducedTypes` (D301,
    include/mdir/Dialect/MDExec/Algebra.h): a loop's destinations, then its
    `reduce`; the destination of `md_exec.scatter_add`; the spreading, the
    energy, and the virial of `md_exec.reciprocal`; the sum of
    `md_dist.all_reduce` and of `md_dist.halo_sum`.

    Each is a licence to the lowering about the order of evaluation
    (ruled on #470):

    - `any_order_exact`: the lowering may evaluate in any order and
      grouping, and every order gives the same bits (integers, `or`, fixed
      point).
    - `any_order`: the lowering may evaluate in any order and grouping. For
      floating-point values the result's rounding then depends on that
      order. This is the floating-point sums of the default mode.
    - `fixed_order`: the lowering must use the order the IR states, or an
      exact form of the sum, so that the result is the same on every run of
      the same build. For `md_dist` this holds for a given decomposition
      (#409).

    `fixed_order` may be met through an `any_order_exact` form: the
    fixed-point spreading of PME in the deterministic mode is the case.

    The licence is about order only. `any_order` permits any order and
    grouping; it does not permit changing the accumulator's precision (an
    f64 sum taken in f32) or the order of growth of its rounding error (a
    tree replaced by a serial chain of atomics). Such a change is a
    separate decision, with its own measurement.

    `md-exec-assign-algebra` states it as `md_exec.algebra`, a discardable
    attribute: a pass that rebuilds the op without it loses the fact, and
    the lowerings then refuse the op rather than assume an algebra.

    ```mlir
    {md_exec.algebra = #md_exec.algebra<any_order, any_order_exact>}
    ```
  }];
  let parameters = (ins ArrayRefParameter<"::mdir::md_exec::Algebra">:$values);
  let hasCustomAssemblyFormat = 1;
}

#endif // MDIR_DIALECT_MDEXEC_MDEXECATTRS_TD
