// Attributes of the md_exec dialect.

#ifndef MDIR_DIALECT_MDEXEC_MDEXECATTRS_TD
#define MDIR_DIALECT_MDEXEC_MDEXECATTRS_TD

include "mlir/IR/AttrTypeBase.td"
include "mdir/Dialect/MDExec/MDExecDialect.td"

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

#endif // MDIR_DIALECT_MDEXEC_MDEXECATTRS_TD
