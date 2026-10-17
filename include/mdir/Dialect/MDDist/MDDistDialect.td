// The md_dist dialect: domains, ownership, halos, and the transfers
// between domains.
//
// See docs/distributed-ml2.md for the design (D275).

#ifndef MDIR_DIALECT_MDDIST_MDDISTDIALECT_TD
#define MDIR_DIALECT_MDDIST_MDDISTDIALECT_TD

include "mlir/IR/OpBase.td"

def MDDist_Dialect : Dialect {
  let name = "md_dist";
  let summary = "Domains, ownership, halos, and transfers between domains.";
  let description = [{
    The `md_dist` dialect says where the values of a distributed
    computation live and what moves between domains. Its values are global
    views: one value of a distributed field stands for the rows of every
    domain, and every op is collective over the domains of its
    decomposition. Ownership (a layout) and halos are values, so that the
    version of a field that a ghost holds, and the epoch of ownership of a
    field, are facts of the IR that the verifier checks.

    Nothing in this dialect names a transport: the same program runs with
    every domain in one process or on separate ranks.
  }];
  let cppNamespace = "::mdir::md_dist";
  let dependentDialects = ["::mdir::md::MDDialect"];

  let useDefaultTypePrinterParser = 1;
  let extraClassDeclaration = [{
    void registerTypes();
  }];
}

class MDDist_Op<string mnemonic, list<Trait> traits = []>
    : Op<MDDist_Dialect, mnemonic, traits>;

#endif // MDIR_DIALECT_MDDIST_MDDISTDIALECT_TD
