// The md dialect: particles, fields, relations, and potentials.
//
// See docs/ops-m0.md for the specification.

#ifndef MDIR_DIALECT_MD_MDDIALECT_TD
#define MDIR_DIALECT_MD_MDDIALECT_TD

include "mlir/IR/OpBase.td"

def MD_Dialect : Dialect {
  let name = "md";
  let summary = "Particles, fields, relations, and potentials.";
  let description = [{
    The `md` dialect expresses what is computed in a molecular dynamics
    simulation. It has a generic relational core (particle sets, per-particle
    fields, relations, and reductions over them) and functions built from
    that core, of which a potential energy function is one kind.

    All ops are pure. Nothing in this dialect says how or where a computation
    is carried out.
  }];
  let cppNamespace = "::mdir::md";

  let useDefaultTypePrinterParser = 1;
  let extraClassDeclaration = [{
    void registerTypes();
  }];
}

class MD_Op<string mnemonic, list<Trait> traits = []>
    : Op<MD_Dialect, mnemonic, traits>;

#endif // MDIR_DIALECT_MD_MDDIALECT_TD
