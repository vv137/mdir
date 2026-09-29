// The dyn dialect: time evolution of the simulation state.
//
// See docs/ops-m0.md, Section 6, for the specification.

#ifndef MDIR_DIALECT_DYN_DYNDIALECT_TD
#define MDIR_DIALECT_DYN_DYNDIALECT_TD

include "mlir/IR/OpBase.td"

def Dyn_Dialect : Dialect {
  let name = "dyn";
  let summary = "Time evolution of the simulation state.";
  let description = [{
    The `dyn` dialect expresses how one time step advances the state. The
    state is the set of values that a step takes and returns: positions,
    velocities, forces, and the cell. Every op takes the old value and
    returns a new one.

    Force evaluation is `md.evaluate`. This dialect has no op of its own for
    it.
  }];
  let cppNamespace = "::mdir::dyn";
  let dependentDialects = ["::mdir::md::MDDialect"];
}

class Dyn_Op<string mnemonic, list<Trait> traits = []>
    : Op<Dyn_Dialect, mnemonic, traits>;

#endif // MDIR_DIALECT_DYN_DYNDIALECT_TD
