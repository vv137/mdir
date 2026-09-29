// The md_exec dialect: how the MD computation is organized.
//
// See docs/ops-m0.md, Section 8, for the specification.

#ifndef MDIR_DIALECT_MDEXEC_MDEXECDIALECT_TD
#define MDIR_DIALECT_MDEXEC_MDEXECDIALECT_TD

include "mlir/IR/OpBase.td"

def MDExec_Dialect : Dialect {
  let name = "md_exec";
  let summary = "How the MD computation is organized.";
  let description = [{
    The `md_exec` dialect expresses loops over particles and over the pairs
    of a physical neighbor structure, and the construction of such
    structures. It says over which set a kernel runs and how the kernel
    accesses fields: which it reads, which it accumulates into, and which
    global sums it contributes to.

    In the value form, which is the form defined here, the ops take and
    return field values.
  }];
  let cppNamespace = "::mdir::md_exec";
  let dependentDialects = ["::mdir::md::MDDialect",
                           "::mdir::mdrt::MDRTDialect"];
}

class MDExec_Op<string mnemonic, list<Trait> traits = []>
    : Op<MDExec_Dialect, mnemonic, traits>;

#endif // MDIR_DIALECT_MDEXEC_MDEXECDIALECT_TD
