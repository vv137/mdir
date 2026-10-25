// The md_exec dialect: how the MD computation is organized.
//
// See docs/ops.md, Section 8, for the specification.

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

    Discardable attributes of the storage form (D337, `StorageSet.h`):

    - `md_exec.tuple_set` on the builder of an incidence buffer and on its
      tuple loops declares the tuple set of the buffer, which its type does
      not carry yet.
    - `md_exec.typed_storage` on a module marks storage that
      `md-exec-assign-storage` assigned, whose loop operands must be tagged
      with their set; the conversions to loops and to a device remove both.
    - `md_exec.input_set_boundary` is a unit marker on `memref.cast` where
      an external plain input first acquires its declared storage set. The
      dialect verifies its placement, value, and set provenance
      (D345).
  }];
  let cppNamespace = "::mdir::md_exec";
  let dependentDialects = ["::mdir::md::MDDialect",
                           "::mdir::mdrt::MDRTDialect"];
  let useDefaultAttributePrinterParser = 1;
  // `md_exec.algebra` (D301).
  let hasOperationAttrVerify = 1;
  let extraClassDeclaration = [{
    void registerAttributes();
  }];
}

class MDExec_Op<string mnemonic, list<Trait> traits = []>
    : Op<MDExec_Dialect, mnemonic, traits>;

#endif // MDIR_DIALECT_MDEXEC_MDEXECDIALECT_TD
