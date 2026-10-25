// The mdrt dialect: the runtime as the IR sees it.

#ifndef MDIR_DIALECT_MDRT_MDRTDIALECT_TD
#define MDIR_DIALECT_MDRT_MDRTDIALECT_TD

include "mlir/IR/OpBase.td"

def MDRT_Dialect : Dialect {
  let name = "mdrt";
  let summary = "The runtime as the IR sees it.";
  let description = [{
    The `mdrt` dialect holds the types that the execution dialects share:
    spatial structures, storage, and completion events. It will also hold
    the ops that stand for calls into the runtime.

    Discardable attributes of the dialect (`MDRTOps.h`):

    - `mdrt.segment` marks a loop whose iterations are segments of a run:
      stretches between two checkpoints.
    - `mdrt.fresh` marks a loop whose neighbor structures begin empty
      rather than continue those built before it: the loop of steps after
      the evaluation at the start of a run, which is a segment of its own
      (D218).
    - `mdrt.read_only` is the attribute of an argument of a function that
      the function never writes (D258): on an argument of the entry of a
      program, a buffer of the model that the caller hands without a copy;
      on an argument of a declared function of the runtime, a buffer that
      the runtime only reads. `check-read-only-arguments` proves the first
      by the second and fails the compile where it cannot.
  }];
  let cppNamespace = "::mdir::mdrt";
  let dependentDialects = ["::mdir::md::MDDialect"];

  let useDefaultTypePrinterParser = 1;
  let extraClassDeclaration = [{
    void registerTypes();
  }];
}

#endif // MDIR_DIALECT_MDRT_MDRTDIALECT_TD
