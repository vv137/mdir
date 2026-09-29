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
  }];
  let cppNamespace = "::mdir::mdrt";

  let useDefaultTypePrinterParser = 1;
  let extraClassDeclaration = [{
    void registerTypes();
  }];
}

#endif // MDIR_DIALECT_MDRT_MDRTDIALECT_TD
