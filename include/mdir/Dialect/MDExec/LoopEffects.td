include "mlir/IR/OpBase.td"

def MDExec_LoopEffectOpInterface : OpInterface<"LoopEffectOpInterface"> {
  let cppNamespace = "::mdir::md_exec";
  let description = [{
    A derived summary of the accesses of one md_exec execution loop. Each
    entry identifies its exact operand, row scope, and set. An incomplete
    summary cannot prove independence, initialization, or readiness.
  }];
  let methods = [
    InterfaceMethod<"Append the checked access summary of this operation",
      "void", "getLoopEffectSummary",
      (ins "::mdir::md_exec::LoopEffectSummary &":$summary)>
  ];
}
