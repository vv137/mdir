include "mlir/IR/OpBase.td"

def MDExec_PairKernelBodyOpInterface : OpInterface<"PairKernelBodyOpInterface"> {
  let cppNamespace = "::mdir::md_exec";
  let description = [{
    Scalar pair computations, each described by its squared distance,
    enclosing kernel region and contribution outputs. A region can contain
    several site-pair computations; clients reacquire views after rewriting
    a region. The interface does not change traversal or cutoff semantics.
  }];
  let methods = [
    InterfaceMethod<"The scalar pair computations in source order",
      "::llvm::SmallVector<::mdir::md_exec::PairKernelBody>",
      "getPairKernelBodies", (ins)>,
    InterfaceMethod<"The physical cutoff of these pair computations",
      "double", "getPairKernelCutoff", (ins)>
  ];
}
