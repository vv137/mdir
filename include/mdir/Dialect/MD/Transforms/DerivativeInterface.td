include "mlir/IR/OpBase.td"

def DerivativeOpInterface : OpInterface<"DerivativeOpInterface"> {
  let cppNamespace = "::mdir::md";
  let description = [{
    A scalar derivative rule, including an explicit declaration of which
    operands carry tangents and which only choose structure or branches.
    A successful null tangent is a proven zero; failure is never zero.
  }];
  let methods = [
    InterfaceMethod<"The role of an operand in the derivative",
      "::mdir::md::DerivativeOperand", "getDerivativeOperand",
      (ins "unsigned":$index)>,
    InterfaceMethod<"Emit the tangent of a result",
      "::mlir::LogicalResult", "emitDerivative",
      (ins "::mlir::Value":$result,
           "::mdir::md::ScalarDerivative &":$derivative,
           "::mlir::Value &":$tangent)>
  ];
}
