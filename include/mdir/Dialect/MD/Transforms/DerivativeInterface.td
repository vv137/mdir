include "mlir/IR/OpBase.td"

def DerivativeOpInterface : OpInterface<"DerivativeOpInterface"> {
  let cppNamespace = "::mdir::md";
  let description = [{
    A scalar derivative rule, including an explicit declaration of which
    operands carry tangents and which only choose structure or branches.
    A successful null tangent is a proven zero; failure is never zero.

    An op may also have a reverse rule, `emitAdjoint`: the cotangents of
    its operands from those of its results. `md.apply_stage` has only that
    rule, the vector-Jacobian product of its stage
    (D288, docs/ml4-ir.md).
  }];
  let methods = [
    InterfaceMethod<"The role of an operand in the derivative",
      "::mdir::md::DerivativeOperand", "getDerivativeOperand",
      (ins "unsigned":$index)>,
    InterfaceMethod<"Emit the tangent of a result",
      "::mlir::LogicalResult", "emitDerivative",
      (ins "::mlir::Value":$result,
           "::mdir::md::ScalarDerivative &":$derivative,
           "::mlir::Value &":$tangent)>,
    InterfaceMethod<[{
        Emits, at the insertion point of `builder`, the cotangents of the
        operands from `cotangents`, one for each result: one value for each
        operand in `operandCotangents`, null for a structural operand.
        Fails after a diagnostic if the op has no reverse rule.
      }],
      "::mlir::LogicalResult", "emitAdjoint",
      (ins "::mlir::OpBuilder &":$builder,
           "::mlir::ValueRange":$cotangents,
           "::llvm::SmallVectorImpl<::mlir::Value> &":$operandCotangents),
      [{}],
      [{
        return $_op->emitOpError() << "has no reverse rule";
      }]>
  ];
}
