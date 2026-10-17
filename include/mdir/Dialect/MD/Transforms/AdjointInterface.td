include "mlir/IR/OpBase.td"

def AdjointOpInterface : OpInterface<"AdjointOpInterface"> {
  let cppNamespace = "::mdir::md";
  let description = [{
    A declared transpose (D[ml2-distribute]): an op of one result that is
    linear, or affine, in some of its operands, and whose transpose is
    another op. `md-differentiate` carries the adjoint of the result back
    to each linear operand through it; an operand that is not linear is
    structural (a layout, a halo, a cell), and no adjoint flows to it.

    The transfers of md_dist are the case: the transpose of the copy to
    the ghosts, G (`md_dist.halo_copy`), is the sum at the owners, G^T
    (`md_dist.halo_sum`), and the transpose of G^T is G
    (docs/distributed-ml2.md, Sections 4.5 and 11).
  }];
  let methods = [
    InterfaceMethod<"Whether the op is linear, or affine, in operand `index`",
      "bool", "isLinearOperand", (ins "unsigned":$index)>,
    InterfaceMethod<[{
        Emits the adjoint of operand `index` that the adjoint `adjoint` of
        the result gives, and returns it, or a null value after a
        diagnostic. `partner`, if not null, is an adjoint of the same
        operand that must be summed with it in one op: the contributions at
        owned rows of the traversal whose contributions at ghost rows
        `adjoint` holds, which no other op may read.
      }],
      "::mlir::Value", "emitTranspose",
      (ins "::mlir::OpBuilder &":$builder, "unsigned":$index,
           "::mlir::Value":$adjoint, "::mlir::Value":$partner)>
  ];
}
