include "mlir/IR/OpBase.td"

def AdjointOpInterface : OpInterface<"AdjointOpInterface"> {
  let cppNamespace = "::mdir::md";
  let description = [{
    A declared transpose (D276, D295). `md-differentiate`
    carries the adjoint of an op's result back to its operands through it,
    in one of two forms.

    An op of one result that is linear, or affine, in some of its operands
    declares the op that is its transpose (`isLinearOperand`,
    `emitTranspose`). An operand that is not linear is structural (a
    layout, a halo, a cell), and no adjoint flows to it. The transfers of
    md_dist are the case: the transpose of the copy to the ghosts, G
    (`md_dist.halo_copy`), is the sum at the owners, G^T
    (`md_dist.halo_sum`), and the transpose of G^T is G
    (docs/distributed-ml2.md, Sections 4.5 and 11).

    An op with a kernel is a gather, a map, and a reduction
    (docs/primitives-map.md, Section 1.2): it gathers its operands at the
    ends of each pair, at the members of each tuple, or at each particle,
    evaluates the kernel there, and adds the kernel's values at each
    particle or over everything. The gather and the reduction are linear;
    the kernel is not. The op declares, for each operand that reaches its
    kernel (`isKernelOperand`), the op of the transpose of its structure
    (`createTranspose`): the transpose of the reduction, a broadcast of the
    adjoint to each pair, tuple, or particle; the place where the reverse
    derivative of the kernel goes; and the transpose of the gather, which
    adds the kernel's values at the particles it read. The pass fills the
    kernel by the scalar rules (D183), so that the transpose of the op's
    linearization is the composition, its vector-Jacobian product.
  }];
  let methods = [
    InterfaceMethod<"Whether the op is linear, or affine, in operand `index`",
      "bool", "isLinearOperand", (ins "unsigned":$index),
      /*methodBody=*/"",
      /*defaultImplementation=*/"return false;">,
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
           "::mlir::Value":$adjoint, "::mlir::Value":$partner),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        $_op->emitOpError() << "is not linear in operand " << index;
        return ::mlir::Value();
      }]>,
    InterfaceMethod<[{
        Whether operand `index` reaches the kernel of the op through its
        gather, so that an adjoint flows to it through `createTranspose`.
      }],
      "bool", "isKernelOperand", (ins "unsigned":$index),
      /*methodBody=*/"",
      /*defaultImplementation=*/"return false;">,
    InterfaceMethod<[{
        Creates, at the builder's place and at `loc`, the op of the
        transpose of the structure for operand `index` and returns it, or null after a
        diagnostic. `cotangent` is the adjoint of the result: a number,
        which the kernel captures, for a sum over everything, or a field,
        which the new op reads, for a field. `kernel` receives the kernel
        of the new op (see `TransposeKernel`), which the caller completes
        with the reverse derivative of the source kernel and a terminator.
        `exchange` is the contract of what the caller's kernel will yield
        (antisymmetric for forces, none for the adjoint of a field), which
        only the caller knows; an op over particles has none. With
        `summed`, the new op adds the kernel's values over all pairs
        instead of at each particle: the form of the virial,
        Σ d ⊗ K.
      }],
      "::mlir::Operation *", "createTranspose",
      (ins "::mlir::OpBuilder &":$builder, "::mlir::Location":$loc,
           "unsigned":$index, "::mlir::Value":$cotangent,
           "::mdir::md::Exchange":$exchange,
           "::mlir::TypeRange":$resultTypes, "bool":$summed,
           "::mdir::md::TransposeKernel &":$kernel),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        $_op->emitOpError() << "has no transpose of its structure";
        return nullptr;
      }]>
  ];
}
