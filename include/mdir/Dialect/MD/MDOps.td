// Ops of the md dialect.
//
// See docs/ops-m0.md, Section 4, for the specification.

#ifndef MDIR_DIALECT_MD_MDOPS_TD
#define MDIR_DIALECT_MD_MDOPS_TD

include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mdir/Dialect/MD/MDTypes.td"

//===----------------------------------------------------------------------===//
// Particle sets
//===----------------------------------------------------------------------===//

def MD_ParticleSetOp : MD_Op<"particle_set", [Symbol]> {
  let summary = "Declares a particle set";
  let description = [{
    Declares a symbol that field and relation types refer to. The number of
    particles is a run-time value.

    ```mlir
    md.particle_set @atoms
    ```
  }];
  let arguments = (ins SymbolNameAttr:$sym_name);
  let assemblyFormat = "$sym_name attr-dict";
}

//===----------------------------------------------------------------------===//
// Functions
//===----------------------------------------------------------------------===//

class MD_FunctionLikeOp<string mnemonic>
    : MD_Op<mnemonic, [FunctionOpInterface, IsolatedFromAbove]> {
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<StrAttr>:$sym_visibility,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );
  let regions = (region AnyRegion:$body);
  let extraClassDeclaration = [{
    ::mlir::Region *getCallableRegion() {
      return isExternal() ? nullptr : &getBody();
    }
    ::llvm::ArrayRef<::mlir::Type> getArgumentTypes() {
      return getFunctionType().getInputs();
    }
    ::llvm::ArrayRef<::mlir::Type> getResultTypes() {
      return getFunctionType().getResults();
    }
    bool isDeclaration() { return isExternal(); }
  }];
  let hasCustomAssemblyFormat = 1;
}

def MD_PotentialOp : MD_FunctionLikeOp<"potential"> {
  let summary = "Defines a potential energy function";
  let description = [{
    Defines the coordinate-dependent potential energy `U(x; θ)`. The first
    argument is the position field and the second is the cell. All parameters
    are explicit arguments. The single result is the energy.

    ```mlir
    md.potential @lj(%x: !md.field<@atoms, 3 x f64>, %cell: !md.cell,
                     %eps: f64, %sigma: f64) -> f64 {
      ...
      md.return %u : f64
    }
    ```
  }];
  let hasVerifier = 1;
}

def MD_FunctionOp : MD_FunctionLikeOp<"function"> {
  let summary = "Defines a pure function of fields";
  let description = [{
    Has the form of `md.potential`, with any number of results of any type.
    Derivative programs and observables are `md.function`s.
  }];
  let hasVerifier = 1;
}

def MD_ReturnOp : MD_Op<"return", [
    Pure, Terminator, ReturnLike,
    ParentOneOf<["PotentialOp", "FunctionOp"]>
  ]> {
  let summary = "Returns from a potential or a function";
  let arguments = (ins Variadic<AnyType>:$values);
  let assemblyFormat = "attr-dict ($values^ `:` type($values))?";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Cells
//===----------------------------------------------------------------------===//

def MD_OrthorhombicCellOp : MD_Op<"orthorhombic_cell", [Pure]> {
  let summary = "A periodic cell with perpendicular edges";
  let description = [{
    The cell is periodic in all three directions. The operands are the
    lengths of its edges along x, y, and z.

    ```mlir
    %cell = md.orthorhombic_cell %lx, %ly, %lz
    ```
  }];
  let arguments = (ins F64:$lx, F64:$ly, F64:$lz);
  let results = (outs MD_CellType:$result);
  let assemblyFormat = "$lx `,` $ly `,` $lz attr-dict";
}

//===----------------------------------------------------------------------===//
// Relations
//===----------------------------------------------------------------------===//

def MD_NeighborhoodOp : MD_Op<"neighborhood", [Pure]> {
  let summary = "Builds the relation of pairs within a cutoff";
  let description = [{
    ```text
    N(x, h) = { {i, j} : i ≠ j, r_ij < r_c }
    ```

    `r_ij` is the length of the minimum-image displacement.

    ```mlir
    %n = md.neighborhood %x, %cell cutoff(2.5)
           : !md.field<@atoms, 3 x f64> -> !md.relation<@atoms, 2, unordered>
    ```
  }];
  let arguments = (ins
    MD_ReferenceFieldType:$positions,
    MD_CellType:$cell,
    F64Attr:$cutoff
  );
  let results = (outs MD_RelationType:$result);
  let assemblyFormat = [{
    $positions `,` $cell `cutoff` `(` $cutoff `)` attr-dict
    `:` qualified(type($positions)) `->` qualified(type($result))
  }];
  let hasVerifier = 1;
}

def MD_YieldOp : MD_Op<"yield", [
    Pure, Terminator, ReturnLike,
    ParentOneOf<["SumRelationOp", "GatherRelationOp", "SumParticlesOp",
                 "MapParticlesOp"]>
  ]> {
  let summary = "Yields the value of a kernel";
  let arguments = (ins Variadic<AnyType>:$values);
  let assemblyFormat = "attr-dict ($values^ `:` type($values))?";
}

def MD_SumRelationOp : MD_Op<"sum_relation", [
    RecursiveMemoryEffects, SingleBlock
  ]> {
  let summary = "Sums a kernel over the tuples of a relation";
  let description = [{
    ```text
    S = Σ_{t ∈ R} k(t)
    ```

    Each tuple contributes once. The kernel receives the distance `r`, the
    displacement `d`, and two values for each gathered field.

    Over an unordered relation the kernel must be invariant under exchange
    of the two particles, so the op must carry `exchange(symmetric)`.

    ```mlir
    %u = md.sum_relation %n, %x, %cell
           exchange(symmetric) truncation(switch, from = 2.0) {
    ^bb0(%r: f64, %d: vector<3xf64>):
      ...
      md.yield %k : f64
    } : !md.relation<@atoms, 2, unordered>, !md.field<@atoms, 3 x f64> -> f64
    ```
  }];
  let arguments = (ins
    MD_RelationType:$relation,
    MD_ReferenceFieldType:$positions,
    MD_CellType:$cell,
    Variadic<MD_ReferenceFieldType>:$gathered,
    MD_Exchange:$exchange,
    DefaultValuedAttr<MD_ExchangeBasis,
                      "::mdir::md::ExchangeBasis::Proof">:$exchange_basis,
    DefaultValuedAttr<MD_Truncation,
                      "::mdir::md::Truncation::None">:$truncation,
    OptionalAttr<F64Attr>:$switch_from
  );
  let regions = (region SizedRegion<1>:$kernel);
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $relation `,` $positions `,` $cell
    (`gather` `(` $gathered^ `:` qualified(type($gathered)) `)`)?
    custom<Exchange>($exchange, $exchange_basis)
    custom<Truncation>($truncation, $switch_from)
    $kernel attr-dict
    `:` qualified(type($relation)) `,` qualified(type($positions)) `->` qualified(type($result))
  }];
  let hasVerifier = 1;
  let hasRegionVerifier = 1;
}

def MD_GatherRelationOp : MD_Op<"gather_relation", [
    RecursiveMemoryEffects, SingleBlock
  ]> {
  let summary = "Sums a kernel, per particle, over the tuples it belongs to";
  let description = [{
    ```text
    a_i = Σ_{j : (i, j) ∈ D(R)} k(i, j)
    ```

    `D(R)` is the directed expansion of the relation. The kernel is evaluated
    with `i` as the central particle, and `d` is `d_ij`.

    The definition is a gather and has no write conflicts. `exchange` records
    a fact that an execution policy may exploit.

    ```mlir
    %f = md.gather_relation %n, %x, %cell exchange(antisymmetric, derived) {
    ^bb0(%r: f64, %d: vector<3xf64>):
      ...
      md.yield %k : vector<3xf64>
    } : !md.relation<@atoms, 2, unordered>, !md.field<@atoms, 3 x f64>
        -> !md.field<@atoms, 3 x f64>
    ```
  }];
  let arguments = (ins
    MD_RelationType:$relation,
    MD_ReferenceFieldType:$positions,
    MD_CellType:$cell,
    Variadic<MD_ReferenceFieldType>:$gathered,
    MD_Exchange:$exchange,
    DefaultValuedAttr<MD_ExchangeBasis,
                      "::mdir::md::ExchangeBasis::Proof">:$exchange_basis
  );
  let regions = (region SizedRegion<1>:$kernel);
  let results = (outs MD_ReferenceFieldType:$result);
  let assemblyFormat = [{
    $relation `,` $positions `,` $cell
    (`gather` `(` $gathered^ `:` qualified(type($gathered)) `)`)?
    custom<Exchange>($exchange, $exchange_basis)
    $kernel attr-dict
    `:` qualified(type($relation)) `,` qualified(type($positions)) `->` qualified(type($result))
  }];
  let hasVerifier = 1;
  let hasRegionVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Particles
//===----------------------------------------------------------------------===//

def MD_SumParticlesOp : MD_Op<"sum_particles", [
    RecursiveMemoryEffects, SingleBlock
  ]> {
  let summary = "Sums a kernel over particles";
  let description = [{
    ```text
    S = Σ_{i ∈ P} k(i)
    ```

    The kernel receives one value for each gathered field.

    ```mlir
    %ke = md.sum_particles gather(%v, %m : !md.field<@atoms, 3 x f64>,
                                           !md.field<@atoms, f64>) {
    ^bb0(%v_i: vector<3xf64>, %m_i: f64):
      ...
      md.yield %k : f64
    } : f64
    ```
  }];
  let arguments = (ins Variadic<MD_ReferenceFieldType>:$gathered);
  let regions = (region SizedRegion<1>:$kernel);
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    `gather` `(` $gathered `:` qualified(type($gathered)) `)`
    $kernel attr-dict `:` type($result)
  }];
  let hasVerifier = 1;
  let hasRegionVerifier = 1;
}

def MD_MapParticlesOp : MD_Op<"map_particles", [
    RecursiveMemoryEffects, SingleBlock
  ]> {
  let summary = "Applies a kernel to each particle";
  let description = [{
    ```text
    b_i = k(i)
    ```

    ```mlir
    %f = md.map_particles gather(%f1, %f2 : !md.field<@atoms, 3 x f64>,
                                            !md.field<@atoms, 3 x f64>) {
    ^bb0(%a: vector<3xf64>, %b: vector<3xf64>):
      %s = arith.addf %a, %b : vector<3xf64>
      md.yield %s : vector<3xf64>
    } : !md.field<@atoms, 3 x f64>
    ```
  }];
  let arguments = (ins Variadic<MD_ReferenceFieldType>:$gathered);
  let regions = (region SizedRegion<1>:$kernel);
  let results = (outs MD_ReferenceFieldType:$result);
  let assemblyFormat = [{
    `gather` `(` $gathered `:` qualified(type($gathered)) `)`
    $kernel attr-dict `:` qualified(type($result))
  }];
  let hasVerifier = 1;
  let hasRegionVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Evaluation
//===----------------------------------------------------------------------===//

def MD_EvaluateOp : MD_Op<"evaluate", [
    Pure, DeclareOpInterfaceMethods<SymbolUserOpInterface>
  ]> {
  let summary = "Requests energy and derivatives of a potential";
  let description = [{
    | Request | Result type | Definition |
    |---|---|---|
    | `energy` | `f64` | `U` |
    | `forces` | Type of the position field | `F_i = −∂U/∂x_i` |
    | `virial` | `vector<9xf64>` | `W = Σ d_ij ⊗ K(i, j)`, in row-major order |
    | `derivative(n)` | `f64` | `∂U/∂θ_n`, where `θ_n` is scalar argument `n` |

    The op does not survive semantic differentiation, which replaces it with
    a call to a generated `md.function`.

    ```mlir
    %u, %f = md.evaluate @lj(%x, %cell, %eps, %sigma) request [energy, forces]
               : (!md.field<@atoms, 3 x f64>, !md.cell, f64, f64)
               -> (f64, !md.field<@atoms, 3 x f64>)
    ```
  }];
  let arguments = (ins
    FlatSymbolRefAttr:$callee,
    Variadic<AnyType>:$operands,
    DenseI32ArrayAttr:$request_kinds,
    DenseI64ArrayAttr:$request_arguments
  );
  let results = (outs Variadic<AnyType>);
  let assemblyFormat = [{
    $callee `(` $operands `)`
    `request` custom<Requests>($request_kinds, $request_arguments)
    attr-dict `:` functional-type($operands, results)
  }];
  let hasVerifier = 1;
}

def MD_CallOp : MD_Op<"call", [
    Pure, DeclareOpInterfaceMethods<SymbolUserOpInterface>
  ]> {
  let summary = "Calls a function or a potential";
  let description = [{
    ```mlir
    %u, %f = md.call @lj.energy_forces(%x, %cell, %eps, %sigma)
               : (!md.field<@atoms, 3 x f64>, !md.cell, f64, f64)
               -> (f64, !md.field<@atoms, 3 x f64>)
    ```
  }];
  let arguments = (ins
    FlatSymbolRefAttr:$callee,
    Variadic<AnyType>:$operands
  );
  let results = (outs Variadic<AnyType>);
  let assemblyFormat = [{
    $callee `(` $operands `)` attr-dict
    `:` functional-type($operands, results)
  }];
}

#endif // MDIR_DIALECT_MD_MDOPS_TD
