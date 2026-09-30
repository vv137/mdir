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

def MD_TupleSetOp : MD_Op<"tuple_set", [Symbol]> {
  let summary = "Declares a set of tuples of particles";
  let description = [{
    Declares the tuples of a topology: bonds, angles, and the like. A tuple
    is an entity, as a particle is. It exists throughout a run, and it has
    fields of its own, such as the parameters of a bond. The number of
    tuples is a run-time value.

    ```mlir
    md.tuple_set @angles on(@atoms) arity(3) orientation(reversal)
    ```

    The members of the tuples are a value of the type
    `!md.relation<@atoms, 3, reversal, @angles>`, and a field of the tuples
    has the type `!md.field<@angles, f64>`.

    `disjoint` states that no particle is a member of two tuples, as in the
    groups of a constraint: a loop over the tuples may then evaluate each
    tuple once and write to all of its members. It is a promise of whoever
    declares the set.

    ```mlir
    md.tuple_set @settles on(@atoms) arity(3) orientation(ordered) disjoint
    ```
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    FlatSymbolRefAttr:$particle_set,
    I64Attr:$arity,
    MD_Orientation:$orientation,
    UnitAttr:$disjoint
  );
  let assemblyFormat = [{
    $sym_name `on` `(` $particle_set `)` `arity` `(` $arity `)`
    `orientation` `(` $orientation `)` (`disjoint` $disjoint^)? attr-dict
  }];
  let hasVerifier = 1;
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

    With `exclude`, the pairs of a relation of a tuple set of arity 2 are
    left out:

    ```text
    N(x, h) = { {i, j} : i ≠ j, r_ij < r_c } \ E
    ```

    ```mlir
    %n = md.neighborhood %x, %cell cutoff(1.0)
           exclude(%e : !md.relation<@atoms, 2, unordered, @excluded>)
           : !md.field<@atoms, 3 x f64> -> !md.relation<@atoms, 2, unordered>
    ```
  }];
  let arguments = (ins
    MD_ReferenceFieldType:$positions,
    MD_CellType:$cell,
    Optional<MD_RelationType>:$excluded,
    F64Attr:$cutoff
  );
  let results = (outs MD_RelationType:$result);
  let assemblyFormat = [{
    $positions `,` $cell `cutoff` `(` $cutoff `)`
    (`exclude` `(` $excluded^ `:` qualified(type($excluded)) `)`)? attr-dict
    `:` qualified(type($positions)) `->` qualified(type($result))
  }];
  let hasVerifier = 1;
}

def MD_LookupOp : MD_Op<"lookup", [Pure]> {
  let summary = "Reads a value of a table";
  let description = [{
    Reads the value of a table for the types `indices`, one for each
    dimension of the table. In a kernel the indices are usually the types
    of the particles, gathered as fields of `i32`.

    ```mlir
    %eps = md.lookup %table[%t_i, %t_j]
             : !md.table<2, f64, symmetric>, i32, i32 -> f64
    ```

    The result is a floating-point number: that of the table, or the type
    that the kernel computes in once precision is assigned. After storage
    is assigned, the table is the buffer that holds it.
  }];
  let arguments = (ins
    AnyType:$table,
    Variadic<AnyTypeOf<[Index, AnySignlessInteger]>>:$indices
  );
  let results = (outs AnyFloat:$result);
  let assemblyFormat = [{
    $table `[` $indices `]` attr-dict `:` qualified(type($table)) `,`
    type($indices) `->` type($result)
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Particle mesh Ewald
//===----------------------------------------------------------------------===//

def MD_ReciprocalOp : MD_Op<"reciprocal", [Pure]> {
  let summary = "The reciprocal sum of particle mesh Ewald";
  let description = [{
    ```text
    E = f / (2π V) Σ_{m ≠ 0} exp(−π² m² / β²) / m² · |S(m)|²
    S(m) = Σ_i q_i exp(2π i m · x_i)
    ```

    The sum of smooth particle mesh Ewald over a grid of `grid` points with
    B-splines of the order `order` (docs/pme-m1.md), with `coulomb` the
    Coulomb constant f. `moduli` holds the factors of the influence function
    along each edge, which do not depend on the cell: row `a` holds
    `|b_a(k)|²` for each index `k` of edge `a`, times the factor of the
    aliasing if it is taken. The influence function is
    `f / (π V) · B(m) exp(−π² m² / β²) / m²` with `B` the product of the
    three, computed from the cell where it is used.

    The op yields the energy, the forces, and the virial `Σ d ⊗ F` of the
    sum. Differentiation takes the forces and the virial from the op; it
    does not differentiate through it.

    ```mlir
    %u, %f, %w = md.reciprocal %x, %q, %cell, %moduli
                   grid([32, 32, 32]) order(4) beta(3.12) coulomb(138.935)
                   : !md.field<@atoms, 3 x f64>, !md.field<@atoms, f64>,
                     !md.table<2, f64>
                   -> f64, !md.field<@atoms, 3 x f64>, vector<9xf64>
    ```
  }];
  let arguments = (ins
    MD_ReferenceFieldType:$positions,
    MD_ReferenceFieldType:$charges,
    MD_CellType:$cell,
    MD_TableType:$moduli,
    DenseI64ArrayAttr:$grid,
    I64Attr:$order,
    F64Attr:$beta,
    F64Attr:$coulomb
  );
  let results = (outs
    F64:$energy,
    MD_ReferenceFieldType:$forces,
    VectorOfLengthAndType<[9], [F64]>:$virial
  );
  let assemblyFormat = [{
    $positions `,` $charges `,` $cell `,` $moduli
    `grid` `(` $grid `)` `order` `(` $order `)` `beta` `(` $beta `)`
    `coulomb` `(` $coulomb `)`
    attr-dict `:` qualified(type($positions)) `,` qualified(type($charges))
    `,` qualified(type($moduli)) `->` type($energy) `,`
    qualified(type($forces)) `,` type($virial)
  }];
  let hasVerifier = 1;
}

def MD_YieldOp : MD_Op<"yield", [
    Pure, Terminator, ReturnLike,
    ParentOneOf<["SumRelationOp", "GatherRelationOp", "SumTuplesOp",
                 "GatherTuplesOp", "SumParticlesOp", "MapParticlesOp"]>
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
// Tuples of a topology
//===----------------------------------------------------------------------===//

class MD_TupleOp<string mnemonic> : MD_Op<mnemonic, [
    AttrSizedOperandSegments, RecursiveMemoryEffects, SingleBlock
  ]> {
  let arguments = (ins
    MD_RelationType:$relation,
    MD_ReferenceFieldType:$positions,
    MD_CellType:$cell,
    Variadic<MD_ReferenceFieldType>:$gathered,
    Variadic<MD_ReferenceFieldType>:$parameters,
    DenseI32ArrayAttr:$coordinate_kinds,
    DenseI64ArrayAttr:$coordinate_members
  );
  let regions = (region SizedRegion<1>:$kernel);
  let assemblyFormat = [{
    $relation `,` $positions `,` $cell
    custom<Coordinates>($coordinate_kinds, $coordinate_members)
    (`gather` `(` $gathered^ `:` qualified(type($gathered)) `)`)?
    (`tuple` `(` $parameters^ `:` qualified(type($parameters)) `)`)?
    $kernel attr-dict
    `:` qualified(type($relation)) `,` qualified(type($positions)) `->` qualified(type($result))
  }];
  let extraClassDeclaration = [{
    /// The internal coordinates that the kernel takes.
    ::llvm::SmallVector<::mdir::md::Coordinate, 2> getCoordinates() {
      return ::mdir::md::getCoordinates(getCoordinateKinds(),
                                        getCoordinateMembers());
    }
    /// The number of members of a tuple.
    unsigned getArity() {
      return ::llvm::cast<::mdir::md::RelationType>(getRelation().getType())
          .getArity();
    }
  }];
  let hasVerifier = 1;
  let hasRegionVerifier = 1;
}

def MD_SumTuplesOp : MD_TupleOp<"sum_tuples"> {
  let summary = "Sums a kernel over the tuples of a tuple set";
  let description = [{
    ```text
    S = Σ_{t ∈ T} k(t)
    ```

    The kernel receives the internal coordinates that the op names, then
    one value for each member of the tuple for each gathered field of the
    particles, then one value for each field of the tuples.

    A place in `coordinates` is a place in the tuple, from 0.

    | Coordinate | Value | Kernel argument |
    |---|---|---|
    | `distance(a, b)` | The length of `d_ab` | `f64` |
    | `displacement(a, b)` | `d_ab = x_a − x_b`, in the minimum image | `vector<3xf64>` |
    | `angle(a, b, c)` | The angle at `b` between `d_ab` and `d_cb`, from 0 to π | `f64` |
    | `cosine(a, b, c)` | The cosine of that angle | `f64` |
    | `dihedral(a, b, c, d)` | The angle between the planes of `a, b, c` and of `b, c, d`, from −π to π | `f64` |

    ```mlir
    %u = md.sum_tuples %angles, %x, %cell coordinates(cosine(0, 1, 2))
           tuple(%k, %c0 : !md.field<@angles, f64>, !md.field<@angles, f64>) {
    ^bb0(%c: f64, %k_t: f64, %c0_t: f64):
      ...
      md.yield %e : f64
    } : !md.relation<@atoms, 3, reversal, @angles>,
        !md.field<@atoms, 3 x f64> -> f64
    ```
  }];
  let results = (outs AnyType:$result);
}

def MD_GatherTuplesOp : MD_TupleOp<"gather_tuples"> {
  let summary = "Sums a kernel, per particle, over the tuples it is in";
  let description = [{
    ```text
    a_i = Σ_{t, s : t[s] = i} k(t)[s]
    ```

    The kernel yields one value for each member of the tuple, in the order
    of the members. A particle receives, from every tuple that it is a
    member of, the value of its place.

    The kernel receives what that of `md.sum_tuples` receives.

    ```mlir
    %f = md.gather_tuples %bonds, %x, %cell
           coordinates(distance(0, 1), displacement(0, 1))
           tuple(%k, %r0 : !md.field<@bonds, f64>, !md.field<@bonds, f64>) {
    ^bb0(%r: f64, %d: vector<3xf64>, %k_t: f64, %r0_t: f64):
      ...
      md.yield %f0, %f1 : vector<3xf64>, vector<3xf64>
    } : !md.relation<@atoms, 2, unordered, @bonds>,
        !md.field<@atoms, 3 x f64> -> !md.field<@atoms, 3 x f64>
    ```
  }];
  let results = (outs MD_ReferenceFieldType:$result);
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
