// Ops of the md dialect.
//
// See docs/ops.md, Section 4, for the specification.

#ifndef MDIR_DIALECT_MD_MDOPS_TD
#define MDIR_DIALECT_MD_MDOPS_TD

include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
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

def MD_DisjointUnionOp : MD_Op<"disjoint_union", [
    Symbol, DeclareOpInterfaceMethods<SymbolUserOpInterface>
  ]> {
  let summary = "Declares that tuple sets share no particle";
  let description = [{
    States that no particle is a member of two tuples of the sets, whether
    the two tuples are of one set or of two: the groups of the constraints
    of a topology, for one, where a rigid water and a group of bonds to
    hydrogen never share an atom. Each set is `disjoint`, and the union
    extends the promise across the sets. It is a promise of whoever
    declares the sets.

    ```mlir
    md.disjoint_union @constraints on(@atoms) of [@settles, @shake1, @shake2]
    ```

    A loop over the tuples of one of the sets reads its fields at the
    members of its tuples only, where what a loop over the tuples of an
    other set gathered is zero: the loops of the sets may run in any order,
    and at once (`md-bypass-updates`).
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    FlatSymbolRefAttr:$particle_set,
    FlatSymbolRefArrayAttr:$tuple_sets
  );
  let assemblyFormat = [{
    $sym_name `on` `(` $particle_set `)` `of` $tuple_sets attr-dict
  }];
  let extraClassDeclaration = [{
    /// Returns true if the union has the tuple set `name`.
    bool contains(::mlir::FlatSymbolRefAttr name) {
      return ::llvm::is_contained(getTupleSets(), name);
    }
  }];
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

def MD_TriclinicCellOp : MD_Op<"triclinic_cell", [Pure]> {
  let summary = "A periodic cell of three lattice vectors";
  let description = [{
    The cell is periodic in all three directions, with the lattice vectors
    a = (lx, 0, 0), b = (bx, ly, 0), and c = (cx, cy, lz), the rows of a
    lower-triangular matrix in the reduced form |bx| <= lx/2, |cx| <= lx/2,
    |cy| <= ly/2 (docs/triclinic.md). The operands are its diagonal and
    then its tilts.

    ```mlir
    %cell = md.triclinic_cell %lx, %ly, %lz, %bx, %cx, %cy
    ```
  }];
  let arguments = (ins F64:$lx, F64:$ly, F64:$lz, F64:$bx, F64:$cx,
                       F64:$cy);
  let results = (outs MD_CellType:$result);
  let assemblyFormat = "$lx `,` $ly `,` $lz `,` $bx `,` $cx `,` $cy attr-dict";
}

def MD_CheckedRectangularCellOp : MD_Op<"checked_rectangular_cell", [MemoryEffects<[]>]> {
  let summary = "A checked rectangular-cell precondition for the selected chain";
  let description = [{
    Returns the same periodic cell. The selected rectangular reciprocal
    chain requires finite positive diagonal lengths and zero tilts.
    Reference materialization checks those conditions and fails by name
    otherwise. This operation is not speculatable and is not a certificate
    that permits a fused or resident execution strategy. It changes no
    cell and does not select a default route.
  }];
  let arguments = (ins MD_CellType:$input);
  let results = (outs MD_CellType:$result);
  let assemblyFormat = "$input attr-dict";
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

def MD_KindPairsOp : MD_Op<"kind_pairs", [
    Pure, AttrSizedOperandSegments, DeclareOpInterfaceMethods<SymbolUserOpInterface>
  ]> {
  let summary = "Divides the pairs of rigid kinds and the rest into segments";
  let description = [{
    Rule R3 (D[kind-pairs]). The operands `kinds` are $K$ rigid kinds of
    D403, numbered $0, \dots, K-1$ in operand order, each the disjoint
    ordered tuples of its copies' particles in place order, described by
    the `#md.rigid_kind` of the same position in `descriptors`. The rest,
    the particles of no taken kind, is kind $K$; each of its particles is
    a molecule of one member. The scalar i32 field `universe` supplies the
    particle set; its values are not read.

    For each pair of kinds $g \le f$ the result numbered
    $s(f, g) = f(f+1)/2 + g$ is the segment $(f, g)$: the pairs of a
    molecule of kind $f$ with a molecule of kind $g$, for $f = g$ each
    unordered pair of distinct molecules once. There are
    $(K+1)(K+2)/2$ results. A tuple of the segment holds the frame atoms of
    the molecule of $f$ in frame order, then those of the molecule of $g$:
    its arity is $m_f + m_g$, with $m$ the frame size of a kind and 1 for
    the rest. It is unordered when $f = g$ and $m_f = 1$, ordered
    otherwise. The massless sites are absent from the tuples: a segment
    body forms them from the frame with the kind's weights.

    `excluded` is removed from the segment of the rest, $(K, K)$. A taken
    kind has all its own pairs excluded and none with another particle;
    the generator of the segments establishes that from D403's facts.
    `anchored` holds ordered (member, anchor) pairs of the rest, a member
    within `radius` of its anchor, as in #795; they
    select candidates only.

    The results are logical tuple domains, independent of positions and of
    the storage of lists, with no implicit cutoff. For $K = 1$ the
    segments 0, 1 and 2 are the WW, SW and SS of
    #795.
  }];
  let arguments = (ins MD_ReferenceFieldType:$universe,
                       Variadic<MD_RelationType>:$kinds,
                       TypedArrayAttrBase<MD_RigidKindAttr, "rigid kinds">:$descriptors,
                       Optional<MD_RelationType>:$excluded,
                       Optional<MD_RelationType>:$anchored,
                       OptionalAttr<F64Attr>:$radius);
  let results = (outs Variadic<MD_RelationType>:$segments);
  let assemblyFormat = [{
    $universe `,` `kinds` `(` $kinds `)` $descriptors
    (`exclude` `(` $excluded^ `)`)?
    (`anchor` `(` $anchored^ `)` `radius` `(` $radius `)`)? attr-dict
    `:` functional-type(operands, results)
  }];
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    /// The number of taken kinds; the rest is kind K.
    unsigned getNumTakenKinds() { return getKinds().size(); }
    /// The descriptor of a taken kind f < K.
    RigidKindAttr getDescriptor(unsigned f) {
      return ::llvm::cast<RigidKindAttr>(getDescriptors()[f]);
    }
  }];
}

// An image-aware reduction over the tuples of one segment.
class MD_SegmentWindowOp<string mnemonic> : MD_Op<mnemonic, [
    RecursiveMemoryEffects, SingleBlock]> {
  let arguments = (ins MD_RelationType:$relation,
                       MD_ReferenceFieldType:$positions, MD_CellType:$cell,
                       Variadic<MD_ReferenceFieldType>:$gathered,
                       I64Attr:$f, I64Attr:$g,
                       F64Attr:$cutoff,
                       UnitAttr:$site_forces);
  let results = (outs AnyType:$result);
  let regions = (region SizedRegion<1>:$kernel);
  let assemblyFormat = [{
    $relation `,` $positions `,` $cell `segment` `(` $f `,` $g `)`
    `cutoff` `(` $cutoff `)`
    (`gather` `(` $gathered^ `:` qualified(type($gathered)) `)`)?
    (`site_forces` $site_forces^)?
    $kernel attr-dict `:` qualified(type($relation)) `,`
    qualified(type($positions)) `->` qualified(type($result))
  }];
  let hasVerifier = 1;
  let hasRegionVerifier = 1;
}

def MD_SumSegmentWindowOp : MD_SegmentWindowOp<"sum_segment_window"> {
  let summary = "Sums the site pairs of a segment over covering anchor images";
  let description = [{
    The relation is the result $s(f, g)$ of `md.kind_pairs`, and
    `segment(f, g)` names it ($g \le f$). With $m_f$ and $m_g$ the frame
    sizes of the two kinds (1 for the rest), the body's arguments are
    first the anchor displacement $\mathbf x_{a_0} - \mathrm{image}(\mathbf x_{b_0})$
    in the candidate image, then the $m_f - 1$ frame offsets
    $\mathbf x_{a_k} - \mathbf x_{a_0}$ of the molecule of $f$, then the
    $m_g - 1$ of the molecule of $g$, each in its whole molecule. Each
    gathered field then supplies one value per tuple member.

    For M site pairs the body yields M energies followed by M site
    displacements. Each energy is included only where its own displacement
    has length strictly below `cutoff`. The verifier proves that every
    yielded displacement is the anchor displacement plus the coefficients
    of a place of $f$ minus those of a place of $g$ over the frame
    offsets: zero for the anchor, a unit vector for a frame atom, and for a
    site its weights $w_1, \dots, w_{m-1}$ of the kind's descriptor,
    compared as f64. A scalar result sums energies; a vector<9xf64> result
    sums the corresponding strain contributions.

    The logical reduction enumerates all covering anchor images, as in
    D242, not just the minimum image. The kinds' `extent` is a checked
    covering bound on each molecule. Only after that bound and the
    explicit site geometry are proved may lowering use a segmented pair
    list. A covering reach below the least cell edge and a cutoff below
    half the cell are required as on the ordinary group route. The relation
    has no hidden real payload.

    Force and strain rules differentiate this body's arithmetic, including
    the candidate image in the anchor displacement. They hold the piecewise
    constant membership fixed, as ordinary pair derivatives do. Higher
    derivatives and tunable parameters are not admitted by this reduction.
  }];
}

def MD_GatherSegmentWindowOp : MD_SegmentWindowOp<"gather_segment_window"> {
  let summary = "Scatters the forces of a segment's site pairs to frame atoms";
  let description = [{
    The force transpose of `md.sum_segment_window`, with the same images,
    operands, geometry and per-site cutoff. For M site pairs its body
    yields M groups of force vectors followed by M site displacements. A
    group has one force for each frame atom of the tuple, $m_f + m_g$; with
    `site_forces` it has one for each place of the two molecules, frame
    atoms and sites in place order, $n_f + n_g$ with $n$ a kind's size (1
    for the rest). Each group is included only inside its site's cutoff.

    `site_forces` is set by a radial derivative whose site pairs involve a
    site. Its slots are interaction-site forces: the constant transpose of
    the site's placement, $\mathbf F_{a_k} \mathrel{+}= w_k \mathbf F_s$
    over its weights, is applied after accumulation (rule (A) of
    #795). The result has frame atoms only.
  }];
}

def MD_TripletsOp : MD_Op<"triplets", [Pure]> {
  let summary = "Derives the triplets centered on each particle from pairs";
  let description = [{
    ```text
    T(N) = { (j, i, k) : {i, j} ∈ N, {i, k} ∈ N, j ≠ k,
             r_ij < r_3, r_ik < r_3 } / ((j, i, k) ~ (k, i, j))
    ```

    For each center `i`, each unordered pair of its neighbors within
    `cutoff` once: the triple sum of Stillinger and Weber (Phys. Rev. B 31,
    5262, 1985) and of the mW water model of Molinero and Moore (J. Phys.
    Chem. B 113, 4008, 2009), and the mode `UniqueCentralParticle` of
    OpenMM's CustomManyParticleForce. The center is at place 1, as in an
    angle, so `angle(0, 1, 2)` is the angle at the center. The far leg
    `j`–`k` is not cut.

    ```mlir
    %n = md.neighborhood %x, %cell cutoff(0.43065)
           : !md.field<@atoms, 3 x f64> -> !md.relation<@atoms, 2, unordered>
    %t = md.triplets %n cutoff(0.43065)
           : !md.relation<@atoms, 2, unordered>
             -> !md.relation<@atoms, 3, reversal>
    ```

    The cutoff is at most that of the neighborhood. The relation is found
    at the positions of the neighborhood and has no tuple set, so a sum
    over it takes no fields of tuples. It changes with the positions, so
    a kernel summed over it must vanish at the cutoff, as the
    Stillinger-Weber form does with all its derivatives, for the forces
    to be the gradient of the energy.
  }];
  let arguments = (ins
    MD_RelationType:$neighbors,
    F64Attr:$cutoff
  );
  let results = (outs MD_RelationType:$result);
  let assemblyFormat = [{
    $neighbors `cutoff` `(` $cutoff `)` attr-dict
    `:` qualified(type($neighbors)) `->` qualified(type($result))
  }];
  let hasVerifier = 1;
}

def MD_EdgesOp : MD_Op<"edges", [Pure]> {
  let summary = "Derives the pairs of images within a cutoff from pairs";
  let description = [{
    ```text
    E(N, x, h) = { (i, j, s) : (j, s) ≠ (i, 0), |x_j − x_i + s H| < r_c }
    ```

    The list that a local learned potential takes (D277,
    docs/metatomic-ml1.md, Section 12): every image of every particle
    within the cutoff of each particle, the particle's own images
    included, in both directions, grouped by `i` in the order of the
    particles. The neighborhood gives the candidates: a particle with any
    image within the cutoff of `i` is a neighbor of `i` in it, by its
    nearest image; the images are enumerated from there, so the cell may be
    narrower than twice the cutoff, or than the cutoff.

    ```mlir
    %n = md.neighborhood %x, %cell cutoff(0.7)
           : !md.field<@atoms, 3 x f64> -> !md.relation<@atoms, 2, unordered>
    %e = md.edges %n, %x, %cell cutoff(0.7)
           : !md.relation<@atoms, 2, unordered>, !md.field<@atoms, 3 x f64>
             -> !md.edges<@atoms>
    ```

    The cutoff is at most that of the neighborhood, and the positions are
    those of the neighborhood. The result is not differentiated: a host
    call that takes it differentiates the model itself (D167).
  }];
  let arguments = (ins
    MD_RelationType:$neighbors,
    MD_ReferenceFieldType:$positions,
    MD_CellType:$cell,
    F64Attr:$cutoff
  );
  let results = (outs MD_EdgesType:$result);
  let assemblyFormat = [{
    $neighbors `,` $positions `,` $cell `cutoff` `(` $cutoff `)` attr-dict
    `:` qualified(type($neighbors)) `,` qualified(type($positions))
    `->` qualified(type($result))
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

    The result is a floating-point number, or a vector of them for a table
    of vectors: that of the table, or the type that the kernel computes in
    once precision is assigned. After storage is assigned, the table is the
    buffer that holds it.
  }];
  let arguments = (ins
    AnyType:$table,
    Variadic<AnyTypeOf<[Index, AnySignlessInteger]>>:$indices
  );
  let results = (outs AnyTypeOf<[AnyFloat, VectorOfRankAndType<[1], [AnyFloat]>]>:$result);
  let assemblyFormat = [{
    $table `[` $indices `]` attr-dict `:` qualified(type($table)) `,`
    type($indices) `->` type($result)
  }];
  let hasVerifier = 1;
}

def MD_ChebyshevOp : MD_Op<"chebyshev", [Pure, SameOperandsAndResultType]> {
  let summary = "A Chebyshev series";
  let description = [{
    The value $\sum_k c_k T_k(y)$ of the Chebyshev polynomials of the first
    kind at `y`, which should lie in $[-1, 1]$, with the coefficients
    `coefficients` from $c_0$ upwards: a smooth function that a kernel
    computes of its arguments, such as the mean of the splitting kernel of
    the prolate kernel of particle mesh Ewald (D266) over $[x, 1]$ in
    $t = 2x - 1$, $x = r / r_c$ (D280).

    ```mlir
    %s = md.chebyshev %y [1.0, -0.5, 0.125] : f64
    ```

    Its derivative is the series of the derivatives. In a loop over pairs
    under `fast_math`, `md-exec-simplify-distance` takes a series of the
    distance into a function of its square that `md-exec-expand-radial`
    tabulates in f32 (D94); a series that no table takes becomes Clenshaw's
    recurrence there [Clenshaw1955].
  }];
  // An array of FloatAttr, which prints every coefficient so that it reads
  // back to the same bits (a dense array of f64 prints six digits).
  let arguments = (ins AnyFloat:$y, F64ArrayAttr:$coefficients);
  let results = (outs AnyFloat:$result);
  let assemblyFormat = [{
    $y $coefficients attr-dict `:` type($result)
  }];
  let extraClassDeclaration = [{
    /// The coefficients c_0, c_1, ... as numbers.
    llvm::SmallVector<double> getSeries();
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Particle mesh Ewald
//===----------------------------------------------------------------------===//

def MD_GridStencilOp : MD_Op<"grid_stencil", [Pure]> {
  let summary = "The discrete support relation of a particle-grid window";
  let description = [{
    Associates a structural stencil relation with its explicit positions
    and cell. The result describes the periodically wrapped support
    indices, not a dense assignment matrix or stored window weights.
    Continuous operands must remain visible to structured consumers;
    differentiation of the weights is separate from integer enumeration.

    This is the semantic relation of #452. It is not selected by the
    ordinary reciprocal route. The structured consumers and their declared
    linear derivatives are defined below. An opt-in reciprocal execution
    chain follows in slice 2; unsupported continuous derivatives are named
    diagnostics until their rules are admitted.
  }];
  let arguments = (ins MD_FieldType:$positions, MD_CellType:$cell);
  let results = (outs MD_GridStencilType:$stencil);
  let assemblyFormat = [{
    $positions `,` $cell attr-dict `:` qualified(type($positions))
    `->` qualified(type($stencil))
  }];
  let hasVerifier = 1;
}

def MD_BSplineWindowOp : MD_Op<"bspline_window", [Pure]> {
  let summary = "The product of three cardinal B-spline windows";
  let description = [{
    The product of cardinal B-splines of the given order at three support
    distances. Each factor has support [0, order]. This one scalar body is
    shared by spreading and gathering. Expansion exposes arithmetic before
    scalar differentiation; periodic enumeration is not part of this body.
    The initial curvature contract is order four.
  }];
  let arguments = (ins VectorOfLengthAndType<[3], [F64]>:$distance,
                       I64Attr:$order);
  let results = (outs F64:$weight);
  let assemblyFormat = "$distance `order` `(` $order `)` attr-dict `:` type($distance)";
  let hasVerifier = 1;
}

class MD_StencilApplyOp<string mnemonic> : MD_Op<mnemonic, [
    RecursiveMemoryEffects, SingleBlock]> {
  let summary = "A complete pure particle-grid stencil reduction";
  let description = [{
    The relation enumerates every particle and its periodic support once.
    The kernel takes a three-component f64 support distance and yields
    one f64 window weight. It has no dynamic captures or effects; immutable
    literal constants may be hoisted. Multiplication
    by the data operand and addition over entries are the declared linear
    structure, outside the weight kernel. Scatter defines a fresh full
    grid, including zeros; gather defines every particle row, including
    zeros. There is no amplitude-dependent list or hidden active payload.

    Positions and cell must be the exact operands of the relation
    constructor. The data operand is the only declared linear operand.
    The transpose swaps scatter and gather and clones the same weight
    body. Position and cell derivatives require an explicit rule or a
    named diagnostic; the structural relation never proves them inactive.
    These reference operations do not select a device execution strategy.
  }];
  let arguments = (ins MD_GridStencilType:$stencil,
                       MD_ReferenceFieldType:$positions,
                       MD_CellType:$cell, AnyType:$data);
  let results = (outs AnyType:$result);
  let regions = (region SizedRegion<1>:$kernel);
  let assemblyFormat = [{
    $stencil `,` $positions `,` $cell `,` $data $kernel attr-dict `:`
    qualified(type($stencil)) `,` qualified(type($positions)) `,`
    qualified(type($data)) `->` qualified(type($result))
  }];
  let hasVerifier = 1;
  let hasRegionVerifier = 1;
}
def MD_ScatterStencilOp : MD_StencilApplyOp<"scatter_stencil">;
def MD_GatherStencilOp : MD_StencilApplyOp<"gather_stencil">;

class MD_StencilDerivativeOp<string mnemonic> : MD_Op<mnemonic, [Pure]> {
  let summary = "An explicit operand derivative request of a stencil operation";
  let description = [{
    The source is the result of a structured scatter or gather. Operand
    numbers are 0 (relation), 1 (positions), 2 (cell), and 3 (amplitudes
    or grid). The seed is a tangent for JVP and a result cotangent for VJP.
    md-differentiate-stencils expands supported linear requests and refuses
    unsupported continuous requests by operand name. A cell tangent or
    cotangent has type vector<9xf64>. No request changes ordinary PME.
  }];
  let arguments = (ins AnyType:$source, AnyType:$seed, I64Attr:$operand);
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $source `wrt` `(` $operand `)` $seed attr-dict `:`
    qualified(type($source)) `,` qualified(type($seed))
    `->` qualified(type($result))
  }];
  let hasVerifier = 1;
}
def MD_StencilJVPOp : MD_StencilDerivativeOp<"stencil_jvp">;
def MD_StencilVJPOp : MD_StencilDerivativeOp<"stencil_vjp">;

def MD_ReduceStencilOp : MD_Op<"reduce_stencil", [
    RecursiveMemoryEffects, SingleBlock, AttrSizedOperandSegments]> {
  let summary = "Reduces an explicit kernel over every entry of a grid stencil";
  let description = [{
    The explicit structured form of assignment/interpolation and their
    kernel derivatives. Every entry of the same indices-only relation is
    visited. The kernel receives the continuous support distance, the
    particle's Cartesian position, the row-cell matrix as vector<9xf64>,
    then one value of each particle and grid input. It yields an f64 scalar
    or vector. All inputs, including geometry, remain explicit.

    `destination("grid")` defines a fresh complete tensor, initialized to
    zero and summed at wrapped indices. `destination("particles")` defines
    every particle row, including zeros. `destination("sum")` is a global
    sum, particles in ascending order and the local last axis fastest.
    There are no lists, quantizers, in-place updates or hidden payloads.
    A pure straight-line kernel may contain the common B-spline body or
    its declared scalar derivatives; no second spline formula is implied.
    Higher derivatives require their own validated rules.
  }];
  let arguments = (ins MD_GridStencilType:$stencil,
                       MD_ReferenceFieldType:$positions, MD_CellType:$cell,
                       Variadic<MD_ReferenceFieldType>:$particles,
                       Variadic<AnyRankedTensor>:$grids,
                       Variadic<AnyType>:$uniforms,
                       StrAttr:$destination);
  let results = (outs AnyType:$result);
  let regions = (region AnyRegion:$kernel);
  let assemblyFormat = [{
    $stencil `,` $positions `,` $cell
    `particles` `(` $particles `)` `grids` `(` $grids `)`
    `uniforms` `(` $uniforms `)`
    `destination` `(` $destination `)` $kernel attr-dict `:`
    qualified(type($stencil)) `,` qualified(type($positions)) `,`
    `(` qualified(type($particles)) `)` `,` `(` type($grids) `)` `,`
    `(` type($uniforms) `)`
    `->` qualified(type($result))
  }];
  let hasVerifier = 1;
  let hasRegionVerifier = 1;
}

class MD_GridKernelOp<string mnemonic> : MD_Op<mnemonic, [
    RecursiveMemoryEffects, SingleBlock]> {
  let arguments = (ins Variadic<AnyType>:$inputs,
                       DenseI64ArrayAttr:$grid, StrAttr:$pairing);
  let regions = (region AnyRegion:$kernel);
  let assemblyFormat = [{
    `(` $inputs `)` `grid` `(` $grid `)`
    `pairing` `(` $pairing `)` attr-dict
    `:` `(` qualified(type($inputs)) `)` `->` qualified(type($result))
    $kernel
  }];
  let hasVerifier = 1;
  let hasRegionVerifier = 1;
}

def MD_MapGridOp : MD_GridKernelOp<"map_grid"> {
  let summary = "Maps one scalar kernel over a real grid or half-spectrum";
  let description = [{
    A pure elementwise map on a static three-dimensional domain. The
    `grid` is the full real extent. With `euclidean` pairing the domain is
    the full grid; with `real_hermitian_half` it is the stored last-axis
    half. Tensor inputs have the domain shape. Uniform f64 scalars,
    vectors and reference tables are explicit inputs, not captures. A
    table stays a table argument in semantic IR; materialization exposes
    its existing buffer and replaces kernel lookups with buffer loads.
    The kernel receives a
    vector<3xindex> index and one element per input. Complex elements are
    represented inside the kernel by vector<2xf64> (real, imaginary), so
    the existing real scalar derivative rules apply in both directions.
    It yields one scalar or vector; a vector<2xf64> may form a complex
    output tensor. Every output element is defined in fresh storage.

    The pairing describes the domain, not a certificate of Hermitian
    validity. A complex result passed to a backward FFT must independently
    satisfy its boundary-plane conjugacy contract. The selected Gaussian
    chain proves this from its rectangular real influence; arbitrary maps
    are not evidence for a fused reciprocal strategy.
  }];
  let results = (outs AnyRankedTensor:$result);
}

def MD_SumGridOp : MD_GridKernelOp<"sum_grid"> {
  let summary = "Reduces a scalar kernel under the declared grid pairing";
  let description = [{
    Uses the same domain and kernel arguments as md.map_grid and sums the
    yielded f64 scalar or vector in lexicographic order, last axis fastest.
    With real_hermitian_half, interior last-axis modes have weight two;
    zero and the even-grid Nyquist plane have weight one. There is no FFT
    normalization in this reduction. The reduction starts at zero and
    covers the whole domain. It has ideal additive algebra; an execution
    quantizer or an alternative reduction tree is not implicit.
  }];
  let results = (outs AnyType:$result);
}

class MD_GridDerivativeOp<string mnemonic> : MD_Op<mnemonic, [Pure]> {
  let summary = "An explicit derivative request for a structured grid kernel";
  let arguments = (ins AnyType:$source, AnyType:$seed, I64Attr:$operand);
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $source `wrt` `(` $operand `)` $seed attr-dict `:`
    qualified(type($source)) `,` qualified(type($seed))
    `->` qualified(type($result))
  }];
  let hasVerifier = 1;
}
def MD_GridJVPOp : MD_GridDerivativeOp<"grid_jvp">;
def MD_GridVJPOp : MD_GridDerivativeOp<"grid_vjp">;

def MD_LinearTransformOp : MD_Op<"linear_transform", [Pure]> {
  let summary = "A value-level linear transform with a declared adjoint";
  let description = [{
    The first kind is the unnormalized three-dimensional real FFT. Forward
    maps `tensor<K0 x K1 x K2 x T>` to
    `tensor<K0 x K1 x (K2/2+1) x complex<T>>`, where `T` is f32 or f64;
    backward maps the latter to the former. The half-spectrum represents a real grid and obeys
    Hermitian symmetry on its zero and, for even K2, Nyquist planes.
    Forward and backward are adjoints under the Euclidean pairing on the
    real grid and the Hermitian-weighted real pairing on the half-spectrum.
    Neither direction normalizes by the grid size. The `adjoint` and
    `pairing` attributes make this contract inspectable before storage is
    assigned. D319.

    ```mlir
    %s = md.linear_transform %grid kind("fft3d") direction("forward")
      adjoint("backward") pairing("real_hermitian_half") :
      tensor<4x4x6xf64> -> tensor<4x4x4xcomplex<f64>>
    ```
  }];
  let arguments = (ins AnyRankedTensor:$input, StrAttr:$kind,
                       StrAttr:$direction, StrAttr:$adjoint, StrAttr:$pairing);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = [{
    $input `kind` `(` $kind `)` `direction` `(` $direction `)`
    `adjoint` `(` $adjoint `)` `pairing` `(` $pairing `)` attr-dict
    `:` type($input) `->` type($result)
  }];
  let hasVerifier = 1;
}

def MD_ReciprocalOp : MD_Op<"reciprocal", [Pure]> {
  let summary = "The reciprocal sum of particle mesh Ewald";
  let description = [{
    ```text
    E = f / (2π V) Σ_{m ≠ 0} exp(−π² m² / β²) / m² · |S(m)|²
    S(m) = Σ_i q_i exp(2π i m · x_i)
    ```

    The sum of smooth particle mesh Ewald over a grid of `grid` points with
    B-splines of the order `order` (docs/pme.md), with `coulomb` the
    Coulomb constant f. `moduli` holds the factors of the influence function
    along each edge, which do not depend on the cell: row `a` holds
    `|b_a(k)|²` for each index `k` of edge `a`, times the factor of the
    aliasing if it is taken. The influence function is
    `f / (π V) · B(m) exp(−π² m² / β²) / m²` with `B` the product of the
    three, computed from the cell where it is used.

    With `dispersion`, the op is the sum of the dispersion −c_i c_j / r⁶
    instead (D162, [Essmann1995]), the charges are the coefficients c_i
    of the particles, and `coulomb` scales the energy:

    ```text
    E = −f π^{3/2} β³ / (6V) Σ_m F(π |m| / β) |S(m)|²
    F(b) = (1 − 2b²) exp(−b²) + 2 √π b³ erfc(b)
    ```

    with the term m = 0, F(0) = 1, included.

    With `prolate`, the splitting and the spreading take the prolate
    spheroidal wave function of order zero (D266,
    [Liang2026]): `order` is the number P of points of the window,
    `beta` is not used, and `moduli` has five rows: the factors
    1/φ̂₁(2πk/K)² of each edge, the Chebyshev series of the weights of
    the window and of their slopes, and the scalars and series of the
    influence (Builder::collectProlate):

    ```text
    E = f / (2π V) Σ_{m ≠ 0} χ̂(2π r_c |m|) / m² · |S(m)|²
    ```

    With `p3m`, the influence function is the one optimal for the forces
    of analytic differentiation, eq. 2.28 of [Ballenegger2012], with the
    Gaussian in a rectangular cell (D304, docs/pme.md,
    Section 1.1): `moduli` has 3 + 3 (2J + 1) rows, row `a` holds
    `1 / S_a(k)` with `S_a(k) = Σ_j Û²(ν_j)`, and row `3 + a (2J + 1) + j
    + J` the weight `u_j = Û²(ν_j) / S_a(k)` of the alias `ν_j = n + j
    K_a` of index `k` (signed `n`), `|j| ≤ J`, `Û(t) = (sin πt / πt)ⁿ`.
    J is 2: 18 rows, which the kernels of a device take as a constant.
    With `μ_j = ν_j / L_a` and `g_j = exp(−π² μ_j² / β²)`:

    ```text
    G(m) = f / (π V) · Π_a (Σ_j u_j g_j / S_a) / Σ_a Σ_j u_j μ_j²
    ```

    The op yields the energy, the forces, and the virial `Σ d ⊗ F` of the
    sum. Differentiation takes the forces and the virial from the op; it
    does not differentiate through it.

    With a fourth result, the op also yields the derivative of the energy
    in the charge of each particle, the potential of the grid at the
    particle, `∂E/∂q_i = Σ_k θ_i(k) φ(k)`, with θ_i the weights of the
    B-splines of the particle and φ the grid after the product with the
    influence function: the energy is `½ Σ_i q_i ∂E/∂q_i`. The
    differentiation in the field of the charges asks for it
    (D231); no other op has it.

    ```mlir
    %u, %f, %w, %p = md.reciprocal %x, %q, %cell, %moduli
                       grid([32, 32, 32]) order(4) beta(3.12) coulomb(138.935)
                       : !md.field<@atoms, 3 x f64>, !md.field<@atoms, f64>,
                         !md.table<2, f64>
                       -> f64, !md.field<@atoms, 3 x f64>, vector<9xf64>,
                          !md.field<@atoms, f64>
    ```

    ```mlir
    %u, %f, %w = md.reciprocal %x, %q, %cell, %moduli
                   grid([32, 32, 32]) order(4) beta(3.12) coulomb(138.935)
                   : !md.field<@atoms, 3 x f64>, !md.field<@atoms, f64>,
                     !md.table<2, f64>
                   -> f64, !md.field<@atoms, 3 x f64>, vector<9xf64>
    ```

    With `listed`, a field of `i32` whose first entries are the places of
    the particles whose charge can be nonzero, in ascending order, and
    whose other entries are −1 − the place of each of the others
    (`md_exec.nonzero`), the sum spreads and gathers those particles only
    (D262): the others have a charge of zero, add
    nothing to the grid, and take a force of zero. The potential at the
    particles is still gathered at every particle. With `gather_all` as
    well, a device spreads the listed particles and gathers every one, as
    without a list, which gives the others their force of zero by the
    product with their charge: where most particles are listed, the
    others share the points of the grid of their neighbors in a warp, and
    gathering them costs less than the list does.

    ```mlir
    %u, %f, %w = md.reciprocal %x, %q, %cell, %moduli listed(%l)
                   grid([32, 32, 32]) order(4) beta(3.12) coulomb(138.935)
                   : !md.field<@atoms, 3 x f64>, !md.field<@atoms, f64>,
                     !md.table<2, f64>, !md.field<@atoms, i32>
                   -> f64, !md.field<@atoms, 3 x f64>, vector<9xf64>
    ```
  }];
  let arguments = (ins
    MD_ReferenceFieldType:$positions,
    MD_ReferenceFieldType:$charges,
    MD_CellType:$cell,
    MD_TableType:$moduli,
    Optional<MD_ReferenceFieldType>:$listed,
    DenseI64ArrayAttr:$grid,
    I64Attr:$order,
    F64Attr:$beta,
    F64Attr:$coulomb,
    UnitAttr:$dispersion,
    UnitAttr:$prolate,
    UnitAttr:$p3m,
    UnitAttr:$gather_all
  );
  let results = (outs
    F64:$energy,
    MD_ReferenceFieldType:$forces,
    VectorOfLengthAndType<[9], [F64]>:$virial,
    Optional<MD_ReferenceFieldType>:$potential
  );
  let assemblyFormat = [{
    $positions `,` $charges `,` $cell `,` $moduli
    (`listed` `(` $listed^ `)`)?
    `grid` `(` $grid `)` `order` `(` $order `)` `beta` `(` $beta `)`
    `coulomb` `(` $coulomb `)` (`dispersion` $dispersion^)?
    (`prolate` $prolate^)?
    (`p3m` $p3m^)?
    (`gather_all` $gather_all^)?
    attr-dict `:` qualified(type($positions)) `,` qualified(type($charges))
    `,` qualified(type($moduli)) (`,` qualified(type($listed))^)?
    `->` type($energy) `,`
    qualified(type($forces)) `,` type($virial)
    (`,` qualified(type($potential))^)?
  }];
  let hasVerifier = 1;
}

def MD_YieldOp : MD_Op<"yield", [
    Pure, Terminator, ReturnLike,
    ParentOneOf<["SumSegmentWindowOp", "GatherSegmentWindowOp",
                 "ScatterStencilOp", "GatherStencilOp", "ReduceStencilOp", "MapGridOp", "SumGridOp",
                 "SumRelationOp", "GatherRelationOp", "SumTuplesOp",
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

    With several results, the kernel yields one value for each, and each
    result is the sum of its values: several fields gathered over the
    same pairs in one evaluation of the kernel. The derivative of a stage
    kernel takes the adjoints of the fields, the forces, and the virial
    together this way (D288). A gather of several fields has
    no exchange contract, `exchange(none)`.

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
  let results = (outs Variadic<MD_ReferenceFieldType>:$results);
  let assemblyFormat = [{
    $relation `,` $positions `,` $cell
    (`gather` `(` $gathered^ `:` qualified(type($gathered)) `)`)?
    custom<Exchange>($exchange, $exchange_basis)
    $kernel attr-dict
    `:` qualified(type($relation)) `,` qualified(type($positions)) `->` qualified(type($results))
  }];
  let extraClassDeclaration = [{
    /// The first result, the only one of a gather of one field.
    ::mlir::Value getResult() { return getResults().front(); }
  }];
  let hasVerifier = 1;
  let hasRegionVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Tuples of a topology
//===----------------------------------------------------------------------===//

class MD_TupleOp<string mnemonic, list<Trait> traits = []>
    : MD_Op<mnemonic, !listconcat([
        AttrSizedOperandSegments, RecursiveMemoryEffects, SingleBlock
      ], traits)> {
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

    A result of the type `i1` tells instead whether the kernel yields true
    for some tuple (`any_{t ∈ T} k(t)`).

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

// The results keep the numbers of the printer, not the names of their
// declaration, which an op of more than one result would print.
def MD_GatherTuplesOp : MD_TupleOp<"gather_tuples", [
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
  ]> {
  let summary = "Sums a kernel, per particle, over the tuples it is in";
  let description = [{
    ```text
    a_i = Σ_{t, s : t[s] = i} k(t)[s]
    ```

    The kernel yields one value for each member of the tuple, in the order
    of the members. A particle receives, from every tuple that it is a
    member of, the value of its place.

    The kernel receives what that of `md.sum_tuples` receives.

    With a second result, of the type `i1`, the kernel yields one more
    value, an `i1`, after those of the members, and the result tells
    whether it is true for some tuple: the constraints of a step test the
    lengths that they give in their own kernel (D278).

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
  let results = (outs MD_ReferenceFieldType:$result, Optional<I1>:$failed);
  let assemblyFormat = [{
    $relation `,` $positions `,` $cell
    custom<Coordinates>($coordinate_kinds, $coordinate_members)
    (`gather` `(` $gathered^ `:` qualified(type($gathered)) `)`)?
    (`tuple` `(` $parameters^ `:` qualified(type($parameters)) `)`)?
    $kernel attr-dict
    `:` qualified(type($relation)) `,` qualified(type($positions)) `->` qualified(type($result)) (`,` type($failed)^)?
  }];
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
// Neural stages (D288, docs/ml4-ir.md)
//===----------------------------------------------------------------------===//

def MD_StageOp : MD_Op<"stage", [
    Symbol, DeclareOpInterfaceMethods<SymbolUserOpInterface>
  ]> {
  let summary = "Declares a neural stage and binds its bodies";
  let description = [{
    Declares a pure function of one element, one directed pair or one
    particle, that a kernel applies with `md.apply_stage`: a stage of a
    learned potential whose body is external to MDIR's semantics (D167).

    ```mlir
    md.stage @edge1 forward(@edge1.f) vjp(@edge1.b)
        : (f64, vector<3xf64>, vector<16xf64>, vector<16xf64>, i32, i32)
          -> vector<16xf64>
    ```

    Each operand and result is the value of one element: `f64`, a vector
    of `f64`, or for an operand an integer (`i32`, `i64`). Operands of
    floating type are differentiable; integer operands, such as species,
    are structural and take no cotangent.

    `forward` names the body, a function of the type of the declaration.
    `vjp` names its vector-Jacobian product: a function of the operands,
    then one cotangent for each result, that returns one cotangent for
    each differentiable operand, in order. The declaration states that the
    bodies are pure and that the VJP is the transpose of the derivative of
    the forward body at the same operands. A stage without `vjp` cannot be
    differentiated in an operand that is active.

    The form of a body is the form of the function that the symbol names.
    The form that is lowered is a body in the module, a `func.func` of
    `arith`, `math`, and `vector` ops, which `md-inline` inlines into the
    kernels. An external declaration (an AOTInductor export) and a body of
    tensors with a batch dimension (StableHLO) are the forms of step 4 of
    issue #377 and are refused here.
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    FlatSymbolRefAttr:$forward,
    OptionalAttr<FlatSymbolRefAttr>:$vjp
  );
  let assemblyFormat = [{
    $sym_name `forward` `(` $forward `)` (`vjp` `(` $vjp^ `)`)? attr-dict
    `:` $function_type
  }];
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    /// Whether operand `index` of the stage takes a cotangent.
    bool isDifferentiable(unsigned index);
    /// The type of the vector-Jacobian product: the operands and a
    /// cotangent for each result, to a cotangent for each differentiable
    /// operand.
    ::mlir::FunctionType getVjpType();
    /// The rows of its batched bodies (D293): the leading
    /// dimension of their tensors, or 0 if its bodies are functions of one
    /// element in the module, which `md-inline` inlines.
    int64_t getRows();
  }];
}

def MD_ApplyStageOp : MD_Op<"apply_stage", [
    Pure, DeclareOpInterfaceMethods<SymbolUserOpInterface>
  ]> {
  let summary = "Applies a neural stage to one element in a kernel";
  let description = [{
    Applies the stage `stage` to the values of one element inside a kernel:
    one directed pair of `md.gather_relation`, one particle of
    `md.map_particles` or `md.sum_particles`.

    ```mlir
    %m = md.apply_stage @edge1(%r, %d, %hi, %hj, %zi, %zj)
        : (f64, vector<3xf64>, vector<16xf64>, vector<16xf64>, i32, i32)
          -> vector<16xf64>
    ```

    With `vjp`, it applies the vector-Jacobian product of the stage
    instead: its operands are those of the stage followed by a cotangent
    of each result, and its results the cotangents of the differentiable
    operands. `md-differentiate` emits this form; it has no derivative of
    its own.

    ```mlir
    %rb, %db, %hib, %hjb = md.apply_stage @edge1 vjp(%r, %d, %hi, %hj, %zi,
                                                     %zj, %mb)
        : (f64, vector<3xf64>, vector<16xf64>, vector<16xf64>, i32, i32,
           vector<16xf64>)
          -> (f64, vector<3xf64>, vector<16xf64>, vector<16xf64>)
    ```

    The semantics are those of one element; how many elements a body takes
    at once is an execution choice.
  }];
  let arguments = (ins
    FlatSymbolRefAttr:$stage,
    UnitAttr:$vjp,
    Variadic<AnyType>:$operands
  );
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = [{
    $stage (`vjp` $vjp^)? `(` $operands `)` attr-dict
    `:` functional-type($operands, results)
  }];
  let hasVerifier = 1;
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
    | `derivative(n)` | Type of the field | `g_i = ∂U/∂a_i`, where `a` is argument `n`, a field of the particles with one component of `f64` |
    | `tangent` | `f64` | `Φ = Σ_i t_i · ∂U/∂x_i`, the derivative of `U` along the tangent `t` of the positions |
    | `tangent_derivative(n)` | as `derivative(n)` | `∂Φ/∂θ_n` or `∂Φ/∂a_i`: the mixed derivative `Σ_i t_i · ∂²U/∂x_i ∂θ_n` (D376) |
    | `tangent_forces` | Type of the position field | `−∂Φ/∂x_i`: minus the Hessian of `U` times the tangent, `−Σ_j ∂²U/∂x_i ∂x_j · t_j` (D383) |
    | `direction` | `f64` | `Ψ = Σ_n ḃ_n ∂U/∂b_n + Σ_i ȧ_i ∂U/∂a_i`, the derivative of `U` along directions of the arguments named by `directions` (D389) |
    | `direction_forces` | Type of the position field | `−∂Ψ/∂x_i`: minus the mixed derivative `Σ_n ∂²U/∂x_i ∂b_n · ḃ_n`, forward mode in the arguments of the gradient in the positions |

    With a `tangent` request, the op takes one operand more than the
    potential, the last: the tangent `t`, a field of the type of the
    positions. The tangent is held fixed: `tangent_derivative(n)` is the
    derivative of `Φ` in argument `n` of the potential at fixed `t`.

    With a `direction` request, the attribute `directions` names arguments
    of the potential, in increasing order, each a number (`f64`) or a field
    of the particles with one component of `f64`, and the op takes one
    operand more for each, after the arguments of the potential: its
    direction, of the type of the argument. The requests of a direction and
    those of the tangent are not mixed in one op.

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
    DenseI64ArrayAttr:$request_arguments,
    OptionalAttr<DenseI64ArrayAttr>:$directions
  );
  let results = (outs Variadic<AnyType>);
  // The builders without directions, as before they were added.
  let builders = [
    OpBuilder<(ins "::mlir::TypeRange":$results,
                   "::mlir::FlatSymbolRefAttr":$callee,
                   "::mlir::ValueRange":$operands,
                   "::mlir::DenseI32ArrayAttr":$request_kinds,
                   "::mlir::DenseI64ArrayAttr":$request_arguments), [{
      build($_builder, $_state, results, callee, operands, request_kinds,
            request_arguments, ::mlir::DenseI64ArrayAttr());
    }]>,
    OpBuilder<(ins "::mlir::TypeRange":$results, "::llvm::StringRef":$callee,
                   "::mlir::ValueRange":$operands,
                   "::llvm::ArrayRef<int32_t>":$request_kinds,
                   "::llvm::ArrayRef<int64_t>":$request_arguments), [{
      build($_builder, $_state, results,
            ::mlir::FlatSymbolRefAttr::get($_builder.getContext(), callee),
            operands, $_builder.getDenseI32ArrayAttr(request_kinds),
            $_builder.getDenseI64ArrayAttr(request_arguments),
            ::mlir::DenseI64ArrayAttr());
    }]>
  ];
  let assemblyFormat = [{
    $callee `(` $operands `)`
    `request` custom<Requests>($request_kinds, $request_arguments)
    attr-dict `:` functional-type($operands, results)
  }];
  let extraClassDeclaration = [{
    /// Whether a request is `tangent` or `tangent_derivative`, so that the
    /// last operand is the tangent of the positions.
    bool hasTangent();
    /// Whether a request is `direction` or `direction_forces`, so that the
    /// operands end with the directions of the arguments `directions`.
    bool hasDirection();
    /// The operands that are the arguments of the potential.
    ::mlir::OperandRange getPotentialOperands();
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
