# ML4 IR: neural stages, feature fields, and their derivative

Step 3 of issue #377 (`D288`, `D289`): the IR in
which a message-passing model (PaiNN, MACE) is written as stages of the
`md` dialect, so that MDIR differentiates, schedules, and later
distributes it. The particle structure (the relation, the gathers and sums
over it, the maps over particles) is stated in `md`; only the neural
bodies of the stages are external. External AD generates the backward of
the bodies (D167); `md-differentiate` derives the rest from the structure.

Status: **built** (PR #441), as ruled by the maintainer on 2026-10-19
(Section 7). Steps 4 and 5 (PaiNN and MACE in stages) lower these ops onto
real artifacts; step 6 distributes them. Step 4a is **built**
(Section 8).

## 1. A model in stages

A layer of a message-passing model is a sum over the neighbors of each
atom of a per-edge function, followed by a per-node update:

$$
\mathbf a_i^{(k)} = \sum_{j : (i,j) \in D(N)} \phi_k\big(r_{ij}, \mathbf d_{ij}, \mathbf h_i^{(k-1)}, \mathbf h_j^{(k-1)}, z_i, z_j\big),
\qquad
\mathbf h_i^{(k)} = \psi_k\big(\mathbf h_i^{(k-1)}, \mathbf a_i^{(k)}\big),
$$

with $\mathbf h_i^{(0)} = \eta(z_i)$ an embedding of the species $z_i$ and
the energy a readout $U = \sum_i \rho(\mathbf h_i^{(N_\text{layer})})$
after $N_\text{layer}$ layers. $D(N)$ is the
directed expansion of the relation of pairs within the cutoff, and
$\mathbf d_{ij} = \mathbf x_i - \mathbf x_j$ in the minimum image. In the
`md` dialect each piece is an existing op:

| Piece | Op | Kernel |
|---|---|---|
| Relation | `md.neighborhood` | |
| Edge stage and sum, $\mathbf a^{(k)}$ | `md.gather_relation ... exchange(none)` | $\phi_k$ of one directed pair |
| Node stage, $\mathbf h^{(k)}$ | `md.map_particles` | $\psi_k$ of one particle |
| Embedding, $\mathbf h^{(0)}$ | `md.map_particles` of the species | $\eta$ of one particle |
| Readout, $U$ | `md.sum_particles` | $\rho$ of one particle |

A kernel calls its neural body with one new op, `md.apply_stage`, whose
semantics are those of one element: one directed pair, one particle. How
many elements a body takes at once (the batch) is an execution choice, the
tile of the loop that `md_exec.pair_for` or `md_exec.particle_for` becomes
(step 4); it is not in the semantics.

```mlir
!h    = !md.field<@atoms, irreps<4x0e + 4x1o>, f64>     // 16 components
!z    = !md.field<@atoms, i32>
!vec  = !md.field<@atoms, 3 x f64>
!pair = !md.relation<@atoms, 2, unordered>

md.stage @embed  forward(@embed.f) : (i32) -> vector<16xf64>
md.stage @edge1  forward(@edge1.f) vjp(@edge1.b)
    : (f64, vector<3xf64>, vector<16xf64>, vector<16xf64>, i32, i32)
      -> vector<16xf64>
md.stage @node1  forward(@node1.f) vjp(@node1.b)
    : (vector<16xf64>, vector<16xf64>) -> vector<16xf64>
// ... @edge2, @node2
md.stage @readout forward(@readout.f) vjp(@readout.b)
    : (vector<16xf64>) -> f64

md.potential @model(%x: !vec, %cell: !md.cell, %z: !z) -> f64 {
  %n  = md.neighborhood %x, %cell cutoff(0.5) : !vec -> !pair
  %h0 = md.map_particles gather(%z : !z) {
  ^bb0(%zi: i32):
    %e = md.apply_stage @embed(%zi) : (i32) -> vector<16xf64>
    md.yield %e : vector<16xf64>
  } : !h
  %a1 = md.gather_relation %n, %x, %cell gather(%h0, %z : !h, !z)
      exchange(none) {
  ^bb0(%r: f64, %d: vector<3xf64>, %hi: vector<16xf64>, %hj: vector<16xf64>,
       %zi: i32, %zj: i32):
    %m = md.apply_stage @edge1(%r, %d, %hi, %hj, %zi, %zj)
        : (f64, vector<3xf64>, vector<16xf64>, vector<16xf64>, i32, i32)
          -> vector<16xf64>
    md.yield %m : vector<16xf64>
  } : !pair, !vec -> !h
  %h1 = md.map_particles gather(%h0, %a1 : !h, !h) {
  ^bb0(%hi: vector<16xf64>, %ai: vector<16xf64>):
    %u = md.apply_stage @node1(%hi, %ai)
        : (vector<16xf64>, vector<16xf64>) -> vector<16xf64>
    md.yield %u : vector<16xf64>
  } : !h
  // ... the second layer, %a2 and %h2, alike
  %u = md.sum_particles gather(%h2 : !h) {
  ^bb0(%hi: vector<16xf64>):
    %e = md.apply_stage @readout(%hi) : (vector<16xf64>) -> f64
    md.yield %e : f64
  } : f64
  md.return %u : f64
}
```

The cutoff function of the edges belongs to the body of the edge stage: the
body must vanish with its derivatives at the cutoff for the forces to be
the gradient of the energy, as the kernel of any sum over a relation that
changes with the positions must (`md.triplets`).

## 2. The ops

### 2.1 `md.stage`: a declaration and its bindings

`md.stage` is a symbol of the module that declares a pure function of one
element and binds its bodies:

- **The signature** is that of one element: each operand and result is a
  kernel value, `f64`, `vector<N x f64>`, or `i32` (`i64`). Operands of a
  floating type are differentiable; operands of an integer type (species,
  indices) are structural and take no cotangent.
- **`forward(@f)`** names the body of the stage: a function of the type of
  the declaration.
- **`vjp(@b)`**, if present, names its vector-Jacobian product: a function
  of the operands followed by one cotangent for each result, returning one
  cotangent for each differentiable operand, in order,
  $(\bar{\mathbf u}_1, \dots) = (\partial f/\partial \mathbf u)^\mathsf T \bar{\mathbf y}$.
  A stage without `vjp` cannot be differentiated in a differentiable
  operand that is active; one whose operands are all structural (the
  embedding) needs none.
- **Effects**: none. The declaration states that the bodies are pure and
  that the VJP is the transpose of the derivative of the forward body at
  the same operands; the verifier checks what it can (Section 2.3) and the
  adjoint identity checks the rest (Section 6).

The *form* of a body is the form of the function that the symbol names;
the declaration is the same for all forms:

| Form | The symbol names | Lowering |
|---|---|---|
| A body in the module (step 3) | a `func.func` of `arith`, `math`, and `vector` ops over the element's values | `md-inline` inlines it into the kernel before `md_exec`, so it lowers on the CPU and on a device as any kernel does |
| StableHLO, legalized to `linalg` (the rulings of 2026-10-17 and of 2026-10-19 on #447; step 4, D293) | a `func.func` of `linalg` on tensors with a static leading dimension, the rows of a tile: `jax.export` of the stage and of its `jax.vjp` to StableHLO, legalized by StableHLO's `stablehlo-legalize-to-linalg` when the model is exported, so that MDIR's build takes no dependency on StableHLO; a body may compute in f32 where the declaration says f64 | the tile of the loop calls it with the tile's rows; `md-exec-lower-stage-bodies` lowers it with MDIR's pipeline (Section 8.3) |
| AOTInductor (step 4) | a `func.func private` declaration with `md.aoti = {package = "...", entry = "..."}`, batched as the StableHLO form | a call of the package's entry with the tile's rows, as the model term calls one today (D273) |

A batched body takes `tensor<T x ...>` for each value of the declaration,
`T` static and the same for every value and for both bodies of a stage:
`f64` as `tensor<T x E>`, `vector<N x f64>` as `tensor<T x N x E>`, with
`E` f32 or f64, and an integer as `tensor<T x iK>`. Step 3 implemented the
first form; step 4a adds the second (Section 8); an external declaration
(AOTInductor, step 4b) is still refused with a diagnostic.

### 2.2 `md.apply_stage`: one element

```mlir
%m = md.apply_stage @edge1(%r, %d, %hi, %hj, %zi, %zj) : (...) -> vector<16xf64>
%rb, %db, %hib, %hjb = md.apply_stage @edge1 vjp(%r, %d, %hi, %hj, %zi, %zj, %mb)
    : (...) -> (f64, vector<3xf64>, vector<16xf64>, vector<16xf64>)
```

The op applies a stage, or with `vjp` its vector-Jacobian product (the
operands of the stage, then a cotangent of each result), to the values of
one element inside a kernel of `md.gather_relation`,
`md.map_particles`, or `md.sum_particles`. It is `Pure`. Its verifier
checks the operand and result types against the declaration
(`SymbolUserOpInterface`), as `func.call` against `func.func`.

It implements `DerivativeOpInterface` (D183) in reverse: the interface
gains `emitAdjoint`, which emits the cotangents of the operands from those
of the results, and `md.apply_stage` emits it as the `vjp` form of itself.
Its forward rule (`emitDerivative`, a tangent) fails with a diagnostic: no
JVP is bound, and MDIR writes no neural derivative (D167). Activity
(D182) and `md-check-derivative-coverage` treat it as a modeled op: its
floating operands are differentiable, its integer operands structural.
The `vjp` form has no rule of its own: a second derivative of a stage is
refused.

### 2.3 What the verifiers check

- `md.stage`: the symbols resolve to `func.func`; the forward body has the
  declaration's type, the VJP the type derived from it (operands, then a
  cotangent per result, to a cotangent per differentiable operand); a body
  in the module contains no op with memory effects; every value is of
  `f64`, a vector of it, or an integer.
- `md.apply_stage`: the declaration exists; operand and result types match
  it, with `vjp`, the types of the VJP; the op sits in a kernel of the ops
  above.
- `md-differentiate` (Section 4) relies on one more fact, the shape of a
  *stage kernel*: the kernel yields the result of `md.apply_stage`
  directly, and every operand of the op that is active in the positions is
  an argument of the kernel. It checks this where it differentiates and
  refuses another shape with a diagnostic. Arithmetic around the call (an
  envelope times the output) would need reverse rules for `arith`; the
  envelope belongs to the body instead. The restriction can be lifted
  later by reverse rules of the scalar ops, without a change of the IR.

## 3. Feature fields

### 3.1 A feature dimension on `!md.field`

A field of features is a field of `!md.field` with more components; the
type states their layout:

```mlir
!md.field<@atoms, 16 x f64>                          // 16 components, no symmetry stated
!md.field<@atoms, irreps<4x0e + 4x1o>, f64>          // 4 scalars, then 4 vectors
!md.field<@atoms, irreps<128x0e + 128x1o, ir_mul>, f32>   // PaiNN's layout
```

- `N x f64` lifts the restriction of the type to 1 or 3 components; a
  field of several components has a floating element type. A
  kernel takes one value of the field as `vector<N x f64>`
  (`getKernelValueType`), and the storage form is `memref<? x N x f64>`, as
  for 3 components; storage, the pools, precision assignment, and
  `mdrt.from_buffer` are already written for any $N$.
- `irreps<...>` states the representation of the rotation group $O(3)$ that
  the components carry: a sum of blocks `M x l p`, $M$ copies of the
  irreducible representation of degree $l$ (dimension $2l + 1$) and parity
  `e` (even) or `o` (odd), with $N = \sum M (2l+1)$, $l \le 8$. The
  notation is e3nn's [[Geiger2022]](references.md#geiger2022). The layout of the components of one
  particle is block after block in the order written (the verifier
  checks $1 \le M$, $0 \le l \le 8$, and $N$ against the blocks; a block
  is written as e3nn writes it, `4x1o`), and within a block
  either `mul_ir` (the default; copy-major, the $2l+1$ components of a copy
  together; e3nn's and MACE's layout) or `ir_mul` (component-major, the $M$
  copies of a component together; SchNetPack's PaiNN, whose vectors are
  `[3, F]` per atom). Within a copy the components are those of the real
  spherical harmonics of e3nn 0.4 in its order and normalization; for
  $l = 1$ that order is $(x, y, z)$, the Cartesian components of a vector.
- PaiNN's features are `irreps<F x 0e + F x 1o, ir_mul>` (its scalars $s$
  and vectors $\mathbf v$); MACE-OFF23 medium's hidden features
  `irreps<128x0e + 128x1o>`; the spherical harmonics of an edge to
  $\ell_\text{max} = 3$ are values inside the edge stage, not fields.

The symmetry is a fact the type states for whoever relies on it (a check
of equivariance, a lowering that chooses a layout for a device, a
conversion between the two layouts); no pass of step 3 relies on it, and
differentiation treats a field of $N$ components alike whatever its
irreps. Two fields of the same $N$ but different irreps are different
types, so a stage cannot take features of the wrong layout without a
conversion that says so.

### 3.2 Why not a type of its own

A separate type (`!md.features<...>`) would have to be accepted by every op
that takes a field: the gathers and sums over relations and particles, the
maps, `mdrt.from_buffer` and `to_buffer`, storage and its pools, precision
assignment, and `md_dist`'s `halo_copy` and `halo_sum`, which step 6 needs
for features as for positions. None of them treats features otherwise than
a field of $N$ components. The layout is a parameter of the field type, so
that it is in the type (principle 1) without a second type.

## 4. The derivative

`md-differentiate` keeps, for each field that the potential computes from
the positions, the derivative of the energy in it, its adjoint
$\bar{\mathbf h}_i = \partial U/\partial \mathbf h_i$ (D143), now a field of
the type of the field. It fills the adjoints from the last field to the
first, and turns the adjoint of each gather into forces and a virial.

### 4.1 Rules

Write $\mathrm{vjp}_\phi(i, j; \bar{\mathbf y}) = (\bar r, \bar{\mathbf d},
\bar{\mathbf h}^\mathrm c, \bar{\mathbf h}^\mathrm n)$ for the VJP of the
edge stage at the directed pair $(i, j)$ with the cotangent
$\bar{\mathbf y}$: the cotangents of the distance, the displacement, the
center's features, and the neighbor's.

| Op of the potential | Adjoint terms |
|---|---|
| Readout $U \mathrel{+}= w \sum_i \rho(\mathbf h_i)$ | $\bar{\mathbf h}_i \mathrel{+}= \mathrm{vjp}_\rho(\mathbf h_i; w)$: a map |
| Node $\mathbf h'_i = \psi(\mathbf h_i, \mathbf a_i)$ | $(\bar{\mathbf h}_i, \bar{\mathbf a}_i) \mathrel{+}= \mathrm{vjp}_\psi(\mathbf h_i, \mathbf a_i; \bar{\mathbf h}'_i)$: a map |
| Edge $\mathbf a_i = \sum_j \phi(i, j)$ | $\bar{\mathbf h}_m \mathrel{+}= \sum_b \big[\bar{\mathbf h}^\mathrm c(m, b; \bar{\mathbf a}_m) + \bar{\mathbf h}^\mathrm n(b, m; \bar{\mathbf a}_b)\big]$: a gather over the same relation that reads $\bar{\mathbf a}$ at both ends |
| | $\mathbf F_m \mathrel{+}= -\sum_b \big[\mathbf g(m, b) - \mathbf g(b, m)\big]$, $\mathbf g(i, j) = \bar r(i,j)\,\mathbf d_{ij}/r_{ij} + \bar{\mathbf d}(i,j)$: a gather, antisymmetric by construction |
| | $\mathsf W \mathrel{+}= \sum_{\{i,j\}} \mathbf d_{ij} \otimes \mathbf K(i,j)$, $\mathbf K$ the term of the forces: a sum over the relation |

The adjoint of a gather over a relation is a gather over the same
relation, and the adjoint of a sum a map: the rules of the structure, with
the VJP of the bodies at the stage boundary. Only the uses of a field by
the ops of the potential enter its adjoint: the adjoints of the later
fields, ops of the derivative, read it as well.

### 4.2 Derivation

$U$ depends on the positions only through the pairs: $\mathbf h^{(0)}$ is a
function of the species, and each later field of the fields before it and
of $(r_{ij}, \mathbf d_{ij})$ of the pairs. Take the energy as a function of
the values of one layer's gather, $\mathbf a_i = \sum_j \phi(i,j)$, with
$\bar{\mathbf a}_i = \partial U/\partial \mathbf a_i$ already known (the
fields after it come later in the body and were done first). Its
first-order change is

$$
\delta U = \sum_i \bar{\mathbf a}_i \cdot \delta \mathbf a_i
= \sum_{(i,j) \in D(N)} \Big[\bar r\,\delta r_{ij} + \bar{\mathbf d}\cdot\delta\mathbf d_{ij}
+ \bar{\mathbf h}^\mathrm c\cdot\delta\mathbf h_i + \bar{\mathbf h}^\mathrm n\cdot\delta\mathbf h_j\Big](i, j; \bar{\mathbf a}_i),
$$

by the definition of the VJP. *Fields.* The terms in $\delta\mathbf h$,
collected at a particle $m$, come from the pairs $(m, b)$, where $m$ is the
center, and $(b, m)$, where it is the neighbor: the row of the table. Both
are pairs of the relation at $m$, so the sum is a gather at $m$ that reads
$\bar{\mathbf a}$ at both ends. *Forces.* With $\delta r_{ij} = \mathbf
d_{ij}\cdot\delta\mathbf d_{ij}/r_{ij}$ and $\delta\mathbf d_{ij} =
\delta\mathbf x_i - \delta\mathbf x_j$, the geometric terms of the pair
$(i,j)$ are $\mathbf g(i,j)\cdot(\delta\mathbf x_i - \delta\mathbf x_j)$;
collected at $m$ they give $\partial U/\partial\mathbf x_m = \sum_b
[\mathbf g(m,b) - \mathbf g(b,m)]$, and $\mathbf K(m, b) = -[\mathbf g(m,b)
- \mathbf g(b,m)] = -\mathbf K(b, m)$. *Virial.* Under the strain $\mathbf
x \to (\mathsf 1 + \boldsymbol\varepsilon)\mathbf x$ with the cell, $\delta
\mathbf d_{ij} = \boldsymbol\varepsilon\,\mathbf d_{ij}$ for every pair and
the species do not change, so $-\partial U/\partial\varepsilon_{ab} =
-\sum_{(i,j)} g_b(i,j)\,d_{ij,a}$ summed over the layers' gathers. Over the
unordered pairs, with $\mathbf d_{ji} = -\mathbf d_{ij}$, that is
$\sum_{\{i,j\}} \mathbf d_{ij}\otimes\mathbf K(i,j)$, the form of the
virial of pairs [[Louwerse2006]](references.md#louwerse2006). The fields enter $\delta U$ only through
$\delta\mathbf h$, which the gather of the fields carries to the earlier
layers, where it becomes geometric terms of their pairs in turn. $\square$

These are the rules that `md-differentiate` applies to generalized Born
(D143) and to EAM on domains (D279), for fields of one component and
kernels of the distance; the VJP of a stage replaces the forward slope of a
scalar kernel, and the displacement enters through $\bar{\mathbf d}$.

### 4.3 One gather for the three terms, and the number of evaluations

The products of a stage gather give every cotangent at once: $\bar r$,
$\bar{\mathbf d}$, and the cotangents of both ends. `md-differentiate`
therefore emits, for each stage gather whose result the energy depends on,
**one** `md.gather_relation` with several results (the ruling of
2026-10-19 on #441):

- the adjoint term of each field it gathers that depends on the
  positions;
- the forces, $\mathbf K(m, b)$ summed over the neighbors $b$ of $m$, if
  forces are requested;
- the virial at each center, $\tfrac12\sum_b \mathbf d_{mb}\otimes\mathbf
  K(m, b)$, a field of 9 components, if the virial is requested, which an
  `md.sum_particles` adds into $\mathsf W$: each unordered pair is the
  pair of two centers, so the sum over the particles is
  $\sum_{\{i,j\}}\mathbf d_{ij}\otimes\mathbf K(i,j)$.

A gather of several fields evaluates its kernel once for all its results
and has no exchange contract (`exchange(none)`). Its kernel at the
directed pair $(m, b)$ evaluates the product at $(m, b)$ with
$\bar{\mathbf a}_m$ and at $(b, m)$ with $\bar{\mathbf a}_b$: **two
evaluations of the VJP per directed pair** of the relation. The separate
form, one op per term (a gather of the field adjoints, an antisymmetric
gather of the forces, a symmetric sum of the virial, each evaluating both
products), takes **six** on the directed traversal. It stays as the
reference, `md-differentiate{separate-stages=true}`, and the tests run
both (Section 6). The forms are gathers, so the results do not depend on
the order of the threads, and the deterministic level (P6) is unaffected.

What is left for step 4, in this order:

1. **The two-ended scatter**, one evaluation per directed pair: the loop
   evaluates the product at $(i, j)$ once and adds $\bar{\mathbf
   h}^\mathrm c$ and $\mathbf g$ at $i$, and $\bar{\mathbf h}^\mathrm n$
   and $-\mathbf g$ at $j$. It is a lowering strategy over the privatized
   scatter-accumulate of D285 (#416) and the two-ended form of #391,
   compared with the gather form (principle 3); the deterministic mode
   keeps the gather.
2. **Store against recompute** of the forward values that a product takes
   (a layer's messages): storing per-edge values costs memory per edge,
   which is what ML4 exists to save, so the number of evaluations is
   reduced structurally first.

### 4.4 Two GPUs (step 6)

After `md-distribute` (D276) the gather of $\mathbf h_j$ over a domain
boundary is $G\mathbf h$, `md_dist.halo_copy`. The rules of 4.1 then go
through the transposes of D276: the cotangent of a neighbor's features at a
ghost row returns to its owner by $G^\mathsf T$ (`halo_sum`), and the field
gather of 4.1, which reads $\bar{\mathbf a}$ at both ends, needs
$G\bar{\mathbf a}$ at the ghosts, as EAM's force gather needs $G\bar\rho$
(D279, Section 15.2 of [distributed-ml2.md](distributed-ml2.md)). Per-layer
exchange is then one $G$ of $\mathbf h$ a layer in the forward and one $G$
of $\bar{\mathbf a}$ and one $G^\mathsf T$ of the cotangents in the
backward; an expanded halo of $L\,r_c$ instead is the opaque path's. Both
are lowerings of the same ops, which step 6 compares.

## 5. What was built

- `!md.field` with any number of components and `irreps<...>` with its
  layout (parse, print, verify; `withElementType` keeps the
  representation when precision assignment changes the element type).
- `md.stage` and `md.apply_stage` with their verifiers; the reverse method
  `emitAdjoint` of `DerivativeOpInterface`, which `md.apply_stage`
  implements by its product, and its forward rule, which fails.
- `md.gather_relation` with several results, lowered to one
  `md_exec.pair_for` with a destination for each; `md-distribute` refuses
  it (step 6).
- `md-inline` inlines the bodies of the stages into the kernels and removes
  the declarations and the bodies that nothing names any more.
- `md-differentiate`: stage kernels for gathers over pairs, maps, and sums
  over particles, the one gather of Section 4.3 and the separate form
  behind its switch, and the refusals with their reasons: a kernel that
  applies a stage but does not yield its value, a field of several
  components through a kernel that is not a stage kernel, a stage without
  a product, a derivative in a number through a stage (no forward rule),
  and a positional gather of several fields.
- `--test-md-stage-adjoint`, a test pass that writes both sides of the
  adjoint identity of each stage.

## 6. Validation

**The synthetic model** (`test/Integration/Inputs/synthetic_mp.py`): two
PaiNN-like layers with `irreps<4x0e + 4x1o>` (16 components), two species,
an edge stage with a cosine cutoff and a radial basis $\sin(k\pi r/r_c)/r$
inside its body, a node stage that mixes the vector channels by their
norms and inner products, and a readout; weights from a fixed seed. The
script writes the bodies and their VJPs (its own reverse sweep, the
stand-in for external AD) as `func.func` of `arith`, `math`, and `vector`,
in any Python, so that lit writes them at each run; with NumPy it
evaluates the model over a whole periodic system independently of the
bodies and of `md-differentiate`: the energy, and the forces and the
virial by one reverse sweep over the whole energy in the positions and a
strain, which agree with central differences of the energy (Richardson to
$h^6$) to $4\times10^{-6}$ of 143 in the forces and $6\times10^{-10}$ of
41 in the virial; the sweep's forces add to $1.8\times10^{-14}$ and its
virial is symmetric to $9\times10^{-16}$. The system: 10 particles in a
cubic cell of 1.2 nm, $r_c$ = 0.5 nm, 40 directed pairs.

| Check | Where | Result | Tolerance |
|---|---|---|---|
| Adjoint identity $\langle\bar{\mathbf y}, J\mathbf v\rangle = \langle J^\mathsf T\bar{\mathbf y}, \mathbf v\rangle$, $J\mathbf v$ from D183's forward rules on the forward body, 3 random points a stage | `stage-adjoint.test`, CPU, double | largest relative difference $1.3\times10^{-15}$ (edge 1), $1.8\times10^{-16}$, $3.9\times10^{-16}$, $2.1\times10^{-16}$, $2.4\times10^{-16}$ (node 1, edge 2, node 2, readout) | $10^{-12}$ |
| The same with one product scaled by 1.001 | the same | fails (prints 0) | |
| Energy, forces, virial against the reverse sweep, one gather (and the separate form) | `synthetic-mp.mlir`, CPU, double | 0, $1.8\times10^{-15}$, $1.7\times10^{-15}$ relative to the largest value, both forms | $10^{-12}$ |
| Forces and six strains against central differences of MDIR's own energy (D184) | the same | pass; a product of edge 2 scaled by 1.001 fails | D184's defaults |
| The same against the reverse sweep | CPU, mixed | $1.0\times10^{-6}$, $9.7\times10^{-6}$, $9.6\times10^{-6}$ | $3\times10^{-5}$ |
| The same | `synthetic-mp-gpu.test`, RTX 4090, double, the neighbor matrix and the groups | $3.4\times10^{-16}$, $5.0\times10^{-16}$, $5.2\times10^{-16}$ | $10^{-12}$ |
| The same | RTX 4090, mixed, groups | $1.4\times10^{-6}$, $1.3\times10^{-5}$, $1.3\times10^{-5}$ | $3\times10^{-5}$ |

The bodies inline into kernels of `arith` and `math`, so the existing
lowerings take them on the CPU and on a device without a change; what
step 4 needs is the other two forms of a body (Section 2.1): a call of a
batched body from a tile of the loop, for StableHLO through `linalg` and
for an AOTInductor package through its entry.

## 7. What was ruled (the maintainer, 2026-10-19, on #441)

- **Q1.** The declaration and the application, `md.stage` and
  `md.apply_stage`.
- **Q2.** A feature dimension on `!md.field`, with `irreps<...>` and its
  layout in the type.
- **Q3.** StableHLO lowers through `linalg` and MDIR's own lowering; the
  choice is measured against AOTInductor in step 4.
- **Q4.** The milestone ruling of 2026-10-16 and the StableHLO ruling of
  2026-10-17 are recorded as `D287`, amending D167 and D169.
- **The number of evaluations.** One gather of the three terms in this
  step (two evaluations per directed pair), the separate form kept as the
  reference; the two-ended scatter (one evaluation) and store against
  recompute in step 4 (Section 4.3).

Chunk sizes from the control file and from Python, #377's last question,
belong to step 4: step 3 adds no user-visible key.

## 8. PaiNN in stages (step 4)

Step 4 of #377 (`D294`, `D293`, PR #447):
the stages of a pretrained PaiNN, nablaDFT's `painn_100k` of D286,
exported from its weights and lowered through MDIR's pipeline, against
the opaque call of the same checkpoint. Status: **4a built**; **4b in
progress** (#462): the fusion of the bodies is built
(`D299`). Section 8.7 says what was ruled; Section 8.8
what 4b takes.

### 8.1 The stages of PaiNN

SchNetPack's PaiNN [[Schütt2021]](references.md#schutt2021) with
$F = 128$ features, $L = 6$ interactions, 100 Gaussians, and a cosine
cutoff at $r_c$ = 5 Å carries per atom scalars $\mathbf q_i \in
\mathbb R^F$ and vectors $\boldsymbol\mu_i \in \mathbb R^{3 \times F}$,
the field `irreps<128x0e + 128x1o, ir_mul>` (512 components). Layer $k$
is three stages:

| Stage | Op | Body of one element |
|---|---|---|
| Context $\mathbf x^{(k)}$ | `md.map_particles` | $\mathbf x_i = \mathrm{Dense}(\mathrm{silu}(\mathrm{Dense}(\mathbf q_i)))$, $128 \to 128 \to 384$ |
| Edge $\mathbf a^{(k)}$ | `md.gather_relation` of $\mathbf x$ and $\mathbf h$ | $\mathbf W = (\mathsf A_k\,\mathbf e(r) + \mathbf b_k)\,f_c(r)$, $(\mathbf s, \mathbf w^R, \mathbf w^\mu) = \mathbf W \odot \mathbf x_j$; $\Delta\mathbf q = \mathbf s$, $\Delta\boldsymbol\mu = \hat{\mathbf r}_{ij} \otimes \mathbf w^R + \mathbf w^\mu \odot \boldsymbol\mu_j$ |
| Node $\mathbf h^{(k)}$ | `md.map_particles` of $\mathbf h$ and $\mathbf a$ | the update $\mathbf q + \Delta\mathbf q$, $\boldsymbol\mu + \Delta\boldsymbol\mu$, then PaiNN's mixing block |

with $\mathbf e(r)$ the 100 Gaussians, $f_c$ the cosine cutoff, and
$\mathsf A_k, \mathbf b_k$ the rows of SchNetPack's shared filter network
that belong to layer $k$. The embedding $\mathbf q^{(0)}_i =
\mathsf E\,z_i$, $\boldsymbol\mu^{(0)} = 0$, is a map of the species, and
the readout $U = \sum_i \epsilon_H\,[\mathrm{MLP}(\mathbf q_i) + \bar u]$
a sum over particles ($\bar u$ the per-atom mean of SchNetPack's
`AddOffsets`, $\epsilon_H$ Hartree in kJ/mol). The context net runs per
atom, as in SchNetPack, not per pair: the edge stage takes $\mathbf x_j$
gathered. The conversions of units (nm to Å, Hartree to kJ/mol) are inside
the bodies, so that they are differentiated with them. SchNetPack's pair
vector is $\mathbf r_{ij} = \mathbf x_j - \mathbf x_i = -\mathbf d_{ij}$.
The 20 stages (the embedding, 6 times three, the readout) and their 19
products make one MLIR module of 22 MB with 4,096 rows a tile.

### 8.2 Bodies from the checkpoint

`scripts/mlip/painn-stages.py` reads the Lightning checkpoint with
SchNetPack (as `painn-to-metatomic.py` does) and writes:

- **The weights** and SchNetPack's own modules evaluated on random rows,
  with their products for random cotangents, in the environment of
  `ml4-requirements.lock`.
- **A re-implementation of each stage in JAX**, batched over rows, its
  forward and its `jax.vjp`, from the same weights. It is the same model
  only once shown equal (D167), which `painn-stages.py check` does:
  - each stage against SchNetPack's modules on 64 random rows (layers 1
    and 6): the forward to $5.0\times10^{-7}$ and the product to
    $6.5\times10^{-7}$ of the largest value, float32's rounding;
  - the energy, the forces, and the virial composed from the stages, with
    the reverse rules of Section 4.1 applied to the stages' products,
    against SchNetPack's calculator with its `Strain` input in float64
    (`water-box.py --painn-dtype float64`), on the CPU:

    | System | $\lvert\Delta E\rvert / \lvert E\rvert$ | max $\lvert\Delta F\rvert$ / max $\lvert F\rvert$ | max $\lvert\Delta W\rvert$ / max $\lvert W\rvert$ |
    |---|---|---|---|
    | 27 waters, cubic 15 Å, 810 directed pairs | $1.6\times10^{-8}$ | $5.4\times10^{-6}$ of 847 | $2.0\times10^{-6}$ of 571 |
    | the same, triclinic | $6.1\times10^{-8}$ | $6.1\times10^{-6}$ | $1.8\times10^{-6}$ |
    | 64 waters, liquid, 10,410 pairs | $4.6\times10^{-8}$ | $3.1\times10^{-6}$ of 2,227 | $1.1\times10^{-6}$ of 1,100 |
    | 8 waters in 6.2 Å, several images | $7.3\times10^{-8}$ | $3.2\times10^{-6}$ | $9.9\times10^{-6}$ |

    The stages compute in float32 and sum in float64; SchNetPack's own
    float32 evaluation is $3.4\times10^{-5}$ of the largest force from its
    float64 one on the first system, so the comparison is with float64.
- **The exported bodies**: each stage's forward and product exported with
  a static number of rows (the tile) to StableHLO, legalized to `linalg`
  on tensors by StableHLO's `stablehlo-legalize-to-linalg` in
  `stablehlo-opt` when the model is exported (the ruling of 2026-10-19;
  the pinned commit and its build are in the script's documentation).
  The weights that a body reads are its constants: a body holds its
  layer's rows of the filter network, not the whole network.
- **The model** as one MLIR module: the `md.stage` declarations, the
  bodies, and an `md.potential` of the structure of Section 8.1, of the
  positions, the cell, and the atomic numbers.

Weights and exported bodies are never committed.

### 8.3 A batched body in a tile

A body of the batched form takes `tensor<T x ...>` for each value of the
declaration (Section 2.1); it cannot run inside one thread of a fused
kernel. `md-inline` leaves the applications of batched stages in the
kernels, `md-differentiate` differentiates them as any stage (Section 4),
and the loops of `md_exec` carry them to the lowering for a device:

- **`md-exec-lower-stage-bodies`** lowers the bodies to functions of
  buffers: elementwise fusion (below), one-shot bufferization with the
  results as buffers that the caller passes, `linalg` to parallel loops,
  and on a device one kernel of a thread for each iteration of each
  parallel loop; the constants go into the kernels that read them, and the
  bodies' intermediate buffers are taken from the runtime's pool for the
  call. The 39 bodies of nablaDFT's PaiNN become 363 kernels and 112
  calls of cuBLAS (532 kernels with loops for the products, 933 with
  upstream's elementwise fusion alone, D293).
- **The fusion of a body** (`D299`; option `fuse` of the
  pass, true by default; `fuse=false` is D293's lowering). The
  legalization of StableHLO is one op for one op, and upstream's
  `linalg-fuse-elementwise-ops` fuses a producer of one use only. The pass
  therefore, on tensors and before bufferization:
  - folds the unit dimensions (`4096x1`, `1x100`, reductions over one
    element) with upstream's `linalg-fold-unit-extent-dims`, so that the
    reshapes around them do not stop the fusion;
  - rewrites each `tensor.pad` of a constant (the adjoint of a slice: the
    512 components of $\bar{\mathbf h}$ as the sum of three pads of its
    slices) as an elementwise generic that reads the source, or the
    source of the slice it pads, where the index falls inside it and gives
    the constant elsewhere; the fusion takes it into its consumer instead
    of a fill and a copy;
  - starts a reduction whose initial value is a `linalg.fill` of a
    constant from that constant at its first reduced iteration (a select
    on the indices of the reduced loops), instead of from a buffer that a
    kernel of its own fills; the other fills become generics;
  - fuses an elementwise producer into its consumer when the consumer
    reads it through a permutation of its loops, or when the producer is
    cheap (at most 8 ops, none of `math`) and the consumer has no
    reduction; never into a contraction, which would evaluate its operand
    again for each column of its result, so that a contraction keeps its
    operands as it reads them (the schedules of 4b match it);
  - fuses a producer of several uses into each such consumer by
    evaluating it again there, up to 64 ops in its payload, in rounds
    until nothing changes. IREE clones into each dispatch region only
    producers that cost nothing to recompute (fills, splat constants,
    scalar arithmetic: `isCloneableIntoDispatchOp`, iree-org/iree v3.12.0,
    `compiler/src/iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.cpp`)
    and fuses another producer of several uses into the consumer that
    dominates the others, which then also yields it
    (`DispatchCreation/FuseMultiUseElementwiseProducer.cpp`). A body's
    intermediates are rows of a tile in global memory, so here the
    producer is evaluated again: it reads its operands once per element
    either way, and the copy saves the write and the read of its result;
  - writes each result in the buffer that the caller passes for it, where
    the bufferization allocated a buffer and copied it there.

  In `edge.b`, for example, 58 kernels become 22, and in `node.b` 50
  become 29.
- **Products as calls of cuBLAS** (`D302`; option
  `contractions` of the pass, `gemm` by default, `loops` for D293's loops;
  `MDIR_STAGE_CONTRACTIONS` overrides it for comparisons). On a device,
  every op of a body of the shape of a GEMM in f32 becomes a call of
  cuBLAS's `cublasSgemm` through the runtime (`mdrtGemmF32`), on the
  stream of the kernels and in their order:
  - the shape: one reduced loop; every other loop indexes the result and
    either both operands (the batch, at most one, the result's first
    dimension) or one of them; after the batch, the result's dimensions
    are those of the first operand that indexes them (A: the rows; several
    rows dimensions are one, as the three components of PaiNN's vectors in
    `node.b`: 12,288 rows) and then at most one of the other (B: the
    columns); rows or columns (a product of $1\times k$ by $k\times 1$
    is a dot product, a sum of rows below); a payload that adds the
    product of the operands to the result, or to zero at the first reduced
    iteration (the form in which the reductions that start from a constant
    reach it), or a `linalg.matmul` or `linalg.batch_matmul`;
  - the layout is read from the buffers: each matrix needs one contiguous
    dimension (it is passed transposed when that is its rows), rows
    dimensions that lie one after the other in A and in C, and C's columns
    contiguous; a product whose buffers do not allow it stays loops;
  - a product with a batch is one call of `cublasSgemmStridedBatched`,
    the matrices of each index of the batch the previous ones' moved by
    their distance along it: in `edge.b`,
    $y_{r,c} = \sum_f x_{r,f}\, v_{r,c,f}$, 4,096 products of
    $3\times128$ by $128\times1$ (the maintainer's ruling on #472);
  - the matching runs on the operands as the legalization gives them: the
    fusion keeps contractions free of fused operands (above), and the
    reductions that it starts from their constants exclude these;
  - a product that starts from a fill of zero gets a fill of its own and
    calls cuBLAS with beta 0 instead (no fill kernel); the payloads that do
    not read their init take an empty tensor, so that a fill shared
    through CSE does not stay for them;
  - the weights that a product reads, globals of the body, are copied to
    the device by the runtime when first read (`mdrtDeviceConstant`), and
    the copy is waited for, since the stream of the kernels does not wait
    for the null stream. A copy is found again by the host address and
    size of the weights and by its owner, the activation of the embedding
    program that made it (D215), or none outside one. It is freed with its
    owner: an activation's copies when the activation closes, the others
    when a program unloads its modules, and all of them, with the handle of
    cuBLAS, when the device changes (`mgpuSetDefaultDevice`). A host
    address names the weights only while their program is loaded: after
    it is released, another program may put its own weights there
    (`model-stages-programs-gpu.test`: with the copies keyed by the
    address alone, a second model of the same shapes read the first
    one's weights);
  - f32 in both precision modes (the bodies are f32, #447), cuBLAS's
    default math mode: no TF32 (the maintainer's ruling on #463).

  The bodies of nablaDFT's PaiNN make 112 calls, 6 of them batched, and
  363 kernels (532 with loops).
- **Sums of rows as a warp a row** (`D303`, the
  maintainer's ruling on #472; on a device, after the products). A
  generic of one result, f32 or f64, whose one reduced loop has at least
  32 iterations and runs along the contiguous dimension of every operand
  it indexes, and whose payload yields the accumulator (or a constant at
  the first reduced iteration) plus a term that does not read it, becomes
  a kernel of a warp for each element of the result, 8 warps a block, as
  IREE gives its row reductions a warp each (`VectorDistribute`):
  - lane $l$ adds the terms $l, l+32, l+64, \dots$ in that order, the
    payload evaluated for each term in the lane;
  - the warp combines its lanes by the butterfly of D301
    (`kernels::emitWarpReduce`, offsets 16 to 1);
  - the first lane stores the start (the init's element, or the
    constant) plus the warp's sum.

  The order is fixed by the lanes, not by the timing of the threads, so a
  body gives the same bits on every run, in every mode, as the loops did.
  The sum is taken in the type of the result (f32 in PaiNN's bodies), as
  the loops took it. The order differs from the loops' order (term by
  term), so the rounding differs. The sums inside a body state no
  `md_exec.algebra`: `algebra::reduces` covers the loops and sums of
  `md_exec` and `md_dist`, not `linalg` in a body. The fixed order is what
  lets this lowering stand in every algebra, including `fixed_order`
  (D301, Section 5.4 of [device-primitives.md](device-primitives.md)).
  In PaiNN these are `edge.b`'s sums over 384 and over 100. The sums over
  the three components of a vector,
  $z_{r,f} = \sum_c a_{r,c,f}\, b_{r,c,f}$, have fewer than 32 terms. They
  are not matrix products either: $f$ indexes both operands and the
  result, a second batch dimension whose distance cannot join $r$'s in one
  stride. They stay a thread for each element.
- **`convert-md-exec-to-gpu`** splits each loop whose kernel applies
  batched stages into a host loop over tiles of $T$ rows. For each level of
  the applications (0 for those whose operands depend on no other, one
  more than the highest they depend on), a kernel packs the operands of
  the tile's rows into buffers and the bodies are called on them; a last
  kernel unpacks the results and runs the rest of the kernel. The fused
  loops of `md-exec-fuse-loops` put the embedding and the first context
  map in one kernel, two levels.
- **The rows of a loop over pairs** are the slots of the pairs of images
  of D277 (`md.edges`), which the loop finds at its positions: each
  particle's pairs within the cutoff first in its slots, counted. A center
  then adds up its rows of the tile within the cutoff, in the order of
  its slots, in one thread: a gather, without atomics, so the results do
  not depend on the order of the threads. The slots are ordered by
  center, so the centers that meet a tile are those of its rows: the row
  of a center's first slot in the tile leads its sum, and the other rows
  do nothing (`D300`). A wide destination is added up by a
  thread for each row and component, the threads of a row side by side,
  so that a center's components are read together; a narrow one by a
  thread for each row. (D293's sum ran a thread for each particle of the
  system, which looped over all the components of its rows: at 6,144
  atoms, 2.05 ms a tile of the reverse loops against 0.44 ms now, and a
  test of every particle for each tile at 196,608 atoms; a step is 1.79,
  1.87, and 4.02 times shorter at 6,144, 12,288, and 196,608 atoms, about
  5.4 µs a pair at every size.) The sums, and
  their order, are those of D293. The slots hold every image of a
  neighbor (a cell narrower than twice the cutoff included, as the opaque
  call takes it); in a cell wider than twice the cutoff there is one, the
  minimum image of the loop's semantics. The rows beyond the slots and
  the slots beyond the cutoff are evaluated and not read. The rows of a
  loop over particles are particles.
- **Wide values** (the 384 and 512 components of the features) are not
  built as vectors in these kernels: a copy into a row and the components
  of a wide value that the kernel yields go component by component,
  through the elementwise ops that make it. With vectors of 512
  components, LLVM took 18 minutes to compile the kernels of PaiNN; it
  takes 11 seconds this way.
- **Precision**: a body may compute in a narrower type than the
  declaration (PaiNN's float32); the pack converts, as the opaque call
  converts at the model's boundary. In `mixed`, the default stage fields
  and ordinary pair kernels use f32. The model key
  `[[energy.model]] stage_field_precision = "DOUBLE"` selects
  `md-exec-assign-precision{mode=mixed stage-fields-f64=true}` and
  gives f64 storage to fields read by a neural stage, including their
  reverse-sweep adjoints, while the body's arithmetic remains f32.
  `md-exec-assign-precision` identifies those fields from the
  operands of `md.apply_stage`, tracing through the casts and additions
  left by differentiation and loop fusion (D[ml4-stage-fields]). The
  final coordinate force follows the existing mixed policy. `single`
  keeps f32 fields, and `double` keeps f64 fields.

The two products of each directed pair of the gather of Section 4.3 are
two calls on the same rows. On the CPU the loops of batched stages are
not lowered: the slots of D277 are found on a device only.

### 8.4 Memory: what is stored

Every product recomputes its forward (a VJP exported from `jax.vjp`
does), so nothing is stored per pair: per atom, the fields of each layer
that the backward reads ($\mathbf h$, $\mathbf x$, $\mathbf a$: 1,408
components a layer); per tile, the rows of the operands and the results
and the bodies' intermediates, a constant in the size of the system. The
opaque call holds about 61 kB per pair (Section 2.2 of
[ml4-baselines.md](ml4-baselines.md)), among them all six layers'
filters ($6 \times 384$ values a pair).

### 8.5 Validation

| Check | Reference | Result | Tolerance |
|---|---|---|---|
| The synthetic PaiNN (4 features, 2 layers, random weights), energy, forces, virial tensor (`Integration/stages-batched-gpu.test`, RTX 4090, 81 atoms) | the JAX stages composed with Section 4.1's rules, float32 bodies and float64 sums | double: 0, $2.5\times10^{-7}$, $6.6\times10^{-8}$; mixed: $5.5\times10^{-9}$, $7.9\times10^{-7}$, $1.9\times10^{-7}$ of the largest value | $10^{-5}$ |
| Two synthetic PaiNN programs with identical shapes and distinct weights, 81 atoms (`Driver/model-stages-programs-gpu.test`, D305) | JAX's composed stages on the CPU, pinned with `JAX_PLATFORMS=cpu`; both models use the same fixed water box | For the second model: MDIR GPU 1291.764984 kJ/mol, JAX CPU f32 1291.753784 (difference 0.011200), JAX GPU f32 1291.761261, and CPU JAX f64 bodies 1291.754766. For the first: MDIR $-90121.688965$, JAX CPU f32 $-90121.688965$, CPU f64 bodies $-90121.682255$. The first model repeats after release of the second. | 0.013 kJ/mol, consistent with the f32 body error and about $10^{-5}$ of the smaller model's energy; the weights' energies differ by 91,413 kJ/mol |
| nablaDFT's PaiNN, the same through `mdir-opt` and the runner, 27 waters in 1.02 nm (an `mdir-opt` harness, not in the suite) | the same | double: $3.7\times10^{-8}$, $3.4\times10^{-6}$, $2.2\times10^{-6}$; mixed: $6.2\times10^{-8}$, $1.9\times10^{-5}$, $2.8\times10^{-6}$ | $3\times10^{-5}$ |
| nablaDFT's PaiNN through the Python model (`backend = "STAGES"`), 27 waters in 15 Å, RTX 4090 | SchNetPack's calculator in float64 | double: $\lvert\Delta E\rvert$ $1.8\times10^{-4}$ kJ/mol of 24,385, max $\lvert\Delta F\rvert$ $4.6\times10^{-3}$ of 847 ($5.4\times10^{-6}$), trace of the virial $2.5\times10^{-3}$ of 1,242; mixed: $1.6\times10^{-3}$, $5.4\times10^{-2}$ ($6.4\times10^{-5}$), $4.9\times10^{-3}$ | ten times ML1's (as the opaque call, D286): 0.048, 0.193, 0.048 |
| The opaque call, the same | the same | double: $1.5\times10^{-3}$, $5.9\times10^{-3}$, $1.0\times10^{-3}$; mixed: $4.8\times10^{-4}$, $4.9\times10^{-3}$, $1.3\times10^{-3}$ | the same |
| The staged forces against the opaque call's | | double $8.6\times10^{-6}$, mixed $6.3\times10^{-5}$ of the largest force | |
| The same on 8 waters in 6.2 Å, several images (`Driver/model-stages-ml4-gpu.test`) | SchNetPack's calculator in float64 | within the tolerances in double and mixed | the same |
| Forces and the trace of the virial, central differences of MDIR's own energy (`scripts/mlip/check-stages-fd.py`, double, the 6 largest force components, 27 waters in 15 Å) | Richardson's extrapolation to $h^4$, $h$ = 0.002 nm, and an isotropic strain of 0.002 | $4.2\times10^{-3}$ of the largest force; the trace $1.7\times10^{-3}$ of 1,242 | the energy is a float32 sum: with $h = 10^{-4}$ nm its rounding made the differences $1.9\times10^{-2}$ |
| NVE, 1,000 steps of 0.1 fs from 300 K, 27 waters 5 Å apart, double | the opaque call's run from the same state | total energy within 1.53 kJ/mol of 24,085 (the opaque call 1.54), a drift of $-2.4$ kJ/mol/ps ($-2.7$); the two runs' totals within 0.099 kJ/mol of each other | |
| The same at 0.5 fs, double and mixed | the same | both runs leave conservation together (28 kJ/mol within 50 steps, then they blow up); their totals agree to 0.006 kJ/mol over 50 steps, 0.015 over 100 | |

For D[ml4-stage-fields], the same 24-atom, multiple-image water box
(`Driver/model-stages-ml4-gpu.test`) was evaluated against SchNetPack's
float64 calculator on one RTX 4090, using the same exported PaiNN model
and inputs for the f32-field baseline at `24cb1e98` and the f64-field
strategy. The reference energy is $-7544.467103$ kJ/mol and its largest
force magnitude is $2013.298414$ kJ/mol/nm. The differences are absolute:

| Mixed stage fields | Energy difference (kJ/mol) | Maximum force difference (kJ/mol/nm) | Virial trace difference (kJ/mol) |
|---|---:|---:|---:|
| f32 (default) | 0.000823 | 0.013110 | 0.002051 |
| f64 (opt-in) | 0.000625 | 0.012011 | 0.001224 |

Both are within the existing tolerances of 0.048, 0.193, and 0.048,
respectively. The f64-field strategy reduces these differences by 24%,
8%, and 40%, respectively; the exported bodies still compute in f32.
In `double`, the same model and input give differences of
$1.44\times10^{-5}$ kJ/mol in energy, 0.004786 kJ/mol/nm in maximum
force, and 0.001842 kJ/mol in virial trace, under the same tolerances.

The initial `mixed` result above used f32 inter-stage fields and adjoints;
D[ml4-stage-fields] makes their f64 storage an opt-in strategy while the
exported bodies stay f32. On liquid water at 300 K
(64 waters, unrelaxed), the staged model and the opaque call follow the
same trajectory to the third decimal of kJ/mol for 30 steps and leave it
together at step 38, where both runs blow up: the checkpoint was trained
on molecules in vacuum (D286), so liquid water is a test of the interface
and of memory, not of dynamics.

### 8.6 Peak device memory and time

Liquid water as in Section 2.2 of [ml4-baselines.md](ml4-baselines.md)
(`model-sweep.py`, `mixed`, 4 steps of NVE at 0.5 fs, an RTX 4090 in one
job of one node, `--idle`). The staged model's peak is that of MDIR's own
allocator (`MDIR_DEVICE_MEMORY`), which holds every buffer of the program:
the positions, the neighbor structures and the slots, the fields, the rows
of the tiles, and the bodies' buffers. The opaque call's is the plugin's
allocator (MDIR's own beside it is 11 to 193 MiB). 4,096 rows a tile.

| Atoms | Pairs | Staged: peak (MiB) | Staged: kB per pair | Opaque: peak, plugin + MDIR (MiB) | Opaque: kB per pair | Staged: ms per step | Staged: µs per pair |
|---|---|---|---|---|---|---|---|
| 192 | 10,410 | 235 | 23.7 | 628 + 11 | 63.3 | 175 | 16.8 |
| 768 | 40,572 | 276 | 7.1 | 2,384 + 44 | 61.6 | 473 | 11.7 |
| 1,536 | 83,146 | 329 | 4.2 | 4,870 + 89 | 61.3 | 843 | 10.1 |
| 3,072 | 163,128 | 437 | 2.8 | 9,521 + 179 | 61.2 | 1,602 | 9.8 |
| 6,144 | 328,944 | 652 | 2.1 | 19,171 + 193 | 61.2 | 3,222 | 9.8 |
| 12,288 | 665,312 | 1,081 | 1.7 | out of memory (an allocation of 976 MiB) | | 6,685 | 10.0 |
| 24,576 | 1,309,400 | 1,941 | 1.6 | | | 14,251 | 10.9 |
| 49,152 | 2,632,674 | 3,659 | 1.5 | | | 32,097 | 12.2 |
| 98,304 | 5,323,064 | 7,099 | 1.4 | | | 80,076 | 15.0 |
| 196,608 | 10,562,124 | 13,975 | 1.4 | | | 227,054 | 21.5 |

The opaque call's time is that of D286 (0.50 to 0.79 µs a pair, 8.2 ms a
step at 192 atoms, 44.4 at 1,536, 167 at 6,144; Section 2.3 of
ml4-baselines.md): the 4 steps of this sweep include TorchScript's
specialization of the first evaluations. The staged model's time per pair
grows beyond 50,000 atoms: each tile's sum over the centers launches a
thread for every particle, which tests whether its slots meet the tile
(2,579 tiles of 4,096 rows at 196,608 atoms).

- **Memory.** The staged model's peak grows with the atoms, about 75 kB an
  atom (1.4 kB a pair of liquid water from 49,152 atoms up), against
  61 kB a pair for the opaque call: at 6,144 atoms, the opaque call's
  largest system on 24 GB, 652 MiB against 18.7 GiB, 29 times less. The
  staged model ran 196,608 atoms (10.6 million pairs) in 13.6 GiB, 32
  times the opaque call's largest system on the same device.
- **Time.** The staged step takes about 10 µs a pair up to 25,000 atoms,
  against 0.5 to 0.8 for the opaque call: about 20 times longer (3.2 s a
  step against 167 ms at 6,144 atoms). The bodies are MDIR's naive
  lowering of `linalg` (a thread for each output, no tiling of the
  products, no tensor cores), 1,136 kernels and their launches for each
  tile, and every product recomputes its forward twice per pair (Section
  4.3). 4b measures this form against AOTInductor's kernels on the same
  weights; the memory does not depend on the body's form.

**Where the time goes** (#462; `main` at D295, 6,144 atoms, the runtime's
own profile, `MDRT_PROFILE` with `MDRT_WAIT`; a benchmark that calls one
body on 4,096 rows):

- **The sums over the centers of each tile, about half of the step.** The
  last kernel of a tile of a loop over pairs, a thread for each particle
  that adds up the center's rows, takes 2.05 ms a tile in the reverse
  loops and 1.2 ms in the forward loops, against 2.6 ms for the two
  products of a tile: about 1.6 s of the 3.2 s step. A center's thread
  adds its rows one component at a time (512 components, about 54 rows
  within the cutoff), so only the few dozen centers whose slots meet a
  tile work, each serially; at 196,608 atoms it also tests 196,608
  particles for each of 2,579 tiles.
- **The bodies, the other half, in three loop shapes.** A call of 4,096
  rows takes 1.29 ms for `edge.b`, 0.35 for `edge.f`, 4.4 for `node.b`,
  1.40 for `node.f`, 1.14 for `ctx.b`, and 0.49 for `ctx.f`. In `edge.b`,
  three kernels are 86% of the time: a contraction over 384 with a
  4096 x 100 result (712 µs), the 4096 x 100 x 384 `linalg.matmul` of the
  filter (286 µs), both a thread per element of the result with the
  accumulator loaded from and stored to global memory at each term, and a
  reduction of each row in a thread of its own (4,096 threads, 209 µs).
- **Launches are not the cost**: about 2.6 µs each on the host. The
  fusion of Section 8.3 takes the bodies from 933 kernels to 532 and the
  launches of an evaluation at 6,144 atoms from about 72,000 to about
  29,500; per call, `edge.b` from 1.29 to 1.22 ms and `node.b` from
  4.4 to 4.2 (two runs of each in one job). A step (the sweep above, both
  in one job) goes from 3,247 to 3,195 ms at 6,144 atoms, 6,711 to 6,626
  at 12,288, and 228,139 to 225,331 at 196,608, 1.2 to 1.6% less, and the
  peak from 652 to 592 MiB at 6,144 atoms, the bodies' intermediates
  that the fusion no longer stores.
- **The products as cuBLAS calls** (`D302`, Section 8.3; the
  same sweep, `main` at D299 and this lowering, and the same build with
  `MDIR_STAGE_CONTRACTIONS=loops`, in one job): a step from 3,193 to
  2,160 ms at 6,144 atoms (1.48 times shorter), 6,622 to 4,633 at
  12,288 (1.43), and 225,377 to 193,063 at 196,608 (1.17), where the sum
  over the centers of each tile (#464) dominates; `loops` gives D299's
  times to 0.1%. The peak grows by 30 MiB, cuBLAS's workspace and the
  weights on the device. In the 81-atom NVE run of Section 8.5, a step
  goes from 95 to 27 ms.
- **The batched products and the sums of rows** (Section 8.3; the
  maintainer's ruling on #472). The sweep above, measured in one `--idle`
  job on one RTX 4090 with:
  - `main` at D301, which carries the tile sums of #464 (D300);
  - the products as cuBLAS calls before this step ("before");
  - the same with the batched products and the warp sums ("after").

  | Atoms | `main` (ms) | before (ms) | after (ms) | before / after | Peak (MiB): `main`, before, after |
  |---|---|---|---|---|---|
  | 6,144 | 1,713 | 696 | 488 | 1.43 | 592, 622, 622 |
  | 12,288 | 3,392 | 1,399 | 982 | 1.43 | 1,021, 1,051, 1,051 |
  | 196,608 | 54,009 | 22,222 | 15,623 | 1.42 | 13,951, 13,951, 13,951 |

  With #464's tile sums, the step at 196,608 atoms is 1.48 µs a pair
  after this step, the same as at 6,144 atoms: the time per pair no longer
  grows with the system. In the 81-atom NVE run of Section 8.5 a step
  takes 16 ms (the opaque call: 8.9 ms). That run shared its node with the
  sweep, on another device of the node.

**Baselines of the same bodies** (#462; not MDIR's path, target times).
The bodies of layer 0 as `painn-stages.py export` writes them, 4,096 rows,
ms per call on one RTX 4090: MDIR's lowering with this fusion, with
the products as loops and as calls of cuBLAS; IREE
3.12.0 from its release wheels (`iree-compile` of the exported StableHLO
for CUDA `sm_89`, timed by `iree-benchmark-module`); XLA through
`jax.jit` of the same JAX functions (JAX 0.11.2), in full f32
(`jax_default_matmul_precision = "highest"`), timed from Python, so each
call carries about 0.05 ms of dispatch.

| Body | MDIR, D293 | MDIR, fused, loops | MDIR, fused, cuBLAS | IREE | XLA |
|---|---|---|---|---|---|
| `embed.f` | 0.054 | 0.046 | 0.046 | 0.040 | 0.055 |
| `ctx.f` | 0.49 | 0.48 | 0.058 | 0.13 | 0.061 |
| `ctx.b` | 1.14 | 1.12 | 0.076 | 0.26 | 0.065 |
| `edge.f` | 0.35 | 0.34 | 0.066 | 0.26 | 0.083 |
| `edge.b` | 1.29 | 1.22 | 0.35 | 0.67 | 0.082 |
| `node.f` | 1.40 | 1.38 | 0.13 | 0.48 | 0.073 |
| `node.b` | 4.4 | 4.2 | 0.31 | 1.11 | 0.17 |
| `readout.f` | 0.077 | 0.077 | 0.015 | 0.036 | 0.062 |
| `readout.b` | 0.24 | 0.21 | 0.052 | 0.066 | 0.081 |

With the products as cuBLAS calls (`D302`, two runs of each
in one job, their mean), the bodies take less time than IREE's and,
except `edge.b` and `node.b`, about XLA's, whose times carry 0.05 ms of
dispatch from Python. What is left in `edge.b` and `node.b` is the
products that are not GEMMs: the contractions over the three components
of a vector and the reductions of rows, a thread per element of the
result.

IREE splits the bodies into about as many dispatches as this fusion gives
kernels (`edge.b` 21 against 22, `edge.f` 7 against 7, `node.f` 14
against 16, `node.b` 23 against 29): its advantage is in the code of each
kernel, not in the grouping. Its contractions take a tile of 32 rows by
128 columns of the result per workgroup of 256 threads, 16 columns per
thread, the contracted dimension in steps of 32, in f32 without tensor
cores or shared-memory staging (`TileAndFuse`); its row reductions take a
warp per row, 4 elements per lane (`VectorDistribute`). These are the
starting points of the schedules of 4b.

**The batched products and the sums of rows, per call** (Section 8.3;
ms per call of 4,096 rows, layer 0, the mean of two runs in the `--idle`
job of the sweep above; XLA in the same job, as above, with its 0.05 ms
of dispatch from Python):

| Body | before | after | XLA |
|---|---|---|---|
| `embed.f` | 0.041 | 0.043 | 0.031 |
| `ctx.f` | 0.052 | 0.053 | 0.038 |
| `ctx.b` | 0.073 | 0.073 | 0.044 |
| `edge.f` | 0.062 | 0.062 | 0.044 |
| `edge.b` | 0.363 | 0.143 | 0.073 |
| `node.f` | 0.127 | 0.129 | 0.072 |
| `node.b` | 0.315 | 0.316 | 0.168 |
| `readout.f` | 0.015 | 0.015 | 0.037 |
| `readout.b` | 0.047 | 0.047 | 0.045 |

In `edge.b` (nsys, one call before this step), the kernel of the sum over
384 took 205 µs of about 0.37 ms: a thread for each of the 4,096 rows,
whose neighbors in a warp read elements 384 apart. The batched product
replaced a kernel of 22.8 µs, and per call `edge.b` went from 0.364 to
0.345 ms with it alone. `node.b` is unchanged. Its sums are all over the
three components. Of about 281 µs of its kernels (nsys), about 117 are
cuBLAS calls, about 31 the sums over the components, and the rest
elementwise kernels over $4096\times3\times128$ values. These kernels
write intermediates that XLA's fusions do not write, so what is left
there is a question of fusion, not of products.

**Mixed stage-field storage (D[ml4-stage-fields]).** On the same RTX
4090 kind, with the same exported PaiNN, four NVE steps, and the base
commit `24cb1e98`, the opt-in f64 stage fields increased both peak
device memory and step time:

| Atoms | f32 fields: peak (MiB), ms/step | f64 fields: peak (MiB), ms/step |
|---|---:|---:|
| 6,144 | 621.5, 480.812 | 714.5, 504.124 |
| 196,608 | 13,951.3, 15,389.638 | 16,927.3, 16,206.971 |
| 262,143 | 18,536.8, 20,591.040 | 22,504.8, 21,636.554 |

The 196,608-atom cost is 2,976 MiB (21.3%) of peak allocation and
5.3% of step time. The f32-field mixed default is retained to protect
the largest system that fits. The opt-in model key is
`stage_field_precision = "DOUBLE"` (the maintainer's ruling on #544).
In a one-step fit probe of 99,999 waters (299,997 atoms, 16,147,120
pairs), f32 stage fields completed with a peak of 21,232.4 MiB on a 24
GiB RTX 4090; f64 stage fields reached `CUDA_ERROR_OUT_OF_MEMORY`
before a step. The f64 strategy completed 262,143 atoms with 22,504.8
MiB peak, so the tested capacity interval differs: f32 fits at least
299,997 atoms, while f64 fits 262,143 but not 299,997. The `.gro`
generator's residue field overflows above 99,999 waters (#550), which
limits this probe; it does not establish the exact maximum for f32.

### 8.7 What was ruled (the maintainer, 2026-10-19, on #447)

- **The split**: 4a in PR #447, 4b next.
- **Q1**: StableHLO through `linalg` and MDIR's own lowering is the first
  body form; its step time is measured against AOTInductor in 4b.
- **Q2**: the export script legalizes StableHLO to `linalg` with
  `stablehlo-opt`; MDIR's build takes no dependency on StableHLO.
- **Q3**: `[[energy.model]]` with `backend = "STAGES"` and `path` naming
  the one MLIR module; the rows of a tile are fixed by the export in 4a,
  and `rows_per_tile` comes in 4b.

### 8.8 What 4b takes

- The AOTInductor form (`torch.export` of each stage from SchNetPack's
  modules) and its time against 4a's on the same weights.
- The time of the `linalg` form (#462), in this order: the fusion of the
  bodies (`D299`, Section 8.3, built); the sums over the
  centers of a tile over the range of centers whose slots meet it, a
  thread for each center and component (#464); the
  products of the shape of a matrix product (`linalg.matmul`, and the
  generic contractions with one contracted dimension) as calls of
  cuBLAS's f32 GEMM on the runtime's stream and buffers, without TF32
  (`D302`, the maintainer's ruling on #462: no schedules of
  MDIR's own for dense products), the fusion of Section 8.3 keeping the
  elementwise ops around them; then AOTInductor's bodies measured against
  them, the choice a strategy with a switch, by measurement (#444); MDIR's
  own schedules only for contractions that neither serves. Built: the
  batched products as strided-batched calls of cuBLAS and the sums of rows
  as a warp a row through D301's butterfly (`D303`, the
  maintainer's ruling on #472, Section 8.3). AOTInductor's bodies stay a
  baseline: its package does not export the VJP. The sums over the three
  components can opt into D318 (Section 8.9); the
  remaining elementwise kernels of `node.b` remain open (Section 8.6).
- Store against recompute; the two-ended scatter (one product per pair).
- Rows compacted to the pairs within the cutoff; the fields in f64 in
  `mixed`.
- `rows_per_tile` from the control file and Python, with bodies of a
  dynamic number of rows.
- The RTX PRO 6000; a CPU lowering of the tiles.

### 8.9 Short reductions of a stage body (D318)

The sums over the three vector components in PaiNN's `node.b` are the
same reduction primitive as D303's sums over 32 or more components. One
`RowSum` matcher accepts a bufferized `linalg.generic` with one reduced
loop and a term independent of the accumulator. It accepts read-only
`memref.load` operations in the term, and rejects writes, unknown effects,
and nested regions. A sum of 2 to 31 terms uses one thread per
output, combining the terms in their source order through
`emitOrderedRegisterSum`; the accumulator stays in a register until the
single store. A longer sum whose inputs are contiguous along the reduced
dimension uses D303's warp strategy. Other reductions retain the loop
lowering. The two strategies share the term emitter and the reduction
operator of `ReductionEmitter.h`; the body pass chooses the strategy.

With the short strategy and body fusion enabled, a pure one-input pointwise
operation immediately after that sum is computed in the same kernel. The
kernel still writes the sum to its original buffer, so other users see the
same value. The pointwise output must be a fresh allocation with matching
identity maps and shape; an intervening operation other than allocation,
an effect, another input, or a read of the old output keeps the separate
kernel. A final pure pointwise producer of two identical results can write
both result buffers in one launch. This requires local allocated inputs,
single-use result buffers, and a stage body whose only symbol users are
`md.stage`: the tiled stage caller allocates a distinct device buffer for
each result. A body with any other symbol user, a captured external input,
or an unproved view keeps the copies. These choices do not combine or hide
the six cuBLAS calls, their buffer lifetimes, or their completion edges.

The loop strategy and separate pointwise launches remain the default. The opt-in pass option
`md-exec-lower-stage-bodies{short-reductions=true}` or environment value
`MDIR_STAGE_SHORT_REDUCTIONS=registers` selects the combined strategy when
the pass's `fuse` option is also true (its default); `fuse=false` leaves the
pointwise operations separate.
No control-file key, Python API, or exported body changes.

On one RTX 4090 at 4,096 rows, `node.b` took 0.313950 ms with loops,
0.308144 ms with register sums alone, and 0.287998 ms with the combined
fusion. Its GPU launches fell from 21 to 17; all forms retain six GEMM
calls. For 6,144 atoms and 328,944 pairs in mixed precision, two alternated
20-step pairs on one idle RTX 4090 measured 482.504 and 482.345 ms/step
with the default, then 475.182 and 475.279 ms/step with the combined
strategy. Peak device allocation fell from 621.5 to 613.5 MiB. The
default stage IR is byte-identical to current main for this exported
model. These paired measurements support a roughly 1.5% whole-step gain
on this device and model, with no wider default claim; the strategy stays
opt-in.
