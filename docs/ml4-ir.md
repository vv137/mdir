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
real artifacts; step 6 distributes them.

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
| StableHLO (the ruling of 2026-10-17; step 4) | a `func.func` of `stablehlo` ops over tensors with a leading batch dimension, `jax.export` of the stage and of its `jax.vjp` | the tile of the loop calls it with the tile's rows; the body lowers through `stablehlo` to `linalg` and MDIR's pipeline (Section 7, Q3) |
| AOTInductor (step 4) | a `func.func private` declaration with `md.aoti = {package = "...", entry = "..."}`, batched as the StableHLO form | a call of the package's entry with the tile's rows, as the model term calls one today (D273) |

A batched body takes `tensor<? x ...>` for each value of the declaration
and the verifier checks that its element types are the declaration's. Step
3 implements the first form only and refuses the others with a diagnostic.

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
