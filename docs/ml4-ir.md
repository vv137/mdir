# ML4 IR: neural stages, feature fields, and their derivative

Step 3 of issue #377 (`D[ml4-neural-op]`, `D[ml4-feature-types]`): the IR in
which a message-passing model (PaiNN, MACE) is written as stages of the
`md` dialect, so that MDIR differentiates, schedules, and later
distributes it. The particle structure (the relation, the gathers and sums
over it, the maps over particles) is stated in `md`; only the neural
bodies of the stages are external. External AD generates the backward of
the bodies (D167); `md-differentiate` derives the rest from the structure.

Status: **proposed**, awaiting the maintainer's ruling on the questions of
Section 7 (the name and form of the op, the representation of the feature
types). Steps 4 and 5 (PaiNN and MACE in stages) lower these ops; step 6
distributes them.

## 1. A model in stages

A layer of a message-passing model is a sum over the neighbors of each
atom of a per-edge function, followed by a per-node update:

$$
\mathbf a_i^{(k)} = \sum_{j : (i,j) \in D(N)} \phi_k\big(r_{ij}, \mathbf d_{ij}, \mathbf h_i^{(k-1)}, \mathbf h_j^{(k-1)}, z_i, z_j\big),
\qquad
\mathbf h_i^{(k)} = \psi_k\big(\mathbf h_i^{(k-1)}, \mathbf a_i^{(k)}\big),
$$

with $\mathbf h_i^{(0)} = \eta(z_i)$ an embedding of the species $z_i$ and
the energy a readout $U = \sum_i \rho(\mathbf h_i^{(L)})$. $D(N)$ is the
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
| A body in the module (step 3) | a `func.func` of `arith`, `math`, and `vector` ops over the element's values | `md-inline-stages` inlines it into the kernel before `md_exec`, so it lowers on the CPU and on a device as any kernel does |
| StableHLO (the ruling of 2026-10-17; step 4) | a `func.func` of `stablehlo` ops over tensors with a leading batch dimension, `jax.export` of the stage and of its `jax.vjp` | the tile of the loop calls it with the tile's rows; the body lowers through `stablehlo` to `linalg` and MDIR's pipeline (Section 7, Q3) |
| AOTInductor (step 4) | a `func.func private` declaration with `md.aoti = {package = "...", entry = "..."}`, batched as the StableHLO form | a call of the package's entry with the tile's rows, as the model term calls one today (D273) |

A batched body takes `tensor<? x ...>` for each value of the declaration
and the verifier checks that its element types are the declaration's. Step
3 implements the first form only and refuses the others with a diagnostic.

### 2.2 `md.apply_stage`: one element

```mlir
%m = md.apply_stage @edge1(%r, %d, %hi, %hj, %zi, %zj) : (...) -> vector<16xf64>
%rb, %db, %hib, %hjb = md.apply_stage @edge1 vjp(%r, %d, %hi, %hj, %zi, %zj ; %mb)
    : (...) -> (f64, vector<3xf64>, vector<16xf64>, vector<16xf64>)
```

The op applies a stage, or with `vjp` its vector-Jacobian product, to the
values of one element inside a kernel of `md.gather_relation`,
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

- `N x f64` lifts the restriction of the type to 1 or 3 components. A
  kernel takes one value of the field as `vector<N x f64>`
  (`getKernelValueType`), and the storage form is `memref<? x N x f64>`, as
  for 3 components; storage, the pools, precision assignment, and
  `mdrt.from_buffer` are already written for any $N$.
- `irreps<...>` states the representation of the rotation group $O(3)$ that
  the components carry: a sum of blocks `M x l p`, $M$ copies of the
  irreducible representation of degree $l$ (dimension $2l + 1$) and parity
  `e` (even) or `o` (odd), with $N = \sum M (2l+1)$, $l \le 8$. The
  notation is e3nn's [[Geiger2022]](references.md#geiger2022). The layout of the components of one
  particle is block after block in the order written, and within a block
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
the VJP of the bodies at the stage boundary.

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

### 4.3 Cost, and what the lowering may change

Each rule of 4.1 is a separate op, so the VJP of a directed pair is
evaluated in the gather of the fields (twice, once from each end), in the
gather of the forces (twice), and in the sum of the virial (once per
unordered pair, from both ends): six evaluations of the VJP per directed
pair where one suffices. The ops say what is computed, not how often:
fusing the three into one loop over the directed pairs that evaluates each
VJP once and adds its four cotangents at both ends (a scatter) is a
lowering (step 4), compared with the ops as they are (principle 3). Store
against recompute of the forward values that a VJP takes (a layer's edge
messages) is a pass of step 4 as well.

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

## 5. Plan of the implementation

1. `!md.field` with $N$ components and `irreps<...>` (parse, print,
   verify), `md.stage` and `md.apply_stage` with their verifiers, and the
   reverse method of `DerivativeOpInterface`.
2. `md-inline-stages`: the bodies of the module inlined into the kernels
   before `md_exec`.
3. `md-differentiate`: the rules of 4.1 for stage kernels, beside the
   existing ones.
4. A synthetic model in NumPy and its bodies, and the tests of Section 6.

## 6. Validation

- **A synthetic model** (`scripts/mlip/synthetic-mp.py`): two layers of the
  form of Section 1, PaiNN-like, with features `irreps<4x0e + 4x1o>`, a
  cosine cutoff inside the edge body, and species, its weights from a fixed
  seed. The script writes the bodies (forward, and its VJP by its own
  reverse sweep, the stand-in for external AD) as `func.func` of
  `arith`/`math`/`vector`, and evaluates the model in NumPy, independently
  of the bodies it writes: the energy, the forces and the virial by central
  differences extrapolated in the step.
- **The adjoint identity** of each stage, $\langle\bar{\mathbf y},
  J\mathbf v\rangle = \langle J^\mathsf T\bar{\mathbf y}, \mathbf v\rangle$,
  with $J\mathbf v$ from MDIR's forward rules on the forward body (D183)
  and $J^\mathsf T\bar{\mathbf y}$ the bound VJP, at random points, to
  $10^{-12}$ relative in double.
- **Finite differences** on a small periodic system on the CPU in double:
  `md-check-derivatives` (D184; forces and six strains of the cell), and
  the energy, forces, and virial against the NumPy reference, with the
  directional derivative $\langle\mathbf F, \mathbf v\rangle$ against
  $-(U(\mathbf x + \epsilon\mathbf v) - U(\mathbf x - \epsilon\mathbf
  v))/2\epsilon$.
- **lit**: parse, verify, and print of the types and ops, and the invalid
  cases of each verifier; the refusals of `md-differentiate`.
- **A device**, if the lowering takes the inlined kernels with fields of 16
  components; else the missing piece is stated for step 4.

## 7. Questions for the maintainer

- **Q1. The name and form of the neural op.** Proposed: a declaration
  `md.stage @s forward(@f) vjp(@b) : (...) -> (...)` and an application
  `md.apply_stage @s(...)` (and `md.apply_stage @s vjp(...)`) in kernels.
  The alternative is one op in kernels that holds both symbols
  (`md.neural @f vjp(@b)(...)`), without a declaration; the declaration
  states the signature, the bindings, and later the form of the batch and
  a store/recompute choice, once for every kernel that applies the stage,
  including those that `md-differentiate` derives.
- **Q2. The feature types.** Proposed: a feature dimension on `!md.field`
  (`N x f64`, and `irreps<M x l p + ...[, ir_mul]>`), with the layout and
  basis of Section 3.1, against a separate type.
- **Q3. StableHLO's lowering.** The ruling of 2026-10-17 asks the design PR
  to choose between StableHLO to `linalg` with MDIR's own lowering and
  handing the body to XLA, and to measure it against AOTInductor on the
  same weights. Proposed: `stablehlo` to `linalg` (MDIR's passes then see
  the body, the reason for adopting it), chosen and measured in step 4,
  where a stage of a real model exists; step 3's bodies are in `arith` and
  `math`.
- **Q4. `D[ml-milestones]`.** The milestone ruling of 2026-10-16 and the
  StableHLO ruling of 2026-10-17 are not in `decisions.md` yet; this PR
  records them as `D[ml-milestones]`, amending D167 and D169.
