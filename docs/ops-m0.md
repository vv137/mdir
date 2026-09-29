# MDIR Op Specification, Milestone M0

Status: draft 5 (2026-09-29). The `md` and `dyn` dialects are implemented,
and so are the value form of `md_exec` and the conversion into it. Storage
assignment and the lowering of `md_exec` to executable code are not.

This document specifies the types and ops needed for milestone M0: a
Lennard-Jones fluid integrated with velocity Verlet or leapfrog, on one node,
on CPU and GPU. It covers `md`, the minimal `dyn`, and `md_exec`.

Operand lists, result lists, and the mathematical definitions are normative.

| Part | State |
|---|---|
| `md` types and ops (Section 4) | Implemented. The examples show the actual syntax. |
| Truncation, differentiation, exchange check (Sections 4.7, 4.8, 5) | Implemented as passes; see Section 5.6 |
| `dyn` ops (Section 6) | Implemented. The examples show the actual syntax. |
| `md_exec` ops in the value form (Section 8) | Implemented. The examples show the actual syntax. |
| Conversion of `md` and `dyn` to `md_exec` (Section 9.1) | Implemented as the pass `convert-md-to-md-exec` |
| Storage form, storage assignment, fusion (Sections 8.5, 9.4, 10) | Not implemented |

It follows the accepted decisions in [decisions.md](decisions.md). Tags such
as (S1) or (B4) name the decision behind a section.

## 1. Notation

| Symbol | Meaning |
|---|---|
| `P` | A particle set. Each particle has a global ID. |
| `a : P → V` | A per-particle field with values in `V`. `a_i` is its value at particle `i`. |
| `h` | The simulation cell: three lattice vectors and a periodicity flag per direction. |
| `d_ij` | Minimum-image displacement `x_i − x_j − h·n`, with integer `n` chosen to minimize the length. |
| `r_ij` | `|d_ij|`. |
| `r_c` | Cutoff. |

M0 requires `r_c` to be smaller than half the shortest perpendicular width of
the cell, so that the minimum image within the cutoff is unique. A violation
is a run-time error.

## 2. Types

### 2.1 `md` types

| Type | Meaning |
|---|---|
| `!md.field<@set, E>` | A per-particle field on particle set `@set`. `E` is `f64`, `i32`, `i64`, or `3 x f64`. |
| `!md.cell` | A simulation cell. |
| `!md.relation<@set, k, O>` | A relation of arity `k` on `@set` with orientation `O`. |

Scalars and small fixed-size values inside kernels use builtin types: `f64`,
`i64`, `index`, and `vector<3xf64>`.

The examples use these aliases:

```mlir
!vec   = !md.field<@atoms, 3 x f64>
!real  = !md.field<@atoms, f64>
!pairs = !md.relation<@atoms, 2, unordered>
```

### 2.2 Relations

A relation is a **set** of tuples of distinct particles. A tuple appears at
most once. This is the logical relation. How it is traversed is an execution
decision and is never part of the type.

| Orientation | Tuples | Example |
|---|---|---|
| `unordered` | `{i, j}`. The tuple has no first member. | Nonbonded pairs, bonds |
| `ordered` | `(i, j)`. `(i, j)` and `(j, i)` are different tuples. | Directed edges of a graph |

For an unordered relation `R`, its directed expansion is

```text
D(R) = { (i, j), (j, i) : {i, j} ∈ R }
```

`D(R)` has exactly twice as many tuples as `R`. It appears in the definitions
below and in the lowering, but it is not a type that users write.

For arity above 2 the orientation generalizes to a group of permutations
under which tuples are identified. That is specified with M1.

### 2.3 There is no state type (S1)

The simulation state is not an aggregate type. It is the collection of SSA
values that are carried from one step to the next: positions, velocities,
forces, and the cell. Each is a separate value.

Reason: invalidation can then be read from the use-def graph with no
knowledge of op semantics. An op that returns a new position field
invalidates what was derived from the old one; an op that does not, does not.

Functions take and return these values individually. An aggregate can be
added later as a convenience for function signatures.

### 2.4 Velocities, not momenta (S2)

The state holds velocities. Thermostats, velocity constraints, and trajectory
formats are all defined on velocities.

### 2.5 `f64` is the reference precision (B2)

The semantic program is a floating-point program in `f64`, with the
semantics that the upstream `arith` and `math` ops define. It is the
reference: the reference interpreter executes it as written.

`f64` is the only floating-point type allowed at the semantic level.

Lowering to single or mixed precision is a transformation that deliberately
relaxes the numerical semantics. It is governed by the precision policy
(Section 7) and its result agrees with the reference only within a
tolerance. The same holds for any transformation that reassociates
floating-point arithmetic; such transformations are allowed only when the
plan enables them, and never in the deterministic mode.

A separate real-number type with its own arithmetic ops was considered and
rejected for M0. It would need about 35 scalar ops and its own vector type,
and it would forgo upstream folding and canonicalization. Kernels are
generated by the expression parser, not written by hand, so changing the
representation later is mechanical.

### 2.6 Units (S4)

The IR carries plain numbers in the internal unit system. The module declares
the system once. Front ends convert on input and output.

| Unit system | Length | Time | Mass | Energy |
|---|---|---|---|---|
| `md` | nm | ps | amu | kJ/mol |
| `reduced` | σ | τ | m | ε |

The `md` system is the one OpenMM uses.

## 3. Kernel bodies

A kernel is a region evaluated once per particle or per tuple. It contains
only pure ops from `arith`, `math`, and `vector`, and ends with a yield.

### 3.1 Expression syntax

Energy expressions are strings in the syntax of OpenMM custom forces (D22).

| Element | Rule |
|---|---|
| Operators | `+`, `-`, `*`, `/`, `^` |
| Grouping | Parentheses |
| Numbers | Decimal or exponential: `5`, `-3.1`, `1e6`, `3.12e-2` |
| Intermediate values | Definitions follow the main expression, separated by `;`. A value is used before it is defined. |
| Pair variables | `r` is the distance. A per-particle parameter `p` is referenced as `p1` and `p2`. |
| Global parameters | Referenced by name |

Example:

```text
4*epsilon*((sigma/r)^12-(sigma/r)^6); sigma=0.5*(sigma1+sigma2); epsilon=sqrt(epsilon1*epsilon2)
```

### 3.2 Mapping to IR

All target ops below exist in LLVM 23.1.2.

| Expression | IR |
|---|---|
| `+ - * /` | `arith.addf`, `arith.subf`, `arith.mulf`, `arith.divf` |
| `^` with an integer constant exponent | `math.fpowi` |
| `^` otherwise | `math.powf` |
| `sqrt exp log sin cos tan` | `math.sqrt`, `math.exp`, `math.log`, `math.sin`, `math.cos`, `math.tan` |
| `asin acos atan atan2` | `math.asin`, `math.acos`, `math.atan`, `math.atan2` |
| `sinh cosh tanh` | `math.sinh`, `math.cosh`, `math.tanh` |
| `erf erfc` | `math.erf`, `math.erfc` |
| `abs floor ceil` | `math.absf`, `math.floor`, `math.ceil` |
| `sec csc cot` | Reciprocal of `math.cos`, `math.sin`, `math.tan` |
| `min max` | `arith.minimumf`, `arith.maximumf` |
| `step(x)` | `arith.cmpf` and `arith.select`: 0 if `x < 0`, else 1 |
| `delta(x)` | `arith.cmpf` and `arith.select`: 1 if `x = 0`, else 0 |
| `select(x, y, z)` | `arith.cmpf` and `arith.select`: `z` if `x = 0`, else `y` |

### 3.3 Lowering of math functions

Checked with LLVM 23.1.2 by lowering each op and running the code generator.

| Target | Result |
|---|---|
| x86-64 | All ops in Section 3.2 lower. `erf` and `erfc` become calls to the C math library; the rest become LLVM intrinsics. |
| NVPTX | Arithmetic, `math.fpowi`, and `math.sqrt` lower. `exp`, `log`, `sin`, `cos`, `tanh`, `powf`, `asin`, `acos`, `atan2`, `erf`, and `erfc` do not: the target has no library to call. |

The Lennard-Jones kernel uses only arithmetic and integer powers, so M0 is
not affected. Any potential that uses a transcendental function on a GPU
needs a project-owned lowering for it. The candidates are polynomial
approximations generated in the IR, or a device math library linked as
bitcode.

## 4. `md` ops

### 4.1 Summary

| Op | Purpose |
|---|---|
| `md.particle_set` | Declares a particle set. |
| `md.potential` | Defines a potential energy function. |
| `md.function` | Defines any other pure function of fields. |
| `md.neighborhood` | Builds the relation of pairs within a cutoff. |
| `md.sum_relation` | Sums a kernel over the tuples of a relation. |
| `md.gather_relation` | For each particle, sums a kernel over the tuples it belongs to. |
| `md.sum_particles` | Sums a kernel over particles. |
| `md.map_particles` | Applies a kernel to each particle. |
| `md.evaluate` | Requests energy and derivatives of a potential. |
| `md.yield`, `md.return` | Terminators of kernels and functions. |

All `md` ops are pure.

### 4.2 `md.particle_set`

```mlir
md.particle_set @atoms
```

Declares a symbol. The number of particles is a run-time value.

### 4.3 `md.potential` and `md.function` (A1)

```mlir
md.potential @lj(%x: !vec, %cell: !md.cell, %eps: f64, %sigma: f64) -> f64 {
  ...
  md.return %u : f64
}
```

`md.potential` defines the coordinate-dependent potential energy
`U(x; θ)`. It has exactly one result, of type `f64`. Kinetic energy is not
part of it.

The first argument is the position field and the second is the cell. All
parameters are explicit arguments. A potential reads nothing else.

`md.function` has the same form with any number of results of any `md` or
builtin type. Derivative programs and observables are `md.function`s.

Both are isolated from above: the body refers only to its arguments.

**Parameter binding (S7).** Whether a parameter is static or a run-time value is
not an attribute. It is static when the caller passes a constant, and a
run-time value when the caller passes an argument of the segment function.
Inlining and constant propagation do the specialization.

### 4.4 `md.neighborhood`

```mlir
%n = md.neighborhood %x, %cell cutoff(2.5) : !vec -> !pairs
```

| | |
|---|---|
| Operands | Position field, cell |
| Attributes | `cutoff`: positive real, in internal units |
| Result | `!md.relation<@set, 2, unordered>` |

```text
N(x, h) = { {i, j} : i ≠ j, r_ij < r_c }
```

The cutoff is a compile-time constant in M0 (S5).

### 4.5 `md.sum_relation`

```mlir
%u = md.sum_relation %n, %x, %cell gather(%q : !real)
       exchange(symmetric) truncation(switch, from = 2.0) {
^bb0(%r: f64, %d: vector<3xf64>, %q_i: f64, %q_j: f64):
  ...
  md.yield %k : f64
} : !pairs, !vec -> f64
```

| | |
|---|---|
| Operands | Relation, position field, cell, and zero or more gathered fields |
| Attributes | `exchange`, `truncation` |
| Kernel arguments | `r`, `d`, then two values per gathered field |
| Kernel result | `f64` or a fixed-size vector of `f64` |
| Result | Same type as the kernel result |

```text
S = Σ_{t ∈ R} k(t)
```

Each tuple contributes once.

Over an unordered relation the kernel must be invariant under exchange of the
two particles, so the op must carry `exchange(symmetric)`. Section 4.7
defines what that attribute means and how it is checked.

If the relation was produced by `md.neighborhood`, the position and cell
operands must be the same SSA values that the neighborhood was built from.

### 4.6 `md.gather_relation`

```mlir
%f = md.gather_relation %n, %x, %cell exchange(antisymmetric, derived) {
^bb0(%r: f64, %d: vector<3xf64>):
  ...
  md.yield %k : vector<3xf64>
} : !pairs, !vec -> !vec
```

| | |
|---|---|
| Operands | As for `md.sum_relation` |
| Attributes | `exchange`: `none`, `symmetric`, or `antisymmetric` |
| Kernel arguments | `r`, `d`, then two values per gathered field |
| Result | A field whose element type is the kernel result type |

```text
a_i = Σ_{j : (i, j) ∈ D(R)} k(i, j)
```

The kernel is evaluated with `i` as the central particle. `d` is `d_ij`.

The definition is a gather. It has no write conflicts by construction.
`exchange` only records a fact that an execution policy may exploit.

### 4.7 Exchange contract (B3)

`exchange` states how a kernel behaves when the two particles are swapped.
Swapping replaces `d` by `−d` and swaps the two values of each gathered
field.

| `exchange` | Property |
|---|---|
| `symmetric` | `k(j, i) = k(i, j)` |
| `antisymmetric` | `k(j, i) = −k(i, j)` |
| `none` | No relation |

The attribute is a semantic contract, not something the compiler derives in
general. Deciding the property for an arbitrary kernel is not possible.

The attribute records what the contract rests on.

| Basis | Syntax | Meaning |
|---|---|---|
| `proof` | `exchange(symmetric)` | The compiler must prove the contract from the kernel. This is the default. |
| `asserted` | `exchange(symmetric, asserted)` | The front end asserts the contract. It is trusted. |
| `derived` | `exchange(antisymmetric, derived)` | A compiler pass, such as differentiation, produced the kernel. The contract holds by construction. |

For the basis `proof`, a checking pass attempts the proof by swapping the
kernel arguments and comparing the two kernels structurally, treating
commutative ops as unordered. If the proof fails, the op is rejected. The
pass is not implemented yet.

The reference interpreter checks asserted contracts numerically on the pairs
it evaluates.

### 4.8 Truncation (B4)

A potential that is cut off at `r_c` jumps by `u(r_c)` whenever a pair
crosses the cutoff. Energy is then not conserved, whatever the integrator.
`truncation` states how the kernel `u` of a relation sum is modified.

| `truncation` | Kernel used for `r < r_c` | Continuity at `r_c` |
|---|---|---|
| `none` | `u(r)` | None |
| `shift` | `u(r) − u(r_c)` | Energy |
| `force_shift` | `u(r) − u(r_c) − (r − r_c) · u'(r_c)` | Energy and force |
| `switch`, from `r_s` | `u(r) · S(r)` | Energy, force, and the derivative of the force |
| `force_switch`, from `r_s` | `u(r) − P(r) − C` | Energy, force, and the derivative of the force |

`r_c` is the cutoff of the neighborhood.

**Switch.** The potential is multiplied by

```text
S(r) = 1                          r ≤ r_s
S(r) = 1 − 10t³ + 15t⁴ − 6t⁵      r_s < r < r_c,   t = (r − r_s) / (r_c − r_s)
```

**Force switch.** A cubic polynomial in `r − r_s` is added to the force, so
that the force and its derivative vanish at the cutoff. With
`F = −u'(r_c)`, `F' = −u''(r_c)`, and `Δ = r_c − r_s`:

```text
A = (F'Δ − 3F) / Δ²        B = (2F − F'Δ) / Δ³

P(r) = 0                                      r ≤ r_s
P(r) = (A/3)(r − r_s)³ + (B/4)(r − r_s)⁴      r_s < r < r_c

C = u(r_c) − (A/3)Δ³ − (B/4)Δ⁴
```

The definition holds for any kernel. It is linear in `u`, so applying it to a
sum of terms equals applying it to each term.

**Correspondence with other packages.**

| MDIR | OpenMM | GROMACS |
|---|---|---|
| `shift` | — | `potential-shift` |
| `switch` | Switching function | `potential-switch` |
| `force_switch` | — | `force-switch` |
| `force_shift` | — | — |

The GROMACS manual defines `force-switch` for a power law `r^-α`. For that
kernel the constants above reduce to the ones in the manual.

A pass expands the attribute into the kernel before differentiation. After
the pass every relation sum has `truncation(none)` and an explicit kernel.
`force_shift` and `force_switch` use the scalar derivative rules of
Section 5.4 to form `u'(r_c)` and `u''(r_c)`.

The expansion is a pass, not a front-end task, so that both front ends share
it and the derivative rules exist once.

Long-range dispersion corrections are not part of M0.

### 4.9 `md.sum_particles` and `md.map_particles`

```mlir
%ke = md.sum_particles gather(%v, %m : !vec, !real) {
^bb0(%v_i: vector<3xf64>, %m_i: f64):
  ...
  md.yield %k : f64
} : f64
```

```text
md.sum_particles:  S   = Σ_{i ∈ P} k(i)
md.map_particles:  b_i = k(i)
```

### 4.10 `md.evaluate`

```mlir
%u, %f = md.evaluate @lj(%x, %cell, %eps, %sigma) request [energy, forces]
           : (!vec, !md.cell, f64, f64) -> (f64, !vec)
```

| Request | Result type | Definition |
|---|---|---|
| `energy` | `f64` | `U` |
| `forces` | Position field type | `F_i = −∂U/∂x_i` |
| `virial` | `vector<9xf64>` | `W`, as defined in Section 5.3, in row-major order |
| `derivative(n)` | `f64` | `∂U/∂θ_n`, where `θ_n` is scalar argument `n` |

`md.evaluate` does not survive semantic differentiation. The pass replaces it
with `md.call` to a generated `md.function`.

```mlir
%u, %f = md.call @lj.energy_forces(%x, %cell, %eps, %sigma)
           : (!vec, !md.cell, f64, f64) -> (f64, !vec)
```

## 5. Semantic differentiation for M0

This section traces the Lennard-Jones potential from its definition to the
force kernel. Every step is an identity.

### 5.1 Energy

```text
U(x) = Σ_{{i,j} ∈ N(x)} u(r_ij)          u(r) = 4ε((σ/r)^12 − (σ/r)^6)
```

This is `md.sum_relation` over an unordered relation. Each pair is counted
once. `u` here is the kernel after truncation has been expanded.

### 5.2 Forces

The set `N(x)` is treated as locally constant. When the kernel is continuous
at the cutoff, the result is the exact gradient of `U`. With
`truncation(none)` it is the gradient everywhere except at configurations
where a pair crosses the cutoff, where `U` itself jumps. With

```text
∂r_ij/∂x_i = d_ij / r_ij        ∂r_ij/∂x_j = −d_ij / r_ij
```

the force on particle `i` is

```text
F_i = −∂U/∂x_i = Σ_{j : (i,j) ∈ D(N)} K(i, j)        K(i, j) = −u'(r_ij) · d_ij / r_ij
```

This is `md.gather_relation`. The factor in front of `d_ij` depends only on
`r_ij`, and `d_ji = −d_ij`, so `K(j, i) = −K(i, j)`. The pass sets
`exchange(antisymmetric)`.

For Lennard-Jones:

```text
K(i, j) = (24ε / r²) · (2(σ/r)^12 − (σ/r)^6) · d_ij
```

### 5.3 Virial and parameter derivatives

```text
W      = Σ_{{i,j} ∈ N} d_ij ⊗ K(i, j)
∂U/∂θ  = Σ_{{i,j} ∈ N} ∂u/∂θ (r_ij)
```

Both are `md.sum_relation`. The virial kernel is invariant under exchange
because both factors change sign.

**Sign convention (B8).** `W` is the MDIR virial. `K(i, j)` is the force on
`i` due to `j`, so `W` is positive for repulsion. The pressure is

```text
P = (2 · E_kin + tr W) / (3V)
```

Other packages use other conventions. The GROMACS virial is `−W / 2`.

### 5.4 Rules

The pass needs two kinds of rules.

| Kind | M0 content |
|---|---|
| Geometry | For a pair with distance geometry: `∂r/∂x_i = d/r`, `∂r/∂x_j = −d/r` |
| Scalar | One derivative rule per `arith` and `math` op in Section 3.2 |

Functions that are not smooth have fixed conventions (B5). `x'` denotes the
derivative of the argument.

| Function | Derivative |
|---|---|
| `select(c, a, b)` | `select(c, a', b')`. The condition is not differentiated. |
| `step(x)`, `delta(x)`, `floor(x)`, `ceil(x)` | 0 |
| `abs(x)` | `x'` if `x ≥ 0`, else `−x'` |
| `min(a, b)` | `a'` if `a < b`, else `b'` |
| `max(a, b)` | `a'` if `a > b`, else `b'` |

The conventions are part of the reference semantics: every back end must
produce the same branch at a tie.

### 5.5 Generated function

```mlir
md.function @lj.energy_forces(%x: !vec, %cell: !md.cell, %eps: f64, %sigma: f64)
    -> (f64, !vec) {
  %n = md.neighborhood %x, %cell cutoff(2.5) : !vec -> !pairs
  %u = md.sum_relation %n, %x, %cell exchange(symmetric) {
    ...
  } : !pairs, !vec -> f64
  %f = md.gather_relation %n, %x, %cell exchange(antisymmetric, derived) {
    ...
  } : !pairs, !vec -> !vec
  md.return %u, %f : f64, !vec
}
```

The two ops are separate at this level. Fusing them into one loop is an
`md_exec` decision.

### 5.6 Passes

| Pass | Effect |
|---|---|
| `md-check-exchange` | Proves the exchange contracts whose basis is `proof`. Fails if a proof fails. |
| `md-expand-truncation` | Expands truncation into the kernels. |
| `md-differentiate` | Generates derivative functions and replaces `md.evaluate` with `md.call`. Expands truncation in the generated functions. |

A generated function is named after the potential and the requests:
`@lj.energy_forces`, `@lj.derivative3`. Equal requests share one function.

When the energy combines several sums, each sum is weighted by the derivative
of the energy with respect to that sum, and the force fields of the sums are
added with `md.map_particles`.

Restrictions of the current implementation:

| Restriction | Reason |
|---|---|
| A kernel that uses the displacement `d` cannot be differentiated with respect to positions. | Only the geometry rule for the distance exists. |
| A parameter derivative is taken with respect to a scalar argument only. | Per-particle parameters would need a field-valued result. |
| The body of a potential must be a single block. | |

The numerical values of the generated kernels are tested. A test pass turns
each kernel into a function, which is lowered and run, and the results are
compared with closed-form expressions to a relative tolerance of 1e-12. The
test covers the Lennard-Jones kernel with each truncation kind: energy,
forces, virial, and a parameter derivative. The reference values for
`force_switch` come from the formulas of the GROMACS manual.

## 6. `dyn` ops

### 6.1 Summary

| Op | Definition |
|---|---|
| `dyn.kick %v, %f, %m, %dt` | `v'_i = v_i + dt · f_i / m_i` |
| `dyn.drift %x, %v, %dt` | `x'_i = x_i + dt · v_i` |
| `dyn.program` | Defines one time step. |
| `dyn.step` | Applies a program. |
| `dyn.return` | Terminator. |

Force evaluation is `md.evaluate`. `dyn` has no op of its own for it.

### 6.2 Positions and periodicity

`dyn.drift` does not wrap positions. At the semantic level a position is a
point in space, and periodicity enters only through the minimum image.

Whether stored positions are wrapped into the cell is a storage decision.
When they are, the storage keeps image counters so that unwrapped positions
can be reconstructed.

### 6.3 `dyn.program`

```mlir
dyn.program @velocity_verlet(%x: !vec, %v: !vec, %f: !vec,
                             %m: !real, %cell: !md.cell, %dt: f64,
                             %eps: f64, %sigma: f64)
    -> (!vec, !vec, !vec)
    attributes {provides = ["symplectic", "time_reversible"]} {
  %c    = arith.constant 0.5 : f64
  %half = arith.mulf %c, %dt : f64
  %v1 = dyn.kick  %v,  %f, %m, %half : !vec
  %x1 = dyn.drift %x,  %v1, %dt      : !vec
  %f1 = md.evaluate @lj(%x1, %cell, %eps, %sigma) request [forces]
          : (!vec, !md.cell, f64, f64) -> !vec
  %v2 = dyn.kick  %v1, %f1, %m, %half : !vec
  dyn.return %x1, %v2, %f1 : !vec, !vec, !vec
}
```

| Attribute | Meaning |
|---|---|
| `requires` | What the program needs from the thermodynamic state, such as a temperature. |
| `provides` | Properties of the program, such as `symplectic`, `time_reversible`, or `thermostatting`. Neither integrator conserves energy exactly, so neither claims to (B9). |
| `velocity_offset` | Time of the stored velocities relative to the positions, in units of `dt`. Zero if absent. |

### 6.4 Leapfrog

```mlir
dyn.program @leapfrog(%x: !vec, %v: !vec,
                      %m: !real, %cell: !md.cell, %dt: f64,
                      %eps: f64, %sigma: f64)
    -> (!vec, !vec)
    attributes {velocity_offset = -0.5,
                provides = ["symplectic", "time_reversible"]} {
  %f  = md.evaluate @lj(%x, %cell, %eps, %sigma) request [forces]
          : (!vec, !md.cell, f64, f64) -> !vec
  %v1 = dyn.kick  %v, %f, %m, %dt : !vec
  %x1 = dyn.drift %x, %v1, %dt    : !vec
  dyn.return %x1, %v1 : !vec, !vec
}
```

| | Velocity Verlet | Leapfrog |
|---|---|---|
| Stored velocity | `v(t)` | `v(t − dt/2)` |
| Force carried between steps | Yes | No |
| Kicks per step | Two half kicks | One full kick |

The two generate the same positions when they start from the same physical
state. The stored velocities differ, so the initial velocities must be
mapped (B6):

```text
v(−dt/2) = v(0) − (dt/2) · F(0) / m
```

An observable that uses velocities must read `velocity_offset`: with
leapfrog, the kinetic energy at time `t` is computed from the velocities on
both sides of `t`.

### 6.5 Segment

The step loop uses upstream `scf.for` with the state as loop-carried values.
No `dyn` op is needed for it.

```mlir
func.func @run_segment(%x0: !vec, %v0: !vec, %f0: !vec, %m: !real,
                       %cell: !md.cell, %dt: f64, %eps: f64, %sigma: f64,
                       %n: index) -> (!vec, !vec, !vec) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %x, %v, %f = scf.for %s = %c0 to %n step %c1
      iter_args(%xa = %x0, %va = %v0, %fa = %f0) -> (!vec, !vec, !vec) {
    %xb, %vb, %fb = dyn.step @velocity_verlet(%xa, %va, %fa, %m, %cell, %dt,
                                              %eps, %sigma)
        : (!vec, !vec, !vec, !real, !md.cell, f64, f64, f64)
        -> (!vec, !vec, !vec)
    scf.yield %xb, %vb, %fb : !vec, !vec, !vec
  }
  return %x, %v, %f : !vec, !vec, !vec
}
```

## 7. Precision (S8)

Precision is a structural plan parameter. The plan assigns a type to each
role.

| Role | Covers |
|---|---|
| `position` | Stored positions |
| `velocity` | Stored velocities |
| `force` | Stored forces |
| `kernel` | Arithmetic inside pair kernels |
| `integrator` | Arithmetic in kick and drift |
| `accumulator` | Energy, virial, and other global sums |

| Mode | `position` | `velocity` | `force` | `kernel` | `integrator` | `accumulator` |
|---|---|---|---|---|---|---|
| `single` | `f32` | `f32` | `f32` | `f32` | `f32` | `f64` |
| `mixed` | `f64` | `f64` | `f32` | `f32` | `f64` | `f64` |
| `double` | `f64` | `f64` | `f64` | `f64` | `f64` | `f64` |

A mode is a default assignment. Each role can be overridden.

In `mixed` mode the displacement `d_ij` is computed in the position type and
then narrowed to the kernel type. The subtraction is the step that loses
precision, so it is done before narrowing.

## 8. `md_exec` ops

### 8.1 Summary

| Op | Purpose |
|---|---|
| `md_exec.build_cells` | Bins particles into cells. |
| `md_exec.spatial_order` | Computes a permutation that orders particles by cell. |
| `md_exec.permute` | Applies a permutation to a field. |
| `md_exec.build_neighbors` | Builds a physical neighbor structure. |
| `md_exec.pair_for` | Runs a kernel over the pairs of a neighbor structure. |
| `md_exec.particle_for` | Runs a kernel over particles. |
| `md_exec.yield` | Terminator of kernels. |

Reductions and accumulation are clauses of the two loop ops, not separate
ops (S6).

Neighbor structures and cells have types from `mdrt`. Those types are
placeholders here, pending the `mdrt` ABI.

### 8.2 `md_exec.build_neighbors`

```mlir
%cells = md_exec.build_cells %x, %cell width(2.8)
           : !vec -> !mdrt.cells<@atoms>
%nl    = md_exec.build_neighbors %cells, %x, %cell
           cutoff(2.5) skin(0.3) kind(matrix) width(96)
           : !mdrt.cells<@atoms>, !vec -> !mdrt.neighbors<@atoms>
```

`width` is the number of neighbors that the structure holds per particle.
The only kind so far is `matrix`, a row of fixed width per particle.

The structure is built at a reference configuration `x_ref` with the extended
cutoff `r_c + skin`. It holds the list

```text
L = { (i, j) : i ≠ j, |d_ij(x_ref)| < r_c + skin }
```

**Validity.** The structure is valid for a configuration `x` when the cell is
unchanged and

```text
max_i |x_i − x_ref,i| ≤ skin / 2
```

Under that condition `L ⊇ D(N(x))`: no pair within the cutoff is missing.

**Rebuild policy (B1).**

| Policy | Behavior | Exact |
|---|---|---|
| `check` | The validity condition is evaluated every step, before the forces. The structure is rebuilt when it fails. | Yes |
| `interval(n)` | The structure is rebuilt every `n` steps with no check in between. | Only if the condition happened to hold |

`check` is the default. `interval` must be selected explicitly.

With `interval`, the maximum displacement since the previous rebuild is
measured at each rebuild, and violations are counted and reported (A11). A
violation means that pairs may have been missed in earlier steps. That cannot
be repaired afterward, which is why `interval` is not the default.

### 8.3 `md_exec.pair_for`

```mlir
%f0 = md_exec.zeros : !vec
%u0 = arith.constant 0.0 : f64

%f, %u = md_exec.pair_for %nl, %x, %cell
           outs(%f0 : !vec) reduce(%u0 : f64)
           cutoff(2.5) weights [0.5]
           policy(directed, owner_only) {
^bb0(%r2: f64, %d: vector<3xf64>):
  ...
  md_exec.yield %k_f, %k_u : vector<3xf64>, f64
} : !mdrt.neighbors<@atoms>, !vec -> !vec, f64
```

| Clause | Meaning |
|---|---|
| `ins` | Fields that are read. The kernel receives two values per field. |
| `outs` | Destination fields. The kernel yields a contribution to the central particle. |
| `reduce` | Global sums. |
| `weights` | One weight per global sum. All 1 if absent. |
| `cutoff` | The predicate `r² < r_c²`. The loop evaluates it, not the kernel. |
| `policy` | The pair execution policy from the plan. |

The loop computes `d` and `r²` from the position field and the cell. The
kernel receives `r²`, not `r`, so that a kernel with only even powers needs
no square root.

Semantics with `traversal = directed`:

```text
a_i = a0_i + Σ_{(i,j) ∈ L, r_ij < r_c} k_a(i, j)
S   = S0   + w · Σ_{(i,j) ∈ L, r_ij < r_c} k_S(i, j)
```

### 8.4 `md_exec.particle_for`

```mlir
%v0 = md_exec.empty : !vec

%v1 = md_exec.particle_for ins(%v, %f, %m : !vec, !vec, !real)
        outs(%v0 : !vec) {
^bb0(%v_i: vector<3xf64>, %f_i: vector<3xf64>, %m_i: f64):
  ...
  md_exec.yield %v_new : vector<3xf64>
} -> !vec
```

It has the same `ins`, `outs`, and `reduce` clauses as `md_exec.pair_for`,
with one difference: a field in `outs` is written, not accumulated into.

```text
b_i = k_b(i)                 for every field in outs
S   = S0 + Σ_i k_S(i)        for every value in reduce
```

`md_exec.zeros` and `md_exec.empty` provide destinations: a field of zeros
for a loop that accumulates, and a field with unspecified values for a loop
that writes every value.

### 8.5 Value form and storage form

Each loop op has two forms (D17).

| | Value form | Storage form |
|---|---|---|
| Operands | Field values | Storage handles |
| Results | One per `outs` and `reduce` operand | None for `outs`; the destination is updated |
| Ordering | By data dependency | By data dependency and `!mdrt.event` |

## 9. Lowering `md` and `dyn` to `md_exec`

### 9.1 Op mapping

| Semantic op | `md_exec` |
|---|---|
| `md.neighborhood` | `md_exec.build_cells` and `md_exec.build_neighbors` |
| `md.sum_relation` | `md_exec.pair_for` with a `reduce` clause |
| `md.gather_relation` | `md_exec.pair_for` with an `outs` clause |
| `md.sum_particles` | `md_exec.particle_for` with a `reduce` clause |
| `md.map_particles` | `md_exec.particle_for` with an `outs` clause |
| `dyn.kick`, `dyn.drift` | `md_exec.particle_for` with an `outs` clause |

The conversion builds the neighbor structure where the neighborhood was, so
the structure is rebuilt at every evaluation. That is exact and slow. Moving
the build out of the evaluation and reusing the structure across steps, under
the rebuild policy of Section 8.2, is the task of a later pass.

The semantic kernel is written in terms of the distance `r`, and the loop
provides `r²`. The conversion inserts a square root at the start of the
kernel when the kernel uses `r`.

### 9.2 Correctness of the directed traversal

**Sum.** For a kernel that is invariant under exchange:

```text
Σ_{{i,j} ∈ R} k(i, j) = ½ · Σ_{(i,j) ∈ D(R)} k(i, j)
```

Each unordered pair appears twice in `D(R)` with the same kernel value. The
lowering therefore sets `weight = 0.5`.

**Gather.** The definition of `md.gather_relation` is already a sum over
`D(R)` for each central particle. The lowering writes only to the central
particle and needs no weight.

**Neighbor list.** When the structure is valid, `L ⊇ D(N(x))`, and the cutoff
predicate removes exactly the tuples of `L` that are not in `D(N(x))`. A
traversal of `L` with the predicate equals a traversal of `D(N(x))`. The
`check` rebuild policy guarantees validity at every force evaluation.

Together: in double precision and without reassociation, the lowered program
computes the same sums as the definitions in Section 5, in a different
order. The results agree up to the rounding of that reordering.

### 9.3 Traversing each pair once

This policy is not part of M0. It is stated here because the semantic
definitions must allow it.

```text
for {i, j} in R:
    a_i += k(i, j)
    a_j += s · k(i, j)        s = +1 for symmetric, −1 for antisymmetric
    S   += k_S(i, j)          weight = 1
```

It requires `exchange` to be `symmetric` or `antisymmetric`, and a conflict
strategy for the write to `a_j`.

### 9.4 Fusion

Loops over the same neighbor structure may be fused into one
`md_exec.pair_for` with several `outs` and `reduce` clauses. Common
subexpressions of the kernels are then shared.

## 10. Storage assignment

Storage assignment turns the value form into the storage form. It runs inside
`md_exec`.

### 10.1 What it decides

| Decision | Input |
|---|---|
| Which buffer holds each field value | Use-def graph |
| Which updates happen in place | Number of consumers of each value |
| Layout of each buffer | Plan |
| Element type of each buffer | Precision policy |

### 10.2 The rule for in-place updates

A loop may write its result into the buffer of its destination operand when
that operand has no other consumer after the loop.

In the velocity Verlet step, every field value has one consumer:

```text
%v  ──kick──► %v1 ──kick──► %v2         one buffer
%x  ──drift─► %x1                        one buffer
%f  (consumed by the first kick), %f1    one buffer
```

Three buffers serve the whole step, and the loop-carried values of
`scf.for` map to the same three buffers on every iteration.

### 10.3 When one buffer is not enough (B10)

A value needs a buffer of its own when it is still live after the point where
its buffer would be overwritten. Having a second consumer is not the
criterion; being live across the overwrite is.

The remedy is usually a second buffer, not a copy. In a Metropolis step the
proposed positions are written to a second buffer while the old positions
stay where they are. Acceptance then selects one of the two.

Inside the step loop the pass introduces neither a copy nor an extra buffer
silently (D18). It reports the value, the overwrite, and the later consumer.
The front end accepts by marking the value, or the program is rejected.

### 10.4 Accumulation from zero

A force field produced by `md.gather_relation` has no old value. Its
destination is a buffer filled with zeros. When several loops contribute to
the same force field, the first starts from zeros and the others accumulate
into the result of the one before.

## 11. Requirements on `mdrt`

The `mdrt` ABI is not designed yet. M0 needs these services from it.

| Service | Used by |
|---|---|
| Allocate and free field storage on host and device | Storage assignment |
| Cell storage with capacity management | `md_exec.build_cells` |
| Neighbor storage with capacity management and overflow reporting | `md_exec.build_neighbors` |
| Sort, prefix sum, and compaction | Cell and neighbor builds, reordering |
| Host access to field storage, with version counters | Driver, Python library |
| Trajectory and energy output | Driver |

## 12. Validation

The reference interpreter evaluates the semantic dialects directly from the
definitions in Sections 4 to 6, in `f64`.

| Check | Compared against | Tolerance depends on |
|---|---|---|
| Energy and forces of one configuration | Reference interpreter | Precision mode |
| Forces | Finite differences of the energy, in the interpreter | Step size |
| Energy and forces | A reference MD engine | Precision mode |
| Energy conservation over a run | Drift bound | Time step, precision mode |
| Velocity Verlet against leapfrog | Positions of the two runs, with initial velocities mapped as in Section 6.4 | Precision mode |

The energy conservation check uses `truncation(force_shift)` or
`truncation(switch)`. With `truncation(none)` the energy jumps at every
cutoff crossing, and the check would not measure the integrator or the back
end.
