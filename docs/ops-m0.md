# MDIR Op Specification, Milestone M0

Status: draft 18 (2026-09-29). Everything in this document is implemented,
except where a section says otherwise. A Lennard-Jones system runs end to
end in single, mixed, and double precision, on the CPU sequentially and
with OpenMP, and on NVIDIA GPUs.

This document specifies the types and ops needed for milestone M0: a
Lennard-Jones [[Jones1924]](references.md#jones1924) fluid integrated with velocity Verlet
[[Swope1982]](references.md#swope1982) or leapfrog [[HockneyEastwood1988]](references.md#hockneyeastwood1988), on one node,
on CPU and GPU. It covers `md`, the minimal `dyn`, and `md_exec`.

Operand lists, result lists, and the mathematical definitions are normative.

| Part | State |
|---|---|
| `md` types and ops (Section 4) | Implemented. The examples show the actual syntax. |
| Truncation, differentiation, exchange check (Sections 4.7, 4.8, 5) | Implemented as passes; see Section 5.6 |
| `dyn` ops (Section 6) | Implemented. The examples show the actual syntax. |
| `md_exec` ops in the value form and in the storage form (Section 8) | Implemented. The examples show the actual syntax. |
| Conversion of `md` and `dyn` to `md_exec` (Section 9.1) | Implemented as the pass `convert-md-to-md-exec` |
| Storage assignment (Section 10) | Implemented as the pass `md-exec-assign-storage` |
| Lowering of the storage form to loops (Section 10.7) | Implemented as the pass `convert-md-exec-to-loops` |
| Lowering of the storage form to GPU kernels (Section 10.8) | Implemented as the pass `convert-md-exec-to-gpu`, for NVIDIA |
| Fusion of loops over pairs and of loops over particles (Section 9.4) | Implemented as the pass `md-exec-fuse-loops` |
| Powers of the squared distance (Section 9.5) | Implemented as the pass `md-exec-simplify-distance` |
| The test of validity as a loop over particles (Section 9.6) | Implemented as the pass `md-exec-expose-validity` |
| The order of the particles (Section 8.6) | Implemented; the driver emits the ops |
| Precision policy (Section 7) | Implemented as the pass `md-exec-assign-precision` |

It follows the accepted decisions in [decisions.md](decisions.md). Tags such
as (S1) or (B4) name the decision behind a section.

## 1. Notation

| Symbol | Meaning |
|---|---|
| `P` | A particle set. Each particle has a global ID. |
| $a : P \to V$ | A per-particle field with values in $V$. $a_i$ is its value at particle $i$. |
| `h` | The simulation cell: three lattice vectors and a periodicity flag per direction. |
| $\mathbf d_{ij}$ | Minimum-image [[AllenTildesley2017]](references.md#allentildesley2017) displacement $\mathbf x_i - \mathbf x_j - h\mathbf n$, with integer $\mathbf n$ chosen to minimize the length. |
| $r_{ij}$ | $\lVert\mathbf d_{ij}\rVert$. |
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

$$D(R) = \{ (i, j), (j, i) : \{i, j\} \in R \}.$$

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
reference.

`f64` is the only floating-point type allowed at the semantic level. The
ops of `md` and `dyn` reject fields of `f32`, and so do the signatures of
potentials, functions, and programs.

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
| `reduced` | $\sigma$ | $\tau$ | $m$ | $\varepsilon$ |

The `md` system is the one OpenMM uses.

## 3. Kernel bodies

A kernel is a region evaluated once per particle or per tuple. It contains
only pure ops from `arith`, `math`, and `vector`, and ends with a yield.

### 3.1 Expression syntax

Energy expressions are strings in the syntax of OpenMM custom forces
[[Eastman2017]](references.md#eastman2017) (D22).

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
| NVPTX, through the code generator alone | Arithmetic, `math.fpowi`, and `math.sqrt` lower. `exp`, `log`, `sin`, `cos`, `tanh`, `powf`, `asin`, `acos`, `atan2`, `erf`, and `erfc` do not: the target has no library to call. |
| NVPTX, through the `gpu` dialect | The conversion to `nvvm` turns the functions into calls to the device math library of the CUDA toolkit, `libdevice`, which is linked as bitcode. A kernel with `sqrt`, `exp`, `erfc`, `fpowi`, and `roundeven` on `f64` and on vectors of `f64` ran on a GPU and agreed with the host. |

The device math library is found through the environment variable
`CUDA_ROOT`, `CUDA_HOME`, or `CUDA_PATH`, which names the CUDA toolkit. It
is needed when the kernels are compiled, not when LLVM is built.

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
| `md.tuple_set`, `md.disjoint_union`, `md.sum_tuples`, `md.gather_tuples` | The tuples of a topology, and the loops over them; see [design-m1.md](design-m1.md), Sections 2 and 3. |
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
$U(\mathbf x; \theta)$. It has exactly one result, of type `f64`. Kinetic energy is not
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

$$N(\mathbf x, h) = \{ \{i, j\} : i \ne j,\ r_{ij} < r_c \}.$$

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

$$S = \sum_{t \in R} k(t).$$

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

$$a_i = \sum_{j : (i, j) \in D(R)} k(i, j).$$

The kernel is evaluated with $i$ as the central particle. `d` is $\mathbf d_{ij}$.

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

### 4.8 Truncation (B4)

A potential that is cut off at `r_c` jumps by `u(r_c)` whenever a pair
crosses the cutoff. Energy is then not conserved, whatever the integrator.
`truncation` states how the kernel `u` of a relation sum is modified.

| `truncation` | Kernel used for `r < r_c` | Continuity at `r_c` |
|---|---|---|
| `none` | $u(r)$ | None |
| `shift` | $u(r) - u(r_c)$ | Energy |
| `force_shift` [[Stoddard1973]](references.md#stoddard1973) | $u(r) - u(r_c) - (r - r_c)\, u'(r_c)$ | Energy and force |
| `switch`, from $r_s$ | $u(r)\, S(r)$ | Energy, force, and the derivative of the force |
| `force_switch` [[Steinbach1994]](references.md#steinbach1994), from $r_s$ | $u(r) - P(r) - C$ | Energy, force, and the derivative of the force |

`r_c` is the cutoff of the neighborhood.

**Switch.** The potential is multiplied by the fifth-degree polynomial of
`potential-switch` in the GROMACS manual [[GromacsManual2025]](references.md#gromacsmanual2025):

$$
S(r) = \begin{cases} 1 & r \le r_s \\ 1 - 10t^3 + 15t^4 - 6t^5 & r_s < r < r_c, \end{cases} \qquad t = \frac{r - r_s}{r_c - r_s}.
$$

**Force switch.** A cubic polynomial in $r - r_s$ is added to the force, so
that the force and its derivative vanish at the cutoff. With
$F = -u'(r_c)$, $F' = -u''(r_c)$, and $\Delta = r_c - r_s$:

$$
\begin{aligned}
A &= \frac{F'\Delta - 3F}{\Delta^2}, \qquad B = \frac{2F - F'\Delta}{\Delta^3}, \\
P(r) &= \begin{cases} 0 & r \le r_s \\ \tfrac{A}{3}(r - r_s)^3 + \tfrac{B}{4}(r - r_s)^4 & r_s < r < r_c, \end{cases} \\
C &= u(r_c) - \tfrac{A}{3}\Delta^3 - \tfrac{B}{4}\Delta^4.
\end{aligned}
$$

The definition holds for any kernel. It is linear in $u$, so applying it to a
sum of terms equals applying it to each term.

**Correspondence with other packages.**

| MDIR | OpenMM | GROMACS |
|---|---|---|
| `shift` | — | `potential-shift` |
| `switch` | Switching function | `potential-switch` |
| `force_switch` | — | `force-switch` |
| `force_shift` | — | — |

The GROMACS manual [[GromacsManual2025]](references.md#gromacsmanual2025) defines `force-switch` for a power law $r^{-\alpha}$. For that
kernel the constants above reduce to the ones in the manual.

A pass expands the attribute into the kernel before differentiation. After
the pass every relation sum has `truncation(none)` and an explicit kernel.
`force_shift` and `force_switch` use the scalar derivative rules of
Section 5.4 to form `u'(r_c)` and `u''(r_c)`.

The expansion is a pass, not a front-end task, so that both front ends share
it and the derivative rules exist once.

Long-range dispersion corrections [[Shirts2007]](references.md#shirts2007) are not part of M0.

### 4.9 `md.sum_particles` and `md.map_particles`

```mlir
%ke = md.sum_particles gather(%v, %m : !vec, !real) {
^bb0(%v_i: vector<3xf64>, %m_i: f64):
  ...
  md.yield %k : f64
} : f64
```

$$
\begin{aligned}
\texttt{md.sum\_particles}: \quad & S = \sum_{i \in P} k(i), \\
\texttt{md.map\_particles}: \quad & b_i = k(i).
\end{aligned}
$$

### 4.10 `md.evaluate`

```mlir
%u, %f = md.evaluate @lj(%x, %cell, %eps, %sigma) request [energy, forces]
           : (!vec, !md.cell, f64, f64) -> (f64, !vec)
```

| Request | Result type | Definition |
|---|---|---|
| `energy` | `f64` | $U$ |
| `forces` | Position field type | $\mathbf F_i = -\partial U/\partial\mathbf x_i$ |
| `virial` | `vector<9xf64>` | $\mathsf W$, as defined in Section 5.3, in row-major order |
| `derivative(n)` | `f64` | $\partial U/\partial\theta_n$, where $\theta_n$ is scalar argument `n` |

`md.evaluate` does not survive semantic differentiation. The pass replaces it
with `md.call` to a generated `md.function`.

```mlir
%u, %f = md.call @lj.energy_forces(%x, %cell, %eps, %sigma)
           : (!vec, !md.cell, f64, f64) -> (f64, !vec)
```

## 5. Semantic differentiation for M0

This section traces the Lennard-Jones potential [[Jones1924]](references.md#jones1924) from its
definition to the force kernel. Every step is an identity.

### 5.1 Energy

$$U(\mathbf x) = \sum_{\{i,j\} \in N(\mathbf x)} u(r_{ij}), \qquad u(r) = 4\varepsilon \big((\sigma/r)^{12} - (\sigma/r)^6\big).$$

This is `md.sum_relation` over an unordered relation. Each pair is counted
once. $u$ here is the kernel after truncation has been expanded.

### 5.2 Forces

The set $N(\mathbf x)$ is treated as locally constant. When the kernel is continuous
at the cutoff, the result is the exact gradient of $U$. With
`truncation(none)` it is the gradient everywhere except at configurations
where a pair crosses the cutoff, where $U$ itself jumps. With

$$\frac{\partial r_{ij}}{\partial\mathbf x_i} = \frac{\mathbf d_{ij}}{r_{ij}}, \qquad \frac{\partial r_{ij}}{\partial\mathbf x_j} = -\frac{\mathbf d_{ij}}{r_{ij}},$$

the force on particle $i$ is

$$\mathbf F_i = -\frac{\partial U}{\partial\mathbf x_i} = \sum_{j : (i,j) \in D(N)} \mathbf K(i, j), \qquad \mathbf K(i, j) = -u'(r_{ij})\, \frac{\mathbf d_{ij}}{r_{ij}}.$$

This is `md.gather_relation`. The factor in front of $\mathbf d_{ij}$ depends only on
$r_{ij}$, and $\mathbf d_{ji} = -\mathbf d_{ij}$, so $\mathbf K(j, i) = -\mathbf K(i, j)$. The pass sets
`exchange(antisymmetric)`.

For Lennard-Jones:

$$\mathbf K(i, j) = \frac{24\varepsilon}{r^2} \big(2(\sigma/r)^{12} - (\sigma/r)^6\big)\, \mathbf d_{ij}.$$

### 5.3 Virial and parameter derivatives

$$
\begin{aligned}
\mathsf W &= \sum_{\{i,j\} \in N} \mathbf d_{ij} \otimes \mathbf K(i, j), \\
\frac{\partial U}{\partial\theta} &= \sum_{\{i,j\} \in N} \frac{\partial u}{\partial\theta}(r_{ij}).
\end{aligned}
$$

Both are `md.sum_relation`. The virial kernel is invariant under exchange
because both factors change sign.

**Sign convention (B8).** $\mathsf W$ is the MDIR virial [[Louwerse2006]](references.md#louwerse2006). $\mathbf K(i, j)$ is the force on
$i$ due to $j$, so $\mathsf W$ is positive for repulsion. The pressure is

$$P = \frac{2 E_\text{kin} + \operatorname{tr}\mathsf W}{3V}.$$

Other packages use other conventions. The GROMACS virial is $-\mathsf W / 2$
[[GromacsManual2025]](references.md#gromacsmanual2025).

### 5.4 Rules

The pass needs two kinds of rules.

| Kind | M0 content |
|---|---|
| Geometry | For a pair with distance geometry: $\partial r/\partial\mathbf x_i = \mathbf d/r$, $\partial r/\partial\mathbf x_j = -\mathbf d/r$ |
| Scalar | One derivative rule per `arith` and `math` op in Section 3.2 |

Functions that are not smooth have fixed conventions (B5). `x'` denotes the
derivative of the argument.

| Function | Derivative |
|---|---|
| `select(c, a, b)` | `select(c, a', b')`. The condition is not differentiated. |
| `step(x)`, `delta(x)`, `floor(x)`, `ceil(x)` | 0 |
| `abs(x)` | `x'` if $x \ge 0$, else `−x'` |
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
| `md-bypass-updates` | Lets a loop over the tuples of a set of an `md.disjoint_union` read a field before an update by an other set of the union: where a field is $\mathbf y = \mathbf x + \mathbf c$, with $\mathbf c$ gathered over an other set, the loop reads $\mathbf x$, which agrees with $\mathbf y$ at its members up to the sign of a zero (D83; the union is in [design-m1.md](design-m1.md), Section 2). |

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
`force_switch` come from the formulas of the GROMACS manual
[[GromacsManual2025]](references.md#gromacsmanual2025).

## 6. `dyn` ops

### 6.1 Summary

| Op | Definition |
|---|---|
| `dyn.kick %v, %f, %m, %dt` | $\mathbf v'_i = \mathbf v_i + \Delta t\, \mathbf f_i / m_i$ |
| `dyn.drift %x, %v, %dt` | $\mathbf x'_i = \mathbf x_i + \Delta t\, \mathbf v_i$ |
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
| `provides` | Properties of the program, such as `symplectic`, `time_reversible`, or `thermostatting`. Neither integrator conserves energy exactly [[Hairer2003]](references.md#hairer2003), so neither claims to (B9). |
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

$$\mathbf v(-\Delta t/2) = \mathbf v(0) - \frac{\Delta t}{2}\, \frac{\mathbf F(0)}{m}.$$

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

## 7. Precision (S8, D31, D32)

Precision is a structural plan parameter. The plan assigns a floating-point
type to each role.

| Role | Covers | Applied by |
|---|---|---|
| `position` | Stored positions | Whoever allocates the state; the pass, for positions that no buffer holds |
| `velocity` | Stored velocities | Whoever allocates the state |
| `force` | Fields that loops over pairs write | The pass |
| `kernel` | Arithmetic in loops over pairs | The pass |
| `integrator` | Arithmetic in loops over particles | The pass |
| `accumulator` | Energy, virial, and other global sums | The pass |

| Mode | `position` | `velocity` | `force` | `kernel` | `integrator` | `accumulator` |
|---|---|---|---|---|---|---|
| `single` | `f32` | `f32` | `f32` | `f32` | `f32` | `f64` |
| `mixed` | `f64` | `f64` | `f32` | `f32` | `f64` | `f64` |
| `double` | `f64` | `f64` | `f64` | `f64` | `f64` | `f64` |

A mode is a default assignment. Each role can be overridden.

### 7.1 Where precision is assigned (D31)

The pass `md-exec-assign-precision` assigns the types. It runs on the value
form of `md_exec`, after the transformations of Section 9 and before
storage assignment.

```text
md, dyn                          f64 only
  ↓ convert-md-to-md-exec
md_exec                          f64 only
  ↓ reuse, fusion, powers of the squared distance
  ↓ md-exec-assign-precision
md_exec                          f32 and f64
  ↓ md-exec-assign-storage
md_exec in the storage form
  ↓ convert-md-exec-to-loops
```

Everything before the pass works on the reference program and needs no
knowledge of precision.

```text
--md-exec-assign-precision="mode=mixed"
--md-exec-assign-precision="mode=single accumulator=f32"
--md-exec-assign-precision="mode=mixed kernel-positions=false"
```

Where the kernels compute in a type narrower than the positions, the pass
converts the positions once, in a loop over particles before the first
loop of a block that needs them, and the loops over pairs and over tuples
that are not disjoint take that field (D79). The difference
$\mathbf d = \mathbf x_i - \mathbf x_j$ is then taken in the type of the
kernel; the loops over disjoint tuples, those of the constraints, take the
positions as they are stored.

### 7.2 Buffers state the type of the state (D32)

The state enters and leaves the compiled program through buffers. The
element type of a buffer is the type that the field is stored in. At the
semantic level the field has `f64` whatever the buffer holds:

```mlir
%x = mdrt.from_buffer %positions : memref<?x3xf32> to !md.field<@atoms, 3 x f64>
```

The field has the values of the buffer. The pass gives the field the type
of the buffer:

```mlir
%x = mdrt.from_buffer %positions : memref<?x3xf32> to !md.field<@atoms, 3 x f32>
```

Whoever allocates the state, which is the driver, applies the roles
`position` and `velocity`, and the type of any other stored field such as
masses or charges. The compiled program follows the buffers. It does not
convert a buffer to another type, which would be a copy that nothing asked
for (D18).

### 7.3 The type of a field

Fields that must have one type are **stored together**:

| Stored together | Reason |
|---|---|
| A loop-carried field, the field it is initialized with, the field it is updated with, and the result of the loop | One buffer is carried |
| The field that a loop accumulates into, and the result of the loop | The loop continues in the buffer |
| The values that the `return` ops of a function return in one position | One result type |

A destination from `md_exec.zeros` or `md_exec.empty` holds no field yet. It
takes the type of the result of the loop that writes to it.

The type of the fields that are stored together is the first of these that
applies:

| # | Condition | Type |
|---|---|---|
| 1 | A buffer holds one of them | The element type of the buffer |
| 2 | One of them is used as positions | `position` |
| 3 | A loop over pairs writes one of them | `force` |
| 4 | A loop over particles writes one of them | `integrator` |
| 5 | The three roles above have one type | That type |

If none applies, the pass fails: nothing tells how the field is stored. If
two buffers that disagree hold fields that are stored together, the pass
fails as well.

The signature of a function follows its arguments and the values it
returns. Fields of integers keep their type.

### 7.4 The type of a kernel

| Loop | Computes in |
|---|---|
| `md_exec.pair_for` | `kernel` |
| `md_exec.particle_for` | `integrator` |

Every floating-point value inside the kernel gets that type, constants
included. Values are converted where they cross the boundary of the kernel.

| Value | Converted |
|---|---|
| The value of a field in `ins` | At the start of the kernel, from the type that the field is stored in |
| A value from outside the kernel | Before the loop, once. A constant is replaced by a constant of the new type. |
| A contribution to a field in `outs` | At the end of the kernel, to the type that the field is stored in |
| A contribution to a global sum | At the end of the kernel, to `accumulator` |

The squared distance and the displacement arrive in the type of the kernel.
The loop takes the difference of the positions in their type, where the
subtraction would lose precision, and converts it; the minimum image, the
squared distance, and the test of the cutoff follow in the type of the
kernel. For a pair across the cell the rounding of the image is then about
10⁻⁷ of the edge in `f32`. A device computes in `f64` at a small fraction
of the rate of `f32`: with the minimum image in `f64`, the forces of the
target of D65 took 172 microseconds a step on an RTX 3090, and 76 with it
in `f32`.

In the mixed mode, a kick reads forces of `f32` and velocities of `f64`:

```mlir
%v1 = md_exec.particle_for
        ins(%v, %f, %m : !md.field<@atoms, 3 x f64>,
                         !md.field<@atoms, 3 x f32>,
                         !md.field<@atoms, f64>)
        outs(%v0 : !md.field<@atoms, 3 x f64>) {
^bb0(%v_i: vector<3xf64>, %f_i: vector<3xf32>, %m_i: f64):
  %wide = arith.extf %f_i : vector<3xf32> to vector<3xf64>
  ...
  md_exec.yield %v_new : vector<3xf64>
} -> !md.field<@atoms, 3 x f64>
```

### 7.5 Global sums and scalars

Scalars outside the loops keep the type `f64`: parameters, the time step,
the cell, and the results of global sums. If `accumulator` is `f32`, the
initial value is converted before the loop and the result after it.

Inside a loop over pairs, the contributions of the neighbors of one particle
are added up in the type of what they are added to: `force` for a field,
`accumulator` for a global sum.

### 7.6 Neighbor structures

A neighbor structure is built and tested in the type of the positions. The
template of Section 8.2 is written for `f64`; for positions of `f32` the
compiler adds an instance in which the type is replaced, under names that
end in `_f32`.

### 7.7 Agreement with the reference

Lowering to single or mixed precision relaxes the numerical semantics
(B2). The integration tests compare runs of 200 steps with the values of
the reference precision.

| Mode | Relative tolerance of the test | Agreement found |
|---|---|---|
| `double` | 1e-9 | 1e-13 |
| `mixed` | 1e-6 | 1e-7 |
| `single` | 1e-5 | 1e-6; 1e-5 for the kinetic energy |

On the CPU the modes differ little in speed so far: 0.83 s, 0.82 s, and
0.78 s for 4096 particles and 200 steps, sequentially. The loops over pairs
are not vectorized across pairs, so narrower values do not yet mean more
values per instruction. On a GPU the mixed mode takes half the time of the
double mode (Section 10.9).

### 7.8 Limitations

| Limitation | Consequence |
|---|---|
| A call of a function that takes or returns fields is not supported. | The pass fails. The lowering does not support such calls either. |
| The role `velocity` is not visible to the pass. | A velocity field that no buffer holds gets the type of `integrator`, by rule 4. |
| A role cannot be declared for an argument of a function. | A function that takes a field and does nothing that reveals its role cannot be assigned a precision in the mixed mode. |

## 8. `md_exec` ops

### 8.1 Summary

| Op | Purpose |
|---|---|
| `md_exec.build_cells` | Bins particles into cells. |
| `md_exec.spatial_order` | Computes the order of the particles by cell, and within a cell by their numbers. |
| `md_exec.permute` | Returns a field in an order. |
| `md_exec.build_neighbors` | Builds a physical neighbor structure. |
| Memory of the device that a program frees | Kept by `mdrt` for a request of the same size: the work that a block is handed to runs after the work that used it, on the one stream, so neither the free nor the reuse waits for the device |
| `md_exec.empty_neighbors` | A neighbor structure that is valid for no configuration. |
| `md_exec.refresh_neighbors` | Returns a neighbor structure that is valid for a configuration, building one only if the one given is not. |
| `md_exec.reference_positions` | The configuration that a neighbor structure was built at. |
| `md_exec.rebuild_count` | The number of times a neighbor structure has been built. |
| `md_exec.zeros`, `md_exec.empty` | Destinations of loops. |
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

`width` is the number of neighbors that the structure holds per particle
at first; a build that finds a particle with more makes the rows wider
(the runtime holds them). The kinds are `matrix`, a row per particle, and
`groups` (D89), lists of groups of 16 in blocks that grow as well.

The structure is built at a reference configuration $\mathbf x^\text{ref}$ with the extended
cutoff $r_c + s$, $s$ the skin. It holds the list

$$L = \{ (i, j) : i \ne j,\ \lVert\mathbf d_{ij}(\mathbf x^\text{ref})\rVert < r_c + s \}.$$

**Validity.** The structure is valid for a configuration $\mathbf x$ when the cell is
unchanged and [[AllenTildesley2017]](references.md#allentildesley2017)

$$\max_i \lVert \mathbf x_i - \mathbf x^\text{ref}_i \rVert \le s / 2.$$

Under that condition $L \supseteq D(N(\mathbf x))$: no pair within the cutoff is missing.

A structure also becomes invalid when the cell changes.

**Refresh.** A loop carries its neighbor structure and refreshes it before
it is used:

```mlir
%nl0 = md_exec.empty_neighbors kind(matrix) width(96)
         : !mdrt.neighbors<@atoms>

scf.for ... iter_args(..., %nl = %nl0) {
  ...
  %nl1 = md_exec.refresh_neighbors %nl, %x1, %cell
           cutoff(2.5) skin(0.3) cell_width(2.8) policy(check)
           : !mdrt.neighbors<@atoms>, !vec
  ...
  scf.yield ..., %nl1
}
```

The refresh returns the structure it was given if that is valid for the
configuration, and a structure built at the configuration otherwise. The
structure is empty when the loop begins, so the first iteration builds it
(R1).

The pass `md-exec-reuse-neighbors` produces this form from a build in the
body of a loop. It does two more things.

| What | Why |
|---|---|
| A structure moves outward through the loops around it, one loop at a time. | A run has a loop for each period of output. The structure lives on from one output to the next. |
| It stops at a loop that has the attribute `mdrt.segment`. | The iterations of that loop are segments of the run. A structure starts empty in each (R1). |
| Structures of one block that are built with the same parameters, one after the other, share their storage. The later build becomes a refresh of what the earlier one left. | The step that returns the energy and the steps that do not are different code with the same neighbors. |

Two structures share storage only if the first is no longer used where the
second begins. The parameters are the cell, the cutoff, the skin, the width
of the cells, and the kind and width of the structure.

**Rebuild policy (B1).**

| Policy | Behavior | Exact | State |
|---|---|---|---|
| `check` | The validity condition is evaluated at every refresh. The structure is rebuilt when it fails. | Yes | Implemented |
| `interval` with `interval(n)` | The structure is rebuilt at the first refresh and every `n` refreshes after a build, with no check in between. The condition is tested at each build, before it, and a build that finds it failed is counted. | Only if the condition happened to hold | Implemented (D88) |

`check` is the default. `interval` must be selected explicitly: the pass
`md-exec-rebuild-at-interval`, which the driver runs only for
`rebuild_interval` in the control file, and warns.

**Who makes the test.** A refresh with the policy `check` makes the test
itself, unless it is given the result as `moved`:

```mlir
%ref   = md_exec.reference_positions %nl : !mdrt.neighbors<@atoms> -> !vec
%moved = md_exec.particle_for ins(%x1, %ref : !vec, !vec)
           reduce(%false : i1) {
^bb0(%x_i: vector<3xf64>, %ref_i: vector<3xf64>):
  ...   // |x_i − ref_i|² > (skin / 2)²
  md_exec.yield %far : i1
} -> i1
%nl1   = md_exec.refresh_neighbors %nl, %x1, %cell moved(%moved)
           cutoff(2.5) skin(0.3) cell_width(2.8) policy(check)
           : !mdrt.neighbors<@atoms>, !vec
```

`md_exec.reference_positions` returns `x_ref`. For a structure that has not
been built its values are unspecified; the refresh builds such a structure
whatever `moved` says. Section 9.6 has the reasons for this form.

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
| `cutoff` | The predicate $r^2 < r_c^2$. The loop evaluates it, not the kernel. |
| `exchange` | One exchange contract for each value, the destinations and then the sums, from the `md.gather_relation` or `md.sum_relation` of the value (Section 4.7); `none` for all if absent. Fusion keeps them with their values. A loop may compute a pair once only if every value is `symmetric` or `antisymmetric` (Section 9.3, D89). |
| `policy` | The pair execution policy from the plan. |

The loop computes `d` and $r^2$ from the position field and the cell. The
kernel receives $r^2$, not `r`, so that a kernel with only even powers needs
no square root.

$r^2$ has the type `f32` or `f64`, and `d` is a vector of the same type: the
type that the kernel computes in (Section 7.4). The values of a field
arrive in the type that the field is stored in.

Semantics with `traversal = directed`:

$$
\begin{aligned}
a_i &= a^0_i + \sum_{(i,j) \in L,\ r_{ij} < r_c} k_a(i, j), \\
S &= S^0 + w \sum_{(i,j) \in L,\ r_{ij} < r_c} k_S(i, j).
\end{aligned}
$$

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

$$
\begin{aligned}
b_i &= k_b(i) && \text{for every field in } \texttt{outs}, \\
S &= S^0 + \sum_i k_S(i) && \text{for every number or vector in } \texttt{reduce}, \\
A &= A^0 \lor \bigvee_i k_A(i) && \text{for every value of the type } \texttt{i1} \text{ in } \texttt{reduce}.
\end{aligned}
$$

A value of the type `i1` in `reduce` tells whether the kernel yields true
for any particle. `md_exec.pair_for` has sums only.

`md_exec.zeros` and `md_exec.empty` provide destinations: a field of zeros
for a loop that accumulates, and a field with unspecified values for a loop
that writes every value.

### 8.5 Value form and storage form

The ops that touch fields are one set with two forms (D17), in the manner
of upstream `linalg`. An op is in the storage form if its fields are
buffers. An op takes fields only or buffers only.

| | Value form | Storage form |
|---|---|---|
| Operands | Fields | The buffers that hold the fields, of type `memref` (D33). A buffer on a device has a memory space in its type: `memref<?x3xf64, 1>`. |
| Results | One per `outs` and `reduce` operand | One per `reduce` operand; a buffer in `outs` is updated where it is |
| Effects | None. The op can be removed, moved, and merged like any pure op. | Reads and writes of the buffers, declared to the upstream analyses |
| Ordering | By data dependency | By the order of the ops in their block (A12) |

```mlir
// Value form
%f, %u = md_exec.pair_for %nl, %x, %cell
           outs(%f0 : !vec) reduce(%u0 : f64)
           cutoff(2.5) weights [0.5] policy(directed, owner_only) { ... }
         : !mdrt.neighbors<@atoms>, !vec -> !vec, f64

// Storage form
%u = md_exec.pair_for %nl, %x, %cell
       outs(%f : memref<?x3xf64>) reduce(%u0 : f64)
       cutoff(2.5) weights [0.5] overwrite [true]
       policy(directed, owner_only) { ... }
     : !mdrt.neighbors<@atoms>, memref<?x3xf64> -> f64
```

| Op | In the storage form |
|---|---|
| `md_exec.pair_for` | `overwrite` has one flag per buffer in `outs`. With the flag, the loop ignores what the buffer holds, as if it held zeros. Without it, the loop adds to what the buffer holds. |
| `md_exec.particle_for` | A buffer may be in `ins` and in `outs`: the kernel of a particle reads and writes the values of that particle only. |
| `md_exec.empty_neighbors` | Takes `size` and `positions` and allocates the storage of a structure for that many particles. `positions` is the type of the buffers that hold the positions; the storage is where they are. |
| Every op | `scratch` holds buffers that the op may use as it likes. A lowering to a device needs them for global sums and maxima. |
| `md_exec.refresh_neighbors` | Rebuilds the structure where it is. The result is the structure that was given. |
| `md_exec.reset_neighbors` | Storage form only. Makes the structure valid for no configuration. It stands where the value form has an empty structure inside a loop. |
| `md_exec.join` | Storage form only. The op before it that is marked `md_exec.side` runs on a second stream beside the ops between the two, which must be independent of it; the ops after the join run after it (`md-exec-assign-streams`, D87). |
| `md_exec.zeros`, `md_exec.empty`, `md_exec.build_cells`, `md_exec.build_neighbors`, `md_exec.tabulate` | Do not occur. A table that `md_exec.tabulate` makes (`md-exec-fold-tables`, D91) is computed on the host, entry by entry, where its tables are, and is placed as they are. A build is storage and a refresh with the policy `always`. |

The kernels are the same in both forms.

The types `!md.cell` and `!mdrt.neighbors` occur in both forms. A cell is a
small value. A neighbor structure is a value in the value form and a handle
to storage in the storage form; the type does not tell which. For that
reason `md_exec.rebuild_count` always counts as reading the structure.

What the storage form is for: every decision that depends on where a field
is stored is a pass over ops that still are loops over particles and pairs.
The layout of a vector field, the device that holds a buffer, the transfers
between host and device, and whether a loop becomes a CPU loop or a GPU
kernel are such decisions.

### 8.6 `md_exec.spatial_order` and `md_exec.permute`

```mlir
%order = md_exec.spatial_order %x, %cell, %ids width(1.4)
           : !vec, !md.field<@atoms, i32> -> !mdrt.permutation<@atoms>
%xs    = md_exec.permute %x, %order
           : !vec, !mdrt.permutation<@atoms> -> !vec
```

`order[k]` is the particle that comes to place `k`: the particles follow
one another by the cell that they are in, and within a cell by `ids`. The
cells are `width` wide or wider and numbered along x first.

```text
xs_k = x_order[k]
```

| Rule | Reason |
|---|---|
| `ids` holds a number for each particle that no other particle has. | The order is then a function of the positions and of `ids` alone. It does not depend on the order that the particles are in, which a run that continues from a checkpoint does not share with the run before (R1). |
| Every field of the particle set is put in the order, `ids` included. | A loop reads the fields of a particle at one index. |
| A neighbor structure that was built before is not valid for the fields in the order. | It holds indices. The driver orders the particles where a segment begins, and a segment begins with empty structures. |

In the storage form the order is a buffer with one `i32` per particle,
and both ops write to a buffer in `outs`:

```mlir
md_exec.spatial_order %x, %cell, %ids outs(%order : memref<?xi32>)
    width(1.4) : memref<?x3xf64>, memref<?xi32>
md_exec.permute %x, %order outs(%xs : memref<?x3xf64>)
    : memref<?x3xf64>, memref<?xi32>
```

Storage assignment gives a field in the order a buffer of its own. In a
loop, the buffer that held the field before is handed back with the
buffers that the loop borrowed (Section 10.4), so the two change places
from one iteration to the next.

[neighbors-m0.md](neighbors-m0.md), Section 5, has the method and the
measurements.

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

The conversion builds the neighbor structure where the neighborhood was. In
the body of a loop that would be a build in every iteration; the pass
`md-exec-reuse-neighbors`, run after the conversion, replaces it with a
refresh (Section 8.2).

The semantic kernel is written in terms of the distance `r`, and the loop
provides $r^2$. The conversion inserts a square root at the start of the
kernel when the kernel uses `r`. Section 9.5 describes how it is removed
again.

### 9.2 Correctness of the directed traversal

**Sum.** For a kernel that is invariant under exchange:

$$\sum_{\{i,j\} \in R} k(i, j) = \tfrac12 \sum_{(i,j) \in D(R)} k(i, j).$$

Each unordered pair appears twice in $D(R)$ with the same kernel value. The
lowering therefore sets `weight = 0.5`.

**Gather.** The definition of `md.gather_relation` is already a sum over
`D(R)` for each central particle. The lowering writes only to the central
particle and needs no weight.

**Neighbor list.** When the structure is valid, $L \supseteq D(N(\mathbf x))$, and the cutoff
predicate removes exactly the tuples of $L$ that are not in $D(N(\mathbf x))$. A
traversal of $L$ with the predicate equals a traversal of $D(N(\mathbf x))$. The
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
    a_j += s * k(i, j)        s = +1 for symmetric, -1 for antisymmetric
    S   += k_S(i, j)          weight = 1
```

It requires `exchange` to be `symmetric` or `antisymmetric`, and a conflict
strategy for the write to `a_j`.

### 9.4 Fusion

The pass `md-exec-fuse-loops` fuses two `md_exec.pair_for` ops into one when

- they run over the same neighbor structure, positions, and cell, with the
  same cutoff and policy;
- the second does not use a result of the first; and
- no op between the two uses a result of the first.

The fused loop has the fields, destinations, and global sums of both. A
field that both kernels read is read once. The kernel is the first kernel
followed by the second; common subexpression elimination, run afterward,
removes what the two compute twice.

Fusion does not change any result: every sum receives the same
contributions in the same order.

The pass fuses two `md_exec.particle_for` ops as well, when they run over
the particles of one set. Here the second may read a field that the first
writes. A loop over particles computes the values of a particle from the
values of that particle alone, so the second kernel reads what the first
has yielded:

```mlir
%v1 = md_exec.particle_for ins(%v, %f, %m) outs(%e0) { ... }   // kick
%x1 = md_exec.particle_for ins(%x, %v1) outs(%e1) { ... }      // drift
```

```mlir
%v1, %x1 = md_exec.particle_for ins(%v, %f, %m, %x) outs(%e0, %e1) {
  ...   // the kick, then the drift with the velocity of the kick
}
```

| Rule | Reason |
|---|---|
| The second loop uses a result of the first only as a field that it reads. | A global sum of the first loop is complete only when the loop is. |
| The fused loop is where the second loop was, if every other user of the first comes after it; else where the first was, if everything that the second uses is there. | With a loop over pairs between a drift and a kick, neither holds: the forces need the positions of all particles. |
| A field that only the second loop reads is not stored. | It is a value inside the kernel. |

With velocity Verlet, the kick and the drift of a step become one loop. In
a step that computes energies, the second kick and the kinetic energy
become one loop.

### 9.5 Powers of the squared distance

The loop provides $r^2$, and a kernel that was written in terms of `r` starts
with a square root. The pass `md-exec-simplify-distance` removes the square
root where the kernel does not need it.

The pass writes every `f64` value of a kernel as a sum of terms

$$c\, r^p,$$

where $c$ is a product of a number and of values that carry no explicit
power of $r$. It carries that form through sums, differences, products,
quotients, and integer powers. Code is emitted only where a value is needed:
for the operands of any other op, and for what the kernel yields.

A product of two sums is not multiplied out. Each of the two is computed
and enters the product as a value. Multiplying out would write a polynomial
in `r − a`, such as a switching function, in powers of `r`. The terms of
that form are much larger than the value of the polynomial, and their sum
loses the digits that the polynomial had: four digits in `f64` for the
switching function of Section 4.8, and nearly all of them in `f32`.

The product of a sum with a single term is multiplied out. It scales every
term of the sum by the same factor and loses nothing.

The factors of a term are multiplied in the order in which the kernel first
uses them. The code that the pass emits is the same in every run of the
compiler, so a program that is compiled twice computes the same bits.

| Power | Computed from |
|---|---|
| Even and positive | $r^2$ |
| Even and negative | $1 / r^2$, which is computed once |
| Odd | The even power below it, times $\sqrt{r^2}$, which is computed once |

For the Lennard-Jones potential without truncation, or with `shift`, energy
and force contain only even powers, and the kernel has no square root. With
`switch`, `force_shift`, or `force_switch`, the truncation itself is a
function of `r`, and the square root stays.

The pass reassociates floating-point arithmetic. Results change in their
last bits, so the pass is not part of the reference semantics (B2) and must
not be run in the deterministic mode.

### 9.6 The test of validity (D41)

A neighbor structure is valid while no particle has moved more than half
the skin since the structure was built (Section 8.2). The test is needed in
every step.

As a global maximum of the displacements the test is a loop of its own
and, on a device, three kernels: the displacements, the maxima of chunks,
and the maximum of those. Nothing needs the maximum. The refresh needs to
know whether one particle is beyond the limit.

The pass `md-exec-expose-validity` makes the test a loop over particles
that yields, for each particle, whether it is beyond the limit
(Section 8.2). `md-exec-fuse-loops` then fuses it with the loop that writes
the positions, the drift. The thread that moves a particle tests it.

| | Global maximum | Test in the drift |
|---|---|---|
| Work for a particle | The displacement, in a loop of its own | The displacement, where the position is written |
| Combining | A reduction over all particles | None. A thread whose particle is beyond the limit sets a flag; the others write nothing. |
| Kernels in a step of velocity Verlet on a device | 6 | 3: kick, drift, and test; forces; kick |
| What the host reads in a step | One number | One flag |
| Steps in which the structure is built | | The same |

The host still reads one value in every step, because it decides whether
to build. The read is a copy of four bytes. The host waits there for the
device, but the device does not wait for the host: the kernels of the
step are queued behind one another.

A refresh of a structure that is empty where the refresh is builds it
whatever has moved. The pass gives it the policy `always`.

## 10. Storage assignment

Storage assignment gives every field a buffer. The pass
`md-exec-assign-storage` converts the ops from the value form to the
storage form (Section 8.5). The pass `convert-md-exec-to-loops` then turns
the storage form into loops (Section 10.7).

### 10.1 What becomes what

| Value form | Storage form |
|---|---|
| A field | `memref<?x3xT>` or `memref<?xT>`, with the element type of the field |
| A field in the signature of a function | A buffer |
| `mdrt.from_buffer`, `mdrt.to_buffer` | Nothing: the buffer is the field |
| `md_exec.zeros`, `md_exec.empty` | A buffer from the pool, or `memref.alloc` |
| `md_exec.particle_for`, `md_exec.pair_for` | The same op on buffers |
| `md_exec.build_cells` and `md_exec.build_neighbors` | `md_exec.empty_neighbors` with storage, in the body of the function, and `md_exec.refresh_neighbors` with the policy `always` |
| `md_exec.empty_neighbors` | The same op with storage, in the body of the function; one for each use of the empty structure. Inside a loop, `md_exec.reset_neighbors` where the structure is used. |
| `scf.for` that carries fields | `scf.for` that carries buffers |
| `scf.for` that carries a neighbor structure | The loop does not carry it: the structure is refreshed where it is |
| A cell | Unchanged |

The pass follows the types that the precision policy assigned (Section 7).
A field whose type differs from that of the buffer that holds it is
rejected.

A function without fields is left as it is.

### 10.2 Regions, ownership, and the pool

The pass works region by region: the body of a function, and the body of
each loop inside it. A region **owns** the buffers that it may overwrite.

| Region | Buffers it owns |
|---|---|
| Body of a function | The buffers of its field arguments, the buffers handed over with `mdrt.from_buffer`, and the buffers it allocates |
| Body of a loop | The buffers that the loop carries |

A field is **dead** after the last op of its region that uses it. When a
field dies, its buffer goes to the **pool** of its region, from which a
later op may take it.

### 10.3 Choosing the buffer of a result

| Loop | Rule |
|---|---|
| Over particles | Kernel `i` reads and writes particle `i` only. The loop writes to the buffer of a field that it reads, if the region owns the buffer and the field is dead afterward. Otherwise it takes a buffer from the pool. |
| Over pairs | The loop reads the positions and the fields in `ins` of other particles, so it cannot write to their buffers. It takes a buffer from the pool. |

When the destination of a loop over pairs is `md_exec.zeros`, the loop stores
the sum for each particle and never reads the buffer. The buffer is not
filled with zeros first.

In the velocity Verlet step, three buffers serve the whole step:

```text
%v  ──kick──► %v1 ──kick──► %v2         in place, one buffer
%x  ──drift─► %x1                        in place, one buffer
%f  dies at the first kick; %f1          takes the buffer of %f from the pool
```

### 10.4 Loops that carry fields

A loop that carries fields carries their buffers. When the body needs a
buffer and its pool is empty, the buffer becomes one more loop-carried
value, initialized with a buffer from the enclosing region. The body never
allocates.

At the end of the body, the loop yields the buffers of the fields it
carries, and then as many unused buffers as it borrowed. Which buffer plays
which role may change from one iteration to the next.

The leapfrog step carries positions and velocities. The forces are needed
only within the step, so their buffer is a third loop-carried value:

```mlir
scf.for ... iter_args(%x = ..., %v = ..., %spare = ...)
    -> (memref<?x3xf64>, memref<?x3xf64>, memref<?x3xf64>) {
  // forces into %spare; kick and drift in place in %v and %x
  scf.yield %x, %v, %spare
}
```

### 10.5 When a field needs a buffer of its own (B10, D18)

The pass never copies a field. It fails, and names the op, in these cases:

| Case | Example |
|---|---|
| A loop updates a field that is used after the loop | The initial positions are read after the step loop. |
| A loop yields a field that belongs to an enclosing region | A loop yields a field that it did not carry. |
| A loop yields one field twice | |
| A loop over pairs accumulates into a field that it reads from other particles | |

A field that is merely read after an op keeps its buffer, and the result of
the op goes to another buffer. Outside loops that buffer is allocated where
it is needed.

A neighbor structure is refreshed where it is. If the structure that a
refresh takes is used after the refresh, the pass fails: the structure would
need storage of its own.

### 10.6 Limitations

| Limitation | Consequence |
|---|---|
| Buffers are never freed. | A function that allocates leaks when it is called repeatedly. The tests run everything from one `main`. |
| Marking a field so that a copy is accepted is not implemented. | A program that needs a copy, such as a Metropolis step, cannot be lowered. |
| A neighbor structure is refreshed where it is. | A loop cannot carry two structures and exchange them. |
| Ops run in the order of their block. | Nothing runs asynchronously. Event tokens come with asynchronous execution (A12). |

### 10.7 From the storage form to loops

The pass `convert-md-exec-to-loops` replaces every op where it is. It
decides nothing about buffers.

| Storage form | Loops |
|---|---|
| `md_exec.particle_for` | `scf.parallel` over the particles |
| `md_exec.pair_for` | `scf.parallel` over the particles, with an `scf.for` over the neighbors of each |
| `md_exec.empty_neighbors` | The buffers of a neighbor matrix: counts, indices, the configuration and the cell of the last build, a flag, and a count of builds |
| `md_exec.refresh_neighbors` | The test of validity and, where it fails, a call to the neighbor build template |
| `md_exec.reset_neighbors` | Two stores |
| `md_exec.rebuild_count` | A load |
| A cell | `vector<3xf64>`, the edge lengths of an orthorhombic cell |

### 10.8 From the storage form to GPU kernels

With `memory=device`, storage assignment puts the buffers on a device.

| Value form | Storage form on a device |
|---|---|
| `mdrt.from_buffer` | A buffer on the device, and a copy from the host to it. The buffer of the host is kept for what is copied back. |
| `mdrt.to_buffer` | A copy from the device to a buffer of the host. The field stays on the device. |
| A loop with global sums | The loop with two buffers in `scratch` for each sum |
| A refresh that tests validity | The refresh with two buffers in `scratch` |

Everything between the two copies stays on the device. A loop over steps
moves nothing between host and device, except one number for each global
sum and each test of validity.

The buffers in `scratch` come from the pool of the region, like any other
buffer. A loop around the op carries them, so the lowering allocates
nothing inside a loop over steps (D18, B10).

The pass `convert-md-exec-to-gpu` replaces every op where it is, with ops
of the upstream `gpu` dialect.

| Storage form | Kernels |
|---|---|
| `md_exec.particle_for`, `md_exec.pair_for` | One kernel with one thread per particle, in blocks of 128 threads. With the policy `owner_only` a thread writes only to its own particle, so the kernel needs no atomic operation. A loop over pairs or tuples has 16 threads per particle, adjacent in a warp: each takes every 16th entry of the row of neighbors or of tuples, the group sums its values by shuffles (`gpu.shuffle xor`, 64-bit numbers as two words), and its first thread writes. Every thread of a launched block runs the loop, those beyond the last particle with no entry, so that the warps stay whole for the shuffles. |
| The global sums of a loop | The kernel stores the contributions of each particle. A second kernel has a block of 128 threads for each part of the particles, at least 512 particles and at most 128 parts: a thread adds up every so manyth particle, and the block adds up its threads by a tree (`gpu.all_reduce`). A third kernel, one block, adds up the parts the same way, and the host reads the results with one copy. The order of a sum depends only on the number of particles. All sums of the loop share the two kernels and the copy. Each sum has slots of its own in one buffer of results for each type, and the copy and the reads move down to where the host first needs a result; a copy that then follows another with nothing between that runs on the device is left out, since the first brought the whole buffer. The sums of a step of coupling, read one after another, then reach the host in one or two copies. A contribution is a number or a vector: the virial is a vector of nine numbers, and its buffers in `scratch` hold nine numbers per particle. |
| Loops over pairs and tuples one after another | One kernel, where the group of threads of a particle does each loop in turn, when the loops take the same positions, none reads what an earlier one writes, and their buffers for global sums are their own; their sums share the kernels of the sums. The forces of a step, the loop over pairs and those over bonds, angles, dihedrals, and the like, are one kernel. |
| A value that tells whether the kernel yields true for any particle | A flag on the device. A thread that yields true sets it. Every thread that writes it writes the same value, so the threads need not take turns. The host reads the flag after the kernel and clears it where it was set. |
| `md_exec.reference_positions` | The buffer of the structure that holds the positions |
| `md_exec.empty_neighbors` | The buffers of a neighbor matrix on the device. The flag and the count of builds are on the host. |
| `md_exec.refresh_neighbors` | Where the structure is not valid, a call to the neighbor build template for devices. Without `moved`, the test of validity before it, with the largest displacement as a global maximum. The excluded pairs are marked, not taken out: a thread for each entry of the matrix replaces an excluded neighbor by the particle itself, which the loops over pairs skip. Taking them out moved the rest of a row, one thread per particle, and took as long as the forces. |
| A cell | `vector<3xf64>`. A kernel takes numbers and buffers as arguments, so a vector from outside enters a kernel as its elements. |
| A constant | A constant of the kernel, not an argument. A power whose exponent is an argument would be a loop. |
| A loop of the host | The loop releases, at the end of every iteration, the stack that the iteration has taken. The lowering of a launch puts the arguments on the stack where the launch is; in a loop over steps the stack would grow until it overflows. |

The order in which a global sum is added up is fixed: by particle within a
chunk, then by chunk. The sum is the same in every run. It differs from
the sum on the host in its last bits, because the host adds up in another
order.

A chunk has as many particles as the square root of their number, and
between 32 and 1024. One thread adds up a chunk and one thread the
chunks, so both take the time of that many additions: with 864 particles
29, where chunks of 256 took nine times as long.

| A step of `examples/argon/argon.toml` that computes energies | Kernels | Copies | Milliseconds |
|---|---|---|---|
| A pair of kernels and a copy for each sum, chunks of 256 | 11 | 5 | 0.33 |
| A pair of kernels and a copy for each loop | 7 | 3 | 0.27 |
| Chunks of the square root | 7 | 3 | 0.22 |
| A step that computes forces only | 3 | 1 | 0.11 |

A thermostat needs the kinetic energy in every step, so this is what a
step of a run at constant temperature costs on a device.

The neighbor build template for devices,
`lib/Runtime/Templates/NeighborsMatrixGPU.mlir`, builds the matrix in 10
kernels, or in 12 where the search is split.
[neighbors-m0.md](neighbors-m0.md) describes the method. The matrix is the
one that the template for the host builds, entry by entry.

The result is lowered by the upstream pipeline
`gpu-lower-to-nvvm-pipeline`. The kernels become PTX text inside the
program, and the driver compiles them when the program starts. Math
functions come from the device math library of the CUDA toolkit.

The lowering waits after every launch. The runtime library does not: all
launches and copies go to one stream, which runs them in the order in
which they were issued, and the host waits only where it reads what the
device has computed, that is, at a copy to the host. The results are the
same, bit for bit. With `MDRT_WAIT` set, the library waits wherever the
lowering does.

| Limitation | Consequence |
|---|---|
| The build allocates its work buffers at every build and frees them. | The cost is per build, not per step. |
| A split search takes one number for each of its threads. | The search is split only for small systems. |
| One device | |
| NVIDIA only | The storage form does not depend on the vendor; the lowering to `rocdl` is not written. |

### 10.9 Run times

Milliseconds per step, for the Lennard-Jones system of the tests at the
density 0.58, with velocity Verlet, a cutoff of 2.0, and a skin of 0.2. The
time is measured inside the program, around 200 steps; it includes the
rebuilds and excludes the start of the program.

| Particles | 1 thread | 16 threads | GPU, double | GPU, mixed |
|---|---|---|---|---|
| 216 | 0.127 | 0.0282 | 0.147 | 0.0558 |
| 512 | 0.341 | 0.0377 | 0.145 | 0.0557 |
| 1728 | 1.19 | 0.0859 | 0.149 | 0.0593 |
| 4096 | 2.81 | 0.175 | 0.15 | 0.0607 |
| 13824 | 8.49 | 0.607 | 0.285 | 0.104 |
| 32768 | 19.9 | 1.41 | 0.464 | 0.175 |
| 110592 | 68.7 | 4.69 | 1.24 | 0.412 |
| 262144 |  | 10.7 | 2.77 | 0.819 |

The host is a machine with 128 cores, the GPU an RTX 3090. The numbers are
the least of three runs. On the GPU the precision matters: the mixed mode
takes less than half the time of the double mode.

Below some thousand particles the time on the GPU does not depend on the
number of particles: it is the time of the kernels of a step, each of
which takes as long as its slowest thread. Below a thousand particles the
host with 16 threads is faster.

Starting the program on a GPU takes about one second, for the context of
the driver and the compilation of the kernels.

#### Where the time goes on a GPU

With `MDRT_PROFILE` set, the runtime library for devices reports the
number and the time of its calls when the program ends; with `MDRT_WAIT`
set as well, the time of each kernel. For `examples/argon/argon.toml` on the
GPU in the mixed mode, 864 particles with 74 neighbors each:

| Kernel | Microseconds, before | Now |
|---|---|---|
| Kick and drift | 7 | 8, with the test of validity |
| Test of validity: displacements, chunks, the maximum | 7, 24, 6 | |
| Forces | 165 | 95 |
| Kick | 7 | 7 |
| Neighbor build, all kernels, for each step | 97 | 7 |
| Of a build, the search | 1600 | 40 |
| A step | 330 | 110 |

A structure is built every 18 steps. "Before" is the state in which the
loops over particles were fused already.

| Change | Time of a step, in microseconds |
|---|---|
| | 330 |
| The kick and the drift are one loop (Section 9.4). | 320 |
| The host does not wait for every kernel, and a constant is a constant of the kernel (Section 10.8). | 280 |
| The minimum image is found with a multiplication (D42). | 170 |
| The test of validity is made in the drift (Section 9.6). | 150 |
| The neighbor build searches a copy of the positions, with narrower cells and more threads ([neighbors-m0.md](neighbors-m0.md)). | 110 |

| Observation | Consequence |
|---|---|
| A kernel that does next to nothing takes 6 to 7 microseconds. | Every kernel that is saved saves that much. |
| The time of a small system is the time of one thread, not of all. The kernel of the forces runs 864 threads at once; each visits its neighbors one after another, at 0.9 microseconds for a neighbor. | The time per step hardly depends on the number of particles until the device is full. A small system would need more than one thread per particle. |
| The search of a build tested every particle of 27 cells. With 27 cells in all, that was every particle. | The width of the cells is chosen from the density, and the search of a small system has a thread for each row of cells ([neighbors-m0.md](neighbors-m0.md)). |
| A division in double precision costs a thread of a device much more than a multiplication. | The minimum image took three divisions for each neighbor, and nearly half the time of the forces. |
| The kick after the forces writes to the particle that the thread of the forces has. | The two can be one kernel. A step then has two. |

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

There is no reference interpreter (V1). Validation rests on tests that run
compiled code and compare with reference values from independent sources,
and, from M1 on, on comparison with an established MD engine.

| Test | Compared against | Tolerance |
|---|---|---|
| Kernels that differentiation generates | Closed-form derivatives; for `force_switch`, the formulas of the GROMACS manual [[GromacsManual2025]](references.md#gromacsmanual2025) | 1e-12 |
| Neighbor build template | A search over all pairs | Exact |
| A neighbor structure that a loop refreshes, 100 steps | Pairs within the cutoff at every step, by a search over all pairs; the number of builds | Exact |
| Energy, forces, and virial of 64 particles | A script that evaluates all pairs | 1e-10 |
| Energy, virial, and pressure of a mixture of 256 particles, through the driver | A script that evaluates all pairs | 4 decimals of the log |
| 200 steps of velocity Verlet and of leapfrog | The same script, integrating with all pairs | 1e-9 |
| The same in the mixed mode | The same values | 1e-6 |
| 200 steps of velocity Verlet in the single mode | The same values | 1e-5 |
| All of the above on a GPU | The same values | The same tolerances |
| Neighbor build template for devices | The matrix that the template for the host builds | Exact, order included |

All but the first run sequentially and with OpenMP on 4 threads, except the
mixed mode, which runs sequentially. The scripts are
in `test/Integration/Inputs`.

The checks that were planned:

| Check | Compared against | Tolerance depends on |
|---|---|---|
| Energy and forces | A reference MD engine | Precision mode |
| Energy conservation over a run | Drift bound | Time step, precision mode |
| Velocity Verlet against leapfrog | Positions of the two runs, with initial velocities mapped as in Section 6.4 | Precision mode |

The energy conservation check uses `truncation(force_shift)` or
`truncation(switch)`. With `truncation(none)` the energy jumps at every
cutoff crossing, and the check would not measure the integrator or the back
end.
