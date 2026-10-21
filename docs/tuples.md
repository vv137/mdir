# Terms over the Tuples of a Topology

Status: kept current; last checked against the code on 2026-10-21. The
decisions are D47 to D49 and those that amended them
([decisions.md](decisions.md)); the structures of the tuples that a device
builds are in [device-tuples.md](device-tuples.md), and the robust
differentiation of every kernel in [ad-robust.md](ad-robust.md).

This document describes the relations that come from a topology, the
kernels over their tuples in internal coordinates, their differentiation,
their execution, the exclusions and scaled pairs, and CMAP. It follows
[architecture.md](architecture.md) and extends [ops.md](ops.md), whose
Section 5 differentiates the terms over pairs. IR snippets show the
proposed syntax.

## 1. Relations of a topology

### 1.1 Tuple sets

In M0 a relation comes from a neighborhood. Its tuples are found anew
whenever the structure is built, and a tuple has no properties of its own.

A bond is different: it exists throughout the run, and it has parameters.
The bonds of a system are therefore a set of entities, as the particles
are.

```mlir
md.particle_set @atoms
md.tuple_set @bonds  on(@atoms) arity(2) orientation(unordered)
md.tuple_set @angles on(@atoms) arity(3) orientation(reversal)
```

| Value | Type | Holds |
|---|---|---|
| The members of the tuples | `!md.relation<@atoms, 3, reversal, @angles>` | For each tuple, the places of its particles in the fields |
| A parameter of the tuples | `!md.field<@angles, f64>` | One value for each tuple |

The type of the relation names the tuple set, so that an op can tell that
a field belongs to the tuples of the relation that it takes.

`disjoint` on a tuple set states that no particle is a member of two of
its tuples, as in the groups of a constraint; a loop over the tuples may
then evaluate each tuple once and write to all its members. A disjoint
union states the same across sets (D83):

```mlir
md.tuple_set @settles on(@atoms) arity(3) orientation(ordered) disjoint
md.tuple_set @shake1  on(@atoms) arity(2) orientation(unordered) disjoint
md.disjoint_union @constraints on(@atoms) of [@settles, @shake1]
```

For tuple sets $T_1, \dots, T_n$ of the union, with $P(t)$ the particles of
a tuple $t$, the union states

$$
P(t) \cap P(t') = \emptyset \quad \text{for all } t \ne t' \in T_1 \cup \dots \cup T_n .
$$

Each set must be `disjoint` and on the particle set of the union. The
driver declares the groups of SETTLE and SHAKE as one union after checking
their members. What `md.gather_tuples` over one set gathers is then zero at
the members of the tuples of every other set of the union, which
`md-bypass-updates` uses ([ops.md](ops.md), Section 5.6).

Both enter the program from buffers, as the fields of the particles do
(D32):

```mlir
%angles = mdrt.from_buffer %members
            : memref<?x3xi32> to !md.relation<@atoms, 3, reversal, @angles>
%k      = mdrt.from_buffer %k_buffer
            : memref<?xf64> to !md.field<@angles, f64>
```

The members are the places of the particles in their fields, as the
program holds them. Where the particles are put in a new order (D44), the
members are renumbered with them. A loop then needs no table from the
numbers of the particles to their places.

### 1.2 Orientation

An orientation is the group of permutations under which two tuples are
the same tuple (Section 2.2 of ops.md).

| Orientation | Arity | The same tuple | Example |
|---|---|---|---|
| `unordered` | 2 | `{i, j}` and `{j, i}` | Bonds, exclusions |
| `ordered` | Any | No other | Virtual sites (the site, then its atoms), the waters of SETTLE, the groups of SHAKE with two or three hydrogens, CMAP; the IR allows arity 1 as well, which the driver does not use: position restraints are fields of the particles ([dynamics.md](dynamics.md), Section 5) |
| `reversal` | 2 or more | `(i, j, k)` and `(k, j, i)` | Angles, dihedrals |

### 1.3 Parameters for each tuple, not for each kind

| | For each tuple | For each kind of tuple |
|---|---|---|
| What a tuple holds | Its parameters | The number of its kind; a table has the parameters |
| Memory | More | Less |
| Reads in a kernel | One for each parameter | One for the kind, then one for each parameter |
| Parameters that differ from tuple to tuple, such as the point that a restraint pulls to | As any other | Not possible |

Proposal: parameters for each tuple. A front end that wants kinds can have
them with the same ops: the kind is a field of the tuple set, and the
parameters are a table ([nonbonded.md](nonbonded.md), Section 1).

## 2. Kernels in internal coordinates

### 2.1 The coordinates of a tuple

A kernel over a neighborhood takes the distance and the displacement of
the pair (B7). A kernel over a relation of a topology takes the internal
coordinates that the op names. The ops are `md.sum_tuples` and
`md.gather_tuples`; they are apart from `md.sum_relation` and
`md.gather_relation` because exchange and truncation, which those two
carry, have no meaning for a tuple of a topology. The same ops also sum
the triplets that `md.triplets` derives from a neighborhood, a relation of
arity 3 of the positions rather than of the topology (D160).

```mlir
%u = md.sum_tuples %angles, %x, %cell
       coordinates(cosine(0, 1, 2))
       tuple(%k, %c0 : !md.field<@angles, f64>, !md.field<@angles, f64>) {
^bb0(%c: f64, %k_t: f64, %c0_t: f64):
  %half = arith.constant 0.5 : f64
  %dc   = arith.subf %c, %c0_t : f64
  %sq   = arith.mulf %dc, %dc : f64
  %hk   = arith.mulf %half, %k_t : f64
  %e    = arith.mulf %hk, %sq : f64
  md.yield %e : f64
} : !md.relation<@atoms, 3, reversal, @angles>, !vec -> f64
```

| Coordinate | Value | Kernel argument |
|---|---|---|
| `distance(a, b)` | $\lVert\mathbf d_{ab}\rVert$ | `f64` |
| `displacement(a, b)` | $\mathbf d_{ab} = \mathbf x_a - \mathbf x_b$, in the minimum image [[AllenTildesley2017]](references.md#allentildesley2017) | `vector<3xf64>` |
| `angle(a, b, c)` | The angle at $b$ between $\mathbf d_{ab}$ and $\mathbf d_{cb}$, from $0$ to $\pi$ | `f64` |
| `cosine(a, b, c)` | The cosine of that angle | `f64` |
| `dihedral(a, b, c, d)` | The angle between the planes of $a, b, c$ and of $b, c, d$, from $-\pi$ to $\pi$ | `f64` |

`a`, `b`, `c`, and `d` are places in the tuple. An op may name several
coordinates, such as the angle of three particles and the distance of the
two outer ones.

The result of `md.sum_tuples` is a sum (`f64` or a vector of `f64`), or an
`i1` that tells whether the kernel yields true for some tuple.
`md.gather_tuples` may have a second result, an `i1` of the same meaning,
which its kernel yields after the values of the members: the constraints
of each step test the lengths they give so (D278).
In `md_exec.tuple_for` such a value takes a flag, not buffers for a sum,
and a loop over disjoint tuples with flags only is still of an
integration run (D110).

After the coordinates the kernel takes the fields of the particles that
the op gathers, one value for each member, and the fields of the tuple.

### 2.2 Why not positions

| | Internal coordinates | The positions of the members |
|---|---|---|
| The kernel of an angle | A function of one number | Dot products, norms, and an inverse cosine |
| Differentiation | The derivative of the kernel, times the derivative of the coordinate, which is known in closed form | Rules for vectors: dot and cross products, norms |
| Where the derivative is singular | In the closed form, where it can be treated once | In what the rules generate |
| A term in coordinates that are not listed | Not possible | Possible |

Proposal: internal coordinates. `displacement` covers terms that need the
vector. Kernels over positions can be added when a term needs them.

### 2.3 Exchange

A coordinate is the same for a tuple and for every tuple that the
orientation identifies with it: `cosine(0, 1, 2)` for `(i, j, k)` and
`(k, j, i)`. A kernel that takes only such coordinates and the fields of
the tuple is symmetric, and the compiler can tell. A kernel that gathers
fields of the particles is checked as in M0 (B3).

## 3. Differentiation

### 3.1 With respect to the positions

$$
\begin{aligned}
\mathbf F_m &= -\sum_q \frac{\partial u}{\partial q}\, \frac{\partial q}{\partial\mathbf x_m} && \text{for every member } m \text{ of the tuple}, \\
\mathsf W &= \sum_q \sum_m \mathbf d_m \otimes \mathbf F_{qm} && \text{with the displacement of } m \text{ from one member of the coordinate } q.
\end{aligned}
$$

The virial takes the form of Thompson, Plimpton, and Mattson
[[Thompson2009]](references.md#thompson2009) for terms of more than two particles.

The forces of one coordinate add up to zero, so that the member that the
displacements are taken from does not matter.

| Coordinate | $\partial q/\partial\mathbf x_m$ |
|---|---|
| `distance(a, b)` | $\pm\mathbf d_{ab} / r$ for $a$ and $b$ |
| `cosine(a, b, c)` | For $a$: $\big(\mathbf d_{cb} / \lVert\mathbf d_{cb}\rVert - c\, \mathbf d_{ab} / \lVert\mathbf d_{ab}\rVert\big) / \lVert\mathbf d_{ab}\rVert$ with $c$ the cosine; for $c$ likewise; for $b$ the negative of their sum |
| `angle(a, b, c)` | That of the cosine, times $-1 / \sin\theta$ |
| `dihedral(a, b, c, d)` | The form of Blondel and Karplus [[Blondel1996]](references.md#blondel1996), which has no singularity where three particles are in line |

The derivative of `angle` is singular where the three particles are in
line. A term that is smooth there, such as the one above, should take
`cosine`.

`md-differentiate` produces, for a sum over tuples of arity `k`, an
`md.gather_tuples` whose kernel yields `k` forces, one for each member:

$$a_i = \sum_{t, s : t[s] = i} k(t)[s].$$

The derivative of a coordinate is a function of displacements. The
generated op takes them as further coordinates, after those of the sum:

| Coordinate | Displacements |
|---|---|
| `distance(a, b)` | `d_ab` |
| `angle(a, b, c)`, `cosine(a, b, c)` | `d_ab`, `d_cb` |
| `dihedral(a, b, c, d)` | `d_ab`, `d_bc`, `d_dc` |

```mlir
%f = md.gather_tuples %angles, %x, %cell
       coordinates(cosine(0, 1, 2), displacement(0, 1), displacement(2, 1))
       tuple(%k, %c0 : ...) {
^bb0(%c: f64, %u: vector<3xf64>, %v: vector<3xf64>, %k_t: f64, %c0_t: f64):
  ...
  md.yield %f0, %f1, %f2 : vector<3xf64>, vector<3xf64>, vector<3xf64>
} : !md.relation<@atoms, 3, reversal, @angles>, !vec -> !vec
```

The value of a coordinate and its derivative have norms and products in
common. Both are emitted by the same code, op for op, so that the
elimination of common subexpressions leaves one of each after the
coordinates have been computed in the kernel.

A kernel over tuples that uses a `displacement` $\mathbf d = \mathbf x_a -
\mathbf x_b$ is differentiated as well: the gradient a component at a time,
$\mathbf F_a = -\partial u/\partial\mathbf d$ and $\mathbf F_b = -\mathbf
F_a$, with its share of the virial. A kernel over a neighborhood
(`md.sum_relation`, `md.gather_relation`) that uses the displacement still
cannot be differentiated with respect to the positions, as in M0, except
the kernel of a neural stage, whose product gives the cotangents (D288).

### 3.2 With respect to a parameter

| With respect to | In M1 | How |
|---|---|---|
| An argument of the potential that is a number, such as a scale of the energy or a coupling parameter | Yes | `request [derivative(n)]`, as in M0. The derivative of a sum over tuples is the sum of the derivative of its kernel |
| A field of the particles with one component of `f64`, such as the charges | Yes, since D230 | `request [derivative(n)]` of `md.evaluate` gives a field of its type, $g_i = \partial U/\partial a_i$: a gather of the derivative of each kernel that takes the field, over the relation, the tuples, or the particles; any other use of the field is an error that names the op ([python-gradient.md](python-gradient.md)) |
| A field of the tuples, such as the force constant of every bond | No | The result would be a field of the tuples. `md-differentiate` refuses a field that a kernel takes other than by gathering it from the particles |
| An entry of a table ([nonbonded.md](nonbonded.md), Section 1.1) | No | The result is a table. It needs a sum for each entry |

## 4. Execution of terms over tuples

### 4.1 The strategies

A term over a tuple contributes to `k` particles. P13 left the strategy
open.

| Strategy | Evaluations of a tuple | Writes | Order of a sum |
|---|---|---|---|
| A. Every particle visits the tuples that it is a member of and takes its own part | `k` | To the own particle only | Fixed |
| B. Every tuple is evaluated once and adds to its members with atomic additions | 1 | To `k` particles | That of the threads |
| C. Every tuple is evaluated once; tuples with a particle in common are in different groups, which run one after another | 1 | To `k` particles | Fixed |

Decision: A, as for pairs (P13, D49).

| Reason | |
|---|---|
| It is the strategy of the pairs. | A thread writes to its own particle. Under A, no atomic addition, and the deterministic level holds. |
| Bonded terms are a small part of the work. | An atom of a protein is a member of some 20 bonded tuples and has some 300 neighbors. |
| A loop over tuples of this kind is a loop over particles. | It can be fused with the loop over pairs and with the kick that follows. |

The decision holds on the CPU and in the deterministic mode. On a device
B became the default: a loop over tuples that are not disjoint and that
adds to its destinations evaluates each tuple once and adds to its members
with atomics, where the stated algebra of every value it reduces allows
another order (`evaluatesOnce` of `convert-md-exec-to-gpu`, option
`tuples-once`; D103, D301). A `fixed_order` sum, as the deterministic mode
states, keeps A, and so do the disjoint tuples of the constraints. A run of
at least 100,000 tuples adds what the tuples of a block give each particle
in a table in block memory first, the strategy `hash` of the privatized
scatter-accumulate (D283, D285, [device-tuples.md](device-tuples.md),
Section 11). C is not built.

### 4.2 The structure

```mlir
%incidence = md_exec.build_incidence %angles
    : !md.relation<@atoms, 3, reversal, @angles>
      -> !mdrt.incidence<@atoms, 3, @angles>

%f, %u = md_exec.tuple_for %incidence, %x, %cell
           coordinates(displacement(0, 1), displacement(2, 1))
           tuple(%k, %c0 : !md.field<@angles, f64>, !md.field<@angles, f64>)
           outs(%f0 : !vec) reduce(%u0 : f64) arity(3) {
^bb0(%u: vector<3xf64>, %v: vector<3xf64>, %k_t: f64, %c0_t: f64):
  ...
  md_exec.yield %f_0, %f_1, %f_2, %e
      : vector<3xf64>, vector<3xf64>, vector<3xf64>, f64
} : !mdrt.incidence<@atoms, 3, @angles>, !vec -> !vec, f64
```

| Item | Decision |
|---|---|
| What the structure holds | For each particle a row: for each of its tuples, in ascending order of their numbers, the number, the place of the particle in it, and its members |
| The layout | Ragged rows (CSR, D314): the number of rows, the offset of each row, and the entries, one buffer of i32 of $N + 2 + (a + 2)(M + 1)$ elements for $M$ memberships; read through one accessor ([device-primitives.md](device-primitives.md), Section 4.7). No parameter, and no row that overflows. Until D314 a dense table, every row as wide as the particle with the most tuples needs. |
| When it is built | Where the members change: where the run begins and where the particles are put in a new order (D44), which renumbers the members. It does not depend on the positions or on the cell. The members and the structure are kept by a store of the runtime and taken again where the identities of the particles are those they were built from (D253, D255); on a GPU, kernels of the program build them on the device from the members, by a count, a scan, a scatter, and a sort of each row (D257, D314, [device-tuples.md](device-tuples.md)). |
| The coordinates | `md_exec.tuple_for` takes displacements only, in the minimum image. The kernel computes the other coordinates from them, with the code that differentiation uses (Section 3.1). |
| The kernel | Yields the forces on all members. A loop by the particles (strategy A) takes the one of the particle; a loop that evaluates each tuple once (B, Section 4.1) adds all of them. A kernel for each place, which computes that force only, is a later optimization. |
| A sum | A tuple contributes once, from its member at place 0. No weight, and no rounding from adding a third three times. |

## 5. Exclusions and scaled pairs

### 5.1 Exclusions

```mlir
%n = md.neighborhood %x, %cell cutoff(1.0) exclude(%excluded)
       : !vec -> !pairs
```

$$N(\mathbf x, h) = \{ \{i, j\} : i \ne j,\ r_{ij} < r_c \} \setminus E.$$

| Item | Proposal |
|---|---|
| The relation | `E` is a relation of arity 2 of a tuple set, as the bonds are. The readers give it: from Amber, `EXCLUDED_ATOMS_LIST` as the file has it, built again from the bonds, angles, and dihedrals only where the topology has extra points ([constraints.md](constraints.md), Section 2); from GROMACS, the pairs within `nrexcl` bonds of each `[ moleculetype ]` and those of its `[ exclusions ]`; from CHARMM, the pairs one to three bonds apart and those of `!NNB` ([charmm.md](charmm.md)). |
| Where a pair is excluded | In the build of the neighbor structure, which carries the exclusions (`exclude` of `md_exec.build_neighbors` and `md_exec.empty_neighbors`), so every build applies them. On the CPU a filter after the search removes from each row the particles in its row of exclusions and keeps the others in their order. On a device the search of the matrix enters an excluded neighbor as the particle itself, which a loop over pairs skips ([neighbors.md](neighbors.md), Section 2.5); the groups clear the bit of an excluded pair in the mask of its entry (D89, D106, [neighbor-groups.md](neighbor-groups.md), Section 2). |
| The width of a row | On the CPU a row holds the neighbors before the excluded ones are removed; on a device an excluded neighbor keeps its slot as the particle itself. |
| The row of exclusions | An incidence structure of `E` (Section 4.2). It is built with the others. |
| A term over the excluded pairs | `md.sum_tuples` over `E`, with `coordinates(distance(0, 1))`. Particle mesh Ewald has one: the reciprocal sum includes every pair, and the excluded pairs are taken out again ([nonbonded.md](nonbonded.md), Section 2.2). |

### 5.2 Pairs three bonds apart

Amber and GROMACS exclude the pairs three bonds apart from the
nonbonded terms and add them back, scaled. They are a tuple set of arity
2 with `distance(0, 1)`:

```mlir
md.tuple_set @pairs14 on(@atoms) arity(2) orientation(unordered)

%u14 = md.sum_tuples %pairs14, %x, %cell coordinates(distance(0, 1))
         tuple(%a, %b, %qq : ...) { ... }
```

| Item | Proposal |
|---|---|
| Parameters | For each pair, fields of `pairs14`: `sigma`, `epsilon` times $s_\text{LJ}$, and `qq` $= f q_i q_j s_\text{C}$. The reader keeps the two factors for each pair, and the builder applies them |
| The factors of scale | Amber stores them for each dihedral (`SCEE`, `SCNB`); GROMACS for the system (`fudgeLJ`, `fudgeQQ`) or for each pair. Parameters for each tuple take either (D47). |
| A pair that two dihedrals share | Counted once. The reader keeps one tuple for each pair, as the engines do. |
| Cutoff | None. A pair three bonds apart is always near. |

## 6. CMAP

A CMAP term [[MacKerell2004]](references.md#mackerell2004) is an energy of
two dihedrals of five atoms, $\phi$ of atoms 1 to 4 and $\psi$ of atoms 2 to 5, read
from a map: a grid of $n \times n$ energies at $\phi$ and $\psi$ from $-180°$ in steps of
$h = 360°/n$, with $\phi$ the slower index in both formats. ff19SB
[[Tian2020]](references.md#tian2020) has a map for each amino acid, of
$n = 24$; amber19sb.ff of GROMACS has the same maps in kJ/mol.

### 6.1 The patches

Between the points of the grid the energy is a bicubic patch in each cell,
from the values and three derivatives at its corners:

| Item | Rule |
|---|---|
| Derivatives at the points | $\partial E/\partial\phi$ and $\partial E/\partial\psi$ from cubic splines along each line of the grid; the cross derivative from the splines of $\partial E/\partial\phi$ along $\psi$. A spline is **not periodic**: it is the natural spline through the line repeated to $2n$ points, from $-360°$ to $345°$, taken at its central $n$ points. A periodic spline differs by up to 4.5 × 10⁻⁸ kcal/mol. |
| The patch | The Hermite bicubic $E(t, u) = \sum c_{ij} t^i u^j$ that matches the value, $h\, \partial E/\partial\phi$, $h\, \partial E/\partial\psi$, and $h^2\, \partial^2 E/\partial\phi\partial\psi$ at the four corners, with $t$ and $u$ the places of $\phi$ and $\psi$ in the cell, from $0$ to $1$: $C = M F M^{\mathsf T}$ with $M$ the matrix of rows $(1, 0, 0, 0)$, $(0, 0, 1, 0)$, $(-3, 3, -2, -1)$, $(2, -2, 1, 1)$ |
| The cell | $a = \lfloor (\phi + 180°)/h \rfloor \bmod n$, $t = (\phi + 180°)/h - \lfloor (\phi + 180°)/h \rfloor$, likewise for $\psi$; $\phi = 180°$ is the first cell with $t = 0$ |
| Units | kcal/mol in a `prmtop`, kJ/mol in `[ cmaptypes ]` |

The driver computes the 16 coefficients of every cell of every map before
the run (`lib/Driver/CMap.cpp`) and passes them as a table of 16 columns
and a row for each cell. The kernel is an `md.sum_tuples` over a tuple set
`cmap` of arity 5, ordered, with the internal coordinates
`dihedral(0, 1, 2, 3)` and `dihedral(1, 2, 3, 4)`; it finds the cell, reads
the coefficients with `md.lookup`, and evaluates the patch by Horner's
rule. Differentiation gives the forces: the cell and the lookups have no
derivative.

### 6.2 What the readers take

| Item | Amber | GROMACS |
|---|---|---|
| Maps | `CMAP_COUNT`, `CMAP_RESOLUTION`, `CMAP_PARAMETER_nn` | `[ cmaptypes ]`: five bonded types, function 1, n, n, and the grid |
| Terms | `CMAP_INDEX`: five atoms numbered from 1 (not times 3) and the map | `[ cmap ]`: five atoms and function 1 |
| Which map | The one of the file | The first `[ cmaptypes ]` whose five types are those of the atoms in their order, not reversed. A type `T-R` is the bonded type T in the residue R, or in any residue for `T-*`, as amber19sb.ff writes them; this is what grompp of GROMACS 2026.3 was seen to do. |
| Grids | Only n = 24: sander takes the derivatives right only for 24 points | n even, the same for every map |

CMAP adds no exclusions and no pairs three bonds apart.

### 6.3 The angle near 0° and ±180°

sander computes a dihedral as the arccosine of its cosine, with the sign of
its sine; near 0° and ±180° the rounding of the cosine becomes an error of
the angle of about $10^{-12}\,\text{deg}^2/\delta$ at $\delta$ degrees from them, up to 1.5 × 10⁻⁶
degrees. MDIR computes the dihedral as GROMACS does, accurately there, so
its CMAP energies may differ from sander's by up to about 3 × 10⁻⁷
kcal/mol within a few thousandths of a degree of 0° or ±180°, and by less
than 10⁻¹² beyond 0.2°.
