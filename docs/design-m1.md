# Design for Milestone M1

Status: decided (2026-09-29), revised for an all-atom target
(2026-09-29), in implementation. Section 16 has the decisions; Section 18
has the order of work and its state.

M1 is all-atom molecular dynamics with an Amber force field: a protein in
water, with particle mesh Ewald [[Darden1993]](references.md#darden1993), [[Essmann1995]](references.md#essmann1995) and
constraints, at constant energy, temperature, and pressure. This document proposes what MDIR needs for it:
relations that come from a topology, terms over them, exclusions and
scaled pairs, parameters of pairs of types, electrostatics, constraints,
the removal of the motion of the center of mass, a thermostat, a
barostat, the input, and a command line.

The first version of this document had a coarse-grained membrane with
Martini [[Marrink2007]](references.md#marrink2007) as its target. Martini is deferred (D53).

It follows [architecture.md](architecture.md) and extends
[ops-m0.md](ops-m0.md). IR snippets show the proposed syntax.

## 1. Scope

### 1.1 The target

A protein in water with ions, with an Amber force field, at constant
energy, temperature, and pressure.

| Property | Value |
|---|---|
| Force field | ff14SB [[Maier2015]](references.md#maier2015) with TIP3P [[Jorgensen1983]](references.md#jorgensen1983), and ff19SB with OPC [[Izadi2014]](references.md#izadi2014) (D65), with the ions that go with them |
| Systems | Alanine dipeptide in water, to validate; dihydrofolate reductase in water (the JAC benchmark of Amber), to measure |
| Time step | 2 fs, with the bonds of hydrogen and the water constrained |
| Nonbonded terms | Lennard-Jones with parameters for each pair of types, with the correction for the dispersion beyond the cutoff; Coulomb with particle mesh Ewald |
| Bonded terms | Bonds, angles, proper and improper dihedrals |
| Excluded and scaled pairs | Pairs one and two bonds apart are excluded; pairs three bonds apart are scaled |
| Ensembles | Constant energy; constant temperature; constant temperature and pressure, with an isotropic cell |

### 1.2 What M1 adds

| Item | Section |
|---|---|
| Relations of a topology, with parameters for each tuple | 2 |
| Kernels in internal coordinates: distance, angle, dihedral | 3 |
| Differentiation of such kernels | 4 |
| Execution of terms that contribute to several particles | 5 |
| Exclusions and scaled pairs | 6 |
| Parameters of pairs of types, the correction for the dispersion | 7 |
| Electrostatics: a cutoff, then particle mesh Ewald | 8 |
| Constraints | 9 |
| Removal of the motion of the center of mass | 10 |
| Thermostat, barostat, and random numbers | 11 |
| A cell that changes | 12 |
| Input of a topology, in the formats of Amber and of GROMACS | 13 |
| A command line with subcommands | 14 |

### 1.3 What M1 leaves out

| Item | Milestone | Consequence for M1 |
|---|---|---|
| Martini and other coarse-grained force fields | Later (D53) | |
| Particle mesh Ewald on more than one device or process | M2c and later | |
| The repartitioning of the mass of hydrogen | Later | The time step stays at 2 fs |
| Polarizable force fields, 12-6-4 terms [[Li2014]](references.md#li2014) | Later | The readers reject topologies that have them |
| Cells that are not orthorhombic | Done (D123, D125, D126, D127) | Triclinic cells run on the CPU and on the device, with the matrix and the groups, at constant volume and pressure (docs/triclinic-m2.md) |
| Random numbers for each particle, as Langevin dynamics needs them | Open | The thermostat of M1 takes a few random numbers for each step |
| Groups with a thermostat each | Later | One group |

### 1.4 An intermediate stage

The stages before particle mesh Ewald and constraints are validated on
their own (D54): flexible water, a Coulomb cutoff, a time step of 0.5 fs,
at constant energy. The energy of every term but the reciprocal part of
the electrostatics is compared with the reference engines there. The
milestone is done when particle mesh Ewald and constraints are in.

## 2. Relations of a topology

### 2.1 Tuple sets

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
`md-bypass-updates` uses ([ops-m0.md](ops-m0.md), Section 5.6).

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

### 2.2 Orientation

An orientation is the group of permutations under which two tuples are
the same tuple (Section 2.2 of ops-m0.md).

| Orientation | Arity | The same tuple | Example |
|---|---|---|---|
| `unordered` | 2 | `{i, j}` and `{j, i}` | Bonds, exclusions |
| `ordered` | Any | No other | Position restraints, with arity 1 |
| `reversal` | 2 or more | `(i, j, k)` and `(k, j, i)` | Angles, dihedrals |

### 2.3 Parameters for each tuple, not for each kind

| | For each tuple | For each kind of tuple |
|---|---|---|
| What a tuple holds | Its parameters | The number of its kind; a table has the parameters |
| Memory | More | Less |
| Reads in a kernel | One for each parameter | One for the kind, then one for each parameter |
| Parameters that differ from tuple to tuple, such as the point that a restraint pulls to | As any other | Not possible |

Proposal: parameters for each tuple. A front end that wants kinds can have
them with the same ops: the kind is a field of the tuple set, and the
parameters are a table (Section 7).

## 3. Kernels in internal coordinates

### 3.1 The coordinates of a tuple

A kernel over a neighborhood takes the distance and the displacement of
the pair (B7). A kernel over a relation of a topology takes the internal
coordinates that the op names. The ops are `md.sum_tuples` and
`md.gather_tuples`; they are apart from `md.sum_relation` and
`md.gather_relation` because exchange and truncation, which those two
carry, have no meaning for a tuple of a topology.

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

After the coordinates the kernel takes the fields of the particles that
the op gathers, one value for each member, and the fields of the tuple.

### 3.2 Why not positions

| | Internal coordinates | The positions of the members |
|---|---|---|
| The kernel of an angle | A function of one number | Dot products, norms, and an inverse cosine |
| Differentiation | The derivative of the kernel, times the derivative of the coordinate, which is known in closed form | Rules for vectors: dot and cross products, norms |
| Where the derivative is singular | In the closed form, where it can be treated once | In what the rules generate |
| A term in coordinates that are not listed | Not possible | Possible |

Proposal: internal coordinates. `displacement` covers terms that need the
vector. Kernels over positions can be added when a term needs them.

### 3.3 Exchange

A coordinate is the same for a tuple and for every tuple that the
orientation identifies with it: `cosine(0, 1, 2)` for `(i, j, k)` and
`(k, j, i)`. A kernel that takes only such coordinates and the fields of
the tuple is symmetric, and the compiler can tell. A kernel that gathers
fields of the particles is checked as in M0 (B3).

## 4. Differentiation

### 4.1 With respect to the positions

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

A kernel that uses a `displacement` cannot be differentiated with respect
to the positions yet, as in M0.

### 4.2 With respect to a parameter

| With respect to | In M1 | How |
|---|---|---|
| An argument of the potential that is a number, such as a scale of the energy or a coupling parameter | Yes | `request [derivative(n)]`, as in M0. The derivative of a sum over tuples is the sum of the derivative of its kernel |
| A field of the tuples or of the particles, such as the force constant of every bond | No | The result is a field, not a number. It is a gather of the derivative of the kernel, and the machinery of 4.1 can produce it when an application needs it, such as the fitting of a force field |
| An entry of a table (Section 7) | No | The result is a table. It needs a sum for each entry |

## 5. Execution of terms over tuples

### 5.1 The strategies

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
| It is the strategy of the pairs. | A thread writes to its own particle. No atomic addition, and the deterministic level holds. |
| Bonded terms are a small part of the work. | An atom of a protein is a member of some 20 bonded tuples and has some 300 neighbors. |
| A loop over tuples of this kind is a loop over particles. | It can be fused with the loop over pairs and with the kick that follows. |

B and C are choices for the planner later.

### 5.2 The structure

```mlir
%incidence = md_exec.build_incidence %angles
    : !md.relation<@atoms, 3, reversal, @angles>
      -> !mdrt.incidence<@atoms, @angles, 3>

%f, %u = md_exec.tuple_for %incidence, %x, %cell
           coordinates(displacement(0, 1), displacement(2, 1))
           tuple(%k, %c0 : !md.field<@angles, f64>, !md.field<@angles, f64>)
           outs(%f0 : !vec) reduce(%u0 : f64) arity(3) {
^bb0(%u: vector<3xf64>, %v: vector<3xf64>, %k_t: f64, %c0_t: f64):
  ...
  md_exec.yield %f_0, %f_1, %f_2, %e
      : vector<3xf64>, vector<3xf64>, vector<3xf64>, f64
} : !mdrt.incidence<@atoms, @angles, 3>, !vec -> !vec, f64
```

| Item | Decision |
|---|---|
| What the structure holds | For each particle a row: the number of its tuples, then for each tuple its number, the place of the particle in it, and its members |
| The width of a row | The largest number of tuples of a particle, found when the structure is built. No parameter, and no row that overflows. |
| When it is built | Where the members change: where the run begins and where the particles are put in a new order (D44), which renumbers the members. It does not depend on the positions or on the cell. |
| The coordinates | `md_exec.tuple_for` takes displacements only, in the minimum image. The kernel computes the other coordinates from them, with the code that differentiation uses (Section 4.1). |
| The kernel | Yields the forces on all members. The loop takes the one of the particle. A kernel for each place, which computes that force only, is a later optimization. |
| A sum | A tuple contributes once, from its member at place 0. No weight, and no rounding from adding a third three times. |

## 6. Exclusions and scaled pairs

### 6.1 Exclusions

```mlir
%n = md.neighborhood %x, %cell cutoff(1.0) exclude(%excluded)
       : !vec -> !pairs
```

$$N(\mathbf x, h) = \{ \{i, j\} : i \ne j,\ r_{ij} < r_c \} \setminus E.$$

| Item | Proposal |
|---|---|
| The relation | `E` is a relation of arity 2 of a tuple set, as the bonds are. The reader derives it: pairs one, two, and three bonds apart (`nrexcl = 3`), and those that the topology lists. |
| Where a pair is excluded | In the build of the neighbor structure. After the search, every particle removes from its row the particles in its row of exclusions and keeps the others in their order. A loop over pairs tests nothing. The neighbor structure carries the exclusions (`exclude` of `md_exec.build_neighbors` and `md_exec.empty_neighbors`), so every build applies them. |
| The width of a row | A row holds the neighbors before the excluded ones are removed. The search itself is not changed, and a filter after it serves all three searches of a device. |
| The row of exclusions | An incidence structure of `E` (Section 5.2). It is built with the others. |
| A term over the excluded pairs | `md.sum_tuples` over `E`, with `coordinates(distance(0, 1))`. Particle mesh Ewald has one: the reciprocal sum includes every pair, and the excluded pairs are taken out again (Section 8). |

### 6.2 Pairs three bonds apart

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
| Parameters | For each pair: the coefficients of Lennard-Jones and the product of the charges, each with its factor of scale applied by the reader |
| The factors of scale | Amber stores them for each dihedral (`SCEE`, `SCNB`); GROMACS for the system (`fudgeLJ`, `fudgeQQ`) or for each pair. Parameters for each tuple take either (D47). |
| A pair that two dihedrals share | Counted once. The reader keeps one tuple for each pair, as the engines do. |
| Cutoff | None. A pair three bonds apart is always near. |

## 7. Parameters of pairs of types

### 7.1 Tables

Amber gives Lennard-Jones as coefficients for each pair of types, and a
force field may set the parameters of some pairs apart from its mixing
rule (NBFIX). GROMACS does the same with `[ nonbond_params ]`. The
front end therefore always builds a table for each pair of types: from
the mixing rule, with the pairs that the topology sets overriding it. The
kernel makes one lookup. No mixing rule is left at run time.

```mlir
%sigma = mdrt.from_buffer %sigma_buffer
           : memref<?x?xf64> to !md.table<2, f64, symmetric>

%u = md.sum_relation %n, %x, %cell gather(%type : !kinds) ... {
^bb0(%r: f64, %d: vector<3xf64>, %t_i: i32, %t_j: i32):
  %s = md.lookup %sigma[%t_i, %t_j] : !md.table<2, f64, symmetric>, i32, i32 -> f64
  ...
}
```

| Item | Proposal |
|---|---|
| The type | `!md.table<rank, E>`, with `symmetric` for a table of rank 2 that is; E is f64 or f32, or a vector of them for the tables that a kernel reads at the same entry (D96) |
| Exchange | A lookup in a symmetric table with the two types of a pair is symmetric |
| Differentiation | A lookup does not depend on the positions |
| Storage | A buffer, on the device where the loops are |
| Mixing rules that fill a table | Lorentz–Berthelot [[Lorentz1881]](references.md#lorentz1881), [[Berthelot1898]](references.md#berthelot1898), geometric, and none: the table as the topology gives it |
| In a control file | The topology gives the types and the table; a control file without a topology keeps the mixing rules of M0 |

Charges need the product of the values of the two particles, which the
mixing rules of M0 do not have. The proposal is the rule `product`.

### 7.2 The correction for the dispersion

Beyond the cutoff the attraction of Lennard-Jones is left out. The
correction [[AllenTildesley2017]](references.md#allentildesley2017), [[Shirts2007]](references.md#shirts2007) adds its mean, for a uniform density beyond the cutoff, to
the energy and the pressure:

$$E_\text{disp} = -\frac{2\pi N^2}{3V}\, \frac{\langle C_6\rangle}{r_c^3}, \qquad P_\text{disp} = \frac{2 E_\text{disp}}{V},$$

with $\langle C_6\rangle$ the mean of $C_6 = 4\varepsilon\sigma^6$ over the pairs of particles,
which the front end computes from the table and the counts of the types.
It depends on the volume only, so it is a number on the host that
changes with the cell. It is on by default when a topology is read, as in
Amber; a control file asks for it with `dispersion_correction` in a pair term.

As the engines do [[GromacsManual2025]](references.md#gromacsmanual2025),
the correction takes the part of the term that decays as $r^{-6}$ and leaves
the repulsion out. As GROMACS does, it takes $N^2 \langle C_6\rangle$, with $\langle C_6\rangle$ the
mean of $C_6$ over the pairs of distinct particles that are not excluded;
it agrees with GROMACS to 10⁻⁷. Its virial is six times its energy, so the pressure
changes by $2 E_\text{disp} / V$. A switch or a shift of the potential inside the
cutoff needs the integral of the change as well; M1 takes a plain cutoff
only.

sander leaves the repulsion out as well, but sums $N_a N_b C_{6,ab}$ over
all pairs of types, with no mean over the pairs that are not excluded
(`vdw_correction` of `ew_setup.F90` in AmberTools). The two differ by
some $N_\text{excl} / N^2$ of the correction, $4 \times 10^{-4}$ of it for the dipeptide in
water; a comparison with sander states the difference.

Not implemented, and kept here as an alternative: the correction from the
whole expression of the term, integrated numerically beyond the cutoff,

$$
\begin{aligned}
E &= \frac{2\pi}{V} \sum_{ab} N_a N_b \int_{r_c}^\infty r^2\, u_{ab}(r)\, dr, \\
W &= \frac{2\pi}{V} \sum_{ab} N_a N_b \int_{r_c}^\infty r^2 \big(-r\, u_{ab}'(r)\big)\, dr,
\end{aligned}
$$

over $x = r_c / r$ in $(0, 1]$, which takes the repulsion and any term
that decays faster than $r^{-3}$. It differs from the correction above by
$(\sigma / r_c)^6 / 3$ of it for Lennard-Jones, $10^{-3}$ at the cutoffs of M1, less
than the error of the uniform density that both assume, and it would not
agree with either engine. Lennard-Jones by particle mesh Ewald, for systems whose density
beyond the cutoff is not uniform, such as membranes, is for later.

## 8. Electrostatics

### 8.1 A cutoff

For the intermediate stage (Section 1.4). sander of AmberTools computes a
plain `q_i q_j / r`, cut at the cutoff with no shift, in a periodic system
with `eedmeth = 4` of `&ewald`; it then leaves out the self term, the
reciprocal sum, and the correction of the excluded pairs. The stage takes
that form, and the comparison with sander is of the energy of each term.
Whether GROMACS has the same form for a periodic system is settled before
the stage begins; otherwise the comparison with GROMACS covers every term
but the Coulomb one.

### 8.2 Particle mesh Ewald

Decided: particle mesh Ewald on one device or one process, in M1 (D55).
Designed in its own document when its stage begins. The parts:

| Part | Where |
|---|---|
| The direct sum with $\operatorname{erfc}(\beta r) / r$ | A term over pairs, as any other |
| The excluded pairs | A term over the tuples of $E$ that takes $\operatorname{erf}(\beta r) / r$ out again (Section 6.1) |
| The self term | A number, from the charges |
| The reciprocal sum | Spreading the charges to a grid with B-splines of order 4 [[Essmann1995]](references.md#essmann1995), a forward FFT, a product with the influence function, an inverse FFT, and the forces from the grid |
| The virial | From the reciprocal energy of each wave vector, and from the direct sum as for pairs |

The methods of the reciprocal sum share the grid, the spreading, and the
FFT, and differ in the influence function
[[GromacsManual2025]](references.md#gromacsmanual2025):

| Method | Cost | In MDIR |
|---|---|---|
| Ewald summation [[Ewald1921]](references.md#ewald1921), a sum over wave vectors | Of order $N^{3/2}$ at best | The reference of the tests: a script with many wave vectors |
| Smooth particle mesh Ewald [[Essmann1995]](references.md#essmann1995): B-splines, an FFT, and the forces from the gradient of the B-splines | $N \log N$ | The method of M1, as both engines use it |
| P3M with forces from the potential (P3M-AD): the influence function that minimizes the error for the grid | $N \log N$ | A variant: another influence function in the same pipeline, when it is measured to help |

What the engines do differently, which MDIR must be able to do both ways
to compare with them:

| Item | GROMACS | Amber (sander, `eedmeth = 1`) | In MDIR |
|---|---|---|---|
| The direct sum at the cutoff | Shifted by a constant to zero, so that the energy is continuous | $\operatorname{erfc}(\beta r) / r$, cut with no shift | A choice of the term, as the truncations of M0 are |
| $\beta$ | From `ewald-rtol`: the relative size of the direct sum at the cutoff | From `dsum_tol`, the same idea | From a tolerance, as both |
| The grid | The largest spacing, `fourierspacing`, with sizes that the FFT handles fast | Grid sizes or a spacing | The same, with sizes of small prime factors |
| The terms of the log | Coulomb (SR): the direct sum, the excluded pairs, and the self term; Coul. recip.: the reciprocal sum | Its own split | The terms kept apart, so that each can be compared |

Beside those:

- A system with a net charge needs the neutralizing background, and a
  surface term (the dipole correction) is an option in both engines.
  Which one each applies by default is checked from their documentation
  before the stage begins.
- The balance of the direct and the reciprocal sum (the cutoff, $\beta$, and
  the grid) is tuned at run time by the engines. In MDIR it is a numeric
  parameter of the plan (A4), which a run can tune without compiling
  again, since $\beta$ and the grid enter the program as values.
- Methods of other kinds, such as fast multipole or multilevel summation,
  are not planned for M1; distributed particle mesh Ewald is M2c.

Two questions shape the design (Section 17):

| Question | Candidates |
|---|---|
| Spreading on a device writes to grid points that many particles share. Atomic additions of floating-point numbers depend on the order of the threads, which breaks the deterministic level (P13). | Spreading by grid points, which gather from the particles near them; or atomic additions of integers in fixed point [[LeGrand2013]](references.md#legrand2013), which do not depend on the order |
| The library for the FFT. FFTW is under the GPL, and MDIR is under the MIT license. | cuFFT on a device, from the toolkit that MDIR needs anyway; a library under a permissive license on the host, such as pocketfft; or an FFT as a template in IR |

## 9. Constraints

Decided: constraints in M1 (D56). Designed in its own document when its
stage begins.

| Item | Decision |
|---|---|
| Water | SETTLE [[Miyamoto1992]](references.md#miyamoto1992), which solves the three distances of a rigid water in closed form |
| Which molecules | Chosen (D63): from a GROMACS topology, those with `[ settles ]`; from an Amber topology, the residues named in `water_residues`, `["WAT"]` by default as in sander. A named residue that is not one heavy atom and two hydrogens with the three distances given is an error. `rigid_water = false` constrains the same water with SHAKE, and `hydrogen_bonds = false` leaves it flexible. |
| The bonds of hydrogen | SHAKE [[Ryckaert1977]](references.md#ryckaert1977), with RATTLE [[Andersen1983]](references.md#andersen1983) for the velocities under velocity Verlet |
| Execution | The constrained bonds fall into clusters with no particle in common: a heavy atom with its hydrogens, a water. A thread takes a cluster and writes to its particles only, so no two threads write to one particle. A loop over clusters is a new kind of loop. |

### 9.1 SETTLE

| Item | Rule |
|---|---|
| Positions | After the drift, each water is brought back to its shape by the closed form of Miyamoto and Kollman [[Miyamoto1992]](references.md#miyamoto1992): the new triangle, about its center of mass, is the rigid one turned by three angles in the frame of the old plane. It is the solution of SHAKE with the constraint forces along the old bonds (checked against SHAKE converged to 10⁻¹⁵: 2 × 10⁻¹⁴ nm). The velocities take the change over the step, $\Delta\mathbf x / \Delta t$, as the first half of RATTLE [[Andersen1983]](references.md#andersen1983). |
| Positions below double precision | M-SHAKE [[Krautler2001]](references.md#krautler2001) on the three bonds of the water, the two O–H and the H–H, with the iterations of Newton of the groups of SHAKE (D112). SETTLE computes the new positions, of the size of the water, and subtracts the old: in f32 the rounding of that change, about 10⁻⁷ Å, divided by the step, is an error of the velocities whose square adds kinetic energy at every step, the drift that [[Jung2026]](references.md#jung2026) finds for single precision. JAC NVE took 830 kcal/mol in 2 ns with SETTLE in f32 (after the fix of the divisions, 286), and 2 with M-SHAKE in f32, which carries the change itself; SETTLE in f64 also conserved the energy, at 581 against 724 ns/day. Both solve the same equations, SHAKE with the forces along the old bonds. |
| Velocities | After the second kick, the impulses along the three bonds that leave no velocity along them: three linear equations, solved in closed form |
| Old positions | Taken in the periods of the cell of the new ones: the bonds before the drift are moved by the same lattice vectors as the new bonds in the minimum image |
| Virial | The mean of those of the forces of the constraints over the two halves of the step: $\mathbf G_i = 2 m_i \Delta\mathbf x_i / \Delta t^2$ of the positions, with the bonds before the drift for arms, and $\mathbf G_i = 2 m_i \Delta\mathbf v_i / \Delta t$ of the velocities, with the bonds after it; $\sum (\mathbf x_i - \mathbf x_O) \otimes \mathbf G_i$ of each. The mean is that of the pressure (D45). |
| At the start | The drawn velocities have their parts along the bonds removed before they are scaled to the temperature. The positions are not constrained, as sander does not. |
| Degrees of freedom | Three fewer for each water |
| Temperature of the log | That of the velocities of the step: the forces do not give the kinetic energies of the half steps once the constraints act (D45) |
| In the IR | An `md.gather_tuples` over a tuple set `settles` of arity 3, the oxygen first, for each of the two halves, in the program of the step; the virial an `md.sum_tuples` |
| Which waters | Those of `[ settles ]` of GROMACS; from Amber the residues of `water_residues`, `["WAT"]` by default, which are an oxygen and two hydrogens with at most virtual sites after them. Their bonds and angles are dropped. |

### 9.2 SHAKE and RATTLE

| Item | Rule |
|---|---|
| Groups | With `hydrogen_bonds = true`, the bonds of hydrogen (`BONDS_INC_HYDROGEN` of Amber; a bond with an atom of atomic number 1 in GROMACS) outside the waters of SETTLE, grouped by their heavy atom: one to three hydrogens. A hydrogen bonded to two atoms, two hydrogens bonded to each other, and more than three hydrogens on an atom are errors. The bonds are dropped from the bond terms, as sander does with `ntf = 2`. |
| Positions | SHAKE [[Ryckaert1977]]: the atoms move along the bonds before the drift, the hydrogens by $\lambda_j \mathbf s_j / m_j$ and the heavy atom by $-\sum_j \lambda_j \mathbf s_j / m_0$. Each of 6 iterations of Newton solves the constraints linearized at the current bonds exactly, $\sum_j 2 (\mathbf r_k \cdot \mathbf s_j)(\delta_{kj} / m_j + 1 / m_0)\, \lambda_j = d_k^2 - r_k^2$, by the elimination of RATTLE, so that the error squares with each; sweeps bond by bond shrink it only by about $m_H / m_X$, close to $1/2$ with repartitioned masses of hydrogen, which 12 sweeps left at 10⁻⁴ of a bond and a step of 4 fs made unstable. The number is fixed, so the kernel has no test of convergence. |
| Optional one-bond projection | `analytic_bonds = true` replaces the six Newton iterations for isolated X–H groups with the rationalized near root of the same fixed-old-direction quadratic (D128). It checks the original squared-distance residual before accepting; failure or a backward predictor takes the existing Newton path. The default is `false`. Groups with two or three bonds and waters keep their existing solvers. |
| Velocities | RATTLE [[Andersen1983]](references.md#andersen1983): the impulses along the bonds solve $A \boldsymbol\tau = -\mathbf b$ with $A_{ij} = \delta_{ij} / m_j + \mathbf e_i \cdot \mathbf e_j / m_0$ exactly, by Gaussian elimination of at most $3 \times 3$ |
| Virial, degrees of freedom, the start | As for SETTLE: the mean of the virials of the forces of the constraints over the two halves of the step, one degree of freedom less for each bond, and drawn velocities without parts along the bonds. The positions of the file are not constrained before the first step, as sander does not; the first step brings the bonds to their lengths. At the start, where no step has given the forces of the constraints, the log takes their virial as $\sum (\mathbf x_k - \mathbf x_0) \cdot \mathbf G^0_k - 2 K_\text{int}$ over each group: $\mathbf G^0 = m P(\mathbf F/m) - \mathbf F$ keeps the accelerations on the constraints, and $-2 K_\text{int}$, twice the kinetic energy of the motion within the group, is the rest for a group that the constraints keep rigid (a water, a bond); for a group of SHAKE with two or three hydrogens it counts the bending as well, a few bar in a protein. |
| In the IR | An `md.gather_tuples` over a tuple set `shake1`, `shake2`, or `shake3`, the heavy atom first, for each half of the step; SETTLE comes first, then the groups of SHAKE, then the placement of the virtual sites |

Three consequences reach other stages, and are recorded now:

| Consequence | Where |
|---|---|
| The forces of the constraints add to the virial | The pressure, and the barostat (Section 11) |
| Every constraint removes a degree of freedom: `f = 3N − N_c − 3` | The temperature, and the thermostat |
| The kinetic energies at the half steps (D45) must be measured from constrained velocities, not computed from the forces | The log, the thermostat, and the barostat |

## 10. Removal of the motion of the center of mass

The total momentum is removed where the velocities are drawn (M0). In a
run it drifts: rounding in the forces, most in mixed precision, and the
thermostat, which scales velocities, does not remove it.

$$\mathbf v_i \leftarrow \mathbf v_i - \frac{\sum_j m_j \mathbf v_j}{\sum_j m_j}$$

every `center_of_mass_interval` steps.

| Item | Proposal |
|---|---|
| What is removed | The linear momentum. Under periodic boundaries the angular momentum is not conserved and is left alone. |
| How | A `particle_for` with a sum of `vector<3xf64>`, as the virial is summed, and a `particle_for` that subtracts and scales for the thermostat |
| How often | `center_of_mass_interval` in `[dynamics]`: with a thermostat, when it acts; without one, only if given (D67) |
| Degrees of freedom | Three fewer, as in M0 |

## 11. Thermostat and barostat

### 11.1 Kinds

| | Proposal | Reason |
|---|---|---|
| Thermostat | Stochastic velocity rescaling (Bussi, Donadio, and Parrinello 2007 [[Bussi2007]](references.md#bussi2007)) | It samples the canonical distribution, and it takes the kinetic energy and a few random numbers for each step, not one for each particle. |
| Thermostat, the other | Langevin dynamics by the middle scheme ([[Zhang2019]](references.md#zhang2019); BAOAB of [[Leimkuhler2013]](references.md#leimkuhler2013)), `method = "LANGEVIN"` (D135, Section 11.6) | The thermostat of most runs of Amber and OpenMM; it acts on each particle in every step, with three random numbers for each. |
| Barostat | Stochastic cell rescaling (Bernetti and Bussi 2020 [[Bernetti2020]](references.md#bernetti2020)), isotropic; semi-isotropic, with the pressure in a plane apart from that along its normal, as an option | It samples the distribution at constant pressure and is of first order: it has no momentum of the cell to store. |

Both carry the statement that they preserve the target distribution
(P10).

Two barostats go by the name of Bussi:

| | Bussi, Zykova-Timan, and Parrinello 2009 [[Bussi2009]](references.md#bussi2009) | Bernetti and Bussi 2020 [[Bernetti2020]](references.md#bernetti2020) |
|---|---|---|
| Kind | Of second order: the cell has a momentum, and the thermostat acts on it as well | Of first order: the barostat of Berendsen [[Berendsen1984]](references.md#berendsen1984) with a term of noise |
| State beside the cell | The momentum of the cell | None |
| Parameters | A time | A time and a compressibility. A wrong compressibility changes how fast the volume relaxes, not what is sampled. |
| The volume | May oscillate | Relaxes |

MDIR takes the one of 2020 (D50). A checkpoint needs nothing for it.

### 11.2 In `dyn`

*Proposed, not implemented as written.* The couplings are not ops of
`dyn`: the driver writes them into the schedule of `@mdir_run` at the end
of each period of coupling (Section 11.4). The factor of the thermostat
and the strain of the barostat are calls to the host (`mdrtBussiFactor`,
`mdrtBarostatStrain` of `libmdrt`, which draw from Philox 4×32-10 by the
key of A13), and loops over particles (`md.map_particles`) remove the
motion of the center of mass and scale the velocities and the positions.
The proposal was:

```mlir
dyn.program @step(...) attributes {
    requires = ["temperature", "pressure"],
    provides = ["thermostatting", "barostatting"]} {
  ...
  %alpha = dyn.velocity_rescaling %kinetic, %freedom, %step
             temperature(300.0) tau(1.0) dt(%span) stream(0)
  %v2    = dyn.scale %v1, %alpha : !vec
  ...
}
```

| Op | Definition |
|---|---|
| `dyn.velocity_rescaling` | The factor of the thermostat, from the kinetic energy, the degrees of freedom, and random numbers |
| `dyn.cell_rescaling` | The factors of the barostat along the three edges, from the pressure along each |
| `dyn.scale` | A field times a number, or times a number for each direction |
| `dyn.scale_cell` | A cell with edges times a number for each direction |
| `dyn.random` | A random number (P5, A5, A13) |

| Item | Proposal |
|---|---|
| The kinetic energy of the thermostat | That of the velocities it scales (D67): with velocity Verlet, those at the end of a step, which are of one time. `K_T` of D45 estimates the temperature of the log. |
| The kinetic energy of the barostat | That of the velocities of the step without that of the center of mass, before the thermostat (Section 11.4), not `K_P` of D45 |
| The pressure of the barostat | With the correction for the dispersion (Section 7.2) and the virial of the constraints (Section 9) |
| How often | Every `n` steps, with `n` times the time step. A step between computes no global sum. On a device a step with global sums takes up to twice the time of one without (Section 10.8 of ops-m0.md). |
| Groups with a thermostat each | One group in M1 |

### 11.3 Random numbers

| Item | Proposal |
|---|---|
| The generator | Philox 4×32 with 10 rounds [[Salmon2011]](references.md#salmon2011), from the key of A13 |
| Where it runs | On the host, in `libmdrt`: the thermostat and the barostat take numbers for the system, not for a particle |
| Streams | 0 for the thermostat, 1 for the barostat. The draw index counts the numbers of one step. |
| Entity | A fixed key, as for a global move (A5) |
| Random numbers in kernels | For Langevin dynamics (D135): Philox emitted by the driver as `arith` ops in the kernel of the step, one block for each particle and step, stream 2, the entity the number of the particle in the input. A block gives four words, four uniform numbers $(w + \tfrac12)\,2^{-32}$, and by Box and Muller two pairs of normal numbers, of which three are taken. It agrees with the generator of the host to the last bit (`langevin.test`). |

### 11.4 The barostat as it is

Stochastic cell rescaling, isotropic or semi-isotropic, with velocity
Verlet and leapfrog (D72, D76, D77, D119):

| Item | Rule |
|---|---|
| When | At the end of the step that completes a period of coupling, after the removal of the motion of the center of mass and the thermostat; `interval` equals `interval`. The last step of a period computes the virial for it. |
| The pressure | $P = (2K + \operatorname{tr}\mathsf W) / (3V)$, with $K$ the kinetic energy of the velocities of the step without that of the center of mass, $\mathsf W$ the virial of the step with those of the constraints (the mean of the two halves, Section 9.1), of the correction for the dispersion, and of the background of a net charge at the volume $V$ of the cell before the scaling. No term $k_B T / V$. |
| The change of the volume | One step of Euler and Maruyama over the period $\Delta t_p$ of the equation for $\lambda = \sqrt V$, eq. (7) and eq. (S7) of [[Bernetti2020]](references.md#bernetti2020): $\lambda' = \lambda - \tfrac12 f\lambda\, \big(P_0 - P - k_B T c/(2V)\big) + \sqrt{k_B T f c/2}\, R$, $\Delta\varepsilon = 2 \ln(\lambda'/\lambda)$, $f = \beta_T \Delta t_p / \tau_p$. Its noise does not depend on the volume, which the paper's reversible integrators need; to first order it is the step of $d\varepsilon = -(\beta_T/\tau_p)(P_0 - P)\, dt + \sqrt{2 k_B T \beta_T / (V \tau_p)}\, dW$, eq. (5), which GROMACS takes. Pressures in bar, $c = 16.6053906717$ bar nm³ mol/kJ, $T$ that of the bath. $R$ is the normal number of stream 1 of the step (A13), drawn on the host (`mdrtBarostatStrain`). |
| Scaling | The positions of every particle and the edges of the cell by $\mu = \exp(\Delta\varepsilon/3)$, the velocities by $1/\mu$; a group that the constraints keep rigid (a water of SETTLE, a group of SHAKE) moves with its center of mass and keeps its shape, since stretched bonds would be taken back by the constraints of the next step with a change of the velocities that heats the system. Virtual sites are placed again in the next step |
| The cell | Kept in memory on the host, where each iteration of a loop takes it, and the steps that follow a loop of periods in the same iteration take it again; the neighbor structures, whose test of validity compares the cell, are built again; the influence function of PME follows (pme-m1.md); the log and the trajectory take the new edges (`mdrtSetBox`) and, for a triclinic cell, the tilts (`mdrtSetTilt`); a checkpoint keeps them, and a restart takes them. A triclinic cell scales as $H\,\mathrm{diag}(\boldsymbol\mu)$: each tilt with the column it is in, $b_x$ and $c_x$ by $\mu_x$, $c_y$ by $\mu_y$ (D127). A run stops if an edge becomes shorter than twice the cutoff, below which the minimum image misses pairs; a run that begins so is rejected |
| The scaled positions | With `work = "EXACT"`, the virtual sites are placed on the scaled positions, and those are evaluated in the new cell: their energy gives the work of the scaling, and their forces are those that the next step begins with (D77). With `"FIRST_ORDER"`, the next step begins with the forces of the positions before the scaling, and places the sites after its drift |
| The conserved energy, exact | Takes away what the scaling gives: $U(\mathbf x') - U(\mathbf x)$, the potential energy of the scaled positions less that of the positions before; $C\,(1/V' - 1/V)$, the change of the terms that are constants of the volume (the correction for the dispersion and the background of a net charge, $C$ their energy times the volume of the start), which `md.evaluate` leaves out (amended 2026-10-02: before, the conserved energy carried $E_c(V) - E_c(V_0)$); and $(1/\mu^2 - 1) K$, the change of the kinetic energy of the velocities it scales. Between scalings the dynamics is that of constant energy, so the conserved energy changes as there. On the mixture of `barostat.test`, 7.4 × 10⁻⁶ of its value over 8 ps instead of 2.2 × 10⁻³; on tri-alanine in 1218 OPC waters at 1 bar, 300 K, `time_constant = 2`, 2 fs, on a GPU in mixed precision (`examples/ala3`), 0.011 kcal/mol per ps instead of 2.1. The evaluation costs 0.13 ms per step at a period of coupling of 10 steps there, 20% of the rate (316 to 253 ns/day): the cost of the reversible integrator of [[Bernetti2020]](references.md#bernetti2020) (Table I; SI Sec. V.B), which this is but for the thermostat, applied once at the end of the period rather than in halves around the step. This conserved energy is not the paper's effective energy, which adds $P_0 \Delta V$, the terms of the noise, and the drift of eq. (7) to measure the violation of detailed balance (Sec. II.C); it tests the dynamics between scalings and the counting of the work, and the distribution of the volume tests the barostat |
| The scaling of Trotter type, the default | The last two steps of a period follow its loop: the step of energy whose pressure gives the strain and the new cell, and `step_trotter`, which kicks half, drifts half, scales the positions by $\mu$ (the rigid groups with their centers) and the velocities by $1/\mu$, drifts the other half, and evaluates in the new cell with the virial ([[Bernetti2020]](references.md#bernetti2020), SI Sec. V.C, eqs. S12a–d; eqs. S13a and S15 have two misprints, D92). The conserved energy takes away $(1/\mu^2 - 1) K$ of the velocities scaled and $-\ln\mu\, (\mathsf W_\text{before} + \mathsf W_\text{after}) / 2$, the virials of the groups and the constant terms before and after the step: $\sum_g \mathbf X_g\cdot\mathbf F_g$, from the forces of the evaluations, $\operatorname{tr}\mathsf W_\text{eval} - \sum_g\sum_{j\in g}(\mathbf x_j - \mathbf X_g)\cdot\mathbf F_j$, which the forces of the constraints do not enter (D116). No evaluation is added; the count drifts with the period of coupling (D92) |
| Semi-isotropic coupling | x and y scale together by the strain of $\ln A$ from $(P_{xx} + P_{yy})/2$ less $\gamma/L_z$, z by that of $\ln L_z$ from $P_{zz}$, each with its own normal number (the first two of stream 1 of the step), one step of Euler and Maruyama of eqs. (9a) and (9b) of [[Bernetti2020]](references.md#bernetti2020) (`mdrtBarostatStrainArea`, `mdrtBarostatStrainHeight`). The pressure of each axis takes its kinetic energy without the center of mass and the diagonal of the virial; the scale, the kinetic energy, and the virials are vectors of three, and the work sums over the axes. `work = "FIRST_ORDER"` is for isotropic coupling only (D119) |
| The period of coupling | The paper finds the fluctuations of the volume too large when $N_P \Delta t$ is not small against $\tau_p$ (a TIP3P box at $\tau_p = 0.5$ ps: $\sigma^2_V$ from 0.23 at $N_P \le 10$ to 0.45 nm⁶ at 100, Fig. 3b), with $N_P = 10$ as a compromise and little gain in rate beyond 20 to 40 in GROMACS (Fig. 4). A longer period that lowers the cost of the evaluation must keep $N_P \Delta t / \tau_p$ small |
| The conserved energy, first order | Takes away $-(\mu - 1) \operatorname{tr}\mathsf W_g$, the change of the potential energy to first order, as GROMACS does, and $(1/\mu^2 - 1) K$, that of the kinetic energy, exactly. $\mathsf W_g$, the virial of the rigid groups that move as wholes, is the $\mathsf W$ of the pressure above, which has the virial of the constraints, with twice the kinetic energy of the motion within the groups, $\sum \tfrac12 m \lVert\mathbf v - \mathbf V\rVert^2$ over each (the virial of the forces within a rigid group is minus that). What is left is the second order, $\tfrac12 (\mu - 1)^2\, d^2U/d\mu^2$, whose mean over the noise of $\Delta\varepsilon$ is proportional to its variance, and so to $f$: a drift that neither the time step nor the period of coupling reduces, only `time_constant`. On the mixture of `barostat.test` at 2 fs, 1.8 × 10⁻³ of the energy over 8 ps with `time_constant = 2`, 4.4 times less with `time_constant = 8`; on 1394 OPC waters with PME at 300 K and 1 bar, 2.1 kcal/mol per ps with `time_constant = 2` and 0.52 with `time_constant = 8`. The same runs at constant volume keep the conserved energy to 10⁻⁶ and 10⁻⁵ |
| Parameters | In `[ensemble]`: `ensemble = "NPT"`, `temperature`, and `pressure` in atm. In `[barostat]`: `method = "C-RESCALE"`, `time_constant` in ps (5 by default), `compressibility` in 1/atm (4.5 × 10⁻⁵ /bar by default), `coupling` (`"ISOTROPIC"` or `"SEMI_ISOTROPIC"`; with the second, `compressibility_z` in 1/atm, that of x and y by default, 0 to keep the height, `surface_tension` in dyn/cm of each surface normal to z, and `surfaces`, 2 by default), `work` (`"TROTTER"` by default, `"TROTTER_FIRST_ORDER"`, `"EXACT"`, `"FIRST_ORDER"`), and `interval`, which must be that of the thermostat. NPT needs a `[thermostat]` |

The drift of the conserved energy on tri-alanine in 1218 OPC waters
(`examples/ala3`, 4,905 particles, from its checkpoint after 100 ps at
constant pressure; 100 ps of each, 2 fs, SETTLE and SHAKE, PME, an RTX
3090, velocity Verlet, the barostat with the exact work and the step in
$\lambda$, thermostat and barostat every 10 steps, `time_constant` 0.5 ps, `time_constant` 2 ps),
from the slope of a line through the rows of the log:

| Ensemble | Mixed: kcal/mol/ps | kJ/mol/ns per atom | ns/day | Double: kcal/mol/ps | kJ/mol/ns per atom | ns/day |
|---|---|---|---|---|---|---|
| NVE | −0.0111 | −0.0094 | 426.5 | −0.0082 | −0.0070 | 83.8 |
| NVT | +0.0047 | +0.0040 | 423.0 | −0.0015 | −0.0013 | 83.7 |
| NPT | −0.0014 | −0.0012 | 253.9 | −0.0054 | −0.0046 | 57.8 |

The barostat costs 40% of the rate at this period: the evaluation after
each scaling, and the neighbor structures that each change of the cell
rebuilds (with the first-order work, 316 ns/day).


### 11.5 The effective energy of the barostat (planned)

The conserved energy of Section 11.4 tests the dynamics between scalings
and the counting of their work. Whether the barostat itself samples its
distribution is what the effective energy of [[Bernetti2020]](references.md#bernetti2020)
measures (Sec. II.C and SI Sec. IV): the work that the steps of the volume
do against detailed balance.

At fixed scaled positions and momenta the variable $\lambda = \sqrt{V}$ has
the stationary distribution $P(\lambda) \propto e^{-E(\lambda)/k_BT}$ with

$$E(\lambda) = K + U + P_0 \lambda^2 - k_BT \ln\lambda,$$

the last term from $dV = 2\lambda\, d\lambda$. Since
$-\partial(K + U)/\partial V = P_\text{int}$ at fixed scaled coordinates,

$$f(\lambda) \equiv -\frac{dE}{d\lambda}
  = -2\lambda\left(P_0 - P_\text{int} - \frac{k_BT}{2\lambda^2}\right),$$

and the step of eq. (S7), with $D = k_BT\beta_T/(4\tau_p)$ and the period
$\Delta t$ of the barostat, is

$$\lambda' = \lambda + \frac{D}{k_BT} f(\lambda)\, \Delta t + \sqrt{2D\Delta t}\, R.$$

The probability of a step and of its reverse are Gaussians of the same
variance $2D\Delta t$ about $\lambda + a f$ and $\lambda' + a f'$, with
$a = D\Delta t/k_BT$, $f = f(\lambda)$, $f' = f(\lambda')$ at the scaled
configuration. With $d = \lambda' - \lambda$,

$$k_BT \ln\frac{T(\lambda\to\lambda')}{T(\lambda'\to\lambda)}
  = k_BT\,\frac{(d + a f')^2 - (d - a f)^2}{4D\Delta t}
  = d\,\frac{f + f'}{2} + \frac{D\Delta t}{4k_BT}\left(f'^2 - f^2\right),$$

and the work of a step, which detailed balance would make zero, is

$$w = \Delta K + \Delta U + P_0 \Delta(\lambda^2) - k_BT\, \Delta\ln\lambda
  + d\,\frac{f + f'}{2} + \frac{\beta_T \Delta t}{16\,\tau_p}\left(f'^2 - f^2\right).$$

Its exponential has mean one from equilibrium,
$\langle e^{-w/k_BT}\rangle = 1$. The coefficient of the last term in the
image of the paper's eq. (S11), $\beta_T\Delta t/(4\tau_p k_BT)$, is not an
energy; `scripts/validation/barostat/effective_energy.py` checks on a model
in one variable that the closed form above equals the definition to
rounding and satisfies the identity, and that the other does not. The
effective energy is the conserved energy of Section 11.4 plus the sum of
$w - \Delta K - \Delta U$ over the steps of the volume.

To compute it, $P_\text{int}$ in $f$ and $f'$ must have one definition
before and after a scaling. With rigid groups scaled about their centers
of mass it is the molecular pressure; at the scaled positions MDIR has the
atomic virial without the constraints, so the molecular one needs
$\sum_\text{groups}\sum_i (\mathbf r_i - \mathbf R)\cdot\mathbf F_i$
as well. Without constraints the atomic pressure serves.

### 11.6 Langevin dynamics

`method = "LANGEVIN"` of `[thermostat]`, with `friction` $\gamma$ in 1/ps
(D135). Each step is that of velocity Verlet with the friction and the
noise between the two halves of its drift, the middle scheme
[[Zhang2019]](references.md#zhang2019), BAOAB in the terms of
[[Leimkuhler2013]](references.md#leimkuhler2013):

$$
\begin{aligned}
\mathbf v &\leftarrow \mathbf v + \tfrac{\Delta t}{2}\,\mathbf F/m, &
\mathbf x &\leftarrow \mathbf x + \tfrac{\Delta t}{2}\,\mathbf v, \\
\mathbf v &\leftarrow c\,\mathbf v + \sqrt{(1 - c^2)\,k_B T/m}\;\mathbf R, &
\mathbf x &\leftarrow \mathbf x + \tfrac{\Delta t}{2}\,\mathbf v,
\end{aligned}
$$

with $c = e^{-\gamma\Delta t}$ and $\mathbf R$ three normal numbers of the
particle (Section 11.3), then the constraints of the positions, the
velocities that take the atoms there over the step, the evaluation, the
second half kick, and RATTLE, as in a step without it. With constraints
the noise has a part along them, which the constraints of the positions
and the velocities of the change remove, as the middle scheme of OpenMM
does; [[Leimkuhler2016]](references.md#leimkuhler2016) projects after
each part instead. Leapfrog takes the same step, whose stored velocities
are those of its middle. A particle of mass 0 keeps its velocity.

| Item | Rule |
|---|---|
| Where | In every program of a step: `step`, `step_energy`, `step_virial`, and `step_trotter`, which scales the cell after the friction and the noise, so that the kinetic energy of the count of its work is that of the velocities it scales |
| The key | The seed; the number of the step, from a counter on the host that every step advances from the first step of the run (D129), so that a continued run takes the numbers of one that does not stop; the number of the particle in the input, so that the order of the positions does not change them |
| The momentum | Not kept: the degrees of freedom are $3N - N_c$, less 3 only if `center_of_mass_interval` removes the motion of the center of mass |
| `interval` | Paces the removal of the motion of the center of mass and the barostat, which act at the end of a period; 10 by default with a barostat, otherwise none |
| The log | No conserved energy: the run does not count the energy that the friction and the noise exchange with the bath (the roadmap) |
| The virial of the constraints | That of the forces that keep the bonds at their lengths at the end of the step, solved from its positions, velocities, and forces, $\mathbf e\cdot(\mathbf a_q - \mathbf a_p) = -\lvert\mathbf v_q - \mathbf v_p\rvert^2/d$, with the matrix of RATTLE; the changes of SHAKE and RATTLE within the step hold the friction and would give a pressure higher by $\gamma\Delta t/2$ of the virial of the constraints, 39 bar per 1/ps for water at 2 fs |

**Measured.** The noise of one step from velocities of 0 against Philox
of the host: within $5\times10^{-7}$ of $\sqrt{k_B T/m}$, the part of the
forces, on the CPU and on a GPU in double and mixed precision
(`langevin.test`, `langevin-gpu.test`). An ideal gas of 256 particles at
10 K coupled to 300 K with $\gamma$ = 5/ps relaxes at 9.6 ± 0.3/ps, against
$2\gamma$ = 10/ps; its mean kinetic energy over 50 ps is 228.0 ± 0.8
kcal/mol against 228.9. A run stopped at its checkpoints and continued
ends in the state of one that does not stop, bit for bit, on the CPU and
on a GPU. The tests of the ensembles of 1039 OPC waters at 2 fs with
$\gamma$ = 1/ps (`scripts/validation/ensembles/run.py --thermostat
langevin`, 5 ns a run): the density at 1 atm with cell rescaling
0.99611 ± 0.00030 g/cm³ against 0.99674 ± 0.00031 with velocity rescaling,
the slope of two pressures within 0.9 standard errors, that of two
temperatures within 2.0; the kinetic energy of the end of the step is
0.13% below $N_fk_BT/2$ (4 to 6 standard errors), so that its distribution
fails the test of Kolmogorov and Smirnov: at 2 fs the velocities of the end
of a step of the middle scheme are not those that it samples exactly,
which are those of the middle.

## 12. A cell that changes

A neighbor structure is valid in the cell that it was built in (Section
8.2 of ops-m0.md). The barostat changes the cell.

| | A. The structure is built after every change of the cell | B. The structure follows the cell |
|---|---|---|
| How | Nothing new: the refresh finds another cell and builds | The barostat scales the positions that the structure was built at, and the reach of the structure shrinks with the smallest factor. The structure is valid while no particle has moved more than half of what is left of the skin. |
| Builds | At least one for each step of the barostat | As without a barostat |
| With a barostat every 10 steps and a build every 10 to 20 steps | Up to twice the builds | |

Proposal: A for M1, and B when the builds are measured to cost. *Done:
B (D80).* The test of validity takes the scale of each axis since the
build, $\mathbf m = \mathbf L \oslash \mathbf L^\text{ref}$, and holds while
$2\max_i \lVert \mathbf x_i - \mathbf m \odot \mathbf x^\text{ref}_i \rVert \le
\min(\mathbf m) R - r_c$, so a scaling of the cell rebuilds a structure only
when it would leave a pair within the cutoff out; the proof is in Section
4.1 of the white paper (`docs/paper/04-neighbors.md`). The inner list of a
dual list takes the same test against its last pruning (D114).

The cell is a value of the state that the loops carry (S1). The order of
the particles (D44) and the incidence structures do not depend on it. The
grid of particle mesh Ewald keeps its number of points and the correction
for the dispersion follows the volume.

## 13. Input

### 13.1 Formats

| Format | Files | Keywords of `[input]` |
|---|---|---|
| Amber | `prmtop` (topology), `inpcrd` or `rst7` (positions, velocities, and the cell) | `topology`, `coordinates` |
| GROMACS | `.top` with `.itp` (topology), `.gro` (positions, velocities, and the cell) | `topology`, `coordinates` |

| Item | Proposal |
|---|---|
| The readers | MDIR has readers of its own. They accept what the engines accept and share no code with them. The engines and their tools are under the GPL and the LGPL; their behavior is learned from their documentation and their code, and written down as a specification, from which the readers are written. |
| What a reader hands on | A description of the system that does not depend on the format: particles, types, tuple sets with their parameters, the table of pairs of types, exclusions, pairs three bonds apart, constraints, the cell. A format is a reader more. |
| Units and forms | A reader converts to the units and the forms of [conventions.md](conventions.md) (D62): for Amber, twice the force constants of bonds and angles, the charges divided by 18.2223, and $\sigma$ and $\varepsilon$ of Lennard-Jones from `ACOEF` and `BCOEF`. The control file stays in Å and kcal/mol (D36). |
| What is an error | What M1 cannot run: 10-12 terms, 12-6-4 terms, polarizability, virtual sites other than those of water of four sites. Each with the milestone that brings it. |
| Formats of other engines | Planned |

The GROMACS preprocessor and sections are as in the specification of the
format; M1 reads the sections that an Amber force field in the format of
GROMACS needs: `defaults`, `atomtypes`, `nonbond_params`, `pairtypes`,
`bondtypes`, `angletypes`, `dihedraltypes`, `moleculetype`, `atoms`,
`bonds`, `pairs`, `angles`, `dihedrals`, `exclusions`, `settles`,
`system`, `molecules`.

A dihedral without parameters takes those of the first entry of
`[ dihedraltypes ]` of its function with the most types that are not
wildcards, matched to its atoms in the order of the file or the reverse;
functions 1 and 9 share a table, and an entry of function 9 brings the
entries of the same types on the lines that follow it. When the force
field defines `_FF_AMBER_LEAP_ATOM_REORDERING`, as amber14sb.ff and
amber19sb.ff of GROMACS do, the atoms of each dihedral are then put in the
order that LEaP gives them, as grompp of GROMACS 2026.3 does, while the
parameters stay those of the order of the file. With the types of the
matching entry, a blank for a wildcard: a dihedral whose entry has a force
constant is reversed when its first type sorts after its last, or the two
are equal and its second sorts after its third; an improper whose types,
or whose atoms, an earlier improper of the same function had, over all
molecule types in the order of the file, has the three atoms around the
third, the central one, sorted by type and then by index. The first
improper of its types keeps its order. Reversing a dihedral leaves its
angle alone, but which atoms are the outer ones of an improper changes its
angle: in ubiquitin with amber19sb.ff, the order of the file gives
impropers 0.896 kcal/mol above those of grompp, and the order of LEaP
agrees with them to 5 × 10⁻⁵ kcal/mol (`gromacs-leap-order.test`).

The functions read are those of the Amber, CHARMM, OPLS-AA, and GROMOS
force fields: bonds of function 1, and of function 2, the quartic bond of
GROMOS, a term given by an expression (D136); angles of function 1, of
function 2, the angle of GROMOS in its cosine, a term given by an
expression as well, and of function 5, whose Urey–Bradley
term ½ k_UB (r₁₃ − r13)² between the outer atoms becomes a set of pairs of
its own; dihedrals of functions 1, 4, and 9, and of function 2, the
harmonic improper ½ k (ξ − ξ0)² with ξ − ξ0 taken in [−π, π), and of
functions 3 (Ryckaert and Bellemans, $\sum_n C_n \cos^n(\varphi - \pi)$)
and 5 (Fourier), as terms given by expressions (D136); pairs of
function 1; CMAP of function 1 (D121, [charmm-m1.md](charmm-m1.md)). The
Lennard-Jones of a topology takes `lennard_jones_modifier` in its kernel:
`POTENTIAL_SHIFT`, `FORCE_SWITCH` (the polynomial of GROMACS), or
`POWER_FORCE_SWITCH` (VFSWITCH of CHARMM, which also switches the pairs
three bonds apart).

A topology of Amber is read in either format (D133): with `%FLAG` lines,
or in the format before Amber 7, a title, 30 pointers in lines of 12, and
the sections in a fixed order whose lengths the pointers give (a section
of no items is a blank line), then the solvent pointers, the molecules,
and the box when IFBOX is set; that format has no NUMEXTRA, no SCEE and
SCNB (1.2 and 2.0), and no atomic numbers (from the masses). The reader
takes the conventions that pmemd 26 follows, read from its source for
compatibility:

| Item | Rule |
|---|---|
| Negative periodicity | A dihedral whose type has a negative periodicity also has the term of the next type, and so on up to the first type of positive periodicity |
| Phase | A phase within $10^{-3}$ of $\pm\pi$ is $\pm\pi$; a cosine or sine of the phase below $10^{-6}$ in magnitude is 0. Topologies write $\pi$ as 3.141594 or 3.1415927; JAC's dihedrals are 965.2152 kcal/mol so and 965.2154 without |
| 1-4 pairs | From the entries with positive third and fourth indices, with SCEE and SCNB of the last type of the entry's chain; a type of periodicity 0 carries none. Two entries of the same pair, or a factor of 0, are errors, where Amber counts the pair twice or drops its Coulomb term |
| Exclusions | Without extra points those of the file; with them built again from the bonds, the ends of the angles and of every dihedral, an extra point taking those of its owner (Section 19) |

What still differs from pmemd, by design: the cosine of an angle is not
clamped to $\pm0.999$ (pmemd evaluates an angle within 2.56° of linear at
that bound); a dihedral whose cross products are below $10^{-3}$ Å² is not
dropped; a net charge below 0.01 e is not spread over the atoms to make
the system neutral (Factor IX: $3.1	imes10^{-4}$ e, below the printed
digits); water is found by its residue and atomic numbers (O, H, H), not
by the names O, H1, H2 in any order; and on an odd grid the aliasing
factor of `influence = "OPTIMAL"` takes the principal frequency at the
index $(N-1)/2$, where pmemd takes $-(N+1)/2$ ([pme-m1.md](pme-m1.md),
Section 1.1). The Coulomb
constant, $eta$ from $\operatorname{erfc}(eta r_c)$ rather than
$\operatorname{erfc}(eta r_c)/r_c$, and the average of the correction
for the dispersion are conventions of MDIR (Sections 7.2 and 8).

A plain cutoff of the Coulomb term does not conserve the energy: a pair
of charged particles that crosses the cutoff changes the energy by
`f q_i q_j / r_c`, some 10 kcal/mol for two waters at 9 Å. sander shows
the same (155 kcal/mol in 10 steps of 0.5 fs for the dipeptide in water).
The intermediate stage therefore compares the terms at a configuration;
a test of the conservation of the energy needs a Coulomb term that goes
to zero smoothly, or particle mesh Ewald.

### 13.2 Keywords of the control file

The tables and their keywords, as `lib/Driver/Control.cpp` takes them; a
keyword that a table does not take is an error with its line. The manual,
with units, defaults, and the combinations that are errors, is Appendix A
of the white paper (`docs/paper/A-control-file.md`), whose example is the
output of `mdir template amber` (`scripts/paper/check-appendix.sh`).

| Table | Keywords |
|---|---|
| `[input]` | `topology`, `coordinates`, `format`, `checkpoint`, `include_paths` (directories of includes of GROMACS), `defines` (macros, as `-D` of grompp) |
| `[output]` | `trajectory`, `checkpoint`, `energy_interval`, `trajectory_interval`, `checkpoint_interval` |
| `[energy]` | `cutoff`, `switch_distance`, `pairlist_distance`, `pruned_distance`, `rebuild_interval`, `lennard_jones_modifier`, `coulomb_modifier` (`"NONE"`, `"POTENTIAL_SHIFT"`), `dispersion_correction`, `electrostatics` (`"CUTOFF"`, `"PME"`, `"REACTION_FIELD"` with `reaction_field_dielectric`, D140), `[[energy.pair]]` (with a topology over its pairs, with `groups`, D137), `[[energy.function]]` (D138), `[[energy.bond]]`, `[[energy.angle]]`, `[[energy.dihedral]]` over particles or the centers of groups (D136, D139), and for a system without a topology `[[energy.type]]`, `[[energy.pair_override]]` |
| `[pme]` | `tolerance`, `beta`, `max_spacing`, `grid`, `order` (4, 6, 8), `influence` (`"SPME"`, `"OPTIMAL"`) |
| `[dynamics]` | `integrator`, `time_step`, `steps`, `seed`, `center_of_mass_interval` |
| `[minimize]` | `method`, `steps`, `initial_step` |
| `[ensemble]` | `ensemble` (`"NVE"`, `"NVT"`, `"NPT"`), `temperature`, `pressure` |
| `[thermostat]` | `method = "V-RESCALE"`, `time_constant`, `interval` |
| `[barostat]` | `method = "C-RESCALE"`, `time_constant`, `compressibility`, `coupling = "ISOTROPIC"`, `work`, `interval` |
| `[constraints]` | `hydrogen_bonds`, `rigid_water`, `water_residues`, `analytic_bonds`. A topology with SETTLE needs `rigid_water = false` to run its waters flexible, with their bonds |
| `[[restraints]]` | `selection` (a mask of Amber), `force_constant`, `reference_scaling` (`"CENTER"` or `"ALL"`, D124) |
| `[boundary]` | `type`, `box` |
| `[execution]` | `target`, `threads`, `precision`, `neighbor_capacity`, `fast_math`, `spatial_order`, `deterministic`, `neighbor_structure` |

The thermostat and the removal of the motion of the center of mass are
done (M1g, D67), and so is the barostat (Section 11.4).

## 14. The command line

One program with subcommands (D59). The programs of M0 become
subcommands; `mdir-opt` stays a program for developers.

| Command | Does |
|---|---|
| `mdir run control.toml` | Reads the input, compiles the run, and runs it (`mdir-run` of M0). `--continue`, `--no-append`, and `--max-walltime` carry a run over more than one job (D129 to D132, driver-m0.md, Section 2.7) |
| `mdir template md` | Prints a control file with every keyword |
| `mdir check control.toml` | Reads the input and prints what it found: particles, types, the mass and the density, the cell, the degrees of freedom, the integrator and the target; with a topology also the tuples of each kind, the charges and their sum. Compiles nothing, so it gives no energies: a run of zero steps does. |
| `mdir emit control.toml --stage=<module\|lowered>` | Prints the program of the run, as `--emit` does in M0 |
| `mdir checkpoint <file>` | Prints what a checkpoint holds (`mdir-checkpoint` of M0) |
| `mdir version` | The version, the LLVM it was built with, the targets it has |

The command line is independent of the physics, so it is a stage of its
own right after the execution of terms over tuples (Section 18).

## 15. Validation

| Test | Compared with | Tolerance |
|---|---|---|
| Energy, forces, and virial of each kind of bonded term, for chains of a few particles | A script that evaluates the definitions and checks its forces against finite differences | 1e-10 |
| The same with particles in line and with a dihedral of $\pi$ | The same | 1e-10 |
| A neighbor structure with exclusions | A search over all pairs | Exact |
| Lookup in a table, pairs three bonds apart | A script | 1e-10 |
| Particle mesh Ewald | An Ewald sum [[Ewald1921]](references.md#ewald1921) with many wave vectors, in a script | That of the parameters |
| SETTLE and SHAKE | A script; the constrained distances | That of the tolerance |
| The factor of the thermostat and of the barostat | A script with the same generator | 1e-12 |
| A run that continues from a checkpoint, with thermostat and barostat | The run that was not interrupted | Exact |
| Alanine dipeptide in water: the energy of each term at the start, and the forces | AmberTools and GROMACS, each from its own format | 1e-5, relative |
| The same system: the conservation of energy at constant energy, and temperature, pressure, and density at constant temperature and pressure | The engines, within the statistical error | |
| The JAC benchmark: run times | The engines on the same device | |

The terms of a run from a topology of GROMACS were compared with those of
GROMACS 2026.3 (mixed precision) for the peptide ALA-GLY-SER-LYS-ASP-TRP
in flexible water, built with `pdb2gmx` for each force field that GROMACS
ships (`scripts/validation/gromacs/run.sh`):

| Force field | Bonds, angles, dihedrals, pairs 1-4, Lennard-Jones, dispersion |
|---|---|
| amber99sb-ildn, amber99sb, amber03, amber14sb | Agree to 5 × 10⁻⁶ or better, each term |
| amber99sb-ildn with TIP4P-Ew, whose sites are virtual (Section 19) | Agree to 4 × 10⁻⁶ or better, each term; the correction for the dispersion counts the sites among the particles in both |
| amber19sb, with CMAP (Section 20) | Agree to 3 × 10⁻⁶ or better, each term; CMAP to 9 × 10⁻⁷ |
| charmm27, with Urey–Bradley angles (function 5), harmonic impropers (function 2), and CMAP (D121) | Agree to 4.3 × 10⁻⁶ or better, each term; angles with their Urey–Bradley terms to 3.5 × 10⁻⁷, dihedrals with the impropers to 1.8 × 10⁻⁷ |
| oplsaa | Agree to 4.3 × 10⁻⁶ or better, each term; the dihedrals of Ryckaert and Bellemans, terms given by an expression (D136), to 1.6 × 10⁻⁶ |
| gromos54a7, with the bonds and angles of function 2 of GROMOS, terms given by expressions (D136) | Agree to 6.5 × 10⁻⁶ or better, each term, but the angles of the flexible SPC water, 14 kJ/mol in all, to 2.1 × 10⁻⁵ |

Coulomb (SR) is compared with MDIR's reaction field at a permittivity of
78.5 (D140): `Coulomb`, `Coulomb excluded`, and `Coulomb self` agree with
it within 2.4 × 10⁻⁵ for every force field, and with OpenMM in double
precision within 1.5 × 10⁻⁹.

The two readers can be compared with each other when the same system is
in both formats. ParmEd converts a topology of Amber to the format of
GROMACS: the factors of the pairs three bonds apart become `fudgeQQ` and
`fudgeLJ`, and NBFIX pairs go to `[ nonbond_params ]`. A system with more
than one value of the factors does not convert faithfully. GROMACS
applies the NBFIX pairs to the pairs three bonds apart, as Amber does:
with gen-pairs, grompp generates them from the matrix of the nonbonded
parameters after `[ nonbond_params ]`, scales ε by `fudgeLJ`, and then
applies `[ pairtypes ]` as they are; the reader does the same, and the
three agree with a rerun of GROMACS 2026.3 (`gromacs-nbfix-pairs.test`).

Neither engine is installed on the development machine. Both are built
into the home directory.

**Forces on every particle** (2026-10-02, `scripts/validation/forces/run.sh`).
MDIR in double precision on the CPU writes the forces of the input
coordinates into a checkpoint after one step of $10^{-9}$ ps from zero
velocities; sander writes its forces the same way (`ntwf`, in single
precision), GROMACS by a rerun. For sander the charges of the topology
that MDIR reads are scaled by $\sqrt{332.0522173/332.0637133}$, its
Coulomb constant, and sander keeps the net force of particle mesh Ewald
(`netfrc = 0`; by default it removes it, which shifted every force by
the net force over the number of atoms, 3.5e-5 of the rms force). The
differences, relative to the rms force:

| System | Against | rms | Largest |
|---|---|---|---|
| Alanine dipeptide in TIP3P, ff14SB, a plain cutoff of 9 Å | sander | 2.0e-7 | 4.1e-6 |
| Alanine dipeptide in OPC, particle mesh Ewald of the same β, grid, and order | sander | 1.25e-7 | 2.5e-6 |
| The dipeptide in TIP3P from its topology of GROMACS, particle mesh Ewald of the same β, grid, and order | GROMACS 2026.3, mixed precision | 6.2e-6 | 7.7e-5 |

The differences with sander are those of its forces in single precision.
GROMACS computes again at the positions of MDIR after the step, which
SETTLE moves: the waters of the `.gro`, rounded to 0.001 nm, are not
rigid to 10⁻³ Å, and against the positions of the file the forces of the
waters differed by 2.6e-4. Its reciprocal energy agrees to 2.6e-6, and the
rest of its Coulomb energy to 1.1e-6 of its largest terms, the self
energy and the excluded pairs, which cancel to 3%; MDIR in mixed
precision on a device differs from MDIR in double precision by 5e-6
(rms) on the same input.

**Ubiquitin in OPC against GROMACS** (2026-10-02,
`scripts/validation/protein/run.py`). 1UBQ with amber19sb.ff and its OPC
from `pdb2gmx` (ff19SB with CMAP, 76 residues, neutral), 5700 waters,
24,031 particles in a triclinic cell of 5.4 × 5.6 × 6.0 nm, minimized and
then equilibrated by GROMACS 2026.3 for 200 ps at 300 K and 1 bar. The
terms of the equilibrated state, MDIR in double precision on the CPU
against a rerun of GROMACS (mixed precision) at the same positions, with
the sites placed from their atoms on both sides; a cutoff of 9 Å with no
shift, the correction for the dispersion, and particle mesh Ewald of the
same β (erfc(β r_c) = 1e-5, which both solve for), grid, and order:

| Term | MDIR, kcal/mol | Relative to GROMACS |
|---|---|---|
| Bonds | 209.2124 | −1.7e-6 |
| Angles | 620.1280 | 5.2e-7 |
| Dihedrals, proper and improper | 444.3518 | 1.5e-7 |
| CMAP | 57.2390 | 3.4e-7 |
| Lennard-Jones 1-4 | 272.9706 | −3.1e-7 |
| Coulomb 1-4 | 2875.2110 | 3.8e-9 |
| Lennard-Jones | 10207.9679 | −1.2e-6 |
| Dispersion | −540.0697 | 4.5e-7 |
| Coulomb, all but 1-4 | −87461.2462 | 6.6e-6; 1.9e-6 against tables of the Ewald correction |
| Total | −73314.2353 | 7.7e-6; 2.0e-6 against tables |

Two differences were found on the way. The impropers were 0.896 kcal/mol
above GROMACS, by an amount that did not add up over subsets of the lines
of the topology: grompp puts the atoms of dihedrals in the order of LEaP
when the force field asks for it (Section 13.1), and the reader now does
too. The Coulomb term is the error of GROMACS's SIMD kernels, which
compute the Ewald correction by an analytical approximation in single
precision: its error is the same for the excluded pairs of every rigid
water, and over 5700 waters it adds up to 0.55 kcal/mol. The waters alone
show it: GROMACS's short-range Coulomb term is below MDIR's by 0.554
kcal/mol with its default kernels, 0.121 with tables of the correction
(`GMX_NBNXN_EWALD_TABLE`), and 0.136 with its plain C kernel; translating
the whole system moves GROMACS's
short-range Coulomb term by 0.02 kcal/mol at most. The reciprocal sums
agree to 4e-6 (592.8846 against 592.8822).

The rates on GPU 0 (an RTX 3090 at 300 W), 60,000 steps of 2 fs from the
equilibrated state with the second half timed, mixed precision in both:
MDIR with groups and the dual list (outer 12 Å, inner 9.6 Å, within 0.4%
of the best of 11 to 12 Å and 9.4 to 9.8 Å), SHAKE on the bonds of
hydrogen, and SETTLE; GROMACS with its nonbonded terms and PME on the GPU
and its update on the CPU, which virtual sites need, and its own pair
list (`verlet-buffer-tolerance` 0.005 kJ/mol/ps per atom):

| Ensemble | MDIR, ns/day | GROMACS, ns/day | MDIR / GROMACS | Energy changed by: MDIR | GROMACS |
|---|---|---|---|---|---|
| NVE | 605.2 ± 0.4 | 842.9 ± 2.9 with nstlist 80 (two repeats; the third lost 0.4 s to writing its output); 689.7 with the nstlist 10 that it keeps at constant energy | 71.8 ± 0.2% | 2.4e-5 to 4.4e-5 | 1.7e-4 to 2.0e-4 |
| NPT | 571.5 ± 0.7 | 865.5 ± 4.4; 771.8 at a tolerance of 5e-5 | 66.0 ± 0.4% | 1.6e-6 to 1.2e-4 (conserved energy) | 6.7e-3 to 7.1e-3 (conserved energy; 5.3e-3 to 6.9e-3 at tolerances of 5e-4 and 5e-5) |

The rates are means and sample standard deviations over three repeats
(2026-10-02, at D118, each program after the other); the other settings of GROMACS
are from one run each.

MDIR's conserved energy at constant pressure moved by −228 kcal/mol/ns
with the count of D92, the bias with rigid groups (below) as on 1039
waters (−234), and by 6.2e-5 to 1.4e-4 of its value over the 120 ps with
that of D116 (the row above; 543.5 ns/day in one run with the count of
D92). GROMACS's moves at
constant temperature without the barostat too, by +370 to +610
kcal/mol/ns (`nsttcouple` 1 or 25, τ_T 1 or 10 ps, `nstcalcenergy` 1 or
100), against −70 for its total energy at constant energy; not
investigated further.

The device time of a step (nsys, NVE; MDIR over 20,000 steps, GROMACS over
the 3528 that its profile captured; MDIR's kernels sorted by their kind):

| Part | MDIR, µs | GROMACS, µs |
|---|---|---|
| Loops over pairs | 91.6 (3 launches; the inner list of 9.6 Å) | 87.7 (one kernel; its list of 10.7 Å, built every 80 steps) |
| PME | 47.3 | 76.7, on a second stream, beside the nonbonded kernel |
| Loops over particles: the drift, the kicks, and the other loops over particles | 48.9 | The update on the CPU |
| Loops over tuples: bonded terms and excluded pairs 22.5, SETTLE 11.8, placement of the sites 4.9 | 39.3 | Bonded terms, SETTLE, and sites on the CPU; the excluded pairs in the nonbonded kernel |
| Builds and prunings | 34.6 | In the nonbonded kernel every 80 steps |
| Layout and reductions | | 18.3 |
| Total on the device | 261.8 on one stream | 182.8 on two |
| Wall time of a step | 310 | 214 |

The loop over pairs costs the same in both, and MDIR's PME is faster.
GROMACS runs beside that loop what MDIR runs after it: PME on a second
stream, and the bonded terms, the update, and the constraints on the 64
threads of the host; the host's time between MDIR's launches is 48 µs of
a step of 310. MDIR's second stream for the reciprocal sum (D81, D87)
gives 580.8 ns/day here against 584.0 without. On JAC (TIP3P, 8 Å)
MDIR's rate is 125% of pmemd.cuda's (Section 10 of the white paper); on
this system, with four sites per water and a cutoff of 9 Å, its pair loop
and its list are as fast as GROMACS's, and the rest of the step is not.

**The ensembles** (2026-10-02, `scripts/validation/ensembles/run.py` and
`analyze.py`). A box of 1039 OPC waters from tleap, rigid (M-SHAKE in the
mixed mode), particle mesh Ewald, a cutoff of 9 Å with the correction for
the dispersion, 2 fs, groups and the dual list, the thermostat and the
barostat every 25 steps ($\tau_T$ = 1 ps, $\tau_P$ = 2 ps), 5 ns for each
run from one state equilibrated for 200 ps at 300 K and 1 atm, on the GPU
in mixed precision. Samples are spaced by their statistical inefficiency;
the slopes are maximum-likelihood estimates of $\ln P_2/P_1$ [[Shirts2013]](references.md#shirts2013).

| Test | MDIR | Expected | Standard errors |
|---|---|---|---|
| Mean kinetic energy at 300 K, kcal/mol | 1856.98 ± 0.66 | 1857.34 ($N_f k_BT/2$, $N_f$ = 6231) | −0.55 |
| Its variance at 300 K | 1103 ± 31 | 1107 ($N_f (k_BT)^2/2$) | −0.13 |
| Mean kinetic energy at 306 K | 1893.86 ± 0.72 | 1894.49 | −0.87 |
| Its variance at 306 K | 1194 ± 36 | 1152 | +1.16 |
| Slope of $\ln P_{306}(U)/P_{300}(U)$, mol/kcal | 0.03315 ± 0.0016 | 0.03289 ($\beta_{300} - \beta_{306}$) | +0.17 |
| Slope of $\ln P_{300\,\text{atm}}(V)/P_{1\,\text{atm}}(V)$, Å⁻³ | −0.00708 ± 0.00035 | −0.00731 ($-\beta\,\Delta P$) | +0.66 |

At 300 K and 1 atm the density is 0.99674 ± 0.00031 g/cm³ and the
compressibility (4.71 ± 0.26) × 10⁻⁵ /bar; GROMACS 2026.3 with its own
OPC (amber19sb.ff, its update on the CPU, which virtual sites need), the
same cutoff, PME, couplings, and length, gives 0.99698 ± 0.00030 g/cm³ and
(4.41 ± 0.23) × 10⁻⁵ /bar, 0.6 and 0.9 standard errors apart.

**The conserved energy at constant pressure drifted with rigid groups**
(found and fixed, D116). In the same runs the conserved energy at constant volume moved by
−0.16 kcal/mol/ns, and at constant pressure by −234 kcal/mol/ns, where
GROMACS's moved by −0.37. Runs of 100 ps from the same state: the default
count of the work, of Trotter type, −16.3 kcal/mol (−18.0 in double
precision, −22.1 with a period of 10 steps, −25.7 at 1 fs), the exact
count +1.3, the first-order count +209; over 20 ps a period of one step
drifts as much as one of 25 (−5.4 and −6.3). With flexible water at 0.5
fs, the Trotter count, the exact count, and constant volume drift alike.
So the Trotter count was biased when rigid groups scale by their centers:
a bias proportional to the time, independent of the period, as a term of
second order in the strain would be, about 2 × 10⁴ kcal/mol times the
sum of $(\ln\mu)^2$. Twice the internal kinetic energy, which the count
carries in $\mathcal W$ before and after the velocities are scaled,
accounts for a twelfth of it. The sampling was not in question (the test
of two pressures, the density, and the compressibility above); the count
that measures it was. The count took the virial of the step with twice
the internal kinetic energy, which is the virial of the groups only when
the forces of the constraints are those of one configuration; in the step
that scales, the constraints of the positions take the forces before the
scaling and those of the velocities the forces after it. Counted from the
virial of the groups of the evaluations (D116), the conserved energy of
the water box moves over 1 ns by +16.2 ± 11, +12.7 ± 13, and +12.0 ± 12
kcal/mol/ns at periods of 5, 25, and 100 steps (the exact count +3.7 ±
3.5), and that of ubiquitin in OPC over 0.6 ns by +11.4 ± 33, from −228.
The count changes nothing in the trajectory, so the tests of the
ensembles above stand.

**The GPU against the CPU over 20 ps** (`scripts/validation/gpu-cpu/run.sh`).
The dipeptide in OPC of `pme-settle.test`, 20,000 steps of 1 fs in double
precision on each from the same state: the rows of the two logs agree to
every printed digit for 500 steps, and then part as the dynamics amplifies
the rounding of sums added in different orders, by 1.1 kcal/mol at 5 ps.
Over the last 16 ps the mean potential energy, $-3807.9 \pm 2.5$ against
$-3801.8 \pm 2.5$ kcal/mol, and the mean temperature, $419.6 \pm 1.0$
against $417.2 \pm 1.0$ K (errors from ten blocks), differ by 1.8 and 1.6
standard errors; the input, heating from 300 K as it relaxes, conserves
its total energy to 1.0e-3 on the CPU and 7.7e-4 on the GPU.

**The GPU against the CPU at 300 K** (2026-10-02,
`scripts/validation/gpu-cpu/nvt.py`). The run above heats as it relaxes,
and the means of a run that is not stationary are not comparable. The
same system, equilibrated for 100 ps at 300 K and 1 atm on the GPU, then
500 ps at constant volume on each in double precision (2 fs, SHAKE and
SETTLE, the thermostat every 10 steps with a seed of its own, so that the
two are independent), errors from samples spaced by their statistical
inefficiency (65 and 133 independent samples of the potential energy):
the mean potential energy −4763.3 ± 4.1 against −4759.2 ± 2.6 kcal/mol
(−0.84 standard errors), its standard deviation 32.6 ± 2.9 against 30.1 ±
1.9 (+0.72), the mean temperature 300.17 ± 0.72 against 301.03 ± 0.48 K
(−1.00), the mean total energy −4055.5 ± 5.9 against −4049.4 ± 3.6
(−0.88). The first run of this comparison, at the volume of tleap
(about 0.6 g/cm³), relaxed over the whole 500 ps; its CPU run also
found D117, a crash at the end of 250,000 steps.

## 16. Decisions

| # | Question | Decision | Recorded as |
|---|---|---|---|
| 1 | The target | All-atom: a protein in water with ff14SB and TIP3P; Martini deferred | D53 |
| 2 | Kernels in internal coordinates or over positions | Internal coordinates (Section 3.2) | D48 |
| 3 | Parameters for each tuple or for each kind | For each tuple (Section 2.3) | D47 |
| 4 | The strategy for terms over tuples | Every particle takes its own part; a sum from the member at place 0 (Section 5) | D49 |
| 5 | The thermostat and the barostat | Stochastic velocity rescaling (2007) and stochastic cell rescaling (2020), isotropic by default (Section 11.1) | D50 |
| 6 | A cell that changes | A build after every change, in M1 (Section 12) | D51 |
| 7 | The formats of the topology | Amber `prmtop` and `inpcrd`, and GROMACS `.top`, `.itp`, and `.gro` (Section 13) | D58 |
| 8 | The engines to compare with | AmberTools and GROMACS, built into the home directory | D58 |
| 9 | Groups with a thermostat each | One group in M1 | D50 |
| 10 | The correction for the dispersion | In M1, on by default (Section 7.2) | D53 |
| 11 | Parameters of pairs of types | A table for each pair of types, filled by the front end from a mixing rule and the pairs that the topology sets (NBFIX) (Section 7.1) | D57 |
| 12 | Particle mesh Ewald and constraints | In M1 (Sections 8.2, 9) | D55, D56 |
| 13 | An intermediate stage | Flexible water, a Coulomb cutoff, 0.5 fs, validated before particle mesh Ewald and constraints (Section 1.4) | D54 |
| 14 | The motion of the center of mass | Removed every `center_of_mass_interval` steps (Section 10) | D60 |
| 15 | The command line | One program with subcommands (Section 14) | D59 |
| 16 | The forms of the terms and the meaning of their parameters | One convention, [conventions.md](conventions.md), which the readers convert into: $\tfrac12 k$ for bonds and angles, $\sigma$ and $\varepsilon$ for Lennard-Jones | D62 |
| 17 | The molecules that SETTLE constrains | Chosen by `[ settles ]` or by `water_residues` (Section 9) | D63 |

## 17. Open questions

| Question | Recommendation | Needed by |
|---|---|---|
| Spreading of charges on a device without breaking the deterministic level | Atomic additions of integers in fixed point: one pass, and the order of the threads does not matter | The stage of particle mesh Ewald |
| The FFT on the host | A library under a permissive license, such as pocketfft; cuFFT on a device | The same |

## 18. Order of work

| Stage | Work | Runs | State |
|---|---|---|---|
| M1a | Tuple sets, internal coordinates, differentiation, loops over tuples on the CPU and on a GPU | Chains of particles with bonds, angles, and dihedrals, at constant energy | Done: energy, forces, and virial agree with a reference in double and mixed precision, on the CPU, with OpenMP, and on a GPU; 200 steps of velocity Verlet agree to 1e-9. Loops over tuples are not fused with each other yet. |
| M1b | The command line (Section 14) | The runs of M0 through `mdir run` | Done |
| M1c | Exclusions in the neighbor build; pairs three bonds apart | Chains with Lennard-Jones | Done: energy, forces, and virial agree with a reference on the CPU, with OpenMP, and on a GPU |
| M1d | Tables, NBFIX, the rule `product`, a Coulomb cutoff, the correction for the dispersion | A mixture of charged types | Done. The IR has `!md.table` and `md.lookup`. The driver takes `[[energy.pair_override]]`, which turns the parameters of a term into tables of pairs of types, the rule `product`, the name `coulomb` for the constant of CODATA 2018, and `dispersion_correction` for each pair term, with a plain cutoff. The correction takes the $r^{-6}$ part of the term and the $N(N - 1)$ ordered pairs, as GROMACS does; excluded pairs come out of the count once topologies are read (M1e). |
| M1e | The readers of both formats; renumbering of the members with the order | Alanine dipeptide in flexible water, at constant energy with 0.5 fs | Done (2026-10-02). Both readers: the terms of a run from a topology agree with sander but for the conventions, and with GROMACS to 10⁻⁶ for amber99sb-ildn, amber99sb, amber03, and amber14sb and for a made-up topology that uses the preprocessor and the defaults of bonded types (Section 15); the reader of Amber takes the format before Amber 7 and Amber's conventions of dihedrals, 1-4 pairs, and exclusions (D133). A run from a topology puts the particles in the order of the positions where it begins and where each segment begins; `md_exec.renumber` gives the members of the tuples at their new places, and the incidence structures and the exclusions are built again for the segment. The dipeptide in flexible water at constant energy, from the positions and velocities that sander takes: at 0.5 fs its energy fluctuates by 1.53 kcal/mol against sander's 1.21, and with the terms shifted at the cutoff the fluctuation falls by 4.0 to 4.2 with each halving of the step down to 0.0625 fs (`scripts/validation/nve`; white paper, Section 9.4) |
| M1f | Comparison of the intermediate stage with AmberTools and GROMACS | | Done: the terms and forces against sander, the terms against GROMACS 2026.3 for five Amber force fields and charmm27, and against CHARMM 51b1 (white paper, Table 9.1; `scripts/validation/forces`, `gromacs`, `charmm`) |
| M1g | Removal of the motion of the center of mass, random numbers, the thermostat | At constant temperature | Done (D67). Philox 4×32-10 agrees with the known answers of Random123, and the factor of the thermostat samples the canonical distribution of the kinetic energy. On a mixture of Lennard-Jones, the conserved energy changes by 2.5 × 10⁻⁵ over 20000 steps at constant temperature, on the CPU and on a GPU; a trajectory is the same for any grouping of steps into loops and across a restart. A drift of every particle is removed to a momentum of 10⁻¹³ amu nm/ps. The schedule uses the driver's own loops and `func.call`s, not yet the `dyn` ops of Section 11.2. |
| M1h | Particle mesh Ewald | With particle mesh Ewald | Done ([pme-m1.md](pme-m1.md), D69 to D71): the reciprocal sum on the host and on a GPU, in fixed point; the terms agree with an Ewald sum and with sander, and a run with rigid water conserves the energy. |
| M1i | Constraints: SETTLE, SHAKE, RATTLE | With 2 fs | Done with velocity Verlet (Sections 9.1 and 9.2): rigid water and rigid bonds of hydrogen keep their lengths to the precision of the trajectory; the target of D65 runs at 2 fs with PME and the thermostat (`test/Driver/ff19sb-gpu.test`). With leapfrog as well (Section 23). |
| M1j | The barostat, a cell that changes | At constant temperature and pressure | Done with velocity Verlet (Section 11.4, D72): isotropic stochastic cell rescaling. 1394 OPC waters at 300 K and 1 bar come to 0.9977 g/cm³ over 300 ps, with a compressibility from the fluctuations of the volume of 4.1 × 10⁻⁵ /bar; a trajectory is the same for any grouping of steps into loops and across a restart; the target of D65 runs at constant pressure on a GPU (`test/Driver/barostat-ff19sb-gpu.test`). With leapfrog as well (Section 23); anisotropic cells are to come. |
| M1k | Comparison with AmberTools and GROMACS; run times of the JAC benchmark | | Done (2026-10-03): the nine systems of the Amber suite from their own files (D133): every term at the start within 3.4 × 10⁻⁸ of pmemd 26 and the forces within 4 × 10⁻⁷ of the rms force (`scripts/validation/suite`); the rates as one model with pmemd.cuda, 104% to 132% of its rate (D134, white paper Table 10.2); the conservation over the measured runs (Table 10.2) and over 2 ns of JAC (Section 9.4 of the white paper); the volumes of JAC and Factor IX at constant pressure against OpenMM, pmemd.cuda, and GROMACS (Section 9.5) |
| M1l | CMAP (D65) | ff19SB | Done (Section 20). The terms of ACE-ALA-GLY-SER-NME with ff19SB in OPC agree with sander, and CMAP with an independent model to 12 digits; a peptide with amber19sb agrees with GROMACS. The energy of the peptide alone is conserved as the square of the time step. |
| M1m | Virtual sites: extra points of Amber, virtual sites of GROMACS (D65) | OPC | Done for water of four sites (Section 19, D68): the extra point of Amber and `[ virtual_sites3 ]` of function 1. The terms of alanine dipeptide in OPC agree with sander, and those of a peptide in TIP4P-Ew with GROMACS. The forces are the derivatives of the energy, and the virial that of a uniform scaling. Other frames of Amber and other kinds of GROMACS are rejected. |

## 19. Virtual sites

A virtual site is a point with a charge and no mass whose position follows
from those of a few atoms. OPC [[Izadi2014]](references.md#izadi2014) and
TIP4P-Ew put the negative charge of water on one, an extra point in Amber.
M1 has the site of water of four sites in both formats (D65, D68).

### 19.1 Placement

For a site `s` built from `i`, `j`, and `k`, with `d_ji = x_j − x_i` and
`d_ki = x_k − x_i` in the minimum image:

| Kind | Position | Parameters |
|---|---|---|
| Extra point of Amber | `x_s = x_i + a (û + v̂) / \|û + v̂\|`, with `û = d_ji / \|d_ji\|` and `v̂ = d_ki / \|d_ki\|` | `a`, the length of the bond from the oxygen to the extra point in the `prmtop` |
| `[ virtual_sites3 ]`, function 1, of GROMACS | `x_s = x_i + a d_ji + b d_ki` | `a`, `b` |

The extra point of Amber depends only on the directions of the two bonds
to the hydrogens. It lies where the linear rule puts it only when the two
bonds have the lengths that $a$ of GROMACS was computed for:
$a = b = d / (2 r_\text{OH} \cos(\theta/2))$.

### 19.2 Forces and virial

The force on a site goes to its atoms by the transpose of the derivative of
its position, $\mathbf F_k \mathrel{+}= (\partial\mathbf x_s/\partial\mathbf x_k)^{\mathsf T} \mathbf F_s$, and the site keeps none. For the
linear rule the atoms take $(1 - a - b)\, \mathbf F_s$, $a\, \mathbf F_s$, and $b\, \mathbf F_s$. For the
extra point of Amber, with $\hat{\mathbf b} = (\hat{\mathbf u} + \hat{\mathbf v}) / \lVert\hat{\mathbf u} + \hat{\mathbf v}\rVert$ and $P(\mathbf n) = I - \mathbf n \mathbf n^{\mathsf T}$:

$$
\begin{aligned}
\mathbf F_j &= \frac{a}{\lVert\mathbf d_{ji}\rVert\, \lVert\hat{\mathbf u} + \hat{\mathbf v}\rVert}\, P(\hat{\mathbf u})\, P(\hat{\mathbf b})\, \mathbf F_s, \\
\mathbf F_k &= \frac{a}{\lVert\mathbf d_{ki}\rVert\, \lVert\hat{\mathbf u} + \hat{\mathbf v}\rVert}\, P(\hat{\mathbf v})\, P(\hat{\mathbf b})\, \mathbf F_s, \\
\mathbf F_i &= \mathbf F_s - \mathbf F_j - \mathbf F_k.
\end{aligned}
$$

A hydrogen takes no force along its own bond: stretching the bond does
not move the site.

The virial of the pair terms is summed with the site as a particle. The
sum over the atoms after the move differs by $\sum_k (\mathbf x_k - \mathbf x_s) \otimes \mathbf F_k$ over
the three atoms of each site, which is 0 for the linear rule, since the
weights sum to 1, and is added for the extra point of Amber.

### 19.3 In a step

| Item | Rule |
|---|---|
| Placement | After every drift, and once where the run begins, whatever the file of coordinates says. The positions that a step returns have their sites placed, so frames and checkpoints need nothing more. |
| Forces | Moved after every evaluation of the forces, before the kick |
| Mass and velocity | A site has mass 0. A kick leaves a particle of mass 0 alone, and the velocities drawn at the start give it none. |
| Degrees of freedom | Three for each particle with a mass, less three |
| Exclusions | Amber: a site is excluded from its oxygen and from what its oxygen is excluded from, as sander rebuilds them; the list of the file must not exclude more. GROMACS: those of `[ exclusions ]`; no bond reaches a site. |
| Bonded terms | Amber: the bond from the oxygen to the extra point gives its length and is dropped, with every term that has an extra point. |
| In the IR | The placement and the move of the forces are `md.gather_tuples` over a tuple set `sites_amber` or `sites_linear` of arity 4, the site first, in the programs of the step; the change of the virial is an `md.sum_tuples` |

### 19.4 What sander and GROMACS do otherwise

| Item | sander | GROMACS |
|---|---|---|
| Constraints before the first step | None: the energies at the start are those of the file | SETTLE, with `continuation = no` |
| Velocity of a site in its output | 0 | That of the placement |
| Which atoms are sites | The atoms of type `EP`, with a frame chosen from the neighbors of the owner; MDIR takes the frame of water only | Those that a section of virtual sites places, whatever the particle type of their atom type says (V or D, and A in the OPC of amber19sb.ff, as grompp takes it), with no mass |

Frames of lone pairs and of TIP5P in Amber, and the other kinds of
virtual sites of GROMACS, are rejected.

### 19.5 Validation

| Test | Result |
|---|---|
| Dipeptide in OPC, the terms at the start, against sander (`test/Driver/amber-opc.test`) | Agree but for the conventions of amber.test |
| A peptide in TIP4P-Ew, against GROMACS (Section 15) | Agree to 4 × 10⁻⁶ |
| Three OPC waters: the change of the total energy with the time step (`test/Driver/virtual-sites.test`) | Shrinks as its square, 6.8 × 10⁻⁴ at 0.1 fs and 1.7 × 10⁻⁴ at 0.05 fs |
| The same: the trace of the virial against $-dU/d\lambda$ under a uniform scaling of the atoms and the cell | Agree to 6 × 10⁻⁶; without the change of Section 19.2, 35.84 instead of 43.57 kcal/mol |
| The order of the positions, against that of the files (`test/Driver/amber-opc-spatial_order.test`) | The same states to 2 × 10⁻¹⁵ |

## 20. CMAP

A CMAP term [[MacKerell2004]](references.md#mackerell2004) is an energy of
two dihedrals of five atoms, $\phi$ of atoms 1 to 4 and $\psi$ of atoms 2 to 5, read
from a map: a grid of $n \times n$ energies at $\phi$ and $\psi$ from $-180°$ in steps of
$h = 360°/n$, with $\phi$ the slower index in both formats. ff19SB
[[Tian2020]](references.md#tian2020) has a map for each amino acid, of
$n = 24$; amber19sb.ff of GROMACS has the same maps in kJ/mol.

### 20.1 The patches

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

### 20.2 What the readers take

| Item | Amber | GROMACS |
|---|---|---|
| Maps | `CMAP_COUNT`, `CMAP_RESOLUTION`, `CMAP_PARAMETER_nn` | `[ cmaptypes ]`: five bonded types, function 1, n, n, and the grid |
| Terms | `CMAP_INDEX`: five atoms numbered from 1 (not times 3) and the map | `[ cmap ]`: five atoms and function 1 |
| Which map | The one of the file | The first `[ cmaptypes ]` whose five types are those of the atoms in their order, not reversed. A type `T-R` is the bonded type T in the residue R, or in any residue for `T-*`, as amber19sb.ff writes them; this is what grompp of GROMACS 2026.3 was seen to do. |
| Grids | Only n = 24: sander takes the derivatives right only for 24 points | n even, the same for every map |

CMAP adds no exclusions and no pairs three bonds apart.

### 20.3 The angle near 0° and ±180°

sander computes a dihedral as the arccosine of its cosine, with the sign of
its sine; near 0° and ±180° the rounding of the cosine becomes an error of
the angle of about $10^{-12}\,\text{deg}^2/\delta$ at $\delta$ degrees from them, up to 1.5 × 10⁻⁶
degrees. MDIR computes the dihedral as GROMACS does, accurately there, so
its CMAP energies may differ from sander's by up to about 3 × 10⁻⁷
kcal/mol within a few thousandths of a degree of 0° or ±180°, and by less
than 10⁻¹² beyond 0.2°.

## 21. Minimization

`[minimize]` in place of `[dynamics]` lowers the potential energy by
steepest descent (D73):

| Item | Rule |
|---|---|
| Start | The positions of the file taken onto the surface of the constraints, SETTLE (M-SHAKE below double precision) and SHAKE with the positions themselves as the reference, and the sites placed again, before the first evaluation (D120); the energy at the start in the log and the terms are of these positions |
| Direction | $\mathbf g = P(\mathbf F / m)$: the force over the mass, $0$ for virtual sites, with $P$ the projection that RATTLE applies to velocities, which takes off the parts along the bonds of SETTLE and SHAKE at the current positions |
| Step | $\mathbf x' = \mathbf x + h\, \mathbf g / \lVert\mathbf g\rVert_{16}$, with $\lVert\mathbf g\rVert_{16} = r \big(\sum_i (\lVert\mathbf g_i\rVert / r)^{16}\big)^{1/16}$ and $r$ the root mean square of $\lVert\mathbf g_i\rVert$; since $\lVert\mathbf g\rVert_{16} \ge \max_i \lVert\mathbf g_i\rVert$, no particle moves farther than $h$. Then SETTLE and SHAKE take the groups back to their shapes from $\mathbf x$, and the sites are placed |
| Acceptance | The step is taken if the energy at $\mathbf x'$ is lower: $h$ grows by 1.2, to at most 1 Å; otherwise $\mathbf x$ stays and $h$ shrinks by 0.2. The choice is made particle by particle (a map that selects), so that the fields keep storage of their own |
| Loops | Over the intervals between frames (one if there are none), over the intervals between energies in each, and over the steps; one `dyn.step @descend` per step, which evaluates the energy and the forces once |
| Log | The potential energy with the constant terms, the root mean square and the largest of the forces $m\mathbf g$ without their parts along the constraints, in kcal/mol/Å, over the particles with mass, the particle of the largest, and $h$ in Å |
| Checkpoint | At the end, with `checkpoint` of `[output]`: the positions, velocities of 0, and the integrator `MIN`. A run that reads it, a minimization or a run of dynamics, takes the positions and the cell and begins anew at step 0 with drawn velocities; a minimization takes the positions and the cell of any checkpoint |
| Keywords | `method = "STEEPEST_DESCENT"`, `steps`, `energy_interval` (a divisor of `steps`), `trajectory_interval` (a multiple of it), `initial_step` in Å (0.1). No thermostat or barostat. A tolerance on the force is planned; in mixed precision the forces are rounded to about 10⁻⁵ of their size, which bounds how far a minimization can go |

On the target of D65 on the CPU in double precision, 500 steps take the
energy from −5348 to −7137 kcal/mol (from the positions of the file before
D120) and the root mean square of the
constrained forces from 8.7 to 2.1 kcal/mol/Å (`test/Driver/minimize.test`
takes 100). Without the masses in the direction, the steps that SHAKE and
SETTLE correct went uphill after 50 steps, and h fell to 0.

## 22. Restraints

`[[restraints]]`, any number of them, hold particles to their positions in
the file of coordinates (D74):

| Item | Rule |
|---|---|
| Energy | $k \lVert\mathbf x - \mathbf x^\text{ref}\rVert^2$ for each selected particle with mass, $k$ in kcal/mol/Å² (`force_constant`), as Amber's `restraint_wt`; the constants of restraints that select the same particle add |
| Reference | The positions $\mathbf r$ of the file of coordinates of `[input]`, also when the run begins from a checkpoint. Under a barostat (D124), $\mathbf x^\text{ref} = \mathbf s \odot \mathbf c + \mathbf o$, with $\mathbf s$ the edges of the cell over those of the file, axis by axis, $\mathbf c$ the center, which follows the cell, and $\mathbf o = \mathbf r - \mathbf c$ the offset, which stays. With `reference_scaling = "CENTER"`, the default, $\mathbf c$ is the mean of the references of every particle so restrained, so that they keep their shape, as a restrained solute should; with `"ALL"`, $\mathbf c = \mathbf r$ and $\mathbf o = 0$, each reference following the cell as the positions do, as restraints spread through the cell need. A particle that two restraints select takes one scaling. The fields hold $\mathbf o$ and $\mathbf c$ (`rest_x`, …, `rest_cx`, …); without a barostat they hold $\mathbf r$ |
| Forces, energy, virial | After the evaluation of the potential and the spreading of the forces of virtual sites, in every step, at the start, and in a minimization: $\mathbf F \mathrel{-}= 2k\mathbf d$ with $\mathbf d = \mathbf x - \mathbf x^\text{ref}$, $U \mathrel{+}= \sum k \lVert\mathbf d\rVert^2$, and $\mathsf W \mathrel{+}= \operatorname{diag}\big(\sum -2k\, \mathbf d \odot (\mathbf d + \mathbf o)\big)$, the derivative of the energy when the positions and the centers scale with the cell and the offsets do not; the off-diagonal elements are left out. Without a barostat $\mathbf o$ is taken as 0, and the trace, $-2U$, is exact. The log lists the energy of the restraints among the terms |
| Selection | `selection`, a mask of Amber in part: `:` residues by numbers (from 1) or names, `@` atoms by numbers or names, `:res@atoms`, `*`, with `!`, `&`, `|` and parentheses, in that order of precedence, and `*` and `?` in names. `!:WAT & !@H*` is every heavy atom but those of the waters. A mask that selects no particle with mass is an error; a run from `coordinates` has no names to select by |

On the target of D65, a run at constant energy with the heavy atoms of the
peptide restrained at 10 kcal/mol/Å² keeps the total energy to 4 × 10⁻⁴
over 2 ps at 1 fs, as the same run without them does; at constant
pressure the conserved energy drifts as it does without them, by the term
of Section 11.4.

Scaled about the origin with the cell, as before D124, the references of a
restrained solute shrink with it. Ubiquitin in OPC from the end of stage 3
of its tutorial, whose cell had become 0.926 of the file's in each edge,
had its restrained heavy atoms at a radius of gyration of 11.33 Å, against
11.73 Å for the file: the restraints pushed outward, and 100 ps more at
1 bar held the cell at 200,900 Å³, where a run without restraints went to
196,300 Å³ (the cell is at about −550 bar at 201,264 Å³). With the centers,
the restrained run averages 196,262 ± 580 Å³ over its second half, the free
one 196,293 ± 658: the same within the noise of runs still relaxing. On the
peptide of D65 in double precision, compressed by 16% in 2 ps at
$\tau_P$ = 0.5 ps with its heavy atoms at 100 kcal/mol/Å², the conserved
energy of the Trotter count changes by −10.4 kcal/mol, −8.8 without the
restraints, and −20.2 with the virial of the centers but without
$\mathbf o$; counted exactly, −3.1 and −2.4. On the POPC bilayer of the
tutorials, whose phosphorus atoms are restrained, `"CENTER"` holds the area
of the file over stage 3: 66.4 Å² a lipid over its second half, against
61.8 with `"ALL"`, and the barostat squeezes the height instead, the
pressure of x and y less that of z at 74 ± 11 bar against −1 ± 5.
Restraints spread through the cell take `"ALL"`.


## 23. Leapfrog

Leapfrog does what velocity Verlet does: virtual sites, SETTLE, SHAKE and
RATTLE, restraints, the thermostat, and the barostat (D76). The two store
different velocities and take the same steps. With `h = dt/2`, positions
`x_n`, forces `f_n = F(x_n)`, and `P_x` the projection that takes off the
velocities along the constraints at `x` (the second half of RATTLE):

| | Velocity Verlet | Leapfrog |
|---|---|---|
| Stored velocities | `v_n`, of the time of `x_n` | `v_{n−½}`, half a step behind (`velocity_offset = −0.5`) |
| Carried state | `x_n, v_n, f_n` | `x_n, v_{n−½}, f_n` |
| A step | `u = v_n + h f_n/m`; `x' = x_n + dt u`; constrain `x'` to `x_{n+1}`; `v_{n+½} = u + (x_{n+1} − x')/dt`; `f_{n+1}`; `v_{n+1} = P(v_{n+½} + h f_{n+1}/m)` | `u = v_{n−½} + dt f_n/m`; `x' = x_n + dt u`; constrain `x'` to `x_{n+1}`; `v_{n+½} = u + (x_{n+1} − x')/dt`; `f_{n+1}` |
| A step with energies | The same, with the energy and the virial of `x_{n+1}` | `u = P(v_{n−½} + h f_n/m) + h f_n/m`, then as velocity Verlet, returning `v_{n+½}` to store and `v_{n+1} = P(v_{n+½} + h f_{n+1}/m)` for the energies |
| Start | `v_0` drawn or read | `v_{−½} = v_0 − h f_0/m` |

Why the steps are the same: without constraints, `v_{n−½} + dt f_n/m` is
velocity Verlet's `v_n + h f_n/m` when `v_n = v_{n−½} + h f_n/m`, which is
what leapfrog's `v_{n−½}` and velocity Verlet's `v_n` are to each other.
With constraints, velocity Verlet's `v_n` is `P(v_{n−½} + h f_n/m)`; the
two drifts differ by `dt (1 − P)(v_{n−½} + h f_n/m)`, which lies along the
directions, weighted by the inverse masses, in which SHAKE and SETTLE move
the particles back. Both are brought to the same point on the surface of
the constraints, to the tolerance of the solvers; the velocities that
follow, `(x_{n+1} − x_n)/dt`, are then the same too. The logs of the two
agree row by row (`test/Driver/leapfrog.test`, and
`test/Driver/leapfrog-constraints.test` with the peptide in OPC water).

What each quantity is taken from:

| Quantity | Velocity Verlet | Leapfrog |
|---|---|---|
| Energy, virial, pressure in the log | $\mathbf x_{n+1}$ and $\mathbf v_{n+1}$ of the step with energies | The same $\mathbf x_{n+1}$ and $\mathbf v_{n+1}$ |
| Virial of the constraints | $\tfrac12$ of the impulses of the positions ($\mathbf G = 2m\boldsymbol\Delta/\Delta t^2$ for $\mathbf x' = \mathbf x_n + \Delta t\, \mathbf v_n + \Delta t^2 \mathbf f_n/2m$) and $\tfrac12$ of those of the velocities (Section 9) | The same: the step with energies drifts from $P(\mathbf v_{n-1/2} + h \mathbf f_n/m)$, the form above, so $\boldsymbol\Delta$ has the same meaning. The plain step, whose drift is a whole kick, computes no virial |
| Thermostat alone | Scales $\mathbf v_n$, with its kinetic energy | Scales $\mathbf v_{n-1/2}$, with its kinetic energy, as GROMACS does with leapfrog |
| Barostat (and the thermostat with it) | Pressure from $\mathbf v_{n+1}$; scales $\mathbf x_{n+1}$ by $\mu$ and $\mathbf v_{n+1}$ by $\alpha/\mu$; carries $\mathbf f' = \mathbf F(\mathbf x'_{n+1})$ (D77) | Pressure from $\mathbf v_{n+1}$; scales $\mathbf x_{n+1}$ by $\mu$ and $\mathbf v_{n+1}$ by $\alpha/\mu$; carries $\mathbf f'$ and stores $\mathbf v'_{n+1/2} = \mathbf v'_{n+1} - h \mathbf f'/m$, so that the next step is that of velocity Verlet |

The barostat couples the velocities of the time of the positions: scaling
the stored $\mathbf v_{n+1/2}$ by $1/\mu$ would leave the half kick $h \mathbf f_{n+1}/m$ in
$\mathbf v_{n+1}$ unscaled, and the kinetic energy of the log would change by
$(1/\mu - 1)\, h \sum \mathbf v_{n+1/2} \cdot \mathbf f_{n+1}$ more than the work that the barostat
counts. That term follows $dU/dt$; while the barostat compresses the target
of D65 by a fifth over 10 ps, the conserved energy drifted by 4.3 × 10⁻³
of its value with it, and by 1.3 × 10⁻³ without it, as with velocity
Verlet (1.4 × 10⁻³), with the work counted to first order. With the exact
work (D77) both integrators carry the forces of the scaled positions, and
without constraints the trajectory of leapfrog at constant pressure is that
of velocity Verlet to the bit (`test/Driver/barostat.test`).

The stored velocities are half a kick behind the ones the coupling leaves
without the motion of the center of mass. The forces of PME do not sum to
zero (their interpolation does not conserve momentum), so the momentum of
the stored velocities is $-h \sum \mathbf f$, 10⁻² amu nm/ps on the target of D65,
where that of velocity Verlet is at the rounding.

A checkpoint of leapfrog holds the forces, as one of velocity Verlet does,
and a run continues it exactly. A checkpoint of leapfrog written before
this holds none and is refused with a message.
