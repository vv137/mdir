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
| Force field | ff14SB [[Maier2015]](references.md#maier2015) for the protein, TIP3P [[Jorgensen1983]](references.md#jorgensen1983) for water, the ions that go with them |
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
| Virtual sites | M2a | Water models with four sites, such as TIP4P [[Jorgensen1983]](references.md#jorgensen1983) and OPC [[Izadi2014]](references.md#izadi2014), cannot be run |
| Martini and other coarse-grained force fields | Later (D53) | |
| Particle mesh Ewald on more than one device or process | M2c and later | |
| The repartitioning of the mass of hydrogen | Later | The time step stays at 2 fs |
| CMAP [[MacKerell2004]](references.md#mackerell2004), polarizable force fields, 12-6-4 terms [[Li2014]](references.md#li2014) | Later | The readers reject topologies that have them |
| Cells that are not orthorhombic | Later | The truncated octahedron of Amber cannot be run |
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
| `distance(a, b)` | `|d_ab|` | `f64` |
| `displacement(a, b)` | `d_ab = x_a − x_b`, in the minimum image [[AllenTildesley2017]](references.md#allentildesley2017) | `vector<3xf64>` |
| `angle(a, b, c)` | The angle at `b` between `d_ab` and `d_cb`, from 0 to π | `f64` |
| `cosine(a, b, c)` | The cosine of that angle | `f64` |
| `dihedral(a, b, c, d)` | The angle between the planes of `a, b, c` and of `b, c, d`, from −π to π | `f64` |

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

```text
F_m = − Σ_q (∂u/∂q) (∂q/∂x_m)          for every member m of the tuple
W   = Σ_q Σ_m d_m ⊗ F_qm               with the displacement of m from one
                                       member of the coordinate q
```

The virial takes the form of Thompson, Plimpton, and Mattson
[[Thompson2009]](references.md#thompson2009) for terms of more than two particles.

The forces of one coordinate add up to zero, so that the member that the
displacements are taken from does not matter.

| Coordinate | `∂q/∂x_m` |
|---|---|
| `distance(a, b)` | `± d_ab / r` for `a` and `b` |
| `cosine(a, b, c)` | For `a`: `(d_cb / |d_cb| − c · d_ab / |d_ab|) / |d_ab|`; for `c` likewise; for `b` the negative of their sum |
| `angle(a, b, c)` | That of the cosine, times `−1 / sin θ` |
| `dihedral(a, b, c, d)` | The form of Blondel and Karplus [[Blondel1996]](references.md#blondel1996), which has no singularity where three particles are in line |

The derivative of `angle` is singular where the three particles are in
line. A term that is smooth there, such as the one above, should take
`cosine`.

`md-differentiate` produces, for a sum over tuples of arity `k`, an
`md.gather_tuples` whose kernel yields `k` forces, one for each member:

```text
a_i = Σ_{t, s : t[s] = i} k(t)[s]
```

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

```text
N(x, h) = { {i, j} : i ≠ j, r_ij < r_c } \ E
```

| Item | Proposal |
|---|---|
| The relation | `E` is a relation of arity 2 of a tuple set, as the bonds are. The reader derives it: pairs one, two, and three bonds apart (`nrexcl = 3`), and those that the topology lists. |
| Where a pair is excluded | In the build of the neighbor structure. The search skips the particles in the row of exclusions of a particle. A loop over pairs tests nothing. |
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
%a = mdrt.from_buffer %a_buffer
       : memref<?x?xf64> to !md.table<2, f64, symmetric>

%u = md.sum_relation %n, %x, %cell gather(%type : !kinds) ... {
^bb0(%r: f64, %d: vector<3xf64>, %t_i: i32, %t_j: i32):
  %a_ij = md.lookup %a[%t_i, %t_j] : f64
  ...
}
```

| Item | Proposal |
|---|---|
| The type | `!md.table<rank, E>`, with `symmetric` for a table of rank 2 that is |
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

```text
E_disp = − (2π N² / 3V) ⟨C6⟩ / r_c³        P_disp = 2 E_disp / V
```

with `⟨C6⟩` the mean of the coefficient over the pairs of particles,
which the front end computes from the table and the counts of the types.
It depends on the volume only, so it is a number on the host that
changes with the cell. It is on by default, as in Amber.

## 8. Electrostatics

### 8.1 A cutoff

For the intermediate stage (Section 1.4). The form is the one that both
reference engines can produce for a periodic system; which that is is
settled from the specifications of their formats before the stage
begins.

### 8.2 Particle mesh Ewald

Decided: particle mesh Ewald on one device or one process, in M1 (D55).
Designed in its own document when its stage begins. The parts:

| Part | Where |
|---|---|
| The direct sum with `erfc(β r) / r` | A term over pairs, as any other |
| The excluded pairs | A term over the tuples of `E` that takes `erf(β r) / r` out again (Section 6.1) |
| The self term | A number, from the charges |
| The reciprocal sum | Spreading the charges to a grid with B-splines of order 4 [[Essmann1995]](references.md#essmann1995), a forward FFT, a product with the influence function, an inverse FFT, and the forces from the grid |
| The virial | From the reciprocal energy of each wave vector, and from the direct sum as for pairs |

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
| The bonds of hydrogen | SHAKE [[Ryckaert1977]](references.md#ryckaert1977), with RATTLE [[Andersen1983]](references.md#andersen1983) for the velocities under velocity Verlet |
| Execution | The constrained bonds fall into clusters with no particle in common: a heavy atom with its hydrogens, a water. A thread takes a cluster and writes to its particles only, so no two threads write to one particle. A loop over clusters is a new kind of loop. |

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

```text
v_i ← v_i − (Σ_j m_j v_j) / (Σ_j m_j)       every `comm_period` steps
```

| Item | Proposal |
|---|---|
| What is removed | The linear momentum. Under periodic boundaries the angular momentum is not conserved and is left alone. |
| How | A `particle_for` with a sum of `vector<3xf64>`, as the virial is summed, and a `particle_for` that subtracts, which fuses with the kick |
| How often | `comm_period` in `[dynamics]`, 100 steps by default |
| Degrees of freedom | Three fewer, as in M0 |

## 11. Thermostat and barostat

### 11.1 Kinds

| | Proposal | Reason |
|---|---|---|
| Thermostat | Stochastic velocity rescaling (Bussi, Donadio, and Parrinello 2007 [[Bussi2007]](references.md#bussi2007)) | It samples the canonical distribution, and it takes the kinetic energy and a few random numbers for each step, not one for each particle. |
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
| The kinetic energy of the thermostat | `K_T`, the mean of the three times (D45) |
| The kinetic energy of the barostat | `K_P`, the mean of the two half steps (D45) |
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
| Random numbers in kernels | Not in M1. They need the generator as a template in IR, as the neighbor build is one. |

## 12. A cell that changes

A neighbor structure is valid in the cell that it was built in (Section
8.2 of ops-m0.md). The barostat changes the cell.

| | A. The structure is built after every change of the cell | B. The structure follows the cell |
|---|---|---|
| How | Nothing new: the refresh finds another cell and builds | The barostat scales the positions that the structure was built at, and the reach of the structure shrinks with the smallest factor. The structure is valid while no particle has moved more than half of what is left of the skin. |
| Builds | At least one for each step of the barostat | As without a barostat |
| With a barostat every 10 steps and a build every 10 to 20 steps | Up to twice the builds | |

Proposal: A for M1, and B when the builds are measured to cost.

The cell is a value of the state that the loops carry (S1). The order of
the particles (D44) and the incidence structures do not depend on it. The
grid of particle mesh Ewald keeps its number of points and the correction
for the dispersion follows the volume.

## 13. Input

### 13.1 Formats

| Format | Files | Keywords of `[input]` |
|---|---|---|
| Amber | `prmtop` (topology), `inpcrd` or `rst7` (positions, velocities, and the cell) | `prmtopfile`, `ambcrdfile` |
| GROMACS | `.top` with `.itp` (topology), `.gro` (positions, velocities, and the cell) | `grotopfile`, `grocrdfile` |

| Item | Proposal |
|---|---|
| The readers | MDIR has readers of its own. They accept what the engines accept and share no code with them. The engines and their tools are under the GPL and the LGPL; their behavior is learned from their documentation and their code, and written down as a specification, from which the readers are written. |
| What a reader hands on | A description of the system that does not depend on the format: particles, types, tuple sets with their parameters, the table of pairs of types, exclusions, pairs three bonds apart, constraints, the cell. A format is a reader more. |
| Units | A reader converts to the units inside MDIR: nm, kJ/mol, ps. Amber files are in Å and kcal/mol, and the charges of `prmtop` carry a factor; the specification of the format gives the constants. The control file stays in Å and kcal/mol (D36). |
| What is an error | What M1 cannot run: CMAP, 10-12 terms, 12-6-4 terms, polarizability, virtual sites and extra points, cells that are not orthorhombic. Each with the milestone that brings it. |
| Formats of other engines | Planned |

The GROMACS preprocessor and sections are as in the specification of the
format; M1 reads the sections that an Amber force field in the format of
GROMACS needs: `defaults`, `atomtypes`, `nonbond_params`, `pairtypes`,
`bondtypes`, `angletypes`, `dihedraltypes`, `moleculetype`, `atoms`,
`bonds`, `pairs`, `angles`, `dihedrals`, `exclusions`, `settles`,
`system`, `molecules`.

### 13.2 Keywords of the control file

| Keywords | Table |
|---|---|
| `electrostatic = "CUTOFF"` or `"PME"`, `pme_ngrid_x`, `pme_ngrid_y`, `pme_ngrid_z` or `pme_spacing`, `pme_order`, `ewald_tolerance`, `dispersion_correction` | `[energy]` |
| `rigid_bond`, `fast_water`, `shake_tolerance`, `shake_iterations` | `[constraints]` |
| `ensemble = "NVT"` or `"NPT"`, `thermostat = "BUSSI"`, `barostat = "BERNETTI-BUSSI"`, `temperature`, `pressure`, `tau_t`, `tau_p`, `compressibility`, `isotropy = "ISO"` or `"SEMI-ISO"` | `[ensemble]` |
| `thermostat_period`, `barostat_period`, `comm_period` | `[dynamics]` |

## 14. The command line

One program with subcommands (D59). The programs of M0 become
subcommands; `mdir-opt` stays a program for developers.

| Command | Does |
|---|---|
| `mdir run control.toml` | Reads the input, compiles the run, and runs it (`mdir-run` of M0) |
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
| The same with particles in line and with a dihedral of π | The same | 1e-10 |
| A neighbor structure with exclusions | A search over all pairs | Exact |
| Lookup in a table, pairs three bonds apart | A script | 1e-10 |
| Particle mesh Ewald | An Ewald sum [[Ewald1921]](references.md#ewald1921) with many wave vectors, in a script | That of the parameters |
| SETTLE and SHAKE | A script; the constrained distances | That of the tolerance |
| The factor of the thermostat and of the barostat | A script with the same generator | 1e-12 |
| A run that continues from a checkpoint, with thermostat and barostat | The run that was not interrupted | Exact |
| Alanine dipeptide in water: the energy of each term at the start, and the forces | AmberTools and GROMACS, each from its own format | 1e-5, relative |
| The same system: the conservation of energy at constant energy, and temperature, pressure, and density at constant temperature and pressure | The engines, within the statistical error | |
| The JAC benchmark: run times | The engines on the same device | |

The two readers can be compared with each other when the same system is
in both formats. Whether a tool of AmberTools converts a topology of Amber
to the format of GROMACS, and with which options, is settled from the
specification of the format.

Neither engine is installed on the development machine. Both are built
into the home directory.

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
| 14 | The motion of the center of mass | Removed every `comm_period` steps (Section 10) | D60 |
| 15 | The command line | One program with subcommands (Section 14) | D59 |

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
| M1c | Exclusions in the neighbor build; pairs three bonds apart | Chains with Lennard-Jones | |
| M1d | Tables, NBFIX, the rule `product`, a Coulomb cutoff, the correction for the dispersion | A mixture of charged types | |
| M1e | The readers of both formats; renumbering of the members with the order | Alanine dipeptide in flexible water, at constant energy with 0.5 fs | |
| M1f | Comparison of the intermediate stage with AmberTools and GROMACS | | This completes the intermediate stage |
| M1g | Removal of the motion of the center of mass, random numbers, the thermostat | At constant temperature | |
| M1h | Particle mesh Ewald | With particle mesh Ewald | |
| M1i | Constraints: SETTLE, SHAKE, RATTLE | With 2 fs | |
| M1j | The barostat, a cell that changes | At constant temperature and pressure | |
| M1k | Comparison with AmberTools and GROMACS; run times of the JAC benchmark | | |
