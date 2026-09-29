# Design for Milestone M1

Status: decided (2026-09-29), not implemented. Section 12 has the
decisions; Section 13 has the order of work and its state.

M1 is a coarse-grained membrane in water with the Martini force field
(P3). This document proposes what MDIR needs for it: relations that come
from a topology, terms over them, exclusions, parameters of pairs of
types, a thermostat, a barostat, and the input.

It follows [architecture.md](architecture.md) and extends
[ops-m0.md](ops-m0.md). IR snippets show the proposed syntax.

## 1. Scope

### 1.1 The target

A lipid bilayer in water with ions, at constant temperature and pressure.

| Property | Value |
|---|---|
| Particles | 10⁴ to 10⁶ beads |
| Time step | 20 fs |
| Nonbonded terms | Lennard-Jones with parameters for each pair of types, shifted to zero at a cutoff of 1.1 nm; Coulomb with a reaction field, relative permittivity 15 |
| Bonded terms | Bonds, angles, and dihedrals |
| Exclusions | Particles that a bond joins |
| Ensemble | Constant temperature and pressure; the pressure in the plane of the membrane apart from that along its normal |

### 1.2 What M1 adds

| Item | Section |
|---|---|
| Relations of a topology, with parameters for each tuple | 2 |
| Kernels in internal coordinates: distance, angle, dihedral | 3 |
| Differentiation of such kernels | 4 |
| Execution of terms that contribute to several particles | 5 |
| Exclusions | 6 |
| Tables of parameters | 7 |
| Thermostat, barostat, and random numbers | 8 |
| A cell that changes | 9 |
| Input of a topology | 10 |

### 1.3 What M1 leaves out

| Item | Milestone | Consequence for M1 |
|---|---|---|
| Constraints and virtual sites | M2a | Molecules that have them cannot be run: cholesterol, polarizable water, proteins with constrained bonds |
| Mesh electrostatics | M2b | |
| More than one device, more than one process | M2c and later | |
| Random numbers for each particle, as Langevin dynamics needs them | Open | The thermostat of M1 takes a few random numbers for each step, not one for each particle |
| Scaled pairs of particles three bonds apart | M2a | Martini has none |

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
| The members of the tuples | `!md.relation<@atoms, 3, reversal>` | For each tuple, the numbers of its particles |
| A parameter of the tuples | `!md.field<@angles, f64>` | One value for each tuple |

Both enter the program from buffers, as the fields of the particles do
(D32):

```mlir
%angles = mdrt.from_buffer %members
            : memref<?x3xi32> to !md.relation<@atoms, 3, reversal>
%k      = mdrt.from_buffer %k_buffer
            : memref<?xf64> to !md.field<@angles, f64>
```

The members are the numbers of the particles, which are their places in
the input (P17, D44), not their places in memory.

### 2.2 Orientation

An orientation is the group of permutations under which two tuples are
the same tuple (Section 2.2 of ops-m0.md).

| Orientation | Arity | The same tuple | Example |
|---|---|---|---|
| `unordered` | 2 | `{i, j}` and `{j, i}` | Bonds, exclusions |
| `ordered` | Any | No other | Position restraints, with arity 1 |
| `reversal` | 3, 4 | `(i, j, k)` and `(k, j, i)` | Angles, dihedrals |

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
coordinates that the op names:

```mlir
%u = md.sum_relation %angles, %x, %cell
       coordinates(cosine(0, 1, 2))
       tuple(%k, %c0 : !md.field<@angles, f64>, !md.field<@angles, f64>) {
^bb0(%c: f64, %k_t: f64, %c0_t: f64):
  %half = arith.constant 0.5 : f64
  %dc   = arith.subf %c, %c0_t : f64
  %sq   = arith.mulf %dc, %dc : f64
  %hk   = arith.mulf %half, %k_t : f64
  %e    = arith.mulf %hk, %sq : f64
  md.yield %e : f64
} : !md.relation<@atoms, 3, reversal>, !vec -> f64
```

| Coordinate | Value | Kernel argument |
|---|---|---|
| `distance(a, b)` | `|d_ab|` | `f64` |
| `displacement(a, b)` | `d_ab = x_a − x_b`, in the minimum image | `vector<3xf64>` |
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

```text
F_m = − Σ_q (∂u/∂q) (∂q/∂x_m)          for every member m of the tuple
W   = Σ_m d_m0 ⊗ F_m                   with the displacement of m from member 0
```

| Coordinate | `∂q/∂x_m` |
|---|---|
| `distance(a, b)` | `± d_ab / r` for `a` and `b` |
| `cosine(a, b, c)` | For `a`: `(d_cb / |d_cb| − c · d_ab / |d_ab|) / |d_ab|`; for `c` likewise; for `b` the negative of their sum |
| `angle(a, b, c)` | That of the cosine, times `−1 / sin θ` |
| `dihedral(a, b, c, d)` | The form of Blondel and Karplus, which has no singularity where three particles are in line |

The derivative of `angle` is singular where the three particles are in
line. A term that is smooth there, such as the one above, should take
`cosine`.

`md-differentiate` produces, for a sum over a relation of arity `k`, a
gather whose kernel yields `k` forces, one for each member:

```text
a_i = Σ_{t, s : t[s] = i} k(t)[s]
```

## 5. Execution of terms over tuples

### 5.1 The strategies

A term over a tuple contributes to `k` particles. P13 left the strategy
open.

| Strategy | Evaluations of a tuple | Writes | Order of a sum |
|---|---|---|---|
| A. Every particle visits the tuples that it is a member of and takes its own part | `k` | To the own particle only | Fixed |
| B. Every tuple is evaluated once and adds to its members with atomic additions | 1 | To `k` particles | That of the threads |
| C. Every tuple is evaluated once; tuples with a particle in common are in different groups, which run one after another | 1 | To `k` particles | Fixed |

Proposal: A, as for pairs (P13).

| Reason | |
|---|---|
| It is the strategy of the pairs. | A thread writes to its own particle. No atomic addition, and the deterministic level holds. |
| Bonded terms are a small part of the work. | A lipid of 12 beads has 11 bonds and fewer angles, and each bead some 70 neighbors. |
| A loop over tuples of this kind is a loop over particles. | It can be fused with the loop over pairs and with the kick that follows. |

B and C are choices for the planner later.

### 5.2 The structure

```mlir
%incidence = md_exec.build_incidence %angles, %ids width(8)
               : ... -> !mdrt.incidence<@atoms>

%f1 = md_exec.tuple_for %incidence, %x, %cell
        coordinates(cosine(0, 1, 2))
        tuple(%k, %c0 : ...) outs(%f0 : !vec) reduce(%u0 : f64)
        weights [0.333333] { ... }
```

| Item | Proposal |
|---|---|
| What the structure holds | For each particle a row: the tuples that it is a member of, with its place in each and the places in memory of the other members |
| When it is built | Where the run begins and where the particles are put in a new order (D44): the places in memory of the members change there |
| The kernel | Yields the forces on all members. The loop takes the one of the particle. A kernel for each place, which computes that force only, is a later optimization. |
| A sum | Every tuple contributes `k` times, with the weight `1/k` |

## 6. Exclusions

```mlir
%n = md.neighborhood %x, %cell cutoff(1.1) exclude(%excluded)
       : !vec -> !pairs
```

```text
N(x, h) = { {i, j} : i ≠ j, r_ij < r_c } \ E
```

| Item | Proposal |
|---|---|
| The relation | `E` is a relation of arity 2 of a tuple set, as the bonds are. The front end derives it from the bonds. |
| Where a pair is excluded | In the build of the neighbor structure. The search skips the particles in the row of exclusions of a particle. A loop over pairs tests nothing. |
| The row of exclusions | An incidence structure of `E` (Section 5.2). It is built with the others. |
| A term over the excluded pairs | `md.sum_relation` over `E`, with `coordinates(distance(0, 1))`. The reaction field has one: a pair that is excluded still feels the field of the medium. |

## 7. Tables of parameters

Martini has the Lennard-Jones parameters for each pair of types. They do
not follow from parameters of the types by a rule.

```mlir
%c6 = mdrt.from_buffer %c6_buffer
        : memref<?x?xf64> to !md.table<2, f64, symmetric>

%u = md.sum_relation %n, %x, %cell gather(%type : !kinds) ... {
^bb0(%r: f64, %d: vector<3xf64>, %t_i: i32, %t_j: i32):
  %a = md.lookup %c6[%t_i, %t_j] : f64
  ...
}
```

| Item | Proposal |
|---|---|
| The type | `!md.table<rank, E>`, with `symmetric` for a table of rank 2 that is |
| Exchange | A lookup in a symmetric table with the two types of a pair is symmetric |
| Differentiation | A lookup does not depend on the positions |
| Storage | A buffer, on the device where the loops are |
| In a control file | `mixing = "table"` for a pair term, and the parameters of the pairs of types from the topology |

Charges need the product of the values of the two particles, which the
mixing rules of M0 do not have. The proposal is the rule `product`.

## 8. Thermostat and barostat

### 8.1 Kinds

| | Proposal | Reason |
|---|---|---|
| Thermostat | Stochastic velocity rescaling (Bussi, Donadio, and Parrinello 2007) | It samples the canonical distribution, and it takes the kinetic energy and a few random numbers for each step, not one for each particle. |
| Barostat | Stochastic cell rescaling (Bernetti and Bussi 2020), with the pressure in the plane apart from that along the normal | It samples the distribution at constant pressure and is of first order: it has no momentum of the cell to store. |

Both carry the statement that they preserve the target distribution
(P10).

Two barostats go by the name of Bussi:

| | Bussi, Zykova-Timan, and Parrinello 2009 | Bernetti and Bussi 2020 |
|---|---|---|
| Kind | Of second order: the cell has a momentum, and the thermostat acts on it as well | Of first order: the barostat of Berendsen with a term of noise |
| State beside the cell | The momentum of the cell | None |
| Parameters | A time | A time and a compressibility. A wrong compressibility changes how fast the volume relaxes, not what is sampled. |
| The volume | May oscillate | Relaxes |

MDIR takes the one of 2020 (D50). A checkpoint needs nothing for it.

### 8.2 In `dyn`

```mlir
dyn.program @step(...) attributes {
    requires = ["temperature", "pressure"],
    provides = ["thermostatting", "barostatting"]} {
  ...
  %alpha = dyn.velocity_rescaling %kinetic, %freedom, %step
             temperature(323.0) tau(1.0) dt(%span) stream(0)
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
| How often | Every `n` steps, with `n` times the time step. A step between computes no global sum. On a device a step with global sums takes up to twice the time of one without (Section 10.8 of ops-m0.md). |
| Groups with a thermostat each | One group in M1 |

### 8.3 Random numbers

| Item | Proposal |
|---|---|
| The generator | Philox 4×32 with 10 rounds, from the key of A13 |
| Where it runs | On the host, in `libmdrt`: the thermostat and the barostat take numbers for the system, not for a particle |
| Streams | 0 for the thermostat, 1 for the barostat. The draw index counts the numbers of one step. |
| Entity | A fixed key, as for a global move (A5) |
| Random numbers in kernels | Not in M1. They need the generator as a template in IR, as the neighbor build is one. |

## 9. A cell that changes

A neighbor structure is valid in the cell that it was built in (Section
8.2 of ops-m0.md). The barostat changes the cell.

| | A. The structure is built after every change of the cell | B. The structure follows the cell |
|---|---|---|
| How | Nothing new: the refresh finds another cell and builds | The barostat scales the positions that the structure was built at, and the reach of the structure shrinks with the smallest factor. The structure is valid while no particle has moved more than half of what is left of the skin. |
| Builds | At least one for each step of the barostat | As without a barostat |
| With a barostat every 10 steps and a build every 10 to 20 steps | Up to twice the builds | |

Proposal: A for M1, and B when the builds are measured to cost.

The cell is a value of the state that the loops carry (S1). The order of
the particles (D44) and the incidence structures do not depend on it.

## 10. Input

| Item | Proposal |
|---|---|
| Topology | The format of GROMACS: `.top` and `.itp` |
| Positions | `.gro`, beside PDB |
| Keywords of `[input]` | `grotopfile`, `grocrdfile` |
| Directives of the preprocessor | `#include`, `#define`, `#ifdef`, `#ifndef`, `#else`, `#endif` |
| Sections | `defaults`, `atomtypes`, `nonbond_params`, `moleculetype`, `atoms`, `bonds`, `angles`, `dihedrals`, `exclusions`, `system`, `molecules` |
| Functions | Bonds 1; angles 1, 2, and 10; dihedrals 1, 2, and 9 |
| What is an error | `constraints`, `settles`, `virtual_sites`, `pairs`, and a function that is not listed, each with the milestone that brings it |
| Units | The files are in nm, kJ/mol, and ps, which are the units inside MDIR. The control file stays in Å and kcal/mol (D36). |
| The readers | MDIR has readers of its own for both formats. They accept what GROMACS accepts and share no code with it. |
| Formats of other engines | Planned. A reader hands the builder a description of the system that does not depend on the format, so a format is a reader more. |

| Keywords of the control file | Table |
|---|---|
| `electrostatic = "CUTOFF"`, `dielec_const`, `epsilon_rf` | `[energy]` |
| `ensemble = "NVT"` or `"NPT"`, `thermostat = "BUSSI"`, `barostat = "BERNETTI-BUSSI"`, `temperature`, `pressure`, `tau_t`, `tau_p`, `compressibility`, `isotropy = "SEMI-ISO"` | `[ensemble]` |
| `thermostat_period`, `barostat_period` | `[dynamics]` |

## 11. Validation

| Test | Compared with | Tolerance |
|---|---|---|
| Energy, forces, and virial of each kind of bonded term, for chains of a few particles | A script that evaluates the definitions | 1e-10 |
| The same with particles in line and with a dihedral of π | The same | 1e-10 |
| A neighbor structure with exclusions | A search over all pairs | Exact |
| Lookup in a table, reaction field | A script | 1e-10 |
| The factor of the thermostat and of the barostat | A script with the same generator | 1e-12 |
| A run that continues from a checkpoint, with thermostat and barostat | The run that was not interrupted | Exact |
| A bilayer: the energy of each term at the start | GROMACS | 1e-5, relative |
| A bilayer: temperature, pressure, area for each lipid, thickness, after 100 ns | GROMACS, within the statistical error | |

GROMACS is not installed on the development machine.

## 12. Decisions

| # | Question | Decision | Recorded as |
|---|---|---|---|
| 1 | The force field and the lipid of the target | Martini 2.2 with DPPC | D46 |
| 2 | Kernels in internal coordinates or over positions | Internal coordinates (Section 3.2) | D48 |
| 3 | Parameters for each tuple or for each kind | For each tuple (Section 2.3) | D47 |
| 4 | The strategy for terms over tuples | Every particle takes its own part (Section 5.1) | D49 |
| 5 | The thermostat and the barostat | Stochastic velocity rescaling (2007) and stochastic cell rescaling (2020) (Section 8.1) | D50 |
| 6 | A cell that changes | A build after every change, in M1 (Section 9) | D51 |
| 7 | The format of the topology | That of GROMACS (Section 10) | D52 |
| 8 | The engine to compare with | GROMACS, built into the home directory | D52 |
| 9 | Groups with a thermostat each | One group in M1 | D50 |
| 10 | The correction for the dispersion beyond the cutoff | Not in M1 | D46 |

## 13. Order of work

| Stage | Work | Runs |
|---|---|---|
| M1a | Tuple sets, internal coordinates, differentiation, loops over tuples on the CPU and on a GPU | Chains of particles with bonds, angles, and dihedrals, at constant energy |
| M1b | Exclusions in the neighbor build; terms over excluded pairs | The same with Lennard-Jones |
| M1c | Tables, the rule `product`, the reaction field | A mixture of charged types |
| M1d | Input of a topology | A bilayer at constant energy |
| M1e | Random numbers, the thermostat | A bilayer at constant temperature |
| M1f | The barostat, a cell that changes | A bilayer at constant temperature and pressure |
| M1g | Comparison with GROMACS; run times | |
