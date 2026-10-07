# 3. Design

A run of MDIR passes through four levels of IR, each a dialect of MLIR
[[Lattner2021]](references.md#lattner2021). The upper two say *what* is computed and *how the state
advances*; the third says *over which set, with which pattern of access*;
the fourth is the interface to a small runtime. Every pass between them
either rewrites within a level or lowers to the next, and a pass may rely
only on what the IR states (Section 12).

*Table 3.1. The levels of the IR.*

| Dialect | Says | Objects |
|---|---|---|
| `md` | What is computed | Particle sets, tuple sets and their disjoint unions; relations that name the members of tuples; fields over a set; the cell; potentials as functions of positions, cell, and parameters; sums over relations and tuples; neighborhoods; the reciprocal sum of Ewald; `md.evaluate` with requests for energy, forces, virial, or a derivative with respect to a parameter |
| `dyn` | How the state advances | `dyn.kick`, `dyn.drift`, and `dyn.program`, a step whose state is the set of its SSA values, with the properties it provides (`symplectic`, `time_reversible`) and requires |
| `md_exec` | Over which set, with which access | Loops over particles, pairs, and tuples (`particle_for`, `pair_for`, `tuple_for`); neighbor structures and their tests; the order of the particles; tables; streams |
| `mdrt` | The interface to the runtime | Opaque types for cells, permutations, neighbor structures, and incidences; buffers in and out; calls to the host |

The four dialects have 50 operations; the passes over them number 22
(5 on `md`, 14 on `md_exec`, and 3 lowerings). Planned levels for
distributed runs (`md_dist`), learned potentials (`mlff`), and ensembles
of runs (`ensemble`) are designs, not code (Section 13).

## 3.1 What is computed, and how the state advances

A term of the potential is written once, as an expression of its
coordinate and its parameters, and the driver writes every term of a run
into one `md.potential`. The terms of a topology are written by the
driver; a control file adds terms over pairs, bonds, angles, dihedrals,
and tuples of any of their distances, angles, and dihedrals
as expressions in the syntax of the custom forces of OpenMM, with
parameters of each particle and tabulated functions of up to three
arguments (D22, D136 to D138, D165),
which compile into the same loops as the terms of the topology and whose
forces the same differentiation gives. Below is the potential of the test of a mixture
of two Lennard–Jones types (`test/Driver/mixture.toml`), as `mdir emit`
prints it, abridged: a sum over the unordered pairs of a neighborhood,
whose kernel receives the distance, the displacement, and the values of
the fields of both particles, and mixes the parameters by the
Lorentz–Berthelot rules.

```mlir
md.potential @energy(%x: !vec, %cell: !md.cell, %p_epsilon: !real, %p_sigma: !real) -> f64 {
  %n = md.neighborhood %x, %cell cutoff(9.0e-01) : !vec -> !pairs
  %u0 = md.sum_relation %n, %x, %cell gather(%p_epsilon, %p_sigma : !real, !real)
      exchange(symmetric) truncation(switch, from = 7.5e-01) {
  ^bb0(%r: f64, %d: vector<3xf64>, %epsilon_i: f64, %epsilon_j: f64,
       %sigma_i: f64, %sigma_j: f64):
    %product_epsilon = arith.mulf %epsilon_i, %epsilon_j : f64
    %epsilon = math.sqrt %product_epsilon : f64
    ...
    md.yield %u : f64
  } : !pairs, !vec -> f64
  md.return %u0 : f64
}
```

A step is a program of `dyn`. Velocity Verlet is

```mlir
dyn.program @step(%x: !vec, %v: !vec, %f: !vec, %m: !real, %cell: !md.cell,
    %dt: f64, %p_epsilon: !real, %p_sigma: !real) -> (!vec, !vec, !vec)
    attributes {provides = ["symplectic", "time_reversible"]} {
  %v1 = dyn.kick %v, %f, %m, %half : !vec
  %x1 = dyn.drift %x, %v1, %dt : !vec
  %f1 = md.evaluate @energy(%x1, %cell, %p_epsilon, %p_sigma) request [forces]
      : (!vec, !md.cell, !real, !real) -> !vec
  %v2 = dyn.kick %v1, %f1, %m, %half : !vec
  dyn.return %x1, %v2, %f1 : !vec, !vec, !vec
}
```

and a step that also logs the energies is a second program whose
`md.evaluate` requests `[energy, forces, virial]`. Constraints, couplings,
and virtual sites are further ops of the same programs (Section 6).

**What the driver specializes.** The driver (`lib/Driver`) reads the
control file (Appendix A) and the topology (Amber prmtop, GROMACS top and
itp, a CHARMM PSF with its files of topology, parameters, and streams, or
PDB) and writes the module as text. What is fixed for the run is
written into the IR as constants: the set of terms and their functional
forms; the cutoffs, skins, and truncations; the grid, order, and
splitting parameter of PME; the parameters that are the same for every
type; the precision mode; and the trip counts of the schedule. What
varies with the system is an argument of the entry function `@mdir_run`:
the positions, velocities, masses, and fields of the particles; the
tables of the parameters of pairs of types; the members of each tuple
set; the cell; and the step. A term absent from the run has no op in the
module, and a parameter shared by every type is a literal in the kernel.

**The schedule.** `@mdir_run` is a nest of `scf.for` loops over segments,
frames, energy outputs, coupling periods, and steps, with calls to the
host for the log, the trajectory, and checkpoints at the levels where they
happen. Before the first step the state is sorted in space
(`md_exec.spatial_order`, `md_exec.permute`; Section 4.2).

## 3.2 Sets, relations, and disjoint unions

Particles and tuples are sets; a relation names, for each tuple, its
members, and has an orientation: an unordered pair, an ordered tuple, or
a tuple equal to its reversal (an angle, a dihedral). A field is defined
over a set. A tuple set may be `disjoint`, no particle being a member of
two of its tuples, and an `md.disjoint_union` states that several tuple
sets share no particle (D83). The driver proves both when it builds the
sets of SHAKE and SETTLE; the IR then carries the fact, and two passes use
it. `md-bypass-updates` lets the loop over the tuples of one set of a
union read the positions from before the updates of the other sets, since
those updates are zero at its members; and the lowering evaluates a
disjoint tuple once, in the thread of its first member, writing every
member without atomics (Section 8.1).

A relation may also follow from the positions. Besides the pairs of a
neighborhood, `md.triplets` derives from them the triplets centered on
each particle: each center with each unordered pair of its neighbors
within a cutoff, the center in the middle, as in the three-body term of
Stillinger and Weber [[StillingerWeber1985]](references.md#stillingerweber1985) and of the mW water
[[Molinero2009]](references.md#molinero2009) (D160). The tuple ops sum over it as over the tuples of a
topology, and differentiation gives its forces and virial by the rule for
tuples of Section 3.3. Its size changes with the positions, so on the CPU
its members are written at every evaluation, from the rows of the
neighbor matrix, into a buffer as long as the triplets; the test of a leg
is that of the kernel, with the cutoff in f32 pulled in below the pole of
the term (D159). On 64 mW particles the forces agree with OpenMM's to
$6\times10^{-15}$ of the largest.

## 3.3 Differentiation and exchange contracts

`md-differentiate` replaces each `md.evaluate` with a call to a function
generated from the potential, which computes the requested quantities.

**Pairs.** For $U = \sum_{\{i,j\}} u(r_{ij})$ over the unordered pairs of a
relation, with $\partial r_{ij}/\partial\mathbf x_i = \mathbf d_{ij}/r_{ij}$,

$$
\mathbf F_i = -\frac{\partial U}{\partial\mathbf x_i}
= \sum_{j:(i,j)\in D} \mathbf K(i,j),
\qquad
\mathbf K(i,j) = -\frac{u'(r_{ij})}{r_{ij}}\,\mathbf d_{ij},
$$

where $D$ is the directed expansion of the relation, each unordered pair
as $(i,j)$ and $(j,i)$. The force becomes an `md.gather_relation`, a sum
per particle over the pairs it belongs to, with no conflicting writes. The
derivative $u'$ is taken in the IR, op by op, through the expansion of the
truncation. The virial of a periodic system is [[Louwerse2006]](references.md#louwerse2006)

$$
\mathsf W = \sum_{\{i,j\}} \mathbf d_{ij}\otimes\mathbf K(i,j).
$$

**Tuples.** For a term $u(q)$ of a coordinate $q$ of a tuple (a distance,
the cosine or the value of an angle, a dihedral), the force on member $m$
is $\mathbf F_m = -u'(q)\,\partial q/\partial\mathbf x_m$, and the virial
$\mathsf W = \sum_m \mathbf d_m\otimes\mathbf F_m$ with the displacement of
each member from one member of the tuple, which does not matter because
the forces of a tuple add to zero [[Thompson2009]](references.md#thompson2009). For the cosine
$c = \mathbf d_{ab}\cdot\mathbf d_{cb}/(r_{ab}r_{cb})$ of an angle at $b$,

$$
\frac{\partial c}{\partial\mathbf x_a}
= \frac{1}{r_{ab}}\Big(\frac{\mathbf d_{cb}}{r_{cb}} - c\,\frac{\mathbf d_{ab}}{r_{ab}}\Big),
\qquad
\frac{\partial c}{\partial\mathbf x_b}
= -\frac{\partial c}{\partial\mathbf x_a} - \frac{\partial c}{\partial\mathbf x_c},
$$

and the angle takes $\partial\theta = -\partial c/\sin\theta$. The dihedral
takes the form of [[Blondel1996]](references.md#blondel1996), which has no singularity where three
of its particles are in line. A kernel computes a coordinate and its
derivative with the same ops, so the elimination of common
subexpressions leaves one of each norm and product.

**Parameters.** `derivative(n)` requests $\partial U/\partial p_n$ for a
scalar argument, by the same rules. The reciprocal sum of Ewald returns
its own energy, forces, and virial (Section 5) and is not differentiated.

**Truncations.** `md-expand-truncation` writes the truncation of a pair
term into its kernel before the derivative is taken, so the force is the
exact derivative of the truncated energy:

*Table 3.2. Truncations, with $t = (r - r_s)/(r_c - r_s)$.*

| Truncation | Energy for $r < r_c$ |
|---|---|
| `shift` | $u(r) - u(r_c)$ |
| `force_shift` | $u(r) - u(r_c) - (r - r_c)\,u'(r_c)$ |
| `switch` from $r_s$ | $u(r)\,S(t)$, $S = 1 - 10t^3 + 15t^4 - 6t^5$ for $r > r_s$ [[GromacsManual2025]](references.md#gromacsmanual2025) |
| `force_switch` from $r_s$ | $u(r) - \tfrac{A}{3}(r-r_s)^3 - \tfrac{B}{4}(r-r_s)^4 - C$ for $r > r_s$, $u(r) - C$ below |

For the force switch, with $F = -u'(r_c)$, $F' = -u''(r_c)$, and
$\Delta = r_c - r_s$, a cubic added to the force makes the force and its
slope vanish at $r_c$: $A = (F'\Delta - 3F)/\Delta^2$,
$B = (2F - F'\Delta)/\Delta^3$, and $C = u(r_c) - \tfrac{A}{3}\Delta^3 -
\tfrac{B}{4}\Delta^4$. For a power law this is the force switch of the
GROMACS manual [[GromacsManual2025]](references.md#gromacsmanual2025); the pass derives $u'$ and $u''$ at $r_c$
from the kernel, so it applies to any $u$. It is not the force switch of
Steinbach and Brooks [[Steinbach1994]](references.md#steinbach1994), CHARMM's VFSWITCH, which multiplies
the force of each inverse power $r^{-n}$ by a switch linear in $r^{n/2}$:
$\Phi_n = r^{-n} - (r_s r_c)^{-n/2}$ below $r_s$ and
$\Phi_n = k_n (r^{-n/2} - r_c^{-n/2})^2$ above, with
$k_n = r_c^{n/2}/(r_c^{n/2} - r_s^{n/2})$. Defined only for a sum of
powers, it is not a kind of `truncation`; the driver writes it into the
Lennard-Jones of a topology, where it is CHARMM's to $2.2\times10^{-9}$
(D121). The two differ by a constant below $r_s$ and by up to 60% above
it: 9.23 kcal/mol on 12,017 particles of CHARMM36m.


**Squared-distance potential switching (D175).**
For the topology Lennard-Jones $u(r)$, `SQUARED_DISTANCE_SWITCH` selects
$u(r)S_2(r)$, the VSWITCH potential [[Brooks1983]](references.md#brooks1983).
A cubic polynomial in $r^2$ subject to $S_2(r_s)=1$, $S_2(r_c)=0$, and
$S_2'(r_s)=S_2'(r_c)=0$ is uniquely fixed. In the interval it is
$S_2(r)=(r_c^2-r^2)^2(r_c^2+2r^2-3r_s^2)/(r_c^2-r_s^2)^3$; below it is
1, and at and beyond the cutoff it is 0. Both energy and force are
continuous. The driver emits the cubic with its argument clamped to the
interval, multiplying the Lennard-Jones before differentiation, for the
ordinary pairs and the 1-4 pairs with their own parameters.

This is a change of the Hamiltonian: the configurational part of the
canonical partition function integrates $\exp[-U/(k_BT)]$ with the
switched $U$. Its thermodynamic force is therefore
$-\nabla[uS_2]=-S_2\nabla u-u\nabla S_2$; switching only the force by
$S_2$ would sample a different potential. The virial follows the same
Hamiltonian derivative under a cell strain. Coulomb is unchanged. The
plain-cutoff tail correction does not account for the contribution removed
between $r_s$ and $r_c$, so this option requires no dispersion correction.
LJPME and custom pair additions are refused. The analytic energy, force,
and virial tests cover both boundaries, CPU and GPU, mixed and double;
the independent two-POPC CHARMM comparison is documented in
`docs/charmm-m1.md`, Section 8.

**Particles.** A sum over the particles that reads the positions
themselves, $U = \sum_i k(\mathbf x_i, \dots)$, a term of the absolute
positions (D148), gives the force $\mathbf F_i = -\nabla k(\mathbf x_i)$
from a map over the particles whose kernel is the derivative of the
sum's, one component at a time along its unit vector. Its virial is the
derivative along the scaling of positions and cell (Section 6.4):
$\sum_i\mathbf x_i\otimes\mathbf F_i$ from the positions and, where the
kernel reads the edges of the cell (`md_exec.cell_edges`), $-\sum_i
\partial k/\partial L_a\,L_a$ on the diagonal (D154). A value that a
kernel receives from outside, such as one computed from the cell, is
independent of the kernel's arguments, and the edges of the cell are an
input of their own, whose derivative only the virial takes.

**Tabulated functions.** A function given by a table enters an
expression as any other function, and the force is the exact derivative of
what the table defines: the kernel computes the cell of each argument,
$\lfloor (x_k - x_k^\text{min})/\delta_k \rfloor$, and the place in it,
$\tau_k \in [0, 1]$, looks up the coefficients of the cell, and evaluates
a polynomial in the places, through which alone differentiation carries
the derivative; the index and the lookups have none. Of one argument the
polynomial is the cubic of a natural or periodic spline (D138). Of two or
three it is the product of cubics of Hermite along the axes, the bicubic
patch [[Press2007]](references.md#press2007) and the tricubic of Lekien and Marsden
[[Lekien2005]](references.md#lekien2005), which matches at the $2^n$ corners of its cell the value,
the derivative along each axis, and the mixed derivatives, each times the
spacings it is taken along; with
$H(\tau) = (2\tau^3 - 3\tau^2 + 1,\ 3\tau^2 - 2\tau^3,\ \tau^3 - 2\tau^2 + \tau,\ \tau^3 - \tau^2)$
the weights of $(p_0, p_1, \delta d_0, \delta d_1)$ along one axis, the
patch of two arguments is
$f = \sum_{a,b} H_a(\tau_1)\,H_b(\tau_2)\,g_{ab}$, with $g_{ab}$ those
corner values (D165). Two patches that share a face agree on it with their
first derivatives, since on the face each is the patch of lower dimension
of the same corner data: the energy and the forces are continuous, so a
run in a table conserves its energy as one in an analytic function does.
The derivatives at the points are those of the splines through the table
along each axis, and the mixed ones of splines through those, in the
order of the continuous functions of OpenMM [[Eastman2017]](references.md#eastman2017), so that a
table gives the same function in both. A discrete function is the value
at the nearest point, whose derivative is zero and whose value jumps
between points: it is meant for arguments that do not move, such as the
kinds of two particles in a parameter of each.

The representation of the input grid does not change that Hamiltonian
(D178). A text file gives the same values in the same
axis order as the inline table, so it gives the same spline coefficients,
$U$, and $-\partial U/\partial\mathbf x_i$. The canonical weight
$\exp[-(K+U)/(k_BT)]$ and its partition function $Z$ therefore remain
the same. A file is decoded and checked before the IR is built, with the
last argument varying fastest as in inline nested lists, its shape
supplied explicitly, and its values reordered into the internal table
layout before fitting the splines; it
introduces no reading or interpolation inside the loop of steps. The
fingerprint of a continuation records the resolved values in the inline
representation, because changing a grid changes $U$ even if its filename
does not change. The manifest separately hashes the file's bytes to
record its provenance.

**Parameters of each particle.** A term may weigh each particle by a
number that the control file gives it by masks of Amber (D165), $w_i$,
the per-particle parameter of the custom forces of OpenMM. A pair term
reads it as $w_1, w_2$, gathered as a field of the particles like the
charges, and the reader tests that the energy of a pair does not change
when the two are exchanged with their parameters (D137); a term over
tuples reads $w_1, \dots, w_N$ by place, which the driver writes as
parameters of each tuple, and a term of the positions reads $w$ itself.
The parameters are constants of the potential, so they change neither
its derivatives nor the virial.

**Scalar rules (D183).** An op interface dispatches rules, supplied
as external models for arithmetic, math, and vectors. Every model declares
differentiable and structural operands; conditions and indices are
structural. The emitter consults these declarations, and the coverage pass
requires a rule or declared zero for every scalar op in a potential. A
missing active rule is an error, distinct from a proved zero.

**Activity (D182).** The dependency proof is a separate analysis,
shared by scalar coordinates, cell edges, positional intermediate fields,
sum weights, and scalar parameters. Its outcomes are active, proven
inactive, and unknown with a reason. Active does not guarantee a nonzero
derivative or a rule; only proven inactivity licenses a zero without a
rule. Unknown required activity fails. Operands and captured kernel values
are followed, and block arguments are independent only in scopes whose
meaning is known. `--md-analyze-activity=argument=N` reports these outcomes.

**Parameters.** `derivative(n)` asks $\partial U/\partial\theta_n$ of a
scalar argument $\theta_n$ of the potential (D2), such as a component of
$\boldsymbol\lambda$ of `[free_energy]` (Section 6.8, D161). The
derivative of a sum over a relation, over tuples, or over particles is a
sum of the same kind whose kernel is the derivative of the sum's, taken
where the kernel reads $\theta_n$ from outside. Every value is one of
three. It is *independent* of $\theta_n$ only if a proof says so: every
path by which $\theta_n$ could reach it is modeled, the operands of ops,
the values that their kernels take from outside, and the arguments of
blocks whose meaning is known (those of the potential and of a kernel);
its derivative is then exactly zero, and `remarks=true` of the pass lists
each op taken so. It is *dependent*, and a rule gives its derivative. Or it
is dependent without a rule, or its independence cannot be proved (an op
that the pass does not know, with effects on memory, or with a region whose
meaning it does not know), and the derivative fails with an error that
names the op and says which of the two it is: a constant derivative and a
failure are never confused. The reciprocal sum of particle mesh Ewald is not
differentiated through: its energy is a quadratic form of the charges,
$E(\mathbf q) = \tfrac12\mathbf q^\mathsf T A\mathbf q$ with $A$
symmetric, so $dE/d\theta = \boldsymbol\delta^\mathsf T A\mathbf q =
\big(E(\mathbf q + \boldsymbol\delta) - E(\mathbf q - \boldsymbol\delta)\big)/2$
exactly, with $\boldsymbol\delta = \partial\mathbf q/\partial\theta$ the
derivative of the map over particles that computes the charges: two more
reciprocal sums of the energy alone, and no change to their templates. In
mixed precision the grid is in f32 and the difference carries an error of
about $\epsilon_{32}E$; on ethanol in 467 waters, $5\times10^{-6}$
kcal/mol of $\partial U/\partial\lambda_\text{C}$ against double
precision. A field that a sum gathers and that depends on $\theta_n$
other than through the charges of a reciprocal sum is refused.

**Exchange contracts.** A pair kernel carries a contract that says how
its value for $(j,i)$ relates to its value for $(i,j)$: `symmetric`,
`antisymmetric`, or `none`, with a basis: `proof`, `derived`, or
`asserted`. A sum over an unordered relation must be `symmetric`, and its
basis is `proof` unless it says otherwise: every pair term that the driver
writes, like one written by a user, is proved by `md-check-exchange`, the
first pass of every pipeline. The pass swaps the two particles in the
kernel ($\mathbf d \to -\mathbf d$, the values gathered for $i$ and $j$
exchanged, tables looked up with their symmetry) and compares the kernel
with itself structurally, commutative ops in either order; a contract
that it cannot prove fails the compilation, with the hint that the
author may assert it instead. What differentiation creates carries the
basis `derived`: since $\mathbf d_{ji} = -\mathbf d_{ij}$ and the factor in
front of it depends on $r$ alone, $\mathbf K(j,i) = -\mathbf K(i,j)$, so
the force is `antisymmetric` and the virial `symmetric` by construction.
The contracts are what makes each pair once legal: a loop over groups
(Section 4.4) computes $\mathbf K(i,j)$ once and adds $\pm$ it to both
particles, and `md-exec-choose-neighbors` chooses groups only when every
destination of the loop has a contract.

## 3.4 From sums to loops, and the passes over loops

`convert-md-to-md-exec` lowers a neighborhood to a build of cells and of a
neighbor structure; a sum or gather over a relation to `md_exec.pair_for`
(a sum over an unordered relation, computed over its directed expansion,
takes a weight $\tfrac12$); sums and maps over particles, kicks, and
drifts to `md_exec.particle_for`; and sums over tuples to
`md_exec.tuple_for` over an incidence, the members of each tuple. A loop
in the *value form* takes fields and returns fields. The passes that
follow, in the order of the pipeline that `tools/mdir/Run.cpp` builds:

*Table 3.3. The passes over `md_exec`, in order.*

| Pass | What it does |
|---|---|
| `md-exec-reuse-neighbors` | Carries a neighbor structure across the steps of a loop, and turns each build into a refresh that rebuilds only when its test asks; with a pruned distance, keeps a dual list (Section 4.5) |
| `md-exec-expose-validity` | Writes the test of validity (Section 4.1) as a loop over particles with an `i1` reduction |
| `md-exec-fuse-loops` | Fuses loops over pairs over the same structure, positions, cutoff, and policy; fuses chains of loops over particles, such as a kick, the drift that follows, and the test; bitwise identical |
| `md-exec-accumulate-destinations` | Lets the loops whose forces a loop over particles adds accumulate into one destination, in the order of the sum |
| `md-exec-narrow-sums` | Sums only the elements of a vector sum that are used, such as the trace of the virial |
| `md-exec-simplify-distance` | Writes pair kernels in powers of $r^2$, and outlines functions of the distance alone (Section 7.2) |
| `md-exec-fold-tables` | Computes what a kernel derives from tables of types, such as $4\varepsilon\sigma^{12}$, once per entry, in f64, into packed tables |
| `md-exec-choose-neighbors` | On a device, chooses groups of 16 and each pair once where the contracts allow it (D89) |
| `md-exec-assign-precision` | Assigns a type to every field and kernel by its role (below) |
| `md-exec-approximate`, `md-exec-expand-radial` | Approximations of f32 kernels of terms, with stated bounds (Section 7.2) |
| `md-exec-assign-storage` | Turns the value form into the storage form: every field gets a buffer, loops update buffers in place, scratch comes from a pool; never copies a field, and fails with the reason where it would have to |
| `md-exec-assign-streams` | Optionally moves an op to a second stream where an independence proof holds (D87; off by default) |

The fused loop of the kick, the drift, and the test of validity of the
mixture, after these passes and before the lowering, abridged:

```mlir
%85:3 = md_exec.particle_for ins(%v, %f, %m, %x, %x_ref : ...) outs(%e0, %e1 : ...)
    reduce(%false : i1) {
^bb0(%v_i: vector<3xf64>, %f_i: vector<3xf64>, %m_i: f64, %x_i: vector<3xf64>,
     %ref_i: vector<3xf64>):
  ...                                  // u = v + (h / m) f, a massless site keeps v
  %x1 = arith.addf %x_i, %dx : vector<3xf64>
  %moved = arith.cmpf ugt, %d2, %bound : f64
  md_exec.yield %u, %x1, %moved : vector<3xf64>, vector<3xf64>, i1
} -> !md.field<@atoms, 3 x f64>, !md.field<@atoms, 3 x f64>, i1
%86 = md_exec.refresh_neighbors %nl, %85#1, %cell moved(%85#2) cutoff(9.0e-01)
    skin(1.0e-01) cell_width(0.333) policy(check) : ...
%88 = md_exec.pair_for %86, %85#1, %cell ins(%eps, %sig : ...) outs(%zeros : ...)
    cutoff(9.0e-01) exchange [antisymmetric] policy(directed, owner_only) { ... }
```

**Precision by role** (`md-exec-assign-precision`). A mode assigns a type
to each role (Table 7.1): positions, forces, the kernels of terms, the
integrators, and the global sums. A field takes the type of its buffer if
the driver gave it one (in the mixed mode the state and the masses are
f64, the forces and the parameters of particles f32), and otherwise the
type of its role, in the order position, force, integrator. A kernel
computes in the type of its role, and the pass converts a value where it
crosses from one role to another: in the mixed mode, one loop over
particles converts the positions to f32 once per evaluation for the loops
over tuples (D79), and a loop over groups converts each position after it
has added the shift of its frame in f64 (D101). Scalars outside loops stay
f64, and tables go to the device in f32 when their kernels are f32.

## 3.5 Lowerings and templates

`convert-md-exec-to-loops` lowers the storage form to `scf` loops for the
CPU, with `scf.parallel` where iterations are independent; with more than
one thread, `convert-scf-to-openmp` makes them OpenMP. A global sum is
the reduction of its `scf.parallel`, which OpenMP combines from partial
sums per thread; the variable of each reduction is an alloca, which
`hoist-static-allocas` moves to the entry of the function so that the loop
of the steps does not grow the stack (D117). `convert-md-exec-to-gpu` lowers
to kernels of the `gpu` dialect (Section 8), and the upstream
`gpu-lower-to-nvvm-pipeline` outlines them, lowers them to NVVM, links
libdevice, and embeds PTX in the module.

The structures and algorithms that are not loops over a set, the builds of
the neighbor matrix and of the groups, the spatial sort, and particle mesh
Ewald on a device, are *templates* written in MLIR
(`lib/Runtime/Templates/`, 41,049 lines, of which 37,559 are generated by
two scripts: the groups and PME on a device, each in an orthorhombic and
a triclinic variant, D125 and D126). The lowering parses the
templates that a run uses into its module and calls them, so they are
compiled with the rest of the program and specialized by its constants:
the cutoff and the skin, the grid and the order of PME, the precision; a
triclinic variant is parsed only into a module whose cell is triclinic.

## 3.6 Compilation at run time, and the runtime

`mdir run` compiles before it runs. It builds the module, parses it, runs
the pipeline, and hands the result to MLIR's `ExecutionEngine`, which
compiles the host code with LLVM at its default level of optimization
(the most aggressive level measured no faster on the Amber suite,
D197), with the fast list scheduler, whose time does not grow
exponentially with the calls of a block as the default's can (D150), and
links it with the runtime. Each kernel is a GPU module of its own.
The modules are serialized in parallel on the threads of the process
(D214). Their architecture is that of the device that
will run them, which the compiler asks of NVML when it lowers the
program, without creating any CUDA state. It is not fixed when MDIR is
built. Each module becomes PTX
through LLVM's NVPTX back end and then a cubin through the toolkit's
`ptxas`, which the driver loads without compiling. Without a device to
ask, an architecture LLVM does not know, or `ptxas`, the kernels stay
PTX, which the driver compiles when the module is loaded
(`cuModuleLoadDataEx`). The log reports the time from the
parsed module to a callable entry point, which includes the passes, the
code generation of LLVM, the creation of the CUDA context, and the
compilation of the PTX: about 10 s at constant energy and 20 s at
constant pressure for the systems of the Amber suite, whatever their size
(Table 10.2). The module of JAC at constant pressure has 9,236 lines as
the driver writes it, and the lowering outlines 545 GPU modules from it.

A Python simulation compiles on an engine of its own (D199), whose host
code generation can be cached on disk (D212,
`MDIR_COMPILE_CACHE_DIR`). The relocatable object is a function of the
LLVM module and of the code generator alone, so an entry is keyed by a
hash of the module's bitcode, which holds the PTX of its kernels, and by
the LLVM version, the CPU and its features, and the options of code
generation, but not by the build of MDIR: a rebuild that generates the
same module hits. An entry also stores its full key and a hash of its
object, so that a collision or a damaged file is a miss, and a hit is
linked and checked as a generated object is. The passes, which do depend
on MDIR, run on every compile. The PTX and the cubin of each GPU module are
cached in the same way. The PTX is keyed by a hash of the module's IR,
which then holds only upstream LLVM and NVVM operations, together with its
target, libdevice, and the LLVM version. The cubin is keyed by a hash of
its PTX and the version and arguments of `ptxas`.

The parts of a Python simulation run in one activation of its entry
(D[resident-buffers], `docs/python-segments.md`). The program wraps its loops
of steps in a loop over parts, and at the end of each part it calls the
host with its positions, velocities, and forces where they are, addresses
of the device on a GPU (`mdrt.host_call` with `in_place`), and waits there,
on a stack of its own, until the host gives it the next part. Its buffers,
its order of the particles, and its neighbor structures therefore persist
from part to part: a run in parts is the run in one part, to the bit in the
deterministic mode, and a part of 10 steps of JAC no longer copies 50 MB to
the device. The runtimes record what an activation allocates, which is
freed when it ends; the state at the end of each part is copied on the
device, so that a part that fails returns to it (D196).

The runtime is small and holds what cannot be IR: 681 lines of C for the
host (`runtime/mdrt.c`: counters of builds and prunings, the stop on a
position that is not a number, the generator Philox 4×32-10
[[Salmon2011]](references.md#salmon2011) and the factors of the thermostat and the barostat,
isotropic or semi-isotropic, drawn from it, the means of the pressures by
axis that the semi-isotropic barostat took, the FFT of the host
[[Reinecke2019]](references.md#reinecke2019), and the matrix of the host)
and 1153 lines for NVIDIA devices (`runtime/mdrt_cuda.c`, on the driver
API only: loading of modules, launches that skip empty grids, one stream
in order plus an optional second one, words of memory of the host mapped
for the device for flags (D118), a caching allocator whose frees do not
wait, plans of cuFFT, growable buffers for neighbor structures, and the
records of what each activation of a Python simulation allocated). The
driver registers callbacks for the log, the trajectory (DCD or XTC), and
checkpoints (H5MD 1.1 [[deBuyl2014]](references.md#debuyl2014), all
values in 64 bits, from which a run continues bitwise). A checkpoint
records the step at which its run began, the frames written, and the
energy that the coupling has taken, so that `mdir run --continue` carries
one run over as many jobs as it takes, appending to its trajectory; a
signal or the wall time stops a run after a checkpoint is written, with
the exit status 75 (D129 to D132; Appendix A.3). A checkpoint also records
what defined its run, a fingerprint of its physics, its coupling, and its
execution: `mdir run --continue` refuses a checkpoint of other physics or
coupling and names each change, and a run that begins from another run's
checkpoint evaluates the forces of its first step where they differ,
rather than take the forces of the physics before
(D172). Format 1 of the file is the contract of
release 0.1.0: the reader checks the format and the SHA-256 of the state,
and the file is on stable storage before it takes its name
(D173).

### Parameters that change without compiling

A compiled program depends on the structure of its model (the terms and
their expressions, the cutoff, the grid, the constraints) and on values. A
Python model may declare values *tunable* (D213,
`docs/python-tunable.md`): charges, $\sigma$ and $\epsilon$ of the
Lennard-Jones types, constants of pair terms, and parameters of tuple
terms, each a vector $\boldsymbol\theta$ with a map $\iota$ from its sites,
so that site $s$ takes $\theta_{\iota(s)}$. Nothing that depends on them is a
constant of the program's text: the charges, the table of the types, and
the fields of tuples are buffers of the entry already (Section 3.6), and a
tunable constant of an expression is read from a row of the program's
scalars in the kernel. A new $\boldsymbol\theta$ is put into the model,
and the builder computes the values of the program from it by the code
that compiles it; the update is accepted only if the text it builds is the
compiled one, so that an update and a compile with the new values give the
same buffers, and the same trajectory to the bit (Section 9).

The quantities derived from $\boldsymbol\theta$ are recomputed with it.
With $q_i$ the charges, the self term and the background of a net charge
of particle mesh Ewald are

$$
E_\text{self} = -f\frac{\beta}{\sqrt\pi}\sum_i q_i^2,\qquad
E_\text{bg} = -\frac{f\pi Q^2}{2V\beta^2},\quad Q = \sum_i q_i,
$$

the second proportional to $1/V = e^{-(\varepsilon_x + \varepsilon_y +
\varepsilon_z)}$, so that $\mathsf W_{aa} = -\partial E_\text{bg}/\partial
\varepsilon_a = E_\text{bg}$ and its virial has the trace $3E_\text{bg}$,
which the barostat takes with the dispersion's $6E_\text{disp}$
(Section 5.4) as arguments of the entry once they depend on
$\boldsymbol\theta$. The pairs three bonds apart take $fq_iq_js_C$; the
table of the types takes $\sigma_{ab} = (\sigma_a + \sigma_b)/2$ and
$\epsilon_{ab} = \sqrt{\epsilon_a\epsilon_b}$ from the types' own, but for
pairs that the model sets apart from the rule (NBFIX), which keep theirs, and
$\langle C_6\rangle$ of $E_\text{disp}$ follows from it; the tails $I_{ij}$ of
pair terms are integrated anew (Section 5.4).

The purpose is reweighting. Frames $S_n$ sampled from the canonical
distribution of $U_{\hat{\boldsymbol\theta}}$ give the average of $O$ at
$\boldsymbol\theta$ without sampling it: with $\Delta U = U_{\boldsymbol\theta} -
U_{\hat{\boldsymbol\theta}}$,

$$
\langle O\rangle_{\boldsymbol\theta}
= \frac{\int O\,e^{-U_{\boldsymbol\theta}/k_BT}\,d\mathbf x}{\int e^{-U_{\boldsymbol\theta}/k_BT}\,d\mathbf x}
= \frac{\langle O\,e^{-\Delta U/k_BT}\rangle_{\hat{\boldsymbol\theta}}}{\langle e^{-\Delta U/k_BT}\rangle_{\hat{\boldsymbol\theta}}}
\approx \sum_n w_nO(S_n),\qquad
w_n = \frac{e^{-\Delta U(S_n)/k_BT}}{\sum_m e^{-\Delta U(S_m)/k_BT}},
$$

the numerator and the denominator multiplied by $Z_{\hat{\boldsymbol\theta}}$
[[ThalerZavadlav2021]](references.md#thalerzavadlav2021). The estimate holds
while the weights spread over many frames, $\exp(-\sum_n w_n\ln w_n)$ of
them; when that number falls, sampling continues at $\boldsymbol\theta$,
which is an update, not a compile. The version of the values that each
energy and frame records says which $\hat{\boldsymbol\theta}$ it was sampled
at.

An update changes the Hamiltonian $H = K + U_{\boldsymbol\theta}$ between two
steps. A step of velocity Verlet, $e^{\frac{\Delta t}{2}\mathcal L_U}
e^{\Delta t\mathcal L_K}e^{\frac{\Delta t}{2}\mathcal L_U}$ with
$\mathcal L_U = \sum_i\mathbf F_i\cdot\partial/\partial\mathbf p_i$, is the
time-reversible, symplectic map of one Hamiltonian only if both of its half
kicks take the forces of that Hamiltonian; the forces carried from the last
step are those of $U_{\hat{\boldsymbol\theta}}$ at $\mathbf x$, so an update
evaluates $\mathbf F_i = -\partial U_{\boldsymbol\theta}/\partial\mathbf x_i$ at
the same $\mathbf x$ before the next step (a call of the entry with no
steps). Leapfrog's velocities of the half step before are momenta of the
trajectory already taken and stay; its next kick takes the new forces.

## 3.7 The objects of MDIR, for developers

A developer meets the same few objects at every level; Table 3.4 lists
them with the op or type that carries each in the IR and the code that
makes or consumes it. The paragraphs after it follow one term through
them.

*Table 3.4. The objects of MDIR. Paths are relative to the root of the
repository.*

| Object | What it is | In the IR | Where |
|---|---|---|---|
| Particle set | The particles of a run, one set, in the order of the input; the state is sorted in space before the first step | `md.particle_set @atoms` | `include/mdir/Dialect/MD/MDOps.td` |
| Tuple set | Tuples of particles of one arity: bonds, angles, the members of a constraint group, the pair of the centers of a pull | `md.tuple_set @bonds on(@atoms) arity(2) orientation(unordered)`; `md.disjoint_union` | `MDOps.td`; built in `lib/Driver/Builder.cpp` (`Program::TupleSet`) |
| Field | A value per element of a set: positions, velocities, masses, charges, types, the parameters of a term; or one computed from the positions, such as the Born radii | `!md.field<@atoms, 3 x f64>` (a vector per particle), `!md.field<@bonds, f64>`; `md.map_particles`, `md.gather_relation` | `Program::Field`; the driver's fields are the arguments of `@mdir_run` |
| Relation | For each tuple, its members; a neighborhood, the pairs within a cutoff | `!md.relation<@atoms, 2, ordered, @links>`; `md.neighborhood %x, %cell cutoff(r)` | `MDTypes.td`; neighbor structures in `lib/Dialect/MDExec` (Section 4) |
| Cell | The periodic cell, orthorhombic or triclinic; without one, a cell no image reaches (D142) | `!md.cell`, `md.orthorhombic_cell`, `md.triclinic_cell`; `md_exec.cell_edges` | `include/mdir/Driver/Cell.h` |
| Table | Parameters of pairs of types, tabulated functions, the grid of CMAP | `md.lookup %t[%a, %b]` | `Program::Table` |
| Kernel | The scalar code of one interaction, in `arith`, `math`, and `vector`: what a sum adds for a pair, a tuple, or a particle | the region of `md.sum_relation`, `md.sum_tuples`, `md.sum_particles`, `md.gather_relation`, `md.map_particles` | written by the Builder from the control file; differentiated by `ScalarDerivative` |
| Potential | The energy as a function of the positions, the cell, the fields, the tables, the relations, and scalars such as the time $t$ | `md.potential @energy(...) -> f64` | one per run and one per term for the log, in `Builder.cpp` |
| Request | What a program asks of a potential: energy, forces, virial, the derivative with respect to a scalar argument | `md.evaluate @energy(...) request [energy, forces, virial, derivative(n)]` | becomes `md.function` and `md.call` in `md-differentiate` (Section 3.3) |
| Program | A step: kicks, drifts, constraints, couplings, around the requests | `dyn.program`, `dyn.kick`, `dyn.drift`, `dyn.step` | `include/mdir/Dialect/Dyn` |
| Entry | The schedule of a run: loops over segments, frames, energies, periods, and steps, with the calls to the host for the outputs | `func.func @mdir_run`, `mdrt.host_call @mdrtWriteEnergies(...)` | `Builder::emitEntry`; the callbacks in `lib/Driver/Output.cpp` |
| Loop | A sum or a map lowered to an iteration over a set with a pattern of access | `md_exec.particle_for`, `md_exec.pair_for`, `md_exec.tuple_for`, `md_exec.reciprocal` | `lib/Conversion/MDToMDExec`; passes in `lib/Dialect/MDExec/Transforms` (Section 3.4) |
| Neighbor structure | The pairs within the reach, the matrix or the groups of 16, with its test of validity | `md_exec.build_neighbors`, `md_exec.refresh_neighbors`, `!mdrt.neighbors<@atoms>` | Section 4; `runtime/mdrt_cuda.c` |
| Storage | The buffers of fields, the scratch of reductions, the cells of results read by the host | `memref`s after `md-exec-assign-storage` | `lib/Dialect/MDExec/Transforms/AssignStorage.cpp` |
| Kernel of a device | A loop lowered to a launch, its sums reduced in blocks | `gpu.launch` | `lib/Conversion/MDExecToGPU` (Section 8) |

**From the control file to a kernel.** The driver reads the control file
into a `Control` (`lib/Driver/Control.cpp`) and the topology and the
coordinates into a `System` (`lib/Driver/System.cpp`), which resolves the
masks of Amber into particles and checks what the IR cannot. The `Builder`
then writes the module as text: the sets, the fields and tables as
arguments of `@mdir_run`, the potential, the programs, and the entry. A
term of the absolute positions, for instance (D148), becomes a flag field
`ext0`, 1 for the particles of the term, a field for each parameter given
as a list, and in `@energy` a `md.sum_particles` over the positions and
those fields whose kernel evaluates the expression and selects on the
flag. `md-differentiate` turns the request `[energy, forces, virial]` of a
step into a function in which the forces are a `md.map_particles` and the
virial a second `md.sum_particles` (Section 3.3). `convert-md-to-md-exec` makes
both `md_exec.particle_for` loops, `md-exec-fuse-loops` joins them with
the other loops over particles, the storage pass gives them buffers, and
the lowering to a device makes one kernel that reduces the energy and the
virial in blocks (Section 8). A new kind of term therefore needs a reader
in `Control.cpp`, what the system must resolve in `System.cpp`, its
emission in `Builder.cpp`, and, if its form is new to the IR, the rules of
its derivative in `lib/Dialect/MD/Transforms/Differentiate.cpp`; the
loops, the storage, and the kernels of a device follow from the IR.
Appendix B.1 lists the tiers of tests, and B.2 where a change goes.

**Singular and branch points (D185).** Scalar differentiation
uses the chosen branch: `abs(0)` takes its nonnegative branch; min/max ties
take the second operand; a select's condition has no tangent; floor, ceil,
integer conversions, steps, and discrete table lookups have piecewise-zero
derivatives even at jumps. Continuous table endpoints use the first/last
polynomial interval and strictly outside the table both value and tangent
are zero. An active square root at zero and coincident radial geometry
have no invented finite continuation; the generated arithmetic can be
nonfinite and the numerical checker fails. An inactive square root still
has a proved zero tangent. Differences across a branch boundary need not
agree with the chosen derivative. See the singular-point tests and
`docs/ad-robust.md` for the full conventions.

**Diagnostic attribution (D186).** The driver attaches
the control-file term name as a named location to every custom-expression
op. Activity retains the operation that prevented a proof, and diagnostics
routed through sums or intermediate fields point at that operation's
location. The numerical semantics are unchanged. An injected unsupported
scalar operation tests that the error names the source term.
