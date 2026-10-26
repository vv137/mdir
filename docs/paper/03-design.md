# 3. Design

A run of MDIR passes through four levels of IR, each a dialect of MLIR
[[Lattner2021]](references.md#lattner2021). The upper two say *what* is computed and *how the state
advances*; the third says *over which set, with which pattern of access*;
the fourth is the interface to a small runtime. Every pass between them
either rewrites within a level or lowers to the next, and a pass may rely
only on what the IR states (Section 12). The pipeline, one step, and the
path from facts to strategies are drawn in brief in
[overview.md](../overview.md).

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

The distributed level now includes `md_dist.partition_centers`
(D317). For each owned center $i$ of a neighborhood
relation, $R_i^\mathrm{ghost}$ contains its pairs whose neighbor is a
ghost row. The operation's two `i32` indicators identify $I_d$ and $B_d$,
with $I_d\cap B_d=\varnothing$ and $I_d\cup B_d=O_d$ for the relation's
layout epoch, whether it comes from initial partitioning or a completed
migration. This
classification follows completion of the position halo used to construct
$R$; it does not move a transfer wait. The sets allow a later lowering to
put pair work that reads only owned fields between a later transfer's
start and completion.

The proved filter (D328) selects either $I_d$
or $B_d$ on a pair traversal only when the indicator is a direct result of
that relation's partition. Thus the two partial traversals form a disjoint
partition of the original sum. The interior traversal may inspect owned
neighbors only: by the definition of $I_d$, its omitted ghost terms form
the empty set. This filter alone preserves the immediate-wait schedule;
placing it between a later start and wait is a separate step.

`md_dist.covered` (D332) expresses the collective Verlet skin test
[[Verlet1967]](references.md#verlet1967) for a
halo built at positions $\mathbf x^\text{ref}$ with reach $R_h$. For current
positions $\mathbf x$, $\Delta_h$ is the largest displacement of an owned
particle in the wrapped coordinate frame of the halo's explicit images.
With the cell unchanged, a candidate image retains its build-time shift.
The old candidate membership is sufficient for a cutoff $r_c$ when
$2\Delta_h < R_h-r_c$: by the triangle inequality, a pair at the current
cutoff was at most $r_c+2\Delta_h < R_h$ apart in that same image at build
time. A periodic wrap invalidates coverage even when the minimum-image
displacement is small. The predicate is one value for all domains, including empty ones.
The current coordinate values remain a separate version and require their
own halo copy before use. This predicate currently runs in the in-process
reference executor; loop-carried distributed state and the production
driver remain to be connected.

### Execution contracts (design)

The semantic levels may state any computation. Before the execution levels
schedule it, they must establish the provenance of values, the effects of
operations, and the validity of the state on which an access depends.
Invariant meaning, such as the set of a buffer or the representation of a
feature, belongs to a type or static attribute. Positions, cell, ownership
epoch, and copy completion change during a run and therefore remain SSA
values and explicit operands. A witness produced by a check is valid only
for those state values. The consumer checks that provenance rather than
assuming that a prior check still applies.

A required execution condition is proved or established by a sound runtime
guard with legal recovery. An unknown required condition without such a
guard is diagnosed as an unsupported contract; its path is never executed.
Contradictory types, mismatched entities, invalid evidence, and demonstrated
uninitialized reads are invalid IR and are refused. A false or unknown
requirement of an optional strategy selects an independently legal baseline
with an `explain` reason. `Unknown` establishes no positive fact. Neither
the availability of a runtime predicate nor that of an asynchronous event
establishes its truth or completion. Existing input defects are diagnosed
by `mdir check`, not classified as unsupported compiler contracts.
Numerical requirements are represented separately from structural and
memory requirements, but all are prerequisites of a legal plan; the cost
model never weakens them. Thus a race-free atomic sum is illegal when a
fixed order is required. This design conditionally verifies supported
contract instances under named rules and recorded assumptions, not every
program. Its trusted base includes predicate rules and their numerics,
effect and derivative models, preservation arguments, accepted external
contracts, the runtime and backends, and the toolchain. This is the design
of D334 (#574); individual checks and witnesses belong
to later decisions. Existing inputs and emitted programs do not change.

The extended D338 design refines a contract into
requirements, established facts, effects, numerical policy, and named
external assumptions. A fact is tied to the versions of the positions,
cell, ordering, ownership, and allocation that establish it; a function or
loop carries it only with checked preconditions and preservation on every
path. For an asynchronous transfer, token definition alone does not make
data ready: completion must precede a consumer and make its memory domain
visible. Source and destination storage remain borrowed until completion.
For a checkpoint, the copy's completion, the writer's release of staging
storage, and the file's commit are three distinct events (D336). A plan
preserves the required numerical order and derivative pairing before cost
can select among legal schedules. These are design obligations, not new
runtime checks in this document.

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
The kinds and member places of these coordinates are inherent attributes
of the tuple operations, checked against tuple arity and accessed through
their typed operation interface in differentiation
(D343). Their numerical definition and kernel order
do not change.

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
`docs/charmm.md`, Section 8.

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

**Neural stages (D288, D289).** A
message-passing potential is written in the same ops, with its neural
bodies as stages: a layer $k$ is a gather over the pairs,
$\mathbf A_i^{(k)} = \sum_{j:(i,j)\in D(N)}\phi_k(r_{ij}, \mathbf d_{ij},
\mathbf h_i^{(k-1)}, \mathbf h_j^{(k-1)}, z_i, z_j)$, and a map,
$\mathbf h_i^{(k)} = \psi_k(\mathbf h_i^{(k-1)}, \mathbf A_i^{(k)})$, from
$\mathbf h_i^{(0)} = \eta(z_i)$ to the readout $U = \sum_i\mathcal
R(\mathbf h_i^{(N_\text{layer})})$. `md.stage` declares a pure function of
one element and binds its body and its vector-Jacobian product, which
external AD supplies (D167); `md.apply_stage` applies either in a kernel.
The features are fields of $N$ components whose type states the
representation of O(3) they carry, `irreps<4x0e + 4x1o>`, and its layout.
The derivative follows from the first-order change of $U$ in the values of
one gather, with $\bar{\mathbf A}_i = \partial U/\partial\mathbf A_i$ known
from the fields after it (taken in the reverse order of the body):

$$
\delta U = \sum_i\bar{\mathbf A}_i\cdot\delta\mathbf A_i
= \sum_{(i,j)\in D(N)}\big[\bar r_{ij}\,\delta r_{ij}
+ \bar{\mathbf d}_{ij}\cdot\delta\mathbf d_{ij}
+ \bar{\mathbf h}^\mathrm c_{ij}\cdot\delta\mathbf h_i
+ \bar{\mathbf h}^\mathrm n_{ij}\cdot\delta\mathbf h_j\big],
$$

where the bracket is the vector-Jacobian product at $(i,j)$ for the
cotangent $\bar{\mathbf A}_i$, by its definition. Collected at a particle
$m$, the terms in $\delta\mathbf h$ come from the pairs $(m,b)$, where $m$
is the center, and $(b,m)$, where it is the neighbor, both pairs of the
relation at $m$: $\bar{\mathbf h}_m \mathrel{+}= \sum_b[\bar{\mathbf
h}^\mathrm c_{mb} + \bar{\mathbf h}^\mathrm n_{bm}]$, a gather over the
same relation that reads $\bar{\mathbf A}$ at both ends. The adjoint of a
gather is a gather, and that of a sum or a map over the particles a map of
the product at each particle. With $\delta r_{ij} = \mathbf d_{ij}\cdot
\delta\mathbf d_{ij}/r_{ij}$ and $\delta\mathbf d_{ij} = \delta\mathbf x_i -
\delta\mathbf x_j$, the geometric terms of $(i,j)$ are $\mathbf
g_{ij}\cdot(\delta\mathbf x_i - \delta\mathbf x_j)$ with $\mathbf g_{ij} =
\bar r_{ij}\mathbf d_{ij}/r_{ij} + \bar{\mathbf d}_{ij}$, so the force on
$m$ is $\sum_b\mathbf K(m,b)$, $\mathbf K(m,b) = -(\mathbf g_{mb} - \mathbf
g_{bm})$, antisymmetric by construction. The species do not move, and
$\mathbf h^{(0)}$ depends on them alone, so the positions enter $U$ only
through the pairs, every layer's alike. Under the strain $\mathbf x \to
(\mathsf 1 + \boldsymbol\varepsilon)\mathbf x$ with the cell, every
displacement becomes $(\mathsf 1 + \boldsymbol\varepsilon)\mathbf d_{ij}$,
so $-\partial U/\partial\varepsilon_{ab} = -\sum_{(i,j)}g_{ij,b}\,d_{ij,a}$
over the layers' pairs, which over the unordered pairs, with $\mathbf
d_{ji} = -\mathbf d_{ij}$, is $\sum_{\{i,j\}}(\mathbf d_{ij}\otimes\mathbf
K(i,j))_{ab}$: the virial of pairs, and with it the pressure of Section
6.4, $P = (2K + \mathrm{tr}\,\mathsf W)/3V$, the derivative of the free
energy in the volume. A product gives every cotangent of a pair at once,
so the derivative emits one gather of several results for each stage
gather, the adjoint terms of the fields it reads, the forces, and
$\tfrac12\mathbf d_{mb}\otimes\mathbf K(m,b)$ at each center for the
virial: two evaluations of the product per directed pair, where one op for
each term took six; the one-evaluation scatter is a lowering of step 4 of
#377. On a synthetic two-layer model with 16 components, the identity
$\langle\bar{\mathbf y}, J\mathbf v\rangle = \langle J^\mathsf
T\bar{\mathbf y}, \mathbf v\rangle$ of each stage holds to $1.3\times
10^{-15}$, MDIR's forward rules against the bound product, and the
energy, forces, and virial agree with an independent reverse sweep of the
whole energy to $2\times10^{-15}$ on the CPU and $5\times10^{-16}$ on an
RTX 4090 in double ([ml4-ir.md](../ml4-ir.md)).

**Bodies in tiles (D293, D294).** A body
from external AD is batched: a function of tensors whose leading dimension
is the rows of a tile, exported by JAX to StableHLO and legalized to
`linalg` when the model is exported. Its semantics stay those of one
element, since the leading dimension is parallel in every op of the body:
row $t$ of the result is the body applied to row $t$ of the operands, and
no row reads another. A loop whose kernel applies such a body therefore
retains the verified static leading row count as `md.stage_rows` when the
body's tensors become buffers (D[stage-rows-attr]); the forward and VJP
bodies must agree on that count. The loop then
runs as a host loop over tiles, each packing the operands of its rows,
calling the body, and unpacking the results into the rest of the kernel;
the sum over the pairs of a center, $\mathbf A_i = \sum_j\phi(i,j)$, is
the same sum taken in the order of the center's slots, one thread a
center, so it stays a gather without atomics. Every product recomputes
its forward, so the device holds, besides a constant for the tile, only
the fields of the particles: nablaDFT's PaiNN (128 features, $N_\text{layer}
= 6$) re-implemented in JAX from SchNetPack's weights agrees with
SchNetPack's calculator in float64 to $5\times10^{-6}$ of the largest
force and takes 1.4 to 2.1 kB of device memory a pair of liquid water from
6,144 atoms up, against 61 kB for the opaque call, at about 10 µs a pair,
about 20 times the opaque call's time with MDIR's naive lowering of
`linalg` ([ml4-ir.md](../ml4-ir.md), Section 8).

The body then chooses a strategy for each reduction of its tensor
components (D318): for a sum of 2 to 31 terms, an
opt-in strategy uses one thread to add them in source order, keeps the
partial sum in a register, and writes only the final value; for a contiguous
sum of at least 32 terms, a warp combines them by D303's fixed butterfly.
The short-row strategy may also evaluate a pure pointwise map of that sum
in the same thread, while storing the sum for any other consumer. If a
final pointwise map returns the same tensor twice, it may write both
disjoint caller buffers directly; the generated stage caller allocates
them separately, and any body with another symbol user retains the copies.
The loop and separate map launches remain the default. These lowerings
implement the same reduction and pointwise maps; neither changes the
stage's function or its adjoint, and the cuBLAS calls remain explicit.

In mixed precision, an opt-in strategy gives the values passed between
stage bodies and the adjoints carried by the reverse sweep f64 field
storage (D329). A body exported with f32 arithmetic still
computes in f32; packing and unpacking convert at the body boundary. The
final coordinate force follows the ordinary mixed policy. This gives the
reverse sweep a stable inter-stage representation without retaining the
body's per-row intermediates for the whole system; the f32-field default
uses less device memory. The staged model's
`stage_field_precision = "DOUBLE"` key selects the f64 storage strategy.

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
destination of the loop has a contract. In terms of primitives, the
transpose of a gather is a scatter-accumulate; over the directed expansion
$D$, which is its own transpose, that scatter is again a gather over $D$,
and the contract is what lets the lowering write it as a scatter to both
ends instead. `docs/primitives-map.md` (D292) takes every part
of MDIR apart into such primitives and their transposes.

**Declared transposes.** The rules above are facts of the ops, not of the
pass (D295). An op with a kernel is a gather, the kernel, and
a reduction; its derivative in an operand that reaches the kernel is the
transpose of its linearization, which is the transpose of the reduction
(the adjoint of the result broadcast to each pair, tuple, or particle),
the reverse derivative of the kernel, and the transpose of the gather (the
values added at the particles that were read). Each op declares the first
and the last, the transpose of its structure, through an op interface:
the op that the derivative creates, over the same relation, tuples, or
particles. `md-differentiate` fills that op's kernel by the scalar rules.
Ops that are linear themselves, the transfers between domains of
`md_dist` and the scatter-accumulate of `md_exec`, declare the op of their
transpose in full. A check refuses an op of the structure without a
declared transpose, and a test checks each pair by the adjoint identity:
for the positions, the derivative along a direction $\mathbf v$, written
by hand, against $-\sum_i \mathbf F_i\cdot\mathbf v_i$.

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
| `md-exec-reuse-neighbors` | Carries a neighbor structure across the steps of a loop, and turns each build into a refresh that rebuilds only when its test asks; with a pruned distance, keeps a dual list (Section 4.5). A refresh that repeats the last one of its structure, as that of the potential of an output at the state of a step, is removed, so an output reads the structure of the steps and cannot change when it is built (D237) |
| `md-exec-expose-validity` | Writes the test of validity (Section 4.1) as a loop over particles with an `i1` reduction |
| `md-exec-fuse-loops` | Fuses loops over pairs over the same structure, positions, cutoff, and policy; fuses chains of loops over particles, such as a kick, the drift that follows, and the test, but for a device not the loop that closes one run of loops over disjoint tuples with the one that opens the next (Section 8.2); bitwise identical |
| `md-exec-accumulate-destinations` | Lets the loops whose forces a loop over particles adds accumulate into one destination, in the order of the sum; a loop that stands before the loop of the term before it in the sum moves down to follow it, so that a step that computes energies adds the forces as a step that does not (Section 7.1) |
| `md-exec-narrow-sums` | Sums only the elements of a vector sum that are used, such as the trace of the virial |
| `md-exec-simplify-distance` | Writes pair kernels in powers of $r^2$, and outlines functions of the distance alone (Section 7.2) |
| `md-exec-fold-tables` | Computes what a kernel derives from tables of types, such as $4\varepsilon\sigma^{12}$, once per entry, in f64, into packed tables |
| `md-exec-choose-neighbors` | On a device, chooses groups of 16 and each pair once where the contracts allow it (D89) |
| `md-exec-assign-precision` | Assigns a type to every field and kernel by its role (below) |
| `md-exec-approximate`, `md-exec-expand-radial` | Approximations of f32 kernels of terms, with stated bounds (Section 7.2) |
| `md-exec-choose-scatter` | On a device, chooses the strategy of each scatter-accumulate, `#md_exec.scatter`: tiles of block memory for the spreading of PME in f32 on a grid of at least 500,000 points, `scatter(tile(8))` on `md_exec.reciprocal` (D282), and a table of the particles of a block for the loops over tuples evaluated once, `scatter(hash(S, P, L))` on `md_exec.tuple_for` (D283); outside the deterministic mode, each where its preconditions hold (Section 8.5, D285) |
| `md-exec-assign-storage` | Turns the value form into the storage form: every field gets a buffer with its row set in an identity layout (D337), loops update buffers in place, scratch comes from a pool keyed by set and type; never copies a field, and fails with the reason where it would have to; never takes a buffer that the entry declares read-only |
| `check-read-only-arguments` | Where the lowering to loops or to a device ends, proves that the code writes no buffer that the entry declares read-only, the model, or fails the compile naming the use (D258) |
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

**Set identity in storage** (D337). Storage assignment
types a field with the set of its rows, for example
`memref<?x3xf32, #md_exec.set_layout<@atoms, 2>>`; loop-reduction scratch and the members
of tuples carry their corresponding particle and tuple sets; incidence
buffers carry the particle set of their rows; their tuple set remains an
explicit `md_exec.tuple_set` attribute on the builder and tuple loop. The layout
maps each index to itself and the numeric memory space remains separate.
The unit module marker `md_exec.typed_storage` declares this form; a
single checked dialect helper writes, queries, and removes it
(D349).
The verifier compares a loop's buffers with its particle or tuple domain,
and rejects a buffer from another set. The current pool can reuse a buffer
only for the same set and full type. Before either CPU or GPU lowering a
shared boundary rejects a cast across sets, including one hidden by a
plain cast or aliasing view, and rejects unknown source provenance unless the cast marks a declared input boundary. That boundary is a `memref.cast` with the unit marker `md_exec.input_set_boundary` (D345); the dialect verifies placement, value, and provenance. It erases the identity layout and boundary casts. Therefore
the address calculations, allocation sizes, and kernels are those of the
plain-memref program. Dynamic row counts come from validated input and
the assignment pass's size map; PME scratch remains untagged structural storage, and lifetime planning is a later pass.

The execution loop interface (D342) derives a record
for each access: the exact SSA operand, its read/write or accumulation
effect, its row scope, and its symbolic set. The scopes
distinguish a particle's own row, a pair neighbor, tuple members, indexed
destinations, and a whole buffer. A reduction also names its existing
algebra. An absent set or algebra, or an opaque kernel effect, leaves the
record incomplete. This interface alone licenses no reordering: fusion,
streams, and distributed scheduling still apply their previous checks.

## 3.5 Lowerings and templates

`convert-md-exec-to-loops` lowers the storage form to `scf` loops for the
CPU, with `scf.parallel` where iterations are independent; with more than
one thread, `convert-scf-to-openmp` makes them OpenMP. A global sum is
the reduction of its `scf.parallel`, which OpenMP combines from partial
sums per thread; the variable of each reduction is an alloca, which
`hoist-static-allocas` moves to the entry of the function so that the loop
of the steps does not grow the stack (D117). The pass runs again where the
lowering ends, for every program, on the rows that the entry allocates
where it writes them (the observed columns, pulling, the free energy): an
allocation in a loop is given back only when its function returns, and the
entry of a simulation does not return (Section 3.6). The pass cannot move
an allocation whose size is not a constant; no program has one, and the
lowering of every program, whose rows stand in loops over steps or frames,
refuses a module that would keep one outside the first block of a
function, so that the absence is checked and not assumed. `convert-md-exec-to-gpu`
releases the stack of each iteration of a loop of the host instead, and lowers
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
runs LLVM's module pipeline at O3 on the host code, inlining the helpers
of the templates and keeping their arrays in registers (D269), and
then its code generator at the default level
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
compilation of the PTX: about 7 s at constant energy and 13 s at
constant pressure for the systems of the Amber suite, whatever their size
(Table 10.4). The module of JAC at constant pressure has 9,236 lines as
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
linked and checked as a generated object is. The directory is bounded,
and the entries used least recently are evicted; its total is kept in a
file that stores update under a lock, so that a store costs a fixed
number of file operations and the directory is listed only to evict
(D234). The passes, which do depend
on MDIR, run for every program that is compiled. Within a process a
`Program` keeps the LLVM module and the host object of the program of
segments of its first simulation, under a hash of the module before the
passes, of the pipeline, and of the machine, and its later simulations
link a copy without the passes and without code generation
(D236); the module is kept beside the object because the
engine reads from it the constructors that load the kernels of a GPU. The PTX and the cubin of each GPU module are
cached in the same way. The PTX is keyed by a hash of the module's IR,
which then holds only upstream LLVM and NVVM operations, together with its
target, libdevice, and the LLVM version. The cubin is keyed by a hash of
its PTX and the version and arguments of `ptxas`. A hit needs the same
module, so the values that depend on the state a run starts from, such as
the tilts of the cell, the cell of the restraints' reference positions,
and the constant virial of the dispersion correction and of the PME
background times the volume (a constant, since both terms scale as $1/V$),
are arguments of the entry rather than
constants of the text (D227). A stage that continues
from an equilibrated cell then reuses the program of another such stage;
only the grid of PME and the neighbor capacity, which shape loops and
buffers, stay in the key. The capacity is estimated from the most
neighbors of a particle at the start and rounded up to four significant
bits, so that it too is the same for such stages; it only sets the room
at first and does not change the results. From Python, a compile
can bypass the cache, and the cache can be cleared while other processes
use it (D217): entries are written whole by a rename
and read by a copy, so a clear can cause a miss but never a partial entry.

The parts of a Python simulation run in one activation of its entry
(D215, `docs/python-segments.md`). The program wraps its loops
of steps in a loop over parts, and at the end of each part it calls the
host with its positions, velocities, and forces where they are, addresses
of the device on a GPU (`mdrt.host_call` with `in_place`), and waits there,
on a stack of its own, until the host gives it the next part. Its buffers,
its order of the particles, and its neighbor structures therefore persist
from part to part: a run in parts is the run in one part, to the bit in the
deterministic mode, and a part of 10 steps of JAC no longer copies 50 MB to
the device. The runtimes record what an activation allocates, which is
freed when it ends; the state at the end of each part is copied on the
device, so that a part that fails returns to it (D196). One
thing is the simulation's and outlives its activations
(D253): the members of the tuples of the
topology renumbered to the places of the particles, and the incidence
structure of each tuple set (Section 3.4). Both are functions of the
members in the order of the files and of the identity of the particle at
each place, and of nothing else; the runtime keeps a copy of these two
inputs with them, and an activation that begins takes the kept structures
only where the inputs it has are equal to the copies, element for element.
That holds at every start at which the particles have not moved (an
update of the tunables, an evaluation, a commit of velocities or
parameters, the start after a checkpoint), which then uploads nothing of
the topology; a start in another order builds the structures in the
memory of the last.

What a start gives the entry is the model and the state
(D258). The entry declares the buffers of the
model read-only: the masses, the fields of the particles, the tables, and
the members and fields of the tuple sets. `md-exec-assign-storage` takes
none of them for another field or for a copy back from a device, and
where the lowering ends `check-read-only-arguments` follows every use of
each and fails the compile on any that it cannot prove a read. A program
that compiles is thus one that writes none of them, and a start hands it
the arrays of the program where they are, in place of copies; a conversion
into the type of the entry is kept and taken only where the start is given
the same array, which programs share and never write (D256). The state,
in which the program on the CPU works, is written anew at every start into
buffers that the simulation keeps.

`mdir run` uses the same store for its one call of the entry
(D255). Its program orders the particles where the run
begins and where each segment of checkpoints begins, so that every segment
asks for the structures of its own order. The argument above decides what
can be taken. The structures are functions of the members in the order of
the files and of the identities alone; the first segment follows the
start with no step between them, so it computes the order of the same
positions and its identities equal those of the start, element for
element: its structures are taken. After a segment of steps the positions
have moved, and an order equal to the one before is not excluded but is
not assumed either: the comparison finds the identities different where
they are, and the segment builds, in the memory that the segment before
gave back. A run therefore does the work it did before, less one build,
in memory taken once.

On a device the build itself moves there (D257). In the
notation of Section 2 ($\pi$, $j_{t,k}$, $n_\mathrm{s}$, $o_p$), the row of
place $p$ of the incidence structure holds every pair $(t, k)$ of a tuple
and a slot with $j'_{t,k} = p$, in the order of $t$, and of $k$ within a
tuple; since $j'_{t,k} = \pi^{-1}(j_{t,k})$, these are the pairs with
$j_{t,k} = \pi(p)$, the row of particle $\pi(p)$ in the order of the
files, in the same order. The rows are ragged (D314): one
buffer holds the offsets $o_0 = 0 \le o_1 \le \dots \le o_N$ and the
entries, row $p$ being the entries $o_p$ to $o_{p+1}$, so the structure
takes the entries it has rather than $N$ rows as long as the longest. A
start in a new order writes $\pi^{-1}$ from $\pi$ and the renumbered
members $j'$, then builds the rows from $j'$ by distribution counting: the
count of each place, an exclusive scan of the counts into the offsets, and
a scatter of each membership $n_\mathrm{s} t + k$ to the next slot of its
place, with an atomic. The atomics leave the slots of a row in the order in
which threads ran, so each row is then sorted by $n_\mathrm{s} t + k$;
for $0 \le k < n_\mathrm{s}$ that number orders the pairs by $t$ and then
by $k$, the order above, and the structure equals the host's element for
element whatever the order of the threads. The identities are compared on
the device with those of the last order, and only a flag reaches the
host.

A consumer such as PyTorch takes those buffers through DLPack without a
copy (D220, `docs/python-dlpack.md`): a view of
the positions, velocities, and forces in the order of the program and the
types it stores, the index in the input of each row, and the values of
the tunables. The buffers change at the next part, so a view is a lease:
while it, or any tensor taken from it, is alive, the simulation refuses
to run or to take new values. Each exported tensor holds the native
simulation, so its storage outlives the Python object, and its deleter may
run on any thread. On a device the consumer's stream waits for an event
recorded on the simulation's stream, and the first part after views were
taken waits for all the work of the context, the consumer's included,
before it writes the buffers.

A consumer that writes takes a *borrow* instead
(D229): the same buffers of the positions and the
velocities without the read-only flag, and buffers of the host with the
diagonal of the cell, the tilts of a triclinic cell (D238), and
the values of the tunables. A borrow excludes every
other operation on the state, views included. Its commit is explicit: it
checks what was written, refuses a change of the tunables that would
change the program as an update does, advances the version counters of
the changed fields (P16), and begins a new activation of the entry from
the committed state, so that the order of the particles, the neighbor
structures, and the forces are those of that state and nothing computed
from the old one is carried; a run after it is, to the bit in the
deterministic mode, that of a simulation compiled from the committed
state. A committed cell is checked as a compile checks it, and for a
triclinic cell it must be reduced and stay triclinic; its tilts are
values that the entry takes when an activation begins (D227), so a commit
of them builds no program. A borrow that ends without a commit is undone
from the copy of the
state that the simulation holds since the last part. A commit costs an
evaluation, so a borrow serves changes between runs, not at every step.

The runtime is small and holds what cannot be IR: 845 lines of C for the
host (`runtime/mdrt.c`: counters of builds and prunings, the stop on a
position that is not a number, the generator Philox 4×32-10
[[Salmon2011]](references.md#salmon2011) and the factors of the thermostat and the barostat,
isotropic or semi-isotropic, drawn from it, the means of the pressures by
axis that the semi-isotropic barostat took, the FFT of the host
[[Reinecke2019]](references.md#reinecke2019), and the matrix of the host),
419 for the structures of the tuples that a simulation keeps across its
activations, shared by both (`runtime/mdrt_tuples.h`),
and 1518 lines for NVIDIA devices (`runtime/mdrt_cuda.c`, on the driver
API only: loading of modules, launches that skip empty grids, one stream
in order plus an optional second one, words of memory of the host mapped
for the device for flags (D118), a caching allocator whose frees do not
wait, plans of cuFFT, growable buffers for neighbor structures, and the
records of what each activation of a Python simulation allocated). The
driver registers callbacks for the log, the trajectory (DCD, XTC, or
H5MD without loss; below), and
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
(D173). `mdir run` writes it on a thread of its own while the steps go
on, from a copy of the state, and the next checkpoint, a stop, and the end
of the run wait for it (D336).

A run is a realization of a Markov chain on the state $\mathbf z_n$, each
step drawn from the kernel $\mathcal K_{U,\mathcal C}$: the integrator and
the potential $U$ map $\mathbf z_n$ to $\mathbf z_{n+1}$, and the coupling
$\mathcal C$ draws its random numbers as a function of $\omega$ and of the
absolute step $n$ alone (A13). Given $\mathbf z_n$, $n$, $\omega$, $U$, and
$\mathcal C$, the steps that follow are therefore determined, up to the
order of the sums, which the deterministic mode fixes. This is why a
checkpoint stores the step and every value in 64 bits, and why the same
run may continue in another process, or in another front end, to the bit.
The forces in $\mathbf z_n$ are $-\partial U/\partial\mathbf x$ of the run
that wrote them, and the energy that the coupling has taken is a sum over
its steps. A stage of another $U'$ or $\mathcal C'$ samples another
stationary distribution (at constant temperature the canonical
$Z'^{-1}e^{-U'/k_BT'}$) with $\mathbf z_n$ as its initial condition, and
must evaluate $-\partial U'/\partial\mathbf x$ rather than reuse the forces
of $U$; the fingerprint identifies $U$ and $\mathcal C$ by what defined
them. A Python simulation (D223) writes and reads the same
file. The fingerprint of a Python model holds the entries that the control
file of the same model writes, for a setting given explicitly; the
defaults of the Python model are those of the control file
(D248), so a setting left out is the same in both, and a
Python script and a control file of the same model continue each other's
run, and a change of either is named. The coupling $\mathcal C$ of a
Python model is any that the control file states: stochastic velocity
rescaling, Langevin dynamics, a Nosé–Hoover chain, whose state the
checkpoint holds, or Brownian dynamics (D249). Tunable parameters, the
front end, and hashes of the model and the plan are additional entries of
format 1 with a hash of their own. A simulation that writes a checkpoint
goes on from the state written, with new neighbor structures, as `mdir run`
builds them at each checkpoint, so that going on and continuing the file
are the same steps.

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
same buffers, and the same trajectory to the bit (Section 9). The model
that takes the values is a copy that the simulation keeps for its updates
and writes in place (D254): the sites written are fixed by the
maps $\iota$, not by the values, and each is written from
$\boldsymbol\theta$ alone, so the copy after an update does not depend on
the updates before it.

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
pairs that the model sets apart from the rule (NBFIX), which keep theirs.
The table itself may be tunable by pairs of types (D226):
its sites are the unordered pairs $(a, b)$, $a\le b$, so that
$\sigma_{ab} = \sigma_{ba} = \theta_{\iota(s(a,b))}$ with one entry per pair,
set after the rule for the pairs that $\iota$ takes. Since $U$ depends on
the table only through $\sigma_{ab}$ and $\epsilon_{ab}$ of unordered pairs,
$\partial U/\partial\theta_k = \sum_{s:\iota(s)=k}\partial U/\partial\sigma_{s}$
counts each pair once, which a site per ordered pair would count twice.
$\langle C_6\rangle$ of $E_\text{disp}$ follows from the table; the tails
$I_{ij}$ of pair terms are integrated anew (Section 5.4).

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

*Which frames.* The weights take $U$ at the configurations that the
dynamics visited. A trajectory format that rounds every coordinate to a
grid of spacing $q$ stores $\tilde S_n = S_n + \boldsymbol\delta$, the
components of $\boldsymbol\delta$ close to independent and uniform on
$[-q/2, q/2)$, with mean 0 and variance $q^2/12$. To second order in
$\boldsymbol\delta$, with $\mathbf F = -\nabla U$ and $\mathsf H$ the matrix of
the second derivatives of $U$ at $S_n$,

$$
U(\tilde S_n) - U(S_n) = -\mathbf F\cdot\boldsymbol\delta
+ \tfrac12\boldsymbol\delta^{\mathsf T}\mathsf H\boldsymbol\delta,\qquad
\big\langle U(\tilde S_n) - U(S_n)\big\rangle_{\boldsymbol\delta}
= \frac{q^2}{24}\mathrm{tr}\,\mathsf H
= \frac{q^2}{24}\sum_i\nabla_i^2U .
$$

The first term has mean 0 and a standard deviation of
$q\lvert\mathbf F\rvert/\sqrt{12}$, noise that differs from frame to frame;
the second is a rise that every frame shares and that no average over
frames removes, since the stiff terms make the trace positive: a bond
$\tfrac k2(r - r_0)^2$ adds $2k[1 + 2(r - r_0)/r]$ to it, $2k$ at its
length, so the rounding raises its energy by $kq^2/12$ in the mean. For
the alanine dipeptide in 382 flexible waters, 1167 bonds with
$\sum k = 5.37\times10^8$ kJ/mol/nm$^2$, the bonds alone give 44.8 kJ/mol
at the $q = 10^{-3}$ nm of XTC; the rise measured for #251 is
$48\pm6$ kJ/mol, of the order of $20\,k_BT$ at 300 K, in a quantity whose
differences enter an exponential. In $\Delta U$ both energies are of the
same $\tilde S_n$ and the rise cancels where $\boldsymbol\theta$ leaves
$\mathsf H$ alone, but not for the parameters of the stiff terms, and not
against an energy that the run recorded at $S_n$. Positions in f32 have
$q$ of $2^{-23}$ times the power of two below $\lvert x\rvert$,
$2.4\times10^{-7}$ nm for a coordinate between 2 and 4 nm:
the rise falls by $(q/10^{-3}\,\text{nm})^2$ to below $10^{-5}$ kJ/mol and
the noise of the first term stays, at most $1.7\times10^{-3}$ kJ/mol on
that system; f64 is the type of the positions of the state in the mixed
and the double mode, so nothing is rounded. DCD holds f32 in Å, the
rounding of a product and not of the state in nm. MDIR therefore writes a
third trajectory format, H5MD 1.1
[[deBuyl2014]](references.md#debuyl2014), with the positions of the state
in f64 or f32 in nm, the cell of every frame, and, from a Python
simulation, the velocities, the forces, the potential energy of the step,
and the version of $\hat{\boldsymbol\theta}$ (D239,
`docs/python-h5md.md`; Appendix A.6). The host function that takes the
frames of DCD and XTC is handed the buffer of the state before their
conversion, so such a frame costs a copy and a write and the compiled
program is the one that writes a DCD.

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

### The derivative in the tunable parameters

Reweighting needs more than $U_{\boldsymbol\theta}$ at the frames: the
gradient of a loss on $\langle O\rangle_{\boldsymbol\theta}$ is, from the
weights above,

$$
\frac{\partial\langle O\rangle_{\boldsymbol\theta}}{\partial\theta_m}
= \Big\langle\frac{\partial O}{\partial\theta_m}\Big\rangle_{\boldsymbol\theta}
- \frac1{k_BT}\Big(\Big\langle O\frac{\partial U}{\partial\theta_m}\Big\rangle_{\boldsymbol\theta}
- \langle O\rangle_{\boldsymbol\theta}\Big\langle\frac{\partial U}{\partial\theta_m}\Big\rangle_{\boldsymbol\theta}\Big),
$$

the derivative of $w_n\propto e^{-\Delta U(S_n)/k_BT}$ normalized, so the
one quantity that MDIR must add is $\partial U/\partial\boldsymbol\theta$ at
a configuration (D230, `docs/python-gradient.md`). A
program compiled for it evaluates it at the state of a simulation when a
script asks, and its steps are those of a program without it.

*Which $U$.* The weights are ratios of the densities that the frames were
sampled from and are reweighted to. A run that cuts a pair term at $r_c$
samples $e^{-U_\text{shift}/k_BT}$, not the cut energy that its log reports
(Section 6.8, D210): its forces are $-\nabla U_\text{shift}$, and the
impulse at $r_c$ that the cut energy implies is never applied. So the
derivative is that of $U_\text{shift}$ with the estimate $E_\text{sh}$ of
the shift, as $\partial F/\partial\xi = \langle\partial U_\text{shift}/
\partial\xi\rangle$ is for an observed constant; the two differ by
$\sum_{r<r_c}\partial u(r_c)/\partial\theta$, which fluctuates with
$N_\text{in}$.

*The chain rule.* The program does not read $\boldsymbol\theta$. It reads
values $b_s$ that the host computes from it (the charges, the table
$\sigma_{ab}$, $\epsilon_{ab}$, the parameters of tuples), and the host
adds constants $c(\boldsymbol\theta)$ to the energy. Hence

$$
\frac{\partial U}{\partial\theta_m}
= \sum_s\frac{\partial U}{\partial b_s}\frac{\partial b_s}{\partial\theta_m}
+ \frac{\partial c}{\partial\theta_m},
$$

and the two factors are computed where the two maps are: $\partial
U/\partial b_s$ by the differentiation of the program, the Jacobian
$\partial b_s/\partial\theta_m$ and $\partial c/\partial\theta_m$ on the
host, beside the code that computes $\mathbf b$ and $c$. The site
derivatives are linear in a weight of the frame, so that a sum over frames
$\sum_ng_n\,\partial U(S_n)/\partial\mathbf b$ can be accumulated before the
chain rule is applied once.

*A field of the particles.* Differentiation (Section 3.3) gains the
derivative of a potential in an argument that is a field $a$ of the
particles, the field $g_i = \partial U/\partial a_i$. A sum over the
unordered pairs of a relation, $S = \sum_{\{i,j\}}k(i,j)$ with $k$
symmetric under exchange, has

$$
\frac{\partial S}{\partial a_i} = \sum_{j:(i,j)\in D(R)}\partial_1k(i,j),
$$

a gather over the directed expansion whose kernel is the derivative of $k$
in the value of its first particle: the derivative in the value of the
second is, by the symmetry, the same kernel with the particles swapped,
which the directed expansion visits at $j$. The kernel is not symmetric
itself ($\partial_1(q_iq_j/r) = q_j/r$), so the gather carries no exchange
contract and each particle of a pair takes its own value. A sum over
tuples gives a gather over them that yields $\partial k/\partial a_{t[s]}$
to the member at each place $s$; a sum over particles a map. Through the
energy, each takes the weight $\partial U/\partial S$ of its sum. The
three outcomes of a parameter (Section 6.8) hold: a field that no sum of
the energy takes has the derivative zero, a sum has its rule, and any
other use is an error that names the op.

*Several arguments.* One evaluation asks the derivative in several
arguments, numbers $c$ and fields $a$ alike. $\partial U/\partial c$ and
$\partial U/\partial a_i$ are first derivatives of the one energy, each
with the other arguments held fixed, and neither enters the other. The
sum that the derivative in $c$ adds, $\sum_{\{i,j\}}\partial k/\partial
c$, gathers $a$ wherever $k$ does, but it is not a term of $U$: its
derivative in $a_i$ would be a part of $\partial^2U/\partial
c\,\partial a_i$, which nothing asks for. The differentiation in each
argument therefore follows the terms of the potential alone, and the
derivative in one tunable is the same whether another is declared or not
(D230; to the bit in the deterministic mode, Section 9).

*Seeds.* The loops over pairs and tuples accumulate per particle, as they
do for the forces; a derivative in an entry $\sigma_{ab}$ of a table or in
the parameter of a tuple is a sum by pairs of types or by tuples, which no
loop accumulates. Such a derivative is made to leave as a field of the
particles. The potential that is differentiated takes a *seed*, a field
$d$ of zeros, and reads

$$
\sigma_{ij} = \sigma_{ab} + d_i\,J_{ab} + d_j\,J_{ba}
$$

for particles $i$, $j$ of the types $a$, $b$, with weights $J$ by
ordered pairs of types. Its value is that of the potential, since $d = 0$,
and

$$
\frac{\partial U}{\partial d_i} = \sum_j\frac{\partial u_{ij}}{\partial\sigma_{ab}}\,J_{ab},
\qquad
\sum_{i\in a}\frac{\partial U}{\partial d_i}
= \sum_bJ_{ab}\,(1+\delta_{ab})\frac{\partial U}{\partial\sigma_{ab}},
$$

a pair of two particles of one type adding at both. With the Jacobian of
the combining rule in its first argument as weights, $J_{ab} =
\partial\sigma_{ab}/\partial\sigma_a$ ($\tfrac12$ for Lorentz,
$\tfrac12\sqrt{\sigma_b/\sigma_a}$ for the geometric mean,
$\tfrac12\sqrt{\epsilon_b/\epsilon_a}$ for Berthelot, 0 for a pair that the
rule does not give), the sum over the particles of the type $a$ is
$\partial U/\partial\sigma_a$: the entry $(a,b)$ follows $\sigma_a$ through
its first argument and $(b,a)$ through its second, and for $a = b$ the
two halves add to $\partial\sigma_{aa}/\partial\sigma_a = 1$. For a
tunable of the table by pairs, $J_{ab} = 1$ for one pair in the row
$a$ gives $(1+\delta_{ab})\,\partial U/\partial\sigma_{ab}$ of that pair;
pairs that share a type take different seeds, so that the columns of the
Jacobian that one seed holds share no row
[[CurtisPowellReid1974]](references.md#curtispowellreid1974). The
parameter $p_t$ of a tuple is read as $p_t + d_{t[s]}$ at one member, and
$\partial U/\partial d_i$ at that member is $\partial u_t/\partial p_t$,
tuples that share the member taking different seeds. The charges need
none: they are a field already, once the pairs three bonds apart form
$fs_Cq_iq_j$ from the charges of their members and not from a product that
the host made.

*What the host adds.* The correction for the dispersion and the estimate of
its shift are $K\sum_{ab}n_{ab}\,4\epsilon_{ab}\sigma_{ab}^6$ over the
ordered pairs of types (Section 5.4), with the derivatives
$G_{ab} = Kn_{ab}\,24\epsilon_{ab}\sigma_{ab}^5$ and $Kn_{ab}\,4\sigma_{ab}^6$
in the entries; the row $a$ of a seed takes $\sum_b(G_{ab} +
G_{ba})\,J_{ab}$, the chain rule with the weights of the kernel. The
tail $I_{ij}$ of a pair term and the estimate of its shift are integrals
that a quadrature computes, and their derivative in a constant of the term
is the Richardson-extrapolated central difference of that quadrature
(Section 5.4), exactly 0 where the expression does not read the constant.

*The reciprocal sum in the charges.* The reciprocal sum of particle mesh
Ewald (Section 5) is a quadratic form of the charges, $E_\text{rec} =
\tfrac12\mathbf q^{\mathsf T}\mathsf A\mathbf q$ with $\mathsf A$ symmetric:
the charges are spread to the grid with the weights $\theta_i(\mathbf k)$
of their B-splines, the grid is multiplied by the influence function in
reciprocal space, and the result $\phi$ is read back with the same
weights. Its derivative is therefore linear in the charges,

$$
\frac{\partial E_\text{rec}}{\partial q_i} = (\mathsf A\mathbf q)_i
= \sum_{\mathbf k}\theta_i(\mathbf k)\,\phi(\mathbf k),
\qquad E_\text{rec} = \frac12\sum_iq_i\frac{\partial E_\text{rec}}{\partial q_i},
$$

the potential of the grid at the particle: the sum of the force,
$\mathbf F_i = -q_i\sum_{\mathbf k}\phi(\mathbf k)\nabla_i\theta_i(\mathbf
k)$, with the weights in place of their gradient and without the charge
(D231). The op of the reciprocal sum yields it as a
fourth result where the differentiation in the field of its charges asks,
gathered by a kernel of its own, in f64 in every precision mode; a program
that does not ask has neither the result nor the kernel. For a scalar that
the charges depend on, $\partial E_\text{rec}/\partial\lambda =
\boldsymbol\delta^{\mathsf T}\mathsf A\mathbf q$ with $\boldsymbol\delta =
\partial\mathbf q/\partial\lambda$ is one number, which two more reciprocal
sums give (Section 6.8); $N$ charges need the gather. The host adds what
it adds to the energy: from $E_\text{self}$ and $E_\text{bg}$ above,

$$
\frac{\partial E_\text{self}}{\partial q_i} = -\frac{2f\beta}{\sqrt\pi}q_i,
\qquad
\frac{\partial E_\text{bg}}{\partial q_i} = -\frac{f\pi Q}{V\beta^2},
$$

the second proportional to $1/V$. The direct sum, shifted to 0 at the
cutoff, the excluded pairs $-fq_iq_j\mathrm{erf}(\beta r)/r$, and the
pairs three bonds apart are sums that gather the charges, differentiated
as fields.

### The energies of stored frames and their product

A fit evaluates $U_{\boldsymbol\theta}$ at the frames again for every
$\boldsymbol\theta$ that it tries, while the frames stay those sampled at
$\hat{\boldsymbol\theta}$. The frame evaluator
(D240, `docs/python-frames.md`) does this in a simulation of
its own, apart from the one that samples, whose program is a loop over
frames (D251): everything but the positions and the cell
is put where the program keeps it once, and each frame, given in the order
of the input with its cell, is evaluated with its neighbor structures built
anew, since a stored frame is not near the one before it. The particles
stay in the order of the input: an order chosen for a frame would have the
tuples of the bonded terms renumbered at that frame, and the neighbor
structures bin the particles themselves, so the kernels gain nothing from
it. $U_{\boldsymbol\theta}(S_n)$ is then a function of the frame alone,
with the same bits wherever the frame stands in a call, which is what makes
the weights uniform to the bit at $\hat{\boldsymbol\theta}$; against the
sampler, whose sums run in its own order, it agrees to the rounding of
those sums (Section 9). What the host adds to the energy
follows the volume of each frame (the tails and $E_\text{disp}$,
$E_\text{bg}$, all $\propto 1/V$; Section 5.4), so frames of a run at
constant pressure are evaluated each at its own volume.

*The product.* A framework differentiates a loss $L$ of the reweighted
averages backwards: it first needs every $U_{\boldsymbol\theta}(S_n)$,
forms the weights, and only then holds the cotangent of each energy. From
$\partial w_m/\partial U(S_n) = -w_n(\delta_{mn} - w_m)/k_BT$,

$$
g_n = \frac{\partial L}{\partial U(S_n)}
= -\frac{1}{k_BT}\,\frac{\partial L}{\partial\langle O\rangle_{\boldsymbol\theta}}\,
w_n\big(O(S_n) - \langle O\rangle_{\boldsymbol\theta}\big),
\qquad
\frac{\partial L}{\partial\theta_m}
= \sum_n g_n\,\frac{\partial U(S_n)}{\partial\theta_m},
$$

which is the covariance of the gradient above times $\partial L/\partial
\langle O\rangle$, for an $O$ that does not read $\boldsymbol\theta$. So
the evaluator owes the framework a vector-Jacobian product, $\mathbf
g\mapsto\sum_n g_n\,\partial U(S_n)/\partial\boldsymbol\theta$, for a
$\mathbf g$ that does not exist during the pass that computes the
energies. It keeps the rows $\partial U(S_n)/\partial\boldsymbol\theta$
of that pass, 8 bytes for each frame and entry, while they fit a bound, and
the product is then a product of matrices on the host; above the bound it
evaluates the frames a second time with $\mathbf g$ in hand. The
derivative adds a tenth to an evaluation, so keeping the rows is nearly
free, and a fit of a few entries, the common case, never pays the second
pass.

*The reference.* The weights are ratios of two densities at the same
stored configuration, so $U_{\hat{\boldsymbol\theta}}(S_n)$ is evaluated
by the same evaluator at the same stored frame, not taken from the run. At
$\boldsymbol\theta = \hat{\boldsymbol\theta}$ the two energies are then
equal to the bit, $w_n = 1/N_S$ for $N_S$ frames, and the number of
effective frames is $N_S$. A file that rounds the positions changes the
sample, not the consistency of the weights; how much it may round is a
question of the potential and of the parameter (*Which frames*, above).
Positions in f32 change $U$ of the dipeptide in water by $2\times10^{-3}$
kJ/mol and $\Delta U$ by at most $10^{-4}$ kJ/mol for every tunable
measured. Positions rounded to $10^{-3}$ nm raise its energy by $50 \pm
12$ kJ/mol, but the rise is of the bonds, which a fit of a soft pair term
or of the charges leaves alone: with both energies at the same rounded
frame, $\Delta U$ is in error by 0.03 and 0.11 kJ/mol, the number of
effective frames is unchanged, and the derivative of a reweighted average
moves by less than 2% of its statistical error over 400 frames. For the
force constant of a harmonic term over the O-H bonds the error of
$\Delta U$ is 1.6 kJ/mol, $0.64\,k_BT$, and against a reference energy
recorded by the run the weights collapse to 9 effective frames of 400.
Frames in f64 are the default because they serve any term chosen later,
not because rounded frames fail in general.

*The gradients in the positions and the strain.* The forces and the
virial tensor of a frame are the two first derivatives of $U(S_n)$ in its
configuration, and the evaluator hands them over where they are asked
(D252). They are evaluated for that in a branch of the loop
over frames that a frame without the request does not take: the
evaluation of the energy reads only the trace of the virial and the
square of the forces, and a second reader of the whole of either changed
its kernels, 3% to 5% of every frame on the CPU, so a frame that asks
pays one more evaluation of its forces, a third of a frame on the CPU
and 13% to 37% on a GPU by the system and what is asked, and a frame
that does not pays nothing and has the same bits. The first is $\partial U/\partial\mathbf x_i = -\mathbf F_i$. For the
second, deform the positions and the cell together by a matrix
$\boldsymbol\varepsilon$, $\mathbf x_i \to (\mathsf 1 +
\boldsymbol\varepsilon)\mathbf x_i$ and each cell vector, a row of
$H$, alike. A displacement between two particles, or between a
particle and an image, $\mathbf d$, becomes $(\mathsf 1 +
\boldsymbol\varepsilon)\mathbf d$, so for an energy $u(\lVert\mathbf
d\rVert)$ of that displacement, whose force on its second end is $\mathbf K
= -u'\,\mathbf d/\lVert\mathbf d\rVert$,

$$
\frac{\partial u}{\partial\varepsilon_{ab}}
= u'\,\frac{d_a d_b}{\lVert\mathbf d\rVert} = -K_a d_b,
\qquad
\frac{\partial U}{\partial\varepsilon_{ab}}
= -\sum K_a d_b = -\mathsf W_{ba},
$$

with $\mathsf W = \sum\mathbf d\otimes\mathbf K$ the virial tensor of
Section 6.4; the same holds term by term for a term of several
displacements and for the reciprocal sum, whose energy is a function of
$\mathbf u_i$ and of the cell. This is why the virial is the quantity of
a pressure. In the canonical ensemble $Z \propto V^N\int
e^{-U(\mathbf u H)/k_BT}\,d\mathbf u$ over the coordinates in the
frame of the cell, and a strain changes $F = -k_BT\ln Z$ by

$$
\frac{\partial F}{\partial\varepsilon_{ab}}
= -Nk_BT\,\delta_{ab}
+ \Big\langle\frac{\partial U}{\partial\varepsilon_{ab}}\Big\rangle ,
$$

whose trace with $\partial V/\partial\varepsilon_{aa} = V$ is $-3PV$:
$P = (Nk_BT - \tfrac13\langle\mathrm{tr}\,\partial U/\partial
\boldsymbol\varepsilon\rangle)/V$, the pressure $(2K +
\mathrm{tr}\,\mathsf W)/3V$ of the log in the mean. A fit of a pressure
or of a stress differentiates exactly this output. What the host adds to
the energy at the volume of a frame is $c/V$ for a constant $c$, with
$\partial(c/V)/\partial\varepsilon_{ab} = -(c/V)\,\delta_{ab}$, a third of
its virial $3c/V$ on each axis. For the tail of a pair energy $u$ cut at
$r_c$ at a uniform density, $E_\text{tail} \propto V^{-1}\int_{r_c}^\infty
u\,r^2dr$ and the estimate of what the shift takes within the cutoff is
$E_\text{sh} \propto V^{-1}u(r_c)\,r_c^3/3$ with the same factor
(D210); the virial of the tail is that of $-\int_{r_c}^\infty
r\,u'\,r^2dr = u(r_c)\,r_c^3 + 3\int_{r_c}^\infty u\,r^2dr$, by parts,
which is $3(E_\text{tail} + E_\text{sh})$: the virial that the run adds
for the tail ($6E_\text{tail}$ for $r^{-6}$) is the derivative of the
energy of the evaluator, the potential shifted at the cutoff with both
estimates, and not of the cut energy with its tail alone. So
$-\mathrm{tr}\,\partial U/\partial\boldsymbol\varepsilon$ equals
the virial of the log in a program without constraints, to the rounding
of a sum of three numbers. With constraints the log adds the virial of
the constraint forces, which belongs to the pressure of the constrained
dynamics and is no derivative of $U$: the output leaves it out, and the
gradient in the positions is the plain gradient in every coordinate, since
a stored frame is taken as given and need not satisfy a constraint. A
virtual site is a function of its atoms, $\mathbf x_s = \mathbf
s(\mathbf x)$, and the energy is $U(\mathbf x, \mathbf s(\mathbf x))$: the
gradient in the row of the site is zero and its atoms take
$(\partial\mathbf s/\partial\mathbf x_i)^{\mathsf T}\partial
U/\partial\mathbf x_s$, the spread of Section 6.

The outputs are not symmetrized. For a potential that a rotation of the
positions and the cell together leaves unchanged,
$\partial U/\partial\boldsymbol\varepsilon$ is symmetric, since an
antisymmetric $\boldsymbol\varepsilon$ is a rotation to first order; a
term of the absolute positions has no such invariance, and its part
$\sum_i(\partial U/\partial x_{i,a})\,x_{i,b}$ is returned as it is. A
restraint to a reference position is the exception: its virial is taken
about the reference (Section 6.4), the derivative when the references
deform with the cell, as a pressure wants, and not the derivative of the
energy of a frame, whose references stay; the gradient in the strain is
refused for a program with restraints, not returned with that term. A
program takes a cell that is lower triangular, and $(\mathsf 1 +
\boldsymbol\varepsilon)$ of such a cell is one only for
$\boldsymbol\varepsilon$ on and above the diagonal; the components below
it are derivatives of the potential as a function of three free cell
vectors, which the pair sums and the reciprocal sum are, and are checked
by rotating the deformed frame back (Section 9).

A framework that differentiates in the cell holds the six numbers of
$H$ at fixed Cartesian positions, not a strain. The chain rule on
the deformation, $\partial U/\partial\varepsilon_{ab} =
\sum_i(\partial U/\partial x_{i,a})x_{i,b} + \sum_k(\partial U/\partial
H_{ka})H_{kb}$, gives

$$
\frac{\partial U}{\partial H}\bigg\rvert_{\mathbf x}
= H^{-\mathsf T}\Big(\frac{\partial U}{\partial\boldsymbol\varepsilon}
- \sum_i\frac{\partial U}{\partial\mathbf x_i}\otimes\mathbf x_i\Big)^{\mathsf T},
$$

whose entries on and below the diagonal use the strain only on and above
it. The two terms nearly cancel (for the dipeptide in water the result is
636 kJ/mol/nm where $\partial U/\partial\boldsymbol\varepsilon$ is
14,000 kJ/mol), so the sum over the particles is taken in f64 in every
mode. The backward of the adapter is these products with what the forward
pass kept; a second derivative, which a loss on the forces or the stress
needs for its gradient in $\boldsymbol\theta$, is not computed, and each
product is an operation whose own derivative is an error, so that the
missing term is never taken for zero.

*What is not differentiated.* The first term of the gradient,
$\langle\partial O/\partial\boldsymbol\theta\rangle$, is not zero for
an observable that reads the parameters at fixed positions: the virial,
or the force of a wall whose stiffness is fitted, for which it cancels the
covariance exactly in an ideal gas (#140). The evaluator returns such
outputs but does not yet differentiate them, and the three outcomes of
Section 6.8 extend to them: for each output it states which tunables it
may read; one that reads none is a constant by proof, and a backward
through one that may read a tunable is an error, never a zero. A second
derivative is an error likewise.

*The terms that depend on the parameters.* Write $U = U_\text{fixed} +
U_\text{dep}(\boldsymbol\theta)$, with $U_\text{dep}$ the terms that a
tunable enters. The weights take $\Delta U = U_{\boldsymbol\theta} -
U_{\hat{\boldsymbol\theta}}$ at one configuration and the gradient takes
$\partial U/\partial\boldsymbol\theta$, and $U_\text{fixed}$ is in
neither: $\Delta U = \Delta U_\text{dep}$ and $\partial U/\partial
\boldsymbol\theta = \partial U_\text{dep}/\partial\boldsymbol\theta$. An
evaluator may therefore be given a program that has only $U_\text{dep}$
(D243): the pair term of a tunable constant with its
tail, the Lennard-Jones for a tunable $\sigma$ or $\epsilon$, and for
tunable charges the electrostatics whole, since $E_\text{rec} =
\tfrac12\mathbf q^{\mathsf T}\mathsf A\mathbf q$ couples every charge to
every other through the mesh. A frame then costs 1.1 to 1.8 times less
(`docs/python-frames.md`). It saves time and leaves the accuracy in mixed precision
as it was: each term is accumulated in f64
from contributions in f32, the fixed terms give the same number at both
values of $\boldsymbol\theta$ and cancel exactly, and the error of
$\Delta U$ is that of the contributions of $U_\text{dep}$ either way. Where
$U_\text{dep} = \sum_k\theta_kA_k(\mathbf x)$ is linear, $\Delta U =
\sum_k(\theta_k - \hat\theta_k)A_k$ and the sums $A_k = \partial
U/\partial\theta_k$, which the run can record at each frame (Section
6.8), reweight it without the coordinates, in those parameters and no
others.

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
| Storage | The buffers of fields, the scratch of reductions, the cells of results read by the host | `memref`s with set-bearing identity layouts after `md-exec-assign-storage`, plain `memref`s after CPU/GPU conversion | `lib/Dialect/MDExec/Transforms/AssignStorage.cpp`, `lib/Dialect/MDExec/StorageSet.cpp` |
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
