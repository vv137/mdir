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
into one `md.potential`. Below is the potential of the test of a mixture
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
itp, or PDB) and writes the module as text. What is fixed for the run is
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
(`lib/Runtime/Templates/`, 14,767 lines, of which the template of the
groups, 9,803 lines, is generated by a script). The lowering parses the
templates that a run uses into its module and calls them, so they are
compiled with the rest of the program and specialized by its constants:
the cutoff and the skin, the grid and the order of PME, the precision.

## 3.6 Compilation at run time, and the runtime

`mdir run` compiles before it runs. It builds the module, parses it, runs
the pipeline, and hands the result to MLIR's `ExecutionEngine`, which
compiles the host code with LLVM at its most aggressive level of
optimization and links it with the runtime. Kernels are embedded as PTX
and compiled by the CUDA driver for the device present when the module is
loaded (`cuModuleLoadDataEx`), so there is no step of `ptxas` and no
choice of architecture at build time. The log reports the time from the
parsed module to a callable entry point, which includes the passes, the
code generation of LLVM, the creation of the CUDA context, and the
compilation of the PTX: about 10 s at constant energy and 20 s at
constant pressure for the systems of the Amber suite, whatever their size
(Table 10.2). The module of JAC at constant pressure has 9,236 lines as
the driver writes it, and the lowering outlines 545 GPU modules from it.

The runtime is small and holds what cannot be IR: 441 lines of C for the
host (`runtime/mdrt.c`: counters of builds and prunings, the stop on a
position that is not a number, the generator Philox 4×32-10
[[Salmon2011]](references.md#salmon2011) and the factors of the thermostat and the barostat drawn
from it, the FFT of the host [[Reinecke2019]](references.md#reinecke2019), and the matrix of the host)
and 905 lines for NVIDIA devices (`runtime/mdrt_cuda.c`, on the driver
API only: loading of modules, launches that skip empty grids, one stream
in order plus an optional second one, slots of pinned memory for flags,
a caching allocator whose frees do not wait, plans of cuFFT, and growable
buffers for neighbor structures). The driver registers callbacks for the
log, the trajectory (DCD), and checkpoints (H5MD 1.1 [[deBuyl2014]](references.md#debuyl2014), all
values in 64 bits, from which a run continues bitwise).
