# ML2: ownership, halos, and their adjoints in the IR

Design of `D275`, stage 1 of distributed execution (M4; issue #31
with #35 and #38; the ruling on #377 of 2026-10-16). Stage 1 is the core
on which every later stage builds: ML3 (the Allegro model of ML1 on two
GPUs), PaiNN and MACE in stages on two GPUs (#377), distributed particle
mesh Ewald, and the classical track of #31 (topology across domains,
ensembles, the Amber suite on several GPUs). Sections 1 to 11 are the
design; Section 12 is what the first PR builds and its validation;
Section 13 records the maintainer's rulings; Section 14 is the second part,
`md-distribute` and the derivative rules of $G$ and $G^\mathsf T$
(D276); Section 15 is the third part, the derivative through
fields computed from the positions, EAM on domains (D279). Section 7.2 is
the path of a domain's work through `md_exec`, on the CPU and on a GPU
(D290).

## 1. Scope

Stage 1 (#31, latest comment): ownership and periodic images, halos
(owner-to-ghost copies $G$ and the reverse sum $G^\mathsf T$ of ghost
contributions at the owner), migration, coverage and freshness,
exactly-once accumulation, and inspectable schedules; validated with a
fixed two-domain Lennard-Jones system, then EAM; CPU ranks before GPUs.

The design follows the principles ([principles.md](principles.md)):

1. **Every fact a pass relies on is in the IR.** Ownership, halos, the
   version of a field that a ghost holds, whether a sum is complete, and
   whether a value is replicated or partial are types, operands, and
   verifiers, not conventions of a runtime.
2. **Every optimization states and checks its precondition.** The
   traversal that computes each pair once and returns the contributions
   of ghosts, and the one that computes every pair at both owners and
   returns nothing, are two realizations of one semantic op; each states
   what it needs, and the lowering checks it with the code the verifier
   uses.
3. **Every optimization has a switch and is compared.** One domain against
   several, and each traversal against the other, are tests.

What stage 1 is not: a planner, asynchronous overlap, MPI or GPU
transport, topology across domains, constraints, or particle mesh Ewald.
The vocabulary leaves room for them (Section 10).

## 2. What exists

| Layer | What it has today | What distribution needs from it |
|---|---|---|
| `md` (semantic) | Particle sets, fields `!md.field<@set, k x T>` of the whole set, relations `!md.relation`, sums and gathers over pairs and tuples with kernels, `md.evaluate`, and semantic differentiation (`md-differentiate`, D2, D182, D183), which already carries adjoints of intermediate fields (an EAM density is a gather, its adjoint a field) | Unchanged. A global program stays the specification and the one-domain oracle |
| `md_exec` (execution) | `pair_for` over a neighbor structure, with the minimum image of a cell, over all rows of the positions; `particle_for`; `tuple_for`; neighbor builds and validity (`ExposeValidity`, D-series of M0); storage assignment | A pair loop over owned centers with neighbors among owned and ghost rows, in coordinates that carry their images; built: `centers`, `images(explicit)`, and the policy `(directed, atomic)` (Section 7.2) |
| `mdrt` (runtime ABI) | Neighbor structures, cells, permutations; `!mdrt.event` declared without semantics | The transport ABI of Section 9 |
| `dyn` | Kick and drift, programs of one step | Migration between steps (Section 8) |
| Runtime | `libmdrt` (host), `libmdrt_cuda` | `mdrt_dist`, the in-process transport |

Before stage 1 nothing distributed was implemented. Architecture Section 8.1 sketched
`md_dist` with `forward_halo` and `reverse_accumulate`; the closed
prototype (#32, #34, #36, #39, #42) explored contracts, a C++ MPI
executable, a Cartesian grid, asynchronous halos, and migration outside
the IR. Ideas taken from it: owned centers distinct from the
owned-plus-ghost extent (#34), field state derived from producers rather
than from attributes (#32), and partial values that no kernel may read
(#32).

## 3. Notation

$D$ domains, numbered $d = 0, \dots, D-1$; the owner of particle $i$ is
$o(i)$, and $O_d = \{i : o(i) = d\}$ the particles that domain $d$ owns.
A *ghost* $g$ of domain $d$ is an image of a particle, its *source*
$s(g)$, displaced by the lattice vector $\mathbf n_g H$ (row vector
$\mathbf n_g \in \mathbb Z^3$, $H$ the matrix of the cell, Section 2 of the
paper); $\Gamma_d$ is the set of ghosts of $d$. The *local* rows of $d$ are
$O_d$ followed by $\Gamma_d$.

| Symbol | Meaning |
|---|---|
| $\mathcal L$ | A layout: the owner and the local index of every particle, and its global identity |
| $\Omega_d$ | The region of domain $d$: a box of the grid in fractional coordinates of the cell |
| $R$ | The reach of a halo: $\Gamma_d$ holds every image within $R$ of $\Omega_d$ |
| $G$ | Owner-to-ghost copy: $(G\mathbf a)_g = \mathbf a_{s(g)}$ |
| $G_{\mathbf x}$ | The copy of positions: $(G_{\mathbf x}\mathbf x)_g = \mathbf x_{s(g)} + \mathbf n_g H$ |
| $G^\mathsf T$ | Its transpose: $(G^\mathsf T \mathbf b)_i = \sum_d \sum_{g \in \Gamma_d,\ s(g) = i} \mathbf b_g$ |

$G$ is linear, and $G^\mathsf T$ is its adjoint for the inner products of
owned and ghost rows:

$$
\langle G\mathbf a, \mathbf b\rangle_\Gamma
= \sum_d \sum_{g \in \Gamma_d} \mathbf a_{s(g)}\cdot\mathbf b_g
= \sum_i \mathbf a_i \cdot \sum_{g : s(g) = i} \mathbf b_g
= \langle \mathbf a, G^\mathsf T\mathbf b\rangle_O .
$$

$G_{\mathbf x}$ is affine: its derivative in $\mathbf x$ is $G$, and its
derivative in the cell gives the term $\sum_g \mathbf n_g^\mathsf T \otimes
\bar{\mathbf x}_g$ of the derivative in $H$.

## 4. The IR: dialect `md_dist`

`md_dist` is a peer of `md_exec` (architecture A8): it says where values
live and what moves between domains, not how a loop traverses. Its
values are *global views*: one SSA value of a distributed field stands
for the rows of every domain, and every op is collective over the domains
of its decomposition. One program text serves every domain count and every
transport; the in-process executor runs each op for all domains, and an
MPI rank runs its own part of each op.

### 4.1 Decomposition

```mlir
md.particle_set @atoms
md_dist.decomposition @dd on(@atoms) grid([2, 1, 1])
```

A Cartesian grid of $D = \prod$ `grid` domains over the fractional
coordinates of the cell (Saunders's and Plimpton's spatial decomposition,
[[Plimpton1995]](references.md#plimpton1995), [[Saunders2018]](references.md#saunders2018)):
domain $(a, b, c)$ is
$\Omega = [a/g_a, (a+1)/g_a) \times [b/g_b, (b+1)/g_b) \times [c/g_c, (c+1)/g_c)$,
numbered $d = a + g_a(b + g_b c)$. A grid of 1 along an axis keeps that axis
whole in each domain; its periodic images are ghosts as well, so no
domain ever takes a minimum image (Section 4.3).

The grid is a property of the existing `md_dist.decomposition` semantic
primitive. `md-distribute` has two strategies for choosing its static
triple; neither changes the partition, halo, or transfer ops:

| Pass option | Preconditions | Choice and check |
|---|---|---|
| `grid=gx,gy,gz` | Three positive integers with a product representable in `i64` | Explicit triple; the baseline and override |
| `domains=N box=Lx,Ly,Lz` | Positive $N$; three positive finite edge lengths of an orthorhombic compile-time box | Among positive factor triples of $N$, minimize $g_x/L_x + g_y/L_y + g_z/L_z$; break equal-score ties lexicographically. Compare its emitted decomposition and results with the explicit triple. |

Exactly one strategy is given. The score is the common-factor-free surface
area of the domain boxes, a geometric estimate of exchange area rather
than a measurement of communication cost. The selected triple appears in
`md_dist.decomposition`; the potential's cell stays a runtime value and a
change to it does not reselect the static grid. This pass-level choice
precedes the future `[distribute]` control-file path of Section 13.

### 4.2 Layouts: ownership and identity

```mlir
%l = md_dist.partition %x, %cell
       : !md.field<@atoms, 3 x f64> -> !md_dist.layout<@dd>
```

A layout is an SSA value: each particle's owner (the domain whose region
holds its position, wrapped into the cell), its local index, and its
global identity (its place in the particle set). It never changes. A
migration yields a new layout (Section 8), so a layout value is an
*ownership epoch*, and two fields are of one epoch when they come from one
layout value.

```mlir
%xo = md_dist.distribute %x, %l image(%cell)
        : !md.field<@atoms, 3 x f64> -> !md_dist.field<@dd, owned, 3 x f64>
%f  = md_dist.collect %fo, %l
        : !md_dist.field<@dd, owned, 3 x f64> -> !md.field<@atoms, 3 x f64>
```

`distribute` gives each domain the rows of the particles it owns, in
ascending global identity; with `image`, positions are wrapped into the
cell, so that each owned position lies in its domain's region.
`collect` is its inverse, in the order of the particle set. Results and
outputs therefore stay in the order of the particle set, whatever the
number of domains.

### 4.3 Halos: coverage and images

```mlir
%h = md_dist.halo %l, %xo, %cell reach(2.0)
       : !md_dist.layout<@dd>, !md_dist.field<@dd, owned, 3 x f64>
       -> !md_dist.halo<@dd, 2.0>
```

For each domain, the ghosts: every image $\mathbf x_j + \mathbf n H$ of every
particle, other than the domain's own particles in the cell
($\mathbf n = 0$), whose distance from $\Omega_d$ is less than $R$. Each
ghost records its source (owner and local index) and its image
$\mathbf n$. The reach is in the type, so that a consumer checks coverage
statically (Section 4.9, V4).

*Coverage.* Let $R \ge r_c$, and let $i \in O_d$ and $j$ be a particle with
$r_{ij} < r_c$ in the minimum image, $\mathbf y = \mathbf x_i - \mathbf d_{ij}$
the image of $j$ nearest to $i$. Since $\mathbf x_i \in \Omega_d$, the
distance of $\mathbf y$ from $\Omega_d$ is at most $r_{ij} < R$, so
$\mathbf y$ is a ghost of $d$ or, if $\mathbf n = 0$ and $o(j) = d$, an owned
row. Every pair of the global relation is therefore within the local rows
of the owner of each of its particles. When $r_c$ is less than half the
least width of the cell, only one image of $j$ is within $r_c$ of $i$, and
the pair is found once at each end; this is the precondition of the
global `md.neighborhood` as well.

*Images.* Ghosts hold their images: positions are copied by $G_{\mathbf x}$,
displacements within a domain are plain differences, and no kernel takes a
minimum image. This is what a halo of $L\,r_c$ for a message-passing model
of $L$ layers needs (#377), where several images of one particle can be
within the reach; the copy of a feature, $G$, has no shift.

### 4.4 Fields: owned and ghost rows; complete and partial

```mlir
!md_dist.field<@dd, owned, 3 x f64>             // owned rows, complete
!md_dist.field<@dd, ghost, 3 x f64>             // ghost rows, a copy G a
!md_dist.field<@dd, owned, partial, 3 x f64>    // contributions to owned rows
!md_dist.field<@dd, ghost, partial, 3 x f64>    // contributions held at ghosts
!md_dist.partial<f64>                           // one partial sum per domain
```

The *role* (`owned`, `ghost`) says which rows a value has. *Partial* says
that a value is a set of contributions to a sum that is not complete yet:
the forces at the owned rows of a traversal that computes each pair once,
before the ghosts' contributions have come back. A partial value is never
read by a kernel, copied to ghosts, collected, or used twice; the only op
that consumes a partial field is `halo_sum`, and the only op that consumes
a partial number is `all_reduce` (V6, V7). This is how exactly-once
accumulation is a property of types and uses rather than of a runtime.

### 4.5 $G$ and $G^\mathsf T$

```mlir
%xg = md_dist.halo_copy %xo, %h image(%cell)      // G_x: positions, with images
%pg = md_dist.halo_copy %psi, %h                  // G: any other field
%f  = md_dist.halo_sum %fo, %fg, %h               // f = f_o + G^T f_g
```

`halo_copy` is $G$ (with `image`, $G_{\mathbf x}$). `halo_sum` is
$\mathbf a + G^\mathsf T\mathbf b$: an owned value plus the contributions that
its ghosts hold, summed at the owner. $G^\mathsf T$ alone is `halo_sum` of
`md_dist.zeros`. The pair is declared to the AD as an adjoint pair
(Section 11): the adjoint of `halo_copy` is `halo_sum`, and that of
`halo_sum` is `halo_copy` (for its ghost operand) and the identity (for its
owned operand).

### 4.6 Pairs: one relation, two traversals

```mlir
%n = md_dist.neighborhood %xo, %xg, %h cutoff(2.0) traversal(directed)
       : ... -> !md_dist.relation<@dd, directed>
```

The relation is the global relation of `md.neighborhood`, realized over
local rows. With the *directed* traversal, domain $d$ holds

$$
D_d = \{(i, k) : i \in O_d,\ k \in O_d \cup \Gamma_d,\ k \ne i,\ r_{ik} < r_c\},
$$

so each unordered global pair appears twice in $\bigcup_d D_d$, once at
the owner of each particle (Section 4.3). With the *unique* traversal it
holds $U_d = \{(i, k) \in D_d : \mathrm{id}(i) < \mathrm{id}(s(k))\}$, and each
pair appears once (the particle of the lesser identity takes it; a ghost's
identity is that of its source).

```mlir
%u = md_dist.sum_pairs %n, %xo, %xg exchange(symmetric) { ... }
       : ... -> !md_dist.partial<f64>
%f = md_dist.gather_pairs %n, %xo, %xg exchange(antisymmetric) { ... }
       : ... -> !md_dist.field<@dd, owned, 3 x f64>               // directed
%fo, %fg = md_dist.gather_pairs %n, %xo, %xg exchange(antisymmetric) { ... }
       : ... -> !md_dist.field<@dd, owned, partial, 3 x f64>,
                !md_dist.field<@dd, ghost, partial, 3 x f64>      // unique
```

The kernels are those of `md.sum_relation` and `md.gather_relation`: the
distance, the displacement $\mathbf d_{ik} = \mathbf x_i - \mathbf x_k$, and two
values of each gathered field, the center's and the neighbor's. Every
gathered field is given as its owned value and its ghost copy, as the
positions are. The semantics are those of the global ops:

| Op | Directed | Unique |
|---|---|---|
| `sum_pairs` | $S_d = \tfrac12\sum_{(i,k)\in D_d} k(i,k)$; needs `symmetric` | $S_d = \sum_{(i,k)\in U_d} k(i,k)$; needs `symmetric` |
| `gather_pairs` | $\mathbf a_i = \sum_{k : (i,k) \in D_{o(i)}} k(i,k)$, complete at the owner; any kernel | Owned partial $\mathbf p$ and ghost partial $\mathbf q$ with $\mathbf a = \mathbf p + G^\mathsf T\mathbf q$; the kernel at $k$ is $\sigma\,k(i,k)$, $\sigma = \pm1$ from `exchange` |

In both, $\sum_d S_d$ is the sum of `md.sum_relation` over the global
relation and $\mathbf a$ the field of `md.gather_relation`.

Over a directed relation, `gather_pairs` may also yield the two partial
results (D276): each pair $(i, k) \in D_d$ adds
$\tfrac12 k(i,k)$ at $i$ and $\tfrac12\sigma\,k(i,k)$ at $k$, owned or ghost,
again with $\mathbf a = \mathbf p + G^\mathsf T\mathbf q$. It is the transpose of
a directed `sum_pairs`, the form the derivative takes (Section 14.3), and
it needs an exchange contract, as the unique traversal does. The directed
traversal computes every pair twice and returns nothing; the unique one
computes each pair once and returns the ghosts' contributions by
$G^\mathsf T$. The choice is the classic one between full and half lists
with reverse communication [[Plimpton1995]](references.md#plimpton1995); here
it is a type of the relation, and the needs of each (an exchange contract
for the unique one, proven by the same code as `md-check-exchange`) are
checked where the op is verified and again where it is lowered.

### 4.7 Reductions and replicated values

`md_dist.all_reduce %p : !md_dist.partial<T> -> T` sums the partial sums of
the domains; its result is one logical value, replicated on every domain.
Its adjoint gives each partial the cotangent of the result, once: the
replicas of a value are one value, so no factor of $D$ enters a derivative
(a sum-and-replicate adjoint test checks it, Section 12).

`md_dist.sum_particles` sums a kernel over the owned rows of fields (the
kinetic energy, the embedding energy of EAM). Over ghost rows, with
`rows(ghost)`, it is the inner product of the ghost space, which the
adjoint test uses; no physical sum is over ghosts.

### 4.8 Freshness: versions are SSA values

The three freshness questions of the
[architecture plan](future-architecture-plan.md#three-separate-freshness-questions)
map to the IR as follows.

| Question | In the IR |
|---|---|
| Ownership and identity | The layout value from which a field comes (through `distribute`, `halo_sum`, the ops that compute owned fields from owned fields, and later `migrate`) |
| Neighborhood validity | The relation records its positions and its halo; using it at other positions needs a witness of coverage (Section 8) |
| Field freshness | A ghost value records the owned value it copies: the operand of its `halo_copy` |

The version of a field is its SSA value. A ghost copy `%xg = halo_copy %x0`
is of the version `%x0`; an op that reads `%x1` at its centers and `%xg` at
its neighbors reads two versions, and the verifier rejects it (V2). A
missing transfer is an op that reads a field at neighbors without a ghost
copy of it (V1). Neither needs a run to be found.

### 4.9 The verifier

Each check is in the verifier of an op or in a structural check shared by
the verifiers and the lowering, and each has a negative test
(`test/Dialect/MDDist/invalid.mlir`).

| | Check | Diagnoses |
|---|---|---|
| V1 | A field read at neighbors is given with a ghost value | Missing transfer |
| V2 | The ghost value is `halo_copy` of exactly the owned value given beside it | Stale halo (a copy of another version) |
| V3 | That `halo_copy` is through the relation's halo, and positions are copied with `image` | Wrong halo; ghosts without images |
| V4 | The halo's reach is at least the cutoff | Incomplete coverage |
| V5 | Owned values, halos, and relations are of one layout | Stale layout (a field of another epoch) |
| V6 | A partial value is consumed only by `halo_sum` (fields) or `all_reduce` (numbers) | A nonlinear read of a partial sum; a contribution collected as if complete |
| V7 | A partial value has exactly one use, and the two partial operands of `halo_sum` are the two results of one traversal | Contributions lost or counted twice |
| V8 | A unique traversal, or the two-ended gather over a directed one, states an exchange contract that the kernel satisfies | A pair computed once whose other half is unknown |
| V9 | The relation's positions are those of the op that traverses it | A relation of another configuration without a coverage witness |

Where a check cannot be decided (a value through a block argument whose
layout is unknown), the verifier rejects with the reason, so that a proven
property and a failure are never confused (workflow, Validation).

## 5. A schedule, written out

Lennard-Jones with energy, forces, and virial, on the directed traversal:

```mlir
%l  = md_dist.partition %x, %cell
%xo = md_dist.distribute %x, %l image(%cell)
%h  = md_dist.halo %l, %xo, %cell reach(2.0)
%xg = md_dist.halo_copy %xo, %h image(%cell)          // transfer 1
%n  = md_dist.neighborhood %xo, %xg, %h cutoff(2.0) traversal(directed)
%up = md_dist.sum_pairs %n, %xo, %xg exchange(symmetric) { u(r) }
%fo = md_dist.gather_pairs %n, %xo, %xg exchange(antisymmetric) { -u'(r) d / r }
%wp = md_dist.sum_pairs %n, %xo, %xg exchange(symmetric) { d ⊗ (-u'(r) d / r) }
%u  = md_dist.all_reduce %up
%w  = md_dist.all_reduce %wp
%f  = md_dist.collect %fo, %l
```

On the unique traversal the gather yields `%fo, %fg` and
`%f = md_dist.halo_sum %fo, %fg, %h` (transfer 2) precedes `collect`. EAM
adds one transfer between its stages
([architecture plan](future-architecture-plan.md#eam-makes-the-stage-boundary-concrete)):

```mlir
%rho = md_dist.gather_pairs %n, %xo, %xg { g(r) }       // directed: complete
%psi = md_dist.map_particles gather(%rho) { F'(rho) }      // owned
%pg  = md_dist.halo_copy %psi, %h                        // transfer 2
%f   = md_dist.gather_pairs %n, %xo, %xg gather(%psi, %pg) { ... }
```

and a `%pg` copied from a `%psi` of the previous step is V2.

`--md-dist-print-schedule` prints, for each transfer, the field version it
copies, its halo and reach, its direction ($G$ or $G^\mathsf T$), and its
consumers; at run time `MDRT_DIST_TRACE` prints the owned and ghost counts
of each domain and the rows each transfer moves. Both are part of stage 1;
the static print is the "inspectable schedule" of #31, and the trace is
what a test checks a schedule against.

### Schedule proof for halo transfers

The schedule proof (D308) is an analysis of the existing
`halo_copy` and `halo_sum` SSA graph. A transfer record retains the exact
source field values, halo and ownership layout, direction, result value,
and every result consumer. Its start prerequisite is the producer of each
source value and of the halo, or the block entry for an argument. Completion
precedes every read of the result; the first such consumer is named
separately. A source or consumer crossing the block boundary is rejected.
The proof also records operations between these points whose operands do
not depend on that result. Such operations may run during a future
asynchronous transfer; the proof alone does not move or overlap them.
An unproved provenance, incompatible ownership epoch, stale ghost copy, or
unsupported control-flow relation rejects the schedule. Both the print
pass and future transfer lowerings consume the same proof.

The current IR has no operation that partitions owned centers into an
interior and a boundary set. A later decision under #481 must add a
relation partition whose interior rows have no ghost neighbors, then
schedule the storage-level event below so its wait precedes boundary
work or a returned owner value. The in-process executor remains the
synchronous baseline; this schedule proof does not claim runtime overlap.

### Storage-level completion events (D310)

An opt-in lowering of the existing $G$ and $G^\mathsf T$ operations emits
`mdrt.halo_copy_start` or `mdrt.halo_sum_start`, followed by
`mdrt.halo_wait`. A start takes the halo handle and the exact physical
source and destination buffers and yields `!mdrt.event`. The wait consumes
the event, those same buffers, and the handle, and yields the completed
destination buffer as an SSA version. The verifier requires one matching
wait later in the same block and direct, distinct `memref.alloc` buffers
of the same three-dimensional f64 type. It forbids access to either buffer
or the halo handle between start and wait, destination access after start
except through the completed result, deallocation before completion,
double deallocation of either allocation, and deallocation through the
completed alias. It refuses unproved aliases and intervening nested
control flow. A start with no semantic consumers is still completed
before the function ends; an empty consumer set alone proves no lifetime.

`halo-events=true` on `--convert-md-dist-to-loops` and
`--convert-md-dist-to-md-exec` selects this form; it defaults to false.
`--lower-mdrt-halo-events` lowers the starts to the current synchronous
in-process transport calls and replaces each wait result with its
destination buffer. The two lowerings therefore have the same numerical
order. These events expose dependencies in the storage IR; they do not
overlap communication with computation. Interior/boundary row work and
runtime asynchronous transport remain separate decisions under #481.

## 6. The in-process reference executor

The first executor runs all domains in one process
(`--convert-md-dist-to-loops`). It is the oracle for every later
transport: a transport must give the same rows and the same sums.

- Each op runs for every domain in turn; collective ops ($G$,
  $G^\mathsf T$, `all_reduce`) are the points where all domains' values are
  complete, so running op by op is a valid bulk-synchronous schedule of
  the global view.
- Partition, halo construction, $G$, and $G^\mathsf T$ are calls of the
  transport ABI (`runtime/mdrt_dist.c`, Section 9): the in-process
  transport is a copy between the domains' buffers.
- Pair ops become loops over each domain's owned centers and all its
  local rows, with the cutoff test; each pair kernel is inlined. The search
  is over all pairs of a domain: the oracle has no cells and no neighbor
  structure, so it shares no code with the neighbor search of `md_exec`,
  which the one-domain reference runs.
- Domains are numbered in a fixed order and their partial sums are added
  in that order, so the result is the same to the bit from run to run for
  a given grid. It is not the same to the bit as one domain, whose sums run
  in another order.
- The ops of `md_dist` of a function run in one block, and a value of
  `md_dist` goes only to ops of `md_dist`. An op of another dialect, or of
  no registered dialect (under `--allow-unregistered-dialect`), that uses
  such a value is refused with a diagnostic; one that does not is left
  where it is (`test/Conversion/MDDistToLoops/unregistered-op.mlir`).

## 7. Lowering a pair loop over owned and ghost rows

### 7.1 Reference executor

Section 6: for each domain, for each owned $i$, for each local row $k \ne i$,
$\mathbf d = \mathbf x_i - \mathbf x_k$ and the cutoff test; the unique traversal
skips $k$ unless $\mathrm{id}(i) < \mathrm{id}(s(k))$, and adds $\sigma\,k$ to
the row of $k$, owned or ghost.

### 7.2 Through `md_exec`

`D290` (#442): `--convert-md-dist-to-md-exec`, the second
executor, makes a domain's work ordinary `md_exec` over its local rows. It
runs where `--convert-md-dist-to-loops` runs, before
`--convert-md-to-md-exec`, and the passes of `md_exec` that follow are the
existing ones, on the CPU and on a device. The reference executor stays
the oracle of the new path and its switch (principle 3).

*Two row sets per domain.* Each domain $d$ has two particle sets of its
own, declared by the pass: `@<decomposition>.d<d>.owned`, its owned rows
$O_d$, and `@<decomposition>.d<d>.local`, its local rows, $O_d$ followed by
$\Gamma_d$. A set of `md_exec` has one number of rows, which
`md-exec-assign-storage` takes from the first buffer of the set, so sets of
different domains, and the owned and the local rows of one domain, are
different sets. Loops over particles (`sum_particles`, `map_particles`) run
over the owned set, loops over pairs over the local set.

*Center extent.* `md_exec.build_neighbors`, `md_exec.refresh_neighbors`,
and `md_exec.pair_for` take `centers(%n)`: the first $n$ rows of the set
are the centers, the rows of the list are those of the centers, and its
entries are any rows of the set. A loop updates a destination at the
centers (and at the neighbors, below); the other rows keep what they hold.
On a domain $n = \lvert O_d\rvert$. On the CPU the build searches the cells
of the centers only (`@mdrt.build_neighbors_matrix_centers`, a variant of
the template added only to a module that builds with `centers`, so that no
other module changes); on a device the build searches every row, and the
loop's places in the order of the cells skip the rows that are not
centers.

*Explicit images.* With `images(explicit)` a loop takes the displacement
$\mathbf d = \mathbf x_i - \mathbf x_k$ as a plain difference, since ghosts
carry their images, and no minimum image; the cell is still an operand, for
the virial of a strain and the later skin test (Section 8). A structure
with `images(explicit)` holds every pair of rows within the reach in plain
differences. The existing search, which wraps the rows into the cell and
takes the minimum image, finds a superset of these pairs: the
minimum-image distance of two rows never exceeds their plain distance. The
loop's test of the cutoff on the plain displacement removes the rest. A
ghost and its source wrap onto one point and list each other and each
other's neighbors, so the rows are longer than they need to be (measured
in Section 7.3); a search over the box of the local rows that does not
wrap is an optimization for later. Such a structure is a matrix, leaves
out no pairs (exclusions across domains are the topology of stage 2), and
is built at every refresh (the policy `always`): the skin test over
explicit images comes with migration (Section 8).

*Writes at the neighbor.* `pair_for` with the policy
`(directed, atomic)` adds, for every entry $(i, k)$ of a center $i$, the
kernel's value at $i$ and $\sigma$ times it at $k$ (from `exchange`, which
must be `symmetric` or `antisymmetric`, as for `(unique, atomic)`). Since
other rows add to a row at their neighbors while it adds its own, both
additions are atomic, on a device and on the host (`memref.atomic_rmw`),
and a destination of such a loop, or of a loop over centers, starts from
zeros and is never overwritten (`md-exec-assign-storage`). This is the
two-ended `gather_pairs`: its owned and ghost partial results are the
prefix and the tail of one destination over the local rows, and
`halo_sum` returns the tail to the owners ($G^\mathsf T$). Over a directed
relation the kernel is scaled by one half (Section 4.6).

*The unique traversal* needs no list of its own: the list is the directed
one, and the kernel's value is selected only where
$\mathrm{id}(i) < \mathrm{id}(s(k))$, with the identities of the local
rows given as a field of `i64` in `ins`. Every pair is then visited, and
its kernel evaluated, at both of its rows of a domain, and taken at one; a
half list built by identity is #391's.

*Transfers.* Distributed values keep the storage of the reference executor
and the calls of `runtime/mdrt_dist.c` (partition, halo, $G$, $G^\mathsf T$,
`all_reduce`). At the boundary of `md_exec` each domain's rows are copied
from the buffer of all domains into a buffer of the domain (`[x_o ; G_x x]`
for the positions, `[a_o ; G a]` for a gathered field) and handed over by
`mdrt.from_buffer`; results come back by `mdrt.to_buffer` and are copied
into the buffer of all domains. These copies are where a rank of the MPI
transport (order 2) sends and receives; there a rank holds its own rows
only.

*Order.* The domains are unrolled: for every op of `md_dist`, the work of
domain 0, then 1, and so on, as in Section 6. The code of one domain is
emitted from the domain's number; an MPI rank emits it once, for its own.
On one thread of the host the results are the same to the bit from run to
run for a fixed grid. The additions at the neighbors are in no fixed order
on a device or under OpenMP, and the deterministic mode of the device
lowering (D84) refuses the policy `(directed, atomic)`; the directed
traversal after `md-dist-owner-computes` writes at the centers alone and
keeps it.

*The passes of `md_exec`.* Those that rebuild a loop or a structure carry
`centers` and `images` over (`md-exec-fuse-loops`, which fuses two loops
only with the same centers and images; `md-exec-accumulate-destinations`;
`md-exec-assign-storage`), and the lowerings to loops and to a device
learn them. `md-exec-reuse-neighbors` leaves a structure with either where
it is, built at every evaluation; `md-exec-choose-neighbors` keeps it a
matrix, since a structure of groups (D89) takes every row as a center in
the minimum image; and a device lowers a loop over the rows of a domain on
its own, not in a fused run of loops over rows. The verifier refuses
`centers` or `images(explicit)` on a structure of groups, with exclusions,
or refreshed with another policy than `always` or with `prune_skin`, the
policy `(unique, atomic)` with either, and a loop whose centers or images
are not those of its structure where the structure's build is visible
(`test/Dialect/MDExec/domain-rows-invalid.mlir`).

What `--convert-md-dist-to-md-exec` refuses, besides what the reference
executor refuses (a cell that is not orthorhombic, fields not of `f64`,
values of `md_dist` handed to another dialect): loops over ghost rows
(`sum_particles rows(ghost)`, the inner product of the adjoint test, which
the reference executor keeps). Its option `functions` names the functions
it runs and leaves the others to the reference executor, so that one
program runs through both and the two are compared.

### 7.3 Validation

The tests of Sections 14.5 and 15.5, through the second executor:
`test/Integration/dist-lj-exec.mlir` (Lennard-Jones, and the program of
`dist-lj.mlir` written by hand), `dist-eam-exec.mlir` (EAM, and the adjoint
identity of the density), and `dist-many-body-exec.mlir` (the many-body
energy whose pair sum reads $\psi = \sqrt\rho$ at both particles). Each runs
the grids of those tests with the unique traversal, the directed traversal
in the derivative's form ($G^\mathsf T$ through the tail of a loop that
writes at the neighbors), and the directed traversal after
`md-dist-owner-computes` (the many-body energy on the directed traversal
only), in `double` (12, 12, and 6 runs) and in `mixed` (4, 4, and 1 runs).
The oracles are those of the earlier tests (the same potential on one
domain through `md_exec`, `Inputs/eam_reference.py` and
`Inputs/lj_reference.py`, central differences of the distributed energy
in positions and in the stretches of the cell, the adjoint identity) and,
new, the same program through the reference executor in `double` (the
oracle of a transport, Section 6). Their twins in
`test/Integration/GPU/` run the device lowering, the domains one after
another on one device, on 7, 7, and 3 runs in `double` and 3, 3, and 1 in
`mixed`. The largest relative differences, over all runs (energy relative
to $\lvert U\rvert$, a force component to the largest one, a virial
component to the trace), CPU and an RTX 4090:

| Quantity | Reference | `double`, CPU | `double`, GPU | `mixed`, CPU | `mixed`, GPU | Tolerance (`double`, `mixed`) |
|---|---|---|---|---|---|---|
| LJ energy, force, virial | One domain | $3.8\times10^{-16}$, $2.9\times10^{-15}$, $3.0\times10^{-16}$ | $1.3\times10^{-16}$, $3.0\times10^{-15}$, $1.7\times10^{-16}$ | $9.7\times10^{-8}$, $4.5\times10^{-6}$, $1.8\times10^{-7}$ | $1.2\times10^{-7}$, $4.2\times10^{-6}$, $1.7\times10^{-7}$ | $10^{-10}$, $10^{-5}$ |
| LJ energy, force, virial | Reference executor | $2.5\times10^{-16}$, $4.2\times10^{-16}$, $4.3\times10^{-17}$ | $2.5\times10^{-16}$, $6.6\times10^{-16}$, $2.2\times10^{-16}$ | $1.4\times10^{-8}$, $3.9\times10^{-6}$, $8.9\times10^{-8}$ | $9.2\times10^{-9}$, $2.8\times10^{-6}$, $1.2\times10^{-7}$ | $10^{-10}$, $10^{-5}$ |
| LJ energy, force | Written by hand (both traversals) | $2.5\times10^{-16}$, $1.0\times10^{-15}$ | $1.3\times10^{-16}$, $5.3\times10^{-16}$ | $7.9\times10^{-9}$, $6.5\times10^{-7}$ | $1.7\times10^{-8}$, $5.4\times10^{-7}$ | $10^{-10}$, $10^{-5}$ |
| EAM energy, force, virial | One domain | $1.8\times10^{-15}$, $3.0\times10^{-15}$, $3.9\times10^{-16}$ | $1.8\times10^{-15}$, $3.7\times10^{-15}$, $1.5\times10^{-16}$ | $7.1\times10^{-7}$, $2.3\times10^{-6}$, $8.4\times10^{-8}$ | $7.3\times10^{-7}$, $2.4\times10^{-6}$, $7.7\times10^{-8}$ | $10^{-10}$, $10^{-5}$ |
| EAM energy, force, virial | Reference executor | $8.9\times10^{-16}$, $9.9\times10^{-16}$, $9.7\times10^{-17}$ | $2.7\times10^{-15}$, $1.4\times10^{-15}$, $1.9\times10^{-16}$ | $1.5\times10^{-7}$, $1.6\times10^{-6}$, $7.7\times10^{-9}$ | $3.6\times10^{-7}$, $1.4\times10^{-6}$, $4.3\times10^{-8}$ | $10^{-10}$, $10^{-5}$ |
| Many-body energy, force, virial | One domain | $2.4\times10^{-16}$, $1.9\times10^{-15}$, $1.0\times10^{-16}$ | $0$, $2.5\times10^{-15}$, $1.0\times10^{-16}$ | $9.8\times10^{-8}$, $1.3\times10^{-6}$, $2.7\times10^{-8}$ | $8.4\times10^{-8}$, $1.0\times10^{-6}$, $2.3\times10^{-8}$ | $10^{-10}$, $10^{-5}$ |
| Many-body energy, force, virial | Reference executor | $1.2\times10^{-16}$, $8.0\times10^{-16}$, $1.0\times10^{-16}$ | $1.2\times10^{-16}$, $1.0\times10^{-15}$, $5.2\times10^{-17}$ | $1.4\times10^{-8}$, $1.1\times10^{-6}$, $4.2\times10^{-9}$ | $3.6\times10^{-8}$, $1.3\times10^{-6}$, $1.9\times10^{-8}$ | $10^{-10}$, $10^{-5}$ |
| Force components, LJ / EAM / many-body | Central differences (`double`) | $1.5\times10^{-9}$, $2.4\times10^{-9}$, $2.5\times10^{-9}$ | $1.5\times10^{-9}$, $1.4\times10^{-9}$, $2.5\times10^{-9}$ | | | $10^{-7}$ |
| Diagonal of the virial, LJ / EAM / many-body | Central differences in the stretches (`double`) | $1.4\times10^{-10}$, $6.2\times10^{-11}$, $7.1\times10^{-12}$ | $7.9\times10^{-11}$, $5.5\times10^{-11}$, $9.7\times10^{-12}$ | | | $10^{-7}$ |
| $\langle\mathbf F, \mathbf v\rangle$ of $U = \langle\mathbf c, \rho\rangle$ | $-\langle\mathbf c, J\mathbf v\rangle$ (`double`) | $4.4\times10^{-15}$ | $2.1\times10^{-15}$ | | | $10^{-12}$ |

In `mixed` the kernels compute in f32 and the global sums in f64; the
buffers of the transport stay in f64, and `md-exec-assign-precision`
assigns the types within `md_exec` from the f64 buffers that
`mdrt.from_buffer` hands it, as at any boundary of a program. Central differences are not checked
in `mixed`: the energy's rounding over $2h$ exceeds the forces. The
mixed-precision measurements above are those of the grid $3\times2\times1$
with each traversal (the many-body energy after `md-dist-owner-computes`);
the tests' fourth mixed run, $16\times1\times1$ after
`md-dist-owner-computes`, passes the same tolerance.

*The rows of the structure.* For the 64 particles of these tests (reach
2.0, no skin), the entries of a row of a center in the search that wraps,
against the rows of the local rows within the reach in plain differences,
which equal those of one domain in the minimum image (19.2): 105.0 on
$1\times1\times1$, 68.0 on $2\times1\times1$, 40.0 on $3\times2\times1$, and
59.7 on $16\times1\times1$, 2.1 to 5.5 times as many. They are computed
from the halo of `runtime/mdrt_dist.c`, whose ghosts agree with
`MDRT_DIST_TRACE` of a run (271 on $1\times1\times1$; 132 to 147 a domain
on $3\times2\times1$), and the search's test of the minimum-image
distance; they are the baseline of a search that does not wrap.

*No other program changes.* The module that the pipeline of
`mdir emit --stage=pipeline` makes of JAC NVE (`GPU`, `MIXED`, through
`convert-md-exec-to-gpu`, and `CPU`, `DOUBLE`, through
`convert-md-exec-to-loops`) is the same text with and without this change,
and so are those of `dist-lj-distribute.mlir` and `dist-eam.mlir` through
the reference executor, `lj-dynamics.mlir` on the CPU, and
`tuples-nonbonded.mlir` on a device in `mixed`.

IR tests: `test/Dialect/MDExec/domain-rows.mlir` (the forms of the ops),
`domain-rows-invalid.mlir` (the verifier),
`test/Conversion/MDDistToMDExec/pairs.mlir` (the sets of the domains, the
structure with a center extent and explicit images, the sums, the mask of
the unique traversal, and the two-ended loop with $G^\mathsf T$),
`invalid.mlir` (ghost rows), and `deterministic-invalid.mlir` (the device's
deterministic mode). Under compute-sanitizer
(`test/Sanitizer/dist-exec-gpu.test`, with `-Dsanitize=1`): Lennard-Jones on
$3\times2\times1$ with the unique traversal and EAM in the derivative's
form, memcheck, initcheck, and racecheck, no errors and no hazards (42 s on
an RTX 4090).

### 7.4 What is left of the path through `md_exec`

- **Groups and dual lists on domains.** A structure of groups (D89) and a
  dual list (D114, D284) are refused for the rows of a domain, and they are
  the fast path on a device: the per-domain timings of this stage are those
  of the matrix. Distributed runs on GPUs need groups with `centers`.
- **A search that does not wrap**, over the box of the local rows, with the
  rows of Section 7.3 as its baseline; on a device, a build of the rows of
  the centers only.
- **A half list by identity** for the unique traversal, and the two-valued
  pair kernel (#391).
- **Migration and `covered`** (Section 8, #38): a structure kept across
  steps needs the skin test over explicit images; today a structure with
  `centers` or explicit images is built at every evaluation, with the halo.
- **MPI on CPU ranks** (transport order 2): the per-domain code is emitted
  from a domain's number, and the copies at the boundary of `md_exec` are
  where a rank sends and receives; a rank then holds its own rows, and the
  buffers of all domains of the reference executor become those of one.
- **Buffers handed to `md_exec`** by `mdrt.from_buffer` are, as for every
  program of `md_exec`, not freed by it; a trajectory on domains needs the
  copies at the boundary kept across steps rather than made at each.
- **The deterministic mode** for the two-ended loop, by fixed point as the
  spreading of PME (D70) or by rows of the neighbors.

## 8. Migration and rebuilds

Between steps particles move. Three things can go stale, and each has its
own op and its own test, as the plan separates them:

```mlir
%ok  = md_dist.covered %h, %x_ref, %x, %cell cutoff(2.0)   // i1, agreed by all domains
%l2  = md_dist.migrate %l, %x, %cell                        // new layout
%x2  = md_dist.redistribute %x, %l, %l2                     // a field to the new epoch
```

- `covered` is the skin test of neighbor validity, extended to the halo:
  the halo built at $\mathbf x^\text{ref}$ with reach $R$ covers a cutoff
  $r_c$ at $\mathbf x$ if no particle has moved more than $(R - r_c)/2$
  (the existing condition with the scale of the cell, [neighbors.md](neighbors.md)).
  Its result is reduced over all domains before it is a value, so every
  domain takes the same branch: a predicate that guards a collective is
  agreed by construction (#38). A relation or halo used at other positions
  than it was built at needs this witness (V9).
- `migrate` gives particles that left their region to their new owner. A
  field of the old layout is not a field of the new one (V5): every live
  owned field goes through `redistribute`, and a field that does not is a
  stale-layout error, not a silent permutation.
- A step loop carries the layout, the halo, and the owned fields together,
  as one value of a state type, so that a loop cannot carry a field apart
  from its layout and the verifier keeps its proof across iterations.

Rebuild and migration happen together when `covered` fails: migrate,
redistribute, rebuild the halo. Migration between rebuilds (particles
that leave their region while the halo still covers them) is not needed,
since ghosts of the old layout are still exact.

The first migration slice, D313 (#494), implements the new
ownership epoch and redistribution of complete owned fields in the
in-process executor. The epoch is the new layout SSA value, not the
decomposition's grid, which remains the same. Within each destination,
rows are ordered by global particle identity; redistribution looks up
the old row of each identity and produces a new field for the new layout.
The old fields and buffers remain valid until their existing users finish.
Every live field needed after migration must be redistributed before it is
used with the new layout; the verifier rejects an old-epoch field there.
This slice does not make the step loop of Section 8 executable: the current
executor requires all `md_dist` ops of a function in one block. A later
part of #38 adds `covered`, a collective branch over all domains (including
empty ones), loop-carried layout and field state, completion of halo events
before buffer retirement, and the production driver's `[distribute]`
boundary. Repeated independent snapshots in tests are not an integrated
trajectory.

## 9. Transports

The ABI below `md_dist` is a small set of collective calls: partition,
halo construction, $G$ (for a number of components, with or without
images), $G^\mathsf T$, an all-reduce, migration, and redistribution. An
implementation provides all of them; the IR does not change with the
transport.

The order, as ruled on PR #379 (2026-10-16):

| Order | Transport | Purpose |
|---|---|---|
| 1 | In-process, on the host | The oracle (this PR) |
| 2 | MPI, CPU ranks | Separate address spaces, real messages; #35 and #38 on CPU. An optional build (a CMake option), whose tests are unsupported without it; MDIR's compiler links the C ABI of the MPI library |
| 3 | One process driving two GPUs, with peer copies | One domain per GPU; the short path to ML3, with no communication library |
| 4 | NCCL | Several processes and nodes |
| 5 | NVSHMEM | Fine-grained overlap |

Directed and staged-forwarding exchanges (the six exchanges of a
Cartesian grid that forward edges and corners, [[Saunders2018]](references.md#saunders2018))
are choices of the transport; the halo's contract (every image within
$R$, its source and image) is the same.

## 10. What the vocabulary leaves room for

- **Topology across domains** (stage 2): a tuple is owned by the owner of
  one member, and its other members are ghosts found by identity, not by
  reach; a halo of identities beside the halo of reach, with the same $G$
  and $G^\mathsf T$.
- **PME** (stage 3): a decomposition of the grid beside that of the
  particles, with its own layout; spreading writes partial grid values
  that a reverse halo of the grid completes, the analogue of
  `halo_sum`.
- **Ensembles** (stage 4): kinetic energy and virial are `all_reduce`s of
  partial sums; a change of the cell changes the regions, which a
  `migrate` follows.
- **Asynchronous transfers**: a transfer becomes a start and a wait
  ordered by `!mdrt.event` tokens, after storage assignment; the
  synchronous op stays the baseline and the switch (principle 2).

## 11. Differentiation, and the order of the passes

`md-distribute` (stage 1, part 2) turns a global potential into its
distributed form: every read of a field at neighbors gets a `halo_copy`,
every sum over pairs becomes `sum_pairs` and `all_reduce`, every gather a
`gather_pairs`. It runs **before** `md-differentiate`, so that the
derivative sees $G$:

- The rule of `halo_copy` is $\bar{\mathbf a} \mathrel{+}= G^\mathsf T\bar{\mathbf b}$,
  a `halo_sum`; the rule of `halo_sum` is the copy and the identity. These
  are the rules the AD already applies to a gather and a sum over a
  relation, with the communication made explicit.
- The derivative of a distributed energy
  $U = \sum_d U_d(\mathbf x_{O_d}, G_{\mathbf x}\mathbf x)$ is then
  $\partial U/\partial\mathbf x = \partial_O U + G^\mathsf T\partial_\Gamma U$:
  forces with reverse communication, the form a message-passing model
  needs (#377, Section "Two GPUs").
- The directed traversal, which computes each pair at both owners and
  returns nothing, is an *optimization* of that form, `md-dist-owner-computes`:
  it replaces $G^\mathsf T$ of the ghosts' adjoints of a pair sum by the same
  pairs evaluated at the other owner. Its precondition, that every pair is
  in the relation of both owners (V4 with Section 4.3), is checked; it has a
  switch; and the two are compared (principle 3).

Running `md-distribute` after the derivative would also work for classical
pairs, but a staged model's backward would then need its communication
derived again from the backward program; with $G$ in the forward program,
the backward's communication is the AD's transpose of the forward's.

## 12. What the first PR builds

1. **The dialect `md_dist`** (Section 4;
   `include/mdir/Dialect/MDDist`, `lib/Dialect/MDDist`): the types and
   ops above, with the provenance of Section 4.8 (`layoutOf`, `haloOf`)
   and the verifiers V1 to V9. The exchange proof of `md-check-exchange`
   is shared (`md::proveExchange`), and it now also follows components and
   lanes of vectors (`vector.extract`, `vector.from_elements`,
   `vector.shuffle`), so that the outer product $\mathbf d\otimes\mathbf d$ of
   a virial is proven symmetric (`check-exchange.mlir`).
2. **The in-process reference executor** (Section 6):
   `--convert-md-dist-to-loops` and `runtime/mdrt_dist.c` (partition,
   halo, $G$, $G^\mathsf T$, and the trace `MDRT_DIST_TRACE`). It runs one
   decomposition per function, the ops of `md_dist` in one block, an
   orthorhombic cell, and `f64`.
3. **Lennard-Jones on several domains against one**
   (`test/Integration/dist-lj.mlir`): 64 particles in a cube of edge 4.8,
   $r_c = 2.0$, halo reach $2.0$ (the edge of coverage), on the grids
   $1\times1\times1$ (one domain whose ghosts are its own images, 271 of
   them), $2\times1\times1$, $3\times2\times1$ (uneven: 6 to 15 owned
   rows), and $16\times1\times1$ (domains of width 0.3, narrower than the
   cutoff; 8 of the 16 own no particle; ghosts from domains that are not
   neighbors), each on the directed and the unique traversal. The
   reference is the same potential on one domain through `md.evaluate`,
   `md-differentiate`, and `md_exec`'s neighbor structures, whose energy
   agrees with `Inputs/lj_reference.py` without its switch
   ($-226.88590520072805$). The largest differences, over all four grids,
   `double`, on the CPU:

   | Quantity | Scale | Directed | Unique | Tolerance of the test |
   |---|---|---|---|---|
   | Energy | $\lvert U\rvert = 226.886$ | $8.5\times10^{-14}$ | $8.5\times10^{-14}$ | $10^{-10}\lvert U\rvert$ |
   | Force component | $\max\lvert F\rvert = 16.997$ | $5.2\times10^{-14}$ | $5.1\times10^{-14}$ | $10^{-10}\max\lvert F\rvert$ |
   | Virial component | $\lvert\mathrm{tr}\,\mathsf W\rvert = 660.908$ | $2.0\times10^{-13}$ | (not computed) | $10^{-10}\lvert\mathrm{tr}\,\mathsf W\rvert$ |

   A virial of the wrong sign fails the test (by 469). The same runs give
   the adjoint identity
   $\langle G\mathbf a, \mathbf b\rangle = \langle\mathbf a, G^\mathsf T\mathbf b\rangle$,
   with $\mathbf a$ the owned positions and $\mathbf b$ the ghost positions
   with their images (271 ghosts of 64 particles on one domain), to a relative $4\times10^{-16}$ or less (tolerance
   $10^{-12}$).
4. **Negative tests** of V1 to V9 (`test/Dialect/MDDist/invalid.mlir`):
   a missing transfer of the positions and of a gathered field, a stale
   halo of the positions and of an intermediate field (as EAM's
   embedding derivative), a copy through another halo, ghost positions
   without images, a reach below the cutoff, a field of another layout,
   contributions lost, counted twice, mixed between two traversals, or
   read before their completion, a unique traversal without or with a
   false exchange contract, a directed sum of an asymmetric kernel, an
   unreduced partial sum, and a relation traversed at other positions.

Next parts of stage 1, each with its own tests: `md-distribute` and the
rules of the AD (Section 11; built, Section 14), EAM (built, Section 15), the schedule print and trace, the
`md_exec` path with the `mixed` mode and GPUs (built, Section 7.2), migration and `covered` with a trajectory
(Section 8), MPI on CPU ranks, then several GPUs.

## 13. What has been ruled (the maintainer on PR #379, 2026-10-16)

1. **Control file.** A table `[distribute]`: `domains`, an integer (the
   grid chosen from the cell) or `[gx, gy, gz]`; `transport`, `"local"`,
   `"mpi"`, and later `"nccl"`; `traversal`, `"directed"` (the default) or
   `"unique"`. Without the table a run is one domain. The halo's reach
   comes from the cutoff and the skin. The keys arrive with the first part
   that runs a distributed trajectory.
2. **Python.** `mdir.compile(..., distribute=mdir.Distribute(domains=...,
   transport=..., traversal=...))`. State arrays stay in the order of the
   particle set.
3. **Communication**, in the order of Section 9.
4. **Outputs and checkpoints** stay in the order of the particle set,
   independent of the number of domains; the grid and the transport go in
   the run manifest (D168). Results are the same to the bit for a fixed
   grid, not across grids.

The order of the passes of Section 11 stands as designed.

## 14. The second part: `md-distribute` and the rules of $G$ and $G^\mathsf T$

D276 makes a global potential run on domains through the
compiler: the program of Section 5 is no longer written by hand but derived
from the potential of `md`, and its forces and virial from the semantic
differentiation, with the adjoints of the transfers taken by declared
rules.

### 14.1 The pipeline

```text
md-distribute{grid, traversal}    md → md_dist, in the body of md.potential
md-check-exchange                 unchanged (the contracts of md)
md-differentiate                  md.evaluate → md.function; adjoints by the rules of 14.3
md-dist-owner-computes            optional: the derivative's form → the owners (14.4)
md-expand-truncation, md-inline   unchanged
convert-md-dist-to-loops          the reference executor (Section 6)
```

`md-distribute` runs before `md-differentiate`, as ruled (Section 11). The
potential keeps its signature: it takes the positions and fields of the
particle set and returns the energy, and the derivative function returns
forces in the order of the particle set, so that `md.evaluate`, the tests,
and later the driver see no change but the decomposition.

### 14.2 `md-distribute`

The pass rewrites the body of every `md.potential` (or of those that its
option `potentials` names) on the grid `grid=gx,gy,gz`, with
`traversal=directed` (the default, as ruled for the control file) or
`traversal=unique`. It declares `md_dist.decomposition
@<potential>.domains` on the particle set of the positions and inserts, at
the start of the body, the layout, the owned positions wrapped into the
cell, one halo of reach $R$ (the largest cutoff, or the option `reach` if
it is larger), and the ghost positions with their images. Then, op by op:

| `md` | `md_dist` |
|---|---|
| A field argument | `distribute` of its owned rows (no image) |
| `md.neighborhood` | `neighborhood` over owned and ghost rows, of the traversal |
| `md.sum_relation` | `sum_pairs` and `all_reduce`; a truncation is expanded into the kernel first |
| `md.gather_relation` | `gather_pairs`: complete on `directed`; on `unique`, the two partial results and `halo_sum` |
| `md.sum_particles` | `sum_particles` over owned rows and `all_reduce` |
| `md.map_particles` | `map_particles` over owned rows (a new op) |
| A field read at neighbors | `halo_copy` of its owned rows, once per version |

Every fact the result relies on is checked by the verifier of Section 4.9:
a field read at neighbors gets the copy of exactly its version (V1, V2),
the halo covers every cutoff (V4), and every contract is proven (V8). The
pass refuses, naming the reason, what it cannot distribute:

| Refused | Why |
|---|---|
| `exclude(...)`, `md.sum_tuples`, `md.gather_tuples` | Tuples need the topology across domains (stage 2 of #31) |
| `md.reciprocal` | It needs a decomposition of its grid (stage 3) |
| A sum or map that reads the positions as a field | The owned positions are wrapped into the cell, so a term of the absolute positions would change |
| A contract that rests on an assertion and cannot be proven | `md_dist` proves every contract (V8) |
| A gather without a contract, on `unique` | The value at the second particle comes from the contract |
| A cell other than the potential's argument; an op of `md` the pass does not know | |

The reference executor further refuses a cell that is not orthorhombic at
run time and fields that are not of `f64` (Section 6).

### 14.3 The rules of the derivative

The transfers declare their transposes through an op interface,
`AdjointOpInterface`
(`include/mdir/Dialect/MD/Transforms/AdjointInterface.td`), as the scalar
rules of D183 are declared through `DerivativeOpInterface`; the rules are
attached to the ops of `md_dist` in
`lib/Dialect/MD/Transforms/Adjoints.cpp`:

| Op | Linear in | Transpose |
|---|---|---|
| `halo_copy` ($G$, or $G_{\mathbf x}$ with `image`) | The owned field | $\bar{\mathbf a} = \mathbf p + G^\mathsf T\bar{\mathbf b}$: `halo_sum` |
| `halo_sum` ($\mathbf a + G^\mathsf T\mathbf b$) | Both operands | $\bar{\mathbf a} = \bar{\mathbf r}$, and $\bar{\mathbf b} = G\bar{\mathbf r}$: `halo_copy` |
| `distribute` | The field | `collect` |
| `collect` | The field | `distribute` |
| `all_reduce` | The partial sum | Each domain's part takes $\bar r$, once |

The loops over the rows of domains (`sum_pairs`, `gather_pairs`,
`sum_particles`, `map_particles`) are not linear; they declare the
transpose of their structure through the same interface, as the ops of
`md` do (D297; [ad-robust.md](ad-robust.md)), and
`md-differentiate` fills the kernel of the op they create. The adjoint of a
field that a pair op reads at both ends of each pair is its owned rows and
their copy to the ghosts (Section 15.2). The premises of each form, the
traversal, the coverage (V4), and the contract (V8), stay in the pass.

$G_{\mathbf x}$ is affine in $\mathbf x$, so its transpose in the positions is
that of $G$; the wrapping of `distribute` is a translation by a lattice
vector, whose derivative is the identity. Layouts, halos, and relations are
structure: piecewise constant functions of the configuration (which domain
owns a particle, which images are ghosts, which pairs are within the
cutoff), whose derivative is zero almost everywhere, as that of the relation
of `md.neighborhood` is. The activity analysis (D182) takes them as
inactive, so that a field distributed by a layout of the positions, a
charge for one, does not count as depending on the positions.

`md-differentiate` then works as on one domain, with one change of form. A
sum over pairs gives the forces at both particles of every pair, as a
`gather_pairs` with two partial results over the same relation, the owned
and the ghost contributions; a sum over rows gives a map over the same
rows. The adjoints are carried back through the transfers in the reverse
order of the body: the ghost contributions of a gather return to the owners
together with its owned contributions in one `halo_sum` (the rule of
`halo_copy`, which is where $G^\mathsf T$ comes from), and the owned rows
return to the particle set by `collect`. A partial adjoint is never added to
anything else: each term travels on its own, and terms are added where they
are complete, by `md.map_particles` over the particle set. For
Lennard-Jones on the unique traversal the result is the program of
Section 5 with the derived kernel in place of the hand-written one:

```mlir
%g:2 = md_dist.gather_pairs %n, %xo, %xg exchange(antisymmetric) { −u'(r) d / r }
%fo  = md_dist.halo_sum %g#0, %g#1, %h          // the rule of halo_copy: G^T
%f   = md_dist.collect %fo, %l                   // the rule of distribute
%wp  = md_dist.sum_pairs %n, %xo, %xg exchange(symmetric) { d ⊗ (−u'(r) d / r) }
%w   = md_dist.all_reduce %wp
```

The virial is the sum of $\mathbf d\otimes\mathbf K$ over the pairs of the
traversal, each pair once [[Louwerse2006]](references.md#louwerse2006): the
ghosts carry their images, so $\mathbf d$ is the displacement of the pair in
the cell.

Not yet differentiated, each refused with a diagnostic: a pair sum that
gathers a field computed from the positions, and a derivative through a
`gather_pairs` or `map_particles` of the positions (built in the next
part, EAM, Section 15); and derivatives in a parameter or a field
(`derivative(n)`).

The derived kernels of `md_dist` must pass the proof of their exchange
contract (V8), where those of `md` carry the basis `derived` and are not
proven again. Every kernel derived here is proven (Lennard-Jones, its
truncations, the many-body kernels of the tests); a potential whose
derivative the proof cannot follow fails at the verifier after
`md-differentiate`, with the error on the generated op.

### 14.4 `md-dist-owner-computes`

On the directed traversal the derivative's form computes half of every
pair at each of its two particles and returns the ghosts' halves by
$G^\mathsf T$. The pass replaces `halo_sum` of a two-ended `gather_pairs` over
a directed relation by the gather with one complete result: every pair at
its center, with the weight one, and nothing returned.

*Claim.* For a kernel $k$ with the exchange contract $\sigma$ and a directed
relation $D$ whose halo covers its cutoff,
$\mathbf p_i + (G^\mathsf T\mathbf q)_i = \sum_{k : (i,k) \in D_{o(i)}} k(i,k)$.

*Proof.* Let $i \in O_d$ and $j$ any particle with $r_{ij} < r_c$. By
coverage (Section 4.3) the pair is in $D_d$ as $(i, k)$, $k$ the image of
$j$ nearest to $i$, and in $D_{o(j)}$ as $(j, k')$, $k'$ the image of $i$
nearest to $j$. The two-ended gather adds $\tfrac12 k(i,k)$ at $i$ from the
first and $\tfrac12\sigma\,k(j,k')$ at $k'$ from the second, which
$G^\mathsf T$ returns to $i$; by the contract $k(j,k') = \sigma\,k(i,k)$, so
that the two are $\tfrac12 k(i,k)$ each, and their sum is the term of the
one-ended gather at $i$. $\square$

*Premises*, each checked by the pass with the code of the verifier: the
relation is directed (its type); the reach covers the cutoff (V4,
`md_dist::verifyCoverage`); the kernel has the contract (V8,
`md_dist::proveContract`); `halo_sum` completes the two results of that
gather through the relation's halo (V7). The pass is the switch: without it
the derivative's form runs, and the tests run both (principle 3). The two
agree to the rounding of the sums, not to the bit, since the sums run in
another order.

### 14.5 Validation

`test/Integration/dist-lj-distribute.mlir`: the Lennard-Jones potential of
Section 12, distributed by `md-distribute` and differentiated by
`md-differentiate`, on the grids $1\times1\times1$, $2\times1\times1$,
$3\times2\times1$, and $16\times1\times1$, each with the unique traversal,
the directed traversal in the derivative's form, and the directed traversal
after `md-dist-owner-computes` (12 runs), against the same potential on one
domain through `md_exec`, against the program of `dist-lj.mlir` written by
hand, and against central differences of the distributed energy. The
largest relative differences over the 12 runs, `double`, on the CPU:

| Quantity | Reference | Largest difference | Tolerance |
|---|---|---|---|
| Energy, of $\lvert U\rvert = 226.886$ | One domain | $3.8\times10^{-16}$ | $10^{-10}$ |
| Force component, of $\max\lvert F\rvert = 16.997$ | One domain | $3.0\times10^{-15}$ | $10^{-10}$ |
| Virial component, of $\lvert\mathrm{tr}\,\mathsf W\rvert = 660.908$ | One domain | $3.0\times10^{-16}$ | $10^{-10}$ |
| Energy, force, virial | Hand-written, directed | $2.5\times10^{-16}$, $1.0\times10^{-15}$, $1.3\times10^{-16}$ | $10^{-10}$ |
| Energy, force | Hand-written, unique | $2.5\times10^{-16}$, $1.0\times10^{-15}$ | $10^{-10}$ |
| Six force components, of $\max\lvert F\rvert$ | Central differences of the distributed energy, step $10^{-6}$ | $2.1\times10^{-9}$ | $10^{-7}$ |
| The diagonal of the virial, of $\lvert\mathrm{tr}\,\mathsf W\rvert$ | Central differences in the three stretches of the cell, step $10^{-6}$ | $1.4\times10^{-10}$ | $10^{-7}$ |

The differences from central differences are those of the differences
themselves, the rounding of $U$ over $2h$.

`test/Integration/dist-adjoint.mlir`: the rules of $G$ and of $G^\mathsf T$
through `md-differentiate`. The potential
$U(\mathbf x) = \langle\mathbf c, G^\mathsf T G_{\mathbf x}\mathbf x\rangle_O$,
written in `md_dist`, has the gradient $G^\mathsf T G\mathbf c$, and its
derived forces satisfy
$\langle\mathbf c, \mathbf F\rangle_O = -\langle G\mathbf c, G\mathbf c\rangle_\Gamma$,
whose right side is computed directly: to $2.4\times10^{-15}$ relative or
less on the four grids (tolerance $10^{-12}$).

`test/Integration/dist-many-body.mlir`: the rewrite of a many-body energy,
$U = \sum_i q_i\psi_i + \sum_{i<j}\psi_i\psi_j/r_{ij}$ with
$\psi_i = \sqrt{\rho_i}$ and $\rho_i = \sum_j e^{-r_{ij}}$ (a gather, a map, a
sum over particles with a field argument, and a pair sum that reads the map
at both particles through its copy to the ghosts), on the grids
$2\times1\times1$, $3\times2\times1$, and $16\times1\times1$ and both
traversals, against one domain: $U = 1896.65$ within $2.4\times10^{-16}$
relative (tolerance $10^{-10}$). Its derivative is the next part
(Section 15).

IR tests: `test/Dialect/MD/Transforms/distribute.mlir` (the rewrite of
Lennard-Jones with a truncation, and of a many-body term with a gather, a
map, a sum over particles, a field argument, and a field read at
neighbors, on both traversals); `differentiate-distributed.mlir` (the
derivative's form, with $G^\mathsf T$, on both traversals, and its absence
after `md-dist-owner-computes`); negative tests of the pass
(`distribute-invalid.mlir`), of the derivative
(`differentiate-distributed-invalid.mlir`), and of the new forms of the
dialect (`test/Dialect/MDDist/invalid.mlir`).

### 14.6 What changed in the code

- The proof of exchange contracts moved from the passes of `md` to the
  library of the dialect (`include/mdir/Dialect/MD/MDExchange.h`), so that
  `md_dist` depends on `md` alone, and the passes of `md`, which now create
  ops of `md_dist`, depend on it.
- `md_dist.map_particles`; the two-ended `gather_pairs` over a directed
  relation, which the executor runs with the weight one half.
- No control-file key, Python API, or output changes: the keys ruled on
  PR #379 arrive with the first part that runs a distributed trajectory.

## 15. The third part: EAM on domains

D279 carries the derivative of a potential on domains through the
fields it computes from the positions: a density gathered over pairs, a
map of it, sums over particles that read them, and the copy of such a
field to the ghosts. The embedded-atom method (EAM)
[[Daw1984]](references.md#daw1984) is the case that needs all of them,
and the second pass of its forces reads, at the neighbors, a field of the
first: the derivative of the embedding energy.

### 15.1 The potential and its forces

$$
U = \sum_i F(\rho_i) + \tfrac12\sum_i\sum_{j\ne i}\phi(r_{ij}),
\qquad \rho_i = \sum_{j\ne i} f(r_{ij}).
$$

With $\bar\rho_i = \partial U/\partial\rho_i = F'(\rho_i)$, the chain rule
gives the force on particle $i$ as a sum over its pairs,

$$
\mathbf F_i = -\sum_{j\ne i}\Bigl[\bigl(\bar\rho_i + \bar\rho_j\bigr) f'(r_{ij})
+ \phi'(r_{ij})\Bigr]\frac{\mathbf d_{ij}}{r_{ij}},
\qquad \mathbf d_{ij} = \mathbf x_i - \mathbf x_j,
$$

antisymmetric in the pair, and the virial is
$\mathsf W = \sum_{i<j}\mathbf d_{ij}\otimes\mathbf K_{ij}$, with $\mathbf K_{ij}$
the term of the pair in $\mathbf F_i$
[[Louwerse2006]](references.md#louwerse2006). The force needs $\bar\rho_j$
at every neighbor $j$: on domains, a copy of $\bar\rho$ to the ghosts
between the pass that computes $\rho$ and the pass that computes the
forces. On one domain `md-differentiate` already derives this (the adjoint
of an intermediate field, D2, D182): the gather of the forces reads the
adjoint of the density at both particles of a pair.

### 15.2 Adjoints of fields on domains

`md-differentiate` keeps, for every field on domains that depends on the
positions, the derivative of the energy in it, its adjoint, as a list of
terms, and fills it in the reverse order of the body, as on one domain:

| Op of the potential that reads the field $\mathbf a$ | Term of $\bar{\mathbf a}$ |
|---|---|
| `sum_particles` of $k(\mathbf a_i, \dots)$, of weight $w$ | $w\,\partial k/\partial\mathbf a_i$: `map_particles` over the same rows |
| `map_particles` $\mathbf b_i = g(\mathbf a_i, \dots)$ | $\bar{\mathbf b}_i\,\partial g/\partial\mathbf a_i$: `map_particles` that also reads $\bar{\mathbf b}$ |
| `halo_copy`, $G\mathbf a$ | $G^\mathsf T$ of the adjoint of the copy (Section 14.3) |
| `halo_sum`, $\mathbf p + G^\mathsf T\mathbf q$ | $\bar{\mathbf p} = \bar{\mathbf r}$, $\bar{\mathbf q} = G\bar{\mathbf r}$ (Section 14.3) |
| `sum_pairs` that gathers $(\mathbf a, G\mathbf a)$, directed | $w\sum_{k : (i,k)\in D_{o(i)}}\partial_1 k(i,k)$: `gather_pairs` at the owners (15.4) |

Terms are added where they are complete, by `md_dist.map_particles` over
owned rows. A gather of the field gives the forces and the virial:

| Op of the potential that computes $\rho$ | Forces and virial from $\bar\rho$ |
|---|---|
| `gather_pairs` (directed, complete), or its two partial results completed by `halo_sum` (unique) | $\bar\rho_\Gamma =$ `halo_copy` of $\bar\rho$; a two-ended `gather_pairs` of $\mathbf K(i,k) = -\bigl(\bar\rho_i f'(i,k) + \bar\rho_k f'(k,i)\bigr)\mathbf d/r$ that gathers $(\bar\rho, \bar\rho_\Gamma)$; a `sum_pairs` of $\mathbf d\otimes\mathbf K$ |

$f'(i,k)$ is the derivative in $r$ of the density's kernel with $i$ at its
center, and $f'(k,i)$ with $k$ there: the kernel is inlined twice with its
ends exchanged, so that $\mathbf K$ is antisymmetric by construction, and
the verifier proves it (V8), as it proves every derived kernel of
`md_dist`. The two partial results of $\mathbf K$ return to the owners as
those of a pair sum do, by the rule of the copy of the positions
($G^\mathsf T$, Section 14.3), and `md-dist-owner-computes` applies to them
on the directed traversal with the premises of Section 14.4.

On the unique traversal the transposes of 14.3 give the copy by
themselves: the adjoint of the density's `halo_sum` is $\bar\rho$ for its
owned operand and $G\bar\rho$ for its ghost operand, and the force gather
takes the two as its gathered pair. On the directed traversal the density
is complete at the owners, and the derivative inserts the copy. Either way
the copy is of exactly the version of $\bar\rho$ computed in this
derivative, which the verifier checks (V2): a copy of another version is a
stale halo.

### 15.3 The program, written out

EAM with forces on the directed traversal, in the derivative's form (the
ops of Section 5 that build the halo and the relation omitted):

```mlir
%rho = md_dist.gather_pairs %n, %xo, %xg exchange(symmetric) { f(r) }
%e1p = md_dist.sum_particles gather(%rho) { F(rho) }
%e2p = md_dist.sum_pairs %n, %xo, %xg exchange(symmetric) { phi(r) }
// the derivative
%k:2 = md_dist.gather_pairs %n, %xo, %xg exchange(antisymmetric) { −phi'(r) d / r }
%rb  = md_dist.map_particles gather(%rho) { F'(rho) }        // ρ̄
%rbg = md_dist.halo_copy %rb, %h                            // ρ̄ to the ghosts
%g:2 = md_dist.gather_pairs %n, %xo, %xg gather(%rb, %rbg)
         exchange(antisymmetric) { −(ρ̄_i f'(r) + ρ̄_k f'(r)) d / r }
%f1  = md_dist.halo_sum %k#0, %k#1, %h                      // G^T
%f2  = md_dist.halo_sum %g#0, %g#1, %h                      // G^T
```

and the collection of $\mathbf f_1 + \mathbf f_2$. After
`md-dist-owner-computes` the two `halo_sum`s are gone and each gather has
one complete result; the copy of $\bar\rho$ stays, since the forces at an
owner need $\bar\rho$ at its neighbors. This is the stage boundary of the
[architecture plan](future-architecture-plan.md#eam-makes-the-stage-boundary-concrete),
derived rather than written.

### 15.4 A field read at neighbors by a pair sum

A pair sum may read a field computed from the positions at both particles
(the many-body energy of Section 14.5 reads $\psi = \sqrt\rho$ so). The
derivatives in the field at the center of a pair, $\partial_1 k(i,k)$, and
at the neighbor, $\partial_2 k(i,k)$, are different functions with no sign
between them, so the two-ended gather of Section 4.6, whose value at the
neighbor comes from an exchange contract, cannot carry them. On the
directed traversal the derivative takes the gather at the owners instead.

*Claim.* For a directed relation $D$ whose halo covers its cutoff and a
symmetric kernel $k$, the derivative of
$S = \tfrac12\sum_d\sum_{(i,k)\in D_d} k(i,k)$ in the owned value $a_i$,
through its own row and through every copy of it to the ghosts, is
$\sum_{k : (i,k)\in D_{o(i)}}\partial_1 k(i,k)$.

*Proof.* $a_i$ enters $S$ at the pairs $(i,k)$ of $D_{o(i)}$, as the
center's value, and at the pairs $(j,k')$ whose $k'$ is $i$ or a copy of
it, as the neighbor's value. By coverage (Section 4.3) the pair of $i$ and
$j$, $r_{ij} < r_c$, is in both forms once each. Since $k$ is symmetric,
$\partial_2 k(j,k') = \partial_1 k(k', j)$, and $k(k', j)$ is $k(i,k)$: the
distance is the same, the displacement $\mathbf x_{k'} - \mathbf x_j$
equals $\mathbf x_i - \mathbf x_k$, and the values at $k'$ and $j$ are those
of $i$ and $k$ (a copy carries its source's value, V2). The two halves are
equal, and their sum is the term of the claim. $\square$

*Premises*, checked where the derivative emits the gather, with the code of
the verifier: the relation is directed (its type), covered (V4,
`md_dist::verifyCoverage`), and the kernel symmetric (V8, proven again by
`md_dist::proveContract`). On the unique traversal the derivative refuses
with a diagnostic: the neighbor's half is computed only in the center's
domain, and returning it needs a two-ended gather whose value at the
neighbor is the kernel evaluated with the ends exchanged, a form the
dialect does not have yet.

### 15.5 Validation

`test/Integration/dist-eam.mlir`: an EAM potential of the analytic form of
Finnis and Sinclair [[Finnis1984]](references.md#finnis1984), with
parameters for the 64 particles of Section 12 in reduced units,

$$
f(r) = (r - 2)^2,\quad F(\rho) = -\sqrt\rho,\quad
\phi(r) = (r - 2)^2(3 - 2r + 0.4r^2)
$$

for $r < 2$ (each zero with its slope at the cutoff), distributed by
`md-distribute` and differentiated by `md-differentiate`. It runs on the
grids $1\times1\times1$, $2\times1\times1$, $3\times2\times1$ (uneven), and
$16\times1\times1$ (half of the domains empty), each with the unique
traversal, the directed traversal in the derivative's form, and the
directed traversal after `md-dist-owner-computes` (12 runs). The reference
is the same potential on one domain through `md_exec`, which agrees with a
Python reference written from the definitions and their closed-form
derivatives (`test/Integration/Inputs/eam_reference.py`). The largest
relative differences over the 12 runs, `double`, on the CPU:

| Quantity | Reference | Largest difference | Tolerance |
|---|---|---|---|
| Energy, $U = 31.8605$ | `eam_reference.py` (one domain) | $3.6\times10^{-15}$ | $10^{-10}$ |
| A force component, of $\max\lvert F\rvert = 2.2406$ | `eam_reference.py` (one domain) | $2.5\times10^{-17}$ | $10^{-10}$ |
| $\mathrm{tr}\,\mathsf W = 584.824$ | `eam_reference.py` (one domain) | $0$ | $10^{-10}$ |
| Energy | One domain | $1.8\times10^{-15}$ | $10^{-10}$ |
| Force component, of $\max\lvert F\rvert$ | One domain | $3.1\times10^{-15}$ | $10^{-10}$ |
| Virial component, of $\lvert\mathrm{tr}\,\mathsf W\rvert$ | One domain | $4.4\times10^{-16}$ | $10^{-10}$ |
| Six force components, of $\max\lvert F\rvert$ | Central differences of the distributed energy, step $10^{-5}$ | $2.3\times10^{-9}$ | $10^{-7}$ |
| The diagonal of the virial, of $\lvert\mathrm{tr}\,\mathsf W\rvert$ | Central differences in the three stretches of the cell, step $10^{-5}$ | $6.0\times10^{-11}$ | $10^{-7}$ |
| $\langle\mathbf F, \mathbf v\rangle$ of $U = \langle\mathbf c, \rho\rangle$ | $-\langle\mathbf c, J\mathbf v\rangle$, summed directly | $6.9\times10^{-15}$ | $10^{-12}$ |

The differences from central differences are those of the differences
themselves, the rounding of $U$ over $2h$. The last row is the adjoint
identity of the density: for $U = \langle\mathbf c, \rho(\mathbf x)\rangle$
with a field $\mathbf c$ and $J = \partial\rho/\partial\mathbf x$, the forces
on domains are $-J^\mathsf T\mathbf c$, carried back through the copy of
$\bar\rho = \mathbf c$ to the ghosts and $G^\mathsf T$, and their product with
a field $\mathbf v$ is compared with
$-\langle\mathbf c, J\mathbf v\rangle = -\sum_i c_i\sum_{j\ne i} f'(r_{ij})\,\mathbf d_{ij}\cdot(\mathbf v_i - \mathbf v_j)/r_{ij}$,
summed in the test over all pairs in the minimum image, with no relation,
no domains, and no derivative.

`test/Integration/dist-many-body-forces.mlir`: the many-body energy of
Section 14.5, whose pair sum reads $\psi = \sqrt\rho$ at both particles, on
the directed traversal (Section 15.4) on the grids $2\times1\times1$,
$3\times2\times1$, and $16\times1\times1$, in the derivative's form and
after `md-dist-owner-computes` (6 runs), against one domain: energy within
$2.4\times10^{-16}$, force components within $2.1\times10^{-15}$, virial
components within $5.2\times10^{-17}$ (tolerance $10^{-10}$); three force
components and the stretch along $x$ against central differences (step
$10^{-5}$) within $2.5\times10^{-9}$ and $7.1\times10^{-12}$ (tolerance $10^{-7}$).

IR tests: `test/Dialect/MD/Transforms/differentiate-eam.mlir` (the copy of
$\bar\rho$ to the ghosts and the force gather that reads it, on both
traversals; $G^\mathsf T$ in the derivative's form and its absence after
owner-computes, with the copy kept; the gather at the owners of the adjoint
of $\psi$). Negative tests: a stale copy of $F'(\rho)$, and none, in an EAM
program written in `md_dist` (`test/Dialect/MDDist/eam-invalid.mlir`, V2
and V1); a pair sum that reads a field computed from the positions on the
unique traversal, and densities whose kernel uses the displacement or reads
a field computed from the positions
(`test/Dialect/MD/Transforms/differentiate-eam-invalid.mlir`).

### 15.6 What changed in the code

- `md-differentiate` (`lib/Dialect/MD/Transforms/Differentiate.cpp`) walks
  the body of a potential on domains once, in reverse, keeping the negated
  adjoints of its values on domains and of the positions as lists of terms
  (complete, or the two partial results of a traversal): the map of the
  density's adjoint yields $-\bar\rho$, the force on the density, and the
  force kernel is written for it. The transfers are
  transposed by `AdjointOpInterface` as in Section 14.3, with the complete
  terms of a value added before its transpose, so that the density's
  `halo_sum` yields one $\bar\rho$ and one copy.
- No change to the dialect, the executor, or `md-dist-owner-computes`: the
  force gather of a density is the two-ended gather of Section 4.6 with a
  proven contract, and the adjoint of a field read by a pair sum is a
  gather with one complete result over a directed relation.
- No control-file key, Python API, or output changes.

### 15.7 What is left

- The derivative through a pair sum that reads a field computed from the
  positions on the unique traversal (15.4). The same missing form, a
  two-ended gather that evaluates the kernel at both particles, means that
  on the directed traversal the adjoint of such a field is emitted directly
  as the gather at the owners: a checked optimization (V4, V8, the proof
  of 15.4) without a switch, so that no derivative's form with $G^\mathsf T$
  is compared with it, as principle 3 asks. That form, and the comparison,
  come with the new gather.
- Derivatives in parameters or fields on domains; migration; MPI. The
  `mixed` mode and GPUs run through `md_exec` (Section 7.2, 7.4).
