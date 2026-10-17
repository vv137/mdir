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
(D276).

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
| `md_exec` (execution) | `pair_for` over a neighbor structure, with the minimum image of a cell, over all rows of the positions; `particle_for`; `tuple_for`; neighbor builds and validity (`ExposeValidity`, D-series of M0); storage assignment | A pair loop over owned centers with neighbors among owned and ghost rows, in coordinates that carry their images (Section 7.2) |
| `mdrt` (runtime ABI) | Neighbor structures, cells, permutations; `!mdrt.event` declared without semantics | The transport ABI of Section 9 |
| `dyn` | Kick and drift, programs of one step | Migration between steps (Section 8) |
| Runtime | `libmdrt` (host), `libmdrt_cuda` | `mdrt_dist`, the in-process transport |

Nothing distributed is implemented. Architecture Section 8.1 sketched
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

## 7. Lowering a pair loop over owned and ghost rows

### 7.1 Reference executor

Section 6: for each domain, for each owned $i$, for each local row $k \ne i$,
$\mathbf d = \mathbf x_i - \mathbf x_k$ and the cutoff test; the unique traversal
skips $k$ unless $\mathrm{id}(i) < \mathrm{id}(s(k))$, and adds $\sigma\,k$ to
the row of $k$, owned or ghost.

### 7.2 Through `md_exec` (stage 1, next part)

A domain's work is ordinary `md_exec` over its local rows, with two
changes, both stated in the ops:

- `pair_for` and `build_neighbors` take a *center extent* (the owned rows)
  apart from the *row extent* (owned and ghost rows), the split that #34
  made in its prototype. Rows of the list are built for centers only; their
  entries are any local rows.
- The geometry is `images(explicit)`: displacements are plain differences,
  since ghosts carry their images. The cell is still known to the loop (for
  the virial of a strain and for a later skin test), but not used for a
  minimum image.

With these the GPU path is the existing one, per domain; the transfers are
between the domains' buffers (Section 9).

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
  (the existing condition with the scale of the cell, [neighbors-m0.md](neighbors-m0.md)).
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
rules of the AD (Section 11; built, Section 14), EAM, the schedule print and trace, the
`md_exec` path (Section 7.2), migration and `covered` with a trajectory
(Section 8), MPI on CPU ranks, then GPUs.

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
`gather_pairs` or `map_particles` of the positions (the next part, EAM);
and derivatives in a parameter or a field (`derivative(n)`).

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
relative (tolerance $10^{-10}$). Its derivative is the next part.

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
