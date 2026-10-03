# Distributed field contracts and development plan

Status: proposed architecture, 2026-10-03, D[md-dist-architecture].
This document specifies future work; it adds no dialect implementation,
control keys, defaults, file formats, or overwrite behavior. It refines
ML2–ML5 without changing the M2/M3/M4 milestone order in [roadmap.md](roadmap.md).

The [v0 specification](md-dist-v0.md) fixes the first implementation
boundary: shared `mdrt` types, field-state analysis, lexical accumulation,
and synchronous straight-line evaluation. It refines the broader gates
below without claiming implementation.

## 1. Purpose and boundary

`md_dist` verifies that each computation receives the required logical field
version at the required entities, and that its contributions complete the
correct logical result exactly once. Ownership, materialization, support,
and completion are its organizing concepts. A transport implements this
contract; it does not define it.

```text
md / dyn / external-potential contract
  -> differentiation and staged dependency analysis
  -> semantic dependency DAG
  -> legal joint plan
       -> md_dist: placement, views, maps, contribution completion
       -> md_exec: traversal, local computation and accumulation
  -> storage assignment
  -> physical dependencies: aliases, outstanding reads, protocol, visibility
  -> scheduling and transport lowering
  -> mdrt / upstream dialects / CPU and GPU backends
```

Start with an analysis and operation interfaces for the dependency DAG,
not another dialect. Both execution dialects realize the same plan.
`md_dist` produces subsets for existing `md_exec` loops; it does not add
`compute_interior` or `compute_boundary` loop families. A one-domain plan
must specialize to the current execution path before runtime setup.

## 2. Core contracts

Names below are proposed, not registered types or operations. Runtime
snapshots are SSA operands; types retain stable entity domains, element
shapes/types, and completeness categories. Shared types belong to `mdrt`;
`md_dist` and `md_exec` remain peers and use neutral interfaces.

| Contract | Meaning |
|---|---|
| `team` | Logical participants; distinct from MPI ranks and device IDs |
| `layout` | Immutable snapshot of authoritative entity owners and local indices |
| `field_view` | Materialization of a logical field version on an explicit layout and subset; never silently changes `!md.field` to mean rank-local storage |
| `transfer_map` | Each consumer slot maps to an owner entity and, for particles, periodic image; physical forwarding is a later choice |
| `coverage` | Evidence that candidates contain the requested semantic relation under a stated validity predicate |
| `contribution` | Partial values with target, reducer, scope, and producer identity; not a readable complete field |
| `md_dist.accumulate` region | Lexical scope binding semantic requirements to evaluation inputs; its completion terminator returns a complete owned field through the parent op; no new scope-handle type |

Data ownership, evaluation ownership, and accumulation destination are
separate. Coordinate updates create a new scientific version; halo refresh
materializes the same version elsewhere; sorting changes local indexing;
migration changes ownership; neighbor rebuilding changes candidates and
coverage. None of these identities substitutes for another.

## 3. Delivery gates

Each row becomes separate implementation issues and focused PRs. All rows
are planned. A gate passes only with recorded evidence, not because an op
or backend exists.

| Gate | Scope | Required evidence |
|---|---|---|
| DIST0 / ML2 | Contract types, scope verifier, in-process logical domains | Negative verifier tests, exact owner/replica routing and transpose properties; no MPI or GPU dependency |
| DIST1 / ML2 | Dependency extraction and synchronous directed LJ | Force, energy, and virial against an independent all-pairs reference; changing partitions, empty domains, migration, sorting, and coverage guards |
| DIST2 / ML2 | EAM stages, unique-pair alternatives, reverse contributions, one crossing topology term | Reject partial density before embedding; infer intermediate exchanges; finite differences and exactly-once completion |
| DIST3 / ML2–ML3 | Synchronous CPU MPI, then GPU transport | Matching participation, completion, coherent retry/failure, CPU/GPU mixed/double agreement; two GPUs before ML3 completion |
| DIST4 / ML5 | Storage hazards and asynchronous scheduling | Delayed-read and buffer-reuse tests; equivalent synchronous results; measured full-step overlap and single-GPU regression checks |
| DIST5 / ML3–ML4 | Whole-model ghost gradients, then visible stages and external VJP | Same artifact versus original backend; communication adjoint and replicated-energy seed checks |
| DIST6 / M4 | PME reference redistribution and later distributed mesh/constraints | Separate particle/mesh layouts, complete force/virial, topology constraints and molecular validation before production claims |

DIST5's whole-model path needs DIST2–DIST3 and ML1, not asynchronous DIST4.
Its staged path follows the single-GPU ML4 contract. DIST6 may begin after
DIST3 independently of learned models. Bounded performance selection follows
validated fixed plans; no autotuner is required to establish correctness.

## 4. Public interface and review scope

This planning change introduces no user-facing control or artifact schema.
Prototype tests construct IR directly. Runtime keys, transport selection,
plan serialization, restart portability, and the external potential ABI
need their own design review before implementation. Current scientific
precision and deterministic-mode contracts constrain every legal plan.

The first implementation PR should implement DIST0's field views, transfer
maps, and contribution completion, with parser/verifier tests and a small
reference executor. It should not promise distributed molecular dynamics
or a speedup. Sections 5–11 specify operations, verification limits,
integration points, numerical acceptance gates, and source evidence.

## 5. Operation semantics and accumulation

The following is a contract inventory, not final assembly syntax. Results
are complete synchronously at this level; transport initiation and waiting
appear later. Completion is relative to a consumer or scope, not a global
barrier.

| Operation family | Inputs and postcondition |
|---|---|
| `partition` / `redistribute` | Team, source layout, destination layout policy, and all live fields; produces a new layout and corresponding materializations without changing scientific values |
| `halo_map` | Layout, support requirement, candidates, and validity guard; produces a consumer map and coverage witness |
| `forward_halo` | Complete field view and compatible map; returns that same logical version available at the map's consumer slots |
| `reverse_accumulate` | Replica contributions and the map snapshot used by forward; returns routed owner contributions, still partial with respect to the enclosing scope |
| `accumulate` / `complete_accumulation` | Region binds independent semantic requirements and evaluation inputs; completion terminator checks contributions and the parent returns a complete field |
| `merge` | Adds values and symbolic work multiplicities; overlapping targets are legal, duplicated semantic work is rejected at completion |
| `collective` | Team, partial values, reducer, and result placement; produces one logical result, either owned or physically replicated |
| `classify_support` | A stage's actual inputs, available views, and support; returns disjoint, exhaustive execution subsets for existing loops |

A map is reusable across field versions only while its layout, entity and
image identities, candidate support, and runtime guard remain compatible.
It is not itself evidence that any payload is fresh. Each read checks its
required version and availability. A new layout invalidates cached bindings
to current execution, but an old immutable snapshot can remain live for a
consumer explicitly bound to it. Initially forbid migration, sorting, or
map rebuilding within one forward/backward evaluation; release that
restriction only with separately verified snapshot lifetime support.

An accumulation scope identifies a result version, entity subset, reducer
and identity, and the expected logical producers derived from the semantic
DAG. Planner-emitted producer IDs are checked against that DAG: an arbitrary
list of the contributions that happen to arrive cannot certify completeness.
Each evaluated pair/tuple or owned-center term has a declared ownership
rule, so duplicated evaluation is distinguished from duplicated addition.

A contribution may have inspection uses, but exactly one terminal route
into its scope's completed result. Routing consumes a provenance edge and
produces another; it cannot erase or duplicate that identity. Adding a
local contribution and its reverse-routed copy is rejected. Zero carries an
empty work ledger; multiple zero identities may be merged. Reinitialization
that discards required work fails completion, rather than a blanket ban on
multiple zero ops. Empty subsets produce
explicit empty contributions and still satisfy participation requirements.
Completing an intermediate field does not complete the enclosing force
scope: other potential terms may still contribute.

Disjoint-subset assembly and additive reduction are distinct operations:
assembly checks disjointness and exhaustive target coverage; reduction
checks producer coverage over possibly overlapping targets. Readiness of
interior and boundary results never licenses adding the same contribution
twice. A type marked `partial` helps reject reads, but does not provide
linear-resource semantics in MLIR; the scope verifier performs that check.

## 6. Verification and runtime trust boundary

The v0a verifier supports explicit scopes and straight-line distributed
stage DAGs with fixed layout/cell. Existing outer step loops and nested
local kernel arithmetic are separate. Team-uniform stage conditionals and
iteration-carried protocol contracts are later extensions. Unknown control flow or opaque support fails
with a diagnostic rather than an optimistic distributed lowering. An
iterative constraint solve must be a structured loop with participation and
termination rules, or a separately validated opaque operation.

| Obligation | Static check | Runtime/reference check |
|---|---|---|
| Freshness | Consumer's logical version and layout match its view and map | Handle epochs and debug payload/version tags agree |
| Coverage | Requested relation, builder kind, guard, cell convention, and exclusions are compatible | Trusted builder plus validity guard establishes candidate completeness |
| Completion | Scope provenance reaches every required result exactly once; no partial read or reset | Enumerated producers/targets and independent sums agree |
| Ownership | Entity domain, team, target and periodic-image convention match | Every global entity has one authoritative owner; map indices and images are valid |
| Participation | Matching team/channel operations and uniform control structure | Counts, collective signature, capacity and failure status agree before dependent communication |
| Lifetime | Saved maps and fields dominate consumers; physical hazards ordered | Artificially delayed transfers cannot read overwritten storage |

For a semantic relation $R$ and candidate set $C$, coverage requires
$R\subseteq C$. Also check post-filter enumeration multiplicity: duplicate
candidate entries can satisfy containment while double-counting interactions.
Every kernel still tests $R$'s predicate, including the
term's cutoff and exclusions. Skin changes $C$, not the physical potential.
General geometry is not proved by the verifier. Recognized builders and
runtime guards provide that proof obligation; tiny all-pairs tests independently
check their results. Topological support names tuple/parent IDs separately
from geometric support. Unsupported global or opaque dependencies are errors
unless an explicit supported global execution plan exists.

v0 halo payloads copy canonical coordinates; geometry applies periodic
displacements explicitly. Shifted coordinate materialization is a later
operation with its own cell pullback.

Replica identity includes global entity ID and periodic image where needed;
edge displacements retain the existing periodic convention. A larger halo
must neither merge distinct images nor count a minimum-image interaction
again. Tests include local periodic replicas in a single domain and narrow
cells. Migration preserves stable IDs, all live scientific state, and
constraint-group membership; sorting rebuilds affected index bindings.

Start with a team-wide OR of local rebuild/staleness predicates, followed by
one common rebuild/migration branch. Empty domains participate. Capacity
failure is discovered before an unsafe write or protocol launch; all
participants retry from the same uncommitted snapshot or report failure
coherently. This is coordinated application error handling, not a promise
of recovery from process loss.

Rank-local communication has team/channel protocol effects and explicit
ordering. Do not give it unrestricted `Pure`, DCE, CSE, or speculation.
A global logical transfer can be optimized only if the transformation
preserves all participants and paths. Ordinary field dependencies remain
separate from protocol ordering; no universal world token is required.

Diagnostics name the consumer, required/available versions, layout/map,
missing producer or coverage witness, and relevant protocol edge. Stable
stage/transfer IDs connect requirement, chosen plan, and runtime trace.
For example, a transfer record explains that neighbor embedding derivatives
are needed by boundary force evaluation, identifies their producer and map,
and records completion and a legal recomputation alternative. Dumps are
initially test/debug output, not a promised serialized artifact schema.

## 7. EAM and derivative contracts

EAM [[Daw1984]](references.md#daw1984) is the first nonlinear intermediate
case. In this section $\rho_i$ is density, $f_{s_j}$ its neighbor contribution,
$F_{s_i}$ embedding energy, $\phi_{s_i,s_j}$ pair energy, $N_i$ the exact
neighbor set, and $s_i$ species. The energy is
$E=\sum_i F_{s_i}(\rho_i)+\frac12\sum_i\sum_{j\in N_i}\phi_{s_i,s_j}(r_{ij})$,
where $\rho_i=\sum_{j\in N_i}f_{s_j}(r_{ij})$.
Differentiating this energy requires the complete
$\psi_i=F'_{s_i}(\rho_i)$ at both endpoints of each force interaction.

```text
Proposed directed plan (not parser syntax):
  coordinate view = forward_halo(owned coordinates, map)
  density contributions = md_exec pair traversal over owned centers
  density = complete_accumulation(density scope, density contributions)
  embedding derivative = md_exec particle traversal(density)
  derivative view = forward_halo(embedding derivative, map)
  force contributions = md_exec pair traversal(coordinate view, derivative view)
  force = complete_accumulation(force scope, force contributions)
```

Directed traversal completes each owned center's density locally and can
compute owned forces without reverse force communication. Directed pair
energy uses its defined half weighting; embedding energy is counted once
per owned center. The unique-pair alternative can produce partial density
and force on replicas; route each to its owner before completing its scope.
Embedding on partial density is illegal because in general
$F(a+b)\ne F(a)+F(b)$. Staging must follow extracted reads of the differentiated
expressions, not a hard-coded EAM communication recipe. Start with an
analytic smooth toy EAM for an independent NumPy and finite-difference
oracle; support for potential-file formats is separate work.

For a feature transfer map $T$, forward is $h_g=T h_o$ and reverse mode is
$\bar h_o\mathrel{+}=T^\mathsf{T}\bar h_g$. The reference executor tests
$\langle Tu,v\rangle=\langle u,T^\mathsf{T}v\rangle$, with duplicates,
empty maps, periodic identities, and multi-hop realizations. Reverse follows
the forward snapshot, including saved values and routing provenance.

A whole-model external contract evaluates owned-center energy from its
complete environment and returns contributions, including ghost gradients.
A staged contract additionally provides callable stage/communication hooks,
intermediate shapes and precision, backward entry points, and saved-value
lifetimes. Metadata alone cannot make an opaque model interruptible.
External AD remains responsible for neural tensor derivatives (D167).
Energy environment and the support required to finish an owned force are
specified separately; a larger environment or reverse contribution routing
are alternative plans, not a universal radius formula.

For $E=\sum_p E_p$, a replicated result represents one logical energy.
Seed its derivative once logically; do not seed every physical copy and
sum those seeds into a rank-count multiplier. Keep field adjoints, shared
parameter reductions, and replicated reporting scalars distinct. A synthetic
collective graph checks this before a learned model is required.

If a periodic coordinate is $x_g=x_i+Hn$, with cell matrix $H$ and integer
image $n$, its pullback includes
$\bar x_i\mathrel{+}=\bar x_g$ and
$\bar H\mathrel{+}=\bar x_g n^\mathsf{T}$.
The potential ABI declares whether the plugin or MDIR supplies this cell
term and converts virial/stress under the existing convention, exactly
once. Relation building and migration remain discrete management operations,
not differentiated operations. Coordinate/cell finite differences keep the
relation fixed away from cutoff/image discontinuities.

## 8. Storage, scheduling, and source integration

Static review baseline: MDIR `d1def6b` (2026-10-03). No `MDDist` source
directory or registered distributed operations exist in this checkout.
The following are implementation touch points, not capabilities already
provided by these files.

| Existing source | Current role | Planned extension |
|---|---|---|
| [MDTypes.td](../include/mdir/Dialect/MD/MDTypes.td) | `!md.field` assigns values to every particle in its set | Preserve semantics; introduce a separate local view and a field-like interface |
| [MDExecOps.td](../include/mdir/Dialect/MDExec/MDExecOps.td) | `MDExec_FieldOrBuffer` admits fields or buffers; loops realize traversal | Explicit supported subsets/views, without a second loop hierarchy |
| [Differentiate.cpp](../lib/Dialect/MD/Transforms/Differentiate.cpp) | Semantic energy derivatives | Expose versioned reads and contribution provenance to staged analysis |
| [ExposeValidity.cpp](../lib/Dialect/MDExec/Transforms/ExposeValidity.cpp) | Displacement/cell tests for neighbor validity | Team agreement before distributed rebuild; retain local predicate semantics |
| [AssignStorage.cpp](../lib/Dialect/MDExec/Transforms/AssignStorage.cpp) | `Scope::lastUse`; reuse pool keyed by buffer type and entity set | Capacity, memory-space/device, and layout-access compatibility; asynchronous lifetime protection |
| [Independence.cpp](../lib/Dialect/MDExec/Independence.cpp) | Conservative SSA/effect/alias reasoning | Preserve as an input to physical hazard analysis |
| [AssignStreams.cpp](../lib/Dialect/MDExec/Transforms/AssignStreams.cpp) | Marks storage-form reciprocal work, hoists it, and inserts joins | General scheduling is separate work, not already implemented here |

First assign storage to synchronous value semantics. Then reconstruct a
physical dependency graph including read/write alias hazards, protocol
ordering, and device visibility before creating overlap. An asynchronous
send's final SSA use does not release its input buffer. A later overwrite
waits for `input_released`; a receiver waits for `consumer_ready`, including
unpacking and visibility. One conservative `!mdrt.event` may guarantee both
initially; later ops can expose separate events with explicit guarantees.

Reuse may cross layout epochs when previous contents and accesses are dead,
capacity is sufficient, and the new access layout and device are compatible.
Conversely, the same entity set no longer proves the same local size.
Later planning may spend memory on pack buffers or multiple field buffers
to remove hazards. Report those allocations/copies under D18; do not hide
them. Nonblocking transport does not itself prove overlap: backend capability
records include progress, device-buffer access, stream ordering, and copies.

Proposed code organization puts shared types under the existing MDRT
dialect, and ops/verification in `include/mdir/Dialect/MDDist/` and
`lib/Dialect/MDDist/`. Neutral payload/provenance/dependency interfaces and
analysis remain outside either execution dialect and transport backends;
add a small CPU reference
executor; and separate transport conversions/runtime support. Add exact
file names with the implementing PR. Registration and field-like changes
precede storage extensions; avoid concurrent broad rewrites of the driver.

## 9. Legal planning, PME, and topology

The first planner is a deterministic legalizer: it always produces a
supported fixed plan or a diagnostic. Candidate enumeration is bounded:
directed versus unique pairs; complete owned EAM density versus partial
reduction; whole environments versus visible-stage feature exchange.
Every candidate preserves scientific cutoffs, exclusions, precision,
reassociation/determinism policy, result placement, and contribution scope.
Approximation or lower communication precision requires its own scientific
contract; it is not a small cost penalty.

Measure the physical critical path plus amortized rebuild/migration cost,
not merely compute time plus message time. Include pack/unpack, progress,
capacity, peak memory, buffer waits, and the slowest domain. Memory is a hard
constraint. Replay the same scientific snapshot when comparing candidates;
commit one trajectory advancement. Dumps explain rejected alternatives and
why each required transfer exists. Tuning must remain optional.

PME needs distinct particle and mesh ownership and redistribution. A first
reference plan gathers inputs to a chosen PME participant, invokes the
existing solver, and redistributes force contributions; it tests the
contract but makes no scalability claim. Distributed FFT/mesh communication
follows in a separate implementation. `shard` may be useful below the mesh
contract and `mpi` below transport; neither is required in the first core.

Bonded tuples, constraints, and virtual-site parents require explicit
topological support. Initially keep supported small constraint groups on
one owner (D83), rather than treating a whole molecule or the nonbonded
radius as a general solution. Oversized or incompatible groups are rejected
until a crossing solver is implemented. Production Amber distribution
requires PME, exclusions, constraints, virtual sites, virial, and restart
validation; successful LJ/EAM or ML3 is insufficient. Restart starts with
the documented same-plan scope; rank-count portability needs stable-ID
state reconstruction and its own validation.

## 10. Implementation slices and acceptance protocol

DIST0 should be deliberately small: types/handles and explicit scopes,
fixed logical domains, map construction from test tables, synchronous
forward/reverse, and completion. Use direct IR fixtures and an executable
CPU reference path to exercise actual ops. The oracle must independently
enumerate expected owner/replica values rather than reuse production map
construction. Split registration/types and executable reference support
into successive focused PRs if needed. Neither needs a control-file key.

The [v0 A–D slices](md-dist-v0.md#5-verification-and-implementation-slices)
permit fixed-state LJ/EAM before temporal migration work; the DIST gates
remain the broader acceptance areas.

DIST1 adds stage extraction from existing pair operations, guarded geometric
map builders, and synchronous LJ through existing loops. First freeze a
layout for one evaluation, then add step-boundary sorting/migration and
rebuilds. DIST2 adds nonlinear stages, reverse routing, and topology support.
This order makes correctness executable before transport complexity.

| Validation layer | Cases and acceptance |
|---|---|
| Verifier | Stale version/layout, missing transfer, insufficient coverage, partial nonlinear read, duplicate/missing producer, reset during accumulation, incompatible reducer/team, divergent participation; assert consumer-specific diagnostics |
| Map properties | One/two/multiple logical domains, empty and uneven domains, repeated replicas/images, sorting, migration, capacity growth, forwarding; integer payload/routing exact, floating adjoint identity within predeclared roundoff bounds |
| Scientific oracle | Independent all-pairs LJ and analytic EAM energy/forces/virial; unique/directed plans, cross-boundary topology term, coordinate and cell finite differences away from discontinuities |
| Dynamics | Same equilibrated snapshot, partition variations, short NVE drift versus one domain; compare conservation and observables rather than require identical chaotic trajectories |
| Transport | Two CPU ranks, then two GPUs; mixed/double, delayed completion, empty ranks, coherent capacity retry; synchronous/asynchronous agreement |
| Learned/mesh | Original model backend with identical weights/outputs; seed multiplicity and parameter gradients; PME against existing single-domain solver and independent molecular oracle |
| Performance | Single-domain specialization has no distribution setup in the hot path; locked GPU 0 timings against main on identical Amber systems; record every slowdown, memory, full-step time, and tuning cost |

Before each numerical implementation starts, commit fixture definitions,
units, oracle identity, and an absolute/relative tolerance for each quantity
and precision: $\lvert q-q_{\rm ref}\rvert\le a+r\lvert q_{\rm ref}\rvert$.
Report force maximum-component and RMS errors as well as energy and virial;
use the absolute term near zero. Finite differences sweep step sizes to
separate truncation and rounding. Do not invent universal tolerances here
without fixtures. Exact IDs, routing counts, and collective seed factors
are tested exactly. CPU and GPU results must be listed separately; an
unavailable target remains an open gate. Intermittent tests report all
repeats/failures. Statistical checks state sample length and uncertainty.

Run the required local suite under GPU 1's lock before marking an
implementation ready. Timings use GPU 0's lock after checking usage. One-node
two-GPU correctness precedes scaling claims; four-GPU and multinode work
requires suitable resources and respects this machine's two-GPU limit.
Example trajectories follow minimization, NVT, restrained NPT, production;
small fixed-configuration oracle fixtures are not production tutorials.

## 11. Source evidence and limits

These are source inspections, not executions or performance measurements.
Pinned revisions below were retrieved on 2026-10-03. No external code is
copied or introduced as a dependency.

| Source inspected | Observation and design use |
|---|---|
| [PETSc sf.c, e76d590](https://github.com/petsc/petsc/blob/e76d59019036c134977c7e10c741a0a740f58be4/src/vec/is/sf/interface/sf.c) | `PetscSFSetGraph` separates root/leaf mapping from `PetscSFBcastBegin/End` and `PetscSFReduceBegin/End`. Reuse the owner–replica abstraction, not its API as MDIR semantics. |
| [LAMMPS pair_eam.cpp, e891a3e](https://github.com/lammps/lammps/blob/e891a3e10973c1a729e391a0aefaa02fd70f8c0f/src/MANYBODY/pair_eam.cpp) | `PairEAM::compute` reverse-communicates density with Newton enabled, computes embedding, then forward-communicates its derivative before force evaluation. This establishes prior implementation of the staged exchange. |
| [chemtrain comm.py, b256dc9](https://github.com/tummfm/chemtrain/blob/b256dc920aade9c947601341fb4390983abb3500/chemtrain/deploy/comm.py) | `_communicate` has a custom VJP invoking the reverse target; `_reduce` has a separate transpose target; `_call_ffi` declares side effects and threads a token. Differentiable communication is already implemented; see also [[Fuchs2025]](references.md#fuchs2025). |
| [MLIR shard execution model](https://mlir.llvm.org/docs/Dialects/Shard/#purity-and-execution-model) | Pure collectives have explicit SPMD execution assumptions; send/receive are not pure. Importing the trait without its execution model is unsound. This is documentation evidence, not a pinned implementation audit. |

The research objective is deriving and verifying alternative communication
and traversal plans from classical/learned semantics. Owner–replica maps,
EAM intermediate exchange, and communication VJPs individually are prior
art. Transport portability needs multiple realizations; good scaling and
novelty require further evidence.
