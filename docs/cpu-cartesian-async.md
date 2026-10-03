# Cartesian CPU LJ and asynchronous execution

Decision: D[cpu-cartesian-async]. This extends the opt-in
[hybrid CPU snapshot tool](cpu-hybrid-lj.md). It does not change the production
`mdir run` pipeline. The implementation is a fixed-layout, fixed-cell force
evaluation with a restricted executable scheduling IR. It is not yet a
production distributed simulation driver.

## User interface

Build with `MDIR_ENABLE_MPI=ON`. Existing snapshot input, precision, threads,
SIMD width, numerical stdout, and no-overwrite behavior are unchanged.

| Option | Default | Meaning |
|---|---|---|
| `--grid=auto` | `auto` | Search ordered factor triples of the MPI rank count |
| `--grid=Px,Py,Pz` | — | Explicit positive grid dimensions, product equal to rank count |
| `--halo=sync` | `sync` | Blocking coordinate payload transfer before both compute stages |
| `--halo=async` | — | Nonblocking payload transfer around interior computation |
| `--repeat=N` | `1` | Replay an unchanged snapshot N times; not trajectory steps |
| `--emit=dist` | — | Print the verified plan and its `md_exec` kernel; do not evaluate forces |

`--emit=source` prints generated source before parsing; `--emit=loops` and
`--emit=llvm` show the compiled kernel after the host schedule has been
extracted. They do not represent a general distribution-to-MPI lowering.
All emit modes currently perform partition and routing metadata setup.
There are no new TOML control keys. Diagnostics go to stderr.

## Grid selection and ownership

The planner enumerates all ordered integer triples whose product is the rank
count. For each axis, divide the physical box length by its grid dimension.
The score is the product of the expanded subdomain lengths divided by the
corresponding full box lengths. An axis split among ranks expands by twice
the cutoff, capped at the full box length; an unsplit axis stays full length.
This normalized calculation avoids overflow for large finite box dimensions.
It estimates a rectangular halo at uniform density, not measured communication
cost. Equal scores prefer more divisions in x, then y. Explicit grids bypass
this heuristic but must satisfy the rank-count constraint.

The rank mapping is `(cx * Py + cy) * Pz + cz`. Canonical coordinates select
half-open cells with `floor(x[k] / L[k] * P[k])`, clamped against upper-end
rounding. Rank IDs are not reordered. Root scatters owned IDs and coordinates;
this root input and scatter are reference-driver limitations.

This follows the established practice of factor-grid selection and geometric
communication estimates. See [MPI_Dims_create](https://www.open-mpi.org/doc/v4.0/man3/MPI_Dims_create.3.php),
[LAMMPS processor grids](https://docs.lammps.org/processors.html), and the
[GROMACS decomposition setup source](https://github.com/gromacs/gromacs/blob/main/src/gromacs/domdec/domdec_setup.cpp).
The implementation here is independently written. MPI_Dims_create alone has
no box-length or cutoff input and therefore does not supply this score.

## Checked fixed routing map

Each owner tests each owned particle against each other destination cell.
For each axis it computes the shortest distance to that cell over shifts
minus one, zero, and plus one box length. A particle is sent if all three
axis distances are at most the cutoff plus a conservative floating-point
margin (`32 * float_epsilon * box_length`). The force kernel still applies
its scientific spherical cutoff; rectangular overcoverage does not change it.

The map is direct: it includes face, edge, corner, and farther domains when
the cutoff spans multiple cells. A destination receives a given global ID
once. Dimensions of size one or two do not generate duplicate periodic
routes. Payloads are canonical coordinates; kernels calculate minimum-image
displacements. The existing cutoff-less-than-half-each-box-edge restriction
remains. Multiple physical images are not independently enumerated.

Counts and IDs are exchanged synchronously to establish the immutable map.
Owned coordinates precede ghost slots. Integer count/capacity overflow is
rejected before payload communication. This setup uses all-to-all metadata
and tests all destinations: it is a correctness baseline, not a scalable
neighbor-discovery implementation.

## Executable intermediate representation

The generated module contains the existing compiled pair function and:

```mlir
md_dist.reference_plan [2, 2, 2] {
^bb0(%layout: !mdrt.layout<@atoms>, %map: !mdrt.transfer_map<@atoms>):
  %event = md_dist.halo_start %layout via %map
    : !mdrt.layout<@atoms>, !mdrt.transfer_map<@atoms> -> !mdrt.event
  md_dist.dispatch @evaluate "interior"
  md_dist.halo_wait %event : !mdrt.event
  md_dist.dispatch @evaluate "boundary"
}
```

The host binds the layout and map block arguments to its checked immutable
runtime snapshots. The grid is static metadata for this invocation. The
existing `mdrt.event` represents logical completion; it is not an MPI request.
The plan has implicit reference-driver field bindings through a fixed kernel
ABI: counts, neighbor entries, coordinates, forces, and ten scalar totals.
It is not yet an independently serializable runtime state or a general field
view ABI. Dispatch invokes the same compiled `md_exec.pair_for` kernel on two
checked subsets; it defines no new particle-loop arithmetic.

`mdir-opt` parses and verifies this IR. The CPU reference executor extracts the
verified sequence, erases the plan before ordinary kernel lowering, and
interprets its actions around JIT invocations. MPI calls are host C++ calls;
there is no `md_dist` to upstream `mpi` conversion pass in this implementation.
The driver generates its own plan and does not accept arbitrary user plans.

The region verifier requires matching particle-set types for its layout/map,
exact use of those bindings by start, exactly one event use and matching wait,
one interior and one boundary dispatch to the same fixed-ABI kernel, and
completion before boundary dispatch. It rejects unknown operations, control
flow, duplicate/missing stages, and unsupported kernel signatures. Operations
have conservative effects; a local DCE cannot remove communication. This is
not a general alias, field-origin, geometry-coverage, or contribution verifier.
The checked host builder remains a trust boundary. Matching the function ABI
does not prove that an arbitrary callee obeys the subset access contract.
Execution is restricted to the driver-generated kernel; `mdir-opt` validation
alone does not authorize executing an arbitrary external function in a plan.

## Async memory and subset contract

Async payload start posts all nonempty receives before sends using
`MPI_Irecv`/`MPI_Isend` with tag 17 on the prototype's world communicator.
Sync payload transport uses `MPI_Alltoallv`; timing differences therefore
include transport-algorithm differences, not only overlap. Every peer uses a
separate slice of persistent send/receive packing storage. The send pack is
independent of coordinate storage and never changes while requests are live.
All MPI calls are on the initializing host thread (`MPI_THREAD_FUNNELED`).
This standalone SPMD prototype requires identical control options on all
ranks; it does not yet validate MPMD protocol-option agreement. A production
embedding must use an isolated team/context and a checked participation
contract.

Wait completes every receive and send, then copies received coordinates into
ghost slots. Its postcondition includes consumer readiness and send-storage
reuse. It is local completion, not a global barrier. Metadata exchange stays
synchronous. In sync mode the plan waits immediately after start.

A center is interior only if its distance from both cell faces in every split
axis exceeds cutoff plus the margin. Unsplitted axes impose no boundary.
Interior rows search owned neighbors only; boundary rows search owned
and ghost neighbors. Exact cutoff filtering occurs in the kernel. Ghost
coordinates are initialized to NaN before communication and not read by the
interior stage. This conservative split is stage-specific to short-range LJ.

Each stage uses zero counts for centers outside its subset. The current bridge
runs the pair kernel twice, stores separate stage forces and totals, and adds
them after both finish. This avoids overwriting the other stage's contribution,
but adds traversal and storage overhead. Candidate rows use periodic spatial bins with cell widths at least the cutoff
plus a conservative margin. A center visits adjacent bins with periodic bin
deduplication, then sorts local indices. Bin storage is sparse; the padded
neighbor matrix width is the maximum candidate count among selected centers.
This avoids an unconditional all-pairs matrix, although concentrated systems
can still require quadratic work/storage. Cells and rows are rebuilt for each
stage and replay. Production still needs skin validity, row/map reuse, compact
subset iteration, and buffer reuse. Safe nonblocking scheduling alone
is not evidence of a speedup.

## Measurements and validation

Stderr reports rank-maximum evaluation and per-action seconds and the global
interior-center count. Evaluation timing excludes JIT compilation, partition,
map construction, packing, global force/energy collection, and final stage
merging. It includes row preparation after an empty-row OpenMP warm-up.
Per-action maxima may belong to different ranks and must not be summed into a
critical path. MPI progress during computation is implementation-dependent;
these diagnostics are not full-step or steady-state throughput measurements.

`scripts/validation/cpu-cartesian-lj.py` uses an independent unordered-pair
Python oracle with explicit periodic-image search. It checks force, energy,
and virial for internal and periodic faces/edges/corners, empty domains,
empty systems, elongated boxes, 1/2/4/5/8 ranks, explicit grids, and auto grids.
Sync/async outputs must match exactly at a fixed configuration. Oracle
tolerance is `2e-11 * (1 + abs(reference))` in double and `3e-5` in mixed.
Negative IR tests exercise incomplete/duplicate events, premature boundary,
invalid subsets, unsupported operations, and incompatible kernels.

Current validation: 96 Cartesian configurations passed, including three
async replays versus one synchronous evaluation at each configuration. Cube
energy reference: `-103.3937675287093`. Maximum absolute component error across
energy, force, and virial was `4.41e-13` in double and `2.52e-5` in mixed.
Five invalid CLI configurations were rejected. The CPU-visible regression
suite passed 190 tests with 69 unsupported (GPU hidden), including ten plan
roundtrip/negative tests. The broader 216-case SIMD/thread matrix also passed, with 24 energy finite differences checking the oracle force convention
and four invalid snapshots rejected; generated forces are separately compared
with that oracle. The GPU-inclusive serial regression suite passed 254 tests with five
unsupported opt-in tests (four sanitizer checks and the scale suite), under
the GPU 1 lock. This validates existing GPU paths, not distributed GPU transport. No GPU transport or performance
improvement is claimed.

## Path toward production

The next checkpoints are reusable spatial neighbor lists and maps, explicit
runtime validity agreement, and repeated force evaluations; then migration and
reindex of every live field between evaluations. Velocity Verlet and trajectory
energy-drift checks require a separately documented simulation input contract.

Bonded ghost exchange is **not implemented**. It needs topology-derived
support, tuple evaluation ownership, and remote force contributions returned
to their owners. A bond crossing a domain beyond the LJ cutoff is the first
required test; angle/dihedral terms, exclusions, and constraints follow.
A spatial LJ map alone cannot claim topology completeness.

Distributed PME, constraints, barostats, learned models, GPU transport,
UCX/NVSHMEM, general async capabilities, and protocol projection remain future
work. Existing production simulations do not select this experimental path.

### Fixed-snapshot replay

`--repeat=N` (default 1, positive integer) replays force evaluation on the
same scientific snapshot and immutable map. It is not a timestep count.
Every replay poisons ghost slots again and must complete all requests before
reusing buffers. Output is the last evaluation, not a sum over repetitions.
Before timing, an empty-row kernel invocation initializes the OpenMP path.
Reported seconds are per-rank means over repetitions followed by a rank
maximum. This removes JIT and first-use OpenMP costs, but still excludes map
setup and global result collection; it cannot be reported as trajectory speed.

## Upper semantic contracts versus this reference plan

`md_dist.reference_plan` is a selected, restricted execution plan. Its fixed
pair-kernel ABI and interior/boundary stages must not define the upper semantic
IR. The upper contract separates four relations:

| Relation | Meaning |
|---|---|
| State placement | Authoritative materialization of a logical field value |
| Replica availability | Locations that can read that same logical value |
| Work placement | Participant responsible for a semantic interaction/work slice |
| Contribution destination | Owner of a result slice, reducer, and completion requirement |

State ownership does not imply work ownership. A worker may own neither
endpoint of a pair. Energy and virial can have reduction placements distinct
from particle forces. Access permission is a separate memory-lifetime contract;
a replica is not authoritative state ownership.

The intended planning direction is semantic requirements, then work placement,
then required materializations, then contribution routing/completion, then
physical scheduling. A coordinate halo is one realization of remote input
requirements. Reverse exchange is one realization of accumulation; it is not
the definition of accumulation. Owner-directed duplicate pair evaluation and
unique-pair evaluation plus remote reduction must realize the same independent
semantic requirement. The planner may choose only within the declared
floating-point/reassociation policy.

The current executable implements only the owner-directed choice. It has no
general requirement extraction, field-origin analysis, contribution ledger,
work-ownership search, remote reduction, or alternative transport. Its layout
and map bindings describe particle snapshots only. General entity domains
(meshes, tuples, features), field views, and contribution types need their own
verifier/interface implementation before this can become the upper IR.

For that implementation, runtime layout and map identities belong in SSA
operands; types carry stable domain/payload/reducer categories. Required work
comes from the semantic program, independently of generated producers.
Completion checks both contribution multiplicity and destination readiness.
Unsupported support/mapping expressions are rejected rather than inferred.
The first supported mapping families should include owner-centered work,
unique unordered pairs, and explicit topology tuples. Each needs a checked
builder or trusted external contract and runtime guards where static proof
cannot establish geometry or ownership.

This boundary permits an MPI send/reduce implementation and a future
one-sided accumulation implementation without putting MPI rank/tag/request
semantics into the upper contracts. It does not claim that every MPI operation
has an equivalent NVSHMEM primitive or identical completion guarantees.

### PP/PME as a required architecture test

The upper IR must distinguish entity domains (particles, tuples, mesh points),
work domains (short-range, bonded, spread, FFT, interpolation, integration),
logical worker teams, and physical resource bindings. A work domain is not a
communicator, and a worker need not correspond one-to-one to a rank or GPU.
PP and PME teams may overlap, be disjoint, or be colocated on shared resources.
None of these choices changes the scientific dependency graph.

A reference PP/PME graph must preserve these boundaries:

1. Materialize the required coordinate/charge versions at PME workers while
   PP workers read their own materializations of the same logical inputs.
2. Spread charges to a new mesh field. Spreading is computation, not
   redistribution of an unchanged field.
3. Redistribute mesh values between layouts as required by FFT stages.
4. Compute reciprocal work, inverse transforms, and interpolation to particle
   force contributions. These are new values or contributions.
5. Route contributions to the force-result placement, independently of state
   authority and work placement. Complete the semantic force requirement only
   after PP, reciprocal, bonded, and other required terms have arrived.
6. Allow integration only with that completed result.

This is a design requirement, not implemented distributed PME. Scalar energy
replication must not create multiple logical energies or duplicate AD seeds.
Collective participation/order belongs to each involved team; a world barrier
is not the meaning of force completion. An FFT transpose is a value-preserving
layout transformation, while forward/inverse FFTs change logical values.

The planner objective is the scheduled critical path including input/output
redistribution, pack/unpack, progress, memory hazards, and resource contention.
Taking the maximum of PP and PME times is only an approximation when their
resources and execution overlap sufficiently. Independent work may still
contend for CPU cores, memory bandwidth, GPU streams, or a NIC.

A future reference implementation can gather PME inputs to one logical worker,
call the existing reciprocal implementation, and return contributions before
attempting distributed FFT. That would exercise distinct work/result placement
without claiming scalable PME. Mesh reuse of upstream `shard` is considered
at that layer, not imposed on particle replica maps.

### Replication and publication are separate contracts

A unique owner per particle is a restriction of the current Cartesian runtime,
not an axiom of the upper IR. Availability is per logical field value, entity,
and representation and can name zero, one, or many materialization locations.
A fully replicated immutable snapshot is ordinary placement. Ghost presence
alone does not establish that a particular field version is readable there.

Authority specifies how a new logical version is published. Candidate policies
include an exclusive updater and replicated execution with an explicit
agreement/publication rule. Multiple materializations do not grant independent
conflicting writes. Immutable SSA values themselves need no mutable owner;
physical buffer access authority is a separate lifetime/effect question.
Dynamic versions and epochs are represented by SSA values/provenance, not
runtime SSA operands embedded in MLIR type parameters.

Logical field version, layout/index snapshot, and validity epoch are distinct.
An old version remains valid for its old consumer. A new coordinate value can
reuse a routing map only if a current support witness permits it; updated
payloads do not prove geometric coverage. Derived neighbor lists and pack/maps
are caches with explicit validity requirements, not canonical state. Changing
representation must distinguish encoding/layout changes that preserve a value
from numerical computations such as spreading or Fourier transforms.

Required architecture sanity checks are replicated-data, spatial, and force
(or interaction) decomposition. The same semantic force requirement must be
realizable in each case with different data/work placements and reduction
routes. Fully replicated coordinates do not by themselves dictate a final
broadcast: every worker might compute the same new state, or one publisher
might distribute it. The publication contract chooses between these cases.

Data/work placement graphs alone are insufficient for automatic synthesis.
The compiler also needs explicit access/support relations, work multiplicity,
reducer and floating-point policy, permitted representation transformations,
cache validity, and participant/progress/completion contracts. Missing facts
must produce a diagnostic or an explicitly supported conservative plan. The
initial Cartesian runtime does not implement this general synthesis.

The 1,728-particle replay check on eight ranks (`2,2,2`, two threads, double,
ten replays) found 64 interior centers and exactly matching sync/async stdout.
Stage timing instrumentation ran successfully; runs shared CPU resources with
validation jobs, so these numbers are not used to claim a speedup.

Tracked next work: [general placement/completion contracts (#37)](https://github.com/vv137/mdir/issues/37)
and [temporal validity/migration (#38)](https://github.com/vv137/mdir/issues/38).
