# Topology transfer in the Cartesian CPU reference executor

D[cpu-bonded]. Experimental CPU implementation; depends on D[cpu-temporal].

`mdir-cpu-lj --bonds=FILE` adds harmonic terms to the existing LJ calculation.
Each nonempty file row is `id_i id_j k r0`; the potential is
$U_b=\tfrac12 k(r-r_0)^2$. IDs must exist, endpoints must differ, unordered
pairs must be unique, and `k` and `r0` must be finite and positive. Distances
use the existing fixed orthorhombic minimum-image convention, independently
of the LJ cutoff. No automatic exclusions or 1-4 scaling are introduced.
This is an explicit additive test Hamiltonian, not a molecular force-field
import format. No production TOML keys or output files change.

The owner of the smaller endpoint ID evaluates each bond once. A separate
ID-indexed topology map gathers its remote endpoint, even beyond the spatial
halo. Forward positions and reverse force contributions use the same epoch's
route. The larger endpoint's contribution returns to its state owner; local
contributions are added locally. The final force is not ready until both LJ
center subsets and all bonded contributions are applied. Migration rebuilds
the owner directory, local bond indices, and topology routes before reuse.

The first implementation replicates the bond table and gathers the ID-owner
directory at epoch construction. It uses a generated CPU bond kernel and
separate per-bond output buffers to avoid concurrent endpoint scatter races.
This is a correctness implementation, not a scalable directory algorithm.
The existing restricted `md_dist` plan explicitly schedules topology
transfer, bond dispatch, and reverse completion; general semantic contribution
verification remains a separate task.

Validation covers mixed/double, synchronous/nonblocking transport, 1/2/8
ranks, remote partners outside the LJ cutoff, repeated bond endpoints,
periodic boundaries, moving ownership, and invalid input. The independent
oracle checks energy, forces, virial, finite differences, and final NVE state.
Angles, dihedrals, constraints, production topology import/exclusions, and
GPU transport remain unsupported.

## Implementation boundary and data representation

`Bonded.cpp` owns the host-side directory, routes, and contribution buffers.
`BondedKernel.cpp` generates `@evaluate_bonds`; `main.cpp` inserts the kernel
and topology phases into the same parsed and verified module as the LJ plan.
After plan verification the host extracts its ordered actions and erases the
plan before lowering arithmetic. This is executable scheduling of a restricted
reference plan, not a general MPI dialect conversion.

Rank zero reads the bond file; blank lines are ignored and other lines must
have exactly four fields. Comments are not part of the format. Both endpoint
orders mean the same unordered bond and a duplicate in either order is an
error. The validated table is broadcast once. Nonzero ranks do not read the
file. Root selection of a bond file determines whether the optional phases
are enabled for the entire team.

At each layout epoch, all ranks gather owned IDs and construct the same
ID-to-owner directory. The smaller-ID endpoint's owner takes the bond's work.
It deduplicates required remote IDs by source rank and exchanges requests with
those owners. Owners map requested IDs to owned local indices. A global count
checks that the distributed work count equals the source bond-table count;
unique validated source pairs and deterministic owner selection establish
one evaluator per bond in this implementation. This is not proof of arbitrary
user kernel mathematics.

The topology coordinate view has owned positions followed by requested
replicas. It is separate from the spatial LJ view, even when a partner appears
in both. A topology requirement never expands the LJ cutoff or inserts a
bonded-only ghost into LJ candidate rows. Directory/map rebuild follows the
same conservative epoch boundary as the temporal driver, including support-only
rebuilds. Every evaluation refreshes coordinate payloads regardless of skin
reuse. The fixed-layout assumption applies only within one evaluation.

## Generated kernel ABI and numerical contract

| Argument | Shape | Meaning |
|---|---|---|
| endpoints | `memref<?x2xi32>` | Local topology-view indices in canonical endpoint order |
| parameters | `memref<?x2xf64>` | `k`, `r0` per locally assigned bond |
| positions | `memref<?x3xf64>` | Owned plus topology replica coordinates |
| contributions | `memref<?x6xf64>` | Three force components for each endpoint, one row per bond |
| results | `memref<?x10xf64>` | Energy followed by row-major 3-by-3 virial, one row per bond |

The generated `scf.parallel` loop lowers to OpenMP through the existing CPU
pipeline. Each iteration writes a separate bond row; no floating-point atomic
scatter is needed. This first bonded loop does not have the explicit neighbor
SIMD transformation used by LJ. Bond results are reduced/scattered by the host
in source-table order and receive-rank order. No rank-count-independent bitwise
reproducibility is promised.

Minimum-image displacement is computed in f64 before conversion to the chosen
pair arithmetic type. In mixed mode, bond parameters, distance, energy, force,
and virial arithmetic use f32, then results widen to f64. Double uses f64
throughout. Coordinate/state transfer and accumulated forces remain f64.
Parameters must remain positive and finite in the selected arithmetic type.
Coincident endpoints are rejected before kernel execution; nonfinite results
abort before completion. Image-boundary differentiability is not claimed.

With displacement from the second endpoint to the first, the contribution is
$\mathbf F_i=-k(r-r_0)\mathbf d_{ij}/r$, with
$\mathbf F_j=-\mathbf F_i$. The virial is
$\mathbf d_{ij}\otimes\mathbf F_i$, consistent with the existing LJ output.
These follow by differentiating the declared harmonic potential and a uniform
strain, respectively; no endpoint half weight is applied to a unique bond.
The minimum-image and force/virial conventions follow the existing MDIR
[[AllenTildesley2017]](references.md#allentildesley2017) reference. No external
engine source code is used.

## Plan and completion rules

The optional third plan block argument is a distinct topology transfer map:

```text
md_dist.reference_plan [2, 2, 2] {
^bb0(%layout: !mdrt.layout<@atoms>,
     %spatial: !mdrt.transfer_map<@atoms>,
     %topology: !mdrt.transfer_map<@atoms>):
  %t = md_dist.topology_start %layout via %topology : ...
  %h = md_dist.halo_start %layout via %spatial : ...
  md_dist.dispatch @evaluate "interior"
  md_dist.halo_wait %h : !mdrt.event
  md_dist.dispatch @evaluate "boundary"
  md_dist.topology_wait %t : !mdrt.event
  md_dist.bond_dispatch @evaluate_bonds
  %r = md_dist.reverse_start %layout via %topology : ...
  md_dist.reverse_wait %r : !mdrt.event
}
```

This is an abbreviated listing; `--emit=dist` prints the complete parseable IR.
The verifier checks matching map/layout particle sets, exact bound SSA map
operands, one wait per event, topology readiness before bond dispatch, exactly
one bond dispatch with the stated ABI, and reverse completion on the same map.
It rejects absent/duplicate completion, wrong routes, partial topology reads,
duplicate bond dispatch, and an incompatible callee. Conservative effects keep
protocol actions through canonicalization. Shared types remain in `mdrt`;
there is no new `comm` dialect and no one-request-per-event requirement.

Host typestate is `Idle -> Forward -> Readable -> Reverse -> Idle`.
Rebuild is legal only in `Idle` with an increasing epoch. An async forward
posts all receives before sends (tag 81); it can remain outstanding during LJ
computation because packed topology buffers and the topology coordinate view
are separate from spatial buffers and force arrays. Wait completes all requests
and unpacks before the generated bond kernel reads the view.

After bond dispatch, the host first sums contributions into owned and replica
slots. Reverse transport sends replica slots back along the transpose route
(tag 82). Receive completion is followed by local addition through the saved
owner indices. The final wait applies both local and returned contributions
and the unique bond energy/virial to the current LJ stage result. The existing
interior result is combined afterward, before finite checks, reporting, or
integration. Request completion alone does not establish this force readiness.

`--halo=sync` uses blocking all-to-all payload exchange for both directions;
`async` uses nonblocking point-to-point requests with retained independent
buffers. Reverse currently starts and is immediately waited; no reverse/compute
overlap benefit is claimed. Runtime MPI progress during the forward overlap is
implementation dependent. Rebuild/migration cannot occur with outstanding
requests, and empty ranks participate in map construction and collectives.

## Validation and remaining limits

`scripts/validation/cpu-bonded.py` independently enumerates periodic images,
differentiates the scalar energy by finite differences, and integrates the
combined LJ-plus-bond Hamiltonian using Python velocity Verlet. It compares
energy, all virial and force components, final masses/coordinates/velocities,
and migration occurrence. Absolute-plus-relative tolerances are
`3e-10*(1+abs(reference))` for double and `5e-5*(1+abs(reference))` for mixed.
The integration fixtures avoid cutoff/image discontinuities.

`scripts/validation/cpu-bonded-invalid.py` checks unknown/self/duplicate IDs,
nonpositive/unrepresentable parameters, and malformed records with process
timeouts. IR tests independently exercise roundtrip, canonicalization, wrong
routes, incomplete transfer, missing/duplicate return, duplicate work dispatch,
and an incompatible kernel ABI.

The directory and topology table are replicated; metadata construction uses
all-to-all traffic and a global directory. There is no scalable distributed
lookup service, partitioned topology storage, retry on allocation failure,
restart persistence, or asynchronous error recovery. Count/capacity bounds
are checked against MPI integer limits and fatal errors use `MPI_Abort`.
This remains the experimental reduced-unit driver, not production molecular
NVE or a force-field parser. No distributed GPU or performance claim is made.

All routes in this executor use `MPI_COMM_WORLD`; arbitrary work teams and
transport backends are not implemented by these reference operations.

## Recorded validation

The complete matrix passed **144 configurations and 36 finite differences**.
Maximum absolute discrepancy over the compared energy, virial, force, mass,
position, and velocity components was `1.7764e-15` in double and `1.1332e-6`
in mixed. The nine malformed-input cases passed their rejection checks.

For the two-particle remote bond with positions `(1,3,3)` and `(8,3,3)` in a
box of edge 12, `k=0.7`, `r0=4`, and LJ cutoff 2.5, the nearest-image bond
length is 5 and LJ contributes zero. At eight ranks with async mixed precision:

| Quantity | Independent reference | Computed | Absolute difference |
|---|---:|---:|---:|
| Energy | 0.35 | 0.34999999403953552 | 5.96e-9 |
| First endpoint x force | -0.7 | -0.69999998807907104 | 1.19e-8 |
| xx virial | -3.5 | -3.5 | 0 |

The full CPU lit suite passed **200 tests, 69 unsupported**, including the
bonded roundtrip, seven invalid-plan tests, MPI numerical smoke test, and
malformed input checks. The tests use the same tolerance contract above.
No intermittent numerical failures were observed.

After rebasing onto `b0eac91` (the zero-epsilon fix), the full CPU suite passed
**201 tests, 70 unsupported**. The initial base's two focused production GPU
regressions (`pair-terms-gpu` and `lj-pme-gpu`) also passed under the GPU 1 lock.
These are existing production-path regression checks, not execution of the
new distributed bonded path on a GPU. The entire GPU suite was not rerun.
The rebased build also passed all three focused GPU 1 tests
(`pair-terms-gpu`, `lj-pme-gpu`, and the new `lj-zero-epsilon-gpu`) under the
lock in 111 seconds. The new topology transport remains CPU-only.
