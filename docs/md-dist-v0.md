# md_dist v0: synchronous distributed field contracts

Status: proposed, 2026-10-03, D[md-dist-architecture]. This refines
[the distributed plan](md-dist-plan.md), using the maintainer-supplied RFC
and Python reference. No syntax below is implemented or accepted by
`mdir-opt`. This is a design specification, not a report of MDIR execution.

## 1. Bounded first implementation

v0a is a fixed-cell, fixed-layout, synchronous, straight-line evaluation.
Start with two logical domains in one process, directed owner-only LJ and
analytic EAM, rebuilding candidates and maps for every evaluation. Test
transpose routing synthetically before enabling unique-pair execution.
v0b adds temporal validity, skin reuse, migration/reindex, team agreement,
and CPU transport. GPU storage and asynchronous execution follow.

The first implementation priorities are `FieldStateAnalysis` and lexical
accumulation completion. Transport, general control-flow proofs, autotuning,
PME distribution, crossing constraints, and executable MLIP stages are
outside v0a. Existing default pipelines stay unchanged. Separately verify
that a one-domain distributed plan specializes to the existing codegen.

## 2. Shared types and interfaces

`md_dist` and `md_exec` remain peers. Shared execution types live in `mdrt`;
neutral interfaces/analysis must not introduce a dependency cycle between
execution dialects. `md_dist` owns movement, routing, completion, and protocol
operations; `md_exec` owns traversal and arithmetic.

| Proposed type | Contract |
|---|---|
| `!mdrt.team` | Logical participants, independent of transport binding |
| `!mdrt.layout<@atoms>` | Immutable ownership and local-index snapshot |
| `!mdrt.field_view<@atoms, vector<3xf64>, owned>` | Complete payload on owned entities |
| `!mdrt.field_view<@atoms, vector<3xf64>, local>` | Complete payload on the owned and replica slots of a particular map; not a coverage guarantee |
| `!mdrt.transfer_map<@atoms>` | Field-independent owner-to-replica copy relation |
| `!mdrt.contribution<@atoms, vector<3xf64>, add>` | Partial sum, unavailable as an ordinary field input |
| `!mdrt.coverage<@atoms>` | Checked state/layout/map/candidate witness |

Start with scalar and fixed-width vector payloads, without a permanent
one/three-component restriction. Keep `!md.field`'s whole-set meaning.
`FieldPayloadTypeInterface` exposes entity and payload only.
`FieldStateOpInterface` and `FieldStateAnalysis` derive logical origin,
layout, materialization map, and readable domain from producers/operands.
Versions and evaluation instances are not type parameters. This uses the
separation of type and operation interfaces described by
[MLIR interfaces](https://mlir.llvm.org/docs/Interfaces/).

Halo refresh and redistribution preserve origin; integration creates a new
origin. A consumer requesting an older origin remains legal even if a newer
value exists. Labels are diagnostic names, not identity. Sorting/migration
create a new layout; coordinate updates alone do not. Map reuse requires
separate current-state coverage and payload freshness checks.

A local view represents owned identity rows plus replica copy rows. For
replica rows $T$, reverse routing is $T^\mathsf{T}$; a VJP of the entire
local view also includes the unchanged owned rows. Do not lose owned
adjoints by treating the replica transpose as the whole view's VJP.

v0 halos copy canonical coordinates without shifting payloads. Periodic
image identity and displacement belong to explicit geometry evaluation.
Future shifted materialization $x_g=x_i+Hn$ requires its own coordinate and
cell pullback, with the plugin/kernel responsibility stated exactly once.

## 3. Lexical accumulation and semantic work

Use a region rather than a new scope-handle type:

```text
Proposed syntax only:
%f = md_dist.accumulate %layout
       realizes(@force_requirement) inputs(%x, %cell) {
  %xh = md_dist.forward_halo %x via %map
  %z = md_exec.zeros on(%owned) : contribution
  %c = md_exec.pair_for ... ins(%xh) outs(%z)
         on(%owned) coverage(%witness) produces(@force_requirement) {
    ... validated force computation ...
  }
  md_dist.complete_accumulation %c
}
```

`complete_accumulation` terminates the scope; its parent returns a complete
owned field. Internal contributions cannot escape to another scope. The
inputs bind the semantic requirement to this evaluation instance. Nested
kernel arithmetic is allowed; distributed stage control flow is straight-line.

Derive requirements from the semantic program before distribution, not from
surviving producers. Track a symbolic multiset of semantic work slices,
including evaluation instance, term/output, source-work domain, target role,
and any justified enumeration weights. Initially support only recognized
whole-owned-center and verified partition forms. Unsupported summaries fail.
A `produces` attribute alone does not prove kernel mathematics: trusted
lowering rules, derived summaries, or an explicit external contract provide
that connection.

| Operation | Provenance rule |
|---|---|
| `zero` | Empty ledger and reducer identity; merging multiple zeros is legal |
| Accumulating traversal | Previous destination work plus the validated semantic work of this traversal |
| `reverse_accumulate` | Preserve work multiplicity; change replica destinations to their owners via the same map snapshot |
| `merge` | Add work multiplicities and numerical values; overlapping target atoms are legal |
| `complete_accumulation` | Require exactly the expected work multiplicities and no outstanding replica destinations |

Duplicate semantic summands are errors; two different terms acting on the
same atom are not duplicates. Dropping a computed contribution and returning
to zero leaves missing work. Multiple ordinary readers of the completed
field are legal. Do not substitute a blanket one-use rule for this ledger.
Floating-point addition obeys the declared precision/reassociation policy;
no rank-count-independent bits or implicit transport narrowing are promised.

Split operand roles: ordinary `ins` accept complete fields/views; additive
destinations accept contributions; explicit routing/reduction operations may
consume contributions. Do not add every new type to `MDExec_FieldOrBuffer`
and thereby allow nonlinear reads of partial values. Completion can forward
the same storage when its lifetime and extent permit; no mandatory copy.

## 4. Coverage and traversal multiplicity

Coverage binds layout, map, candidate structure, exact relation and its
cutoff/exclusions, coordinates/cell, validity guard, and enumeration rule.
$R\subseteq C$ checks missing interactions but does not exclude duplicates
in a candidate array. Also verify the post-filter multiplicity or the
recognized directed/unique transformation and its weighting.

Trusted builders and guards establish geometric facts. Static verification
checks witness provenance, compatibility, dominance, and permitted use;
`valid=true` is not a witness. Distinguish duplicate destination-slot
assignment from several valid replicas of one source. Conservative redundant
candidate storage is legal only if filtering/deduplication restores the
required semantic multiplicity.

Owned centers and local gather slots are separate index domains. Traversal
and integration must not iterate ghosts merely because the allocation has
`n_local` rows. Directed symmetric-pair energy has half weighting; owned
force does not. This rule is not automatically valid for asymmetric terms.

For EAM, complete density before the embedding derivative, exchange that
complete derivative using a compatible map, then complete force. Directed
density can complete locally; unique-pair density requires returned remote
contributions. Completion asserts the contract and need not introduce a
barrier. LJ/EAM scientific equations and validation gates remain in the
[parent plan](md-dist-plan.md#7-eam-and-derivative-contracts).

## 5. Verification and implementation slices

Local op verifiers check types, entity/payload/reducer, and operand roles.
Region verification checks origins, layouts, semantic work, and completion
through reusable analysis after nested ops are verified; see
[MLIR verification ordering](https://mlir.llvm.org/docs/DefiningDialects/Operations/#verification-ordering).
Checked runtime builders enforce IDs/slots, unique owners, geometry,
capacity, and participation agreement. Unknown stage branches are rejected
in v0a; the existing outer step loop is outside the straight-line plan.

Participant-local communication carries conservative team/channel effects,
including effects visible through an accumulation region whose result is
unused. Test DCE/canonicalization of the enclosing op, not just individual
transfers. Value semantics does not imply protocol purity; see
[MLIR effects](https://mlir.llvm.org/docs/Rationale/SideEffectsAndSpeculation/).

Diagnostics identify requested and supplied origin/layout, missing or duplicate
semantic work, unreturned replica contributions, or mismatched coverage.
Codes such as MDD001 are examples, not a stable diagnostic API yet.

The checkout already defines `MDRT_EventType` in
[MDRTTypes.td](../include/mdir/Dialect/MDRT/MDRTTypes.td). Distributed event
producers, completion guarantees, and scheduling remain future work; v0a
needs no event operand on every field type.

| Slice | Scope | Relation to existing delivery gates |
|---|---|---|
| A | Neutral interfaces/FieldStateAnalysis, shared types, lexical scope and movement/reduction syntax; roundtrip and negative tests | DIST0 contract portion; first implementation PRs |
| B | Fixed-state in-process executable reference, directed LJ/EAM, synthetic reverse tests | DIST0 reference plus fixed-state portions of DIST1/DIST2; not completion of those broader gates |
| C | Skin/freshness, migration/reindex of all live fields, agreement and CPU MPI | Temporal DIST1 and CPU DIST3; unique-pair/topology DIST2 validation still required |
| D | Synchronous GPU first, then storage hazards and measured overlap | GPU DIST3 before DIST4 |

The slices allow fixed-state EAM before migration. DIST numbers are acceptance
areas, not a requirement to finish all migration code before testing EAM.
Public driver keys, final assembly spelling, and transport ABI are separate
implementation reviews. Unsupported distributed types must fail clearly
if passed to a backend without their lowering.

The [post-v0 async skeleton](md-dist-async-design.md) reuses `mdrt.event`
for physical transfer start/await/join. It preserves this synchronous field
contract and does not add async requirements to v0a.

## 6. Supplied Python reference: evidence and follow-up tests

The maintainer supplied Python source and a log reporting 24 passing tests.
This revision reviewed the pasted source; it did not rerun those tests.
The sandbox attachment paths were not available in this workspace. The
reference is useful design evidence, not an MDIR test artifact or completion
of any DIST gate.

It models origins by object identity, layouts/maps, ghost readability,
work-label multisets, scalar/vector transpose, directed LJ, toy EAM, and
finite-difference EAM forces. Its scope is fixed orthorhombic minimum-image
geometry and double-precision NumPy; it runs centrally without transport.

Important limits of the supplied code:

- Distributed numerical producers attach `req.expected` directly as their
  ledger. The completion unit tests validate declared labels, not whether
  the numerical loops executed every semantic summand.
- Both numerical paths and their references share `near_pairs`, displacement,
  and potential helpers. Comparison can miss a shared relation/geometry bug;
  finite differences add derivative evidence but not independent coverage.
- `certify` checks readability; there is no independent candidate enumeration
  input or post-filter multiplicity check. Duplicate replica rejection is
  not the same as duplicate physical interaction rejection.
- `eam_distributed` discards its coverage object after certification, and
  directly computes the embedding on a completed field. This demonstrates
  a legal path, not rejection of a partial nonlinear read by a generic op.
- The image test checks transpose accumulation into one owner. It does not
  validate nonzero-image geometry, cell derivatives, virial, or periodic
  shifted materialization. No migration, trajectory, mixed precision, MPI,
  GPU, or overlap evidence is supplied.

Before calling the executable reference an independent oracle, add separate
relation enumeration and mutation tests: delete/duplicate a candidate while
retaining expected requirements; omit one producer without changing the
requirement; substitute stale views; attempt a partial-density nonlinear
read; combine local and returned copies of the same summand. Test owned
plus replica VJPs, enclosing-region DCE, multiple zeros, multiple complete
readers, and empty domains explicitly. Numerical fixtures and tolerance
policy follow the parent plan. Do not manufacture reported MDIR results
from the supplied 24-test log.
