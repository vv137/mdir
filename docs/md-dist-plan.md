# Distributed field contracts and development plan

Status: proposed architecture, 2026-10-03, D[md-dist-architecture].
This document specifies future work; it adds no dialect implementation,
control keys, defaults, file formats, or overwrite behavior. It refines
ML2–ML5 without changing the M2/M3/M4 milestone order in [roadmap.md](roadmap.md).

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
shapes/types, and completeness categories.

| Contract | Meaning |
|---|---|
| `team` | Logical participants; distinct from MPI ranks and device IDs |
| `layout` | Immutable snapshot of authoritative entity owners and local indices |
| `field_view` | Materialization of a logical field version on an explicit layout and subset; never silently changes `!md.field` to mean rank-local storage |
| `transfer_map` | Each consumer slot maps to an owner entity and, for particles, periodic image; physical forwarding is a later choice |
| `coverage` | Evidence that candidates contain the requested semantic relation under a stated validity predicate |
| `contribution` | Partial values with target, reducer, scope, and producer identity; not a readable complete field |
| `accumulation_scope` | Expected producers and target coverage for one logical result, including explicit initialization and completion |

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
or a speedup. The remaining sections of this plan will specify operations,
verification limits, integration points, and numerical acceptance gates.
