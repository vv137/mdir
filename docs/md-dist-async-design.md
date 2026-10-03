# Distributed transfer completion: post-v0 design skeleton

Status: proposed, 2026-10-03, D[md-dist-architecture]. This is a follow-up
to [synchronous v0](md-dist-v0.md), not an implementation of asynchronous
execution. It belongs to DIST3 protocol preparation and DIST4 scheduling.
The maintainer selected an MDIR design skeleton, not a separate upstream
`comm` dialect or an LLVM fork. Default pipelines and control keys do not
change. Code below is not registered in MDIR.

## 1. Upstream evidence and placement

The installed LLVM/MLIR 23.1.2 headers contain `shard.update_halo`, static
and dynamic halo sizes, and the ShardToMPI pass declaration. A separate
inspection of upstream LLVM revision
[36207c0](https://github.com/llvm/llvm-project/blob/36207c0ed8417ceaf9c393d3704833dca9939633/mlir/lib/Conversion/ShardToMPI/ShardToMPI.cpp)
found blocking sends/receives, contiguous temporary buffers, and a
higher-to-lower dimension traversal that propagates corner data. Its neighbor
conversion returns an absent neighbor at grid boundaries and supports one
grid axis per split; it is not a periodic particle routing implementation.
This source inspection does not establish the exact source revision or
runtime behavior of the installed library.

Use `shard` where tensor/grid semantics fit, especially future mesh work.
Particle fields continue through `md_dist` owner–replica maps. Neither must
lower through the other. Both could eventually use a shared physical transfer
contract; protocol scheduling follows data-placement requirements.

```text
md_dist field/map/contribution plan       shard tensor/grid operations
                 |                                  |
       storage and physical transfer planning / supported adapters
                                |
             mdrt transfer completion + protocol effects
                                |
           selected runtime transport / upstream mpi operations
```

The existing `!mdrt.event` represents completion. Its mere existence does
not supply protocol execution, progress, or alias verification. Do not add
`!comm.event` solely to rename this concept. MPI requests are backend details,
and an `async` token is useful only with a specified runtime/progress bridge.

## 2. Completion contract

The first asynchronous physical operation returns one conservative event.
It guarantees both input release and consumer readiness when awaited:
all required sends have released their source storage, receives and unpack
have finished, and the required memory visibility has been established.
It does not complete an enclosing semantic force accumulation scope.

A start must return after bounded initiation, without synchronously waiting
for the entire transfer. It may pack initial-stage data synchronously. A
multi-stage transfer needs a progress mechanism that can receive, unpack,
and initiate dependent stages; an opaque event alone does not provide one.
An implementation without autonomous progress may advance only during poll
or wait and must report that limit rather than claim full overlap.

`join` combines completion obligations, not MPI request ownership. Waiting
on the same logical event through multiple consumers is legal; a backend
must wait/retire each physical request safely once and retain completed
state. The enclosing execution owns event resources until all required
completion and cleanup occur. A later optimization may expose separate
input-release and consumer-ready events without changing these guarantees.

The transfer descriptor binds layout/map snapshot, source/destination
extents, pack/unpack rules, participant group, peer/edge identity, message
matching, and protocol phases. It is immutable and live through completion.
A descriptor cannot be rebuilt or rebound while an event uses it.

## 3. TableGen declaration skeleton

This standalone ODS sketch includes the current MDIR declarations and
reuses `MDRT_EventType`. `transfer_plan` and the three operations are proposed
physical-runtime vocabulary, not new public API. Generated declarations and
definitions can be checked with `mlir-tblgen`; custom verifiers, registration,
lowering, runtime support, and behavioral tests remain unimplemented.

```tablegen
include "mdir/Dialect/MDRT/MDRTOps.td"

def MDRT_TransferPlanType : MDRT_Type<"TransferPlan", "transfer_plan"> {
  let summary = "Immutable physical transfer descriptor and routing snapshot";
}

def MDRT_TransferStartOp : MDRT_Op<"transfer_start", [
    MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Initiate a physical transfer; completion remains outstanding";
  let arguments = (ins MDRT_TransferPlanType:$plan,
                       AnyNon0RankedMemRef:$source,
                       AnyNon0RankedMemRef:$destination,
                       Variadic<MDRT_EventType>:$dependencies);
  let results = (outs MDRT_EventType:$event);
  let assemblyFormat = [{
    $plan `from` $source `to` $destination
    `after` `(` $dependencies `)` attr-dict
    `:` type($source) `,` type($destination)
  }];
  let hasVerifier = 1;
}

def MDRT_TransferAwaitOp : MDRT_Op<"transfer_await", [
    MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Wait for input release, consumer readiness, and visibility";
  let arguments = (ins MDRT_EventType:$event);
  let assemblyFormat = "$event attr-dict";
  let hasVerifier = 1;
}

def MDRT_TransferJoinOp : MDRT_Op<"transfer_join", [
    MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Form one event that completes after all input events";
  let arguments = (ins Variadic<MDRT_EventType>:$events);
  let results = (outs MDRT_EventType:$event);
  let assemblyFormat = "$events attr-dict";
  let hasVerifier = 1;
}
```

Generic read/write effects intentionally prevent unsafe elimination in this
skeleton; they are not a finished asynchronous effect model. Implementation
must track protocol resources and accesses over the interval from start to
completion, including aliases and effects through enclosing regions.
`hasVerifier` only declares a hook; it proves nothing by itself.

A transfer is allowed only when its dependencies have completed. Lowering
may initially satisfy dependencies with waits before initiation; scheduling
can move independent work before those waits. The first prototype uses
host-accessible packed buffers with matching payload element types and
explicitly disjoint send/receive staging allocations. Shape, bounds,
capacity, packing permutation, and routing are checked against the plan.
Unsupported memory spaces or unknown alias relations are rejected or given
an explicit conservative synchronous fallback.

An empty join denotes an already complete event. A join does not create
missing phase dependencies. Later phases that forward received data must
wait for the previous phase's receive/unpack before packing their payloads.

## 4. ConvertAsyncTransferToMPI algorithm skeleton

This is an algorithm for a future conversion/runtime bridge, not compilable
C++ and not an existing pass. The particle/tensor-specific lowering has
already selected a physical plan; this layer does not infer a halo radius.

```text
lowerStart(plan, source, destination, dependencies):
  verify descriptor bindings, supported memory spaces, capacity and aliases
  establish completion of dependencies
  allocate event state and distinct live send/receive staging buffers
  for each ready protocol phase:
    post all matching receives for this phase
    pack send payloads whose data dependencies are satisfied
    initiate sends using plan-assigned communicator/peer/tag bindings
    retain requests, buffers, unpack actions and remaining phase dependencies
  return event state without waiting for all transfer completion

progress(event):
  observe completed receives; unpack and establish required visibility
  enable later phases only after their required data is available
  observe send completions; release corresponding staging/input borrows
  mark event complete when all required phases, visibility and cleanup finish

lowerAwait(event):
  drive progress and MPI waits until the logical event is complete
  preserve completed state for other event consumers

lowerJoin(events):
  return an aggregate completion state referring to child events
```

A statically sized, single-phase transfer can expand directly to
`mpi.irecv`, `mpi.isend`, and waits at the appropriate consumption point.
Runtime-dependent request counts or multi-stage progress may initially use
an `mdrt` ABI instead. Do not turn every opaque event into a fixed tuple of
requests without proving that its cardinality and control flow are static.
The declaration of an upstream MPI op does not establish support for every
MPI request management or progress API in the pinned toolchain.

The inspected blocking implementation reuses one temporary for send and
receive, then deallocates it. A nonblocking conversion must change those
lifetimes and overlapping accesses, not merely replace op names. It must
also assign an unambiguous message namespace when exchanges overlap; the
blocking template's constant tag cannot simply be copied into independent
concurrent protocols without a matching/order proof.

For ragged data, count exchange precedes capacity checks and payload posting.
The same descriptor/evaluation binds both phases. Zero counts, absent peers,
self peers, and distinct periodic edges to the same peer need explicit
rules. Count-dependent buffer allocation is not accomplished by a token.

## 5. Scheduling and verification obligations

Send borrows forbid writes/frees through any alias until input release;
receive borrows forbid reads/writes/frees until consumer readiness. Read-only
access to a send source may be legal. A blanket Ready/Pending state on an
entire particle array would unnecessarily prohibit interior reads, while
missing overlapping subviews could still admit a race. Track access regions
when proved, and conservatively serialize otherwise. MLIR SSA is not linear.

Boundary work depends on the views it reads becoming ready. It need not wait
for interior force work unless a real accumulation or storage dependency
requires it. Integration waits for the required complete force and state;
there is no generic safe `integrate.partial` motion that can advance borrowed
coordinates or change ownership during an evaluation.

Acceptance tests for the future implementation include:

- Roundtrip and verifier tests plus DCE of unused transfer results and parent
  regions; protocol participation must survive.
- Overlapping subview writes/frees while sending; premature receive reads;
  delayed completion; repeated await and shared joins; bounded cleanup.
- Staged corner forwarding versus direct-map reference, including the
  transpose route and contributions returned exactly once.
- Concurrent exchanges, peer/tag collisions, zero counts, empty domains,
  periodic axis sizes one and two, and coherent capacity failure.
- Single-phase completion and multi-stage progress with no dependence on
  eager message buffering; explicit progress and MPI thread-level assumptions.
- Synchronous/asynchronous numerical agreement, no ghost integration,
  single-domain codegen regression, and full-step timing under existing locks.

No asynchronous throughput or deadlock-freedom claim follows from TableGen
checking. A later upstream stencil experiment may assess generality, but it
is not a prerequisite for MDIR v0 or evidence for particle correctness.

## 6. Checks performed for this design

The TableGen block was extracted unchanged and processed using the installed
LLVM/MLIR 23.1.2 `mlir-tblgen`, with the repository and installed include
paths. `-gen-op-decls`, `-gen-op-defs`, and `-gen-typedef-decls` for `mdrt`
all succeeded. Generated output stayed in task scratch storage. This checks
ODS syntax and assembly-format generation, not C++ compilation, verifier
behavior, runtime registration, MPI execution, or overlap performance.
