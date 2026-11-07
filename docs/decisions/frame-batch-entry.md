# D[frame-batch-entry]

Recorded 2026-11-06.

Part of #138, PR 1 of its accepted section 6.

## Contract

The device batch entry consumes `Control::FrameMode::DeviceBatch` through
`SimpleFrameEntryIRConfig::deviceBatch`. It is selected by a new C++
`Simulation::evaluateFrames` caller, independently of the existing host
`evaluateFrame` path. This decision is being implemented; the entry is not
yet available. It extends the field plumbing merged in D378 (#755).

One call evaluates K frames in one activation, using the caller's stream.
Positions, tunable values, and results remain in device buffers. Each
frame's position view is validated into an owned one-frame device scratch
buffer before neighbor indexing. This also widens f32 input for a
nondeterministic Mixed program; it never copies batch positions to the
host. Results are written into their corresponding output rows. The host may read the K by 6 cell values once
per call to validate them (Ruling 20, Q2). Tunable values are boundary inputs:
`applyTunables` runs on the device, so a change of parameters does not
restart the activation (Q3).

The boundary carries static requests for energy, position and strain
gradients, tunable rows, and tangent or cotangent contractions. Corrections
of the form $c_0(\boldsymbol\theta) + c_1(\boldsymbol\theta)/V$ and their derivatives are evaluated on
the device. The chain rule is uploaded as indices and weights; its f64
sums preserve the host contribution order where possible, and the accepted
comparison of tunable rows and VJPs is relative error at most 1e-14.
Quantities taken from the same compiled program must match the deterministic
host evaluator to the bit for the same precision. The tunable-dependent
dispersion and shift correction has two implementations: C++ in the host
entry and a kernel in the device entry. For the energies this correction
contributes to, the relative difference must be at most 1e-14. The same
relative bound applies to that correction's strain and virial contribution,
as ruled after the first integrated check. That check measured a largest
strain-contribution relative difference of 1.1158433802099432e-16 (1 ULP):
the three diagonal entries of one frame differed, while all twelve position
gradient entries were bitwise identical. The run stopped without a retry;
the maintainer authorized continuing with this explicit correction bound.
The compiled virial tensor and every other contribution remain bitwise.
This bound allows the few ULP introduced by the independent implementations;
it does not relax comparisons of other quantities. Report the largest
measured relative difference and ULP count. Repeated device calls in
deterministic mode must still be bitwise identical. Device integer powers
use multiplication, not `pow`. The host path and existing lowering remain
unchanged.

The focused eight-energy Double preflight measured a largest relative
difference of 1.582914503133543e-16, with at most 1 ULP,
against the actual host FrameEvaluator on RTX 5090. Two device kernel
invocations in the same run produced bitwise identical results. This
is a correction arithmetic component check, not final batch validation.
The native LJ constructor's eight-energy check measured a largest relative
difference of 2.989680490180319e-16 and at most 2 ULP; its repeated device
invocations were bitwise identical. The map/contraction check covered 879
values, with exact mapped values, a maximum contraction relative difference
of 1.4814145717262463e-16, and bitwise identical repeated device invocations.

The integrated check resumed under the explicit strain/virial ruling:
16/16 energies and 48/48 dispersion strain contributions satisfy 1e-14.
The maxima are 2.989680490180319e-16 (2 ULP) for energy and
2.1075144471205114e-16 (1 ULP) for the strain contribution. All 96 position
gradient values and 96 strain off-diagonal values are bitwise identical to
the host. All 256 repeated deterministic device values are bitwise
identical. These are component-snapshot results, not final-head gates.

A scoped runtime stream selection restores the previous stream after the
call, including errors. Side streams join the selected stream, and FFT
plans take it on each execution. The batch checks positions on the device,
records the first failing frame and reason, and waits once per call to
report data errors (Q5). A failed batch discards its activation.

The runtime scope uses event dependencies in both directions, including
nested scopes and the legacy default stream. The C++ batch caller and its
private DLPack exercise entry now use this scope. Focused component checks
precede the remaining batch implementation and its final-head validation.

The first end-to-end Double check on RTX 5090, without a dispersion tail,
compared one energy, twelve position-gradient entries and nine strain
entries with the host frame entry: all 22 were bitwise identical. This
checks a component snapshot, not a final-head gate. Only the new batch
output and chain-rule kernels carry `mdir.frame_host_arithmetic`: their
postprocessing preserves the separate multiply/add roundings of the host
entry. Existing potential kernels and ordinary lowering do not carry this
mark and retain their established compilation behavior.

The subsequent two-frame torsion check includes the tunable Jacobian:
46 values per call, all bitwise identical to the host in each of three
calls. Two calls repeat the same inputs; the third changes theta. Repeated
device results are bitwise identical, and the activation-start argument
counters do not increase across the calls. Derived tables, general
correction derivatives, auxiliary outputs and final-head gates remain
in progress.

Later component checks cover device JVP/VJP contractions, including a call
without a Jacobian output buffer, and bitwise virial/volume auxiliary
outputs. Device validation records nonfinite positions and invalid theta;
host cell validation seeds the same ordered status, so the final report
names the earliest bad frame. Invalid inputs use safe model values during
evaluation and the failed activation is discarded. A focused check covers
error attribution, recovery, overlapping output refusal and an empty batch
whose VJP is zero. The native constructor checks and all these focused
checks passed.

The combining-rule and charge-map component checks cover arithmetic and
geometric sigma mixing and epsilon mixing in Double and Mixed. Each uses
three calls, including an exact repeat and changed theta. With cutoff
electrostatics and with PME, energies, position gradients and strain are
bitwise identical to the host; parameter rows meet the relative 1e-14
bound and repeated device values are bitwise identical. These checks each
passed their first focused run. General custom pair-tail updates,
transfer-counter validation and final-head gates remain in progress.

`mdrt.from_buffer` accepts fields in explicit device memory space 1. Device
storage assignment borrows them without a host copy or ownership transfer;
they remain read-only, and a later destination takes separate storage.
Host storage assignment refuses this boundary. The caller keeps the input
alive until device work completes. This supplies the input boundary for
the native batch entry while preserving ordinary host-buffer imports.

`mdrt.to_buffer` can expose a field's device storage through an explicit
memory-space-1 result without a host copy. A function-level view retains
its storage for the activation. A loop-local view allows direct reads and
descriptor views within its iteration, but cannot escape or pass to an
opaque consumer. The storage stays pinned through those reads and returns
to the loop's carried buffer pool at the yield, keeping allocations outside
the frame loop. Host storage assignment refuses device results.

The GPU lowering of a function marked `mdir.frame_batch` recognizes f64
device loads marked `mdir.frame_parameter`. It clones their pure scalar
expressions into each consuming kernel, including side-stream kernels,
preserving the operations and eliminating the host loads. A remaining host
consumer is an error. This component supplies device scalar parameters;
the batch caller and device tunable updates remain in progress.

Device tables can also be borrowed and viewed without a copy. Their
explicit memref element type gives the physical storage width, including
f32 storage for logical f64 tables. A view with a different width or a
request for an absent host buffer is rejected. Storage assignment selects
table-only functions containing these new import/export boundaries while
retaining its existing selection for ordinary table helpers. This supplies
table storage for device tunable updates without a host round trip.

The native row loop takes a canonical contiguous byte buffer and a host
cell buffer with K rows and six columns. Each iteration creates an
identity-layout position view at its byte offset and reads that row's
cell. The builder's device-batch mode emits this K-loop behind one
`mdrtFrameBatchBoundary`; its C interface hands over positions, cells and
device theta at one activation boundary. Storage assignment preserves the
batch marker for later GPU scalar lowering. Immutable resolved maps are
uploaded at activation start. Direct field, tuple and pair-table maps update
owned device storage in physical precision; sites mapped to `-1` keep their values.
An effectful update call returns the descriptor for a fresh field/table
import, so Pure imports and their derived values cannot move above the update.
Scalar tunables become marked device loads in their consuming kernels.

The ordered contraction constructor uses CSR indices and f64 weights, with
one thread per result and no atomics in its sums. A separate LJ constructor
evaluates reference-volume dispersion/shift constants using multiplication
for integer powers. Device output writers, the caller, derived table updates,
correction hookup and the model chain rule are still being connected; no
batch API is exposed.

A private `FrameEvaluator` method accepting DLPack buffers exercises the
boundary. This PR adds no JAX handler, batching rule, public Python API,
control-file key, file format, checkpoint format, or overwrite behavior.
The existing host path remains available as the independent oracle.

## Required validation

The batch implementation and final validation remain in progress. Before readiness:

- CPU and RTX 5090, Mixed and Double: compare the batch with the host path
  and finite differences; check native batching, changing parameters without
  restarting, caller stream ordering, and error frame attribution.
- Assert zero host transfers of positions and results per call. Compare
  tunable rows and VJPs at 1e-14 relative; compare Mixed against Double at
  1e-4 for position/cell gradients, 5e-3 kJ/mol for energy, and 1e-6 for
  tunable gradients.
- Compare every existing printed module and lowered program and the order
  matrix with an independently archived final main; run the five bitwise
  checkpoints, both D309 shards, and the sanitizer tier on RTX 5090.
- Rebase onto current main before readiness and run every gate at that
  clean head. Stop on existing-output differences, numerical comparisons
  outside the accepted bounds, intermittent failures, or design questions.
  For a deterministic failure of a new work-in-progress test, fix it,
  rerun once, and record both counts. Never retry a numerical or
  intermittent failure.
