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
Positions, tunable values, and results remain in device buffers. Positions
of each frame are views into the batch, and results are written into their
corresponding output rows. The host may read the K by 6 cell values once
per call to validate them (Ruling 20, Q2). Tunable values are boundary inputs:
`applyTunables` runs on the device, so a change of parameters does not
restart the activation (Q3).

The boundary carries static requests for energy, position and strain
gradients, tunable rows, and tangent or cotangent contractions. Corrections
of the form c0(theta) + c1(theta)/V and their derivatives are evaluated on
the device. The chain rule is uploaded as indices and weights; its f64
sums preserve the host contribution order where possible, and the accepted
comparison of tunable rows and VJPs is relative error at most 1e-14.
Position and strain gradients and energies must match the deterministic
host evaluator to the bit for the same precision.

A scoped runtime stream selection restores the previous stream after the
call, including errors. Side streams join the selected stream, and FFT
plans take it on each execution. The batch checks positions on the device,
records the first failing frame and reason, and waits once per call to
report data errors (Q5). A failed batch discards its activation.

The runtime scope uses event dependencies in both directions, including
nested scopes and the legacy default stream. This first implementation
component is not yet connected to a batch caller. Focused component checks
precede the remaining batch implementation and its final-head validation.

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
  clean head. Stop on any output difference, numerical mismatch, test
  failure, or intermittent failure; never retry one.
