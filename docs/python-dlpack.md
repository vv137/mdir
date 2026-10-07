# Read-only DLPack views of a simulation (D[python-dlpack])

Issue #131, the first part of item 6 of the M2 sequence
([python-m2.md](python-m2.md), Sections 3 to 5). Writable views, which
advance the versions of the fields they change, are #136. Status: in
progress; the questions put to the maintainer on the PR are marked below.

A simulation keeps its state where its program keeps it, on the device or
on the host, from part to part (D215,
[python-segments.md](python-segments.md#resident-buffers)). `state()`
copies it to the host in the order of the input. A view hands a consumer
(PyTorch, CuPy, NumPy, JAX) the buffers themselves through the
[DLPack Python specification](https://dmlc.github.io/dlpack/latest/python_spec.html):
no copy, in the order of the program, in the type that the program stores.

## Interface

```python
sim.run(1000)
with sim.view() as view:                # a lease on the state of step view.step
    x = torch.from_dlpack(view.positions)    # (N, 3), the program's buffer
    f = torch.from_dlpack(view.forces)
    ids = torch.from_dlpack(view.ids)        # (N,) int32: the input index of each row
    q = torch.from_dlpack(view.tunables["q"])  # (M,) float64, on the host
    x_input = torch.empty_like(x); x_input[ids.long()] = x   # the order of the input
    del x, f, ids, q                        # the consumer's aliases
sim.run(1000)                           # refused while a lease is alive
```

`Simulation.view()` returns a `View` of the state at the end of the last
part, `view.step`:

| Attribute | Shape and dtype | Unit, meaning |
|---|---|---|
| `positions` | `(N, 3)`, the state's type: float64 in double and mixed precision | nm |
| `velocities` | `(N, 3)`, the state's type | nm/ps, at `time + velocity_offset * timestep` |
| `forces` | `(N, 3)`, the forces' type: float32 in mixed precision, float64 in double precision and in the deterministic mixed mode (#102) | kJ/mol/nm, those that the next step begins with |
| `ids` | `(N,)` int32 | the index in the input of the particle at each row |
| `tunables` | a dict of names to `(M,)` float64, on the host | the values of the tunables (D213) |
| `step`, `time`, `velocity_offset` | int, float, float | as in `State` |
| `device` | `(1, 0)` on the CPU, `(2, k)` on CUDA device `k` | the DLPack device |
| `valid` | bool | whether the buffers still hold the state of `step` |

Each buffer is a `Buffer` with `shape`, `dtype` (a NumPy dtype),
`device`, `data_ptr` (the address of its first element), `__dlpack__`, and
`__dlpack_device__`. Rows are in the order of the program, which the start
of an activation chooses and the parts keep (D215); `ids` maps them to the
input. The views of the state are on the device of the simulation; the
values of the tunables are the host vectors from which the program's
values are built (the program's own copies of them, fields in the type of
the parameters and derived tables, are not viewed).

## Leases

A view is a *lease*: while it is alive, the simulation refuses everything
that would write or free the buffers. `View.release()`, or the end of a
`with` block, releases the view's own lease. Each tensor that a consumer
takes through `__dlpack__` holds a lease of its own until the consumer
calls its deleter, when its last alias is gone, whatever happened to the
`View` (an exported tensor cannot be revoked). `Simulation.leases` counts
the live ones.

| Operation while a lease is alive | Result |
|---|---|
| `run`, `minimize`, `run(0, energy=True)`, an update of tunables | `SimulationError` naming the leases; nothing changes |
| `state()`, `view()`, reporters, `request_stop`, `step`, `time` | allowed: they only read |
| `del sim` | the simulation, its buffers, and its program stay alive until the last lease is released |

**Logical validity, storage lifetime, consumer work.** Three things are
kept apart:

- *Validity*: `view.valid` is true while the buffers hold the state of
  `view.step`. A lease keeps it true. After the view is released, a part
  or the start of an activation makes it false.
- *Storage*: the buffers belong to the activation of the simulation's
  entry, and the state that a lease holds lives at least as long as the
  lease: an exported tensor holds the simulation, so even after `del sim`
  its blocks are not returned to the runtime's pool before the deleter is
  called.
- *Consumer work*: work that a consumer issued on the device may still
  read the buffers after the deleter has been called. On a device, the
  first operation after views were exported that writes or frees the
  buffers (a part, the end of an activation, the end of the simulation)
  first waits for all work issued in the CUDA context, on any stream,
  which covers the consumer's work.

## The protocol

`Buffer.__dlpack__(*, stream=None, max_version=None, dl_device=None,
copy=None)`:

- **Version.** With `max_version` of major version 1 or more, the capsule
  is a `dltensor_versioned` holding a `DLManagedTensorVersioned` of
  version 1.1 with the read-only flag set; otherwise a `dltensor` holding
  the legacy `DLManagedTensor`, which cannot say that it is read-only.
- **Ownership.** The managed tensor's context holds the lease and the
  simulation, not a Python object: its deleter may be called on any
  thread, with or without the GIL. A capsule that no consumer took
  releases its lease when it is destroyed; a consumer that took it
  (renamed it `used_dltensor...`) releases it through the deleter,
  exactly once.
- **Streams.** On a device the simulation's kernels run on a stream of the
  runtime that does not synchronize with the default streams. `stream`
  is the consumer's: `None` and 1 the legacy default stream, 2 the
  per-thread default stream, another integer a `cudaStream_t`; `-1` asks
  for no synchronization. The simulation records an event on its stream
  and makes the consumer's stream wait for it, so that the consumer's work
  begins after the part and the copy of the state that follows it. On the
  CPU, `stream` must be `None` or `-1`.
- **Device.** `__dlpack_device__()` is `(kDLCPU, 0)` or
  `(kDLCUDA, k)`, where `k` is the device that the runtime's context is
  on, as an ordinal among the devices that `CUDA_VISIBLE_DEVICES` leaves
  visible: the index that PyTorch calls `cuda:k` and CuPy `Device(k)`. It
  is `Execution.device` unless `MDRT_DEVICE` overrides it.
- `dl_device` other than the buffer's device and `copy=True` raise
  `BufferError`: a view does not copy.

**Read-only intent.** The legacy capsule cannot carry it, and consumers
may ignore the flag. Writing through a read-only view is not detected; its
effect on the simulation is undefined (lost at the next part, or carried
into it, or in the copy that a failed part returns to). Tracked writes are
#136.

The structures are those of `dlpack.h` (DLPack 1.1,
[dmlc/dlpack](https://github.com/dmlc/dlpack)), declared in
`python/DLPack.h` after its ABI.

## Errors

| Exception | When |
|---|---|
| `SimulationError` | `view()` before the first run or evaluation, or after a failure: the state is then on the host only, in the order of the input; another operation under way; a run, a minimization, an evaluation, or an update of tunables while leases are alive |
| `BufferError` | `__dlpack__` of a released view; a `stream` the device does not take; `dl_device` of another device; `copy=True` |
| `ValueError` | `stream=0` on a device, which the specification forbids |
