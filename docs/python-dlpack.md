# Read-only DLPack views of a simulation (D220)

Issue #131, the first part of item 6 of the M2 sequence
([python-m2.md](python-m2.md), Sections 3 to 5). Writable views, which
advance the versions of the fields they change, are #136. Status:
implemented; the questions put to the maintainer on PR #177 were decided
as recommended ([Maintainer rulings](#maintainer-rulings)).

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
`with` block, releases the view's own lease; without them it lasts while
the view or one of its `Buffer` objects is alive. Each tensor that a consumer
takes through `__dlpack__` holds a lease of its own until the consumer
calls its deleter, when its last alias is gone, whatever happened to the
`View` (an exported tensor cannot be revoked). `Simulation.leases` counts
the live ones.

| Operation while a lease is alive | Result |
|---|---|
| `run`, `minimize`, `run(0, energy=True)`, an update of tunables | `SimulationError` naming the leases; nothing changes |
| `state()`, `view()`, reporters, `request_stop`, `step`, `time` | allowed: they only read |
| `del sim` | the native simulation, its buffers, its compiled program, and the files of its reporters stay open until the last lease is released |

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
may ignore the flag: PyTorch 2.11 takes a read-only tensor without a
warning and writes into it (`x.zero_()`), NumPy 2 makes the array read-only.
Writing through a read-only view is not detected; its
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

## Implementation

- `compiler::Simulation::takeView` returns the addresses that the
  activation handed the host at its last boundary (`mdrtPartBoundary`,
  D215), their widths, the particle count, the device, and a
  generation, and takes a lease; `acquireLease`/`releaseLease` count the
  others. `run`, `minimize`, `evaluate`, and `updateTunables` check the
  count after the guard against overlapping operations (`checkLeases`).
  The generation advances in `resumeActivation` and `endActivation`, which
  also wait for the consumers' work (`waitForConsumers`) if buffers of the
  device were exported since the last wait.
- `runtime/mdrt_cuda.c` adds `mdrtDeviceOrdinal` (the device of the
  context), `mdrtDeviceHandOff` (an event recorded on the stream of the
  kernels and waited for on the consumer's), and `mdrtDeviceWaitAll`
  (`cuCtxSynchronize`).
- `python/DLPack.h` holds the DLPack structures, `View`, `Buffer`, and
  the capsules. A lease holds a `std::shared_ptr` to the native
  simulation, which `Simulation` (Python) shares.

## Validation

`test/Driver/Inputs/python_dlpack.py`, on the dipeptide in water (1,168
atoms) with PME and the default thermostat coupling every 10 steps, charges
tunable, in double, mixed, and deterministic mixed precision. The
consumers use none of MDIR's code.

| Test | Consumer | What is checked |
|---|---|---|
| `python-dlpack.test` | NumPy 2.5 `from_dlpack`, CPU | the array's address is the buffer's; arrays are read-only (the versioned flag); dtypes; rows put in the input's order by `ids` equal `state()` to the bit; tunable values; 6 leases block `run`, `run(0, energy=True)`, and an update and allow `state()`; release by view and by array, slices of arrays; capsules not taken, legacy and versioned (`max_version` None, (1, 0), (1, 3)), each releasing once; refusals; an array that outlives its simulation while another runs; `minimize` refused under a lease; `view()` refused after a failed part |
| `python-dlpack-gpu.test` | ctypes reader of the capsule and the CUDA driver API | version 1.1, the read-only flag, shape, strides, dtype, device; `cuPointerGetAttribute` gives the view's device ordinal; values copied by `cuMemcpyDtoHAsync` on the consumer's own non-blocking stream, handed off by `__dlpack__(stream=...)`, equal `state()` to the bit; the deleter called through ctypes without the GIL releases once and the renamed capsule nothing more; legacy capsules with the streams `None`, 1, 2, -1; `stream=0`; a tensor read after its simulation was deleted and another ran 20 steps |
| `python-dlpack-torch.test`, `python-dlpack-torch-gpu.test` (`REQUIRES: torch`) | PyTorch 2.11 (CUDA 12.8 build) | `data_ptr()` is the buffer's; dtypes; values against `state()`; tensors taken on a `torch.cuda.Stream` of their own; slices keep leases; outstanding consumer work (below); a tensor that outlives its simulation, whose blocks the next simulation does not take while it lives |
| `Sanitizer/python-dlpack-gpu.test` | ctypes and the driver, under compute-sanitizer memcheck and initcheck | views of two parts handed to a consumer stream, a tensor read after its simulation ended |

**Consumer work.** A consumer kernel enqueued on a PyTorch stream spins
for about 1e9 cycles (about 0.5 s) and then copies the positions; its tensor is deleted
and the view released while it spins, and the simulation runs 10 steps at
once; the test checks that the consumer's stream is still busy when the
run begins. The copy holds the positions of the view to the bit: the part
waited for the consumer's stream. With the wait removed (a build for this
check only), the copy differs from them by up to 0.058 nm (6.8 nm in the
deterministic mode, where the buffer that the consumer read was the other
of the pair that the loop rotates), and the run returns after 1 ms while
the consumer's stream is still busy.

PyTorch is not in the interpreter that the suite uses, so the two `torch`
tests are unsupported there; lit takes another interpreter with
`-Dtorch_python=<python>`, which must import the module of the build (the
same Python version). They were run with a private environment of PyTorch
2.11 over the suite's interpreter.

## Performance

One RTX 3090 (GPU 0, 300 W cap, idle), mixed precision, NVE at 1 fs; ms
per `run` call over 2,000 calls of 1 step and 300 of 10, and ms per step of
one run of 20,000 steps; main (4c435ae) and this branch on the same build
options, two rounds each, alternated. A view per part takes `view()`,
`torch.from_dlpack(view.positions)`, and releases both.

| System | Main | Branch | Branch, a view per part |
|---|---|---|---|
| Dipeptide (1,168 atoms), parts of 1 | 0.138, 0.154 | 0.135, 0.165 | 0.386, 0.236 |
| Dipeptide, parts of 10 | 0.923, 1.005 | 0.959, 0.956 | 1.095, 0.919 |
| Dipeptide, one run (ms/step) | 0.0950, 0.0982 | 0.0944, 0.0883 | 0.1081, 0.0835 |
| JAC (23,558 atoms), parts of 1 | 0.262, 0.263 | 0.262, 0.263 | 0.322, 0.312 |
| JAC, parts of 10 | 2.310, 2.316 | 2.311, 2.325 | 2.342, 2.351 |
| JAC, one run (ms/step) | 0.2279, 0.2282 | 0.2279, 0.2282 | 0.2280, 0.2281 |

Without views a part costs what it did (an atomic exchange and an
increment more). A view per part costs about 0.05 ms on JAC: the view, the
event that the consumer's stream waits for, and the synchronization of the
context before the next part. The dipeptide's spread between rounds is that
of the shared host.

## Maintainer rulings

The six questions put on PR #177 were decided as recommended
([comment](https://github.com/vv137/mdir/pull/177#issuecomment-6030816084)):
`view()` without an activation raises `SimulationError`; an exported tensor
keeps the native simulation alive until its deleter runs; tunables are
viewed as their host float64 vectors; conflicts raise `SimulationError`
with the count of live leases; writes through a read-only view are
undefined and not detected (#136 tracks writes); the CUDA context is
synchronized before the first operation that writes or frees exported
buffers.
