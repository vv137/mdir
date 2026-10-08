# DLPack views and writable borrows of a simulation (D220, D229)

Issue #131, the first part of item 6 of the M2 sequence
([python-m2.md](python-m2.md), Sections 3 to 5): read-only views. Writable
borrows with a commit, which advance the versions of the fields they
change, are the second part (#136, D229,
[below](#writable-borrows-d229)). Status of the views:
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
| `run`, `minimize`, `run(0, energy=True)`, an update of tunables, `borrow()` | `SimulationError` naming the leases; nothing changes |
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
into it, or in the copy that a failed part returns to). The tracked way
to write is a borrow
([Writable borrows](#writable-borrows-d229)).

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

## Writable borrows (D229)

Issue #136, the second part of item 6 and the last open item of M2a.
Status: implemented; the four questions put to the maintainer on PR #204
were decided as recommended
([Maintainer rulings on writable borrows](#maintainer-rulings-on-writable-borrows)).

A view is for reading. A *borrow* hands a consumer the same buffers for
writing, and the simulation takes what was written at an explicit commit.

### Interface

```python
sim.run(1000)
with sim.borrow() as borrow:              # exclusive: no view, no other borrow, no run
    x = torch.from_dlpack(borrow.positions)   # (N, 3), the program's buffer, writable
    ids = torch.from_dlpack(borrow.ids)       # (N,) int32, read-only
    # The rows are in the order of the program: ids[row] is the index in
    # the input of the particle at that row.
    row = torch.nonzero(ids == 17)[0, 0]      # the row of particle 17 of the input
    x[row] += delta
    q = torch.from_dlpack(borrow.tunables["q"])   # (M,) float64 on the host, writable
    q *= 1.01
    del x, ids, q                         # a commit needs every tensor deleted
    borrow.commit()                       # ("positions", "tunables")
sim.run(1000)                             # from the committed state
```

`Simulation.borrow()` returns a `Borrow` of the state at the end of the
last part. It needs what `view()` needs (an activation: a run or an
evaluation first, no failure) and no live lease.

| Attribute | Shape and dtype | Written through | What a commit does with it |
|---|---|---|---|
| `positions` | `(N, 3)`, the state's type, on the device of the simulation, in the order of the program | the program's buffer | taken as the positions (nm) |
| `velocities` | `(N, 3)`, the state's type, as above | the program's buffer | taken as the velocities (nm/ps, at `time + velocity_offset * timestep`) |
| `cell` | `(3,)` float64 on the host: the edges of an orthorhombic cell (nm) | a buffer of the borrow, holding the edges | taken as the edges if their bits changed |
| `tunables` | a dict of names to `(M,)` float64 on the host | buffers of the borrow, holding the values | `Simulation.tunables.update` of the entries whose bits changed (D213) |
| `ids` | `(N,)` int32, read-only | | |
| `step`, `time`, `velocity_offset`, `device` | as in `View` | | |
| `written` | a tuple of names | | the fields that a commit would take now |
| `live` | bool | | false after a commit or `abandon()` |

`commit()` takes the written fields and ends the borrow; it returns the
names of the fields it changed. `abandon()` ends the borrow without
taking anything. Leaving the `with` block without a commit, or dropping
the `Borrow`, abandons.

**The order of the rows.** The buffers of the state are in the order of
the program, which the start of an activation chooses, not in that of the
input: `ids` gives the particle of the input at each row, as in a view. A
commit begins another activation, so the order of the next borrow is
another: read `ids` in every borrow.

**Types.** The buffers of the state have the type that the program
stores, as in a view: float64 in double and in mixed precision (the
positions and velocities are f64 in both; only the forces are f32 in the
mixed mode, and forces cannot be borrowed); the Python model takes no
other precision.
The consumer writes that type and a commit converts nothing. The values of
the tunables and the cell are float64 on the host; a commit builds the
program's values from them as an update of D213 does (fields of the
parameters are f32 in mixed precision). The capsules carry no read-only
flag, except that of `ids`.

**What counts as written.** A buffer of the state is taken as written once
`__dlpack__` has been called on it in this borrow: a tensor that was
handed out for writing cannot be told from one that was written. The cell
and each tunable are written if their bits differ from the values of the
simulation at the commit.

**The cell.** The edges of an orthorhombic periodic cell. For a triclinic
cell, or without a periodic cell, `Borrow.cell` raises `UnsupportedError`:
the tilts are values that the program takes at its start (D227), and
writing them is not supported yet
([#206](https://github.com/vv137/mdir/issues/206)).

### Exclusion

While a borrow is live, and after it until the last tensor taken from it
is deleted:

| Operation | Result |
|---|---|
| `run`, `minimize`, `run(0, energy=True)`, `tunables.update`, `view()`, `borrow()`, `state()`, `save_checkpoint` | `SimulationError` naming the borrow and its leases; nothing changes |
| `step`, `time`, `leases`, `versions`, `commits`, `tunables[name]`, `request_stop` | allowed |
| `del sim` | as under a lease: the native simulation lives until the last tensor is deleted |

`borrow()` is refused with `SimulationError` while a lease of a view is
alive, and for a simulation that minimizes with `UnsupportedError`.
`state()` is refused, unlike under a view, because the buffers hold values
that are not a state of the simulation until they are committed.

### Commit

A commit is refused with `SimulationError` while a tensor taken from the
borrow is alive (an exported tensor cannot be revoked, and a write after
the commit would not be tracked); the borrow stays live. On a device it
first waits for all work issued in the CUDA context, which covers the
consumer's writes on any stream.

It then checks everything before it changes anything:

- positions and velocities are finite (`InputError`);
- the edges of the cell are finite and at least twice the cutoff
  (`InputError`);
- the values of the tunables pass the checks of an update of D213, and the
  program built from them is the compiled one, text for text: a structural
  change is refused (`InputError`).

After a refusal nothing has changed: the values and versions of the
simulation are those of before, and the borrow is still live with what
the consumer wrote, to be corrected and committed again, or abandoned.

A commit that passes takes the fields at once:

- the versions of the changed fields advance (`Simulation.versions`,
  P16), and `Simulation.commits` gains `(step, fields)`; changed tunables
  advance `tunables.version` and its history exactly as `update` does;
- the activation of the entry ends and another begins from the committed
  state with `%first_call = 2`, as after an update of tunables (D213,
  D215): the particles are put in order, the neighbor structures and the
  derived fields are built anew, and the forces that the next step begins
  with are evaluated at the committed state. Nothing that was computed
  from the old values is carried. `state().forces` is that of the
  committed state, and with velocity Verlet `state().energies` as well
  (with leapfrog, `None`);
- the copy of the state that a failed part returns to (D196) is that of
  the committed state.

A run after a commit therefore equals, to the bit in the deterministic
mode, a simulation compiled from the committed state that evaluates it
first and takes the same steps. If the evaluation fails (particles at one
place, for instance) the commit is undone: `SimulationError`, the
state, values, and versions of before the borrow, the borrow ended, and
the simulation not failed; its next part begins an activation from that
state, as after an update of tunables that was undone.

**A commit is not a step.** It rebuilds the neighbor structures and
evaluates the forces: its cost is that of an evaluation
(`run(0, energy=True)`) and of the start of an activation, milliseconds
([Cost](#cost-of-a-borrow)), not that of a step. A borrow is meant for a
change between runs (a move of a Monte Carlo scheme, a state placed by
another program, new values of parameters), not for a change at every
step, which belongs in the program as a term or a restraint. An upload in
place that keeps the activation was not taken, for the reason D215 gives
for the tunables: it would lose the equality with a simulation compiled
from the committed state.

MDIR does not adjust what was written: positions are not projected onto
the constraints and velocities keep their components along them, the
positions are not scaled with a new cell, and the energy that the write
adds or removes is not booked in the conserved quantity. A simulation
compiled from the same state behaves the same.

### Abandon

A borrow that ends without a commit leaves no trace: before the next
operation, once the last tensor is deleted, the positions and velocities
that were handed out are copied back from the copy of the state that the
simulation holds since the end of the last part (D215), on the device,
after a wait for the consumer's work, and the activation continues with
its order and neighbor structures. A run after an abandoned borrow equals
a run without it to the bit.

### Versions, checkpoints, and the fingerprint

`Simulation.versions` is a dict of `positions`, `velocities`, `cell`, and
`tunables`: the number of commits (and, for the tunables, updates) that
changed each from the host. Steps do not advance them.
`Simulation.commits` lists `(step, fields)` for each commit that changed
anything.

The fingerprint (D172, D223) covers the physics, the coupling, and the
execution, not the state: a commit does not change it. A checkpoint
written after a commit holds the committed state with its forces, and
continues exactly as any other; it records the values, version, and
history of the tunables as before (D223) and no version of the state:
`versions` and `commits` begin at 0 and empty in the simulation that
continues it.

### Untracked writes

A consumer that writes through a read-only view, or through a tensor it
kept after an abandon, cannot be detected. The policy: the tracked path is
`borrow()`; a write through `view()` stays undefined, as D220 ruled
([Read-only intent](#the-protocol)). A tensor kept after an abandon holds
its lease, so the simulation refuses to run until it is deleted; what it
writes until then is discarded with the rest of the borrow.

### Implementation of a borrow

- `compiler::Simulation::takeBorrow` returns the addresses of `takeView`
  and sets `borrowed`; `markWritten` records the fields whose buffers were
  exported. `checkLeases`, which `run`, `minimize`, `evaluate`,
  `updateTunables`, and now `takeView`, `getState`, and `saveCheckpoint`
  under a borrow call, refuses while the borrow has a lease and otherwise
  undoes it (`undoBorrow`: a copy from the snapshot of D215 into the
  buffers, `mdrtDeviceCopyWithin` on a device).
- `commitBorrow` waits for the consumers, reads the buffers and the
  snapshot into the order of the input (`readState`), checks, and then
  does what `updateTunables` does: the state of the host is the committed
  one, the activation ends, the program's values are those built from the
  new tunables (`rebuildTunables`), and `evaluatePart` begins the next
  activation with `%first_call = 2`. The evaluation of a committed state
  is taken with leapfrog by every program of segments, whose half kick
  back is a select on `%first_call` since D223.
- `python/DLPack.h`: `Borrow` owns the buffers of the host (the cell, the
  tunables) in a `Staging` that every managed tensor taken from one of
  them shares, so that they outlive the `Borrow`; a writable buffer
  exports without the read-only flag. The tensors of a view hold what
  they held (D220).
- No kernel, runtime function, pass, or program text changes, and nothing
  on the path of a step.

### Validation of borrows

`test/Driver/Inputs/python_dlpack_write.py`, on the dipeptide in water
(1,168 atoms) with PME on a grid of 32³, a pair term `a*exp(-r/l)`, the
charges and `l` tunable, velocity Verlet at constant energy, in the
deterministic mode, in double and in mixed precision. The consumers use
none of MDIR's code: NumPy 2.5 on the CPU (`python-dlpack-write.test`), a
reader of the capsule in ctypes that writes with `cuMemcpyHtoDAsync` of
the CUDA driver API on a stream of its own on a GPU
(`python-dlpack-write-gpu.test`), and PyTorch 2.14.1 (CUDA 13.0 build) on
the CPU and on a GPU, on a `torch.cuda.Stream` of its own
(`python-dlpack-write-torch*.test`, `REQUIRES: torch`). Every row holds
for each consumer, on the CPU and on an RTX 3090, in double and mixed
precision; the difference is the largest absolute difference of the
positions, velocities, forces, cell, and energies, and the tolerance is
0.

| Quantity | Reference | Difference | Tolerance |
|---|---|---|---|
| Forces and energies after a commit at step 8 that moves one particle by (0.011, −0.007, 0.013) nm, and positions, velocities, forces, energies 12 steps later | A simulation compiled from the state of step 8 with the particle moved (`InitialState.positions`), evaluated first, then 12 steps | 0 (the move changes the forces by up to 5.4e3 kJ/mol/nm) | 0 |
| The same for a commit of all velocities scaled by 0.9 | A simulation compiled from the state with those velocities | 0 | 0 |
| The same for a commit of the cell and the positions scaled by 1.01 | A simulation compiled from the state with that cell and those positions, the grid of PME given | 0 | 0 |
| State after a commit of the charges scaled by 1.02 and `l` by 1.1, and 12 steps later; the values, version, and history | `Simulation.tunables.update` of the same values on a second simulation of the program | 0 | 0 |
| A commit of the move and of charges scaled by 0.97 at once, and 12 steps later | A simulation compiled with those values from the state with the particle moved | 0 | 0 |
| State after an abandoned borrow that wrote the positions (+0.3 nm), the velocities (0), the cell (doubled), and the charges (0), and 12 steps later; after a borrow that was dropped and one whose block raised, 5 steps later | A second simulation of the program without a borrow | 0 | 0 |
| State, values, and versions after refused commits: a position that is not a number; charges of 100 e and more; charges that are not numbers; an edge of 1.5 nm (twice the cutoff is 1.6 nm) and one that is infinite; and, with the correction for the dispersion, a negative `l` that changes the program (its tail cannot be integrated), 6 steps later | The values of before, and a second simulation without a borrow | 0, and no version advanced | 0 |
| State and values after a commit whose evaluation fails (all particles at one place) | A second simulation without a borrow | 0; `failed` is false | 0 |
| On a GPU with PyTorch: velocities halved by a kernel that waits behind about 0.5 s of work on the consumer's stream, the tensor deleted and the borrow committed while the stream is busy | A commit of the same values written at once | 0; with the wait removed (a build for this check only), 2.87 nm/ps | 0 |

Also checked: `SimulationError` naming the borrow from `run`,
`run(0, energy=True)`, `tunables.update`, `view()`, `borrow()`, `state()`,
and `save_checkpoint` while a borrow or a tensor taken from it is alive;
`borrow()` refused under a lease of a view, before the first run, and for
a minimization; a commit refused while one tensor is alive, and accepted
after it is deleted; a tensor that outlives its borrow and its Python
simulation while another simulation runs 20 steps, whose buffers are not
those of the tensor; six borrows committed and abandoned in turn against
a simulation with the three commits alone, to the bit; a commit with
leapfrog in a program without tunables, and in the default mode of mixed
precision. `python-memory*.test` runs 30 borrows committed and abandoned
in turn: the memory of the host and of the device stays flat (+0.0 MiB of growth from borrow 5 to borrow 30 on both, bound 8 MiB).
No device code changed, so the compute-sanitizer suite was not run.

### Cost of a borrow

Nothing on the path of a step changes: the program, its kernels, and
`runPart` are those of main, and a call of `run` reads one more atomic
flag. On the dipeptide (1,168 atoms, PME) on one RTX 3090 (GPU 0, 300 W
cap, idle), mixed precision, the mean of 200 calls with PyTorch as the
consumer:

| Operation | ms |
|---|---|
| `run(1)` | 0.133 |
| A view, a tensor of the positions, both released, `run(1)` | 0.163 |
| A borrow, the positions written, abandoned, `run(1)` | 0.183 |
| A borrow, the positions written, committed | 1.315 |
| `run(0, energy=True)` | 1.245 |
| `tunables.update` of the charges | 1.830 |

An abandoned borrow costs a part 0.05 ms, the copy back and the wait for
the context; a commit costs what an evaluation costs.

### Maintainer rulings on writable borrows

The four questions put on PR #204 were decided as recommended
([comment](https://github.com/vv137/mdir/pull/204#issuecomment-6049907991)):
writes through a read-only view stay undefined and not detected, and
`borrow()` is the tracked path; the cell of a borrow is the edges of an
orthorhombic cell, a triclinic cell is refused with a message that says it
is not supported yet, and the tilts are
[#206](https://github.com/vv137/mdir/issues/206); the versions of the
state are not recorded in checkpoints; leaving a `with` block without a
commit abandons the borrow.
