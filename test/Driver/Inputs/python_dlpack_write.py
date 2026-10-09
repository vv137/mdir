"""Tracked writable DLPack borrows of a simulation with a commit
(D229, docs/python-dlpack.md), written by consumers that
do not use MDIR's code.

Usage: python_dlpack_write.py ROOT TARGET CONSUMER

Consumers:
  numpy   (CPU) NumPy's from_dlpack
  driver  (GPU) a reader of the capsule in ctypes that writes with the CUDA
          driver API
  torch   (CPU and GPU) PyTorch, on a GPU on a stream of its own

Each checks, in the deterministic mode, in double and mixed precision: a
commit of positions, of velocities, of the cell, and of tunables against the
same change through the existing interface (a simulation compiled from the
changed state; Simulation.tunables.update) and the steps after it, to the
bit; what a live borrow blocks; an abandoned borrow against a simulation
without one; the refusals, which change nothing; a commit whose evaluation
fails; and tensors that outlive the borrow and the simulation.
"""
import ctypes
import gc
import sys

import numpy as np
import mdir

root, target_name, consumer_name = sys.argv[1], sys.argv[2], sys.argv[3]
target = getattr(mdir.Target, target_name)
FIELDS = ("positions", "velocities", "forces")


def expect(error, call, text=""):
    try:
        call()
    except error as exc:
        assert text in str(exc), str(exc)
        return
    raise AssertionError(f"expected {error} ({text})")


def soft_term():
    term = mdir.PairTerm()
    term.name, term.expression = "soft", "a*exp(-r/l)"
    term.constants = [("a", 2.0), ("l", 0.05)]
    return term


def make(precision, state=None, method="VelocityVerlet", tunables=True, values=None,
         minimize=False, deterministic=True, tails=False):
    loaded = mdir.load_amber(root + "/dipeptide.prmtop", root + "/dipeptide.inpcrd")
    system, start = loaded.make_system(), loaded.make_state()
    system.cutoff, system.pairlist_distance, system.switch_distance = 0.8, 0.9, 0.7
    if tails:
        # A shift instead of the switch: the correction for the dispersion
        # integrates the tail of the pair term.
        system.switch_distance, system.truncation = 0.8, mdir.Truncation.Shift
    system.electrostatics = mdir.Electrostatics.PME
    # The grid of the first cell, for the simulations compiled from a
    # committed cell as well.
    system.pme_grid = [32, 32, 32]
    if tunables:
        values = values or {}
        system.pair_terms = [soft_term()]
        system.tunables = [mdir.Tunable("q", "charge", values=values.get("q")),
                           mdir.Tunable("l", "l", term="soft", values=values.get("l"))]
    if state is None:
        state = start.draw_velocities(system, 300.0, 7)
    integrator, ensemble, execution = mdir.Integrator(), mdir.Ensemble(), mdir.Execution()
    integrator.method = getattr(mdir.IntegratorMethod, method)
    integrator.timestep = 0.001
    integrator.minimize = minimize
    ensemble.kind = mdir.EnsembleKind.NVE
    execution.target, execution.precision = target, getattr(mdir.Precision, precision)
    execution.deterministic = deterministic
    return mdir.compile(system, state, integrator, ensemble, execution, mdir.Schedule())


def initial(snapshot, positions=None, velocities=None, edges=None):
    """The existing interface: an initial state with the change."""
    state = mdir.InitialState()
    state.positions = snapshot.positions if positions is None else positions
    state.velocities = snapshot.velocities if velocities is None else velocities
    cell = snapshot.cell
    if edges is not None:
        cell.diagonal = edges
    state.cell = cell
    return state


def same(a, b, what):
    """Equal to the bit: positions, velocities, forces, cell, energies."""
    for field in FIELDS:
        x, y = getattr(a, field), getattr(b, field)
        assert np.array_equal(x, y), (what, field, float(np.abs(x - y).max()))
    assert np.array_equal(a.cell.diagonal, b.cell.diagonal), (what, "cell")
    assert a.energies == b.energies, (what, a.energies, b.energies)


# --- Consumers -----------------------------------------------------------------
# open(buffer) gives a handle with read() -> a NumPy copy, write(array), and
# close(), which deletes the consumer's tensor.

class NumpyConsumer:
    name = "NumPy"

    class Handle:
        def __init__(self, buffer):
            self.array = np.from_dlpack(buffer)
            assert self.array.ctypes.data == buffer.data_ptr, "a copy"

        def read(self):
            return self.array.copy()

        def write(self, values):
            self.array[...] = values

        def close(self):
            del self.array

    def open(self, buffer):
        return self.Handle(buffer)


class TorchConsumer:
    name = "PyTorch"

    def __init__(self):
        import torch
        self.torch = torch

    class Handle:
        def __init__(self, torch, buffer):
            self.torch = torch
            self.stream = torch.cuda.Stream() if buffer.device[0] == 2 else None
            with self.on():
                self.tensor = torch.from_dlpack(buffer)
            assert self.tensor.data_ptr() == buffer.data_ptr, "a copy"
            if self.stream is not None:
                assert self.tensor.device == torch.device("cuda", buffer.device[1])

        def on(self):
            return self.torch.cuda.stream(self.stream) if self.stream is not None else self.torch.no_grad()

        def read(self):
            with self.on():
                return self.tensor.cpu().numpy().copy()

        def write(self, values):
            with self.on():
                self.tensor.copy_(self.torch.from_numpy(np.ascontiguousarray(values)))

        def close(self):
            del self.tensor

    def open(self, buffer):
        return self.Handle(self.torch, buffer)


class DLDevice(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int32), ("id", ctypes.c_int32)]


class DLDataType(ctypes.Structure):
    _fields_ = [("code", ctypes.c_uint8), ("bits", ctypes.c_uint8), ("lanes", ctypes.c_uint16)]


class DLTensor(ctypes.Structure):
    _fields_ = [("data", ctypes.c_void_p), ("device", DLDevice), ("ndim", ctypes.c_int32),
                ("dtype", DLDataType), ("shape", ctypes.POINTER(ctypes.c_int64)),
                ("strides", ctypes.POINTER(ctypes.c_int64)), ("byte_offset", ctypes.c_uint64)]


class DLManagedTensorVersioned(ctypes.Structure):
    _fields_ = [("major", ctypes.c_uint32), ("minor", ctypes.c_uint32),
                ("context", ctypes.c_void_p), ("deleter", ctypes.c_void_p),
                ("flags", ctypes.c_uint64), ("tensor", DLTensor)]


class DriverConsumer:
    """The capsule read in ctypes; buffers of the device copied by the CUDA
    driver API on a stream of its own that the producer hands off to."""
    name = "ctypes and the CUDA driver"

    def __init__(self):
        self.api = ctypes.pythonapi
        self.api.PyCapsule_GetPointer.restype = ctypes.c_void_p
        self.api.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
        self.api.PyCapsule_SetName.argtypes = [ctypes.py_object, ctypes.c_char_p]
        self.used = ctypes.c_char_p(b"used_dltensor_versioned")
        self.cuda = ctypes.CDLL("libcuda.so.1")
        self.check(self.cuda.cuInit(0))
        self.stream = None

    def check(self, result):
        assert result == 0, f"CUDA error {result}"

    def enter(self, ordinal):
        if self.stream is not None:
            return
        device, context = ctypes.c_int(), ctypes.c_void_p()
        self.check(self.cuda.cuDeviceGet(ctypes.byref(device), ordinal))
        self.check(self.cuda.cuDevicePrimaryCtxRetain(ctypes.byref(context), device))
        self.check(self.cuda.cuCtxSetCurrent(context))
        self.stream = ctypes.c_void_p()
        self.check(self.cuda.cuStreamCreate(ctypes.byref(self.stream), 1))  # non-blocking

    class Handle:
        def __init__(self, owner, buffer):
            self.owner = owner
            on_device = buffer.device[0] == 2
            if on_device:
                owner.enter(buffer.device[1])
            self.capsule = buffer.__dlpack__(stream=owner.stream.value if on_device else None,
                                             max_version=(1, 0))
            self.pointer = owner.api.PyCapsule_GetPointer(self.capsule, b"dltensor_versioned")
            owner.api.PyCapsule_SetName(self.capsule, owner.used)
            self.managed = DLManagedTensorVersioned.from_address(self.pointer)
            t = self.managed.tensor
            assert t.data == buffer.data_ptr and (t.device.type, t.device.id) == buffer.device
            # Writable buffers carry no read-only flag.
            self.read_only = bool(self.managed.flags & 1)
            self.on_device = on_device
            self.dtype = {(2, 64): np.float64, (2, 32): np.float32, (0, 32): np.int32}[
                (t.dtype.code, t.dtype.bits)]
            self.shape = tuple(t.shape[k] for k in range(t.ndim))

        def read(self):
            out = np.empty(self.shape, self.dtype)
            if not self.on_device:
                ctypes.memmove(out.ctypes.data, self.managed.tensor.data, out.nbytes)
                return out
            cuda, stream = self.owner.cuda, self.owner.stream
            self.owner.check(cuda.cuMemcpyDtoHAsync_v2(
                out.ctypes.data_as(ctypes.c_void_p), ctypes.c_uint64(self.managed.tensor.data),
                ctypes.c_size_t(out.nbytes), stream))
            self.owner.check(cuda.cuStreamSynchronize(stream))
            return out

        def write(self, values):
            values = np.ascontiguousarray(values, self.dtype)
            assert values.shape == self.shape
            if not self.on_device:
                ctypes.memmove(self.managed.tensor.data, values.ctypes.data, values.nbytes)
                return
            cuda, stream = self.owner.cuda, self.owner.stream
            self.owner.check(cuda.cuMemcpyHtoDAsync_v2(
                ctypes.c_uint64(self.managed.tensor.data), values.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_size_t(values.nbytes), stream))
            # The pageable array must outlive the copy; the commit waits
            # for the stream as well.
            self.owner.check(cuda.cuStreamSynchronize(stream))

        def close(self):
            # A foreign call releases the GIL: the deleter runs without it.
            ctypes.CFUNCTYPE(None, ctypes.c_void_p)(self.managed.deleter)(self.pointer)
            del self.capsule

    def open(self, buffer):
        return self.Handle(self, buffer)


consumer = {"numpy": NumpyConsumer, "torch": TorchConsumer, "driver": DriverConsumer}[consumer_name]()


def edit(buffer, change):
    """Reads a buffer of a borrow, writes change(values), and deletes the
    consumer's tensor."""
    handle = consumer.open(buffer)
    values = change(handle.read())
    handle.write(values)
    handle.close()
    gc.collect()
    return values


def rows(borrow):
    handle = consumer.open(borrow.ids)
    ids = handle.read()
    handle.close()
    return ids


def moved(borrow, particle, shift):
    """Moves the particle `particle` of the input through the borrow."""
    row = int(np.nonzero(rows(borrow) == particle)[0][0])

    def change(x):
        x[row] += np.asarray(shift, x.dtype)
        return x
    return edit(borrow.positions, change)


# --- Checks --------------------------------------------------------------------

def commits_against_the_interface(precision):
    program = make(precision)
    shift = np.array([0.011, -0.007, 0.013])

    # Positions: one particle moved. The reference is a simulation compiled
    # from the state with the particle moved, which evaluates it first.
    sim = mdir.Simulation(program)
    sim.run(8)
    sim.run(0, energy=True)
    at = sim.state()
    with sim.borrow() as borrow:
        assert borrow.live and borrow.step == 8 and borrow.written == ()
        assert borrow.positions.dtype == np.float64 and borrow.velocities.dtype == np.float64
        assert borrow.device == ((2, borrow.device[1]) if target_name == "GPU" else (1, 0))
        moved(borrow, 5, shift)
        assert borrow.written == ("positions",), borrow.written
        assert borrow.commit() == ("positions",)
        assert not borrow.live
    assert sim.leases == 0
    assert sim.versions == {"positions": 1, "velocities": 0, "cell": 0, "tunables": 0}
    assert sim.commits == [(8, ("positions",))]
    committed = sim.state()
    positions = at.positions.copy()
    positions[5] += shift
    assert np.array_equal(committed.positions, positions), "the positions written"
    assert np.array_equal(committed.velocities, at.velocities)
    assert not np.array_equal(committed.forces, at.forces), "forces carried through a commit"
    fresh = mdir.Simulation(make(precision, initial(at, positions=positions)))
    fresh.run(0, energy=True)
    same(committed, fresh.state(), "positions, at the commit")
    sim.run(12, energy=True)
    fresh.run(12, energy=True)
    same(sim.state(), fresh.state(), "positions, 12 steps later")
    change = float(np.abs(committed.forces - at.forces).max())

    # Velocities: all scaled.
    sim = mdir.Simulation(program)
    sim.run(8)
    at = sim.state()
    with sim.borrow() as borrow:
        ids = rows(borrow)
        edit(borrow.velocities, lambda v: v * 0.9)
        assert borrow.commit() == ("velocities",)
    assert sim.versions["velocities"] == 1 and sim.versions["positions"] == 0
    committed = sim.state()
    velocities = at.velocities * 0.9
    assert np.array_equal(committed.velocities, velocities)
    assert np.array_equal(committed.positions, at.positions)
    assert np.array_equal(np.sort(ids), np.arange(len(ids)))
    fresh = mdir.Simulation(make(precision, initial(at, velocities=velocities)))
    fresh.run(0, energy=True)
    same(committed, fresh.state(), "velocities, at the commit")
    sim.run(12, energy=True)
    fresh.run(12, energy=True)
    same(sim.state(), fresh.state(), "velocities, 12 steps later")

    # The cell with the positions: both scaled by 1.01.
    sim = mdir.Simulation(program)
    sim.run(8)
    at = sim.state()
    with sim.borrow() as borrow:
        assert borrow.cell.device == (1, 0) and borrow.cell.shape == (3,)
        edges = edit(borrow.cell, lambda c: c * 1.01)
        edit(borrow.positions, lambda x: x * 1.01)
        assert borrow.written == ("positions", "cell"), borrow.written
        assert borrow.commit() == ("positions", "cell")
    committed = sim.state()
    assert np.array_equal(committed.cell.diagonal, edges)
    assert np.array_equal(edges, np.asarray(at.cell.diagonal) * 1.01)
    fresh = mdir.Simulation(make(precision, initial(at, positions=at.positions * 1.01, edges=edges)))
    fresh.run(0, energy=True)
    same(committed, fresh.state(), "cell, at the commit")
    sim.run(12, energy=True)
    fresh.run(12, energy=True)
    same(sim.state(), fresh.state(), "cell, 12 steps later")

    # Tunables: the reference is Simulation.tunables.update.
    sim, other = mdir.Simulation(program), mdir.Simulation(program)
    for s in (sim, other):
        s.run(8)
    with sim.borrow() as borrow:
        assert borrow.tunables["q"].device == (1, 0)
        q = edit(borrow.tunables["q"], lambda q: q * 1.02)
        scale = edit(borrow.tunables["l"], lambda l: l * 1.1)
        assert borrow.written == ("tunables",)
        assert borrow.commit() == ("tunables",)
    other.tunables.update({"q": other.tunables["q"] * 1.02, "l": other.tunables["l"] * 1.1})
    assert np.array_equal(sim.tunables["q"], q) and np.array_equal(sim.tunables["l"], scale)
    assert all(np.array_equal(sim.tunables[k], other.tunables[k]) for k in ("q", "l"))
    assert sim.tunables.version == other.tunables.version == 1 == sim.versions["tunables"]
    assert sim.tunables.history == other.tunables.history == [(0, 0), (8, 1)]
    assert sim.state().tunables_version == 1
    same(sim.state(), other.state(), "tunables, at the commit")
    sim.run(12, energy=True)
    other.run(12, energy=True)
    same(sim.state(), other.state(), "tunables, 12 steps later")

    # Positions and tunables in one commit, against the compile with both.
    sim = mdir.Simulation(program)
    sim.run(8)
    at = sim.state()
    with sim.borrow() as borrow:
        moved(borrow, 5, shift)
        q = edit(borrow.tunables["q"], lambda q: q * 0.97)
        assert borrow.commit() == ("positions", "tunables")
    positions = at.positions.copy()
    positions[5] += shift
    fresh = mdir.Simulation(make(precision, initial(at, positions=positions), values={"q": q}))
    fresh.run(0, energy=True)
    same(sim.state(), fresh.state(), "positions and tunables, at the commit")
    sim.run(12, energy=True)
    fresh.run(12, energy=True)
    same(sim.state(), fresh.state(), "positions and tunables, 12 steps later")
    print(f"{precision}: commits of positions, velocities, the cell, and tunables equal the "
          f"existing interface to the bit, and 12 steps later")
    print(f"{precision}: the move of one particle changes the forces by up to {change:.3e} "
          f"kJ/mol/nm", file=sys.stderr)


def blocked(precision):
    program = make(precision)
    sim = mdir.Simulation(program)
    expect(mdir.SimulationError, sim.borrow, "before its first run")
    sim.run(4)
    with sim.view():
        expect(mdir.SimulationError, sim.borrow, "lease of a view")
    borrow = sim.borrow()
    handle = consumer.open(borrow.positions)
    text = "writable borrow"
    expect(mdir.SimulationError, lambda: sim.run(1), text)
    expect(mdir.SimulationError, lambda: sim.run(0, energy=True), text)
    expect(mdir.SimulationError, lambda: sim.tunables.update({"q": sim.tunables["q"]}), text)
    expect(mdir.SimulationError, sim.view, text)
    expect(mdir.SimulationError, sim.borrow, text)
    expect(mdir.SimulationError, sim.state, text)
    # Without HDF5 a checkpoint is unsupported before anything else.
    expect((mdir.SimulationError, mdir.UnsupportedError), lambda: sim.save_checkpoint("never.h5"))
    assert sim.step == 4 and sim.leases == 2
    # A commit needs the consumer's tensor deleted.
    expect(mdir.SimulationError, borrow.commit, "1 tensor taken from the borrow is alive")
    assert borrow.live and sim.versions["positions"] == 0
    # The borrow ends, the tensor lives: still blocked, by the tensor.
    borrow.abandon()
    assert not borrow.live and sim.leases == 1
    expect(mdir.SimulationError, lambda: sim.run(1), text)
    expect(BufferError, lambda: borrow.positions.__dlpack__(), "the borrow has ended")
    expect(mdir.SimulationError, borrow.commit, "has ended")
    handle.close()
    gc.collect()
    assert sim.leases == 0
    sim.run(1)
    minimizer = mdir.Simulation(make(precision, minimize=True))
    minimizer.minimize(2)
    expect(mdir.UnsupportedError, minimizer.borrow, "minimizes")
    print(f"{precision}: a live borrow blocks run, evaluation, update, view, borrow, state, and "
          f"checkpoint; a commit waits for the consumer's tensors")


def abandoned(precision):
    program = make(precision)
    sim, plain = mdir.Simulation(program), mdir.Simulation(program)
    for s in (sim, plain):
        s.run(8)
    with sim.borrow() as borrow:
        edit(borrow.positions, lambda x: x + 0.3)
        edit(borrow.velocities, lambda v: v * 0.0)
        edit(borrow.cell, lambda c: c * 2.0)
        edit(borrow.tunables["q"], lambda q: q * 0.0)
        assert borrow.written == ("positions", "velocities", "cell", "tunables")
    assert sim.leases == 0
    same(sim.state(), plain.state(), "after an abandoned borrow")
    assert np.array_equal(sim.tunables["q"], plain.tunables["q"])
    assert sim.versions == {"positions": 0, "velocities": 0, "cell": 0, "tunables": 0}
    assert sim.commits == []
    sim.run(12, energy=True)
    plain.run(12, energy=True)
    same(sim.state(), plain.state(), "12 steps after an abandoned borrow")
    # A borrow that is dropped, and one whose block raises.
    borrow = sim.borrow()
    edit(borrow.positions, lambda x: x * 0.0)
    del borrow
    gc.collect()
    try:
        with sim.borrow() as borrow:
            edit(borrow.velocities, lambda v: v + 1.0)
            raise KeyError("in the block")
    except KeyError:
        pass
    sim.run(5, energy=True)
    plain.run(5, energy=True)
    same(sim.state(), plain.state(), "after a dropped borrow and a block that raised")
    print(f"{precision}: an abandoned borrow leaves the run that of a simulation without it, "
          f"to the bit")


def refused(precision):
    program = make(precision)
    sim, plain = mdir.Simulation(program), mdir.Simulation(program)
    for s in (sim, plain):
        s.run(8)
    values = dict(sim.tunables)
    edges = np.asarray(plain.state().cell.diagonal)

    def unchanged():
        assert all(np.array_equal(sim.tunables[k], values[k]) for k in values)
        assert sim.tunables.version == 0 and sim.tunables.history == [(0, 0)]
        assert sim.versions == {"positions": 0, "velocities": 0, "cell": 0, "tunables": 0}
        assert sim.commits == []

    borrow = sim.borrow()
    # Positions that are not finite, with valid tunables in the same commit.
    original = []

    def spoil(x):
        original.append(x.copy())
        x = x.copy()
        x[3, 1] = np.nan
        return x
    edit(borrow.positions, spoil)
    edit(borrow.tunables["q"], lambda q: q * 1.01)
    expect(mdir.InputError, borrow.commit, "not finite")
    assert borrow.live
    unchanged()
    edit(borrow.positions, lambda x: original[0])
    edit(borrow.tunables["q"], lambda q: values["q"])
    # Values that an update refuses.
    for name, change, text in (("q", lambda q: q + 100.0, "less than 100 e"),
                               ("q", lambda q: q * np.nan, "")):
        edit(borrow.tunables[name], change)
        expect(mdir.InputError, borrow.commit, text)
        assert borrow.live
        unchanged()
        edit(borrow.tunables[name], lambda q: values[name])
    # A cell below twice the cutoff, and one that is not finite.
    for change in (lambda c: np.array([c[0], 1.5, c[2]]), lambda c: c * np.inf):
        edit(borrow.cell, change)
        expect(mdir.InputError, borrow.commit, "twice the cutoff")
        assert borrow.live
        unchanged()
        edit(borrow.cell, lambda c: edges)
    # Everything was written back: the commit takes the positions as they
    # were, and the run is that of a simulation that evaluates there.
    assert borrow.commit() == ("positions",)
    plain.run(0, energy=True)
    same(sim.state(), plain.state(), "after refusals and a commit of the same positions")
    assert all(np.array_equal(sim.tunables[k], values[k]) for k in values)
    assert sim.tunables.version == 0
    # A structural change of a tunable: with the correction for the
    # dispersion, the tail of the pair term cannot be integrated for a
    # negative length, so the values would change the program. Nothing of
    # the commit is taken, the positions written with it included.
    program = make(precision, tails=True)
    sim, plain = mdir.Simulation(program), mdir.Simulation(program)
    for s in (sim, plain):
        s.run(8)
    values = dict(sim.tunables)
    with sim.borrow() as borrow:
        moved(borrow, 5, [0.011, -0.007, 0.013])
        edit(borrow.tunables["l"], lambda l: -l)
        edit(borrow.tunables["q"], lambda q: q * 1.01)
        expect(mdir.InputError, borrow.commit, "its tail cannot be integrated")
        assert borrow.live
        unchanged()
    same(sim.state(), plain.state(), "after a refused structural change")
    sim.run(6, energy=True)
    plain.run(6, energy=True)
    same(sim.state(), plain.state(), "6 steps after a refused structural change")
    unchanged()
    print(f"{precision}: refused commits (positions not finite, a structural change of a "
          f"tunable, values an update refuses, a cell below twice the cutoff) change nothing")


def lifetimes(precision):
    program = make(precision)
    sim = mdir.Simulation(program)
    sim.run(6)
    at = sim.state()
    # A tensor that outlives the borrow and the Python simulation, while
    # another simulation runs: its buffer is not given to the other.
    borrow = sim.borrow()
    pointers = {borrow.positions.data_ptr, borrow.velocities.data_ptr}
    handle = consumer.open(borrow.positions)
    kept = consumer.open(borrow.tunables["q"])
    written = handle.read() + 0.25
    handle.write(written)
    borrow.abandon()
    del sim, borrow
    gc.collect()
    other = mdir.Simulation(program)
    other.run(20)
    with other.view() as view:
        assert not ({view.positions.data_ptr, view.velocities.data_ptr} & pointers), \
            "blocks under a live tensor were given to another simulation"
    assert np.array_equal(handle.read(), written)
    assert np.array_equal(kept.read(), other.tunables["q"])
    handle.close()
    kept.close()
    gc.collect()
    # Commits and abandoned borrows in turn: each commit ends an activation
    # and begins another, whose buffers the consumer takes anew.
    sim = mdir.Simulation(program)
    sim.run(6)
    same(sim.state(), at, "the same program")
    for cycle in range(6):
        with sim.borrow() as borrow:
            edit(borrow.positions, lambda x: x + (1e-4 if cycle % 2 == 0 else 0.5))
            if cycle % 2 == 0:
                borrow.commit()
        sim.run(3)
    assert sim.versions["positions"] == 3 and [c[0] for c in sim.commits] == [6, 12, 18]
    reference = mdir.Simulation(program)
    reference.run(6)
    for cycle in range(6):
        if cycle % 2 == 0:
            with reference.borrow() as borrow:
                edit(borrow.positions, lambda x: x + 1e-4)
                borrow.commit()
        reference.run(3)
    same(sim.state(), reference.state(), "commits and abandoned borrows in turn")
    print(f"{precision}: a tensor outlives its borrow and its simulation; commits and "
          f"abandoned borrows in turn")


def leapfrog_and_default_mode():
    # Leapfrog without tunables: the commit evaluates the forces without
    # the half kick back; the velocities are those of the half step.
    sim = mdir.Simulation(make("Double", method="Leapfrog", tunables=False))
    sim.run(8)
    at = sim.state()
    with sim.borrow() as borrow:
        assert borrow.velocity_offset == -0.5
        expect(KeyError, lambda: borrow.tunables["q"])
        moved(borrow, 5, [0.011, -0.007, 0.013])
        borrow.commit()
    committed = sim.state()
    positions = at.positions.copy()
    positions[5] += [0.011, -0.007, 0.013]
    assert np.array_equal(committed.positions, positions)
    assert np.array_equal(committed.velocities, at.velocities)
    assert not np.array_equal(committed.forces, at.forces) and committed.energies is None
    sim.run(4)
    # The default mode: mixed precision, not deterministic.
    sim = mdir.Simulation(make("Mixed", deterministic=False))
    sim.run(8)
    at = sim.state()
    with sim.borrow() as borrow:
        edit(borrow.velocities, lambda v: -v)
        borrow.commit()
    assert np.array_equal(sim.state().velocities, -at.velocities)
    assert np.array_equal(sim.state().positions, at.positions)
    sim.run(4)
    print("leapfrog without tunables, and the default mode: commits take what was written")


def failed_evaluation():
    # Two particles at one place: the evaluation of the committed state
    # fails, and the commit is undone.
    program = make("Double")
    sim, plain = mdir.Simulation(program), mdir.Simulation(program)
    for s in (sim, plain):
        s.run(8)
    borrow = sim.borrow()

    def collapse(x):
        x[:] = x[0]
        return x
    edit(borrow.positions, collapse)
    edit(borrow.tunables["q"], lambda q: q * 1.01)
    expect(mdir.SimulationError, borrow.commit, "the commit is undone")
    assert not borrow.live and not sim.failed and sim.leases == 0
    assert sim.versions == {"positions": 0, "velocities": 0, "cell": 0, "tunables": 0}
    assert np.array_equal(sim.tunables["q"], plain.tunables["q"])
    after, reference = sim.state(), plain.state()
    for field in FIELDS:
        assert np.array_equal(getattr(after, field), getattr(reference, field)), field
    # The undone commit ended the activation (#220): a view names it, and
    # the evaluation that the message gives brings the state back.
    for call in (sim.view, sim.borrow):
        expect(mdir.SimulationError, call,
               "after a commit of a borrow that failed and was undone; its state is on the "
               "host only, which state() copies; a run, or an evaluation with "
               "run(0, energy=True), brings it back")
    sim.run(0, energy=True)
    sim.view().release()
    sim.run(4)
    print("a commit whose evaluation fails is undone: the state and values of before the borrow")


if consumer_name == "torch" and target_name == "GPU":
    def outstanding_work():
        """Writes still under way on the consumer's stream when its tensor
        is deleted and the borrow committed: the commit waits for them."""
        torch = consumer.torch
        program = make("Mixed")
        sim, other = mdir.Simulation(program), mdir.Simulation(program)
        for s in (sim, other):
            s.run(8)
        stream = torch.cuda.Stream()
        # The first use of a kernel waits for the device: not the one below.
        torch.ones(3, dtype=torch.float64, device="cuda").mul_(0.5)
        torch.cuda.synchronize()
        with sim.borrow() as borrow:
            with torch.cuda.stream(stream):
                v = torch.from_dlpack(borrow.velocities)
                torch.cuda._sleep(1_000_000_000)  # about 0.5 s
                v.mul_(0.5)
            del v
            assert not stream.query(), "the consumer's work ended before the commit"
            borrow.commit()
        with other.borrow() as borrow:
            edit(borrow.velocities, lambda v: v * 0.5)
            borrow.commit()
        same(sim.state(), other.state(), "a commit under outstanding consumer work")
        print("consumer writes outstanding at the commit are waited for")


for precision in ("Double", "Mixed"):
    commits_against_the_interface(precision)
    blocked(precision)
    abandoned(precision)
    refused(precision)
    lifetimes(precision)
leapfrog_and_default_mode()
failed_evaluation()
if consumer_name == "torch" and target_name == "GPU":
    outstanding_work()
print(f"dlpack write {consumer_name} passed")
