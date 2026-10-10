"""The tilts of a triclinic cell in a writable borrow (D238,
docs/python-dlpack.md), on 403 waters in a rhombic dodecahedron, written by
consumers that do not use MDIR's code: NumPy's from_dlpack on the CPU, and
on a GPU a reader of the capsule in ctypes that writes with the CUDA driver
API.

Usage: python_dlpack_tilts.py ROOT TARGET commits WORK DIPEPTIDE
       python_dlpack_tilts.py ROOT TARGET frames DCD
       python_dlpack_tilts.py ROOT TARGET npt WORK

`commits`, in the deterministic mode, in double and mixed precision, with
PME and with a cutoff: a commit of tilts with positions against a simulation
compiled from the same state and cell (forces, energies, virial, and the
state 12 steps later, to the bit); the same lattice in another reduced
basis; a round trip without a change; the refusals, which change nothing;
a cell narrower between two faces than twice the cutoff against a NumPy sum
over the images; the cell of the frames of an open reporter.

`frames`: the frames of a run of `mdir run` at constant pressure, whose
barostat scales the tilts with the cell, put one by one into a second
simulation, each energy against the evaluation of a simulation compiled
from that frame.

`npt` (D[python-triclinic-npt]): a Python run at constant pressure in the
triclinic cell, whose tilts the barostat scales: the tilts of the state,
a run in parts and one continued from a checkpoint against one run, a
commit of tilts followed by steps under the barostat against a simulation
compiled from the committed state, and 20 frames of the run put into a
second simulation.
"""
import ctypes
import gc
import pathlib
import sys

import numpy as np
import mdir

root, target_name, mode, last = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
dipeptide = sys.argv[5] if len(sys.argv) > 5 else None
target = getattr(mdir.Target, target_name)
FIELDS = ("positions", "velocities", "forces")
CUTOFF, REACH = 0.8, 0.9
SOFT_A, SOFT_L = 2.0, 0.2


def expect(error, call, text=""):
    try:
        call()
    except error as exc:
        assert text in str(exc), str(exc)
        return str(exc)
    raise AssertionError(f"expected {error} ({text})")


# --- Consumers -----------------------------------------------------------------

class NumpyConsumer:
    def write(self, buffer, values):
        array = np.from_dlpack(buffer)
        assert array.ctypes.data == buffer.data_ptr, "a copy"
        array[...] = values
        del array

    def read(self, buffer):
        return np.from_dlpack(buffer).copy()


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

    def __init__(self):
        self.api = ctypes.pythonapi
        self.api.PyCapsule_GetPointer.restype = ctypes.c_void_p
        self.api.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
        self.api.PyCapsule_SetName.argtypes = [ctypes.py_object, ctypes.c_char_p]
        self.used = ctypes.c_char_p(b"used_dltensor_versioned")
        self.cuda = ctypes.CDLL("libcuda.so.1")
        assert self.cuda.cuInit(0) == 0
        self.stream = None

    def enter(self, ordinal):
        if self.stream is not None:
            return
        device, context = ctypes.c_int(), ctypes.c_void_p()
        assert self.cuda.cuDeviceGet(ctypes.byref(device), ordinal) == 0
        assert self.cuda.cuDevicePrimaryCtxRetain(ctypes.byref(context), device) == 0
        assert self.cuda.cuCtxSetCurrent(context) == 0
        self.stream = ctypes.c_void_p()
        assert self.cuda.cuStreamCreate(ctypes.byref(self.stream), 1) == 0

    def copy(self, buffer, values):
        """Writes `values`, or reads if it is None; deletes the tensor."""
        on_device = buffer.device[0] == 2
        if on_device:
            self.enter(buffer.device[1])
        capsule = buffer.__dlpack__(stream=self.stream.value if on_device else None,
                                    max_version=(1, 0))
        pointer = self.api.PyCapsule_GetPointer(capsule, b"dltensor_versioned")
        self.api.PyCapsule_SetName(capsule, self.used)
        managed = DLManagedTensorVersioned.from_address(pointer)
        t = managed.tensor
        dtype = {(2, 64): np.float64, (2, 32): np.float32, (0, 32): np.int32}[
            (t.dtype.code, t.dtype.bits)]
        shape = tuple(t.shape[k] for k in range(t.ndim))
        out = np.empty(shape, dtype) if values is None else np.ascontiguousarray(values, dtype)
        assert out.shape == shape
        if not on_device:
            if values is None:
                ctypes.memmove(out.ctypes.data, t.data, out.nbytes)
            else:
                ctypes.memmove(t.data, out.ctypes.data, out.nbytes)
        else:
            call = self.cuda.cuMemcpyDtoHAsync_v2 if values is None else self.cuda.cuMemcpyHtoDAsync_v2
            first = out.ctypes.data_as(ctypes.c_void_p) if values is None else ctypes.c_uint64(t.data)
            second = ctypes.c_uint64(t.data) if values is None else out.ctypes.data_as(ctypes.c_void_p)
            assert call(first, second, ctypes.c_size_t(out.nbytes), self.stream) == 0
            assert self.cuda.cuStreamSynchronize(self.stream) == 0
        ctypes.CFUNCTYPE(None, ctypes.c_void_p)(managed.deleter)(pointer)
        del capsule
        return out

    def write(self, buffer, values):
        self.copy(buffer, values)

    def read(self, buffer):
        return self.copy(buffer, None)


consumer = NumpyConsumer() if target_name == "CPU" else DriverConsumer()


# --- The system ---------------------------------------------------------------

def make(precision, pme=True, kind="NVE", state=None, capacity=0, soft=False, grid=28):
    loaded = mdir.load_gromacs(root + "/water.top", root + "/dodecahedron.gro",
                               defines=["FLEXIBLE"])
    system, start = loaded.make_system(), loaded.make_state()
    system.cutoff, system.pairlist_distance, system.switch_distance = CUTOFF, REACH, CUTOFF
    system.truncation = mdir.Truncation.None_
    system.dispersion = mdir.DispersionCorrection.None_
    system.electrostatics = mdir.Electrostatics.PME if pme else mdir.Electrostatics.Cutoff
    if pme and grid:
        # The grid of the first cell, for the simulations compiled from a
        # committed cell as well.
        system.pme_grid = [grid] * 3
    if soft:
        term = mdir.PairTerm()
        term.name, term.expression = "soft", "a*exp(-r/l)"
        term.constants, term.observe = [("a", SOFT_A), ("l", SOFT_L)], []
        system.pair_terms = [term]
    if state is None:
        state = start.draw_velocities(system, 300.0, 7)
    integrator, ensemble, execution = mdir.Integrator(), mdir.Ensemble(), mdir.Execution()
    integrator.method = mdir.IntegratorMethod.VelocityVerlet
    integrator.timestep = 0.0005
    ensemble.kind = getattr(mdir.EnsembleKind, kind)
    if kind != "NVE":
        ensemble.temperature = 300.0
    if kind == "NPT":
        ensemble.pressure = 1.0
    execution.target, execution.precision = target, getattr(mdir.Precision, precision)
    execution.deterministic = True
    execution.neighbor_capacity = capacity
    return mdir.compile(system, state, integrator, ensemble, execution, mdir.Schedule())


def cell_of(diagonal, tilt):
    cell = mdir.Cell()
    cell.diagonal, cell.tilt = np.asarray(diagonal, float), np.asarray(tilt, float)
    return cell


def initial(positions, velocities, cell):
    state = mdir.InitialState()
    state.positions, state.velocities, state.cell = positions, velocities, cell
    return state


def sheared(x, old, new):
    """Each water moved rigidly with the lattice coordinates of its oxygen."""
    x = x.copy().reshape(-1, 3, 3)
    s = x[:, 0, :] @ np.linalg.inv(old)
    x += (s @ new - x[:, 0, :])[:, None, :]
    return x.reshape(-1, 3)


def strained(x, now, target):
    """The waters moved with the lattice from the cell `now` to `target`,
    through the basis of the lattice of `now` that is nearest to `target`:
    a change of basis moves nothing, and the rest is a small strain."""
    d, t = now.diagonal, now.tilt.copy()
    n = np.round((t[2] - target.tilt[2]) / d[1])
    t[2] -= n * d[1]
    t[1] -= n * t[0]
    t[1] -= np.round((t[1] - target.tilt[1]) / d[0]) * d[0]
    t[0] -= np.round((t[0] - target.tilt[0]) / d[0]) * d[0]
    basis = np.array([[d[0], 0.0, 0.0], [t[0], d[1], 0.0], [t[1], t[2], d[2]]])
    return sheared(x, basis, target.vectors)


def put(sim, positions=None, diagonal=None, tilt=None, commit=True):
    """A borrow that writes what is given and commits."""
    with sim.borrow() as borrow:
        if positions is not None:
            ids = consumer.read(borrow.ids)
            consumer.write(borrow.positions, positions[ids])
        if diagonal is not None:
            consumer.write(borrow.cell, diagonal)
        if tilt is not None:
            consumer.write(borrow.tilt, tilt)
        gc.collect()
        return borrow.commit() if commit else borrow.written


def same(a, b, what):
    """Equal to the bit: the state, the cell, and the energies with the virial."""
    for field in FIELDS:
        x, y = getattr(a, field), getattr(b, field)
        assert np.array_equal(x, y), (what, field, float(np.abs(x - y).max()))
    assert np.array_equal(a.cell.diagonal, b.cell.diagonal), (what, "diagonal")
    assert np.array_equal(a.cell.tilt, b.cell.tilt), (what, "tilt", a.cell.tilt, b.cell.tilt)
    assert a.energies == b.energies, (what, a.energies, b.energies)
    assert a.energies is None or "virial" in a.energies


def against_compiled(sim, program, precision, pme, kind, positions, cell, steps, what):
    """The simulation after a commit against one compiled from the committed
    state with the grid and the capacity of the first: the evaluation, then
    `steps` steps."""
    at = sim.state()
    plan = program.plan
    fresh_program = make(precision, pme, kind, initial(positions, at.velocities, cell),
                         capacity=plan["neighbor_capacity"])
    # Nothing that depends on the tilts is in the text.
    assert fresh_program.ir == program.ir, (what, "the text of the program differs")
    fresh = mdir.Simulation(fresh_program)
    fresh.run(0, energy=True)
    same(at, fresh.state(), what + ", the evaluation")
    sim.run(steps, energy=True)
    fresh.run(steps, energy=True)
    same(sim.state(), fresh.state(), what + f", {steps} steps later")


def reduced(cell, bound=1.0 + 1e-6):
    d, t = cell.diagonal, cell.tilt
    return abs(t[0]) <= 0.5 * d[0] * bound and abs(t[1]) <= 0.5 * d[0] * bound and \
        abs(t[2]) <= 0.5 * d[1] * bound


# --- Commits -------------------------------------------------------------------

def commits(precision, pme):
    label = f"{precision} {'PME' if pme else 'cutoff'}"
    program = make(precision, pme)
    sim, twin = mdir.Simulation(program), mdir.Simulation(program)
    sim.run(8, energy=True)
    twin.run(8, energy=True)
    at = sim.state()
    first = at.cell
    assert np.any(first.tilt != 0.0)

    # A round trip without a change: the cell written with its own values
    # is not written, and the activation continues.
    with sim.view() as view:
        assert view.step == 8
    assert put(sim, diagonal=first.diagonal, tilt=first.tilt) == ()
    assert sim.versions["cell"] == 0 and sim.commits == []
    sim.run(6, energy=True)
    twin.run(6, energy=True)
    same(sim.state(), twin.state(), "a round trip without a change")

    # Refusals: nothing changes.
    def refused(text, **written):
        return expect(mdir.InputError, lambda: put(sim, **written), text)
    d = first.diagonal
    messages = [
        refused("all zero", tilt=[0.0, 0.0, 0.0]),
        refused("not reduced: the tilt b_x", tilt=[0.51 * d[0], first.tilt[1], first.tilt[2]]),
        refused("not reduced: the tilt c_y", tilt=[0.0, first.tilt[1], -0.6 * d[1]]),
        # a_x smaller with the tilts kept: c_x is past half of it.
        refused("not reduced: the tilt c_x", diagonal=[0.95 * d[0], d[1], d[2]]),
        refused("the tilt c_x written for the cell is not finite",
                tilt=[0.0, np.nan, first.tilt[2]]),
        refused("less than twice the cutoff", diagonal=[d[0], d[1], 1.5]),
    ]
    assert sim.versions["cell"] == 0 and sim.commits == []
    sim.run(6, energy=True)
    twin.run(6, energy=True)
    same(sim.state(), twin.state(), "after refused commits")
    at = sim.state()

    # A small change of the tilts, with the waters moved with the lattice.
    cell = cell_of(first.diagonal, first.tilt + np.array([0.012, -0.009, -0.007]))
    x = strained(at.positions, first, cell)
    assert put(sim, positions=x, tilt=cell.tilt) == ("positions", "cell")
    assert sim.versions["cell"] == 1 and sim.commits[-1] == (20, ("positions", "cell"))
    against_compiled(sim, program, precision, pme, "NVE", x, cell, 12, label + ": small")

    # The same lattice in another reduced basis: c - a, c_x from a_x/2 to
    # -a_x/2, the positions as they are. The state is the same physically,
    # and the program takes other images, other bins, and other vectors of
    # the reciprocal lattice.
    at = sim.state()
    before = at.cell
    other = cell_of(before.diagonal, [before.tilt[0], before.tilt[1] - before.diagonal[0],
                                      before.tilt[2]])
    # The commit above left c_x below the bound, so c - a is past it: the
    # test takes the lattice back to the bound first.
    edge = cell_of(before.diagonal, [before.tilt[0], 0.5 * before.diagonal[0], before.tilt[2]])
    x = strained(at.positions, before, edge)
    assert put(sim, positions=x, tilt=edge.tilt) == ("positions", "cell")
    energy = sim.state().energies["potential"]
    other = cell_of(edge.diagonal, [edge.tilt[0], -0.5 * edge.diagonal[0], edge.tilt[2]])
    assert reduced(other)
    assert put(sim, tilt=other.tilt) == ("cell",)
    moved = sim.state().energies["potential"]
    lattice = abs(moved - energy)
    # In kJ/mol. With PME the splines follow the vectors of the cell, so
    # the two bases differ within the accuracy of the mesh (1e-4 of the
    # potential); with a cutoff, by the rounding of the sums, in f32 in
    # mixed precision.
    tolerance = 1e-4 * abs(energy) if pme else 1e-9 if precision == "Double" else 2e-2
    assert lattice < tolerance, (label, energy, moved)
    against_compiled(sim, program, precision, pme, "NVE", x, other, 12,
                     label + ": another basis")

    # A large change in all three entries: c_x by 2.5 nm and c_y by
    # 2.55 nm, which is another basis of a lattice strained by up to 8%.
    at = sim.state()
    before = at.cell
    large = cell_of(before.diagonal, [0.2, 1.2, -1.25])
    assert reduced(large)
    x = strained(at.positions, before, large)
    assert put(sim, positions=x, tilt=large.tilt) == ("positions", "cell")
    against_compiled(sim, program, precision, pme, "NVE", x, large, 12, label + ": large")

    # The diagonal and the tilts at once: the cell scaled as a barostat
    # scales it, H diag(mu), with the positions.
    at = sim.state()
    before = at.cell
    mu = np.array([1.02, 0.99, 1.01])
    scaled = cell_of(before.diagonal * mu,
                     before.tilt * np.array([mu[0], mu[0], mu[1]]))
    x = at.positions * mu
    assert put(sim, positions=x, diagonal=scaled.diagonal, tilt=scaled.tilt) == \
        ("positions", "cell")
    against_compiled(sim, program, precision, pme, "NVE", x, scaled, 12, label + ": scaled")
    assert sim.versions["cell"] == 5

    # A round trip that hands the positions out and writes them back as
    # they are, with the cell as it is: a commit of the positions, which
    # begins an activation; the run equals that of a simulation compiled
    # from this state.
    at = sim.state()
    assert put(sim, positions=at.positions, diagonal=at.cell.diagonal, tilt=at.cell.tilt) == \
        ("positions",)
    against_compiled(sim, program, precision, pme, "NVE", at.positions, at.cell, 12,
                     label + ": positions written back")
    print(f"{label}: commits of tilts (small, another basis of the lattice, large, with the "
          f"diagonal) equal a simulation compiled from the committed state to the bit, and 12 "
          f"steps later; the lattice in another basis changes the potential by "
          f"{lattice:.1e} kJ/mol (tolerance {tolerance:.0e})")
    return messages


def round_trip(precision, pme):
    """A view, then a borrow that hands the positions and the cell out and
    writes them back as they are: against the run without it."""
    program = make(precision, pme)
    sim, twin = mdir.Simulation(program), mdir.Simulation(program)
    sim.run(8, energy=True)
    twin.run(8, energy=True)
    with sim.view() as view:
        assert view.step == 8
    at = sim.state()
    assert put(sim, positions=at.positions, diagonal=at.cell.diagonal, tilt=at.cell.tilt) == \
        ("positions",)
    # The positions and the velocities are the bits of before; the forces
    # are evaluated anew in the order and with the neighbor structures of
    # another activation, so they and the steps after them equal those of
    # the run without the borrow within rounding, and those of a simulation
    # compiled from this state to the bit (`commits`).
    a, b = sim.state(), twin.state()
    assert np.array_equal(a.positions, b.positions) and np.array_equal(a.velocities, b.velocities)
    assert np.array_equal(a.cell.tilt, b.cell.tilt)
    forces = float(np.abs(a.forces - b.forces).max())
    sim.run(12, energy=True)
    twin.run(12, energy=True)
    a, b = sim.state(), twin.state()
    later = float(np.abs(a.positions - b.positions).max())
    bounds = (1e-9, 1e-13) if precision == "Double" else (5e-2, 1e-6)
    assert forces < bounds[0] and later < bounds[1], (precision, forces, later)
    return forces, later


def thermostat(precision):
    """With a thermostat: a commit before the first step against a
    simulation compiled from the committed state, whose coupling begins at
    the same step."""
    program = make(precision, True, "NVT")
    sim = mdir.Simulation(program)
    sim.run(0, energy=True)
    at = sim.state()
    cell = cell_of(at.cell.diagonal, [0.03, -1.28, -1.29])
    x = strained(at.positions, at.cell, cell)
    assert put(sim, positions=x, tilt=cell.tilt) == ("positions", "cell")
    against_compiled(sim, program, precision, True, "NVT", x, cell, 40, precision + " NVT")
    print(f"{precision}: with a thermostat, a commit of tilts before the first step and 40 "
          f"steps equal a simulation compiled from the committed state to the bit")


def structure():
    """A program compiled for an orthorhombic cell: its tilts are three
    zeros that cannot be written, and its borrows are what they were."""
    loaded = mdir.load_amber(dipeptide + "/dipeptide.prmtop", dipeptide + "/dipeptide.inpcrd")

    def program(state=None):
        system, start = loaded.make_system(), loaded.make_state()
        system.cutoff, system.pairlist_distance, system.switch_distance = CUTOFF, REACH, 0.7
        system.electrostatics = mdir.Electrostatics.Cutoff
        if state is None:
            state = start.draw_velocities(system, 300.0, 7)
        integrator, ensemble, execution = mdir.Integrator(), mdir.Ensemble(), mdir.Execution()
        integrator.method = mdir.IntegratorMethod.VelocityVerlet
        integrator.timestep = 0.001
        ensemble.kind = mdir.EnsembleKind.NVE
        execution.target, execution.precision = target, mdir.Precision.Double
        execution.deterministic = True
        return mdir.compile(system, state, integrator, ensemble, execution, mdir.Schedule())

    first = program()
    sim, twin = mdir.Simulation(first), mdir.Simulation(first)
    sim.run(4, energy=True)
    twin.run(4, energy=True)
    # Reading: zeros, with the read-only flag of DLPack, which NumPy takes
    # as an array that cannot be written. Nothing counts as written.
    with sim.borrow() as borrow:
        assert borrow.tilt.shape == (3,) and borrow.cell.shape == (3,)
        tilt = np.from_dlpack(borrow.tilt)
        assert tilt.dtype == np.float64 and np.array_equal(tilt, np.zeros(3))
        assert not tilt.flags.writeable
        refusal = expect(ValueError, lambda: tilt.__setitem__(0, 0.1), "read-only")
        assert np.from_dlpack(borrow.cell).flags.writeable
        del tilt
        gc.collect()
        assert borrow.written == () and borrow.commit() == ()
    assert sim.versions["cell"] == 0 and sim.commits == []
    sim.run(4, energy=True)
    twin.run(4, energy=True)
    same(sim.state(), twin.state(), "an orthorhombic borrow whose tilts were read")
    # A consumer that ignores the flag (a legacy capsule cannot carry it)
    # and writes: the commit refuses, and nothing changes.
    with sim.borrow() as borrow:
        capsule = borrow.tilt.__dlpack__()
        ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
        ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
        legacy = ctypes.pythonapi.PyCapsule_GetPointer(capsule, b"dltensor")
        data = ctypes.c_void_p.from_address(legacy).value
        assert data == borrow.tilt.data_ptr
        (ctypes.c_double * 3).from_address(data)[1] = 0.25
        del capsule
        gc.collect()
        assert borrow.written == ("cell",)
        message = expect(mdir.InputError, borrow.commit, "compiled for an orthorhombic cell")
        assert borrow.live
    assert sim.versions["cell"] == 0 and sim.commits == []
    sim.run(4, energy=True)
    twin.run(4, energy=True)
    same(sim.state(), twin.state(), "after a refused write of the tilts")
    # A commit of the edges with the positions is what it was (D229): the
    # simulation compiled from the committed state, to the bit.
    at = sim.state()
    edges, x = at.cell.diagonal * 1.01, at.positions * 1.01
    assert put(sim, positions=x, diagonal=edges) == ("positions", "cell")
    after = sim.state()
    assert np.array_equal(after.cell.tilt, np.zeros(3))
    fresh = mdir.Simulation(program(initial(x, at.velocities, cell_of(edges, [0.0, 0.0, 0.0]))))
    fresh.run(0, energy=True)
    same(after, fresh.state(), "an orthorhombic commit of the edges")
    sim.run(12, energy=True)
    fresh.run(12, energy=True)
    same(sim.state(), fresh.state(), "an orthorhombic commit of the edges, 12 steps later")
    print("an orthorhombic program: Borrow.tilt is three zeros that cannot be written (" +
          refusal + "); a commit of the edges equals a simulation compiled from the committed "
          "state to the bit, and 12 steps later")
    return message


def images(x, vectors, molecules):
    """The observed pair term summed over every image within the cutoff,
    shifted to zero at the cutoff as the program takes it (D210)."""
    total = 0.0
    shift = SOFT_A * np.exp(-CUTOFF / SOFT_L)
    for n in np.ndindex(5, 5, 5):
        d = x[:, None, :] - x[None, :, :] + (np.array(n) - 2) @ vectors
        r = np.sqrt((d * d).sum(-1))
        take = (r < CUTOFF) & (molecules[:, None] != molecules[None, :])
        total += 0.5 * (SOFT_A * np.exp(-r[take] / SOFT_L) - shift).sum()
    return total


def narrow(precision):
    """What the builder tests: the diagonal, not the widths between the
    faces. A cell with a_x = 1.9 nm, at least twice the pairlist distance,
    whose faces of b and c are 1.54 nm apart, less than twice the cutoff:
    the pair term against a NumPy sum over the images."""
    program = make(precision, False, soft=True)
    sim = mdir.Simulation(program)
    sim.run(0, energy=True)
    at = sim.state()
    molecules = np.arange(len(at.positions)) // 3
    reference = images(at.positions, at.cell.vectors, molecules)
    first = abs(at.observables["soft.energy"] - reference) / reference
    cell = cell_of([1.9, at.cell.diagonal[1], at.cell.diagonal[2]], [0.6, -0.95, 1.3])
    h = cell.vectors
    width = abs(np.linalg.det(h)) / np.linalg.norm(np.cross(h[1], h[2]))
    assert width < 2 * CUTOFF < 2 * REACH <= cell.diagonal.min(), width
    x = strained(at.positions, at.cell, cell)
    assert put(sim, positions=x, diagonal=cell.diagonal, tilt=cell.tilt) == ("positions", "cell")
    now = sim.state()
    reference = images(x, h, molecules)
    worst = abs(now.observables["soft.energy"] - reference) / reference
    tolerance = 1e-12 if precision == "Double" else 1e-6
    assert max(first, worst) < tolerance, (precision, first, worst)
    against_compiled_soft(sim, program, precision, x, cell)
    print(f"{precision}: a cell {width:.2f} nm between two faces, less than twice the cutoff, "
          f"with a diagonal that passes: the pair term equals the sum over the images within "
          f"{max(first, worst):.1e} (tolerance {tolerance:.0e})")


def against_compiled_soft(sim, program, precision, positions, cell):
    at = sim.state()
    fresh_program = make(precision, False, state=initial(positions, at.velocities, cell),
                         capacity=program.plan["neighbor_capacity"], soft=True)
    assert fresh_program.ir == program.ir
    fresh = mdir.Simulation(fresh_program)
    fresh.run(0, energy=True)
    same(at, fresh.state(), "the narrow cell")
    assert at.observables == fresh.state().observables


def dcd_cells(path):
    """The cell records of a DCD file: a, cos gamma, b, cos beta, cos alpha, c."""
    data, cells, i = pathlib.Path(path).read_bytes(), [], 0
    while i < len(data):
        n = int.from_bytes(data[i:i + 4], "little")
        if n == 48:
            cells.append(np.frombuffer(data[i + 4:i + 52], dtype=np.float64))
        i += n + 8
    return np.array(cells)


def lower(record):
    """The lower-triangular cell of a DCD record, in nm."""
    a, cg, b, cb, ca, c = record
    bx, by = b * cg, b * np.sqrt(1.0 - cg * cg)
    cx = c * cb
    cy = c * (ca - cb * cg) / np.sqrt(1.0 - cg * cg)
    cz = np.sqrt(c * c - cx * cx - cy * cy)
    return cell_of(np.array([a, by, cz]) / 10.0, np.array([bx, cx, cy]) / 10.0)


def reporter(work):
    """The frames of a reporter that is open take the committed cell."""
    program = make("Double", False)
    sim = mdir.Simulation(program)
    path = pathlib.Path(work) / f"tilts-{target_name}.dcd"
    sim.reporters.append(mdir.TrajectoryReporter(str(path), 5))
    sim.run(5)
    at = sim.state()
    cell = cell_of(at.cell.diagonal * np.array([1.01, 1.0, 1.0]), [0.02, -1.3, 1.29])
    x = strained(at.positions, at.cell, cell)
    put(sim, positions=x, diagonal=cell.diagonal, tilt=cell.tilt)
    sim.run(5)
    sim.close_reporters()
    cells = dcd_cells(path)
    assert len(cells) == 2, len(cells)
    for record, expected in zip(cells, (at.cell, cell)):
        found = lower(record)
        assert np.allclose(found.diagonal, expected.diagonal, rtol=0, atol=1e-12), record
        assert np.allclose(found.tilt, expected.tilt, rtol=0, atol=1e-12), record
    print("the frames of an open reporter take the committed cell")


def continued(work):
    """A checkpoint written after a commit holds the committed tilts, and a
    simulation of the program, which was built with other tilts, continues
    from it with them."""
    program = make("Double", False)
    sim = mdir.Simulation(program)
    sim.run(4, energy=True)
    at = sim.state()
    cell = cell_of(at.cell.diagonal, [0.02, -1.3, 1.29])
    put(sim, positions=strained(at.positions, at.cell, cell), tilt=cell.tilt)
    path = pathlib.Path(work) / f"tilts-{target_name}.h5"
    try:
        sim.save_checkpoint(str(path))
    except mdir.UnsupportedError:
        print("a checkpoint after a commit continues with the committed tilts (not checked: "
              "no HDF5)")
        return
    assert np.array_equal(mdir.read_checkpoint(str(path)).cell.tilt, cell.tilt)
    other = mdir.Simulation(program, checkpoint=str(path))
    sim.run(10, energy=True)
    other.run(10, energy=True)
    same(sim.state(), other.state(), "continued from a checkpoint")
    print("a checkpoint after a commit continues with the committed tilts")


# --- Frames --------------------------------------------------------------------

def dcd_frames(path):
    data, records, i = pathlib.Path(path).read_bytes(), [], 0
    while i < len(data):
        n = int.from_bytes(data[i:i + 4], "little")
        records.append(data[i + 4:i + 4 + n])
        i += n + 8
    cells = [np.frombuffer(r, dtype=np.float64) for r in records if len(r) == 48]
    size = max(len(r) for r in records)
    xyz = [np.frombuffer(r, dtype=np.float32) for r in records if len(r) == size]
    count = len(xyz) // 3
    assert len(cells) == count
    positions = [np.stack(xyz[3 * k:3 * k + 3], axis=1).astype(np.float64) / 10.0
                 for k in range(count)]
    return positions, [lower(c) for c in cells]


def frames(path):
    positions, cells = dcd_frames(path)
    assert len(positions) == 20, len(positions)
    tilts = np.array([c.tilt for c in cells])
    assert np.ptp(tilts[:, 1]) > 1e-4, "the barostat did not change the tilts"
    for precision in ("Double", "Mixed"):
        program = make(precision, True)
        sim = mdir.Simulation(program)
        sim.run(0, energy=True)
        worst, spread = 0.0, []
        for x, cell in zip(positions, cells):
            assert reduced(cell)
            fields = put(sim, positions=x, diagonal=cell.diagonal, tilt=cell.tilt)
            assert fields == ("positions", "cell"), fields
            at = sim.state()
            fresh_program = make(precision, True, state=initial(x, at.velocities, cell),
                                 capacity=program.plan["neighbor_capacity"])
            assert fresh_program.ir == program.ir
            fresh = mdir.Simulation(fresh_program)
            fresh.run(0, energy=True)
            other = fresh.state()
            assert np.array_equal(at.forces, other.forces), precision
            for name in ("potential", "virial", "volume"):
                assert at.energies[name] == other.energies[name], (precision, name)
            spread.append(at.energies["potential"])
            worst = max(worst, abs(at.energies["potential"] - other.energies["potential"]))
        assert sim.versions["cell"] == 20 and sim.step == 0
        print(f"{precision}: 20 frames of a run at constant pressure (c_x from "
              f"{tilts[:, 1].min():.4f} to {tilts[:, 1].max():.4f} nm) put into one simulation: "
              f"the potential, the virial, the volume, and the forces of each equal those of a "
              f"simulation compiled from the frame, difference {worst:.1f} (potentials from "
              f"{min(spread):.0f} to {max(spread):.0f} kJ/mol)")
    print(f"dlpack tilts frames {target_name} passed")


def npt(work):
    for precision in ("Double", "Mixed"):
        program = make(precision, True, "NPT")
        whole, parts = mdir.Simulation(program), mdir.Simulation(program)
        start = whole.state().cell
        whole.run(60, energy=True)
        end = whole.state()
        # The barostat scales the tilts with their columns: the state has
        # them, and the shape of the dodecahedron stays.
        change = float(np.abs(end.cell.tilt - start.tilt).max())
        assert change > 1e-6, change
        shape = max(abs(end.cell.tilt[1] / end.cell.diagonal[0] - 0.5),
                    abs(end.cell.tilt[2] / end.cell.diagonal[1] - 0.5),
                    abs(end.cell.tilt[0]))
        assert shape < 1e-14, shape
        with whole.borrow() as borrow:
            assert np.array_equal(consumer.read(borrow.tilt), end.cell.tilt)
            assert np.array_equal(consumer.read(borrow.cell), end.cell.diagonal)
        # In parts, to the bit.
        for _ in range(3):
            parts.run(20, energy=True)
        same(parts.state(), end, precision + " NPT in parts")
        # A checkpoint ends the activation: the simulation that wrote it
        # and one that continues from it both begin an activation with the
        # tilts that the barostat left, and agree to the bit; the run that
        # did not stop kept its neighbor structures, and differs from them
        # by rounding.
        stopped = mdir.Simulation(program)
        stopped.run(40, energy=True)
        path = pathlib.Path(work) / f"npt-{target_name}-{precision}.h5"
        rounding = None
        try:
            stopped.save_checkpoint(str(path))
            assert np.array_equal(mdir.read_checkpoint(str(path)).cell.tilt,
                                  stopped.state().cell.tilt)
            continued = mdir.Simulation(program, checkpoint=str(path))
            continued.run(20, energy=True)
            stopped.run(20, energy=True)
            same(continued.state(), stopped.state(), precision + " NPT from a checkpoint")
            rounding = float(np.abs(continued.state().positions - end.positions).max())
            assert rounding < (1e-12 if precision == "Double" else 1e-6), rounding
            assert np.allclose(continued.state().cell.tilt, end.cell.tilt, rtol=0,
                               atol=1e-12 if precision == "Double" else 1e-6)
        except mdir.UnsupportedError:
            pass
        print(f"{precision}: at constant pressure the tilts follow the barostat (by {change:.1e} "
              f"nm in 60 steps, the shape kept within {shape:.0e}); a run in parts equals one "
              f"run to the bit, and one continued from a checkpoint the run that wrote it, "
              f"within " + ("no HDF5" if rounding is None else f"{rounding:.1e} nm") +
              " of the run that did not stop")

        # A commit of tilts, then a barostat that changes the cell.
        sim = mdir.Simulation(program)
        sim.run(0, energy=True)
        at = sim.state()
        cell = cell_of(at.cell.diagonal, [0.03, -1.28, -1.29])
        x = strained(at.positions, at.cell, cell)
        assert put(sim, positions=x, tilt=cell.tilt) == ("positions", "cell")
        against_compiled(sim, program, precision, True, "NPT", x, cell, 40, precision + " NPT")
        moved = float(np.abs(sim.state().cell.tilt - cell.tilt).max())
        assert moved > 1e-6, moved
        print(f"{precision}: a commit of tilts before the first step and 40 steps under the "
              f"barostat, which moves the tilts by {moved:.1e} nm, equal a simulation compiled "
              f"from the committed state to the bit")

        # A frame of the run committed into the simulation that ran it, 20
        # steps later: the borrow lends the tilts that the barostat left,
        # the evaluation is that of a simulation compiled from the frame,
        # and the barostat goes on from the cell of the frame.
        run = mdir.Simulation(program)
        run.run(20, energy=True)
        frame = run.state()
        run.run(20, energy=True)
        later = run.state().cell
        assert not np.array_equal(later.tilt, frame.cell.tilt)
        with run.borrow() as borrow:
            assert np.array_equal(consumer.read(borrow.tilt), later.tilt)
        fields = put(run, positions=frame.positions, diagonal=frame.cell.diagonal,
                     tilt=frame.cell.tilt)
        assert fields == ("positions", "cell"), fields
        with run.borrow() as borrow:
            assert np.array_equal(consumer.read(borrow.tilt), frame.cell.tilt)
        at = run.state()
        fresh = mdir.Simulation(make(precision, True, "NPT",
                                     initial(frame.positions, at.velocities, frame.cell),
                                     capacity=program.plan["neighbor_capacity"]))
        fresh.run(0, energy=True)
        other = fresh.state()
        assert np.array_equal(at.forces, other.forces), precision
        assert np.array_equal(at.cell.tilt, other.cell.tilt), precision
        # The conserved quantity of the run has the energy of its baths.
        for name in ("potential", "kinetic", "virial", "pressure", "volume"):
            assert at.energies[name] == other.energies[name], (precision, name)
        run.run(20, energy=True)
        after = run.state().cell
        assert not np.array_equal(after.tilt, frame.cell.tilt)
        kept = max(abs(after.tilt[1] / after.diagonal[0] - 0.5),
                   abs(after.tilt[2] / after.diagonal[1] - 0.5), abs(after.tilt[0]))
        assert kept < 1e-14, kept
        print(f"{precision}: a frame of the run committed into the simulation that ran it, "
              f"20 steps later: the evaluation equals that of a simulation compiled from the "
              f"frame to the bit, and the barostat goes on from its cell")

        # The frames of the run, put into a second simulation.
        source = mdir.Simulation(program)
        taken = []
        for _ in range(20):
            source.run(10)
            taken.append(source.state())
        nve = make(precision, True)
        second = mdir.Simulation(nve)
        second.run(0, energy=True)
        tilts = np.array([f.cell.tilt for f in taken])
        for frame in taken:
            fields = put(second, positions=frame.positions, diagonal=frame.cell.diagonal,
                         tilt=frame.cell.tilt)
            assert fields == ("positions", "cell"), fields
            at = second.state()
            fresh_program = make(precision, True,
                                 state=initial(frame.positions, at.velocities, frame.cell),
                                 capacity=nve.plan["neighbor_capacity"])
            assert fresh_program.ir == nve.ir
            fresh = mdir.Simulation(fresh_program)
            fresh.run(0, energy=True)
            other = fresh.state()
            assert np.array_equal(at.forces, other.forces), precision
            assert at.energies == other.energies, precision
        print(f"{precision}: 20 frames of a Python run at constant pressure (c_x from "
              f"{tilts[:, 1].min():.5f} to {tilts[:, 1].max():.5f} nm) put into one simulation: "
              f"the energies and the forces of each equal those of a simulation compiled from "
              f"the frame")
    print(f"triclinic npt {target_name} passed")


if mode == "frames":
    frames(last)
elif mode == "npt":
    npt(last)
else:
    refusals = None
    for precision in ("Double", "Mixed"):
        for pme in (True, False):
            refusals = commits(precision, pme)
            forces, later = round_trip(precision, pme)
            print(f"{precision} {'PME' if pme else 'cutoff'}: a round trip that writes the "
                  f"positions and the cell back as they are leaves the positions and the "
                  f"velocities to the bit; the forces evaluated anew differ by {forces:.1e} "
                  f"kJ/mol/nm and the positions 12 steps later by {later:.1e} nm")
        thermostat(precision)
        narrow(precision)
    refusals.append(structure())
    reporter(last)
    continued(last)
    for message in refusals:
        print("refused:", message)
    print(f"dlpack tilts {target_name} passed")
