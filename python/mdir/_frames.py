"""The frame evaluator (docs/python-frames.md): the potential energy that
the forces sample, the virial, and the observed columns at stored frames,
with the vector-Jacobian product in the tunables.

A fit by reweighting (Thaler and Zavadlav, Nat. Commun. 12, 6884 (2021))
needs U and dU/dtheta at frames sampled at other values of theta; the
weights and the loss stay in the framework.
"""
from collections.abc import Mapping

import numpy as np

from . import _core

#: The bytes of float64 positions that one call into the extension takes.
_BATCH_BYTES = 1 << 26


def _array(source, name):
    """A NumPy array of `source` on the host, without a copy where it is one
    already. A tensor that requires a gradient is refused: the gradient in
    the positions and the strain is not returned yet, and is not zero."""
    if getattr(source, "requires_grad", False):
        raise _core.UnsupportedError(
            f"{name} require a gradient: the gradients of the energy in the positions and "
            "the cell are not returned by the frame evaluator yet; detach them")
    if isinstance(source, np.ndarray):
        return source
    if hasattr(source, "__dlpack__"):
        try:
            return np.from_dlpack(source)
        except Exception:
            # A buffer of a device, or one that NumPy cannot alias.
            if hasattr(source, "cpu"):
                return np.asarray(source.cpu())
            raise _core.InputError(f"{name}: cannot read the buffer on the host") from None
    return np.asarray(source)


def _cell(cell):
    """The cell of one frame as its diagonal a_x, b_y, c_z and its tilts
    b_x, c_x, c_y (D238), each (3,) float64, the tilts None for an
    orthorhombic cell given by its edges: the one place that knows the
    shapes a cell is given in. Takes an `mdir.Cell` (or any object with
    `diagonal` and `tilt` in its convention), the array of the three edges,
    and the (3, 3) matrix of the cell vectors in rows, lower triangular, as
    a frame of `mdir.read_h5md` unpacks it for a triclinic cell (D239)."""
    if cell is None:
        return None, None
    if hasattr(cell, "tilt"):
        return (np.asarray(cell.diagonal, dtype=np.float64).reshape(3),
                np.asarray(cell.tilt, dtype=np.float64).reshape(3))
    given = np.asarray(_array(cell, "cells"), dtype=np.float64)
    if given.shape == (3, 3):
        if given[0, 1] != 0.0 or given[0, 2] != 0.0 or given[1, 2] != 0.0:
            raise _core.InputError(
                "frames: the cell vectors of a frame are the rows of a lower-triangular "
                "matrix, a = (a_x, 0, 0), b = (b_x, b_y, 0), c = (c_x, c_y, c_z)")
        return np.array(np.diag(given)), np.array([given[1, 0], given[2, 0], given[2, 1]])
    if given.shape != (3,):
        raise _core.InputError("frames: the cell of a frame is an mdir.Cell, its three edges, "
                               f"shape (3,), or its vectors, shape (3, 3); found {given.shape}")
    return given, None


def _cells(cells, tilts, count):
    """The cells of `count` frames given as arrays: `cells` (3,) or (K, 3),
    the diagonal a_x, b_y, c_z (the edges of an orthorhombic cell), and
    `tilts` None or (3,) or (K, 3), b_x, c_x, c_y."""
    def rows(values, name):
        if values is None:
            return None
        if hasattr(values, "tilt"):
            raise _core.InputError(f"frames: {name} is an array; give an mdir.Cell with "
                                   "each frame of an iterable")
        a = np.asarray(_array(values, name), dtype=np.float64)
        if a.shape == (3,):
            a = np.broadcast_to(a, (count, 3))
        if a.shape != (count, 3):
            raise _core.InputError(f"frames: expected {name} of shape (3,) or ({count}, 3); "
                                   f"found {a.shape}")
        return a
    if cells is None and tilts is not None:
        raise _core.InputError("frames: tilts are given with the diagonals of the cells, "
                               "cells")
    return rows(cells, "cells"), rows(tilts, "tilts")


class _Arrays:
    """Frames given as one array (K, N, 3), read a batch at a time: an array
    on disk is not read whole."""

    reusable = True

    def __init__(self, positions, cells, tilts, particles):
        if positions.ndim == 2:
            positions = positions[None]
        if positions.ndim != 3 or positions.shape[1:] != (particles, 3):
            raise _core.InputError(f"frames: expected positions of shape (K, {particles}, 3) or "
                                   f"({particles}, 3) in nm, in the order of the input; found "
                                   f"{positions.shape}")
        if positions.dtype not in (np.float64, np.float32):
            raise _core.InputError("frames: positions are float64 or float32; found "
                                   f"{positions.dtype}")
        self.positions, self.count = positions, positions.shape[0]
        self.cells, self.tilts = _cells(cells, tilts, self.count)
        self.step = max(1, _BATCH_BYTES // max(24 * particles, 1))

    def batches(self):
        for at in range(0, self.count, self.step):
            end = min(at + self.step, self.count)
            yield (np.ascontiguousarray(self.positions[at:end], dtype=np.float64),
                   None if self.cells is None else np.ascontiguousarray(self.cells[at:end]),
                   None if self.tilts is None else np.ascontiguousarray(self.tilts[at:end]))


class _Items:
    """Frames given one at a time by an iterable: each item an object with
    `positions` and `cell` (an `mdir.State`, or the frame of a reader of a
    file), a pair (positions, cell), or the positions alone."""

    def __init__(self, items, particles):
        self.items, self.particles = items, particles
        # An iterator is read once; a second pass needs the frames again.
        self.reusable = iter(items) is not items
        self.count = len(items) if hasattr(items, "__len__") else None
        self.step = max(1, _BATCH_BYTES // max(24 * particles, 1))

    def _frame(self, item, index):
        if hasattr(item, "positions"):
            positions, cell = item.positions, getattr(item, "cell", None)
        elif isinstance(item, (tuple, list)) and len(item) == 2:
            positions, cell = item
        else:
            positions, cell = item, None
        x = _array(positions, "positions")
        if x.shape != (self.particles, 3):
            raise _core.InputError(f"frame {index}: expected positions of shape "
                                   f"({self.particles}, 3); found {x.shape}")
        return (np.asarray(x, dtype=np.float64),) + _cell(cell)

    def batches(self):
        x, cells, tilts = [], [], []

        def flush():
            given = [c is not None for c in cells]
            if any(given) and not all(given):
                raise _core.InputError("frames: some frames give a cell and some do not")
            tilted = [t is not None for t in tilts]
            # A frame without tilts among frames with them is orthorhombic.
            rows = (np.array([t if t is not None else np.zeros(3) for t in tilts])
                    if any(tilted) else None)
            batch = (np.ascontiguousarray(np.stack(x)), np.array(cells) if all(given) else None,
                     rows)
            x.clear(), cells.clear(), tilts.clear()
            return batch

        for index, item in enumerate(self.items):
            frame = self._frame(item, index)
            x.append(frame[0]), cells.append(frame[1]), tilts.append(frame[2])
            if len(x) == self.step:
                yield flush()
        if x:
            yield flush()


def _source(positions, cells, tilts, particles):
    """The frames of a call, as batches that can be asked for."""
    array_like = isinstance(positions, np.ndarray) or hasattr(positions, "__dlpack__") or (
        hasattr(positions, "__array__") and not hasattr(positions, "positions"))
    if array_like:
        return _Arrays(_array(positions, "positions"), cells, tilts, particles)
    if hasattr(positions, "positions"):
        positions = [positions]         # one State
    if cells is not None or tilts is not None:
        raise _core.InputError("frames: an iterable gives the cell of each frame with the "
                               "frame; cells and tilts must be None")
    try:
        iter(positions)
    except TypeError:
        raise _core.InputError("frames: expected an array (K, N, 3), or an iterable of states "
                               "or of pairs (positions, cell)") from None
    return _Items(positions, particles)


class FrameGradient(Mapping):
    """The product of a cotangent on the energies of the frames with their
    derivative in the tunables: a read-only mapping of the names of the
    tunables to arrays of the shape of their values, in kJ/mol per unit of
    the tunable times the unit of the cotangent, as `TunableGradient`
    (D230) is at one state."""

    def __init__(self, values, zero, version):
        self._values, self.zero, self.version = values, zero, version

    def __getitem__(self, name):
        return self._values[name]

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)

    def __repr__(self):
        return f"FrameGradient(version={self.version}, {list(self._values)})"


def _frozen(array):
    array.setflags(write=False)
    return array


class FrameEnergies:
    """What `FrameEvaluator.evaluate` returns; see docs/python-frames.md.

    With `terms="all"` it has `energy`, the potential energy, and `virial`;
    with `terms="dependent"` it has `dependent_energy`, the energy of the
    terms that a tunable enters, which is not the potential energy: each
    name raises in the other mode, and `unavailable` lists what the mode
    does not give."""

    def __init__(self, evaluator, source, energy, virial, volume, observed, rows, gradient):
        self._evaluator, self._source, self._gradient = evaluator, source, gradient
        self._energy, self._virial, self.volume = _frozen(energy), _frozen(virial), _frozen(volume)
        self.terms, self.unavailable = evaluator.terms, evaluator.unavailable
        self.count = len(energy)
        self.observables = {name: _frozen(np.ascontiguousarray(observed[:, k]))
                            for k, name in enumerate(evaluator._columns)}
        self.depends = dict(evaluator._depends)
        self.version = evaluator.tunables.version
        self.zero = evaluator._zero
        self._rows = None if rows is None else _frozen(rows)
        self.jacobian = None if rows is None else {
            name: self._rows[:, begin:end] for name, (begin, end) in evaluator._slices.items()}

    def _only(self, name, mode, value):
        if self.terms != mode:
            raise _core.UnsupportedError(
                f"'{name}' is not given by a frame evaluator with terms='{self.terms}': " +
                ("it evaluates the terms that a tunable enters alone, whose energy is "
                 "'dependent_energy' and is not the potential energy"
                 if self.terms == "dependent" else
                 "its energy is 'energy', that of the whole potential"))
        return value

    @property
    def energy(self):
        """The potential energy that the forces sample (D210), kJ/mol."""
        return self._only("energy", "all", self._energy)

    @property
    def virial(self):
        return self._only("virial", "all", self._virial)

    @property
    def dependent_energy(self):
        """The energy of the terms that a tunable enters, kJ/mol: its
        differences in the tunables and its derivative are those of the
        potential energy."""
        return self._only("dependent_energy", "dependent", self._energy)

    def vjp(self, cotangent):
        """sum_n cotangent[n] dU_n/dtheta, for a cotangent (K,) float64 on
        the energy: a product with the Jacobian if it was kept, a second pass
        over the frames otherwise."""
        evaluator = self._evaluator
        if not self._gradient:
            raise _core.InputError("these frames were evaluated with gradient=False")
        g = np.asarray(_array(cotangent, "cotangent"), dtype=np.float64)
        if g.shape != (self.count,):
            raise _core.InputError(f"vjp: expected a cotangent of shape ({self.count},); found "
                                   f"{g.shape}")
        if not np.all(np.isfinite(g)):
            raise _core.InputError("vjp: the cotangent is not finite")
        if self._rows is not None:
            total = g @ self._rows
        else:
            total = evaluator._second_pass(self._source, self.version, g)
        values = {name: _frozen(np.ascontiguousarray(total[begin:end]))
                  for name, (begin, end) in evaluator._slices.items()}
        return FrameGradient(values, self.zero, self.version)

    def __repr__(self):
        return (f"FrameEnergies(count={self.count}, version={self.version}, "
                f"jacobian={'kept' if self._rows is not None else 'not kept'})")


class FrameEvaluator:
    """Evaluates a program at stored frames, in a simulation of its own.

    `program` is compiled with tunables and `System.tunable_gradient`: the
    energy of a frame is the potential that the forces sample (D210), which
    that program carries. The Jacobian of the energies in the tunables,
    8 K M bytes, is kept up to `jacobian_bytes`; above it `vjp` evaluates
    the frames again."""

    def __init__(self, program, jacobian_bytes=1 << 28, terms="all"):
        if terms not in ("all", "dependent"):
            raise _core.InputError("FrameEvaluator: terms is 'all' or 'dependent'")
        given = list(program.plan.get("observables", []))
        if terms == "dependent":
            # A program of its own with the terms that a tunable enters,
            # built from the model of the one given; its simulations take
            # its code as those of any program do (D236).
            program = program._dependent()
        self.terms = terms
        plan = program.plan
        declared = plan.get("tunables", [])
        if not declared:
            raise _core.InputError("FrameEvaluator: the program declares no tunable parameters "
                                   "(System.tunables)")
        if any("gradient" not in d for d in declared):
            raise _core.InputError(
                "FrameEvaluator: the program was compiled without the derivative in the "
                "tunables, whose potential gives the energy of a frame: set "
                "System.tunable_gradient = True and compile again")
        if int(jacobian_bytes) < 0:
            raise _core.InputError("FrameEvaluator: jacobian_bytes is not negative")
        self.program, self.jacobian_bytes = program, int(jacobian_bytes)
        self._simulation = _core.Simulation(program)
        # A part of this simulation is one evaluation.
        self._simulation.part_seconds = 1e9
        self._particles = self._simulation._particle_count
        # The cell of a frame that gives none is that of the program, not
        # that of the last frame evaluated.
        cell = self._simulation.state().cell
        edges = np.array(cell.diagonal, dtype=np.float64)
        self._edges = edges if np.all(edges > 0.0) else None
        self._tilt = np.array(cell.tilt, dtype=np.float64)
        self._slices, at = {}, 0
        for d in declared:
            self._slices[d["name"]] = (at, at + d["entries"])
            at += d["entries"]
        self._entries = at
        self._zero = frozenset(d["name"] for d in declared if d["gradient"] == "zero")
        self._columns = list(plan.get("observables", []))
        # What each output may read at fixed positions. The energy and the
        # virial: every tunable that the energy reads. A column of `observe`
        # is of one term: the constants and parameters of that term that are
        # tunable, and the charges, which an expression may read; the
        # Lennard-Jones parameters of the types enter no observed term
        # (D230 refuses a pair term that reads them). The volume: none.
        reads = frozenset(d["name"] for d in declared) - self._zero
        if terms == "all":
            self._depends = {"energy": reads, "virial": reads, "volume": frozenset()}
            self.unavailable = ("dependent_energy",)
        else:
            self._depends = {"dependent_energy": reads, "volume": frozenset()}
            self.unavailable = ("energy", "virial") + tuple(
                c for c in given if c not in self._columns)
        for column in self._columns:
            term = column.rsplit(".", 1)[0]
            self._depends[column] = frozenset(
                d["name"] for d in declared if d["name"] in reads and
                (d["term"] == term or d["parameter"] == "charge"))

    def _second_pass(self, source, version, cotangent):
        """The product of `cotangent` with the derivative of the energies
        of the frames `source`, which were evaluated at the version
        `version` of the values, by evaluating them again."""
        if self.tunables.version != version:
            raise _core.SimulationError(
                "the Jacobian of these frames was not kept (jacobian_bytes), and the values of "
                f"the tunables changed since they were evaluated (version {version}, now "
                f"{self.tunables.version}): a second pass would differentiate another potential")
        total, at = np.zeros(self._entries), 0
        for x, cells, tilts in source.batches():
            rows = self._simulation._evaluate_frames(x, *self._cells(x, cells, tilts), True, at)[4]
            if at + len(x) > len(cotangent):
                break
            total += cotangent[at:at + len(x)] @ rows
            at += len(x)
        if at != len(cotangent):
            raise _core.InputError(f"vjp: the frames gave another number of frames in the second "
                                   f"pass than the {len(cotangent)} of the first")
        return total

    def _cells(self, x, cells, tilts):
        """The edges and the tilts of the frames of a batch: those given
        (edges alone are an orthorhombic cell), or those of the program."""
        if cells is not None or self._edges is None:
            return cells, tilts
        return (np.ascontiguousarray(np.broadcast_to(self._edges, (len(x), 3))),
                np.ascontiguousarray(np.broadcast_to(self._tilt, (len(x), 3))))

    @property
    def tunables(self):
        """The values of the tunables that the frames are evaluated at
        (`TunableValues`, D213); those of the program at first."""
        return self._simulation.tunables

    def evaluate(self, positions, cells=None, tilts=None, *, gradient=True):
        """The energies, virials, volumes, and observed columns of the
        frames `positions`, and with `gradient` their derivative in the
        tunables; see docs/python-frames.md."""
        simulation = self._simulation
        source = _source(positions, cells, tilts, self._particles)
        keep = bool(gradient)
        if keep and source.count is not None:
            keep = 8 * source.count * self._entries <= self.jacobian_bytes
        if gradient and not keep and not source.reusable:
            raise _core.InputError(
                "frames: the Jacobian of these frames is above jacobian_bytes, so that vjp "
                "needs the frames again, and an iterator gives them once: give a sequence")
        parts, at = [], 0
        for x, c, t in source.batches():
            if keep and 8 * (at + len(x)) * self._entries > self.jacobian_bytes:
                # The rows pass the bound: they are dropped, and vjp reads
                # the frames again.
                if not source.reusable:
                    raise _core.InputError(
                        f"frame {self.jacobian_bytes // (8 * self._entries)}: the Jacobian "
                        f"passes jacobian_bytes ({self.jacobian_bytes}), so that vjp needs the "
                        "frames again, and an iterator gives them once: give a sequence")
                keep = False
                parts = [p[:4] + (None,) for p in parts]
            parts.append(simulation._evaluate_frames(x, *self._cells(x, c, t), keep, at))
            at += len(x)
        if not parts:
            raise _core.InputError("frames: no frames were given")
        if source.count is None:
            source.count = at
        columns = [np.concatenate([p[k] for p in parts]) for k in range(4)]
        rows = np.concatenate([p[4] for p in parts]) if keep else None
        return FrameEnergies(self, source, *columns, rows, bool(gradient))

    def __repr__(self):
        return (f"FrameEvaluator(terms='{self.terms}', tunables={list(self._slices)}, "
                f"observables={self._columns})")
