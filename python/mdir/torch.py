"""The frame evaluator as an operation of PyTorch (docs/python-frames.md).

`evaluate(evaluator, theta, positions, cells=None, tilts=None)` returns the
energies of the frames as a tensor that autograd differentiates in the
tunables `theta` through the vector-Jacobian product of the evaluator, and
in the positions and the cell, where they are tensors that require a
gradient, through the gradients of the energy that the evaluator gives
(D252). The operation is a `torch.library.custom_op` with
`register_autograd`, so that it is opaque to `torch.compile`. Importing
this module imports torch; `import mdir` does not.
"""
import itertools
import weakref
from typing import List, Tuple

import numpy as np
import torch
from torch import Tensor

from . import _core, _frames

__all__ = ["evaluate", "FrameOutputs"]

# An operation takes tensors and numbers only, so it finds the evaluator
# and the frames of a call by a number. The number is that of the
# evaluator, the same at every call, so that a function traced by
# `torch.compile` is not traced again at each call; the serial of the call
# travels as a tensor, and tells a backward whether the record is still
# that of its forward.
_RECORDS = {}
_NUMBERS = itertools.count(1)
_SERIALS = itertools.count(1)


class _Record:
    """The last call of an evaluator through the adapter."""

    def __init__(self, evaluator):
        self.evaluator = weakref.ref(evaluator)
        self.others = (["virial"] if evaluator.terms == "all" else []) + list(evaluator._columns)
        self.depends = dict(evaluator._depends)
        self.names, self.frames, self.count, self.serial = [], None, None, 0
        self.out = None
        # What a backward by a second pass reads: the frames, and the
        # version of the values that they were evaluated at.
        self.source, self.version = None, None
        # The gradients of the energy that the call asks the evaluator for:
        # in the positions, in the strain; and whether the gradient in the
        # six numbers of the cell at fixed positions is formed from them.
        self.wants, self.cell = (False, False), False
        # The gradients that the caller named, and whether the positions or
        # the cell of the call take a gradient.
        self.asked, self.differentiated = (), False


def _cell_gradient(positions, cells, tilts, gradient, strain):
    """The gradient of the energy of each frame in the six numbers of its
    cell at fixed Cartesian positions, (K, 6): the diagonal a_x, b_y, c_z,
    then the tilts b_x, c_x, c_y.

    With the cell vectors in the rows of h, G = dU/dx, and S = dU/d(epsilon)
    under x -> (1 + epsilon) x, h -> h (1 + epsilon)^T, the chain rule gives
    S = sum_i G_i (x) x_i + (dU/dh)^T h, so that
    dU/dh = h^-T (S - sum_i G_i (x) x_i)^T. An entry (k, a) of it with
    k >= a takes S only along strains that keep the cell lower triangular.
    The sum over the particles is taken in f64."""
    count = len(strain)
    x = np.asarray(positions)
    if x.ndim == 2:
        x = x[None]
    diagonal = np.broadcast_to(np.asarray(cells, dtype=np.float64), (count, 3))
    tilt = (np.zeros((count, 3)) if tilts is None else
            np.broadcast_to(np.asarray(tilts, dtype=np.float64), (count, 3)))
    result = np.empty((count, 6))
    for n in range(count):
        h = np.array([[diagonal[n, 0], 0.0, 0.0], [tilt[n, 0], diagonal[n, 1], 0.0],
                      [tilt[n, 1], tilt[n, 2], diagonal[n, 2]]])
        moment = gradient[n].astype(np.float64).T @ np.asarray(x[n], dtype=np.float64)
        by_vector = np.linalg.solve(h.T, (strain[n] - moment).T)
        result[n] = (by_vector[0, 0], by_vector[1, 1], by_vector[2, 2],
                     by_vector[1, 0], by_vector[2, 0], by_vector[2, 1])
    return result


@torch.library.custom_op("mdir::frame_energies", mutates_args=())
def _frame_energies(call: int, gradient: bool, theta: List[Tensor], positions: Tensor,
                    cells: Tensor, tilts: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor,
                                                           Tensor, Tensor, Tensor]:
    # `positions`, `cells`, and `tilts` are the tensors of the call where
    # they require a gradient and empty otherwise: they are inputs so that
    # autograd has them, and the frames are read from the record.
    record = _RECORDS[call]
    evaluator = record.evaluator()
    changes = {}
    for name, tensor in zip(record.names, theta):
        value = np.ascontiguousarray(tensor.detach().cpu().numpy())
        if value.tobytes() != np.ascontiguousarray(evaluator.tunables[name]).tobytes():
            changes[name] = value
    if changes:
        evaluator.tunables.update(changes)
    asked = tuple(name for name, wanted in zip(_frames._GRADIENTS, record.wants) if wanted)
    try:
        out = record.out = evaluator.evaluate(
            *record.frames, gradients=(("tunables",) if gradient else ()) + asked)
    except Exception:
        # The record of a call that fails keeps nothing of it.
        record.frames = None
        raise
    record.source = record.version = None
    if gradient and out.jacobian is not None:
        rows = np.concatenate([out.jacobian[name] for name in record.names], axis=1)
    else:
        rows = np.zeros((out.count, 0))
        if gradient:
            record.source, record.version = out._source, out.version
    device = theta[0].device
    columns = ([out._virial] if evaluator.terms == "all" else []) + [
        out.observables[c] for c in evaluator._columns]
    others = np.stack(columns, axis=1) if columns else np.zeros((out.count, 0))
    particles = evaluator._particles
    in_positions = out.gradients.get("positions")
    in_strain = out.gradients.get("strain")
    in_cell = np.zeros((0, 6))
    if record.cell:
        in_cell = _cell_gradient(_frames._array(record.frames[0], "positions"),
                                 _frames._array(record.frames[1], "cells"),
                                 None if record.frames[2] is None else
                                 _frames._array(record.frames[2], "tilts"), in_positions, in_strain)
    if in_positions is None:
        in_positions = np.zeros((0, particles, 3), dtype=evaluator._force_type)
    if in_strain is None:
        in_strain = np.zeros((0, 3, 3))
    return (torch.from_numpy(out._energy.copy()).to(device), torch.from_numpy(others).to(device),
            torch.from_numpy(rows).to(device), torch.tensor(record.serial, dtype=torch.int64),
            torch.from_numpy(in_positions.copy()).to(device),
            torch.from_numpy(in_strain.copy()).to(device), torch.from_numpy(in_cell).to(device))


@torch.library.register_fake("mdir::frame_energies")
def _frame_energies_fake(call, gradient, theta, positions, cells, tilts):
    record = _RECORDS[call]
    evaluator = record.evaluator()
    count = record.count
    if count is None:
        count = torch.library.get_ctx().new_dynamic_size()
    entries = sum(int(t.shape[0]) for t in theta)
    kept = gradient and record.count is not None and (
        8 * record.count * evaluator._entries <= evaluator.jacobian_bytes)
    like = theta[0]
    force = torch.float64 if evaluator._force_type == np.float64 else torch.float32
    return (like.new_empty((count,)), like.new_empty((count, len(record.others))),
            like.new_empty((count, entries if kept else 0)),
            torch.empty((), dtype=torch.int64),
            like.new_empty((count if record.wants[0] else 0, evaluator._particles, 3), dtype=force),
            like.new_empty((count if record.wants[1] else 0, 3, 3)),
            like.new_empty((count if record.cell else 0, 6)))


def _second(what):
    """The backward of a product of the backward: a second derivative of
    the energy, which is not implemented and is not zero."""
    def refuse(ctx, *cotangents):
        raise _core.UnsupportedError(
            f"a second derivative of the energy of the frames is not implemented: the "
            f"derivative of its gradient in {what} was asked for (a backward with "
            "create_graph=True, a Hessian, or a gradient of a gradient). It is not zero")
    return refuse


@torch.library.custom_op("mdir::frame_vjp", mutates_args=())
def _frame_vjp(call: int, entries: int, serial: Tensor, energy: Tensor, others: Tensor,
               jacobian: Tensor, gradients: List[Tensor], needed: bool) -> Tensor:
    # `gradients` are the cotangents of the gradients of the energy in the
    # positions, the strain, and the cell, which are outputs as well.
    # Without `needed` no tunable takes a gradient (the backward is that of
    # the positions or the cell), and the product is not formed.
    record = _RECORDS.get(call)
    asked = torch.nonzero(others.abs().sum(dim=0)).flatten().tolist()
    names = record.others if record else [f"column {k}" for k in range(others.shape[1])]
    depends = record.depends if record else {}
    listed = [f"'{names[k]}' (which may read {', '.join(sorted(depends.get(names[k], [])))})"
              for k in asked]
    listed += [f"the gradient of the energy in {what}"
               for what, cotangent in zip(("the positions", "the strain", "the cell"), gradients)
               if cotangent.numel() and bool(cotangent.abs().sum() != 0)]
    if listed:
        raise _core.UnsupportedError(
            f"the derivative of the output {', '.join(listed)} in the tunables, the positions, "
            "or the cell is not implemented; it is not zero. Detach the output to leave that "
            "term out of a gradient")
    if not needed:
        return energy.new_zeros((entries,))
    if jacobian.shape[1] == entries:
        return energy @ jacobian
    evaluator = record.evaluator() if record else None
    if evaluator is None or record.serial != int(serial) or record.source is None:
        raise _core.SimulationError(
            "the Jacobian of these frames was not kept (jacobian_bytes), and the evaluator has "
            "evaluated other frames since: the backward cannot read the frames again")
    total = evaluator._second_pass(record.source, record.version,
                                   np.ascontiguousarray(energy.detach().cpu().numpy()))
    begins = [evaluator._slices[name] for name in record.names]
    return torch.from_numpy(np.concatenate([total[a:b] for a, b in begins])).to(energy.device)


@torch.library.register_fake("mdir::frame_vjp")
def _frame_vjp_fake(call, entries, serial, energy, others, jacobian, gradients, needed):
    return energy.new_empty((entries,))


torch.library.register_autograd("mdir::frame_vjp", _second("the tunables"))


@torch.library.custom_op("mdir::frame_input_vjp", mutates_args=())
def _frame_input_vjp(energy: Tensor, gradient: Tensor) -> Tensor:
    # The product of the cotangent of the energies, (K,), with their
    # gradient in an input, (K, ...): an operation of its own, and not a
    # multiplication of tensors, so that its derivative is refused. As a
    # multiplication it would be differentiable in the cotangent alone, and
    # a second derivative in the input would come out as zero.
    return energy.to(gradient.dtype).reshape((-1,) + (1,) * (gradient.dim() - 1)) * gradient


@torch.library.register_fake("mdir::frame_input_vjp")
def _frame_input_vjp_fake(energy, gradient):
    return gradient.new_empty(gradient.shape)


torch.library.register_autograd("mdir::frame_input_vjp", _second("the positions or the cell"))


def _setup(ctx, inputs, output):
    call, gradient, theta, positions, cells, tilts = inputs
    ctx.call, ctx.jacobian, ctx.serial = call, output[2], output[3]
    ctx.sizes, ctx.needed = [int(t.shape[0]) for t in theta], gradient
    ctx.in_positions, ctx.in_cell = output[4], output[6]
    # The inputs that take a gradient, each with its shape, type, and device.
    ctx.inputs = [(tuple(t.shape), t.dtype, t.device) if t.numel() else None
                  for t in (positions, cells, tilts)]


def _backward(ctx, energy, others, jacobian, serial, in_positions, in_strain, in_cell):
    product = _frame_vjp(ctx.call, sum(ctx.sizes), ctx.serial, energy, others, ctx.jacobian,
                         [in_positions, in_strain, in_cell], ctx.needed)

    def of(given, values):
        # The gradient in the shape, the type, and the device of its input:
        # one frame given without its axis, or one cell for every frame.
        shape, dtype, device = given
        if values.dim() > len(shape):
            values = values.sum(dim=0)
        return values.reshape(shape).to(device=device, dtype=dtype)

    positions, cells, tilts = ctx.inputs
    results = [None, None, None]
    if positions is not None:
        results[0] = of(positions, _frame_input_vjp(energy, ctx.in_positions))
    if cells is not None or tilts is not None:
        by_cell = _frame_input_vjp(energy, ctx.in_cell)
        if cells is not None:
            results[1] = of(cells, by_cell[:, :3])
        if tilts is not None:
            results[2] = of(tilts, by_cell[:, 3:])
    return (None, None, list(torch.split(product, ctx.sizes)), *results)


torch.library.register_autograd("mdir::frame_energies", _backward, setup_context=_setup)


class FrameOutputs:
    """What `evaluate` returns: `energy`, `virial`, and `volume`, tensors
    (K,) float64 on the device of `theta`; `observables`, a dict of the
    names of the observed columns to such tensors; and `gradients`, a dict
    of the gradients of the energy that were asked, "positions" (K, N, 3)
    and "strain" (K, 3, 3). `energy` is differentiable in `theta` and in
    the positions and the cell that require a gradient. Any other output
    that reads no tunable of `theta` (`depends`), in a call whose positions
    and cell take no gradient, is a constant, without a graph; otherwise it
    raises `UnsupportedError` from a backward that reaches it."""

    def __init__(self, out, energy, virial, volume, observables, gradients):
        self._out, self._energy, self._virial = out, energy, virial
        self.volume, self.observables, self.gradients = volume, observables, gradients
        self.depends, self.version, self.count, self.zero = (
            out.depends, out.version, out.count, out.zero)
        self.terms, self.unavailable = out.terms, out.unavailable

    @property
    def energy(self):
        return self._out._only("energy", "all", self._energy)

    @property
    def virial(self):
        return self._out._only("virial", "all", self._virial)

    @property
    def dependent_energy(self):
        return self._out._only("dependent_energy", "dependent", self._energy)

    def __repr__(self):
        return f"FrameOutputs(count={self.count}, version={self.version})"


def _takes_gradient(given):
    return isinstance(given, Tensor) and given.requires_grad and torch.is_grad_enabled()


@torch.compiler.disable
def _begin(evaluator, theta, positions, cells, tilts, gradients):
    """Checks the arguments and records the call; returns the number of the
    evaluator, whether the derivative in the tunables is needed, and the
    three inputs of the operation that take a gradient."""
    if not isinstance(evaluator, _frames.FrameEvaluator):
        raise _core.InputError("mdir.torch.evaluate takes an mdir.FrameEvaluator")
    if not theta:
        raise _core.InputError("mdir.torch.evaluate: theta names no tunable")
    devices = set()
    for name, tensor in theta.items():
        if name not in evaluator._slices:
            raise _core.InputError(f"the evaluator has no tunable named '{name}'")
        begin, end = evaluator._slices[name]
        if (not isinstance(tensor, Tensor) or tensor.dtype != torch.float64 or
                tuple(tensor.shape) != (end - begin,)):
            raise _core.InputError(f"theta['{name}']: expected a float64 tensor of shape "
                                   f"({end - begin},)")
        devices.add(tensor.device)
    if len(devices) != 1:
        raise _core.InputError("mdir.torch.evaluate: the tensors of theta are on one device")
    # "tunables" names what autograd decides here: the derivative in `theta`
    # is kept where a tensor of it requires a gradient.
    asked = tuple(name for name in _frames._named(gradients) if name != "tunables")
    in_positions = _takes_gradient(positions)
    in_cells, in_tilts = _takes_gradient(cells), _takes_gradient(tilts)
    if evaluator.terms != "all" and (asked or in_positions or in_cells or in_tilts):
        raise _core.UnsupportedError(
            "the gradients in the positions, the strain, and the cell are not given by a frame "
            "evaluator with terms='dependent': it evaluates the terms that a tunable enters "
            "alone, whose gradients are not those of the potential energy")
    # A gradient is taken in tensors that are the arrays of the call: frames
    # of an iterable carry none, and `_array` refuses those that require one.
    for given, name in ((positions, "positions"), (cells, "cells"), (tilts, "tilts")):
        if _takes_gradient(given) and not given.dtype.is_floating_point:
            raise _core.InputError(f"{name} that require a gradient are floating-point tensors")
    if in_tilts and cells is None:
        raise _core.InputError("frames: tilts are given with the diagonals of the cells, cells")
    number = getattr(evaluator, "_torch_number", None)
    if number is None:
        number = evaluator._torch_number = next(_NUMBERS)
        _RECORDS[number] = _Record(evaluator)
        weakref.finalize(evaluator, _RECORDS.pop, number, None)
    record = _RECORDS[number]

    def held(given):
        return given.detach() if isinstance(given, Tensor) else given

    record.names, record.frames = list(theta), (held(positions), held(cells), held(tilts))
    record.serial, record.out = next(_SERIALS), None
    record.cell = in_cells or in_tilts
    record.wants = (in_positions or record.cell or "positions" in asked,
                    record.cell or "strain" in asked)
    record.asked, record.differentiated = asked, in_positions or record.cell
    record.count = None
    if isinstance(positions, (np.ndarray, Tensor)):
        record.count = positions.shape[0] if positions.ndim == 3 else 1
    elif hasattr(positions, "__len__") and not hasattr(positions, "positions"):
        record.count = len(positions)
    gradient = torch.is_grad_enabled() and any(t.requires_grad for t in theta.values())
    empty = torch.zeros(0)
    return number, gradient, [given if takes else empty for given, takes in
                              ((positions, in_positions), (cells, in_cells), (tilts, in_tilts))]


@torch.compiler.disable
def _finish(number, names, energy, others, in_positions, in_strain):
    """The outputs of the call, and the end of its record: the arguments
    and the arrays are not kept beyond it."""
    record = _RECORDS[number]
    out, record.out, record.frames = record.out, None, None
    named = set(names)
    constant = others.detach()

    def output(name, column):
        # With positions or a cell that take a gradient every output is a
        # function of them, whose derivative is not implemented either.
        reads = record.differentiated or out.depends[name] & named
        return others[:, column] if reads else constant[:, column]

    observables = {name: output(name, k) for k, name in enumerate(record.others)
                   if name != "virial"}
    volume = torch.from_numpy(out.volume.copy()).to(energy.device)
    virial = output("virial", 0) if out.terms == "all" else None
    # The gradients of the energy read what the energy reads.
    reads = record.differentiated or (out.terms == "all" and out.depends["energy"] & named)
    gradients = {name: tensor if reads else tensor.detach()
                 for name, tensor in (("positions", in_positions), ("strain", in_strain))
                 if name in record.asked}
    return FrameOutputs(out, energy, virial, volume, observables,
                        {name: gradients[name] for name in record.asked})


def evaluate(evaluator, theta, positions, cells=None, tilts=None, *, gradients=()):
    """Evaluates the frames at the values `theta` of the tunables.

    `theta` is a dict of names of tunables of the evaluator to float64
    tensors of the shape of their values, on one device; a tunable that is
    not named keeps the value of the evaluator. `positions`, `cells`, and
    `tilts` are those of `FrameEvaluator.evaluate`, and so are the names of
    `gradients`: "positions" and "strain" give those gradients of the
    energy as outputs; "tunables" is taken and changes nothing, since the
    derivative in `theta` is autograd's, kept where a tensor of `theta`
    requires a gradient. The values of the evaluator are updated where they
    differ, without a compilation (D213).

    Where `positions` is one tensor (K, N, 3) that requires a gradient, and
    where `cells` and `tilts` are tensors that do, the energy is
    differentiable in them: the function is U(x, h) at the Cartesian
    positions given. Second derivatives are refused (`UnsupportedError`).
    """
    number, gradient, inputs = _begin(evaluator, theta, positions, cells, tilts, gradients)
    energy, others, _, _, in_positions, in_strain, _ = _frame_energies(
        number, gradient, list(theta.values()), *inputs)
    return _finish(number, list(theta), energy, others, in_positions, in_strain)
