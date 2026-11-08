"""The frame evaluator as an operation of PyTorch (docs/python-frames.md).

`evaluate(evaluator, theta, positions, cells=None, tilts=None)` returns the
energies of the frames as a tensor that autograd differentiates in the
tunables `theta` through the vector-Jacobian product of the evaluator, and
in the positions and the cell, where they are tensors that require a
gradient, through the gradients of the energy that the evaluator gives
(D252). The operation is a `torch.library.custom_op` with
`register_autograd`, so that it is opaque to `torch.compile`; it has no
rule of forward mode, which is refused by name (D[torch-forward-refusal]).
Importing this module imports torch; `import mdir` does not.
"""
import itertools
import weakref
from typing import List, Tuple

import numpy as np
import torch
import torch.autograd.forward_ad as _forward_ad
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


def _refuse_forward(what, *tensors):
    """Refuses a tangent of forward-mode AD on a tensor that enters an
    operation (D[torch-forward-refusal], #777). The operations have a
    backward alone (`register_autograd`), and PyTorch gives no tangent
    through them and raises nothing: the tangent would be a silent zero.
    The tangent is seen where the tensor enters, by the public
    `torch.autograd.forward_ad.unpack_dual`, at the dual level of
    `torch.autograd.forward_ad` or of `torch.func.jvp`; outside any dual
    level it returns at once. Frames given as a list or a tuple of tensors
    are looked at one by one; an iterable of another kind is not consumed
    here, and its tensors are refused when they are read (`_array`) only
    where they require a gradient."""
    for tensor in tensors:
        if isinstance(tensor, (list, tuple)):
            _refuse_forward(what, *tensor)
        elif isinstance(tensor, Tensor) and _forward_ad.unpack_dual(tensor).tangent is not None:
            raise _core.UnsupportedError(
                f"forward mode through mdir.torch.evaluate (torch.func.jvp, jacfwd, or "
                f"torch.autograd.forward_ad, here a tangent on {what}) is not implemented; "
                "reverse mode (torch.autograd.grad / backward) is. The tangent would not be "
                "zero")


def _cell_gradient(positions, cells, tilts, gradient, strain):
    """The gradient of the energy of each frame in the six numbers of its
    cell at fixed Cartesian positions, (K, 6): the diagonal a_x, b_y, c_z,
    then the tilts b_x, c_x, c_y.

    With the cell vectors in the rows of h, G = dU/dx, and S = dU/d(epsilon)
    under x -> (1 + epsilon) x, h -> h (1 + epsilon)^T, the chain rule gives
    S = sum_i G_i (x) x_i + (dU/dh)^T h, so that
    dU/dh = h^-T (S - sum_i G_i (x) x_i)^T. An entry (k, a) of it with
    k >= a takes S only along strains that keep the cell lower triangular.
    It is formed by the function of the extension that the handler of
    `mdir.jax` uses (D371), so that the two adapters give
    the same bits: the sum over the particles in f64 in the order of the
    input, and a back substitution."""
    count = len(strain)
    x = np.asarray(positions, dtype=np.float64)
    if x.ndim == 2:
        x = x[None]
    x = np.broadcast_to(x, (count,) + x.shape[1:])
    diagonal = np.broadcast_to(np.asarray(cells, dtype=np.float64), (count, 3))
    tilt = (np.zeros((count, 3)) if tilts is None else
            np.broadcast_to(np.asarray(tilts, dtype=np.float64), (count, 3)))
    by_vector = _core._frame_cell_gradient(x, gradient, strain,
                                           np.concatenate([diagonal, tilt], axis=1))
    return np.stack([by_vector[:, 0, 0], by_vector[:, 1, 1], by_vector[:, 2, 2],
                     by_vector[:, 1, 0], by_vector[:, 2, 0], by_vector[:, 2, 1]], axis=1)


@torch.library.custom_op("mdir::frame_energies", mutates_args=())
def _frame_energies(call: int, gradient: bool, theta: List[Tensor]) -> Tuple[
        Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    # The node of the tunables. The frames are read from the record; the
    # positions and the cell that take a gradient enter the graph through
    # nodes of their own (_frame_positions, _frame_cell, _frame_energy_in),
    # so that the engine of autograd runs this backward only on paths to
    # `theta`, and those only on paths to the positions or the cell.
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
    # A backward by a second pass, or one with a cotangent on the gradient
    # in the positions, by the product program (D382),
    # reads the frames again.
    if (gradient and (out.jacobian is None or record.wants[0])) or record.differentiated:
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
def _frame_energies_fake(call, gradient, theta):
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


def _nonzero(tensor):
    return tensor.numel() and bool(tensor.abs().sum() != 0)


@torch.library.custom_op("mdir::frame_vjp", mutates_args=())
def _frame_vjp(call: int, entries: int, serial: Tensor, energy: Tensor, others: Tensor,
               jacobian: Tensor, gradients: List[Tensor], needed: bool) -> Tensor:
    # `gradients` are the cotangents of the gradients of the energy in the
    # positions, the strain, and the cell, which are outputs as well.
    # Without `needed` no tunable takes a gradient (the backward is that of
    # the positions or the cell), and the product is not formed. A
    # cotangent on the gradient in the positions takes the product of
    # force matching (D381) with that of the energy,
    # in one pass of the product program.
    record = _RECORDS.get(call)
    asked = torch.nonzero(others.abs().sum(dim=0)).flatten().tolist()
    names = record.others if record else [f"column {k}" for k in range(others.shape[1])]
    depends = record.depends if record else {}
    listed = [f"'{names[k]}' (which may read {', '.join(sorted(depends.get(names[k], [])))})"
              for k in asked]
    if _nonzero(gradients[1]):
        listed.append("the gradient of the energy in the strain")
    if listed:
        raise _core.UnsupportedError(
            f"the derivative of the output {', '.join(listed)} in the tunables, the positions, "
            "or the cell is not implemented; it is not zero. Detach the output to leave that "
            "term out of a gradient")
    # The gradient in the six numbers of the cell is no output: a cotangent
    # on it comes from a backward of the backward in the cell.
    if _nonzero(gradients[2]):
        raise _core.UnsupportedError(
            "a second derivative of the energy of the frames is not implemented: the derivative "
            "of the gradient of the energy in the cell was asked for (a backward with "
            "create_graph=True, a Hessian, or a gradient of a gradient). It is not zero")
    if not needed:
        return energy.new_zeros((entries,))
    product = _nonzero(gradients[0])
    if not product and jacobian.shape[1] == entries:
        return energy @ jacobian
    evaluator = record.evaluator() if record else None
    if evaluator is None or record.serial != int(serial) or record.source is None:
        raise _core.SimulationError(
            "the backward needs the frames again (the Jacobian was not kept, jacobian_bytes, or "
            "the gradient in the positions has a cotangent), and the evaluator has evaluated "
            "other frames since: the backward cannot read the frames again")
    g = np.ascontiguousarray(energy.detach().to(torch.float64).cpu().numpy())
    if product:
        v = np.ascontiguousarray(gradients[0].detach().to(torch.float64).cpu().numpy())
        total = evaluator._product_pass(record.source, record.version, g, v)
    else:
        total = evaluator._second_pass(record.source, record.version, g)
    begins = [evaluator._slices[name] for name in record.names]
    return torch.from_numpy(np.concatenate([total[a:b] for a, b in begins])).to(energy.device)


@torch.library.register_fake("mdir::frame_vjp")
def _frame_vjp_fake(call, entries, serial, energy, others, jacobian, gradients, needed):
    return energy.new_empty((entries,))


torch.library.register_autograd("mdir::frame_vjp", _second("the tunables"))


@torch.library.custom_op("mdir::frame_input_vjp", mutates_args=())
def _frame_input_vjp(energy: Tensor, gradient: Tensor) -> Tensor:
    # The product of the cotangent of the energies, (K,), with their
    # gradient in an input, (K, ...): an operation of its own, whose
    # derivative in `gradient` reaches the operation of the energies, which
    # gave it as an output: the product of force matching in the tunables
    # (D382), and the refusal of the Hessian in the
    # positions and of the derivative in the cell.
    return energy.to(gradient.dtype).reshape((-1,) + (1,) * (gradient.dim() - 1)) * gradient


@torch.library.register_fake("mdir::frame_input_vjp")
def _frame_input_vjp_fake(energy, gradient):
    return gradient.new_empty(gradient.shape)


def _input_setup(ctx, inputs, output):
    ctx.save_for_backward(*inputs)


def _input_backward(ctx, cotangent):
    _refuse_forward("the cotangent of a gradient of the energy", cotangent)
    energy, gradient = ctx.saved_tensors
    shape = (-1,) + (1,) * (gradient.dim() - 1)
    by_energy = (cotangent * gradient).to(energy.dtype).reshape(len(energy), -1).sum(dim=1)
    by_gradient = energy.to(gradient.dtype).reshape(shape) * cotangent
    return by_energy, by_gradient


torch.library.register_autograd("mdir::frame_input_vjp", _input_backward,
                                setup_context=_input_setup)


def _setup(ctx, inputs, output):
    call, gradient, theta = inputs
    ctx.call, ctx.jacobian, ctx.serial = call, output[2], output[3]
    ctx.sizes, ctx.needed = [int(t.shape[0]) for t in theta], gradient


def _backward(ctx, energy, others, jacobian, serial, in_positions, in_strain, in_cell):
    # The tunables alone: the gradient in the positions and the cell is
    # that of the nodes of their own, which the engine runs only where the
    # backward takes it.
    _refuse_forward("a cotangent of the outputs", energy, others, jacobian, in_positions,
                    in_strain, in_cell)
    product = _frame_vjp(ctx.call, sum(ctx.sizes), ctx.serial, energy, others, ctx.jacobian,
                         [in_positions, in_strain, in_cell], ctx.needed)
    return None, None, list(torch.split(product, ctx.sizes))


@torch.library.custom_op("mdir::frame_hvp", mutates_args=())
def _frame_hvp(call: int, serial: Tensor, gradient: Tensor) -> Tensor:
    # H_n v_n, the derivative in the positions of the product of force
    # matching at a zero cotangent of the energy (D383), by a pass of the
    # product program that gives it; a term without a rule is refused by
    # name there. Where the same backward takes the gradient in the
    # tunables too, their node forms it by a pass of its own: the two do
    # not depend on the order in which the engine runs them.
    record = _RECORDS.get(call)
    evaluator = record.evaluator() if record else None
    if evaluator is None or record.serial != int(serial) or record.source is None:
        raise _core.SimulationError(
            "the backward needs the frames again (the Jacobian was not kept, jacobian_bytes, or "
            "the gradient in the positions has a cotangent), and the evaluator has evaluated "
            "other frames since: the backward cannot read the frames again")
    v = np.ascontiguousarray(gradient.detach().to(torch.float64).cpu().numpy())
    _, dx = evaluator._product_pass(record.source, record.version,
                                        np.zeros(len(v)), v, True)
    return torch.from_numpy(np.ascontiguousarray(dx)).to(gradient.device)


@torch.library.register_fake("mdir::frame_hvp")
def _frame_hvp_fake(call, serial, gradient):
    return gradient.new_empty(gradient.shape, dtype=torch.float64)


torch.library.register_autograd("mdir::frame_hvp", _second("the positions, of its Hessian"))


def _refuse_outputs(call, others, strain, cell):
    """The refusals of `_frame_vjp` for the cotangents on the outputs whose
    derivative is not implemented, in any input."""
    _frame_vjp(call, 0, torch.zeros((), dtype=torch.int64), others.new_zeros((others.shape[0],)),
               others, others.new_zeros((others.shape[0], 0)),
               [others.new_zeros((0,)), strain, cell], False)


@torch.library.custom_op("mdir::frame_positions", mutates_args=())
def _frame_positions(call: int, serial: Tensor, positions: Tensor, others: Tensor,
                     gradient: Tensor, strain: Tensor, cell: Tensor) -> Tuple[
        Tensor, Tensor, Tensor, Tensor]:
    # Zeros, added to the outputs of the energies: the node through which
    # a gradient of the outputs reaches the positions. `others` and the
    # rest give the shapes.
    return (torch.zeros_like(others), torch.zeros_like(gradient), torch.zeros_like(strain),
            torch.zeros_like(cell))


@torch.library.register_fake("mdir::frame_positions")
def _frame_positions_fake(call, serial, positions, others, gradient, strain, cell):
    return (torch.empty_like(others), torch.empty_like(gradient), torch.empty_like(strain),
            torch.empty_like(cell))


def _positions_setup(ctx, inputs, output):
    call, serial, positions = inputs[:3]
    ctx.call, ctx.serial = call, serial
    ctx.given = (tuple(positions.shape), positions.dtype, positions.device)


def _shaped(given, values):
    # The gradient in the shape, the type, and the device of its input:
    # one frame given without its axis, or one cell for every frame.
    shape, dtype, device = given
    if values.dim() > len(shape):
        values = values.sum(dim=0)
    return values.reshape(shape).to(device=device, dtype=dtype)


def _positions_backward(ctx, others, gradient, strain, cell):
    _refuse_forward("a cotangent of the outputs", others, gradient, strain, cell)
    _refuse_outputs(ctx.call, others, strain, cell)
    if not _nonzero(gradient):
        # The first backward of a force (a zero cotangent on the gradient).
        return None, None, None, None, None, None, None
    return (None, None, _shaped(ctx.given, _frame_hvp(ctx.call, ctx.serial, gradient)),
            None, None, None, None)


torch.library.register_autograd("mdir::frame_positions", _positions_backward,
                                setup_context=_positions_setup)


@torch.library.custom_op("mdir::frame_cell", mutates_args=())
def _frame_cell(call: int, cells: Tensor, tilts: Tensor, others: Tensor, gradient: Tensor, strain: Tensor,
                cell: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    # The same for the cell, whose derivatives of the gradients are not
    # implemented: a backward that reaches the cell with a cotangent on
    # them raises.
    return (torch.zeros_like(others), torch.zeros_like(gradient), torch.zeros_like(strain),
            torch.zeros_like(cell))


@torch.library.register_fake("mdir::frame_cell")
def _frame_cell_fake(call, cells, tilts, others, gradient, strain, cell):
    return (torch.empty_like(others), torch.empty_like(gradient), torch.empty_like(strain),
            torch.empty_like(cell))


def _cell_setup(ctx, inputs, output):
    ctx.call = inputs[0]


def _cell_backward(ctx, others, gradient, strain, cell):
    _refuse_forward("a cotangent of the outputs", others, gradient, strain, cell)
    _refuse_outputs(ctx.call, others, strain, cell)
    if _nonzero(gradient):
        raise _core.UnsupportedError(
            "a second derivative of the energy of the frames is not implemented: the "
            "derivative of the gradient of the energy in the positions, in the cell, was "
            "asked for. It is not zero. The derivative in the tunables is given "
            "(D382): take the gradient in the tunables alone "
            "(torch.autograd.grad(loss, theta)), or detach the cell")
    return None, None, None, None, None, None, None


torch.library.register_autograd("mdir::frame_cell", _cell_backward, setup_context=_cell_setup)


@torch.library.custom_op("mdir::frame_energy_in", mutates_args=())
def _frame_energy_in(positions: Tensor, cells: Tensor, tilts: Tensor, energy: Tensor,
                     gradient: Tensor, cell: Tensor) -> Tensor:
    # Zeros, added to the energy: its backward gives the gradient of the
    # energy in the positions and the cell, the cotangent times the
    # gradients `gradient` and `cell`, which are outputs with a graph of
    # their own, so that a backward of this backward reaches the energies
    # (in the tunables) and the positions (the Hessian) through them.
    return torch.zeros_like(energy)


@torch.library.register_fake("mdir::frame_energy_in")
def _frame_energy_in_fake(positions, cells, tilts, energy, gradient, cell):
    return torch.empty_like(energy)


def _energy_in_setup(ctx, inputs, output):
    positions, cells, tilts, energy, gradient, cell = inputs
    ctx.inputs = [(tuple(t.shape), t.dtype, t.device) if t.numel() else None
                  for t in (positions, cells, tilts)]
    ctx.save_for_backward(gradient, cell)


def _energy_in_backward(ctx, energy):
    _refuse_forward("the cotangent of the energy", energy)
    gradient, cell = ctx.saved_tensors
    positions, cells, tilts = ctx.inputs
    results = [None, None, None]
    if positions is not None:
        results[0] = _shaped(positions, _frame_input_vjp(energy, gradient))
    if cells is not None or tilts is not None:
        by_cell = _frame_input_vjp(energy, cell)
        if cells is not None:
            results[1] = _shaped(cells, by_cell[:, :3])
        if tilts is not None:
            results[2] = _shaped(tilts, by_cell[:, 3:])
    return (*results, None, None, None)


torch.library.register_autograd("mdir::frame_energy_in", _energy_in_backward,
                                setup_context=_energy_in_setup)


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
        _refuse_forward(f"theta['{name}']", tensor)
    if len(devices) != 1:
        raise _core.InputError("mdir.torch.evaluate: the tensors of theta are on one device")
    for given, name in ((positions, "the positions"), (cells, "the cells"), (tilts, "the tilts")):
        _refuse_forward(name, given)
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
def _inputs(number, serial, inputs, energy, others, in_positions, in_strain, in_cell):
    """The positions and the cell that take a gradient, as nodes of their
    own beside that of the tunables: the outputs plus zeros from them. They
    are outside any compiled graph, whose backward would be one node with
    every input, so that the engine runs each only on a path to its input."""
    positions, cells, tilts = inputs
    if not (positions.numel() or cells.numel() or tilts.numel()):
        return energy, others, in_positions, in_strain
    terms = [(others, in_positions, in_strain, in_cell)]
    if positions.numel():
        terms.append(_frame_positions(number, serial, positions, others.detach(),
                                      in_positions.detach(), in_strain.detach(),
                                      in_cell.detach()))
    if cells.numel() or tilts.numel():
        terms.append(_frame_cell(number, cells, tilts, others.detach(), in_positions.detach(),
                                 in_strain.detach(), in_cell.detach()))
    others, in_positions, in_strain, in_cell = (sum(parts[1:], parts[0])
                                                for parts in zip(*terms))
    energy = energy + _frame_energy_in(positions, cells, tilts, energy.detach(), in_positions,
                                       in_cell)
    return energy, others, in_positions, in_strain


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
    positions given.

    The gradient in the positions, `gradients["positions"]`, and the
    gradient of `energy` in positions that require a gradient
    (`torch.autograd.grad(..., create_graph=True)`), are differentiable in
    `theta`: a loss on the forces trains the tunables, by the product of
    force matching (D382, `FrameEnergies.vjp(positions=)`).
    Their derivative in the positions is that of the product,
    g dU/dx + H v (D383, D385), where the backward takes a
    gradient in the positions, the extra points of Amber included
    (D387); a term without a rule for it (the
    reciprocal sum of particle mesh Ewald) raises
    UnsupportedError, naming it, there alone. Their derivative in the cell,
    and every other second derivative, are refused (`UnsupportedError`)
    where a backward needs them. The positions and the cell are nodes of
    the graph of their own beside that of `theta`, so that a backward that
    takes the gradient in `theta` alone does not reach them.

    Forward mode is not implemented: a tangent of `torch.func.jvp`,
    `jacfwd`, or `torch.autograd.forward_ad` on an argument, or on a
    cotangent of a backward, raises UnsupportedError, naming forward mode
    and the tensor; it is not zero (D[torch-forward-refusal]).
    """
    number, gradient, inputs = _begin(evaluator, theta, positions, cells, tilts, gradients)
    energy, others, _, serial, in_positions, in_strain, in_cell = _frame_energies(
        number, gradient, list(theta.values()))
    energy, others, in_positions, in_strain = _inputs(
        number, serial, inputs, energy, others, in_positions, in_strain, in_cell)
    return _finish(number, list(theta), energy, others, in_positions, in_strain)
