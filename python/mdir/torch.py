"""The frame evaluator as an operation of PyTorch (docs/python-frames.md).

`evaluate(evaluator, theta, positions, cells=None)` returns the energies of
the frames as a tensor that autograd differentiates in the tunables `theta`
through the vector-Jacobian product of the evaluator. The operation is a
`torch.library.custom_op` with `register_autograd`, so that it is opaque to
`torch.compile`. Importing this module imports torch; `import mdir` does
not.
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
        self.others = ["virial"] + list(evaluator._columns)
        self.depends = dict(evaluator._depends)
        self.names, self.frames, self.count, self.serial = [], None, None, 0
        self.out = None
        # What a backward by a second pass reads: the frames, and the
        # version of the values that they were evaluated at.
        self.source, self.version = None, None


@torch.library.custom_op("mdir::frame_energies", mutates_args=())
def _frame_energies(call: int, gradient: bool,
                    theta: List[Tensor]) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    record = _RECORDS[call]
    evaluator = record.evaluator()
    changes = {}
    for name, tensor in zip(record.names, theta):
        value = np.ascontiguousarray(tensor.detach().cpu().numpy())
        if value.tobytes() != np.ascontiguousarray(evaluator.tunables[name]).tobytes():
            changes[name] = value
    if changes:
        evaluator.tunables.update(changes)
    out = record.out = evaluator.evaluate(*record.frames, gradient=gradient)
    record.source = record.version = None
    if gradient and out.jacobian is not None:
        rows = np.concatenate([out.jacobian[name] for name in record.names], axis=1)
    else:
        rows = np.zeros((out.count, 0))
        if gradient:
            record.source, record.version = out._source, out.version
    device = theta[0].device
    others = np.stack([out.virial] + [out.observables[c] for c in evaluator._columns], axis=1)
    return (torch.from_numpy(out.energy.copy()).to(device), torch.from_numpy(others).to(device),
            torch.from_numpy(rows).to(device), torch.tensor(record.serial, dtype=torch.int64))


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
    return (like.new_empty((count,)), like.new_empty((count, len(record.others))),
            like.new_empty((count, entries if kept else 0)),
            torch.empty((), dtype=torch.int64))


@torch.library.custom_op("mdir::frame_vjp", mutates_args=())
def _frame_vjp(call: int, entries: int, serial: Tensor, energy: Tensor, others: Tensor,
               jacobian: Tensor) -> Tensor:
    # No derivative is registered for this operation: a second
    # differentiation, which would need the derivative of the Jacobian in
    # the tunables, is an error of autograd, not a zero.
    record = _RECORDS.get(call)
    asked = torch.nonzero(others.abs().sum(dim=0)).flatten().tolist()
    if asked:
        names = record.others if record else [f"column {k}" for k in range(others.shape[1])]
        depends = record.depends if record else {}
        listed = ", ".join(f"'{names[k]}' (which may read "
                           f"{', '.join(sorted(depends.get(names[k], [])))})" for k in asked)
        raise _core.UnsupportedError(
            f"the derivative in the tunables of the output {listed} at fixed positions is not "
            "implemented; it is not zero. Detach the output to leave that term out of a "
            "gradient")
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
def _frame_vjp_fake(call, entries, serial, energy, others, jacobian):
    return energy.new_empty((entries,))


def _setup(ctx, inputs, output):
    call, gradient, theta = inputs
    ctx.call, ctx.jacobian, ctx.serial = call, output[2], output[3]
    ctx.sizes = [int(t.shape[0]) for t in theta]


def _backward(ctx, energy, others, jacobian, serial):
    product = _frame_vjp(ctx.call, sum(ctx.sizes), ctx.serial, energy, others, ctx.jacobian)
    return None, None, list(torch.split(product, ctx.sizes))


torch.library.register_autograd("mdir::frame_energies", _backward, setup_context=_setup)


class FrameOutputs:
    """What `evaluate` returns: `energy`, `virial`, and `volume`, tensors
    (K,) float64 on the device of `theta`, and `observables`, a dict of the
    names of the observed columns to such tensors. `energy` is
    differentiable in `theta`. An output that reads no tunable of `theta`
    (`depends`) is a constant, without a graph; one that may read one
    raises `UnsupportedError` from a backward that reaches it."""

    def __init__(self, out, energy, virial, volume, observables):
        self.energy, self.virial, self.volume, self.observables = (
            energy, virial, volume, observables)
        self.depends, self.version, self.count, self.zero = (
            out.depends, out.version, out.count, out.zero)

    def __repr__(self):
        return f"FrameOutputs(count={self.count}, version={self.version})"


@torch.compiler.disable
def _begin(evaluator, theta, positions, cells, tilts):
    """Checks the arguments and records the call; returns the number of the
    evaluator and whether the derivative is needed."""
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
    for given, name in ((positions, "positions"), (cells, "cells"), (tilts, "tilts")):
        if getattr(given, "requires_grad", False):
            raise _core.UnsupportedError(
                f"{name} require a gradient: the gradients of the energy in the positions and "
                "the cell are not returned by the frame evaluator yet; detach them")
    number = getattr(evaluator, "_torch_number", None)
    if number is None:
        number = evaluator._torch_number = next(_NUMBERS)
        _RECORDS[number] = _Record(evaluator)
        weakref.finalize(evaluator, _RECORDS.pop, number, None)
    record = _RECORDS[number]
    record.names, record.frames = list(theta), (positions, cells, tilts)
    record.serial, record.out = next(_SERIALS), None
    record.count = None
    if isinstance(positions, (np.ndarray, Tensor)):
        record.count = positions.shape[0] if positions.ndim == 3 else 1
    elif hasattr(positions, "__len__") and not hasattr(positions, "positions"):
        record.count = len(positions)
    gradient = torch.is_grad_enabled() and any(t.requires_grad for t in theta.values())
    return number, gradient


@torch.compiler.disable
def _finish(number, names, energy, others):
    """The outputs of the call, and the end of its record: the arguments
    and the arrays are not kept beyond it."""
    record = _RECORDS[number]
    out, record.out, record.frames = record.out, None, None
    named = set(names)
    constant = others.detach()

    def output(name, column):
        return others[:, column] if out.depends[name] & named else constant[:, column]

    observables = {name: output(name, k + 1) for k, name in enumerate(record.others[1:])}
    volume = torch.from_numpy(out.volume.copy()).to(energy.device)
    return FrameOutputs(out, energy, output("virial", 0), volume, observables)


def evaluate(evaluator, theta, positions, cells=None, tilts=None):
    """Evaluates the frames at the values `theta` of the tunables.

    `theta` is a dict of names of tunables of the evaluator to float64
    tensors of the shape of their values, on one device; a tunable that is
    not named keeps the value of the evaluator. `positions`, `cells`, and
    `tilts` are those of `FrameEvaluator.evaluate`. The values of the
    evaluator are updated where they differ, without a compilation (D213).
    """
    number, gradient = _begin(evaluator, theta, positions, cells, tilts)
    energy, others, _, _ = _frame_energies(number, gradient, list(theta.values()))
    return _finish(number, list(theta), energy, others)
