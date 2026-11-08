"""Forward-mode AD through mdir.torch is refused (D[torch-forward-refusal],
#777): every tangent that reaches an operation of mdir.torch raises
UnsupportedError, naming forward mode, and never gives a zero.

Usage: python_frames_torch_forward.py ROOT TARGET PRECISION WORK

On the torsion fixtures of python_frame_product.py with dihedral_k tunable,
two frames in a cubic cell: torch.func.jvp, jacfwd, and the dual numbers
of torch.autograd.forward_ad, with the tangent on theta, on the positions,
on the cells, and on the tilts; of the energy and of the output
gradients["positions"]; positions and cells given as a list or a tuple of
frames; vmap of jvp; forward over reverse (a dual input
to a backward with create_graph=True, and a dual cotangent of a backward,
through the energy and through the gradient in the positions); under
torch.compile (aot_eager). torch.func.hessian stops at its reverse level,
in a RuntimeError of PyTorch (an autograd.Function without the support of
functorch), before any tangent: it is no zero either. A jvp whose tangent
does not reach the operations gives its value. Then the same evaluator
gives the same gradients of reverse mode as before the refusals, to the
bit.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.autograd.forward_ad as fwAD

import mdir
import mdir.torch

root, target, precision, work = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
sys.argv = [sys.argv[0], root, "dihedral", target, precision, work]
sys.path.insert(0, str(Path(__file__).parent))
import python_frame_product as base  # noqa: E402  (reads sys.argv)

mode = f"{target} {precision}"
device = "cuda" if target == "GPU" else "cpu"
TEXT = "is not implemented; reverse mode (torch.autograd.grad / backward) is"

system, state = base.torsion_fixture()
system.tunables = [mdir.Tunable("k", "dihedral_k")]
system.tunable_gradient = True
evaluator = mdir.FrameEvaluator(base.compile_(system, state))
x = torch.from_numpy(base.frames(state, count=2, spread=0.02)).to(device)
cells = torch.tensor([3.0, 3.0, 3.0], dtype=torch.float64, device=device)
tilts = torch.zeros(3, dtype=torch.float64, device=device)
k0 = torch.tensor(np.array(evaluator.tunables["k"]), dtype=torch.float64, device=device)


def energy(k, positions=x, c=None, t=None):
    return mdir.torch.evaluate(evaluator, {"k": k}, positions, c, t).energy


def gradient(k, positions=x):
    out = mdir.torch.evaluate(evaluator, {"k": k}, positions, gradients=("positions",))
    return out.gradients["positions"]


def reverse():
    """The gradients of reverse mode: of the energy in theta, the positions,
    and the cell; and of a loss on the forces in theta (D382)."""
    k = k0.clone().requires_grad_()
    positions, c = x.clone().requires_grad_(), cells.clone().requires_grad_()
    weights = torch.tensor([0.7, -1.3], dtype=torch.float64, device=device)
    by = torch.autograd.grad((weights * energy(k, positions, c)).sum(), (k, positions, c))
    positions = x.clone().requires_grad_()
    g, = torch.autograd.grad(energy(k, positions).sum(), positions, create_graph=True)
    force, = torch.autograd.grad((g.to(torch.float64) ** 2).sum(), k)
    return [t.detach().cpu() for t in (*by, force)]


def ones(t):
    return torch.ones_like(t)


def dual(fn, primal):
    with fwAD.dual_level():
        return fwAD.unpack_dual(fn(fwAD.make_dual(primal, ones(primal)))).tangent


def over_reverse():
    # A dual input to a backward with create_graph=True.
    with fwAD.dual_level():
        k = fwAD.make_dual(k0.clone().requires_grad_(), ones(k0))
        g, = torch.autograd.grad(energy(k).sum(), k, create_graph=True)
        return fwAD.unpack_dual(g).tangent


def dual_cotangent_energy():
    # A dual cotangent of a backward through the energy.
    k = k0.clone().requires_grad_()
    e = energy(k)
    with fwAD.dual_level():
        g, = torch.autograd.grad(e, k, fwAD.make_dual(ones(e), ones(e)))
        return fwAD.unpack_dual(g).tangent


def dual_cotangent_force():
    # The same through the gradient in the positions (create_graph=True).
    k, positions = k0.clone().requires_grad_(), x.clone().requires_grad_()
    g, = torch.autograd.grad(energy(k, positions).sum(), positions, create_graph=True)
    with fwAD.dual_level():
        by, = torch.autograd.grad(g, k, fwAD.make_dual(ones(g), ones(g)))
        return fwAD.unpack_dual(by).tangent


def compiled(fn):
    return torch.compile(fn, backend="aot_eager")


FORMS = [
    ("torch.func.jvp in theta", "theta['k']",
     lambda: torch.func.jvp(energy, (k0,), (ones(k0),))),
    ("torch.func.jvp in the positions", "the positions",
     lambda: torch.func.jvp(lambda p: energy(k0, p), (x,), (ones(x),))),
    ("torch.func.jvp in the cells", "the cells",
     lambda: torch.func.jvp(lambda c: energy(k0, x, c), (cells,), (ones(cells),))),
    ("torch.func.jvp in the tilts", "the tilts",
     lambda: torch.func.jvp(lambda t: energy(k0, x, cells, t), (tilts,), (ones(tilts),))),
    ("torch.func.jvp of gradients['positions'] in theta", "theta['k']",
     lambda: torch.func.jvp(gradient, (k0,), (ones(k0),))),
    ("torch.func.jacfwd in theta", "theta['k']", lambda: torch.func.jacfwd(energy)(k0)),
    ("torch.func.jacfwd in the positions", "the positions",
     lambda: torch.func.jacfwd(lambda p: energy(k0, p))(x)),
    ("torch.func.jacfwd of gradients['positions'] in theta", "theta['k']",
     lambda: torch.func.jacfwd(gradient)(k0)),
    ("torch.func.vmap of torch.func.jvp in theta", "theta['k']",
     lambda: torch.func.vmap(lambda v: torch.func.jvp(energy, (k0,), (v,))[1])(
         torch.eye(len(k0), dtype=torch.float64, device=device))),
    ("forward_ad in theta", "theta['k']", lambda: dual(energy, k0)),
    ("forward_ad in the positions", "the positions", lambda: dual(lambda p: energy(k0, p), x)),
    ("forward_ad in the cells", "the cells", lambda: dual(lambda c: energy(k0, x, c), cells)),
    ("forward_ad in the tilts", "the tilts",
     lambda: dual(lambda t: energy(k0, x, cells, t), tilts)),
    ("forward_ad of gradients['positions'] in theta", "theta['k']", lambda: dual(gradient, k0)),
    ("torch.func.jvp in the positions as a list of frames", "the positions",
     lambda: torch.func.jvp(lambda p: energy(k0, [p[0], p[1]]), (x,), (ones(x),))),
    ("forward_ad in the positions as a list of frames", "the positions",
     lambda: dual(lambda p: energy(k0, [p[0], p[1]]), x)),
    ("forward_ad in the cells as a tuple of frames", "the cells",
     lambda: dual(lambda c: energy(k0, x, (c, c)), cells)),
    ("forward_ad over torch.autograd.grad with create_graph=True", "theta['k']", over_reverse),
    ("a dual cotangent of a backward through the energy", "a cotangent of the outputs",
     dual_cotangent_energy),
    ("a dual cotangent of a backward through the gradient in the positions",
     "the cotangent of a gradient of the energy", dual_cotangent_force),
    ("torch.func.jvp under torch.compile (aot_eager)", "theta['k']",
     lambda: torch.func.jvp(compiled(lambda k: energy(k)), (k0,), (ones(k0),))),
    ("forward_ad under torch.compile (aot_eager)", "theta['k']",
     lambda: dual(compiled(lambda k: energy(k)), k0)),
]


def refused(name, carrier, call):
    try:
        result = call()
    except mdir.UnsupportedError as raised:
        text = str(raised)
        assert TEXT in text and "forward mode" in text and carrier in text, (name, text)
        return
    raise AssertionError(f"{name}: no UnsupportedError, gave {result!r}")


before = reverse()
for name, carrier, call in FORMS:
    refused(name, carrier, call)
print(f"{mode}: refused, UnsupportedError naming forward mode and the tensor of the tangent, "
      f"never a zero: {len(FORMS)} forms; " + "; ".join(name for name, _, _ in FORMS))

# torch.func.hessian stops at its reverse level (torch.func.grad and
# jacrev alone do the same): PyTorch's RuntimeError, before any tangent.
for name, call in (("in theta", lambda: torch.func.hessian(lambda k: energy(k).sum())(k0)),
                   ("in the positions",
                    lambda: torch.func.hessian(lambda p: energy(k0, p).sum())(x))):
    try:
        result = call()
    except mdir.UnsupportedError:
        raise
    except RuntimeError as raised:
        assert "functorch" in str(raised), str(raised)
        continue
    raise AssertionError(f"torch.func.hessian {name}: no error, gave {result!r}")
print(f"{mode}: torch.func.hessian in theta and in the positions: RuntimeError of PyTorch at "
      "its reverse level, never a zero")

# A jvp whose tangent does not reach the operations gives its value.
value = energy(k0).detach()
_, tangent = torch.func.jvp(lambda s: s * energy(k0), (torch.ones((), dtype=torch.float64,
                                                                device=device),),
                            (torch.ones((), dtype=torch.float64, device=device),))
assert torch.equal(tangent, value)
print(f"{mode}: a jvp whose tangent does not reach the operations gives the energy, to the bit")

after = reverse()
same = all(torch.equal(a, b) for a, b in zip(before, after))
print(f"{mode}: after the refusals, the gradients of reverse mode in theta, the positions, the "
      f"cell, and of a loss on the forces in theta, to the bit: {same}")
assert same
print("forward refusal passed")
