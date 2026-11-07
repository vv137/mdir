"""The derivative in the positions of the gradient in the positions through
mdir.torch (D385, #771 part 4): every form of the backward
of a loss on the forces against FrameEnergies.vjp (D381, D383), and the
passes of the product program that each form makes.

The positions are a node of the graph of their own beside that of the
tunables, so a backward that takes the gradient in theta alone makes the
pass of the product in theta alone, and a term without a rule for the
derivative in the positions is refused only where a backward takes it.

The loss is L = sum (-G - f)^2 + w sum U, with G = dU/dx taken by
torch.autograd.grad(..., create_graph=True). Its cotangent on G is
Gbar = 2 (G + f), and on U it is w, so that dL/dtheta and dL/dx are those of
FrameEnergies.vjp(w, positions=Gbar, gradients=("tunables", "positions")).

Usage: python_frame_hvp_torch.py ROOT SCENARIO [TARGET PRECISION [WORK]]

Scenarios:
  forms  the torsion fixtures of D372, the pair fluid of D383, and eight
         OPC waters with the linear sites of GROMACS (D384): every
         form given, against vjp, and under torch.compile
  pme    the dipeptide with PME: the gradient in theta alone given, against
         vjp; the forms that take the positions refused, naming the
         reciprocal sum
  amber  OPC from a prmtop of Amber, cut by ParmEd in WORK beforehand
         (`python_frame_sites.py ROOT cut`, D384): the
         gradient in theta alone given, against vjp; the forms that take the
         positions refused, naming the extra points
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

import mdir
import mdir.torch

sys.path.insert(0, str(Path(__file__).parent))
import python_frame_product as base  # noqa: E402  (reads sys.argv as this script does)
import python_frame_hvp as hvp  # noqa: E402
import python_frame_sites as sites  # noqa: E402

scenario = sys.argv[2]
mode = f"{base.mode} torch"
FORMS = ("grad(L, theta)", "grad(L, x)", "grad(L, [theta, x])", "L.backward()",
         "L.backward(inputs=theta)")
# What each form takes, and the passes of the product program it makes:
# that of theta alone, and that which gives the derivative in the positions.
TAKES = {"grad(L, theta)": (True, False), "grad(L, x)": (False, True),
         "grad(L, [theta, x])": (True, True), "L.backward()": (True, True),
         "L.backward(inputs=theta)": (True, False)}


def counted(evaluator):
    """The passes of the product program, recorded by the evaluator's
    method that makes them."""
    passes = []
    original = evaluator._product_pass

    def recorded(source, version, energy, position, positions=False):
        passes.append("positions" if positions else "theta")
        return original(source, version, energy, position, positions)
    evaluator._product_pass = recorded
    return passes


def theta_of(evaluator):
    return {name: torch.tensor(np.array(evaluator.tunables[name]), dtype=torch.float64,
                               requires_grad=True) for name in evaluator._slices}


def loss_of(evaluator, theta, x, f, weight, energy=None):
    positions = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    if energy is None:
        out_energy = mdir.torch.evaluate(evaluator, theta, positions).energy
    else:
        out_energy = energy(theta, positions)
    g, = torch.autograd.grad(out_energy.sum(), positions, create_graph=True)
    loss = ((-g.to(torch.float64) - f) ** 2).sum() + weight * out_energy.sum()
    return loss, positions, g.detach().to(torch.float64).numpy()


def backward(form, loss, theta, positions):
    values = list(theta.values())
    for t in values:
        t.grad = None
    if form == "grad(L, theta)":
        return [t.numpy() for t in torch.autograd.grad(loss, values)], None
    if form == "grad(L, x)":
        return None, torch.autograd.grad(loss, positions)[0].numpy()
    if form == "grad(L, [theta, x])":
        *by_theta, by_x = torch.autograd.grad(loss, values + [positions])
        return [t.numpy() for t in by_theta], by_x.numpy()
    if form == "L.backward()":
        loss.backward()
        return [t.grad.numpy() for t in values], positions.grad.numpy()
    loss.backward(inputs=values)
    assert positions.grad is None
    return [t.grad.numpy() for t in values], None


def reference(evaluator, x, gradient, f, weight, positions):
    out = evaluator.evaluate(x)
    cotangent = 2.0 * (gradient + f.numpy())
    names = ("tunables", "positions") if positions else ("tunables",)
    result = out.vjp(np.full(len(x), weight), positions=cotangent, gradients=names)
    return [np.asarray(result[n]) for n in evaluator._slices], result.positions


def check_forms(label, evaluator, x, tolerance):
    """Every form against vjp. The product in theta is that of one pass of
    the product program at the same cotangents: equal to the bit. The
    gradient in the positions adds w G of the gradient output to the
    H Gbar of the pass, where vjp forms w dU/dx in the pass: equal within
    `tolerance` (the rounding of the gradient output in mixed precision)."""
    f = torch.from_numpy(np.random.default_rng(771).normal(size=x.shape))
    worst = 0.0
    for weight in (0.0, 0.7):
        for form in FORMS:
            theta = theta_of(evaluator)
            loss, positions, gradient = loss_of(evaluator, theta, x, f, weight)
            passes = counted(evaluator)
            by_theta, by_x = backward(form, loss, theta, positions)
            del evaluator._product_pass
            in_theta, in_x = TAKES[form]
            expected = (["positions"] if in_x else []) + (["theta"] if in_theta else [])
            assert sorted(passes) == sorted(expected), (form, passes)
            want_theta, want_x = reference(evaluator, x, gradient, f, weight, in_x)
            if in_theta:
                for got, want in zip(by_theta, want_theta):
                    assert np.array_equal(got, want), (form, weight, base.relative(got, want))
            if in_x:
                error = base.relative(by_x, np.asarray(want_x))
                worst = max(worst, error)
                assert error <= tolerance, (form, weight, error)
    print(f"{mode}: {label}: every form, with and without a term of the energy, against "
          f"FrameEnergies.vjp: theta to the bit, x {worst:.1e} of the largest entry "
          f"(tolerance {tolerance:.0e}); one pass of each product that the form takes")


def check_compiled(label, evaluator, x):
    """The forward under torch.compile: the operation of the tunables in the
    compiled graph, the nodes of the positions outside it, the same passes
    and the same gradients as eager, to the bit."""
    f = torch.from_numpy(np.random.default_rng(772).normal(size=x.shape))
    compiled = torch.compile(lambda theta, positions: mdir.torch.evaluate(
        evaluator, theta, positions).energy, backend="aot_eager")
    for form in ("grad(L, theta)", "grad(L, x)"):
        results = []
        for energy in (compiled, None):
            theta = theta_of(evaluator)
            loss, positions, _ = loss_of(evaluator, theta, x, f, 0.0, energy)
            passes = counted(evaluator)
            results.append((backward(form, loss, theta, positions), passes))
            del evaluator._product_pass
        ((theta_a, x_a), pa), ((theta_b, x_b), pb) = results
        assert pa == pb == (["theta"] if form == "grad(L, theta)" else ["positions"]), (pa, pb)
        if theta_a is not None:
            assert all(np.array_equal(u, v) for u, v in zip(theta_a, theta_b)), form
        if x_a is not None:
            assert np.array_equal(x_a, x_b), form
    print(f"{mode}: {label}: under torch.compile (aot_eager), grad(L, theta) and grad(L, x) "
          "make the passes of eager and give its gradients, to the bit")


def check_refused(label, evaluator, x, text):
    """The gradient in theta alone given, to the bit of vjp; every form that
    takes the positions refused, naming the term."""
    f = torch.from_numpy(np.random.default_rng(773).normal(size=x.shape))
    for weight in (0.0, 0.7):
        for form in FORMS:
            theta = theta_of(evaluator)
            loss, positions, gradient = loss_of(evaluator, theta, x, f, weight)
            if TAKES[form][1]:
                try:
                    backward(form, loss, theta, positions)
                except mdir.UnsupportedError as error:
                    assert text in str(error), str(error)
                    continue
                raise AssertionError(f"{form}: expected UnsupportedError containing {text!r}")
            by_theta, _ = backward(form, loss, theta, positions)
            want, _ = reference(evaluator, x, gradient, f, weight, False)
            for got, value in zip(by_theta, want):
                assert np.array_equal(got, value), (form, base.relative(got, value))
    print(f"{mode}: {label}: grad(L, theta) and L.backward(inputs=theta), with positions that "
          f"require a gradient, against FrameEnergies.vjp, to the bit; grad(L, x), "
          f"grad(L, [theta, x]), and L.backward() refused, naming {text}")


def quiet(build):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return build()


def run_forms():
    tolerance = 1e-14 if base.double else 1e-9
    system, state = base.torsion_fixture()
    system.tunables = [mdir.Tunable("k", "dihedral_k")]
    system.tunable_gradient = True
    evaluator = mdir.FrameEvaluator(base.compile_(system, state))
    x = base.frames(state, count=2, spread=0.02)
    check_forms("dihedral_k on the torsion fixtures, 2 frames", evaluator, x, tolerance)
    check_compiled("dihedral_k on the torsion fixtures", evaluator, x)
    system, state, _ = hvp.pair_fixture()
    evaluator = mdir.FrameEvaluator(base.compile_(system, state))
    x = base.frames(state, count=2, spread=0.01)
    check_forms("the pair fluid (LJ with NBFIX and Coulomb, shifted at the cutoff), 2 frames",
                evaluator, x, tolerance)
    system, state, _, _ = quiet(lambda: sites.fixture(8, 2.0))
    evaluator = quiet(lambda: mdir.FrameEvaluator(base.compile_(system, state)))
    x = base.frames(state, count=2, spread=0.01)
    check_forms("eight OPC waters with the linear sites of GROMACS, 2 frames", evaluator, x,
                tolerance)


def run_pme():
    loaded = mdir.load_amber(base.root + "/dipeptide.prmtop", base.root + "/dipeptide.inpcrd")
    system, state = loaded.make_system(), loaded.make_state()
    system.cutoff, system.pairlist_distance = 0.8, 0.9
    system.electrostatics = mdir.Electrostatics.PME
    system.dispersion = mdir.DispersionCorrection.None_
    system.tunables = [mdir.Tunable("sigma", "sigma_pair")]
    system.tunable_gradient = True
    evaluator = mdir.FrameEvaluator(base.compile_(system, state))
    x = base.frames(state, count=2, spread=0.004)
    check_refused("the dipeptide with PME, 2 frames", evaluator, x, "reciprocal sum")


def run_amber():
    # The prmtop that `python_frame_sites.py ROOT cut` wrote into WORK with
    # ParmEd, which the interpreter of PyTorch need not have.
    prmtop, inpcrd = (str(base.work / f"opc-40-protein.{e}") for e in ("prmtop", "inpcrd"))
    system, state, _ = sites.amber_fixture(mdir.load_amber(prmtop, inpcrd))
    evaluator = quiet(lambda: mdir.FrameEvaluator(base.compile_(system, state)))
    x = base.frames(state, count=2, spread=0.004)
    check_refused("Amber OPC (ACE-ALA-NME and 40 waters), 2 frames", evaluator, x,
                  "virtual sites placed by the directions of their bonds")


{"forms": run_forms, "pme": run_pme, "amber": run_amber}[scenario]()
