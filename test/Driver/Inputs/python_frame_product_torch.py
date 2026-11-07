"""Force matching through mdir.torch (D382, #726 part E):
the gradient of a loss on the forces in the tunables, by the backward of
the gradient of the energy in the positions.

Usage: python_frame_product_torch.py ROOT SCENARIO [TARGET PRECISION [WORK]]

Scenarios:
  dihedral, lj, charges
            the scenarios of python_frame_product.py (D381), with the product
            formed by mdir.torch in place of FrameEnergies.vjp, in its two
            forms, which must agree with each other and with vjp: the
            references of those scenarios (closed forms, NumPy sums with
            their tangents, central differences, mixed against double) then
            check the backward of mdir.torch
  gradcheck torch.autograd.gradcheck, in double, of the gradient in the
            positions as a function of theta, in both forms: its Jacobian in
            theta against finite differences of the forces, on the torsion
            fixtures and on propane and water
  refusals  the second derivatives that stay refused, each where a
            backward needs it, the derivative in the positions with PME
            among them, the gradient in theta alone given there, and that
            the evaluator stays usable (CPU)
"""
import sys
from pathlib import Path

import numpy as np
import torch

import mdir
import mdir.torch

sys.path.insert(0, str(Path(__file__).parent))
import python_frame_product as base  # noqa: E402  (reads sys.argv as this script does)

scenario = sys.argv[2]
base.mode = f"{base.mode} torch"
FORMS = ("output", "create_graph")
VJP = base.product
# gradcheck: the step in the tunables and the tolerances of the comparison
# of the Jacobian of the gradient in the positions with its differences.
EPS, ATOL, RTOL = 1e-5, 1e-5, 1e-5


def torch_product(evaluator, x, g, v, form):
    """dL/dtheta for L = sum_n g_n U_n + sum_n,i v_n,i . G_n,i, by autograd
    through mdir.torch; G is the output gradients["positions"] or the
    gradient of the energy taken with create_graph=True."""
    theta = {name: torch.tensor(np.array(evaluator.tunables[name]), dtype=torch.float64,
                                requires_grad=True) for name in evaluator._slices}
    weights, cotangent = torch.from_numpy(np.array(g)), torch.from_numpy(np.array(v))
    if form == "output":
        out = mdir.torch.evaluate(evaluator, theta, x, gradients=("positions",))
        gradient = out.gradients["positions"].to(torch.float64)
    else:
        positions = torch.tensor(x, dtype=torch.float64, requires_grad=True)
        out = mdir.torch.evaluate(evaluator, theta, positions)
        gradient, = torch.autograd.grad(out.energy.sum(), positions, create_graph=True)
        gradient = gradient.to(torch.float64)
    loss = (weights * out.energy).sum() + (cotangent * gradient).sum()
    grads = torch.autograd.grad(loss, list(theta.values()))
    return {name: value.detach().numpy() for name, value in zip(theta, grads)}


def checked_product(evaluator, x, g, v):
    # Both forms, against each other and against FrameEnergies.vjp, then
    # the first for the references of the scenario.
    reference = base.flat(VJP(evaluator, x, g, v), evaluator._slices)
    forms = [torch_product(evaluator, x, g, v, form) for form in FORMS]
    for form, result in zip(FORMS, forms):
        error = base.relative(base.flat(result, evaluator._slices), reference)
        tolerance = 1e-14 if base.double else 1e-6
        print(f"{base.mode}: the product through mdir.torch ({form}) against "
              f"FrameEnergies.vjp, {error:.1e} of the largest entry (tolerance {tolerance:.0e})")
        assert error <= tolerance, (form, error)
    return forms[0]


def gradcheck():
    torch.manual_seed(0)
    for name, build in (("dihedral_k on the torsion fixtures", torsion),
                        ("sigma and epsilon of two pairs of types on propane and water", propane)):
        evaluator, x = build()
        # gradcheck takes the backward of its first call after the calls
        # of its differences; the backward of mdir.torch is that of the
        # last call of an evaluator, so the calls after the first take an
        # evaluator of their own, of the same program.
        differences = mdir.FrameEvaluator(evaluator.program)
        names = list(evaluator._slices)
        start = [torch.tensor(np.array(evaluator.tunables[n]), dtype=torch.float64,
                              requires_grad=True) for n in names]
        for form in FORMS:
            calls = []

            def gradient(*values):
                theta = dict(zip(names, values))
                taken = differences if calls else evaluator
                calls.append(form)
                if form == "output":
                    out = mdir.torch.evaluate(taken, theta, x, gradients=("positions",))
                    return out.gradients["positions"]
                positions = torch.tensor(x, dtype=torch.float64, requires_grad=True)
                with torch.enable_grad():
                    out = mdir.torch.evaluate(taken, theta, positions)
                    g, = torch.autograd.grad(out.energy.sum(), positions,
                                             create_graph=torch.is_grad_enabled() and
                                             any(v.requires_grad for v in values))
                return g
            # The step of the differences relative to the values, which
            # are of 0.1 to 10 in kJ/mol and nm.
            assert torch.autograd.gradcheck(gradient, start, eps=EPS, atol=ATOL, rtol=RTOL,
                                            fast_mode=False)
            print(f"{base.mode}: gradcheck of the gradient in the positions in theta ({form}): "
                  f"{name}, passed (eps {EPS:.0e}, atol {ATOL:.0e}, rtol {RTOL:.0e})")


def torsion():
    system, state = base.torsion_fixture()
    system.tunables = [mdir.Tunable("k", "dihedral_k")]
    system.tunable_gradient = True
    evaluator = mdir.FrameEvaluator(base.compile_(system, state))
    return evaluator, base.frames(state, count=2, spread=0.02)


def propane():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        def tunables(system):
            # sigma and epsilon of the pairs OW-OW and CT-OW, two entries each.
            top = system.topology
            names, pairs = list(top.type_names), [tuple(p) for p in top.type_pairs]
            ow, ct = names.index("OW"), names.index("CT")
            taken = np.full(len(pairs), -1)
            taken[pairs.index((ow, ow))] = 0
            taken[pairs.index((min(ct, ow), max(ct, ow)))] = 1
            return [mdir.Tunable("sigma", "sigma_pair", map=taken),
                    mdir.Tunable("epsilon", "epsilon_pair", map=taken)]
        system, state = base.propane(tunables)
        evaluator = mdir.FrameEvaluator(base.compile_(system, state))
    return evaluator, base.frames(state, count=1)


def expect(error, call, text):
    try:
        call()
    except error as raised:
        assert text in str(raised), str(raised)
        return
    raise AssertionError(f"expected {error.__name__} containing {text!r}")


def pme():
    loaded = mdir.load_amber(base.root + "/dipeptide.prmtop", base.root + "/dipeptide.inpcrd")
    system, state = loaded.make_system(), loaded.make_state()
    system.cutoff, system.pairlist_distance = 0.8, 0.9
    system.electrostatics = mdir.Electrostatics.PME
    system.dispersion = mdir.DispersionCorrection.None_
    system.tunables = [mdir.Tunable("sigma", "sigma_pair")]
    system.tunable_gradient = True
    evaluator = mdir.FrameEvaluator(base.compile_(system, state))
    return evaluator, base.frames(state, count=2, spread=0.004)


def refusals():
    # The derivative in the positions of the gradient in the positions is
    # given where every term has a rule for it (D383,
    # python-frame-hvp-torch.test); with PME it is refused, naming the
    # reciprocal sum, where a backward takes a gradient in the positions,
    # and there alone (D[torch-public-prune]).
    evaluator, x = pme()
    theta = {"sigma": torch.tensor(np.array(evaluator.tunables["sigma"]), requires_grad=True)}
    f_ref = torch.from_numpy(np.random.default_rng(3).normal(size=x.shape))

    def force_loss(positions):
        out = mdir.torch.evaluate(evaluator, theta, positions)
        g, = torch.autograd.grad(out.energy.sum(), positions, create_graph=True)
        return ((-g - f_ref) ** 2).sum(), g

    positions = torch.tensor(x, requires_grad=True)
    loss, g = force_loss(positions)
    first, = torch.autograd.grad(loss, theta["sigma"], retain_graph=True)
    expect(mdir.UnsupportedError, lambda: torch.autograd.grad(loss, positions, retain_graph=True),
           "reciprocal sum")
    expect(mdir.UnsupportedError, lambda: loss.backward(retain_graph=True), "reciprocal sum")
    # The positions moved by a tensor of their own: the gradient that
    # reaches it is refused too.
    shift = torch.zeros_like(positions, requires_grad=True)
    moved, _ = force_loss(torch.from_numpy(x) + shift)
    expect(mdir.UnsupportedError, lambda: torch.autograd.grad(moved, shift), "reciprocal sum")
    print("refusals: with PME, the derivative in the positions, by torch.autograd.grad, by "
          "backward, and through a tensor that moves the positions, naming the reciprocal sum")
    # The gradient in theta alone, with the positions requiring a gradient:
    # the product of D381, that of FrameEnergies.vjp at the cotangent that
    # the loss puts on G, dL/dG = 2 (G + f).
    cotangent = 2.0 * (g.detach().numpy() + f_ref.numpy())
    reference = evaluator.evaluate(x).vjp(np.zeros(len(x)), positions=cotangent)["sigma"]
    assert np.array_equal(first.numpy(), reference)
    positions = torch.tensor(x, requires_grad=True)
    loss, _ = force_loss(positions)
    loss.backward(inputs=[theta["sigma"]])
    assert torch.equal(theta["sigma"].grad, first) and positions.grad is None
    print("refusals: with PME, the gradient in theta alone, with positions that require a "
          "gradient, by torch.autograd.grad and by backward(inputs=theta), against "
          "FrameEnergies.vjp, to the bit")
    evaluator, x = torsion()
    theta = {"k": torch.tensor(np.array(evaluator.tunables["k"]), requires_grad=True)}
    f_ref = torch.from_numpy(np.random.default_rng(3).normal(size=x.shape))

    def force_loss(positions):
        out = mdir.torch.evaluate(evaluator, theta, positions)
        g, = torch.autograd.grad(out.energy.sum(), positions, create_graph=True)
        return ((-g - f_ref) ** 2).sum()

    first, = torch.autograd.grad(force_loss(torch.tensor(x, requires_grad=True)), theta["k"])
    out = mdir.torch.evaluate(evaluator, theta, x, gradients=("strain",))
    expect(mdir.UnsupportedError,
           lambda: torch.autograd.grad(out.gradients["strain"].sum(), theta["k"]),
           "the gradient of the energy in the strain")
    out = mdir.torch.evaluate(evaluator, theta, x)
    g, = torch.autograd.grad(out.energy.sum(), theta["k"], create_graph=True)
    expect(mdir.UnsupportedError, lambda: torch.autograd.grad(g.sum(), theta["k"]),
           "a second derivative of the energy")
    print("refusals: the derivative of the gradient in the strain, and the second derivative "
          "in theta")
    out = mdir.torch.evaluate(evaluator, theta, x, gradients=("positions",))
    loss = (out.gradients["positions"] ** 2).sum()
    mdir.torch.evaluate(evaluator, theta, x[:1])
    expect(mdir.SimulationError, lambda: torch.autograd.grad(loss, theta["k"]),
           "evaluated other frames since")
    # The evaluator stays usable, and gives what it gave.
    positions = torch.tensor(x, requires_grad=True)
    again, = torch.autograd.grad(force_loss(positions), theta["k"])
    assert torch.equal(again, first)
    print("refusals: a backward after other frames; the evaluator then gives the same "
          "gradient, to the bit")


if scenario in ("dihedral", "lj", "charges"):
    base.product = checked_product
    {"dihedral": base.run_dihedral, "lj": base.run_lj, "charges": base.run_charges}[scenario]()
elif scenario == "gradcheck":
    gradcheck()
else:
    refusals()
