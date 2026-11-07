"""The products of force matching with linear virtual sites
(D[force-matching-sites], #771 part 2): the product in theta of D381 and the
derivative in the positions of D383 for OPC water, whose site M is
x_O + a (x_H1 - x_O) + a (x_H2 - x_O) ([ virtual_sites3 ], function 1, of
GROMACS), against a NumPy model of the waters.

With X the places of all particles, the sites placed, and J = dX/dx, the
gradient that the evaluator returns is G = J^T dU/dX, zero on the sites, and

  Phi = g U + Gbar . G,  dPhi/dtheta,  dPhi/dx = J^T (g dU/dX + H_X J Gbar),

J being constant. NumPy forms U and G of the atoms; the references are
central differences of its Phi in theta and in x.

Usage: python_frame_sites.py ROOT SCENARIO [TARGET PRECISION [WORK]]

Scenarios:
  pair      two OPC waters in a cell far larger than the cutoff
  box       eight OPC waters in a cell of 2.0 nm, through the minimum image;
            in mixed precision, against double
  refusals  the extra points of Amber (placed by the directions of the
            bonds) are refused, by name, in theta and in the positions; the
            cotangent on a site is not read
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import mdir

sys.path.insert(0, str(Path(__file__).parent))
import python_frame_product as base  # noqa: E402  (reads sys.argv as this script does)

scenario = sys.argv[2]
work = base.work
double = base.double
F = 138.935457644
# OPC of amber19sb.ff (opc.itp and ffnonbonded.itp of GROMACS 2026.3).
SIGMA, EPSILON = 0.316655208196, 0.890358601592
CHARGES = {"OW": 0.0, "HW": 0.679142, "MW": -1.358284}
A = 0.14772234
DOH, HOH = 0.08724331, np.radians(103.6)
RC = 0.9


def hvp(evaluator, x, g, v):
    out = evaluator.evaluate(x, gradients=("tunables", "positions"))
    result = out.vjp(g, positions=v, gradients=("tunables", "positions"))
    return out, base.flat(result, evaluator._slices), np.asarray(result.positions)


def richardson(function, x, step):
    """The gradient of `function` (frames (K, N, 3) -> (K,)) by central
    differences, Richardson-extrapolated from the steps `step` and `step`/2."""
    def central(h):
        result = np.zeros_like(x)
        for n in range(x.shape[0]):
            for i in range(x.shape[1]):
                for c in range(3):
                    e = np.zeros_like(x)
                    e[n, i, c] = h
                    result[n, i, c] = (function(x + e)[n] - function(x - e)[n]) / (2.0 * h)
        return result
    return (4.0 * central(0.5 * step) - central(step)) / 3.0


def evaluator_phi(evaluator, g, v):
    def phi(y):
        out = evaluator.evaluate(y, gradients=("positions",))
        return g * out.energy + np.einsum("nij,nij->n", v, out.gradients["positions"])
    return phi


def fixture(waters, edge):
    """OPC waters, flexible and without intramolecular terms, at random
    orientations about a lattice: the system, its state, and the names."""
    work.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(772)
    side = int(np.ceil(waters ** (1.0 / 3.0)))
    centers = np.array([(i, j, k) for i in range(side) for j in range(side)
                        for k in range(side)], dtype=float)[:waters]
    centers = (centers + 0.5) * (edge / side) if waters > 2 else \
        np.array([[1.0, 1.0, 1.0], [1.29, 1.08, 1.05]])
    x = []
    for center in centers:
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        h1 = DOH * np.array([np.sin(HOH / 2), np.cos(HOH / 2), 0.0])
        h2 = DOH * np.array([-np.sin(HOH / 2), np.cos(HOH / 2), 0.0])
        o = center + rng.uniform(-0.03, 0.03, 3)
        h1, h2 = o + q @ h1, o + q @ h2
        x += [o, h1, h2, o + A * (h1 - o) + A * (h2 - o)]
    names = ["OW", "HW", "HW", "MW"] * waters
    text = ["[ defaults ]", "1 2 no 1.0 1.0", "[ atomtypes ]",
            f"OW 8 16.0 0.0 A {SIGMA} {EPSILON}", "HW 1 1.008 0.0 A 0.0 0.0",
            "MW 0 0.0 0.0 D 0.0 0.0", "[ moleculetype ]", "SOL 2", "[ atoms ]",
            f"1 OW 1 SOL OW 1 {CHARGES['OW']} 16.0", f"2 HW 1 SOL HW1 1 {CHARGES['HW']} 1.008",
            f"3 HW 1 SOL HW2 1 {CHARGES['HW']} 1.008", f"4 MW 1 SOL MW 1 {CHARGES['MW']} 0.0",
            "[ virtual_sites3 ]", f"4 1 2 3 1 {A} {A}",
            "[ exclusions ]", "1 2 3 4", "2 1 3 4", "3 1 2 4", "4 1 2 3",
            "[ system ]", "OPC", "[ molecules ]", f"SOL {waters}"]
    (work / "opc.top").write_text("\n".join(text) + "\n")
    atoms = ["OW", "HW1", "HW2", "MW"]
    (work / "opc.gro").write_text("OPC\n" + f"{len(x)}\n" + "".join(
        f"{i // 4 + 1:5d}{'SOL':<5}{atoms[i % 4]:>5}{i + 1:5d}"
        f"{p[0]:12.7f}{p[1]:12.7f}{p[2]:12.7f}\n" for i, p in enumerate(x))
        + f"{edge} {edge} {edge}\n")
    loaded = mdir.load_gromacs(str(work / "opc.top"), str(work / "opc.gro"))
    system, state = loaded.make_system(), loaded.make_state()
    system.cutoff, system.pairlist_distance = RC, RC + 0.1
    system.truncation = mdir.Truncation.None_
    system.electrostatics = mdir.Electrostatics.Cutoff
    system.dispersion = mdir.DispersionCorrection.None_
    top = system.topology
    pairs, types = top.type_pairs, list(top.type_names)
    ow = types.index("OW")
    oo = [k for k, (a, b) in enumerate(pairs) if a == ow and b == ow]
    taken = np.full(len(pairs), -1)
    taken[oo[0]] = 0
    charge_map = np.array([{"OW": 0, "HW": 1, "MW": 2}[n] for n in names], dtype=np.int64)
    system.tunables = [mdir.Tunable("sigma_pair", "sigma_pair", map=taken),
                       mdir.Tunable("epsilon_pair", "epsilon_pair", map=taken),
                       mdir.Tunable("q", "charge", map=charge_map)]
    system.tunable_gradient = True
    return system, state, names, edge


def model(y, theta, waters, edge):
    """U of the frames (K, N, 3) and G = J^T dU/dX, the sites placed;
    theta = (sigma_OO, epsilon_OO, q_O, q_H, q_M)."""
    sigma, epsilon = theta[0], theta[1]
    q = np.tile(theta[2:5][[0, 1, 1, 2]], waters)
    energies, gradients = np.zeros(len(y)), np.zeros_like(y)
    for k, frame in enumerate(y):
        X = frame.copy()
        o, h1, h2 = X[0::4], X[1::4], X[2::4]
        X[3::4] = o + A * (h1 - o) + A * (h2 - o)
        dX = np.zeros_like(X)
        n = len(X)
        for i in range(n - 1):
            for j in range(i + 1, n):
                if i // 4 == j // 4:
                    continue
                d = X[i] - X[j]
                d -= edge * np.round(d / edge)
                r = np.sqrt(d @ d)
                if r >= RC:
                    continue
                slope = 0.0
                if i % 4 == 0 and j % 4 == 0:
                    lj = lambda t: 4 * epsilon * ((sigma / t) ** 12 - (sigma / t) ** 6)
                    energies[k] += lj(r) - lj(RC)
                    slope += 4 * epsilon * (-12 * sigma ** 12 / r ** 13 + 6 * sigma ** 6 / r ** 7)
                energies[k] += F * q[i] * q[j] * (1 / r - 1 / RC)
                slope += -F * q[i] * q[j] / r ** 2
                dX[i] += slope * d / r
                dX[j] -= slope * d / r
        m = dX[3::4]
        dX[0::4] += (1 - 2 * A) * m
        dX[1::4] += A * m
        dX[2::4] += A * m
        dX[3::4] = 0.0
        gradients[k] = dX
    return energies, gradients


def phi(y, theta, g, v, waters, edge):
    e, grad = model(y, theta, waters, edge)
    return g * e + np.einsum("nij,nij->n", v, grad)


def theta_differences(x, theta, g, v, waters, edge):
    """d(sum_n Phi_n)/dtheta by central differences, Richardson-extrapolated."""
    result = np.zeros(len(theta))
    for t in range(len(theta)):
        def central(h):
            e = np.zeros(len(theta))
            e[t] = h
            return (phi(x, theta + e, g, v, waters, edge).sum()
                    - phi(x, theta - e, g, v, waters, edge).sum()) / (2.0 * h)
        h = 1e-4 * max(abs(theta[t]), 1.0)
        result[t] = (4.0 * central(0.5 * h) - central(h)) / 3.0
    return result


def run(waters, edge, label):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        system, state, names, edge = fixture(waters, edge)
        evaluator = mdir.FrameEvaluator(base.compile_(system, state))
    order = list(evaluator._slices)
    theta = np.array([evaluator.tunables["sigma_pair"][0], evaluator.tunables["epsilon_pair"][0],
                      *evaluator.tunables["q"]])
    assert np.allclose(theta, [SIGMA, EPSILON, CHARGES["OW"], CHARGES["HW"], CHARGES["MW"]])
    x = base.frames(state, count=2, spread=0.01)
    g, v = base.cotangents(x)
    out, total, dx = hvp(evaluator, x, g, v)
    energies, gradients = model(x, theta, waters, edge)
    tight = 1e-10 if double else 1e-5
    base.check(f"{label}: the NumPy energy against the evaluator", out.energy, energies, tight)
    base.check(f"{label}: the NumPy gradient (zero on the sites) against the evaluator",
               np.asarray(out.gradients["positions"]), gradients, tight)
    # The entries in theta: sigma_OO, epsilon_OO, then the charges O, H, M.
    reference = theta_differences(x, theta, g, v, waters, edge)
    product = base.flat(out.vjp(g, positions=v), order)
    tolerance = 1e-8 if double else 2e-5
    base.check(f"{label}, {len(x)} frames of {x.shape[1]} particles: the product in theta "
               "against central differences of the NumPy Phi", product, reference, tolerance)
    base.check(f"{label}: the product in theta of the variant against D381's", total, product, 0.0)
    reference = richardson(lambda y: phi(y, theta, g, v, waters, edge), x, 1e-4)
    base.check(f"{label}: the derivative in the positions against central differences of the "
               "NumPy Phi", dx, reference, tolerance)
    assert np.array_equal(dx[:, 3::4], np.zeros_like(dx[:, 3::4])), "a site moved Phi"
    if double:
        base.check(f"{label}: the same against central differences of Phi from the evaluator",
                   dx, richardson(evaluator_phi(evaluator, g, v), x, 1e-4),
                   1e-7)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            exact = mdir.FrameEvaluator(base.compile_(system, state, base.target, "Double"))
        _, total_exact, dx_exact = hvp(exact, x, g, v)
        base.check(f"{label}: the product in theta in mixed precision against that in double",
                   total, total_exact, 1e-5)
        base.check(f"{label}: the derivative in the positions in mixed precision against that "
                   "in double", dx, dx_exact, 1e-5)
    # The cotangent on a site meets a gradient that is zero: it is not read.
    w = v.copy()
    w[:, 3::4] = np.random.default_rng(5).normal(size=w[:, 3::4].shape)
    moved = out.vjp(g, positions=w, gradients=("tunables", "positions"))
    # The placement adds (J Gbar - Gbar) to the entry of a site, so the
    # cotangent there leaves a rounding of the last place.
    base.check(f"{label}: the product in theta with another cotangent on the sites",
               base.flat(moved, order), total, 1e-14 if double else 1e-6)
    base.check(f"{label}: the derivative in the positions with another cotangent on the sites",
               np.asarray(moved.positions), dx, 1e-14 if double else 1e-6)


def run_refusals():
    opc = base.root + "/../opc"
    loaded = mdir.load_amber(opc + "/opc.prmtop", opc + "/opc.inpcrd")
    system, state = loaded.make_system(), loaded.make_state()
    system.cutoff, system.pairlist_distance = 0.8, 0.9
    system.electrostatics = mdir.Electrostatics.Cutoff
    system.dispersion = mdir.DispersionCorrection.None_
    system.tunables = [mdir.Tunable("sigma", "sigma_pair")]
    system.tunable_gradient = True
    evaluator = mdir.FrameEvaluator(base.compile_(system, state))
    y = np.array(state.positions)[None]
    out = evaluator.evaluate(y)
    for gradients in (("tunables",), ("positions",)):
        base.expect(mdir.UnsupportedError,
                    lambda: out.vjp(np.ones(1), positions=np.ones_like(y), gradients=gradients),
                    "virtual sites placed by the directions of their bonds")
    assert np.isfinite(evaluator.evaluate(y).energy).all()
    print("refusals: the extra points of Amber are refused by name, in theta and in the "
          "positions; the evaluator stays usable")


if scenario == "pair":
    run(2, 4.0, "two OPC waters")
elif scenario == "box":
    run(8, 2.0, "eight OPC waters in 2.0 nm")
else:
    run_refusals()
