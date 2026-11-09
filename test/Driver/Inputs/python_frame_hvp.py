"""The derivative of the product of force matching in the positions
(D383, #771 part 1):

  dPhi_n/dx_i = g_n dU_n/dx_i + sum_j d2U_n/dx_i dx_j . Gbar_n,j,
  Phi_n = g_n U_n + Gbar_n . dU_n/dx,

from the product program that gives it, against NumPy oracles and against
central differences of Phi from the evaluator.

Usage: python_frame_hvp.py ROOT SCENARIO [TARGET PRECISION [WORK]]

Scenarios:
  pairs     a Lennard-Jones and Coulomb fluid of two types with a pair of
            types set apart (NBFIX), under a plain cutoff: the shifted
            Lennard-Jones and the shifted Coulomb (D210) in NumPy, their
            Phi, and its gradient in the positions by central differences
            of the NumPy Phi; and central differences of Phi from the
            evaluator; in mixed precision, against double
  dihedral  the torsion fixtures of D372: the closed form of Phi with dual
            numbers in NumPy, its gradient by central differences; and the
            product in theta, equal to D381's to the bit
  refusals  what is refused, by name, and that the evaluator stays usable
"""
import sys
from pathlib import Path

import numpy as np
import mdir

sys.path.insert(0, str(Path(__file__).parent))
import python_frame_product as base  # noqa: E402  (reads sys.argv as this script does)

scenario = sys.argv[2]
work = base.work
double, mode = base.double, base.mode
F = 138.93545755023302


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


# --- pairs -----------------------------------------------------------------

TYPES = {"A": (0.30, 0.8, 0.45), "B": (0.36, 1.2, -0.45)}  # sigma nm, epsilon kJ/mol, charge e
NBFIX = (0.33, 0.6)
RC, EDGE, COUNT = 0.9, 2.2, 24


def pair_fixture():
    work.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(771)
    # A lattice with displacements, so that no two particles come close.
    side = 3
    sites = np.array([(i, j, k) for i in range(side) for j in range(side) for k in range(side)],
                     dtype=float)[:COUNT]
    x = (sites + 0.5) * (EDGE / side) + rng.uniform(-0.08, 0.08, sites.shape)
    names = ["A" if i % 2 == 0 else "B" for i in range(COUNT)]
    text = ["[ defaults ]", "1 2 no 1.0 1.0", "[ atomtypes ]"]
    for name, (sigma, epsilon, _) in TYPES.items():
        text.append(f"{name} 1 20.0 0.0 A {sigma} {epsilon}")
    text += ["[ nonbond_params ]", f"A B 1 {NBFIX[0]} {NBFIX[1]}"]
    for name, (_, _, charge) in TYPES.items():
        text += ["[ moleculetype ]", f"{name}M 0", "[ atoms ]",
                 f"1 {name} 1 {name}M {name} 1 {charge} 20.0"]
    text += ["[ system ]", "fluid", "[ molecules ]"]
    text += [f"{n}M 1" for n in names]
    (work / "fluid.top").write_text("\n".join(text) + "\n")
    (work / "fluid.gro").write_text("fluid\n" + f"{COUNT}\n" + "".join(
        f"{i + 1:5d}{n + 'M':<5}{n:>5}{i + 1:5d}{p[0]:8.3f}{p[1]:8.3f}{p[2]:8.3f}\n"
        for i, (n, p) in enumerate(zip(names, x))) + f"{EDGE} {EDGE} {EDGE}\n")
    loaded = mdir.load_gromacs(str(work / "fluid.top"), str(work / "fluid.gro"))
    system, state = loaded.make_system(), loaded.make_state()
    system.cutoff, system.pairlist_distance = RC, RC + 0.1
    system.truncation = mdir.Truncation.None_
    system.electrostatics = mdir.Electrostatics.Cutoff
    system.dispersion = mdir.DispersionCorrection.None_
    system.tunables = [mdir.Tunable("sigma", "sigma"), mdir.Tunable("epsilon", "epsilon")]
    system.tunable_gradient = True
    return system, state, names


def pair_energy(y, names):
    """The shifted Lennard-Jones and Coulomb of the pairs within the cutoff,
    with its gradient, for frames (K, N, 3)."""
    sigma = {(a, b): (TYPES[a][0] + TYPES[b][0]) / 2 for a in TYPES for b in TYPES}
    epsilon = {(a, b): np.sqrt(TYPES[a][1] * TYPES[b][1]) for a in TYPES for b in TYPES}
    sigma[("A", "B")] = sigma[("B", "A")] = NBFIX[0]
    epsilon[("A", "B")] = epsilon[("B", "A")] = NBFIX[1]
    q = np.array([TYPES[n][2] for n in names])
    energies, gradients = np.zeros(len(y)), np.zeros_like(y)
    for k, frame in enumerate(y):
        for i in range(COUNT - 1):
            for j in range(i + 1, COUNT):
                d = frame[i] - frame[j]
                d -= EDGE * np.round(d / EDGE)
                r = np.sqrt(d @ d)
                if r >= RC:
                    continue
                s, e = sigma[names[i], names[j]], epsilon[names[i], names[j]]
                lj = lambda t: 4 * e * ((s / t) ** 12 - (s / t) ** 6)
                slope = 4 * e * (-12 * s ** 12 / r ** 13 + 6 * s ** 6 / r ** 7)
                energies[k] += lj(r) - lj(RC) + F * q[i] * q[j] * (1 / r - 1 / RC)
                slope += -F * q[i] * q[j] / r ** 2
                gradients[k, i] += slope * d / r
                gradients[k, j] -= slope * d / r
    return energies, gradients


def run_pairs():
    system, state, names = pair_fixture()
    evaluator = mdir.FrameEvaluator(base.compile_(system, state))
    x = base.frames(state, count=2, spread=0.01)
    g, v = base.cotangents(x)
    out, total, dx = hvp(evaluator, x, g, v)
    # The model of NumPy is the model of the program.
    energies, gradients = pair_energy(x, names)
    base.check("the pair fluid: the NumPy energy against the evaluator", out.energy, energies,
               1e-10 if double else 1e-5)

    def phi(y):
        e, grad = pair_energy(y, names)
        return g * e + np.einsum("nij,nij->n", v, grad)
    reference = richardson(phi, x, 1e-4)
    tolerance = 1e-8 if double else 2e-5
    base.check(f"the pair fluid (LJ with NBFIX and Coulomb, shifted at the cutoff), {len(x)} "
               "frames: the derivative of the product in the positions against central "
               "differences of the NumPy Phi", dx, reference, tolerance)
    if double:
        base.check("the same against central differences of Phi from the evaluator", dx,
                   richardson(evaluator_phi(evaluator, g, v), x, 1e-4), 1e-7)
    else:
        exact = mdir.FrameEvaluator(base.compile_(system, state, base.target, "Double"))
        _, _, dexact = hvp(exact, x, g, v)
        base.check("the derivative in the positions in mixed precision against that in double",
                   dx, dexact, 1e-5)
    base.check("the product in theta of the variant against D381's", total,
               base.flat(out.vjp(g, positions=v), evaluator._slices), 0.0)


# --- dihedral --------------------------------------------------------------

def torsion_phi(y, v, g, top, constants):
    """g U + Gbar . dU/dx of the torsions, by dual numbers along Gbar."""
    values = np.zeros(len(y))
    for n in range(len(y)):
        basis, slopes = base.torsion_basis(y[n], v[n], top)
        values[n] = g[n] * (constants * basis).sum() + (constants * slopes).sum()
    return values


def run_dihedral():
    system, state = base.torsion_fixture()
    system.tunables = [mdir.Tunable("k", "dihedral_k")]
    system.tunable_gradient = True
    evaluator = mdir.FrameEvaluator(base.compile_(system, state))
    x = base.frames(state, spread=0.02)
    g, v = base.cotangents(x)
    out, total, dx = hvp(evaluator, x, g, v)
    top = system.topology
    constants = np.array(top.dihedral_force_constants)
    reference = richardson(lambda y: torsion_phi(y, v, g, top, constants), x, 1e-4)
    base.check(f"dihedrals, {len(x)} frames: the derivative of the product in the positions "
               "against central differences of the closed form of Phi (dual numbers)", dx,
               reference, 1e-8 if double else 2e-5)
    if double:
        base.check("the same against central differences of Phi from the evaluator", dx,
                   richardson(evaluator_phi(evaluator, g, v), x, 1e-4), 1e-7)
    base.check("the product in theta of the variant against D381's", total,
               base.flat(out.vjp(g, positions=v), evaluator._slices), 0.0)


# --- refusals --------------------------------------------------------------

def run_refusals():
    root = base.root
    loaded = mdir.load_amber(root + "/dipeptide.prmtop", root + "/dipeptide.inpcrd")
    system, state = loaded.make_system(), loaded.make_state()
    system.cutoff, system.pairlist_distance = 0.8, 0.9
    system.electrostatics = mdir.Electrostatics.PME
    system.dispersion = mdir.DispersionCorrection.None_
    system.tunables = [mdir.Tunable("sigma", "sigma_pair")]
    system.tunable_gradient = True
    evaluator = mdir.FrameEvaluator(base.compile_(system, state))
    y = np.array(state.positions)[None]
    g, v = np.ones(1), np.ones_like(y)
    out = evaluator.evaluate(y)
    base.expect(mdir.UnsupportedError,
                lambda: out.vjp(g, positions=v, gradients=("tunables", "positions")),
                "reciprocal")
    first = out.vjp(g, positions=v)["sigma"]
    assert np.isfinite(first).all()
    print("refusals: with particle mesh Ewald, the derivative in the positions is refused, "
          "naming the reciprocal sum; the product in theta is given")
    opc = root + "/../opc"
    loaded = mdir.load_amber(opc + "/opc.prmtop", opc + "/opc.inpcrd")
    system, state = loaded.make_system(), loaded.make_state()
    system.cutoff, system.pairlist_distance = 0.8, 0.9
    system.electrostatics = mdir.Electrostatics.Cutoff
    system.dispersion = mdir.DispersionCorrection.None_
    system.tunables = [mdir.Tunable("sigma", "sigma_pair")]
    system.tunable_gradient = True
    sites = mdir.FrameEvaluator(base.compile_(system, state))
    y = np.array(state.positions)[None]
    out = sites.evaluate(y)
    # The extra points of Amber: given since the curvature of their
    # placement (#771), zero in the rows of the sites.
    given = np.asarray(out.vjp(np.ones(1), positions=np.ones_like(y),
                               gradients=("positions",)).positions)
    rows = np.zeros(y.shape[1], dtype=bool)
    rows[np.asarray(system.topology.virtual_sites)[:, 0]] = True
    zero = np.all(given[0] == 0.0, axis=1)
    assert np.isfinite(given).all() and rows.any()
    assert np.array_equal(zero, rows), "the zero rows are not those of the sites"
    print("refusals: the extra points of Amber are given, zero in the rows of the sites")
    # The arguments of vjp.
    system, state = base.torsion_fixture()
    system.tunables = [mdir.Tunable("k", "dihedral_k")]
    system.tunable_gradient = True
    evaluator = mdir.FrameEvaluator(base.compile_(system, state))
    x = base.frames(state, spread=0.02)
    g, v = base.cotangents(x)
    out = evaluator.evaluate(x)
    base.expect(mdir.InputError, lambda: out.vjp(g, gradients=("forces",)), "gradients names")
    base.expect(mdir.InputError, lambda: out.vjp(g, gradients="positions"), "a tuple of names")
    alone = out.vjp(g, positions=v, gradients=("positions",))
    both = out.vjp(g, positions=v, gradients=("tunables", "positions"))
    assert len(alone) == 0 and np.array_equal(alone.positions, both.positions)
    assert out.vjp(g, positions=v).positions is None
    # Without a cotangent on the gradients, g dU/dx.
    energy_only = out.vjp(g, gradients=("positions",)).positions
    grad = evaluator.evaluate(x, gradients=("positions",)).gradients["positions"]
    base.check("vjp(g, gradients=('positions',)) against g dU/dx", energy_only,
               g[:, None, None] * grad, 1e-14)
    print("refusals: the names of gradients; positions alone, and beside the tunables, the "
          "same to the bit; without it, no positions")


if __name__ == "__main__":
    {"pairs": run_pairs, "dihedral": run_dihedral, "refusals": run_refusals}[scenario]()
