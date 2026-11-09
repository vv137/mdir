"""Forward mode in the tunables of the frame evaluator,
`FrameEnergies.jvp(tangent)` (D389, #726): the tangent
of the energies, dU_n/dtheta . tangent, and of their gradients in the
positions, d2U_n/dx dtheta . tangent, against independent references.

Every scenario checks, beside its references:

- the tangent of the energy against the rows of the Jacobian times the
  tangent, and with gradients=() (no pass of the product program);
- the transpose identity with the product of D381 (gate 3 of the design),
  tangent . vjp(g, positions=v) = g . dU + v . dG, with a cotangent g of the
  energies that is not zero, where the product is formed;
- in double precision, the tangent of the gradient against central
  differences of the gradient of the evaluator in theta along the tangent,
  extrapolated by Ridders' method with the first step of D371's tests;
- in mixed precision, the tangent of the gradient against that of double.

Usage: python_frame_jvp.py ROOT SCENARIO [TARGET PRECISION [WORK]]

Scenarios:
  dihedral  the force constants of proper dihedrals (D372) on the torsion
            fixtures of python_frame_product.py: against the closed form,
            sum over the rows of the tangent of their k times the gradient
            of 1 + cos(n phi - delta) (dual numbers)
  lj        propane and water, per-type (geometric) and pair sigma and
            epsilon with a pair set apart by NBFIX, a plain cutoff with the
            correction for the dispersion: against a NumPy sum of the forces
            of the Lennard-Jones, differentiated in theta by a complex step
  charges   the charges under a Coulomb cutoff on the dipeptide in water,
            tied by a map: against a NumPy sum of the forces of the shifted
            Coulomb and of the 1-4 pairs, differentiated in the charges
  sites     eight OPC waters with a linear site (GROMACS): against central
            differences in theta of the gradient of the NumPy model of
            python_frame_sites.py
  amber     Amber OPC with its extra points, whose placement depends on the
            positions but not on theta (ParmEd cuts the topology)
  pme       the charges with particle mesh Ewald, per particle and tied,
            whose reverse product is not formed (D381): the gradient is
            quadratic in the charges, so that its half difference at the
            charges shifted by the tangent and back is exact but for the
            rounding
  torchpme  the same per particle against torch-pme (Loche et al., J.
            Chem. Phys. 162, 142501 (2025)): the half difference of its
            gradient in the positions, at two grids, within three times what
            each program changes by between them (python_frame_gradients.py)
  refusals  what is refused, and that the evaluator stays usable (CPU)
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import mdir

sys.path.insert(0, str(Path(__file__).parent))
import python_frame_product as base  # noqa: E402  (reads sys.argv as this script does)

scenario = sys.argv[2]
double = base.double
mode = base.mode
check = base.check


def flat(mapping, names):
    return np.concatenate([np.asarray(mapping[name], dtype=np.float64) for name in names])


def random_tangent(evaluator, seed=726):
    rng = np.random.default_rng(seed)
    return {name: rng.normal(size=np.shape(evaluator.tunables[name])) *
            np.maximum(np.abs(np.asarray(evaluator.tunables[name])), 1e-3)
            for name in evaluator._slices}


def gradient_at(evaluator, x, theta, tangent, s):
    """The gradient in the positions of the evaluator at theta + s tangent."""
    evaluator.tunables.update({name: theta[name] + s * tangent[name] for name in theta})
    out = evaluator.evaluate(x, gradients=("tunables", "positions"))
    return out, np.asarray(out.gradients["positions"], dtype=np.float64)


def ridders(evaluator, x, tangent, exact):
    """Central differences of the gradient in theta along `tangent`,
    extrapolated to a zero step by Ridders' method (Ridders, Adv. Eng.
    Software 4, 75 (1982); the tableau of Press et al., Numerical Recipes,
    Section 5.7), steps h0 / 1.4^k, k = 0 to 15, and the estimate whose
    neighbours in the tableau agree best. The first step as in D371's tests:
    the tangent of the gradient changes by itself over l = |dG| / |d2G|,
    d2G the central difference of the exact tangent over 1e-6, h0 = l / 20,
    and at most a twentieth of the scale of the values over that of the
    tangent, where the gradient is linear in theta. The values of the
    tunables are those of before at the end."""
    names = list(evaluator._slices)
    theta = {name: np.array(evaluator.tunables[name]) for name in names}
    try:
        tangents = []
        for s in (1e-6, -1e-6):
            out, _ = gradient_at(evaluator, x, theta, tangent, s)
            tangents.append(np.asarray(out.jvp(tangent).gradients["positions"]))
        curvature = (tangents[0] - tangents[1]) / 2e-6
        scale = max(np.abs(flat(theta, names)).max(), 1e-3) / max(
            np.abs(flat(tangent, names)).max(), 1e-300)
        h = min(np.linalg.norm(exact) / max(np.linalg.norm(curvature), 1e-300) / 20.0,
                scale / 20.0)
        con, table, best, error = 1.4, [[None] * 16 for _ in range(16)], None, np.inf
        for i in range(16):
            if i:
                h /= con
            table[0][i] = (gradient_at(evaluator, x, theta, tangent, h)[1] -
                           gradient_at(evaluator, x, theta, tangent, -h)[1]) / (2.0 * h)
            factor = con * con
            for j in range(1, i + 1):
                table[j][i] = (table[j - 1][i] * factor - table[j - 1][i - 1]) / (factor - 1.0)
                factor *= con * con
                estimate = max(np.abs(table[j][i] - table[j - 1][i]).max(),
                               np.abs(table[j][i] - table[j - 1][i - 1]).max())
                if estimate <= error:
                    error, best = estimate, table[j][i]
        return best
    finally:
        evaluator.tunables.update(theta)


def forward(evaluator, x, label, reference=None, product=True, reference_tolerance=None,
            exact_program=None, seed=726, energy_tolerance=None):
    """The checks of every scenario (the docstring of the module); returns
    the tangent of the gradient."""
    names = list(evaluator._slices)
    tangent = random_tangent(evaluator, seed)
    out = evaluator.evaluate(x, gradients=("tunables", "positions"))
    result = out.jvp(tangent)
    assert isinstance(result, mdir.FrameTangent) and result.version == out.version
    dg = np.asarray(result.gradients["positions"])
    assert dg.shape == (len(x), x.shape[1], 3) and dg.dtype == np.float64
    rows = np.concatenate([out.jacobian[name] for name in names], axis=1)
    check(f"{label}: the tangent of the energy against the rows of the Jacobian times the "
          "tangent", result.energy, rows @ flat(tangent, names),
          energy_tolerance or (1e-12 if double else 1e-5))
    alone = out.jvp(tangent, gradients=())
    assert dict(alone.gradients) == {}
    check(f"{label}: the same with gradients=(), without the product program", alone.energy,
          rows @ flat(tangent, names), 1e-14 if double else 1e-6)
    if reference is not None:
        check(f"{label}: the tangent of the gradient in the positions against {reference[0]}",
              dg, reference[1], reference_tolerance or (1e-10 if double else 2e-5))
    if product:
        rng = np.random.default_rng(seed + 1)
        g, v = rng.normal(size=len(x)), rng.normal(size=x.shape)
        p = flat(out.vjp(g, positions=v), names)
        lhs = float(p @ flat(tangent, names))
        rhs = float(g @ result.energy + np.sum(v * dg))
        relative = abs(lhs - rhs) / abs(lhs)
        # In mixed precision, relative to the sum of the magnitudes of the
        # terms, of which the sums cancel a part: the rounding of the
        # kernels is that of each term.
        terms = float(np.abs(p * flat(tangent, names)).sum() + np.abs(g * result.energy).sum() +
                      np.abs(v * dg).sum())
        error = relative if double else abs(lhs - rhs) / terms
        tolerance = 1e-12 if double else 1e-5
        print(f"{mode}: {label}: the transpose identity, tangent . vjp(g, positions=v) = "
              f"{lhs:.12g} against g . dU + v . dG = {rhs:.12g}, {relative:.1e} relative, "
              f"{abs(lhs - rhs) / terms:.1e} of the sum of the magnitudes of the terms "
              f"(tolerance {tolerance:.0e} {'relative' if double else 'of that sum'})")
        assert error <= tolerance, (label, error)
    if double:
        check(f"{label}: the same against central differences of the gradient in theta "
              "along the tangent (Ridders)", dg, ridders(evaluator, x, tangent, dg), 1e-6)
    elif exact_program is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            exact = mdir.FrameEvaluator(exact_program)
        reference_dg = np.asarray(exact.evaluate(x).jvp(tangent).gradients["positions"])
        check(f"{label}: the tangent of the gradient in mixed precision against that in "
              "double", dg, reference_dg, 1e-5)
    return dg


def exact_program(system, state):
    if double:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return base.compile_(system, state, base.target, "Double")


# --- dihedral --------------------------------------------------------------

def run_dihedral():
    system, state = base.torsion_fixture()
    system.tunables = [mdir.Tunable("k", "dihedral_k")]
    system.tunable_gradient = True
    program = base.compile_(system, state)
    evaluator = mdir.FrameEvaluator(program)
    x = base.frames(state, spread=0.02)
    tangent = random_tangent(evaluator)
    entry = np.asarray(program.plan["tunables"][0]["map"])
    top = system.topology
    reference = np.zeros(x.shape)
    for n in range(len(x)):
        for i in range(x.shape[1]):
            for c in range(3):
                e = np.zeros(x.shape[1:])
                e[i, c] = 1.0
                _, slopes = base.torsion_basis(x[n], e, top)
                reference[n, i, c] = sum(tangent["k"][at] * slopes[row]
                                         for row, at in enumerate(entry) if at >= 0)
    forward(evaluator, x, f"dihedral_k, {len(x)} frames",
            ("the closed form (dual numbers)", reference),
            exact_program=exact_program(system, state))


# --- lj --------------------------------------------------------------------

def run_lj():
    rc = 0.8
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        system, state = base.propane(lambda s: [mdir.Tunable("s", "sigma_pair"),
                                                mdir.Tunable("e", "epsilon_pair")],
                                     gradient=False)
        model = mdir.Simulation(base.compile_(system, state, "CPU", "Double"))
    top = system.topology
    names, pairs, types = list(top.type_names), top.type_pairs, np.asarray(top.particle_types)
    count = len(names)
    sigma0, epsilon0 = model.tunables["s"].copy(), model.tunables["e"].copy()
    site = {(a, b): k for k, (a, b) in enumerate(pairs)}
    diagonal = np.array([site[a, a] for a in range(count)])
    ct, ow, hc = names.index("CT"), names.index("OW"), names.index("HC")
    nbfix = site[min(ct, ow), max(ct, ow)]
    keep = np.where(epsilon0[diagonal] > 0.0, np.cumsum(epsilon0[diagonal] > 0.0) - 1, -1)
    taken_sigma = np.full(len(pairs), -1)
    taken_sigma[nbfix], taken_sigma[site[ow, ow]] = 0, 1
    taken_epsilon = np.full(len(pairs), -1)
    taken_epsilon[nbfix], taken_epsilon[site[min(ct, hc), max(ct, hc)]] = 0, 1

    def tunables(system):
        return [mdir.Tunable("sigma", "sigma", map=keep, mixing="geometric"),
                mdir.Tunable("epsilon", "epsilon", map=keep),
                mdir.Tunable("sigma_pair", "sigma_pair", map=taken_sigma),
                mdir.Tunable("epsilon_pair", "epsilon_pair", map=taken_epsilon)]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        system, state = base.propane(tunables)
        program = base.compile_(system, state)
    evaluator = mdir.FrameEvaluator(program)
    x = base.frames(state)
    box = np.array(state.cell.diagonal)
    n = x.shape[1]
    excluded = base.propane_excluded(n)
    theta0 = {name: np.array(evaluator.tunables[name]) for name in evaluator._slices}
    tangent = random_tangent(evaluator)

    def table(theta):
        # As python_frame_product.py, in complex numbers.
        sigma_type = sigma0[diagonal].astype(complex)
        epsilon_type = epsilon0[diagonal].astype(complex)
        sigma_type[keep >= 0] = theta["sigma"][keep[keep >= 0]]
        epsilon_type[keep >= 0] = theta["epsilon"][keep[keep >= 0]]
        sigma, epsilon = sigma0.astype(complex), epsilon0.astype(complex)
        for (a, b), k in site.items():
            if k != nbfix:
                sigma[k] = np.sqrt(sigma_type[a] * sigma_type[b])
                epsilon[k] = np.sqrt(epsilon_type[a] * epsilon_type[b])
        sigma[taken_sigma >= 0] = theta["sigma_pair"][taken_sigma[taken_sigma >= 0]]
        epsilon[taken_epsilon >= 0] = theta["epsilon_pair"][taken_epsilon[taken_epsilon >= 0]]
        return sigma, epsilon

    frame_pairs = []
    for y in x:
        rows = []
        for i in range(n - 1):
            d = y[i + 1:] - y[i]
            d -= box * np.round(d / box)
            r = np.sqrt((d * d).sum(axis=1))
            for j in np.nonzero(r < rc)[0]:
                if (i, i + 1 + j) in excluded:
                    continue
                a, b = sorted((types[i], types[i + 1 + j]))
                rows.append((i, i + 1 + j, site[a, b], r[j], d[j]))
        frame_pairs.append(rows)

    def gradient(theta):
        # dU/dx of the Lennard-Jones within the cutoff: u'(r) (x_i - x_j)/r
        # at i, with u = 4 epsilon ((sigma/r)^12 - (sigma/r)^6); the shift
        # and the correction read no position.
        sigma, epsilon = table(theta)
        result = np.zeros(x.shape, dtype=complex)
        for f, rows in enumerate(frame_pairs):
            i, j, k = (np.array([row[c] for row in rows]) for c in range(3))
            r = np.array([row[3] for row in rows])
            d = np.array([row[4] for row in rows])
            slope = 4.0 * epsilon[k] * (-12.0 * sigma[k] ** 12 * r ** -13 +
                                        6.0 * sigma[k] ** 6 * r ** -7)
            term = (slope / r)[:, None] * d
            np.add.at(result[f], i, -term)
            np.add.at(result[f], j, term)
        return result

    step = 1e-30
    shifted = {name: theta0[name] + 1j * step * tangent[name] for name in theta0}
    reference = gradient(shifted).imag / step
    forward(evaluator, x, f"propane and water, per-type and pair sigma and epsilon with a "
            f"pair set apart, {len(x)} frames",
            ("a NumPy sum of the forces of the Lennard-Jones, by a complex step in theta",
             reference), exact_program=exact_program(system, state))


# --- charges ---------------------------------------------------------------

def charges_system(electrostatics=mdir.Electrostatics.Cutoff, tied=True):
    loaded = mdir.load_amber(base.root + "/dipeptide.prmtop", base.root + "/dipeptide.inpcrd")
    system, state = loaded.make_system(), loaded.make_state()
    system.truncation = mdir.Truncation.Switch
    system.cutoff, system.pairlist_distance, system.switch_distance = 0.8, 0.9, 0.7
    system.electrostatics = electrostatics
    system.dispersion = mdir.DispersionCorrection.None_
    top = system.topology
    residues = [top.residue_names[r] for r in top.residue_indices]
    entries, charge_map = {}, []
    for i, (atom, residue) in enumerate(zip(top.atom_names, residues)):
        key = (residue, atom) if residue == "WAT" else (residue, i)
        charge_map.append(entries.setdefault(key, len(entries)))
    charge_map = np.array(charge_map, dtype=np.int64)
    system.tunables = [mdir.Tunable("q", "charge", map=charge_map if tied else None)]
    system.tunable_gradient = True
    return system, state, charge_map


def run_charges():
    rc, f = 0.8, 138.93545755023302
    system, state, charge_map = charges_system()
    program = base.compile_(system, state)
    evaluator = mdir.FrameEvaluator(program)
    x = base.frames(state, count=2)
    tangent = random_tangent(evaluator)
    q = np.asarray(evaluator.tunables["q"])[charge_map]
    dq = tangent["q"][charge_map]
    box = np.array(state.cell.diagonal)
    n = x.shape[1]
    top = system.topology
    graph = {i: set() for i in range(n)}
    for i, j in top.bonds:
        graph[int(i)].add(int(j))
        graph[int(j)].add(int(i))
    excluded, scaled = set(), set()
    for i in range(n):
        seen, frontier = {i}, {i}
        for depth in range(3):
            frontier = {k for m in frontier for k in graph[m]} - seen
            seen |= frontier
            for j in frontier:
                if j > i:
                    (scaled if depth == 2 else excluded).add((i, j))
    # f q_i q_j (1/r - 1/rc) within the cutoff, f q_i q_j / (1.2 r) for the
    # 1-4 pairs (D210): the gradient at i is f q_i q_j d / r^3 with
    # d = x_j - x_i, and its tangent takes dq_i q_j + q_i dq_j for q_i q_j.
    reference = np.zeros(x.shape)
    for k in range(len(x)):
        y = x[k]
        for i in range(n - 1):
            d = y[i + 1:] - y[i]
            d -= box * np.round(d / box)
            r = np.sqrt((d * d).sum(axis=1))
            for j in np.nonzero(r < rc)[0]:
                pair = (i, i + 1 + int(j))
                if pair in excluded or pair in scaled:
                    continue
                rate = dq[i] * q[pair[1]] + q[i] * dq[pair[1]]
                term = f * rate * d[j] / r[j] ** 3
                reference[k, i] += term
                reference[k, pair[1]] -= term
        for i, j in scaled:
            d = y[j] - y[i]
            d -= box * np.round(d / box)
            r = np.sqrt((d * d).sum())
            rate = dq[i] * q[j] + q[i] * dq[j]
            term = f * rate * d / (1.2 * r ** 3)
            reference[k, i] += term
            reference[k, j] -= term
    forward(evaluator, x, f"the charges under a Coulomb cutoff, {len(tangent['q'])} entries "
            f"for {n} particles, {len(x)} frames",
            ("a NumPy sum of the forces of the shifted Coulomb and of the 1-4 pairs", reference),
            reference_tolerance=1e-10 if double else 2e-5,
            exact_program=exact_program(system, state))


# --- sites -----------------------------------------------------------------

def run_sites():
    import python_frame_sites as sites
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        system, state, names, edge = sites.fixture(8, 2.0)
        program = base.compile_(system, state)
    evaluator = mdir.FrameEvaluator(program)
    x = base.frames(state, count=2, spread=0.01)
    tangent = random_tangent(evaluator)
    theta = np.array([evaluator.tunables["sigma_pair"][0], evaluator.tunables["epsilon_pair"][0],
                      *evaluator.tunables["q"]])
    along = np.array([tangent["sigma_pair"][0], tangent["epsilon_pair"][0], *tangent["q"]])

    def central(h):
        return (sites.model(x, theta + h * along, 8, edge)[1] -
                sites.model(x, theta - h * along, 8, edge)[1]) / (2.0 * h)
    h = 1e-4
    reference = (4.0 * central(0.5 * h) - central(h)) / 3.0
    forward(evaluator, x, "eight OPC waters in 2.0 nm, a linear site",
            ("central differences in theta of the gradient of the NumPy model (Richardson)",
             reference), reference_tolerance=1e-8 if double else 2e-5,
            exact_program=exact_program(system, state))


def run_amber():
    import python_frame_sites as sites
    prmtop, inpcrd, _ = sites.amber_subset(40, True)
    system, state, placed = sites.amber_fixture(mdir.load_amber(prmtop, inpcrd))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        program = base.compile_(system, state)
    evaluator = mdir.FrameEvaluator(program)
    model = sites.AmberModel(system, placed, np.array(state.cell.diagonal),
                             sites.ep_distance(prmtop))
    x, _ = sites.clear_frames(state, model, 2, 2e-5)
    forward(evaluator, x, f"Amber OPC (ACE-ALA-NME and waters, {x.shape[1]} particles), the "
            "extra points", exact_program=exact_program(system, state))


# --- pme -------------------------------------------------------------------

def run_pme():
    for tied in (False, True):
        system, state, _ = charges_system(mdir.Electrostatics.PME, tied)
        program = base.compile_(system, state)
        evaluator = mdir.FrameEvaluator(program)
        x = base.frames(state, count=2)
        tangent = random_tangent(evaluator)
        theta = {"q": np.array(evaluator.tunables["q"])}
        plus = gradient_at(evaluator, x, theta, tangent, 1.0)[1]
        minus = gradient_at(evaluator, x, theta, tangent, -1.0)[1]
        evaluator.tunables.update(theta)
        label = (f"the charges with PME, {'tied in' if tied else 'per particle,'} "
                 f"{len(theta['q'])} entries, {len(x)} frames")
        forward(evaluator, x, label,
                ("the half difference of the gradient at the charges shifted by the tangent "
                 "and back (exact: quadratic in the charges)", (plus - minus) / 2.0),
                product=False, reference_tolerance=1e-12 if double else 2e-5,
                # The rows take the potential of the mesh at the particles
                # (D231), the tangent two sums of the mesh: each carries
                # the rounding of the mesh, about 1e-10 of its energy (D379).
                energy_tolerance=1e-9 if double else 1e-5,
                exact_program=exact_program(system, state))


def run_torchpme():
    sys.argv[2] = "import"
    import python_frame_gradients as gradients

    def ours(points):
        system, state = gradients.oracle_model(points, "amber", charges=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            evaluator = mdir.FrameEvaluator(base.compile_(system, state))
        x = np.array(state.positions)
        cell = (np.array(state.cell.diagonal), np.array(state.cell.tilt))
        q0 = np.array(evaluator.tunables["q"])
        tangent = {"q": np.random.default_rng(726).normal(size=q0.shape) * 0.1}
        out = evaluator.evaluate(x[None], cell[0][None], None)
        dg = np.asarray(out.jvp(tangent).gradients["positions"][0])
        return system, x, gradients.vectors(*cell), q0, tangent["q"], dg

    import torch
    import torchpme
    points = 64
    system, x, h, q0, dq, dg = ours(points)
    dg2 = ours(2 * points)[5]
    length = float(np.sqrt((h * h).sum(axis=1)).max())
    spacings = [2.0 * length / (m - 1.5) for m in (points, 2 * points)]
    for m, spacing in zip((points, 2 * points), spacings):
        mesh = [int(k) for k in torchpme.lib.get_ns_mesh(torch.tensor(h), spacing)]
        assert mesh == [m] * 3, (mesh, m)
    references = []
    for spacing in spacings:
        plus = gradients.torchpme_reference(system, x, h, q0 + dq, spacing)[0]
        minus = gradients.torchpme_reference(system, x, h, q0 - dq, spacing)[0]
        references.append((plus - minus) / 2.0)
    rg, rg2 = references
    scale = np.abs(rg).max()
    gradients.compared(f"the dipeptide in water, {len(x)} particles: the tangent of G in the "
                       f"charges against torch-pme, grid {points}", dg, rg,
                       np.abs(dg - dg2).max() / scale, np.abs(rg - rg2).max() / scale, 2e-5)
    gradients.compared(f"the dipeptide in water: the same at {2 * points} points", dg2, rg2,
                       np.abs(dg - dg2).max() / scale, np.abs(rg - rg2).max() / scale, 2e-5)
    print("torchpme passed")


# --- refusals --------------------------------------------------------------

def run_refusals():
    expect = base.expect
    system, state = base.torsion_fixture()
    system.tunables = [mdir.Tunable("k", "dihedral_k")]
    system.tunable_gradient = True
    evaluator = mdir.FrameEvaluator(base.compile_(system, state))
    x = base.frames(state, spread=0.02)
    out = evaluator.evaluate(x, gradients=("tunables", "positions"))
    tangent = random_tangent(evaluator)
    first = np.asarray(out.jvp(tangent).gradients["positions"]).copy()
    expect(mdir.InputError, lambda: out.jvp([1.0]), "a mapping")
    expect(mdir.InputError, lambda: out.jvp({"nope": [1.0]}), "is not a tunable")
    expect(mdir.InputError, lambda: out.jvp({"k": tangent["k"][:-1]}), "of shape")
    bad = tangent["k"].copy()
    bad[0] = np.inf
    expect(mdir.InputError, lambda: out.jvp({"k": bad}), "not finite")
    expect(mdir.InputError, lambda: out.jvp(tangent, gradients=("tunables",)), "names")
    expect(mdir.InputError, lambda: out.jvp(tangent, gradients="positions"), "a tuple")
    once = evaluator.evaluate(iter(list(x)))
    expect(mdir.InputError, lambda: once.jvp(tangent), "an iterator gives them once")
    assert np.isfinite(once.jvp(tangent, gradients=()).energy).all()
    values = np.array(evaluator.tunables["k"])
    evaluator.tunables.update({"k": values * 1.5})
    expect(mdir.SimulationError, lambda: out.jvp(tangent), "changed since")
    evaluator.tunables.update({"k": values})
    # A name left out is zero.
    assert not np.asarray(evaluator.evaluate(x).jvp({}).gradients["positions"]).any()
    again = evaluator.evaluate(x).jvp(tangent)
    assert np.array_equal(np.asarray(again.gradients["positions"]), first)
    dependent = mdir.FrameEvaluator(base.compile_(system, state), terms="dependent")
    expect(mdir.UnsupportedError, lambda: dependent.evaluate(x).jvp(tangent),
           "terms='dependent'")
    print("refusals: a tangent that is not a mapping, a name that is not a tunable, a shape, a "
          "value that is not finite, the names of gradients, frames given once, and values "
          "changed since the frames were evaluated are refused, and the tangent of the gradient "
          "with terms='dependent'; a name left out is zero; the evaluator gives the same "
          "tangent after them, to the bit")


if __name__ == "__main__":
    {"dihedral": run_dihedral, "lj": run_lj, "charges": run_charges, "sites": run_sites,
     "amber": run_amber,
     "pme": run_pme, "torchpme": run_torchpme, "refusals": run_refusals}[scenario]()
