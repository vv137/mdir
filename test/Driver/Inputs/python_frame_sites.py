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
  amber     OPC from a prmtop of Amber, whose extra points follow the
            directions of the bonds: the product in theta against a NumPy
            model, the derivative in the positions refused by name
  placement on frames of exactly rigid waters, the extra point of Amber and
            the linear site of GROMACS (ParmEd's conversion, with
            a = d / (2 r_OH cos(theta_HOH / 2))) are one place: the same
            energy. The two spread the force on a site differently, so their
            gradients and products differ, and are not compared
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




# --- the extra points of Amber ---------------------------------------------
#
# OPC from the prmtop of test/Driver/Inputs/opc, cut by ParmEd to fewer waters:
# the extra point is x_O + d (u + v) / |u + v|, u and v the unit vectors
# from O to the hydrogens, which follows their directions. The product in
# theta takes J Gbar at the frame; the derivative in the positions is refused.

AMBER_RC = 0.8


def amber_fixture(loaded):
    system, state = loaded.make_system(), loaded.make_state()
    system.cutoff, system.pairlist_distance = AMBER_RC, AMBER_RC + 0.1
    system.truncation = mdir.Truncation.None_
    system.electrostatics = mdir.Electrostatics.Cutoff
    system.dispersion = mdir.DispersionCorrection.None_
    top = system.topology
    sites = np.asarray(top.virtual_sites)            # (site, O, H1, H2)
    n = top.particle_count
    types, pairs = np.asarray(top.particle_types), top.type_pairs
    ow = types[sites[0, 1]]
    taken = np.full(len(pairs), -1)
    taken[[k for k, (a, b) in enumerate(pairs) if a == ow and b == ow][0]] = 0
    charge_map = np.full(n, -1, dtype=np.int64)
    for column, entry in ((1, 0), (2, 1), (3, 1), (0, 2)):
        charge_map[sites[:, column]] = entry
    system.tunables = [mdir.Tunable("sigma_pair", "sigma_pair", map=taken),
                       mdir.Tunable("epsilon_pair", "epsilon_pair", map=taken),
                       mdir.Tunable("q", "charge", map=charge_map)]
    system.tunable_gradient = True
    return system, state, sites


def amber_place(x, sites, distance):
    """The places of the extra points (works on complex frames too)."""
    X = x.copy()
    o, h1, h2 = x[sites[:, 1]], x[sites[:, 2]], x[sites[:, 3]]
    norm = lambda a: np.sqrt((a * a).sum(axis=1))[:, None]
    u, v = (h1 - o) / norm(h1 - o), (h2 - o) / norm(h2 - o)
    X[sites[:, 0]] = o + distance * (u + v) / norm(u + v)
    return X


class AmberModel:
    """The part of U that depends on theta: the Lennard-Jones of the pairs
    of oxygens of the waters and the Coulomb of every pair with a particle
    of a water (the other pairs and terms do not change with theta, nor do
    the cotangents)."""

    def __init__(self, system, sites, box, distance):
        top = system.topology
        self.sites, self.box, self.distance = sites, box, distance
        n = top.particle_count
        self.q = np.array(top.charges)
        molecule = np.full(n, -1)
        for w, row in enumerate(sites):
            molecule[row] = w
        i, j = np.triu_indices(n, 1)
        keep = ((molecule[i] >= 0) | (molecule[j] >= 0)) & \
            ~((molecule[i] == molecule[j]) & (molecule[i] >= 0))
        self.i, self.j = i[keep], j[keep]
        oxygen = np.zeros(n, dtype=bool)
        oxygen[sites[:, 1]] = True
        self.oo = oxygen[self.i] & oxygen[self.j]
        self.water = molecule >= 0
        self.roles = [sites[:, 1], np.concatenate([sites[:, 2], sites[:, 3]]), sites[:, 0]]

    def charges(self, theta):
        q = self.q.copy()
        for entry, members in enumerate(self.roles):
            q[members] = theta[2 + entry]
        return q

    def energy_gradient(self, X, theta):
        sigma, epsilon = theta[0], theta[1]
        q = self.charges(theta)
        d = X[self.i] - X[self.j]
        d -= self.box * np.round(d / self.box)
        r = np.sqrt((d * d).sum(axis=1))
        inside = r < AMBER_RC
        r, d = r[inside], d[inside]
        i, j, oo = self.i[inside], self.j[inside], self.oo[inside]
        lj = lambda t: 4 * epsilon * ((sigma / t) ** 12 - (sigma / t) ** 6)
        qq = F * q[i] * q[j]
        energy = (qq * (1 / r - 1 / AMBER_RC)).sum() + (lj(r[oo]) - lj(AMBER_RC)).sum()
        slope = -qq / r ** 2
        slope[oo] += 4 * epsilon * (-12 * sigma ** 12 / r[oo] ** 13 + 6 * sigma ** 6 / r[oo] ** 7)
        force = (slope / r)[:, None] * d
        gradient = np.zeros_like(X)
        np.add.at(gradient, i, force)
        np.add.at(gradient, j, -force)
        return energy, gradient

    def phi(self, x, theta, g, v):
        """sum_n g_n U_n + (J Gbar_n) . dU_n/dX, the part that depends on theta."""
        total = 0.0
        for n in range(len(x)):
            X = amber_place(x[n], self.sites, self.distance)
            tangent = amber_place(x[n] + 1e-30j * v[n], self.sites, self.distance).imag / 1e-30
            energy, gradient = self.energy_gradient(X, theta)
            total += g[n] * energy + (tangent * gradient).sum()
        return total


def amber_theta(evaluator):
    return np.array([evaluator.tunables["sigma_pair"][0], evaluator.tunables["epsilon_pair"][0],
                     *evaluator.tunables["q"]])


def ep_distance(prmtop):
    import parmed
    structure = parmed.load_file(prmtop)
    for bond in structure.bonds:
        if {bond.atom1.name, bond.atom2.name} == {"EPW", "O"}:
            return bond.type.req / 10.0
    raise AssertionError("no bond to the extra point")


def amber_subset(waters, protein):
    """A prmtop of Amber with the first `waters` waters of the fixture, and
    its ACE-ALA-NME if `protein`, written by ParmEd into the work directory."""
    import parmed
    opc = base.root + "/../opc"
    structure = parmed.load_file(opc + "/opc.prmtop", opc + "/opc.inpcrd")
    first = 4 if protein else 1
    if not protein:
        structure.strip("!:WAT")
    structure.strip(f":{first + waters}-99999")
    work.mkdir(parents=True, exist_ok=True)
    name = f"opc-{waters}{'-protein' if protein else ''}"
    structure.save(str(work / f"{name}.prmtop"), overwrite=True)
    structure.save(str(work / f"{name}.inpcrd"), overwrite=True)
    return str(work / f"{name}.prmtop"), str(work / f"{name}.inpcrd"), structure


def clear_frames(state, model, count, clearance):
    """Frames about the state whose pairs with a particle of a water are all
    farther than `clearance` from the cutoff, where the Coulomb force jumps:
    in mixed precision a distance within rounding of the cutoff may fall on
    either side, and its pair adds or drops a jump to the product."""
    for seed in range(726, 826):
        x = base.frames(state, count=count, spread=0.004, seed=seed)
        nearest = np.inf
        for frame in x:
            X = amber_place(frame, model.sites, model.distance)
            d = X[model.i] - X[model.j]
            d -= model.box * np.round(d / model.box)
            nearest = min(nearest, np.abs(np.sqrt((d * d).sum(axis=1)) - AMBER_RC).min())
        if nearest > clearance:
            return x, nearest
    raise AssertionError("no frames clear of the cutoff")


def run_amber():
    prmtop, inpcrd, _ = amber_subset(40, True)
    system, state, sites = amber_fixture(mdir.load_amber(prmtop, inpcrd))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        evaluator = mdir.FrameEvaluator(base.compile_(system, state))
    order = list(evaluator._slices)
    assert order == ["sigma_pair", "epsilon_pair", "q"], order
    theta = amber_theta(evaluator)
    box = np.array(state.cell.diagonal)
    model = AmberModel(system, sites, box, ep_distance(prmtop))
    x, nearest = clear_frames(state, model, 2, 2e-5)
    rng = np.random.default_rng(774)
    g, v = rng.normal(size=len(x)), rng.normal(size=x.shape)
    out = evaluator.evaluate(x)
    product = base.flat(out.vjp(g, positions=v), order)
    reference = np.zeros(len(theta))
    for t in range(len(theta)):
        def central(h):
            e = np.zeros(len(theta))
            e[t] = h
            return (model.phi(x, theta + e, g, v) - model.phi(x, theta - e, g, v)) / (2.0 * h)
        h = 1e-4 * max(abs(theta[t]), 1.0)
        reference[t] = (4.0 * central(0.5 * h) - central(h)) / 3.0
    label = f"Amber OPC (ACE-ALA-NME and {len(sites)} waters, {x.shape[1]} particles)"
    base.check(f"{label}, {len(x)} frames: the product in theta against central differences of "
               "the NumPy Phi", product, reference, 1e-8 if double else 2e-5)
    if double:
        base.check(f"{label}: the same against central differences of the rows along the "
                   "cotangent", product, base.differences(evaluator, x, g, v, 1e-6), 1e-7)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            exact = mdir.FrameEvaluator(base.compile_(system, state, base.target, "Double"))
        base.check(f"{label}: the product in theta in mixed precision against that in double",
                   product, base.flat(exact.evaluate(x).vjp(g, positions=v), order), 1e-5)
    base.expect(mdir.UnsupportedError,
                lambda: out.vjp(g, positions=v, gradients=("tunables", "positions")),
                "virtual sites placed by the directions of their bonds")
    assert np.array_equal(base.flat(out.vjp(g, positions=v), order), product)
    print("Amber OPC: the derivative in the positions is refused by name; the product in "
          "theta stays, to the bit")


def run_placement():
    """On frames of exactly rigid waters, the extra point of Amber and the
    linear site with a = b = d / (2 r_OH cos(theta_HOH / 2)) are one place:
    the waters of the fixture alone, from the prmtop and from ParmEd's
    topology of GROMACS, at the same values of the tunables, give the same
    energy. Their gradients differ, as their spreads do (docs/constraints.md,
    Sections 2.1 and 2.2), and so do the products."""
    import parmed
    prmtop, inpcrd, structure = amber_subset(40, False)
    amber, state, sites = amber_fixture(mdir.load_amber(prmtop, inpcrd))
    lengths = {}
    for bond in structure.bonds:
        lengths[frozenset((bond.atom1.name, bond.atom2.name))] = bond.type.req / 10.0
    doh, dhh = lengths[frozenset(("O", "H1"))], lengths[frozenset(("H1", "H2"))]
    half = np.arcsin(dhh / 2 / doh)
    a = ep_distance(prmtop) / (2 * doh * np.cos(half))
    structure.save(str(work / "opc.top"), format="gromacs", overwrite=True)
    structure.save(str(work / "opc.gro"), overwrite=True)
    text = (work / "opc.top").read_text().split("\n")
    at = [k for k, line in enumerate(text) if line.startswith("[ virtual_sites3 ]")][0]
    row = [k for k in range(at + 1, len(text)) if text[k].strip() and not text[k].startswith(";")][0]
    assert text[row].split()[:5] == ["4", "1", "2", "3", "1"], text[row]
    text[row] = f"4 1 2 3 1 {a:.17g} {a:.17g}"
    (work / "opc.top").write_text("\n".join(text))
    lines = (work / "opc.gro").read_text().rstrip("\n").split("\n")
    lines[-1] = " ".join(f"{edge:.10f}" for edge in np.array(state.cell.diagonal))
    (work / "opc.gro").write_text("\n".join(lines) + "\n")
    gromacs, gstate, gsites = amber_fixture(
        mdir.load_gromacs(str(work / "opc.top"), str(work / "opc.gro")))
    gromacs.rigid_water = True
    assert np.array_equal(gsites, sites)
    # Rigid waters at the geometry of the model, each turned and moved.
    rng = np.random.default_rng(775)
    x0 = np.array(state.positions)
    x = np.repeat(x0[None], 2, axis=0)
    for n in range(len(x)):
        for site, o, h1, h2 in sites:
            q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
            arm1 = doh * np.array([np.sin(half), np.cos(half), 0.0])
            arm2 = doh * np.array([-np.sin(half), np.cos(half), 0.0])
            centre = x0[o] + rng.normal(scale=0.01, size=3)
            x[n, o], x[n, h1], x[n, h2] = centre, centre + q @ arm1, centre + q @ arm2
    energies, theta = [], None
    for system, st in ((amber, state), (gromacs, gstate)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            evaluator = mdir.FrameEvaluator(base.compile_(system, st))
        names = list(evaluator._slices)
        if theta is None:
            theta = {name: np.array(evaluator.tunables[name]) for name in names}
        else:
            evaluator.tunables.update(theta)
        energies.append(evaluator.evaluate(x).energy)
    base.check(f"OPC from Amber and from GROMACS ({len(sites)} waters, a = {a:.8f}), "
               f"{len(x)} frames of rigid waters: the energy, through the places of the sites",
               energies[0], energies[1], 1e-12 if double else 1e-5)


if scenario == "pair":
    run(2, 4.0, "two OPC waters")
elif scenario == "box":
    run(8, 2.0, "eight OPC waters in 2.0 nm")
elif scenario == "amber":
    run_amber()
elif scenario == "placement":
    run_placement()
else:
    raise SystemExit(f"unknown scenario {scenario!r}")
