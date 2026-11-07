"""The product of force matching in the frame evaluator
(D381, #726): `FrameEnergies.vjp(cotangent, positions=)`,

  p = sum_n [ g_n dU_n/dtheta + sum_i Gbar_n,i . d2U_n/dx_i dtheta ],

against independent references and against central differences of the
existing derivative in the tunables (D230) along the cotangent.

Usage: python_frame_product.py ROOT SCENARIO [TARGET PRECISION [WORK]]

Scenarios:
  dihedral  the force constants of proper dihedrals (D372) on the torsion
            fixtures: against the closed form, differentiated along the
            cotangent with dual numbers in NumPy, and against differences
  lj        propane and water under a plain cutoff with the correction for
            the dispersion and a Coulomb cutoff: per-type (geometric) and
            pair sigma and epsilon with a pair set apart, against a NumPy
            sum of the shifted Lennard-Jones and its tangent, and against
            differences; in mixed precision, against the product in double
  charges   the charges under a Coulomb cutoff on the dipeptide in water,
            tied by a map: against the NumPy sum of the shifted Coulomb and
            the 1-4 pairs and its tangent, and against differences
  refusals  what is refused, and that the evaluator stays usable (CPU)
"""
import sys
import time
from pathlib import Path

import numpy as np
import mdir

root, scenario = sys.argv[1], sys.argv[2]
target = sys.argv[3] if len(sys.argv) > 3 else "CPU"
precision = sys.argv[4] if len(sys.argv) > 4 else "Double"
work = Path(sys.argv[5]) if len(sys.argv) > 5 else Path("work")
double = precision == "Double"
mode = f"{target} {precision}"


def compile_(system, state, target=target, precision=precision):
    integrator, execution = mdir.Integrator(), mdir.Execution()
    integrator.timestep = 0.0005
    execution.target = getattr(mdir.Target, target)
    execution.precision = getattr(mdir.Precision, precision)
    execution.deterministic = True
    return mdir.compile(system, state, integrator, mdir.Ensemble(), execution, mdir.Schedule())


def frames(state, count=3, spread=0.004, seed=726):
    """Frames about the state: its positions displaced at random."""
    rng = np.random.default_rng(seed)
    x = np.array(state.positions)
    return x[None] + spread * rng.normal(size=(count,) + x.shape)


def cotangents(x, seed=727):
    rng = np.random.default_rng(seed)
    return rng.normal(size=len(x)), rng.normal(size=x.shape)


def product(evaluator, x, g, v):
    """The product under test, a dict of the names of the tunables to their
    entries: FrameEnergies.vjp here; python_frame_product_torch.py puts the
    backward of mdir.torch in its place (D382)."""
    out = evaluator.evaluate(x, gradients=("tunables", "positions"))
    return {name: np.asarray(value) for name, value in out.vjp(g, positions=v).items()}


def flat(gradient, names):
    return np.concatenate([np.asarray(gradient[name]) for name in names])


def differences(evaluator, x, g, v, step):
    """g . rows + the central differences of the rows of the Jacobian along
    v, Richardson-extrapolated from the steps `step` and `step`/2: the
    product that the existing derivative in the tunables implies."""
    def rows(y):
        out = evaluator.evaluate(y)
        return np.concatenate([out.jacobian[name] for name in evaluator._slices], axis=1)
    base = np.einsum("n,nm->m", g, rows(x))

    def central(h):
        return ((rows(x + h * v) - rows(x - h * v)) / (2.0 * h)).sum(axis=0)
    return base + (4.0 * central(0.5 * step) - central(step)) / 3.0


def relative(a, b):
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


def check(what, value, reference, tolerance):
    error = relative(value, reference)
    print(f"{mode}: {what}, {error:.1e} of the largest entry (tolerance {tolerance:.0e})")
    assert error <= tolerance, (what, error)
    return error


def expect(error, call, text):
    try:
        call()
    except error as raised:
        assert text in str(raised), str(raised)
        return str(raised)
    raise AssertionError(f"expected {error.__name__} containing {text!r}")


# --- dihedral --------------------------------------------------------------

def torsion_fixture():
    work.mkdir(parents=True, exist_ok=True)
    rows = ["1 2 3 4 1 20 1.5 1", "1 3 2 4 4 180 2.25 2",
            "1 2 3 4 9 -40 0.75 3", "2 3 4 5 1 60 1.5 2",
            "1 2 3 4 1 20 1.5 1"]
    (work / "system.top").write_text(
        "[ defaults ]\n1 2 no 0.5 0.8333333333333333\n"
        "[ atomtypes ]\nC 6 12.0 0.0 A 0.0 0.0\n"
        "[ moleculetype ]\nTORSION 0\n[ atoms ]\n" +
        "".join(f"{i} C 1 TOR C{i} {i} 0.0 12.0\n" for i in range(1, 6)) +
        "[ dihedrals ]\n" + "\n".join(rows) +
        "\n[ system ]\ntorsion fixtures\n[ molecules ]\nTORSION 1\n")
    positions = [[1., 1., 1.], [1.2, 1.1, 1.], [1.3, 1.2, 1.25],
                 [1.5, 1.1, 1.4], [1.65, 1.3, 1.5]]
    (work / "system.gro").write_text("torsion fixtures\n5\n" + "".join(
        f"{1:5d}{'TOR':<5}{'C'+str(i):>5}{i:5d}"
        f"{x:8.3f}{y:8.3f}{z:8.3f}\n"
        for i, (x, y, z) in enumerate(positions, 1)) + "3.0 3.0 3.0\n")
    loaded = mdir.load_gromacs(str(work / "system.top"), str(work / "system.gro"))
    system, state = loaded.make_system(), loaded.make_state()
    system.cutoff, system.pairlist_distance = 0.8, 0.9
    system.dispersion = mdir.DispersionCorrection.None_
    return system, state


class Dual:
    """A number and its derivative along one direction."""

    def __init__(self, value, derivative=0.0):
        self.value, self.derivative = value, derivative

    @staticmethod
    def of(value):
        return value if isinstance(value, Dual) else Dual(value)

    def __add__(self, other):
        other = Dual.of(other)
        return Dual(self.value + other.value, self.derivative + other.derivative)

    __radd__ = __add__

    def __neg__(self):
        return Dual(-self.value, -self.derivative)

    def __sub__(self, other):
        return self + -Dual.of(other)

    def __mul__(self, other):
        other = Dual.of(other)
        return Dual(self.value * other.value,
                    self.derivative * other.value + self.value * other.derivative)

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = Dual.of(other)
        return Dual(self.value / other.value,
                    (self.derivative * other.value - self.value * other.derivative) /
                    other.value ** 2)

    def sqrt(self):
        value = np.sqrt(self.value)
        return Dual(value, self.derivative / (2.0 * value))

    def cos(self):
        return Dual(np.cos(self.value), -np.sin(self.value) * self.derivative)


def torsion_basis(positions, direction, top):
    """1 + cos(n phi - delta) of every row, and its derivative along
    `direction` of the positions."""
    x = [[Dual(p, d) for p, d in zip(row, along)] for row, along in zip(positions, direction)]
    dot = lambda a, b: sum(u * w for u, w in zip(a, b))
    sub = lambda a, b: [u - w for u, w in zip(a, b)]
    cross = lambda a, b: [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
                          a[0] * b[1] - a[1] * b[0]]
    values, slopes = [], []
    for members, n, phase in zip(top.dihedrals, top.dihedral_periodicities, top.dihedral_phases):
        a, b, c, d = [x[i] for i in members]
        b0, b1, b2 = sub(a, b), sub(c, b), sub(d, c)
        unit = [w / dot(b1, b1).sqrt() for w in b1]
        v = sub(b0, [dot(b0, unit) * u for u in unit])
        w = sub(b2, [dot(b2, unit) * u for u in unit])
        sine, cosine = dot(cross(unit, v), w), dot(v, w)
        phi = Dual(np.arctan2(sine.value, cosine.value),
                   (cosine.value * sine.derivative - sine.value * cosine.derivative) /
                   (sine.value ** 2 + cosine.value ** 2))
        term = 1.0 + (float(n) * phi - float(phase)).cos()
        values.append(term.value)
        slopes.append(term.derivative)
    return np.array(values), np.array(slopes)


def run_dihedral():
    system, state = torsion_fixture()
    system.tunables = [mdir.Tunable("k", "dihedral_k")]
    system.tunable_gradient = True
    program = compile_(system, state)
    evaluator = mdir.FrameEvaluator(program)
    x = frames(state, spread=0.02)
    g, v = cotangents(x)
    out = evaluator.evaluate(x, gradients=("tunables", "positions"))
    p = product(evaluator, x, g, v)["k"]
    entry = np.asarray(program.plan["tunables"][0]["map"])
    top = system.topology
    reference = np.zeros(len(p))
    for n in range(len(x)):
        values, slopes = torsion_basis(x[n], v[n], top)
        for row, at in enumerate(entry):
            if at >= 0:
                reference[at] += g[n] * values[row] + slopes[row]
    tolerance = 1e-10 if double else 2e-5
    check(f"dihedral_k, {len(x)} frames, against the closed form along the cotangent (dual "
          "numbers)", p, reference, tolerance)
    if double:
        check("dihedral_k against central differences of the rows along the cotangent",
              p, differences(evaluator, x, g, v, 1e-4), 1e-7)
    # The energy's term alone is the product with the rows of before.
    alone = out.vjp(g)["k"]
    zero = out.vjp(g, positions=np.zeros_like(v))["k"]
    check("with a zero cotangent on the positions, against vjp(g) with the Jacobian", zero,
          alone, 1e-13 if double else 1e-6)


# --- lj --------------------------------------------------------------------

def propane_excluded(n):
    """The pairs of the GROMACS model that its topology excludes: up to
    three bonds apart in propane (1-4 pairs included, which have parameters
    of their own), and within a water."""
    text = open(root + "/../gromacs/system.top").read()
    bonds, counts, section = [], {}, None
    for line in text.splitlines():
        line = line.split(";")[0].strip()
        if line.startswith("["):
            section = line.strip("[] ")
            continue
        if section == "bonds" and line:
            bonds.append(tuple(int(x) - 1 for x in line.split()[:2]))
        if section == "molecules" and line:
            counts[line.split()[0]] = int(line.split()[1])
    n_pro = 11
    graph = {i: set() for i in range(n_pro)}
    for i, j in bonds:
        graph[i].add(j)
        graph[j].add(i)
    excluded = set()
    for m in range(counts["PRO"]):
        o = n_pro * m
        for i in range(n_pro):
            seen, frontier = {i}, {i}
            for _ in range(3):
                frontier = {k for f in frontier for k in graph[f]} - seen
                seen |= frontier
            excluded |= {(o + min(i, j), o + max(i, j)) for j in seen if j != i}
    for w in range(counts["SOL"]):
        o = n_pro * counts["PRO"] + 3 * w
        excluded |= {(o, o + 1), (o, o + 2), (o + 1, o + 2)}
    assert n_pro * counts["PRO"] + 3 * counts["SOL"] == n
    return excluded


def propane(tunables, gradient=True):
    gromacs = root + "/../gromacs"
    loaded = mdir.load_gromacs(gromacs + "/system.top", gromacs + "/system.gro",
                               defines=["FLEXIBLE"])
    system, state = loaded.make_system(), loaded.make_state()
    system.cutoff, system.pairlist_distance, system.switch_distance = 0.8, 0.9, 0.8
    system.truncation = mdir.Truncation.None_
    system.electrostatics = mdir.Electrostatics.Cutoff
    system.dispersion = mdir.DispersionCorrection.EnergyPressure
    system.tunables = tunables(system)
    system.tunable_gradient = gradient
    return system, state


def run_lj():
    rc = 0.8
    import warnings
    system, state = propane(lambda s: [mdir.Tunable("s", "sigma_pair"),
                                       mdir.Tunable("e", "epsilon_pair")], gradient=False)
    top = system.topology
    names, pairs, types = list(top.type_names), top.type_pairs, np.asarray(top.particle_types)
    count = len(names)
    model = mdir.Simulation(compile_(system, state, "CPU", "Double"))
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
        system, state = propane(tunables)
        program = compile_(system, state)
    evaluator = mdir.FrameEvaluator(program)
    order = list(evaluator._slices)
    x = frames(state)
    g, v = cotangents(x)
    out = evaluator.evaluate(x, gradients=("tunables", "positions"))
    p = flat(product(evaluator, x, g, v), order)
    # Once compiled, a pass of the product program over the frames.
    started = time.perf_counter()
    out.vjp(g, positions=v)
    product_seconds = time.perf_counter() - started
    box = np.array(state.cell.diagonal)
    n, volume = x.shape[1], float(np.prod(box))
    excluded = propane_excluded(n)
    theta0 = {name: np.array(evaluator.tunables[name]) for name in order}

    def sums(y, w):
        # Over the pairs not excluded within the cutoff of each pair of
        # types: r^-12 - rc^-12, r^-6 - rc^-6, and their derivatives along w.
        s12, s6, t12, t6 = (np.zeros(len(pairs)) for _ in range(4))
        for i in range(n - 1):
            d = y[i + 1:] - y[i]
            d -= box * np.round(d / box)
            r = np.sqrt((d * d).sum(axis=1))
            rate = (d * (w[i + 1:] - w[i])).sum(axis=1) / r
            for j in np.nonzero(r < rc)[0]:
                if (i, i + 1 + j) in excluded:
                    continue
                a, b = sorted((types[i], types[i + 1 + j]))
                k = site[a, b]
                s12[k] += r[j] ** -12 - rc ** -12
                s6[k] += r[j] ** -6 - rc ** -6
                t12[k] += -12.0 * r[j] ** -13 * rate[j]
                t6[k] += -6.0 * r[j] ** -7 * rate[j]
        return s12, s6, t12, t6

    ordered = np.zeros(len(pairs))
    numbers = np.bincount(types, minlength=count).astype(float)
    for (a, b), k in site.items():
        ordered[k] = numbers[a] * (numbers[b] - 1.0) if a == b else 2.0 * numbers[a] * numbers[b]
    for i, j in excluded:
        a, b = sorted((types[i], types[j]))
        ordered[site[a, b]] -= 2.0
    total = n * (n - 1.0) - 2.0 * len(excluded)
    frame_sums = [sums(x[k], v[k]) for k in range(len(x))]

    def table(theta):
        sigma_type, epsilon_type = sigma0[diagonal].copy(), epsilon0[diagonal].copy()
        sigma_type[keep >= 0] = theta["sigma"][keep[keep >= 0]]
        epsilon_type[keep >= 0] = theta["epsilon"][keep[keep >= 0]]
        sigma, epsilon = sigma0.copy(), epsilon0.copy()
        for (a, b), k in site.items():
            if k != nbfix:
                sigma[k] = np.sqrt(sigma_type[a] * sigma_type[b])
                epsilon[k] = np.sqrt(epsilon_type[a] * epsilon_type[b])
        sigma[taken_sigma >= 0] = theta["sigma_pair"][taken_sigma[taken_sigma >= 0]]
        epsilon[taken_epsilon >= 0] = theta["epsilon_pair"][taken_epsilon[taken_epsilon >= 0]]
        return sigma, epsilon

    def objective(theta):
        # sum_n g_n U_n + Phi_n: the shifted Lennard-Jones with the
        # correction and its shift estimate (D210), and its tangent, to
        # which the correction, which reads no position, adds nothing.
        sigma, epsilon = table(theta)
        value = 0.0
        for k, (s12, s6, t12, t6) in enumerate(frame_sums):
            within = (4.0 * epsilon * (sigma ** 12 * s12 - sigma ** 6 * s6)).sum()
            mean = (ordered * 4.0 * epsilon * sigma ** 6).sum() / total
            correction = -2.0 * np.pi / (3.0 * volume) * n * n * mean / rc ** 3
            energy = within + correction * (2.0 - volume / (n * 4.0 * np.pi / 3.0 * rc ** 3))
            tangent = (4.0 * epsilon * (sigma ** 12 * t12 - sigma ** 6 * t6)).sum()
            value += g[k] * energy + tangent
        return value

    reference = []
    for name in order:
        for m in range(len(theta0[name])):
            def central(h):
                e = []
                for sign in (1.0, -1.0):
                    theta = {key: value.copy() for key, value in theta0.items()}
                    theta[name][m] += sign * h
                    e.append(objective(theta))
                return (e[0] - e[1]) / (2.0 * h)
            h = 1e-3 * theta0[name][m]
            reference.append((4.0 * central(0.5 * h) - central(h)) / 3.0)
    reference = np.array(reference)
    tolerance = 1e-8 if double else 2e-5
    check(f"propane and water, per-type and pair sigma and epsilon with a pair set apart, "
          f"{len(x)} frames, against the NumPy sum of the shifted Lennard-Jones and its tangent",
          p, reference, tolerance)
    if double:
        check("the same against central differences of the rows along the cotangent", p,
              differences(evaluator, x, g, v, 1e-4), 1e-6)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reference_program = compile_(system, state, target, "Double")
        reference_evaluator = mdir.FrameEvaluator(reference_program)
        exact = flat(reference_evaluator.evaluate(x).vjp(g, positions=v), order)
        check("the product in mixed precision against that in double", p, exact, 1e-5)
    started = time.perf_counter()
    evaluator.evaluate(x, gradients=("tunables", "positions"))
    evaluate_seconds = time.perf_counter() - started
    print(f"{mode}: {len(x)} frames of {n} atoms, the evaluation with the rows and the "
          f"gradient in the positions {1e3 * evaluate_seconds / len(x):.2f} ms a frame, the "
          f"product {1e3 * product_seconds / len(x):.2f} ms a frame")


# --- charges ---------------------------------------------------------------

def run_charges():
    rc, f = 0.8, 138.935457644
    loaded = mdir.load_amber(root + "/dipeptide.prmtop", root + "/dipeptide.inpcrd")
    system, state = loaded.make_system(), loaded.make_state()
    system.truncation = mdir.Truncation.Switch
    system.cutoff, system.pairlist_distance, system.switch_distance = rc, 0.9, 0.7
    system.electrostatics = mdir.Electrostatics.Cutoff
    system.dispersion = mdir.DispersionCorrection.None_
    top = system.topology
    names = top.atom_names
    residues = [top.residue_names[r] for r in top.residue_indices]
    entries, charge_map = {}, []
    for i, (atom, residue) in enumerate(zip(names, residues)):
        key = (residue, atom) if residue == "WAT" else (residue, i)
        charge_map.append(entries.setdefault(key, len(entries)))
    charge_map = np.array(charge_map, dtype=np.int64)
    system.tunables = [mdir.Tunable("q", "charge", map=charge_map)]
    system.tunable_gradient = True
    program = compile_(system, state)
    evaluator = mdir.FrameEvaluator(program)
    x = frames(state, count=2)
    g, v = cotangents(x)
    p = product(evaluator, x, g, v)["q"]
    q = np.asarray(evaluator.tunables["q"])[charge_map]
    box = np.array(state.cell.diagonal)
    n = x.shape[1]
    # The pairs up to two bonds apart are excluded; those three bonds apart
    # are the 1-4 pairs, f q_i q_j / (1.2 r); the others take
    # f q_i q_j (1 / r - 1 / rc) within the cutoff (D210).
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
    per_particle = np.zeros(n)
    for k in range(len(x)):
        y, w = x[k], v[k]
        for i in range(n - 1):
            d = y[i + 1:] - y[i]
            d -= box * np.round(d / box)
            r = np.sqrt((d * d).sum(axis=1))
            rate = (d * (w[i + 1:] - w[i])).sum(axis=1) / r
            for j in np.nonzero(r < rc)[0]:
                pair = (i, i + 1 + int(j))
                if pair in excluded or pair in scaled:
                    continue
                kernel = g[k] * f * (1.0 / r[j] - 1.0 / rc) - f / r[j] ** 2 * rate[j]
                per_particle[i] += kernel * q[pair[1]]
                per_particle[pair[1]] += kernel * q[i]
        for i, j in scaled:
            d = y[j] - y[i]
            d -= box * np.round(d / box)
            r = np.sqrt((d * d).sum())
            rate = (d * (w[j] - w[i])).sum() / r
            kernel = g[k] * f / (1.2 * r) - f / (1.2 * r ** 2) * rate
            per_particle[i] += kernel * q[j]
            per_particle[j] += kernel * q[i]
    reference = np.zeros(len(p))
    np.add.at(reference, charge_map, per_particle)
    tolerance = 1e-9 if double else 2e-5
    check(f"the charges under a Coulomb cutoff, {len(p)} entries for {n} particles, "
          f"{len(x)} frames, against the NumPy sum of the shifted Coulomb and the 1-4 pairs and "
          "its tangent", p, reference, tolerance)
    if double:
        # The force of the shifted Coulomb jumps at the cutoff, and so does
        # the derivative of a row in the positions: a step small enough that
        # no pair crosses the cutoff (1e-4 gives 1.7e-2 and 1e-5 7.7e-4 from
        # the pairs that do, measured).
        check("the same against central differences of the rows along the cotangent, at a "
              "step that carries no pair across the cutoff", p,
              differences(evaluator, x, g, v, 1e-6), 1e-7)


# --- refusals --------------------------------------------------------------

def run_refusals():
    system, state = torsion_fixture()
    system.tunables = [mdir.Tunable("k", "dihedral_k")]
    system.tunable_gradient = True
    evaluator = mdir.FrameEvaluator(compile_(system, state))
    x = frames(state, spread=0.02)
    g, v = cotangents(x)
    out = evaluator.evaluate(x, gradients=("tunables", "positions"))
    first = out.vjp(g, positions=v)["k"].copy()
    expect(mdir.InputError, lambda: out.vjp(), "give a cotangent")
    expect(mdir.InputError, lambda: out.vjp(positions=v[:, :-1]), "of shape")
    bad = v.copy()
    bad[1, 2, 0] = np.nan
    expect(mdir.InputError, lambda: out.vjp(positions=bad), "not finite")
    once = evaluator.evaluate(iter(list(x)))
    expect(mdir.InputError, lambda: once.vjp(positions=v), "an iterator gives them once")
    values = np.array(evaluator.tunables["k"])
    evaluator.tunables.update({"k": values * 1.5})
    expect(mdir.SimulationError, lambda: out.vjp(positions=v), "changed since")
    evaluator.tunables.update({"k": values})
    # The product program takes the values of the evaluator, which changed
    # and came back: the same product, to the bit.
    assert np.array_equal(evaluator.evaluate(x).vjp(g, positions=v)["k"], first)
    print("refusals: no cotangent, a shape, a value that is not finite, frames given once, and "
          "values changed since the frames were evaluated are refused; the evaluator gives the "
          "same product after them, to the bit")

    # What the product does not form yet, refused when it is asked.
    opc = root + "/../opc"
    loaded = mdir.load_amber(opc + "/opc.prmtop", opc + "/opc.inpcrd")
    system, state = loaded.make_system(), loaded.make_state()
    system.cutoff, system.pairlist_distance = 0.8, 0.9
    system.electrostatics = mdir.Electrostatics.Cutoff
    system.dispersion = mdir.DispersionCorrection.None_
    system.tunables = [mdir.Tunable("sigma", "sigma_pair")]
    system.tunable_gradient = True
    sites = mdir.FrameEvaluator(compile_(system, state))
    y = np.array(state.positions)[None]
    out = sites.evaluate(y)
    text = expect(mdir.UnsupportedError, lambda: out.vjp(positions=np.ones_like(y)),
                  "virtual sites")
    assert np.isfinite(sites.evaluate(y).energy).all()
    print(f"refusals: virtual sites: {text.split(': ', 1)[1][:110]}")

    loaded = mdir.load_amber(root + "/dipeptide.prmtop", root + "/dipeptide.inpcrd")
    system, state = loaded.make_system(), loaded.make_state()
    system.cutoff, system.pairlist_distance = 0.8, 0.9
    system.electrostatics = mdir.Electrostatics.PME
    system.dispersion = mdir.DispersionCorrection.None_
    system.tunables = [mdir.Tunable("q", "charge")]
    system.tunable_gradient = True
    mesh = mdir.FrameEvaluator(compile_(system, state))
    y = np.array(state.positions)[None]
    out = mesh.evaluate(y)
    expect(mdir.UnsupportedError, lambda: out.vjp(positions=np.ones_like(y)), "not formed")
    print("refusals: the charges with particle mesh Ewald are refused, never a zero")


if __name__ == "__main__":
    {"dihedral": run_dihedral, "lj": run_lj, "charges": run_charges,
     "refusals": run_refusals}[scenario]()
