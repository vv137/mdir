"""D372: row identity, atomic updates and independent physics.

The oracle differentiates a projected-vector atan2 torsion with forward-mode
NumPy dual numbers. It neither calls MDIR for reference values nor differences
MDIR's forces. Coefficient differences separately check the reported energy.
"""
import json
from pathlib import Path
import sys

import mdir
import numpy as np


def expect(call, text):
    try:
        call()
    except mdir.InputError as error:
        assert text in str(error), str(error)
        return
    raise AssertionError(f"expected InputError containing {text!r}")


def fixture(work, only_improper=False):
    work.mkdir(parents=True, exist_ok=True)
    rows = ["1 2 3 4 1 20 1.5 1", "1 3 2 4 4 180 2.25 2",
            "1 2 3 4 9 -40 0.75 3", "2 3 4 5 1 60 1.5 2",
            "1 2 3 4 1 20 1.5 1"]
    if only_improper:
        rows = rows[1:2]
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
    return mdir.load_gromacs(str(work / "system.top"), str(work / "system.gro"))


def model(loaded, declarations):
    system, state = loaded.make_system(), loaded.make_state()
    system.cutoff, system.pairlist_distance = 0.8, 0.9
    system.dispersion = mdir.DispersionCorrection.None_
    system.tunables = declarations
    system.tunable_gradient = bool(declarations)
    return system, state


def compile_(system, state, target="CPU", precision="Double"):
    execution = mdir.Execution()
    execution.target = getattr(mdir.Target, target)
    execution.precision = getattr(mdir.Precision, precision)
    execution.deterministic, execution.reorder, execution.fast_math = True, False, False
    integrator, ensemble = mdir.Integrator(), mdir.Ensemble()
    integrator.timestep, ensemble.com_period = 0.0001, 0
    return mdir.compile(system, state, integrator, ensemble, execution, mdir.Schedule())


def declarations(loaded):
    system, state = model(loaded, [mdir.Tunable("k", "dihedral_k")])
    top = system.topology
    np.testing.assert_array_equal(top.dihedral_periodicities, [1, 2, 3, 2, 1])
    np.testing.assert_allclose(top.dihedral_force_constants, [1.5, 2.25, 0.75, 1.5, 1.5], rtol=0, atol=0)
    np.testing.assert_allclose(top.dihedral_phases, np.deg2rad([20, 180, -40, 60, 20]), rtol=0, atol=1e-15)
    np.testing.assert_array_equal(top.improper_dihedrals, [False, True, False, False, False])
    for name, dtype in [("dihedral_force_constants", np.float64),
                        ("dihedral_periodicities", np.int64), ("dihedral_phases", np.float64)]:
        value = getattr(top, name)
        assert value.shape == (5,) and value.dtype == dtype and not value.flags.writeable
        try:
            value[0] = 99
        except ValueError:
            pass
        else:
            raise AssertionError("topology metadata was writable")
    program = compile_(system, state)
    np.testing.assert_array_equal(program.plan["tunables"][0]["map"], [0, -1, 1, 2, 3])
    assert program.plan["tunables"][0]["unit"] == "kJ/mol"
    assert program.plan["tunables"][0]["gradient"] == "rule"
    assert program.topology.dihedral_force_constants.tolist() == top.dihedral_force_constants.tolist()

    def refused(tunable, text):
        system.tunables = tunable if isinstance(tunable, list) else [tunable]
        expect(lambda: compile_(system, state), text)

    T = lambda **kw: mdir.Tunable("k", "dihedral_k", **kw)
    cases = [(T(map=np.array([0, 1, 2, 3, 4])), "improper row 1"),
             (T(map=np.array([0, -1, 1])), "map has 3 entries"),
             (T(map=np.array([], dtype=np.int64)), "map has 0 entries"),
             (T(map=np.full(5, -1)), "map takes no site"),
             (T(map=np.array([-2, -1, 0, 1, 2])), "below -1"),
             (T(map=np.array([0, -1, 2, 2, -1])), "no site takes the entry 1"),
             (T(map=np.array([0, -1, 0, -1, -1])), "differ in the model"),
             (T(values=np.empty(0)), "`values` has 0 entries"),
             (T(values=np.ones(3)), "`values` has 3 entries"),
             (T(term="other"), "not `term`"),
             ([T(), mdir.Tunable("other", "dihedral_k")], "declared by another tunable")]
    for tunable, text in cases:
        refused(tunable, text)
    for bad in [np.nan, np.inf, -np.inf]:
        expect(lambda: T(values=np.array([bad, 1., 1., 1.])), "finite")
    expect(lambda: T(mixing="arithmetic"), "sigma")
    system.tunables = [T(map=np.array([0, -1, 0, 0, -1]), values=np.array([-1.25]))]
    assert compile_(system, state).plan["tunables"][0]["entries"] == 1
    system.tunables = [T(values=np.array([-1.5, 0., 2., -3.]))]
    compile_(system, state)
    other = fixture(work / "improper-only", only_improper=True)
    s, x = model(other, [T()])
    expect(lambda: compile_(s, x), "at least one proper periodic dihedral")
    print("dihedral declarations: row order, copied read-only metadata, signed values and map refusals passed")


class Dual:
    def __init__(self, value, derivative=0.):
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
                    (self.derivative * other.value - self.value * other.derivative) / other.value**2)

    def sqrt(self):
        value = np.sqrt(self.value)
        return Dual(value, self.derivative / (2 * value))

    def cos(self):
        return Dual(np.cos(self.value), -np.sin(self.value) * self.derivative)


def oracle(positions, top, coefficients):
    size = positions.size
    identity = np.eye(size).reshape(positions.shape + (size,))
    x = [[Dual(value, d) for value, d in zip(row, derivatives)]
         for row, derivatives in zip(positions, identity)]
    dot = lambda a, b: sum(u * v for u, v in zip(a, b))
    sub = lambda a, b: [u - v for u, v in zip(a, b)]
    cross = lambda a, b: [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
    energy, basis = Dual(0., np.zeros(size)), []
    for members, n, phase, k in zip(top.dihedrals, top.dihedral_periodicities,
                                     top.dihedral_phases, coefficients):
        a, b, c, d = [x[i] for i in members]
        b0, b1, b2 = sub(a, b), sub(c, b), sub(d, c)
        norm = dot(b1, b1).sqrt()
        unit = [v / norm for v in b1]
        v = sub(b0, [dot(b0, unit) * u for u in unit])
        w = sub(b2, [dot(b2, unit) * u for u in unit])
        sine, cosine = dot(cross(unit, v), w), dot(v, w)
        phi = Dual(np.arctan2(sine.value, cosine.value),
                   (cosine.value * sine.derivative - sine.value * cosine.derivative) /
                   (sine.value**2 + cosine.value**2))
        contribution = 1 + (float(n) * phi - float(phase)).cos()
        basis.append(contribution.value)
        energy += float(k) * contribution
    return energy.value, -energy.derivative.reshape(positions.shape), np.array(basis)


def physics(loaded, target, precision):
    mapping = np.array([0, -1, 1, 0, -1])
    values = np.array([1.5, -0.75])

    def make(theta, together=False):
        tunables = [mdir.Tunable("k", "dihedral_k", map=mapping, values=theta)]
        system, state = model(loaded, tunables)
        if together:
            term = mdir.TupleTerm()
            term.name, term.arity, term.expression = "spring", 2, "0.5*k*(r-r0)^2"
            term.particles = np.array([[0, 1], [1, 2]], dtype=np.int64)
            term.parameters = [("k", np.array([1., 2.])), ("r0", np.array([.2, .3]))]
            system.tuple_terms = [term]
            system.tunables = tunables + [mdir.Tunable("spring_k", "k", term="spring"),
                                         mdir.Tunable("q", "charge")]
        return system, state

    system, state = make(values)
    program = compile_(system, state, target, precision)
    sim = mdir.Simulation(program)
    sim.part_seconds = 1e9
    sim.run(0, energy=True)
    sim.tunables.gradient()  # Establish the gradient's compiled image before counting updates.
    stats = sim.compile_stats.copy()
    top = system.topology
    tolerance = (1e-10, 1e-9) if precision == "Double" else (2e-5, 2e-5)
    atol, rtol = tolerance
    worst = {"energy": 0., "force": 0., "gradient": 0., "finite_difference": 0.}
    records = []
    for theta in [values, np.array([0., -1.25]), np.array([-2., .5])]:
        sim.tunables.update({"k": theta})
        g = sim.tunables.gradient()
        sampled = sim.state()
        coefficients = top.dihedral_force_constants.copy()
        coefficients[mapping >= 0] = theta[mapping[mapping >= 0]]
        energy, force, basis = oracle(np.asarray(sampled.positions, dtype=np.float64), top, coefficients)
        derivative = np.array([basis[mapping == i].sum() for i in range(2)])
        for label, found, reference in [("energy", g.energy, energy), ("force", sampled.forces, force),
                                        ("gradient", g["k"], derivative)]:
            error = float(np.max(np.abs(np.asarray(found) - reference)))
            worst[label] = max(worst[label], error)
            np.testing.assert_allclose(found, reference, atol=atol, rtol=rtol, err_msg=label)
        records.append({"k": theta.tolist(), "energy_reference": energy,
                        "gradient_reference": derivative.tolist()})
        other_system, other_state = make(theta)
        fresh = mdir.Simulation(compile_(other_system, other_state, target, precision))
        fresh.run(0, energy=True)
        other = fresh.tunables.gradient()
        assert g.energy == other.energy and np.array_equal(g["k"], other["k"])
        np.testing.assert_array_equal(sampled.forces, fresh.state().forces)
        for slot in range(2):
            for h in [1., .5, .25]:
                energies = []
                for sign in [1., -1.]:
                    perturbed = theta.copy()
                    perturbed[slot] += sign * h
                    sim.tunables.update({"k": perturbed})
                    energies.append(sim.tunables.gradient().energy)
                fd = (energies[0] - energies[1]) / (2 * h)
                worst["finite_difference"] = max(worst["finite_difference"], abs(fd - derivative[slot]))
                np.testing.assert_allclose(fd, derivative[slot], atol=atol, rtol=rtol)
        sim.tunables.update({"k": theta})
    assert sim.compile_stats == stats
    before = sim.tunables.snapshot()
    before_forces = np.array(sim.state().forces)
    for invalid in [np.array([np.nan, 1.]), np.array([np.inf, 1.]), np.ones(3)]:
        expect(lambda: sim.tunables.update({"k": invalid}), "finite" if invalid.size == 2 else "shape")
        now = sim.tunables.snapshot()
        assert now[0]["k"].tolist() == before[0]["k"].tolist() and now[1:] == before[1:]
        np.testing.assert_array_equal(before_forces, sim.state().forces)
    assert sim._tunable_builds[3] == 1 and sim._tunable_builds[4] > 0, sim._tunable_builds

    # Full and dependent frame paths must both retain the bonded gradient.
    for terms in ["all", "dependent"]:
        evaluator = mdir.FrameEvaluator(program, terms=terms)
        if terms == "dependent":
            assert evaluator.program.plan["terms"] == ["dihedrals"]
        out = evaluator.evaluate(np.asarray(state.positions)[None], gradients=("tunables",))
        np.testing.assert_allclose(out.jacobian["k"][0],
                                   np.array([oracle(np.asarray(state.positions), top,
                                       top.dihedral_force_constants)[2][mapping == i].sum() for i in range(2)]),
                                   atol=atol, rtol=rtol)

    together, initial = make(values, together=True)
    both = mdir.Simulation(compile_(together, initial, target, precision))
    both.run(0, energy=True)
    base_stats = both.compile_stats.copy()
    both.tunables.update({"k": np.array([-2., .5]), "spring_k": np.array([3., 4.]), "q": np.zeros(5)})
    combined = both.tunables.gradient()
    fresh_system, fresh_state = make(np.array([-2., .5]), together=True)
    fresh_system.tunables = [mdir.Tunable("k", "dihedral_k", map=mapping, values=np.array([-2., .5])),
                            mdir.Tunable("spring_k", "k", term="spring", values=np.array([3., 4.])),
                            mdir.Tunable("q", "charge")]
    fresh = mdir.Simulation(compile_(fresh_system, fresh_state, target, precision))
    fresh.run(0, energy=True)
    fresh_gradient = fresh.tunables.gradient()
    assert combined.energy == fresh_gradient.energy
    for name in combined:
        np.testing.assert_array_equal(combined[name], fresh_gradient[name])
    assert both.compile_stats == base_stats
    report = {"target": target, "precision": precision, "worst_absolute_errors": worst,
              "atol": atol, "rtol": rtol, "references": records}
    (work / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, sort_keys=True))
    print(f"dihedral physics {target} {precision}: oracle, coefficient differences, atomic updates, retained arrays, frames and coexistence passed")


def checkpoint(loaded, target, precision):
    system, state = model(loaded, [mdir.Tunable("k", "dihedral_k")])
    program = compile_(system, state, target, precision)
    sim = mdir.Simulation(program)
    sim.part_seconds = 1e9
    sim.run(3)
    theta = np.array([-1.5, 0., 2., -3.])
    sim.tunables.update({"k": theta})
    sim.run(2)
    g = sim.tunables.gradient()
    path = str(work / "continued.h5")
    sim.save_checkpoint(path)
    resumed = mdir.Simulation(program, checkpoint=path)
    h = resumed.tunables.gradient()
    assert h.energy == g.energy and h.step == g.step
    np.testing.assert_array_equal(h["k"], g["k"])
    np.testing.assert_array_equal(resumed.state().forces, sim.state().forces)
    np.testing.assert_array_equal(resumed.tunables["k"], theta)
    assert resumed.tunables.version == sim.tunables.version
    different, initial = model(loaded, [mdir.Tunable(
        "k", "dihedral_k", map=np.array([1, -1, 0, 2, 3]))])
    altered = compile_(different, initial, target, precision)
    expect(lambda: mdir.Simulation(altered, checkpoint=path), "other physics")
    print(f"dihedral checkpoint {target} {precision}: updated coefficients and gradients continue bitwise; changed map refused")


scenario, work = sys.argv[1], Path(sys.argv[2])
loaded = fixture(work)
if scenario == "declarations":
    declarations(loaded)
elif scenario == "physics":
    physics(loaded, sys.argv[3], sys.argv[4])
elif scenario == "checkpoint":
    checkpoint(loaded, sys.argv[3], sys.argv[4])
else:
    raise ValueError(scenario)
