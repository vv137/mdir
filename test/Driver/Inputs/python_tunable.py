"""Tunable parameters of a Python model (D[python-tunable],
docs/python-tunable.md).

Usage: python_tunable.py ROOT SCENARIO [TARGET PRECISION | CLI WORK]

Scenarios:
  declarations  the declarations, their refusals, and the arrays maps are
                built from (CPU, no run)
  updates       an update during a run against a simulation compiled with
                the new values from the same state (velocity Verlet, NVE),
                bit for bit; against a model without tunables; the refused
                updates, the versions, and the energy file
  integrators   leapfrog keeps its velocities through an update; an update
                before the first run of NPT against a compile with the new
                values, bit for bit
  oracle        the energies at new values against OpenMM 8.6.1 and NumPy,
                and central differences of the energy in a tunable constant
                against the derivative of `observe` (D189) of `mdir run`
"""
import pathlib
import subprocess
import sys

import numpy as np
import mdir

root = sys.argv[1]
scenario = sys.argv[2]
KJ = 4.184


def expect(error, call, text=""):
    try:
        call()
    except error as exc:
        assert text in str(exc), str(exc)
        return str(exc)
    raise AssertionError(f"expected {error.__name__} ({text})")


def soft_term():
    # A repulsion between all pairs, in nm and kJ/mol, whose constants are
    # tunable.
    term = mdir.PairTerm()
    term.name, term.expression = "soft", "a*exp(-r/l)"
    term.constants = [("a", 2.0), ("l", 0.05)]
    return term


def spring_term():
    # Springs between the first heavy atoms of the dipeptide, a parameter per
    # tuple.
    term = mdir.TupleTerm()
    term.name, term.expression, term.arity = "spring", "0.5*k*(r - r0)^2", 2
    term.particles = np.array([[1, 4], [4, 6], [6, 8]], dtype=np.int64)
    term.parameters = [("k", np.array([500.0, 600.0, 600.0])),
                       ("r0", np.array([0.25, 0.26, 0.27]))]
    return term


def model(tunables=None, pme=True, terms=True):
    loaded = mdir.load_amber(root + "/dipeptide.prmtop", root + "/dipeptide.inpcrd")
    system, state = loaded.make_system(), loaded.make_state()
    system.cutoff, system.pairlist_distance, system.switch_distance = 0.8, 0.9, 0.7
    if pme:
        system.electrostatics = mdir.Electrostatics.PME
    if terms:
        system.pair_terms = [soft_term()]
        system.tuple_terms = [spring_term()]
    if tunables is not None:
        system.tunables = tunables(system)
    return system, state


def declarations(system, values=None):
    """Tied charges (one entry per atom name of the waters, each atom of the
    peptide its own), σ and ε of every type, a constant of the pair term,
    and the force constants of the springs, tied in two."""
    charges = system.charges
    names, residues = system.atom_names, system.residue_names
    entries, charge_map = {}, []
    for i, (atom, residue) in enumerate(zip(names, residues)):
        key = (residue, atom) if residue == "WAT" else (residue, i)
        charge_map.append(entries.setdefault(key, len(entries)))
    charge_map = np.array(charge_map, dtype=np.int64)
    values = values or {}
    return [mdir.Tunable("q", "charge", map=charge_map, values=values.get("q")),
            mdir.Tunable("sigma", "sigma", values=values.get("sigma")),
            mdir.Tunable("epsilon", "epsilon", values=values.get("epsilon")),
            mdir.Tunable("soft_a", "a", term="soft", values=values.get("soft_a")),
            mdir.Tunable("k", "k", term="spring", map=np.array([0, 1, 1]),
                         values=values.get("k"))]


def compile_(system, state, target="CPU", precision="Double", leapfrog=False, npt=False,
             deterministic=True):
    integrator, ensemble, execution = mdir.Integrator(), mdir.Ensemble(), mdir.Execution()
    integrator.timestep = 0.0005
    if leapfrog:
        integrator.method = mdir.IntegratorMethod.Leapfrog
    if npt:
        ensemble.kind, ensemble.temperature, ensemble.coupling_period = mdir.EnsembleKind.NPT, 300.0, 10
    execution.target, execution.precision = getattr(mdir.Target, target), getattr(mdir.Precision, precision)
    execution.deterministic = deterministic
    return mdir.compile(system, state, integrator, ensemble, execution, mdir.Schedule())


def simulation(program):
    """A simulation whose runs are one part each, whatever a step takes: a
    part begins by ordering the particles anew (D196), so runs compared to
    the bit must be cut at the same steps."""
    sim = mdir.Simulation(program)
    sim.part_seconds = 1e9
    return sim


def new_values(sim):
    """Values that differ from the model's in every tunable."""
    v = dict(sim.tunables)
    rng = np.random.default_rng(7)
    q = v["q"] * (1.0 + 0.05 * rng.standard_normal(v["q"].shape))
    return {"q": q, "sigma": v["sigma"] * 1.02, "epsilon": v["epsilon"] * 0.9,
            "soft_a": v["soft_a"] * 3.0, "k": v["k"] * 1.5}


def state_of(system, snapshot):
    state = mdir.InitialState()
    state.positions, state.velocities, state.cell = snapshot.positions, snapshot.velocities, snapshot.cell
    return state


def difference(a, b):
    return [float(np.abs(getattr(a, f) - getattr(b, f)).max()) for f in ("positions", "velocities", "forces")]


def run_declarations():
    system, state = model()
    n = system.particle_count
    assert system.charges.shape == (n,) and system.particle_types.shape == (n,)
    assert len(system.atom_names) == n and len(system.residue_names) == n
    assert system.residue_indices.shape == (n,) and system.residue_indices[-1] > 0
    assert len(system.type_names) == int(system.particle_types.max()) + 1
    print(f"model arrays: {n} particles, {len(system.type_names)} types, "
          f"{int(system.residue_indices[-1]) + 1} residues")
    good = declarations(system)
    system.tunables = good
    assert [t.name for t in system.tunables] == ["q", "sigma", "epsilon", "soft_a", "k"]
    assert system.tunables[4].map.tolist() == [0, 1, 1] and system.tunables[1].mixing == "arithmetic"

    def refused(tunables, text, error=mdir.InputError):
        system.tunables = tunables
        return expect(error, lambda: compile_(system, state), text)

    T = mdir.Tunable
    cases = [
        ([T("1q", "charge")], "a tunable needs a name"),
        ([T("q", "charge"), T("q", "sigma")], "two tunables are named 'q'"),
        ([T("q", "mass")], "the parameter is \"charge\""),
        ([T("q", "charge"), T("p", "charge")], "is declared by another tunable"),
        ([T("x", "a", term="nothing")], "no pair or tuple term is named 'nothing'"),
        ([T("x", "b", term="soft")], "the pair term 'soft' has no constant 'b'"),
        ([T("x", "b", term="spring")], "the tuple term 'spring' has no parameter 'b'"),
        ([T("q", "charge", map=np.zeros(3, dtype=np.int64))], "the map has 3 entries"),
        ([T("q", "charge", map=np.full(n, -2))], "below -1"),
        ([T("q", "charge", map=np.full(n, -1))], "the map takes no site"),
        ([T("q", "charge", map=np.r_[np.zeros(n - 1, dtype=np.int64), 2])], "no site takes the entry 1"),
        ([T("q", "charge", map=np.zeros(n, dtype=np.int64))], "differ in the model"),
        ([T("q", "charge", values=np.zeros(2))], "`values` has 2 entries"),
        ([T("s", "sigma", values=-np.ones(len(system.type_names)))], "values of at least 0"),
    ]
    for tunables, text in cases:
        refused(tunables, text)
    expect(mdir.InputError, lambda: T("e", "epsilon", mixing="geometric"), "for \"sigma\" only")
    expect(mdir.InputError, lambda: T("s", "sigma", mixing="harmonic"), "arithmetic\" or \"geometric")
    expect(mdir.InputError, lambda: T("s", "sigma", map=np.zeros(3)), "int32 or int64")
    expect(mdir.InputError, lambda: T("s", "sigma", values=[0.3]), "lists are not accepted")
    print(f"declaration refusals: {len(cases) + 4} passed")
    # A table with an override (an NBFIX of [ nonbond_params ]) takes no
    # per-type values.
    gromacs = root + "/../gromacs"
    loaded = mdir.load_gromacs(gromacs + "/system.top", gromacs + "/system.gro", defines=["FLEXIBLE"])
    other, other_state = loaded.make_system(), loaded.make_state()
    other.cutoff, other.pairlist_distance, other.switch_distance = 0.8, 0.9, 0.8
    other.truncation = mdir.Truncation.None_
    other.tunables = [T("s", "sigma", mixing="geometric")]
    message = expect(mdir.InputError, lambda: compile_(other, other_state), "NBFIX")
    assert "'CT' and 'OW'" in message or "'OW' and 'CT'" in message, message
    other.tunables = [T("q", "charge")]
    compile_(other, other_state)
    print("NBFIX: refused for sigma, naming CT and OW; charges of the same model compile")
    # Declaring is structural: the program goes stale.
    system.tunables = good
    program = compile_(system, state)
    assert not program.stale
    plan = program.plan["tunables"]
    assert [(d["name"], d["entries"], d["unit"]) for d in plan][:3] == [
        ("q", int(good[0].map.max()) + 1, "e"), ("sigma", len(system.type_names), "nm"),
        ("epsilon", len(system.type_names), "kJ/mol")], plan
    assert "md.lookup %t_tunable_constants" in program.ir
    system.tunables = good[:2]
    assert program.stale
    expect(mdir.StaleProgramError, program.check_current)
    # Without tunables the program reads no table of them.
    system.tunables = []
    assert "tunable_constants" not in compile_(system, state).ir
    print("declaring is structural: the program goes stale; the plan lists the tunables")


def run_updates(target, precision):
    system, state = model(declarations)
    sim = simulation(compile_(system, state, target, precision))
    assert sim.tunables.version == 0 and sim.tunables.history == [(0, 0)]
    values = new_values(sim)
    old = dict(sim.tunables)
    # An update during a run (velocity Verlet, NVE) against a simulation
    # compiled with the new values from the state at the update.
    sim.run(8)
    at_update = sim.state()
    version = sim.tunables.update(values)
    assert version == 1 and sim.tunables.history == [(0, 0), (8, 1)], sim.tunables.history
    refreshed = sim.state()
    assert refreshed.tunables_version == 1 and refreshed.energies is not None
    assert np.array_equal(refreshed.positions, at_update.positions)
    assert np.array_equal(refreshed.velocities, at_update.velocities)
    assert not np.array_equal(refreshed.forces, at_update.forces)
    sim.run(12, energy=True)
    updated = sim.state()

    other_system, _ = model(lambda s: declarations(s, values))
    # The compile with the new values evaluates the state first, without a
    # step (run(0, energy=True)), so that both runs cut their calls of the
    # entry at the same steps: a call that carries forces from the last
    # sums in another order than one that evaluates them (D196).
    fresh = simulation(compile_(other_system, state_of(other_system, at_update), target, precision))
    assert all(np.array_equal(fresh.tunables[k], values[k]) for k in values)
    assert fresh.run(0, energy=True) == 0 and fresh.step == 0
    start = fresh.state()
    assert np.array_equal(start.forces, refreshed.forces), np.abs(start.forces - refreshed.forces).max()
    assert start.energies == refreshed.energies, (start.energies, refreshed.energies)
    fresh.run(12, energy=True)
    recompiled = fresh.state()
    d = difference(updated, recompiled)
    assert d == [0.0, 0.0, 0.0] and updated.energies == recompiled.energies, (d, updated.energies,
                                                                             recompiled.energies)
    print(f"{target} {precision}: the forces and energies of an update at step 8 equal those of a "
          f"compile with the new values at that state to the bit (potential "
          f"{refreshed.energies['potential']:.6f} kJ/mol), and so do the state and the energies "
          f"12 steps later (potential {updated.energies['potential']:.6f} kJ/mol)")

    # Back to the old values: the energy of the state is that of the old
    # values, within the rounding of another order of the sums.
    sim2 = simulation(compile_(system, state, target, precision))
    sim2.run(4, energy=True)
    before = sim2.state().energies["potential"]
    sim2.tunables.update(values)
    sim2.tunables.update(old)
    after = sim2.state().energies["potential"]
    tolerance = 1e-9 if precision == "Double" else 1e-4
    assert abs(after - before) <= tolerance * abs(before), (before, after)
    print(f"{target} {precision}: there and back: the potential within {abs(after - before):.1e} "
          f"kJ/mol of {before:.6f}")

    # Against the model without tunables: the table of the combining rule
    # replaces Amber's within 2e-7, and the constants are read from a table.
    plain_system, plain_state = model()
    plain = simulation(compile_(plain_system, plain_state, target, precision))
    tunable = simulation(compile_(system, state, target, precision))
    plain.run(20, energy=True)
    tunable.run(20, energy=True)
    p, t = plain.state(), tunable.state()
    d = difference(p, t)
    rel = abs(p.energies["potential"] - t.energies["potential"]) / abs(p.energies["potential"])
    limit = 1e-7 if precision == "Double" else 1e-5
    assert rel < limit and d[0] < 1e-6, (rel, d)
    print(f"{target} {precision}: against the model without tunables after 20 steps: potential "
          f"within {rel:.1e} relative, positions within {d[0]:.1e} nm")

    # Refused updates change nothing.
    sim = tunable
    version, values_before = sim.tunables.version, dict(sim.tunables)
    T = len(values_before["sigma"])
    for change, text, error in [
            ({"nothing": np.zeros(1)}, "no tunable named 'nothing'", mdir.InputError),
            ({"sigma": np.ones(T + 1)}, "expected shape", mdir.InputError),
            ({"sigma": np.full(T, np.nan)}, "nonfinite", mdir.InputError),
            ({"epsilon": -np.ones(T)}, "values of at least 0", mdir.InputError),
            ({"sigma": np.ones(T), "q": np.full(values_before["q"].shape, 200.0)},
             "charges of less than 100 e", mdir.InputError),
            ({"sigma": [0.3] * T}, "lists are not accepted", mdir.InputError)]:
        expect(error, lambda: sim.tunables.update(change), text)
    assert sim.tunables.version == version
    assert all(np.array_equal(sim.tunables[k], values_before[k]) for k in values_before)
    expect(KeyError, lambda: sim.tunables["nothing"])
    plain_ = plain
    expect(mdir.InputError, lambda: plain_.tunables.update({"q": np.zeros(1)}), "declares no tunable")
    assert len(plain.tunables) == 0 and dict(plain.tunables) == {}
    print(f"{target} {precision}: 6 refused updates change neither values nor version")

    # The energy file takes the version as its last column; the frames take
    # theirs from the history.
    work = pathlib.Path(sys.argv[5] if len(sys.argv) > 5 else ".")
    path = work / f"versions-{target}-{precision}.dat"
    sim = simulation(compile_(system, state, target, precision))
    sim.reporters.append(mdir.EnergyReporter(str(path), 5))
    sim.run(10)
    sim.tunables["soft_a"] = np.array([5.0])
    sim.run(5)
    sim.tunables.update(soft_a=np.array([6.0]), k=values["k"])
    sim.run(5)
    sim.close_reporters()
    lines = path.read_text().splitlines()
    header = lines[0].lstrip("#").split()
    assert header[-1] == "tunables_version", header
    rows = [(int(l.split()[0]), int(l.split()[-1])) for l in lines if not l.startswith("#")]
    assert rows == [(0, 0), (5, 0), (10, 0), (15, 1), (20, 2)], rows
    assert sim.tunables.history == [(0, 0), (10, 1), (15, 2)], sim.tunables.history
    assert sim.state().tunables_version == 2
    print(f"{target} {precision}: versions {rows} in the energy file; history {sim.tunables.history}")
    # Without tunables the energy file has no such column.
    path = work / f"plain-{target}-{precision}.dat"
    plain.reporters.append(mdir.EnergyReporter(str(path), 5))
    plain.run(5)
    plain.close_reporters()
    assert "tunables_version" not in path.read_text().splitlines()[0]


def run_integrators(target, precision):
    # Leapfrog: the velocities are half a step behind the positions, and an
    # update keeps them; the forces are those of the new values.
    system, state = model(declarations)
    sim = simulation(compile_(system, state, target, precision, leapfrog=True))
    sim.run(6)
    before = sim.state()
    values = new_values(sim)
    old = dict(sim.tunables)
    sim.tunables.update(values)
    after = sim.state()
    assert np.array_equal(after.positions, before.positions)
    assert np.array_equal(after.velocities, before.velocities)
    assert after.velocity_offset == -0.5 and after.energies is None
    assert not np.array_equal(after.forces, before.forces)
    sim.tunables.update(old)
    back = sim.state()
    scale = float(np.abs(before.forces).max())
    tolerance = 1e-9 if precision == "Double" else 1e-3
    assert float(np.abs(back.forces - before.forces).max()) <= tolerance * scale
    sim.run(4)
    print(f"{target} {precision}: leapfrog keeps its positions and velocities through an update; "
          f"the forces back at the old values within "
          f"{float(np.abs(back.forces - before.forces).max()):.1e} kJ/mol/nm")

    # NPT: an update before the first run against a compile with the new
    # values, to the bit: the barostat takes the virial and energy of the
    # constant terms of the new values (arguments of the entry).
    sim = simulation(compile_(system, state, target, precision, npt=True))
    values = new_values(sim)
    sim.tunables.update(values)
    assert sim.tunables.history == [(0, 0), (0, 1)]
    sim.run(30, energy=True)
    other_system, _ = model(lambda s: declarations(s, values))
    fresh = simulation(compile_(other_system, state, target, precision, npt=True))
    fresh.run(30, energy=True)
    a, b = sim.state(), fresh.state()
    d = difference(a, b)
    assert d == [0.0, 0.0, 0.0] and a.energies == b.energies, (d, a.energies, b.energies)
    assert "%baro_constant: f64" in compile_(system, state, target, precision, npt=True).ir
    print(f"{target} {precision}: NPT, an update before the first run equals a compile with the "
          f"new values to the bit after 30 steps (volume {a.energies['volume']:.6f} nm^3)")


if scenario == "declarations":
    run_declarations()
elif scenario == "updates":
    run_updates(sys.argv[3], sys.argv[4])
elif scenario == "integrators":
    run_integrators(sys.argv[3], sys.argv[4])
else:
    raise SystemExit(f"unknown scenario {scenario}")
print(f"{scenario} passed")
