"""`observe` in the Python model (D[python-observe], docs/python-observe.md):
the columns of `[output] observables` (D189) written by a reporter of a
Python simulation and read from its state, against `mdir run` on the same
model.

    python_observe.py dipeptide-directory mixture-directory target mdir work

The control file of the same model carries the expression as the Python
model hands it to the builder (the coordinates in nm, the energy in kJ/mol)
and the same numbers for its constants, so the columns are equal to the
printed digits, and in the deterministic mode the files are equal byte for
byte.

  dipeptide   a pair term, the two walls of D189 (terms of the positions),
              and a term over bonds, under a plain cutoff and the shift
  mixture     60 A + 60 B of pair-dispersion.test, an NBFIX-like pair term
              between A and B with its tail and its shift estimate (D209,
              D210), also in NPT
"""
import pathlib
import subprocess
import sys
import warnings

import numpy as np
import mdir

root = sys.argv[1]
mixture = pathlib.Path(sys.argv[2])
target_name = sys.argv[3]
target = getattr(mdir.Target, target_name)
cli = sys.argv[4]
work = pathlib.Path(sys.argv[5])
SEED = 271828
KJ = 4.184
HDF5 = True


def expect(error, call, text=""):
    try:
        call()
    except error as exc:
        assert text in str(exc), str(exc)
        return
    raise AssertionError(f"expected {error.__name__}")


def converted(expression, *lengths):
    """The expression as the Python model hands it to the builder: each
    coordinate in Å times 0.1, the energy over 4.184."""
    import re
    for name in lengths:
        expression = re.sub(rf"\b{name}\b", f"({name}*0.1)", expression)
    return f"({expression})/4.184"


# The terms of the dipeptide in water: (kind, name, expression, coordinates,
# constants, observe, the rest).
SOFT = "a*exp(-r/l)"
LOWER = "0.5*k*max(0, z0 - z)^2"
UPPER = "0.5*k*max(0, z - z0)^2"
FLAT = "k*max(0, r - r0)^2"
BONDS = [[1, 18], [4, 14]]  # zero-based


def dipeptide_terms(system):
    soft = mdir.PairTerm()
    soft.name, soft.expression = "soft", SOFT
    soft.constants = [("a", 2.0), ("l", 0.05)]
    soft.observe = ["l", "a"]
    flat = mdir.TupleTerm()
    flat.name, flat.expression, flat.arity = "flat", FLAT, 2
    flat.particles = np.array(BONDS, dtype=np.int64)
    flat.parameters = [("k", np.array([4184.0, 4184.0])), ("r0", np.array([0.4, 0.4]))]
    flat.observe = ["r0"]
    lower, upper = mdir.ExternalTerm(), mdir.ExternalTerm()
    lower.name, lower.expression, lower.selection = "lower", LOWER, ":WAT"
    lower.constants = [("k", 4184.0), ("z0", 0.6)]
    lower.observe = ["z0"]
    upper.name, upper.expression, upper.selection = "upper", UPPER, ":WAT"
    upper.constants = [("k", 4184.0), ("z0", 2.0)]
    upper.observe = ["z0", "k"]
    system.pair_terms, system.tuple_terms = [soft], [flat]
    system.external_terms = [lower, upper]


DIPEPTIDE_COLUMNS = ["soft.energy", "soft.d_l", "soft.d_a", "flat.energy", "flat.d_r0",
                     "lower.energy", "lower.d_z0", "upper.energy", "upper.d_z0", "upper.d_k"]


def dipeptide_control(name, precision, shift, steps, interval):
    path = work / f"{name}.toml"
    path.write_text(f"""[input]
topology = "{root}/dipeptide.prmtop"
coordinates = "{root}/dipeptide.inpcrd"
[output]
energy_interval = {interval}
energy = "{name}.dat"
observables = "{name}.obs"
[energy]
cutoff = 8.0
pairlist_distance = 9.0
electrostatics = "CUTOFF"
lennard_jones_modifier = "{'POTENTIAL_SHIFT' if shift else 'NONE'}"
[[energy.pair]]
name = "soft"
expression = "{converted(SOFT, 'r')}"
observe = ["l", "a"]
a = 2.0
l = 0.05
[[energy.bond]]
name = "flat"
expression = "{converted(FLAT, 'r')}"
particles = [[2, 19], [5, 15]]
observe = ["r0"]
k = 4184.0
r0 = 0.4
[[energy.external]]
name = "lower"
expression = "{converted(LOWER, 'x', 'y', 'z')}"
selection = ":WAT"
observe = ["z0"]
k = 4184.0
z0 = 0.6
[[energy.external]]
name = "upper"
expression = "{converted(UPPER, 'x', 'y', 'z')}"
selection = ":WAT"
observe = ["z0", "k"]
k = 4184.0
z0 = 2.0
[dynamics]
time_step = 0.0005
steps = {steps}
seed = {SEED}
[ensemble]
ensemble = "NVE"
temperature = 300.0
[boundary]
type = "PERIODIC"
[execution]
target = "{target_name}"
precision = "{precision.upper()}"
deterministic = true
""")
    return path


def dipeptide(precision, shift, change=None, minimize=False, kind="NVE"):
    loaded = mdir.load_amber(root + "/dipeptide.prmtop", root + "/dipeptide.inpcrd")
    system, state = loaded.make_system(), loaded.make_state()
    system.cutoff, system.pairlist_distance = 0.8, 0.9
    system.truncation = mdir.Truncation.Shift if shift else mdir.Truncation.None_
    system.electrostatics = mdir.Electrostatics.Cutoff
    dipeptide_terms(system)
    if change:
        change(system)
    state = state.draw_velocities(system, 300.0, SEED)
    integrator, ensemble, execution = mdir.Integrator(), mdir.Ensemble(), mdir.Execution()
    integrator.timestep, ensemble.temperature, ensemble.seed = 0.0005, 300.0, SEED
    integrator.minimize = minimize
    ensemble.kind = getattr(mdir.EnsembleKind, kind)
    execution.target, execution.precision = target, getattr(mdir.Precision, precision)
    execution.deterministic = True
    return mdir.compile(system, state, integrator, ensemble, execution, mdir.Schedule())


def rows(path):
    return [line for line in pathlib.Path(path).read_text().splitlines()
            if not line.startswith("#")]


def run_cli(path):
    subprocess.run([cli, "run", str(path)], cwd=work, check=True, stdout=subprocess.DEVNULL)


def as_row(step, time, values):
    """The row that the file holds of the values of a state (kJ/mol)."""
    return " ".join([str(step), f"{time:.6f}"] + [f"{v / KJ:.6f}" for v in values])


def run_files():
    # The gate of item 8: the files of 20 steps with rows every 5, written
    # inside the parts of one run, equal those of `mdir run`; and so do the
    # values that a callback reads at each row.
    for precision in ("Double", "Mixed"):
        for shift in (False, True):
            name = f"dip-{precision}-{'shift' if shift else 'cutoff'}".lower()
            run_cli(dipeptide_control(name, precision, shift, 20, 5))
            program = dipeptide(precision, shift)
            assert program.plan["observables"] == DIPEPTIDE_COLUMNS
            sim = mdir.Simulation(program)
            sim.reporters.append(mdir.ObservablesReporter(str(work / f"py-{name}.obs"), 5))
            sim.reporters.append(mdir.EnergyReporter(str(work / f"py-{name}.dat"), 5))
            assert sim.run(20) == 20
            sim.close_reporters()
            mine = (work / f"py-{name}.obs").read_bytes()
            theirs = (work / f"{name}.obs").read_bytes()
            assert mine == theirs, name
            assert (work / f"py-{name}.dat").read_bytes() == (work / f"{name}.dat").read_bytes()
            # The values of the state at each row, with and without the
            # reporter, and after run(0, energy=True).
            sim = mdir.Simulation(program)
            seen = []
            sim.reporters.append(mdir.CallbackReporter(
                lambda s, state: seen.append((state.step, state.time, state.observables)), 5))
            sim.run(0, energy=True)
            first = sim.state()
            assert list(first.observables) == DIPEPTIDE_COLUMNS
            seen.append((0, first.time, first.observables))
            sim.run(20)
            seen.sort(key=lambda item: item[0])
            read = [as_row(step, time, values.values()) for step, time, values in seen]
            assert read == rows(work / f"{name}.obs"), (read[0], rows(work / f"{name}.obs")[0])
            print(f"{target_name} {precision} {'shift' if shift else 'cutoff'}: "
                  f"{len(read)} rows of {len(DIPEPTIDE_COLUMNS)} columns (pair, bonds, walls) "
                  f"equal to mdir run's file byte for byte, and the state's values to its rows")


def run_parts():
    # Parts and periods: runs of 7 and 13 steps write the file of one run;
    # the reporter's period is its own (energies every 4, observables every
    # 6: the steps of energy fall every 2); a file that exists is backed up.
    name = "dip-double-cutoff"
    program = dipeptide("Double", False)
    reference = (work / f"{name}.obs").read_bytes()
    sim = mdir.Simulation(program)
    sim.reporters.append(mdir.ObservablesReporter(str(work / "parts.obs"), 5))
    sim.run(7)
    sim.run(13)
    sim.close_reporters()
    assert (work / "parts.obs").read_bytes() == reference
    run_cli(dipeptide_control("six", "Double", False, 24, 6))
    sim = mdir.Simulation(program)
    sim.reporters.append(mdir.ObservablesReporter(str(work / "parts.obs"), 6))
    sim.reporters.append(mdir.EnergyReporter(str(work / "parts.dat"), 4))
    sim.run(24)
    sim.close_reporters()
    assert (work / "parts.obs").read_bytes() == (work / "six.obs").read_bytes()
    assert [r.split()[0] for r in rows(work / "parts.dat")] == ["0", "4", "8", "12", "16", "20", "24"]
    backups = sorted(p.name for p in work.glob("#parts.obs*"))
    assert backups == ["#parts.obs.1#"], backups
    assert pathlib.Path(work / "#parts.obs.1#").read_bytes() == reference
    # A reporter added after the first run begins with its next row.
    sim = mdir.Simulation(program)
    sim.run(10)
    sim.reporters.append(mdir.ObservablesReporter(str(work / "late.obs"), 5))
    sim.run(10)
    sim.close_reporters()
    assert rows(work / "late.obs") == rows(work / f"{name}.obs")[3:]
    print("parts: runs of 7 and 13 steps write the file of one run; a period of its own; "
          "backup; a late reporter")
    if not HDF5:
        return
    # A run continued from a checkpoint at step 10 writes the file of one
    # run (append=True), or the rows that follow to a part of its own.
    sim = mdir.Simulation(program)
    sim.reporters.append(mdir.ObservablesReporter(str(work / "ck.obs"), 5))
    sim.run(10)
    sim.save_checkpoint(str(work / "ck.h5"))
    sim.run(5)  # A row after the checkpoint, which the continuation writes again.
    sim.close_reporters()
    del sim
    sim = mdir.Simulation(program, checkpoint=str(work / "ck.h5"))
    sim.reporters.append(mdir.ObservablesReporter(str(work / "ck.obs"), 5))
    sim.run(10)
    sim.close_reporters()
    assert (work / "ck.obs").read_bytes() == reference
    sim = mdir.Simulation(program, checkpoint=str(work / "ck.h5"), append=False)
    sim.reporters.append(mdir.ObservablesReporter(str(work / "ck.obs"), 5))
    sim.run(10)
    sim.close_reporters()
    part = sorted(p.name for p in work.glob("ck.part*.obs"))
    assert len(part) == 1, part
    assert rows(work / part[0]) == rows(work / f"{name}.obs")[3:]
    # Without `observe`, the same run continues the checkpoint, and the
    # reverse: nothing of it is in the fingerprint (D189).
    def unobserved(system):
        terms = system.pair_terms
        terms[0].observe = None
        system.pair_terms = terms
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        plain = mdir.Simulation(dipeptide("Double", False, unobserved),
                                checkpoint=str(work / "ck.h5"))
    plain.run(10, energy=True)
    assert "soft.energy" not in plain.state().observables
    # The sums run in the order of the particles of an activation, which
    # begins anew at a checkpoint: equal to round-off, not to the bit.
    mine, whole = plain.state().observables["lower.d_z0"], seen_at_20["lower.d_z0"]
    assert abs(mine / whole - 1) < 1e-12, (mine, whole)
    # The reverse: a checkpoint written without `observe` on the term,
    # continued by the program that observes it.
    plain.save_checkpoint(str(work / "plain.h5"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        observing = mdir.Simulation(program, checkpoint=str(work / "plain.h5"))
    observing.run(0, energy=True)
    assert abs(observing.state().observables["soft.energy"] / seen_at_20["soft.energy"] - 1) < 1e-12
    print(f"checkpoint: continued at step 10 with append, the file of one run; without, "
          f"{part[0]} with the rows that follow; continued without observe on a term")


def run_refusals():
    def changed(edit):
        def change(system):
            pairs, tuples, externals = system.pair_terms, system.tuple_terms, system.external_terms
            edit(pairs[0], tuples[0], externals)
            system.pair_terms, system.tuple_terms, system.external_terms = pairs, tuples, externals
        return lambda: dipeptide("Double", False, change)

    def set_observe(term, value):
        term.observe = value

    expect(mdir.InputError, changed(lambda p, t, e: set_observe(p, ["z1"])),
           "the term 'soft' observes 'z1', which is not a constant of the term given as one number")
    expect(mdir.InputError, changed(lambda p, t, e: set_observe(p, ["l", "l"])),
           "the term 'soft': 'l' is observed twice")
    expect(mdir.InputError, changed(lambda p, t, e: set_observe(e[0], ["x"])),
           "the term 'lower' observes 'x', which is not a constant of the term")

    def per_tuple(p, t, e):
        t.parameters = [("k", np.array([4184.0, 4184.0])), ("r0", np.array([0.4, 0.45]))]
    expect(mdir.InputError, changed(per_tuple),
           "the term 'flat' observes 'r0', which has a value for each particle or tuple")
    for term in (mdir.PairTerm(), mdir.TupleTerm(), mdir.ExternalTerm()):
        assert term.observe is None
        for bad in ("l", 3, [3], [None]):
            expect(TypeError, lambda: set_observe(term, bad), ".observe takes None or a list")
        term.observe = []
        assert term.observe == []
        term.observe = ("a", "b")
        assert term.observe == ["a", "b"]
        term.observe = None
        assert term.observe is None

    # The reporter: a program that observes nothing, two reporters, a
    # period that is not positive.
    def nothing(system):
        system.pair_terms, system.tuple_terms, system.external_terms = [], [], []
    program = dipeptide("Double", False, nothing)
    assert program.plan["observables"] == []
    sim = mdir.Simulation(program)
    sim.reporters.append(mdir.ObservablesReporter(str(work / "none.obs"), 5))
    expect(mdir.InputError, lambda: sim.run(5), "no term of the program gives 'observe'")
    assert not (work / "none.obs").exists()
    sim.reporters = []
    sim.run(5, energy=True)
    assert sim.state().observables is None
    sim = mdir.Simulation(dipeptide("Double", False))
    sim.reporters.append(mdir.ObservablesReporter(str(work / "two.obs"), 5))
    sim.reporters.append(mdir.ObservablesReporter(str(work / "three.obs"), 5))
    expect(mdir.InputError, lambda: sim.run(5), "a simulation takes one ObservablesReporter")
    expect(mdir.InputError, lambda: mdir.ObservablesReporter("x.obs", 0), "positive")
    expect(mdir.InputError, lambda: mdir.ObservablesReporter("", 5), "name of a file")
    # `observe = []` is the energy alone; a state without a step of energy
    # has no values; a minimization does not observe.
    def energy_only(system):
        terms = system.external_terms
        terms[0].observe = []
        terms[1].observe = None
        system.external_terms = terms
        system.pair_terms, system.tuple_terms = [], []
    program = dipeptide("Double", False, energy_only)
    assert program.plan["observables"] == ["lower.energy"]
    sim = mdir.Simulation(program)
    sim.run(3)
    assert sim.state().observables is None and sim.state().energies is None
    sim.run(2, energy=True)
    assert list(sim.state().observables) == ["lower.energy"]
    # A minimization refuses `observe`, as `mdir run` refuses `observables`
    # under [minimize]; without it the same system minimizes.
    expect(mdir.InputError, lambda: dipeptide("Double", False, minimize=True),
           "the term 'soft' gives 'observe', which is evaluated at the energies of a run of "
           "dynamics; a minimization does not take it")

    def unobserved(system):
        pairs, tuples, externals = system.pair_terms, system.tuple_terms, system.external_terms
        for term in pairs + tuples + externals:
            term.observe = None
        system.pair_terms, system.tuple_terms, system.external_terms = pairs, tuples, externals
    program = dipeptide("Double", False, unobserved, minimize=True)
    assert program.plan["observables"] == []
    sim = mdir.Simulation(program)
    sim.minimize(3)
    assert sim.state().observables is None
    # A program becomes stale when `observe` changes on the system.
    loaded = mdir.load_amber(root + "/dipeptide.prmtop", root + "/dipeptide.inpcrd")
    system, state = loaded.make_system(), loaded.make_state()
    system.cutoff, system.pairlist_distance = 0.8, 0.9
    system.truncation = mdir.Truncation.None_
    dipeptide_terms(system)
    program = mdir.compile(system, state, mdir.Integrator(), mdir.Ensemble(), mdir.Execution(),
                           mdir.Schedule())
    assert not program.stale
    terms = system.pair_terms
    terms[0].observe = None
    system.pair_terms = terms
    assert program.stale
    print("refusals: an unknown constant, a name twice, a parameter of each tuple, a value "
          "that is not a list of names, a reporter without columns, two reporters, "
          "a minimization")


def run_external():
    # The terms of the positions (D[python-external]): what is refused.
    def changed(edit, **options):
        def change(system):
            terms = system.external_terms
            edit(terms[0])
            system.external_terms = terms
        return lambda: dipeptide("Double", False, change, **options)

    def both(term):
        term.particles = np.array([0, 1])
    expect(mdir.InputError, changed(both), "takes 'selection', a mask of particles, or 'particles'")

    def neither(term):
        term.selection = ""
    expect(mdir.InputError, changed(neither), "takes 'selection'")

    def twice(term):
        term.selection, term.particles = "", np.array([3, 3])
    expect(mdir.InputError, changed(twice), "invalid particle identities")

    def beyond(term):
        term.selection, term.particles = "", np.array([10 ** 6])
    expect(mdir.InputError, changed(beyond), "invalid particle identities")

    def unknown(term):
        term.expression = "0.5*k*max(0, z0 - z)^2 + w"
    expect(mdir.InputError, changed(unknown), "undeclared parameter 'w'")

    def reserved(term):
        term.constants = [("k", 4184.0), ("z0", 0.6), ("x", 1.0)]
    expect(mdir.InputError, changed(reserved), "invalid external term parameter name")
    expect(mdir.InputError, changed(lambda term: None, kind="NPT"),
           "the external term 'lower' needs 'scaling' under a barostat")
    term = mdir.ExternalTerm()
    assert term.scaling is None
    term.scaling = mdir.ExternalScaling.Cell
    assert term.scaling == mdir.ExternalScaling.Cell

    def assign(value):
        term.scaling = value
    expect(TypeError, lambda: assign("CELL"), "takes an ExternalScaling or None")
    # A wall over particles given by their indices equals the mask's.
    waters = None

    def by_index(system):
        nonlocal waters
        residues = system.topology.residue_names
        index = system.topology.residue_indices
        waters = np.array([i for i in range(system.particle_count) if residues[index[i]] == "WAT"])
        terms = system.external_terms
        for wall in terms:
            wall.selection, wall.particles = "", waters
        system.external_terms = terms
    sim = mdir.Simulation(dipeptide("Double", False, by_index))
    sim.run(0, energy=True)
    other = mdir.Simulation(dipeptide("Double", False))
    other.run(0, energy=True)
    assert sim.state().observables == other.state().observables
    assert sim.state().energies == other.state().energies
    print(f"external terms: refusals; {len(waters)} particles by index equal to the mask ':WAT'")


def run_external_parity():
    # The terms of the positions by themselves (D[python-external]): the
    # energy and the forces of two walls against NumPy at the start, and a
    # run of 10 steps with them against `mdir run` with the same
    # [[energy.external]] terms: the energy file byte for byte, and the
    # positions, velocities, and forces of its checkpoint to the bit.
    def walls(system):
        system.pair_terms, system.tuple_terms = [], []
        terms = system.external_terms
        for term in terms:
            term.observe = []
        system.external_terms = terms

    def none(system):
        system.pair_terms, system.tuple_terms, system.external_terms = [], [], []
    for precision in ("Double", "Mixed"):
        program = dipeptide(precision, False, walls)
        sim, bare = mdir.Simulation(program), mdir.Simulation(dipeptide(precision, False, none))
        sim.run(0, energy=True)
        bare.run(0, energy=True)
        state, without = sim.state(), bare.state()
        topology = program.topology
        waters = np.array([topology.residue_names[r] == "WAT" for r in topology.residue_indices])
        z = state.positions[:, 2]
        k, low, high = 4184.0, 0.6, 2.0
        below, above = np.maximum(0, low - z) * waters, np.maximum(0, z - high) * waters
        energy = {"lower.energy": 0.5 * k * np.sum(below ** 2),
                  "upper.energy": 0.5 * k * np.sum(above ** 2)}
        force = np.zeros_like(state.positions)
        force[:, 2] = k * below - k * above
        mine = state.forces - without.forces
        tolerance = 1e-12 if precision == "Double" else 2e-6
        e = max(abs(state.observables[key] / value - 1) for key, value in energy.items())
        f = np.abs(mine - force).max() / np.abs(force).max()
        assert e < tolerance and f < tolerance, (e, f)
        total = abs((state.energies["potential"] - without.energies["potential"])
                    / sum(energy.values()) - 1)
        assert total < (1e-9 if precision == "Double" else 1e-4), total
        line = (f"{target_name} {precision} external terms: {int(waters.sum())} particles, the "
                f"walls' energies {energy['lower.energy'] / KJ:.6f} and "
                f"{energy['upper.energy'] / KJ:.6f} kcal/mol within {e:.1e} of NumPy, their "
                f"forces within {f:.1e} of the largest")
        if not HDF5:
            print(line)
            continue
        name = f"walls-{precision}".lower()
        text = dipeptide_control(name, precision, False, 10, 5).read_text()
        head, rest = text.split("[[energy.pair]]")
        external = rest[rest.index("[[energy.external]]"):]
        external = "\n".join(l for l in external.splitlines() if not l.startswith("observe"))
        head = head.replace(f'observables = "{name}.obs"\n',
                            f'checkpoint = "{name}.h5"\ncheckpoint_interval = 10\n')
        (work / f"{name}.toml").write_text(head + external + "\n")
        run_cli(work / f"{name}.toml")
        sim = mdir.Simulation(dipeptide(precision, False, walls))
        sim.reporters.append(mdir.EnergyReporter(str(work / f"py-{name}.dat"), 5))
        sim.run(10)
        sim.close_reporters()
        assert (work / f"py-{name}.dat").read_bytes() == (work / f"{name}.dat").read_bytes()
        theirs, ours = mdir.read_checkpoint(str(work / f"{name}.h5")), sim.state()
        assert theirs.step == 10
        for field in ("positions", "velocities", "forces"):
            assert np.array_equal(getattr(theirs, field), getattr(ours, field)), field
        # The fingerprint of the Python model has an entry of its own for
        # these terms, which `mdir run --continue` refuses by name (D223).
        (work / f"{name}.h5").unlink()
        sim.save_checkpoint(str(work / f"{name}.h5"))
        entries = [entry[1] for entry in mdir.read_checkpoint(str(work / f"{name}.h5")).fingerprint]
        assert "[python] external_terms" in entries, entries
        longer = (work / f"{name}.toml").read_text().replace("steps = 10", "steps = 20")
        (work / f"{name}.toml").write_text(longer)
        again = subprocess.run([cli, "run", "--continue", f"{name}.toml"], cwd=work,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert again.returncode != 0 and "[python] external_terms" in again.stdout, again.stdout
        print(line + "; 10 steps: the energy file byte for byte and the positions, velocities, "
              "and forces of the checkpoint of mdir run to the bit")


# The mixture of pair-dispersion.test: an NBFIX written as a pair term
# between A and B, the difference of two Lennard-Jones (kJ/mol, nm).
NBFIX = "4*eps*((sig/r)^12 - (sig/r)^6) - 4*eps0*((sig0/r)^12 - (sig0/r)^6)"
NBFIX_CONSTANTS = [("sig", 0.37), ("eps", 0.5 * KJ), ("sig0", 0.34), ("eps0", 0.24 * KJ)]


def mixture_control(name, precision, shift, kind, steps, interval):
    coupling = ('[thermostat]\nmethod = "V-RESCALE"\ninterval = 10\n'
                '[barostat]\nmethod = "C-RESCALE"\n' if kind == "NPT" else "")
    com = "center_of_mass_interval = 10" if kind == "NPT" else ""
    constants = "\n".join(f"{key} = {value!r}" for key, value in NBFIX_CONSTANTS)
    path = work / f"{name}.toml"
    path.write_text(f"""[input]
topology = "{mixture}/plain.top"
coordinates = "{mixture}/system.gro"
[output]
energy_interval = {interval}
energy = "{name}.dat"
observables = "{name}.obs"
[energy]
cutoff = 12.0
pairlist_distance = 13.0
electrostatics = "CUTOFF"
lennard_jones_modifier = "{'POTENTIAL_SHIFT' if shift else 'NONE'}"
[[energy.pair]]
name = "nbfix"
groups = [":A", ":B"]
expression = "{converted(NBFIX, 'r')}"
observe = ["sig", "eps"]
{constants}
[dynamics]
time_step = 0.001
steps = {steps}
seed = {SEED}
{com}
[ensemble]
ensemble = "{kind}"
temperature = 100.0
{coupling}[boundary]
type = "PERIODIC"
[execution]
target = "{target_name}"
precision = "{precision.upper()}"
deterministic = true
""")
    return path


def mixture_program(precision, shift, kind="NVE", dispersion=None):
    loaded = mdir.load_gromacs(str(mixture / "plain.top"), str(mixture / "system.gro"))
    system, state = loaded.make_system(), loaded.make_state()
    system.cutoff, system.pairlist_distance = 1.2, 1.3
    system.truncation = mdir.Truncation.Shift if shift else mdir.Truncation.None_
    if dispersion is not None:
        system.dispersion = dispersion
    term = mdir.PairTerm()
    term.name, term.groups, term.expression = "nbfix", [":A", ":B"], NBFIX
    term.constants = NBFIX_CONSTANTS
    term.observe = ["sig", "eps"]
    system.pair_terms = [term]
    state = state.draw_velocities(system, 100.0, SEED)
    integrator, ensemble, execution = mdir.Integrator(), mdir.Ensemble(), mdir.Execution()
    integrator.timestep, ensemble.temperature, ensemble.seed = 0.001, 100.0, SEED
    ensemble.kind = getattr(mdir.EnsembleKind, kind)
    if kind != "NVE":
        ensemble.com_period = ensemble.coupling_period = 10
    execution.target, execution.precision = target, getattr(mdir.Precision, precision)
    execution.deterministic = True
    return mdir.compile(system, state, integrator, ensemble, execution, mdir.Schedule())


def mixture_tails():
    """What the correction adds to the columns of the NBFIX term, in closed
    form (D209, D210), kJ/mol and nm: nu (4 pi / V) N_A N_B (I + f r_c^3
    u(r_c) / 3), with I the integral of r^2 u beyond r_c, f = 1 - V / (N 4
    pi r_c^3 / 3), and nu = N / (N - 1) (no pair is excluded); and its
    derivatives in sigma and epsilon of the first Lennard-Jones."""
    import math
    gro = (mixture / "system.gro").read_text().splitlines()
    count = int(gro[1])
    kinds = [line[5:10].strip() for line in gro[2:2 + count]]
    volume = float(gro[2 + count].split()[0]) ** 3
    na, nb, rc = kinds.count("A"), kinds.count("B"), 1.2
    n = na + nb
    f = 1 - volume / (n * 4 * math.pi / 3 * rc ** 3)
    factor = 4 * math.pi / volume * n / (n - 1) * na * nb
    c = dict(NBFIX_CONSTANTS)

    def part(sig, eps):
        i = 4 * eps * (sig ** 12 / (9 * rc ** 9) - sig ** 6 / (3 * rc ** 3))
        u = 4 * eps * ((sig / rc) ** 12 - (sig / rc) ** 6)
        di = 4 * eps * (12 * sig ** 11 / (9 * rc ** 9) - 6 * sig ** 5 / (3 * rc ** 3))
        du = 4 * eps * (12 * sig ** 11 / rc ** 12 - 6 * sig ** 5 / rc ** 6)
        return (factor * (i + f * rc ** 3 * u / 3), factor * (di + f * rc ** 3 * du / 3))
    energy, d_sig = part(c["sig"], c["eps"])
    energy0, _ = part(c["sig0"], c["eps0"])
    return {"nbfix.energy": energy - energy0, "nbfix.d_sig": d_sig,
            "nbfix.d_eps": energy / c["eps"]}


def run_mixture():
    # A pair term whose tail is in the correction for the dispersion: its
    # columns hold the tail and, under either modifier, the estimate of the
    # shift (D209, D210), in the energy and in both derivatives.
    for precision in ("Double", "Mixed"):
        for shift in (False, True):
            name = f"mix-{precision}-{'shift' if shift else 'cutoff'}".lower()
            run_cli(mixture_control(name, precision, shift, "NVE", 20, 5))
            sim = mdir.Simulation(mixture_program(precision, shift))
            sim.reporters.append(mdir.ObservablesReporter(str(work / f"py-{name}.obs"), 5))
            sim.run(20)
            sim.close_reporters()
            assert (work / f"py-{name}.obs").read_bytes() == (work / f"{name}.obs").read_bytes(), name
            # What the correction adds to the columns: with it off, they
            # are those of the pairs within the cutoff alone.
            sim.run(0, energy=True)
            on = sim.state().observables
            bare = mdir.Simulation(mixture_program(precision, shift, dispersion=
                                                   mdir.DispersionCorrection.None_))
            bare.run(20, energy=True)
            off = bare.state().observables
            tails = {key: (on[key] - off[key]) / KJ for key in on}
            closed = mixture_tails()
            for key, value in closed.items():
                assert abs(tails[key] * KJ / value - 1) < 1e-8, (key, tails[key] * KJ, value)
            print(f"{target_name} {precision} {'shift' if shift else 'cutoff'}: the mixture's "
                  f"rows equal to mdir run's file byte for byte; the correction adds, as its closed "
                  f"forms, "
                  + ", ".join(f"{key} {value:.6f}" for key, value in tails.items()))
    # At constant pressure the tails follow the volume of each row.
    for precision in ("Double", "Mixed"):
        name = f"mix-npt-{precision}".lower()
        run_cli(mixture_control(name, precision, False, "NPT", 40, 10))
        sim = mdir.Simulation(mixture_program(precision, False, "NPT"))
        sim.reporters.append(mdir.ObservablesReporter(str(work / f"py-{name}.obs"), 10))
        sim.reporters.append(mdir.EnergyReporter(str(work / f"py-{name}.dat"), 10))
        sim.run(40)
        sim.close_reporters()
        mine, theirs = rows(work / f"py-{name}.obs"), rows(work / f"{name}.obs")
        if mine == theirs:
            print(f"{target_name} {precision} NPT: {len(mine)} rows equal to mdir run's file "
                  f"byte for byte")
            continue
        assert precision == "Mixed" and len(mine) == len(theirs)
        worst = 0.0
        for a, b in zip(mine, theirs):
            a, b = (np.array([float(x) for x in r.split()]) for r in (a, b))
            worst = max(worst, float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1.0))))
        assert worst < 1e-3, worst
        print(f"{target_name} {precision} NPT: {len(mine)} rows within {worst:.1e} of mdir "
              f"run's (the two programs round differently in single precision, #105)")


def run_tunables(precision):
    # Tunables and `observe` in one program: the columns follow an update,
    # and the row says of which values it is.
    def tuned(system):
        terms = system.pair_terms
        terms[0].observe = ["l"]
        system.pair_terms = terms
        system.external_terms, system.tuple_terms = [], []
        system.tunables = [mdir.Tunable("soft_a", "a", term="soft")]
        system.truncation = mdir.Truncation.None_
    sim = mdir.Simulation(dipeptide(precision, False, tuned))
    sim.reporters.append(mdir.ObservablesReporter(str(work / "tuned.obs"), 5))
    sim.run(5, energy=True)
    before = sim.state().observables
    sim.tunables["soft_a"] = np.array([3.0])
    sim.run(0, energy=True)
    after = sim.state().observables
    tolerance = 1e-12 if precision == "Double" else 1e-5
    for key in ("soft.energy", "soft.d_l"):
        assert abs(after[key] / before[key] / 1.5 - 1) < tolerance, (key, after[key], before[key])
    sim.run(5)
    sim.close_reporters()
    text = (work / "tuned.obs").read_text().splitlines()
    assert text[0] == "# step time soft.energy soft.d_l tunables_version", text[0]
    assert text[1] == "# - ps kcal/mol kcal/mol/l -", text[1]
    assert [(r.split()[0], r.split()[-1]) for r in text[2:]] == [("0", "0"), ("5", "0"), ("10", "1")]

    # An observed constant may be a tunable with one entry that every site
    # takes: dU/dl of `observe` equals that of gradient() (D230), of the
    # same shifted potential with the same tail, follows an update without
    # compiling, and equals central differences of the term's observed
    # energy by updates (D213).
    def both(system):
        tuned(system)
        system.tunables = [mdir.Tunable("soft_l", "l", term="soft")]
        system.tunable_gradient = True
    program = dipeptide(precision, False, both)
    assert program.plan["observables"] == ["soft.energy", "soft.d_l"]
    sim = mdir.Simulation(program)
    tight, loose = (1e-12, 1e-7) if precision == "Double" else (1e-5, 1e-3)
    worst = 0.0
    for l in (0.05, 0.06):
        sim.tunables["soft_l"] = np.array([l])
        gradient = sim.tunables.gradient()["soft_l"][0]
        sim.run(0, energy=True)
        d_l = sim.state().observables["soft.d_l"]

        def fresh(system, l=l):
            tuned(system)
            system.tunables = []
            terms = system.pair_terms
            terms[0].constants = [("a", 2.0), ("l", l)]
            system.pair_terms = terms
        compiled = mdir.Simulation(dipeptide(precision, False, fresh))
        compiled.run(0, energy=True)
        other = compiled.state().observables["soft.d_l"]
        worst = max(worst, abs(gradient / d_l - 1), abs(other / d_l - 1))
        assert worst < tight, (l, d_l, gradient, other)
    h, energies = 1e-5, []
    for value in (0.06 + h, 0.06 - h):
        sim.tunables["soft_l"] = np.array([value])
        sim.run(0, energy=True)
        energies.append(sim.state().observables["soft.energy"])
    differences = abs((energies[0] - energies[1]) / (2 * h) / d_l - 1)
    assert differences < loose, differences
    # The same in the run: the row after an update is of the new value.
    sim.tunables["soft_l"] = np.array([0.05])
    sim.reporters.append(mdir.ObservablesReporter(str(work / "fit.obs"), 5))
    sim.run(5)
    sim.tunables["soft_l"] = np.array([0.06])
    sim.run(5)
    sim.close_reporters()
    fit = [r.split() for r in rows(work / "fit.obs")]
    assert [(r[0], r[-1]) for r in fit] == [("5", "5"), ("10", "6")], fit

    # A parameter of a tuple term: one entry for all tuples is observed,
    # several entries are refused with the way to their derivatives.
    def bonds(map_):
        def change(system):
            system.pair_terms, system.external_terms = [], []
            system.tunables = [mdir.Tunable("rest", "r0", term="flat", map=np.array(map_))]
            system.tunable_gradient = True
        return change
    expect(mdir.InputError, lambda: dipeptide(precision, False, bonds([0, 1])),
           "the term 'flat' observes 'r0', which the tunable 'rest' takes with several entries")
    expect(mdir.InputError, lambda: dipeptide(precision, False, bonds([0, -1])),
           "Simulation.tunables.gradient()['rest']")
    sim = mdir.Simulation(dipeptide(precision, False, bonds([0, 0])))
    sim.tunables["rest"] = np.array([0.35])
    gradient = sim.tunables.gradient()["rest"][0]
    sim.run(0, energy=True)
    d_r0 = sim.state().observables["flat.d_r0"]
    assert d_r0 != 0.0 and abs(gradient / d_r0 - 1) < tight, (gradient, d_r0)
    print(f"{target_name} {precision} tunables: the columns follow an update, with its version "
          f"in the row; an observed tunable of one entry: soft.d_l {d_l:.6f} kJ/mol/nm against "
          f"gradient() and a compile with the value within {worst:.1e}, central differences "
          f"within {differences:.1e}; flat.d_r0 against gradient() within "
          f"{abs(gradient / d_r0 - 1):.1e}; several entries refused")


scenario = sys.argv[6] if len(sys.argv) > 6 else "all"
try:
    mdir.read_checkpoint(str(work / "missing.h5"))
except mdir.UnsupportedError:
    HDF5 = False
except Exception:
    pass
run_files()
probe = mdir.Simulation(dipeptide("Double", False))
probe.run(20, energy=True)
seen_at_20 = probe.state().observables
run_parts()
run_refusals()
run_external()
run_external_parity()
run_mixture()
for precision in ("Double", "Mixed"):
    run_tunables(precision)
print("python observe passed")
