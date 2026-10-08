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
    program = dipeptide("Double", False, minimize=True)
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
          "that is not a list of names, a reporter without columns, two reporters; "
          "a minimization does not observe")


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
print("python observe passed")
