"""Minimization in a Python simulation (D[python-minimize]): the minimizer
of `mdir run`, run in several parts, against the last row and the
checkpoint of `mdir run` on the same input; the next stage from the
minimized positions against `mdir run` from its checkpoint; refusals."""
import pathlib
import subprocess
import sys

import numpy as np
import mdir

root = sys.argv[1]
target_name = sys.argv[2]
target = getattr(mdir.Target, target_name)
cli = sys.argv[3]
work = pathlib.Path(sys.argv[4])
SEED = 271828
# The first part of a simulation takes at most 100 steps (D196): FIRST
# steps run in one part, and the steps after them in parts of their own.
FIRST = 100
AFTER = 10
KCAL_A2 = 4.184 / (0.1 * 0.1)  # kJ/mol/nm^2 per kcal/mol/A^2
# The columns of the log of a minimization, and of dynamics, from MD units.
MINIMIZATION = {"potential": ("energy", 4.184), "rms_force": ("rms_force", 41.84),
                "max_force": ("max_force", 41.84), "step_size": ("step_size", 0.1)}
DYNAMICS = {"total": 4.184, "potential": 4.184, "kinetic": 4.184, "temperature": 1.0,
            "conserved": 4.184}


def expect(error, call, text=""):
    try:
        call()
    except error as exc:
        assert text in str(exc), str(exc)
        return
    raise AssertionError(f"expected {error.__name__}")


def control(name, precision, stage, steps=FIRST, electrostatics="PME", constraints=True):
    if stage == "min":
        body = f'[minimize]\nmethod = "STEEPEST_DESCENT"\nsteps = {steps}\n'
        start = extra = ""
        interval = 1
    else:
        body = (f'[dynamics]\ntime_step = 0.0005\nsteps = 20\nseed = {SEED}\n'
                '[ensemble]\nensemble = "NVT"\ntemperature = 300.0\n'
                '[thermostat]\nmethod = "V-RESCALE"\ninterval = 10\n')
        start = f'checkpoint = "min-{precision}.h5"\n'
        interval = 10
        extra = "checkpoint_interval = 20\n"
    path = work / f"{name}.toml"
    path.write_text(f"""[input]
topology = "{root}/dipeptide.prmtop"
coordinates = "{root}/dipeptide.inpcrd"
{start}[energy]
cutoff = 8.0
pairlist_distance = 9.0
electrostatics = "{electrostatics}"
[constraints]
hydrogen_bonds = {str(constraints).lower()}
rigid_water = {str(constraints).lower()}
[output]
energy_interval = {interval}
energy = "{name}.dat"
checkpoint = "{name}.h5"
{extra}{body}[[restraints]]
selection = "!:WAT & !@H*"
force_constant = 10.0
[boundary]
type = "PERIODIC"
[execution]
target = "{target_name}"
precision = "{precision.upper()}"
deterministic = true
""")
    return path


def cli_run(name, precision, stage, steps=FIRST, electrostatics="PME", constraints=True):
    path = control(name, precision, stage, steps, electrostatics, constraints)
    subprocess.run([cli, "run", str(path)], cwd=work, check=True, stdout=subprocess.DEVNULL)
    text = (work / f"{name}.dat").read_text().splitlines()
    names = text[0].lstrip("#").split()
    rows = {int(row.split()[0]): dict(zip(names, row.split())) for row in text[2:]}
    out = subprocess.run([cli, "checkpoint", "--print=positions", f"{name}.h5"], cwd=work,
                         check=True, stdout=subprocess.PIPE, text=True).stdout
    positions = np.array([[float(x) for x in line.split()[2:]] for line in out.splitlines()])
    return rows, positions


def model(electrostatics="PME", constraints=True):
    loaded = mdir.load_amber(root + "/dipeptide.prmtop", root + "/dipeptide.inpcrd")
    system, state = loaded.make_system(), loaded.make_state()
    system.cutoff, system.pairlist_distance = 0.8, 0.9
    system.truncation = mdir.Truncation.None_
    system.electrostatics = {"PME": mdir.Electrostatics.PME, "CUTOFF": mdir.Electrostatics.Cutoff}[electrostatics]
    system.rigid_hydrogen_bonds = system.rigid_water = constraints
    system.restraints = [mdir.Restraint("!:WAT & !@H*", 10.0 * KCAL_A2)]
    return system, state


def execution(precision):
    e = mdir.Execution()
    e.target, e.precision, e.deterministic = target, getattr(mdir.Precision, precision), True
    return e


def minimization(precision, steps=FIRST, electrostatics="PME", constraints=True):
    system, state = model(electrostatics, constraints)
    integrator, schedule = mdir.Integrator(), mdir.Schedule()
    integrator.minimize, schedule.steps = True, steps
    program = mdir.compile(system, state, integrator, mdir.Ensemble(), execution(precision),
                           schedule)
    return system, mdir.Simulation(program)


def row_text(row):
    return {column: "%.6f" % (row[key] / scale)
            for column, (key, scale) in MINIMIZATION.items()} | {
        "max_atom": str(row["max_force_particle"] + 1)}


def compare(label, mine, written, bound):
    """The row of Python (`row_text`) against that of `mdir run`: every
    printed digit, or each value within `bound(column)`. Returns the largest
    difference relative to the value."""
    worst = 0.0
    for column, text in mine.items():
        if bound is None:
            assert text == written[column], (label, column, text, written[column])
        elif column != "max_atom":
            a, b = float(text), float(written[column])
            assert abs(a - b) <= bound(column), (label, column, a, b, bound(column))
            worst = max(worst, abs(a - b) / max(abs(b), 1e-300))
    return worst


# `mdir run` in each precision: one run of FIRST steps (with a checkpoint,
# where the NVT stage begins) and one of FIRST + AFTER, rows at every step.
references = {}
for precision in ("Double", "Mixed"):
    rows, positions = cli_run(f"min-{precision}", precision, "min")
    longer, _ = cli_run(f"long-{precision}", precision, "min", FIRST + AFTER)
    nvt, _ = cli_run(f"nvt-{precision}", precision, "nvt")
    references[precision] = rows, positions, longer, nvt


def bound_of(precision, written, double, relative):
    """In double, `relative` of the value and the last printed digit. In
    mixed, three times the difference of `mdir run` in mixed and in double
    at that row, the error of mixed precision itself (as D196 bounds it):
    with PME the model and `mdir run` differ in mixed precision from the
    first evaluation, in dynamics as well (#105)."""
    if precision == "Double":
        return lambda c: 1e-6 + relative * abs(float(written[c]))
    return lambda c: 1e-6 + 3 * abs(float(written[c]) - float(double[c]))


# Within a part the program is that of `mdir run`. A part after the first
# begins with its own neighbor structures and evaluates its first energy
# anew, as a segment of dynamics does (D196), so after a boundary the rows
# agree within the rounding of a sum in another order, which the choices of
# the steps carry on.
for precision in ("Double", "Mixed"):
    rows, reference, longer, nvt_rows = references[precision]
    double_rows, double_positions, double_longer, double_nvt = references["Double"]
    system, simulation = minimization(precision)
    assert simulation.minimize() == FIRST and simulation.step == FIRST
    state = simulation.state()
    assert state.time == 0.0 and state.energies is None
    assert state.velocities.shape == state.positions.shape and not state.velocities.any()
    mine = row_text(state.minimization)
    if precision == "Double":
        compare("one part", mine, rows[FIRST], None)
        tolerance = 1e-11
    else:
        compare("one part", mine, rows[FIRST],
                bound_of(precision, rows[FIRST], double_rows[FIRST], 0))
        tolerance = 3 * np.abs(reference - double_positions).max()
    difference = np.abs(state.positions - reference).max()
    assert difference <= tolerance, (difference, tolerance)
    print(f"{target_name} {precision}: one part of {FIRST} steps: python {mine}; "
          f"mdir run {rows[FIRST]}; positions within {difference:.1e} nm "
          f"(tolerance {tolerance:.1e})")
    # Parts after it: AFTER more steps, one call of one part each.
    for k in range(AFTER):
        assert simulation.minimize(1) == 1
    after = simulation.state()
    step = FIRST + AFTER
    worst = compare("after", row_text(after.minimization), longer[step],
                    bound_of(precision, longer[step], double_longer[step], 1e-9))
    print(f"{target_name} {precision}: {AFTER} parts of one step after it: row at step "
          f"{after.step} within {worst:.1e} of mdir run")
    # Many parts from the start: 30 parts of one step.
    _, parts = minimization(precision)
    for k in range(30):
        parts.minimize(1)
    worst = compare("parts", row_text(parts.state().minimization), rows[30],
                    bound_of(precision, rows[30], double_rows[30], 1e-9))
    print(f"{target_name} {precision}: 30 parts of one step: row at step 30 within "
          f"{worst:.1e} of mdir run")

    # The next stage: NVT from the minimized positions (those of the
    # checkpoint of `mdir run`, which begins anew there), with velocities
    # drawn as `mdir run` draws them.
    start = mdir.InitialState()
    start.positions = state.positions
    start.cell = state.cell
    start = start.draw_velocities(system, 300.0, SEED)
    integrator, ensemble = mdir.Integrator(), mdir.Ensemble()
    integrator.timestep, ensemble.temperature, ensemble.seed = 0.0005, 300.0, SEED
    ensemble.kind = mdir.EnsembleKind.NVT
    # The reference of the restraints is that of the control file.
    system.restraint_reference = model()[1].positions
    nvt = mdir.Simulation(mdir.compile(system, start, integrator, ensemble,
                                       execution(precision), mdir.Schedule()))
    nvt.run(10, energy=True)
    energies = nvt.state().energies
    mine = {column: "%.6f" % (energies[column] / scale) for column, scale in DYNAMICS.items()}
    if precision == "Double":
        compare("nvt", mine, nvt_rows[10], None)
    else:
        compare("nvt", mine, nvt_rows[10], bound_of(precision, nvt_rows[10], double_nvt[10], 0))
    print(f"{target_name} {precision}: NVT from the minimized state, step 10: python "
          f"{mine}; mdir run {nvt_rows[10]}")

# In mixed precision with cutoff electrostatics (and no constraints, with
# which this minimization lowers the energy for longer) the model and `mdir run`
# agree from the first evaluation: one part of 20 steps is every printed
# digit, and 20 parts of one step agree within the rounding of the sums of
# forces in single precision. (The energy of a cutoff without a shift is
# not continuous, and the minimization stops lowering it after about 30
# steps.)
rows, _ = cli_run("cutoff-Mixed", "Mixed", "min", 20, "CUTOFF", False)
_, whole = minimization("Mixed", 20, "CUTOFF", False)
whole.minimize()
mine = row_text(whole.state().minimization)
compare("cutoff one part", mine, rows[20], None)
_, parts = minimization("Mixed", 20, "CUTOFF", False)
for k in range(20):
    parts.minimize(1)
worst = compare("cutoff parts", row_text(parts.state().minimization), rows[20],
                lambda c: 1e-6 + 1e-6 * abs(float(rows[20][c])))
print(f"{target_name} Mixed, cutoff: one part of 20 steps: python {mine}, every printed digit "
      f"of mdir run; 20 parts of one step within {worst:.1e} (tolerance 1e-06)")

# Refusals.
system, simulation = minimization("Double", steps=10)
expect(mdir.InputError, lambda: simulation.run(5), "minimize")
expect(mdir.InputError, lambda: simulation.minimize(-1), "nonnegative")
assert simulation.minimize(0) == 0 and simulation.state().minimization is None
assert simulation.minimize() == 10 and simulation.state().minimization is not None
dynamics = mdir.Simulation(mdir.compile(system, model()[1], mdir.Integrator(), mdir.Ensemble(),
                                        execution("Double"), mdir.Schedule()))
expect(mdir.InputError, lambda: dynamics.minimize(5), "dynamics")
assert dynamics.state().minimization is None
print("minimization refusals passed")
