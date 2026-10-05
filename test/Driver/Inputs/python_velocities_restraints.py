"""Drawn velocities and typed restraints in the Python model
(D198): the velocities of `mdir run` bit for bit,
restraints against `[[restraints]]` through `mdir run`, copies, versions,
and refusals."""
import pathlib
import subprocess
import sys
import warnings

import numpy as np
import mdir

root = sys.argv[1]
target_name = sys.argv[2]
target = getattr(mdir.Target, target_name)
cli = sys.argv[3]
model_test = sys.argv[4]
work = pathlib.Path(sys.argv[5])
SEED = 271828
BOLTZMANN = 0.0083144626181532  # kJ/mol/K
# The restraints of the control file and of the model: the heavy atoms of
# the peptide about their center, its alanine once more (the constants
# add), and six water oxygens each with the cell (D74, D124).
RESTRAINTS = (("!:WAT & !@H*", 10.0, "CENTER"), (":ALA & !@H*", 2.5, "CENTER"),
              (":4-9@O", 5.0, "ALL"))
KCAL_A2 = 4.184 / (0.1 * 0.1)  # kJ/mol/nm^2 per kcal/mol/A^2, as the CLI converts


def expect(error, call, text=""):
    try:
        call()
    except error as exc:
        assert text in str(exc), str(exc)
        return
    raise AssertionError(f"expected {error.__name__}")


def control(name, kind, precision, constraints=True, restraints=RESTRAINTS, steps=20):
    thermostat = '[thermostat]\nmethod = "V-RESCALE"\ninterval = 10' if kind != "NVE" else ""
    barostat = '[barostat]\nmethod = "C-RESCALE"\ninterval = 10' if kind == "NPT" else ""
    tables = "".join(f'[[restraints]]\nselection = "{s}"\nforce_constant = {k}\n'
                     f'reference_scaling = "{scaling}"\n' for s, k, scaling in restraints)
    path = work / f"{name}.toml"
    path.write_text(f"""[input]
topology = "{root}/dipeptide.prmtop"
coordinates = "{root}/dipeptide.inpcrd"
[energy]
cutoff = 8.0
pairlist_distance = 9.0
electrostatics = "CUTOFF"
[constraints]
hydrogen_bonds = {str(constraints).lower()}
rigid_water = {str(constraints).lower()}
[dynamics]
time_step = 0.0005
steps = {steps}
seed = {SEED}
[output]
energy_interval = 10
energy = "{name}.dat"
checkpoint = "{name}.h5"
checkpoint_interval = {steps}
[ensemble]
ensemble = "{kind}"
temperature = 300.0
{thermostat}
{barostat}
[boundary]
type = "PERIODIC"
[execution]
target = "{target_name}"
precision = "{precision.upper()}"
deterministic = true
{tables}""")
    return path


def model(constraints=True, restraints=RESTRAINTS):
    loaded = mdir.load_amber(root + "/dipeptide.prmtop", root + "/dipeptide.inpcrd")
    system, state = loaded.make_system(), loaded.make_state()
    system.cutoff, system.pairlist_distance = 0.8, 0.9
    system.truncation = mdir.Truncation.None_
    system.rigid_hydrogen_bonds = system.rigid_water = constraints
    system.restraints = [mdir.Restraint(s, k * KCAL_A2, getattr(mdir.ReferenceScaling, scaling.title()))
                         for s, k, scaling in restraints]
    return system, state


# Drawn velocities: those that `mdir run` draws for the same control file
# (its readSystem, then assignVelocities, written by mdir-model-test), bit
# for bit, with and without constraints.
for constraints in (False, True):
    path = control(f"draw-{constraints}", "NVT", "Double", constraints)
    cli_bytes = subprocess.run([model_test, str(path), "--drawn"], check=True,
                               stdout=subprocess.PIPE).stdout
    reference = np.frombuffer(cli_bytes, dtype=np.float64).reshape(-1, 3)
    system, state = model(constraints)
    before = state.positions
    drawn = state.draw_velocities(system, 300.0, SEED)
    differing = np.count_nonzero(drawn.velocities.view(np.uint64) != reference.view(np.uint64))
    assert differing == 0, differing
    assert drawn is not state and state.velocities.shape == (0, 3)
    assert np.array_equal(drawn.positions, before)
    assert np.array_equal(drawn.cell.vectors, state.cell.vectors)
    count = drawn.velocities.shape[0]
    print(f"drawn velocities (constraints {constraints}): {count} particles, "
          f"{differing} differing bits against mdir run")
    again = state.draw_velocities(system, 300.0, SEED)
    assert np.array_equal(again.velocities, drawn.velocities)
    other = state.draw_velocities(system, 300.0, SEED + 1)
    assert not np.array_equal(other.velocities, drawn.velocities)
    hot = state.draw_velocities(system, 600.0, SEED)
    assert np.allclose(hot.velocities, np.sqrt(2.0) * drawn.velocities, rtol=1e-12, atol=0)
    cold = state.draw_velocities(system, 0.0, SEED)
    assert not cold.velocities.any()

system, state = model()
for temperature in (-1.0, float("nan"), float("inf")):
    expect(mdir.InputError, lambda: state.draw_velocities(system, temperature, SEED), "temperature")
for seed in (-1, 2**63):
    expect(mdir.InputError, lambda: state.draw_velocities(system, 300.0, seed), "seed")
expect(mdir.InputError, lambda: state.draw_velocities(None, 300.0, SEED), "System")
broken = mdir.InitialState()
broken.positions = state.positions[:10]
expect(mdir.InputError, lambda: broken.draw_velocities(system, 300.0, SEED))
print("draw_velocities refusals passed")

# The typed list: defaults, copies, versions, and strict references.
restraint = mdir.Restraint()
assert restraint.selection == "" and restraint.force_constant == 0.0
assert restraint.reference_scaling == mdir.ReferenceScaling.Center
copies = system.restraints
assert [r.selection for r in copies] == [r[0] for r in RESTRAINTS]
copies[0].force_constant = 1.0
assert system.restraints[0].force_constant == 10.0 * KCAL_A2
assert system.restraint_reference.shape == (0, 3)
assert not system.restraint_reference.flags.writeable
program_inputs = (mdir.Integrator(), mdir.Ensemble(), mdir.Execution(), mdir.Schedule())
program = mdir.compile(system, state, *program_inputs)
system.restraints = system.restraints
assert program.stale
program = mdir.compile(system, state, *program_inputs)
system.restraint_reference = state.positions
assert program.stale and np.array_equal(system.restraint_reference, state.positions)
expect(mdir.InputError, lambda: setattr(system, "restraint_reference", state.positions[:5]), "expected shape")
expect(mdir.InputError, lambda: setattr(system, "restraint_reference", state.positions.astype(np.float32)))
expect(mdir.InputError, lambda: setattr(system, "restraint_reference", state.positions.tolist()))
bad = state.positions.copy()
bad[3, 1] = np.nan
expect(mdir.InputError, lambda: setattr(system, "restraint_reference", bad))
assert np.array_equal(system.restraint_reference, state.positions)
system.restraint_reference = np.zeros((0, 3))
expect(TypeError, lambda: setattr(system, "restraints", [1]))


def refuse(restraints, text):
    system.restraints = restraints
    expect(mdir.InputError, lambda: mdir.compile(system, state, *program_inputs), text)
    expect(mdir.InputError, lambda: state.draw_velocities(system, 300.0, SEED), text)


refuse([mdir.Restraint("", 1.0)], "selection")
refuse([mdir.Restraint("@CA", 0.0)], "force constant")
refuse([mdir.Restraint("@CA", float("nan"))], "force constant")
refuse([mdir.Restraint("@CA &", 1.0)], "")
refuse([mdir.Restraint("@CA", 1.0), mdir.Restraint("@CA", 1.0, mdir.ReferenceScaling.All)],
       "reference_scaling")
# A selection of nothing restrains nothing: a warning, as `mdir run` gives.
system.restraints = [mdir.Restraint(":XYZ", 1.0)]
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    mdir.compile(system, state, *program_inputs)
assert len(caught) == 1 and issubclass(caught[0].category, UserWarning), caught
assert "restrains nothing" in str(caught[0].message)
print("restraint copies, versions and refusals passed")

# A run with drawn velocities and restraints against `mdir run` of the same
# control file: the rows of its energies to every printed digit, and its
# checkpoint at step 20.
COLUMNS = {"total": 4.184, "potential": 4.184, "kinetic": 4.184, "temperature": 1.0,
           "virial": 4.184, "pressure": 1.01325, "conserved": 4.184, "volume": 1e-3}


def cli_run(name, kind, precision, restraints=RESTRAINTS):
    path = control(name, kind, precision, restraints=restraints)
    subprocess.run([cli, "run", str(path)], cwd=work, check=True, stdout=subprocess.DEVNULL)
    text = (work / f"{name}.dat").read_text().splitlines()
    names = text[0].lstrip("#").split()
    rows = {int(row.split()[0]): dict(zip(names, row.split())) for row in text[2:]}
    state = {}
    for field in ("positions", "velocities", "forces"):
        out = subprocess.run([cli, "checkpoint", f"--print={field}", f"{name}.h5"], cwd=work,
                             check=True, stdout=subprocess.PIPE, text=True).stdout
        state[field] = np.array([[float(x) for x in line.split()[2:]] for line in out.splitlines()])
    return rows, state


def python_run(kind, precision, restraints=RESTRAINTS, reference=None):
    system, state = model(restraints=restraints)
    if reference is not None:
        system.restraint_reference = reference
    state = state.draw_velocities(system, 300.0, SEED)
    integrator, ensemble, execution = mdir.Integrator(), mdir.Ensemble(), mdir.Execution()
    integrator.timestep, ensemble.temperature, ensemble.seed = 0.0005, 300.0, SEED
    ensemble.kind = getattr(mdir.EnsembleKind, kind)
    ensemble.com_period = 10
    execution.target, execution.precision = target, getattr(mdir.Precision, precision)
    execution.deterministic = True
    simulation = mdir.Simulation(mdir.compile(system, state, integrator, ensemble, execution,
                                              mdir.Schedule()))
    simulation.run(10, energy=True)
    first = simulation.state().energies
    simulation.run(10, energy=True)
    return first, simulation.state()


def same_row(energies, row):
    for column, written in row.items():
        if column in COLUMNS:
            mine = "%.6f" % (energies[column] / COLUMNS[column])
            assert mine == written, (column, mine, written)


def close_row(energies, row, relative):
    for column, written in row.items():
        if column in COLUMNS:
            mine, value = energies[column] / COLUMNS[column], float(written)
            assert abs(mine - value) <= 1e-6 + relative * abs(value), (column, mine, value)


# Tolerances of the state at step 20, which the model reaches in two parts
# (the second begins with new structures, D196) and `mdir run` in one. In
# double, as python-segments.md budgets them: c eps max|q| with c = 2 x 20
# for positions (one update a step in each run) and c = 4 (N - 1) for
# velocities and forces (two orders of a sum over at most N - 1 neighbors,
# times two). In mixed, 3 E, with E the difference of `mdir run` in mixed and
# in double at that step under NVT. Under NPT the scalings of the cell at
# steps 10 and 20 take the pressure of each precision and amplify that
# difference (to 32 kJ/mol/nm in the forces), while the model and `mdir run`
# differ only by the boundary of a part, as under NVT: NPT takes the E of NVT.
EPS = np.finfo(np.float64).eps
FIELDS = ("positions", "velocities", "forces")
mixed_error = {}
for kind in ("NVT", "NPT"):
    for precision in ("Double", "Mixed"):
        rows, reference = cli_run(f"{kind}-{precision}".lower(), kind, precision)
        first, last = python_run(kind, precision)
        # Step 10 is in the first part, as in `mdir run`: every printed digit,
        # except on a GPU in mixed precision with constraints, which differs
        # in the last digits with or without restraints (#97).
        if target_name == "GPU" and precision == "Mixed":
            close_row(first, rows[10], 1e-6)
        else:
            same_row(first, rows[10])
        close_row(last.energies, rows[20], 1e-9 if precision == "Double" else 1e-5)
        line = [f"{kind} {precision}"]
        for field in FIELDS:
            difference = np.abs(getattr(last, field) - reference[field]).max()
            if precision == "Double":
                count = reference[field].shape[0]
                c = 2 * 20 if field == "positions" else 4 * (count - 1)
                tolerance = c * EPS * np.abs(reference[field]).max()
            else:
                if kind == "NVT":
                    mixed_error[field] = np.abs(reference[field] - double_reference[field]).max()
                tolerance = 3 * mixed_error[field]
            assert difference <= tolerance, (kind, precision, field, difference, tolerance)
            line.append(f"{field} {difference:.3e} (tolerance {tolerance:.3e})")
        print("; ".join(line))
        if precision == "Double":
            double_reference = reference
            # The restraints are on: without them the potential at step 10
            # differs by far more than the printed digits that matched.
            free = python_run(kind, precision, restraints=())[0]
            effect = abs(first["potential"] - free["potential"])
            assert effect > 0.1, effect
            print(f"{kind}: the restraints change the potential at step 10 by {effect:.3f} kJ/mol")
            # The default reference is the positions of the state given to
            # compile; the same positions given explicitly change no bit.
            system, start = model()
            _, explicit = python_run(kind, precision, reference=start.positions)
            for field in FIELDS:
                assert np.array_equal(getattr(explicit, field), getattr(last, field)), field
            # References of the peptide moved by 0.01 nm along x add about
            # k (0.01 nm)^2 = 0.42 kJ/mol for each of its heavy atoms.
            shifted = start.positions.copy()
            shifted[:22, 0] += 0.01
            moved = python_run(kind, precision, reference=shifted)[0]
            assert abs(moved["potential"] - first["potential"]) > 1.0

print("drawn velocities and restraints against mdir run passed")
