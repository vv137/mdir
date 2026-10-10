"""The coupling and the work of the barostat in a Python simulation
(D[python-barostat], #275, docs/python-segments.md) against `mdir run` of
the same control, to the bit in the deterministic mode.

Usage: python_barostat.py INPUTS TARGET PRECISION MDIR WORK SCENARIO [COUPLING...]

  refusals      the refusals and the types of `Ensemble.barostat_coupling` and
                `Ensemble.barostat_work`, `Program.plan`, and the fingerprint; needs
                no device.
  orthorhombic  the dipeptide in water of `Inputs/dipeptide` (1,168
                particles, PME on 32^3, SHAKE and SETTLE).
  triclinic     403 rigid waters in the rhombic dodecahedron of
                `Inputs/triclinic` (PME on 28^3, SETTLE).
  keys          the dipeptide with the other keys of the couplings: a
                compressibility of each axis under the anisotropic
                coupling, one of them 0, which keeps its axis; and the
                compressibility of z, 0 and not, with a surface tension on
                one and on two surfaces under the semi-isotropic coupling.
                40 steps in parts against `mdir run` with a checkpoint at
                step 20, the state and the cell to the bit, the files byte
                for byte, the fingerprint, and `mdir run --continue`.
  every-step    the dipeptide with a coupling period of 1: the exact and
                the first-order work, which close a period with one step,
                scale the cell at every step (the work of Trotter type
                there is refused by a simulation); 40 steps in parts of 13
                and 27 against `mdir run`, the state and the cell to the
                bit.

For each coupling (ISOTROPIC, SEMI_ISOTROPIC, ANISOTROPIC; all, or those
named) and each work that it takes (TROTTER, TROTTER_FIRST_ORDER, EXACT, and
with the isotropic coupling FIRST_ORDER), 40 steps of velocity Verlet at
2 fs with stochastic velocity rescaling and stochastic cell rescaling
every 10 steps, at 2000 atm so that the cell moves:

- a Python simulation in parts of 10 and 30 steps, with reporters, against
  `mdir run`: the positions, the velocities, the forces, and the cell with
  its tilts to the bit, and the energy file and the DCD trajectory, whose
  record of the cell is of each frame, byte for byte;
- the cell of the last frame against the cell of the state, and how the
  axes moved: together, x and y together, or each by its own;
- `mdir run` -> Python and Python -> `mdir run --continue` at step 20, to
  the same bits and bytes, with the fingerprint of `mdir run`.

A checkpoint ends a segment of `mdir run`, as it ends the activation of a
simulation (D215, D218), so each run here has its checkpoint at step 20.
"""
import pathlib
import shutil
import subprocess
import sys
import warnings

import numpy as np
import mdir

inputs, target_name, precision, cli = sys.argv[1:5]
work = pathlib.Path(sys.argv[5])
scenario = sys.argv[6]
chosen = sys.argv[7:]
SEED = 2024
STEPS, PERIOD, TIMESTEP = 40, 10, 0.002
PRESSURE_ATM = 2000.0
FIELDS = ("positions", "velocities", "forces")
label = f"{target_name} {precision}"
COUPLINGS = {"ISOTROPIC": "Isotropic", "SEMI_ISOTROPIC": "SemiIsotropic",
             "ANISOTROPIC": "Anisotropic"}
WORKS = {"TROTTER": "Trotter", "TROTTER_FIRST_ORDER": "TrotterFirstOrder",
         "EXACT": "Exact", "FIRST_ORDER": "FirstOrder"}
TRICLINIC = scenario == "triclinic"
EVERY_STEP = (("ISOTROPIC", "EXACT"), ("ISOTROPIC", "FIRST_ORDER"),
              ("SEMI_ISOTROPIC", "EXACT"), ("ANISOTROPIC", "EXACT"))


def works_of(coupling):
    return [w for w in WORKS if w != "FIRST_ORDER" or coupling == "ISOTROPIC"]


def control(path, coupling, work_key, period=PERIOD, steps=STEPS, given=True, extra=""):
    """The control file of what `model` gives, with a checkpoint at half of
    its steps. With `given` false the two keys are left out."""
    name = path.stem
    source = (f'topology = "{inputs}/triclinic/water.top"\n'
              f'coordinates = "{inputs}/triclinic/dodecahedron.gro"\nformat = "GROMACS"'
              if TRICLINIC else
              f'topology = "{inputs}/dipeptide/dipeptide.prmtop"\n'
              f'coordinates = "{inputs}/dipeptide/dipeptide.inpcrd"')
    grid = 28 if TRICLINIC else 32
    keys = f'coupling = "{coupling}"\nwork = "{work_key}"\n' if given else ""
    path.write_text(f"""[input]
{source}
[energy]
cutoff = 8.0
pairlist_distance = 9.0
electrostatics = "PME"
[pme]
grid = [{grid}, {grid}, {grid}]
[constraints]
{"" if TRICLINIC else "hydrogen_bonds = true"}
rigid_water = true
[dynamics]
integrator = "VELOCITY_VERLET"
time_step = {TIMESTEP}
steps = {steps}
seed = {SEED}
center_of_mass_interval = {period}
[output]
energy_interval = {period}
energy = "{name}.dat"
trajectory = "{name}.dcd"
trajectory_interval = {period}
checkpoint = "{name}.h5"
checkpoint_interval = {steps // 2}
[ensemble]
ensemble = "NPT"
temperature = 300.0
pressure = {PRESSURE_ATM}
[thermostat]
method = "V-RESCALE"
interval = {period}
[barostat]
method = "C-RESCALE"
{keys}{extra}[boundary]
type = "PERIODIC"
[execution]
target = "{target_name}"
precision = "{precision.upper()}"
deterministic = true
""")
    return path


def parts_of_model(coupling=None, work_key=None, period=PERIOD, kind="NPT", settings=None):
    if TRICLINIC:
        loaded = mdir.load_gromacs(inputs + "/triclinic/water.top",
                                   inputs + "/triclinic/dodecahedron.gro")
    else:
        loaded = mdir.load_amber(inputs + "/dipeptide/dipeptide.prmtop",
                                 inputs + "/dipeptide/dipeptide.inpcrd")
    system, state = loaded.make_system(), loaded.make_state()
    system.cutoff, system.pairlist_distance = 0.8, 0.9
    system.truncation = mdir.Truncation.None_
    system.electrostatics = mdir.Electrostatics.PME
    system.pme_grid = [28 if TRICLINIC else 32] * 3
    system.periodic = True
    system.rigid_water = True
    if not TRICLINIC:
        system.rigid_hydrogen_bonds = True
    state = state.draw_velocities(system, 300.0, SEED)
    integrator, ensemble, execution = mdir.Integrator(), mdir.Ensemble(), mdir.Execution()
    integrator.method = mdir.IntegratorMethod.VelocityVerlet
    integrator.timestep = TIMESTEP
    ensemble.seed = SEED
    ensemble.kind = getattr(mdir.EnsembleKind, kind)
    ensemble.temperature = 300.0
    ensemble.pressure = PRESSURE_ATM * 1.01325  # bar
    ensemble.com_period = ensemble.coupling_period = period
    if coupling:
        ensemble.barostat_coupling = getattr(mdir.BarostatCoupling, COUPLINGS[coupling])
    if work_key:
        ensemble.barostat_work = getattr(mdir.BarostatWork, WORKS[work_key])
    for name, value in (settings or {}).items():
        setattr(ensemble, name, value)
    execution.target = getattr(mdir.Target, target_name)
    execution.precision = getattr(mdir.Precision, precision)
    execution.deterministic = True
    return system, state, integrator, ensemble, execution


def model(*arguments, **keywords):
    schedule = mdir.Schedule()
    schedule.steps, schedule.energy_period = STEPS, keywords.get("period", PERIOD)
    return mdir.compile(*parts_of_model(*arguments, **keywords), schedule)


def run_cli(path, *options):
    result = subprocess.run([cli, "run", *options, str(path)], cwd=path.parent,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert result.returncode == 0, result.stdout
    return result.stdout


def expect(error, call, *texts):
    try:
        call()
    except error as exc:
        for text in texts:
            assert text in str(exc), str(exc)
        return str(exc)
    raise AssertionError(f"expected {error.__name__}")


def continued(program, checkpoint):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        simulation = mdir.Simulation(program, checkpoint=str(checkpoint))
    assert not caught, [str(w.message) for w in caught]
    return simulation


def compare(what, state, reference):
    """To the bit: the state and the cell with its tilts."""
    for field in FIELDS:
        mine, theirs = getattr(state, field), getattr(reference, field)
        differing = np.count_nonzero(mine.view(np.uint64) != theirs.view(np.uint64))
        assert differing == 0, (what, field, differing, np.abs(mine - theirs).max())
    for field in ("diagonal", "tilt"):
        mine, theirs = getattr(state.cell, field), getattr(reference.cell, field)
        assert np.array_equal(mine, theirs), (what, field, mine, theirs)


def lower(record):
    """The lower-triangular cell of a DCD record, in nm."""
    a, cg, b, cb, ca, c = record
    bx, by = b * cg, b * np.sqrt(1.0 - cg * cg)
    cx = c * cb
    cy = c * (ca - cb * cg) / np.sqrt(1.0 - cg * cg)
    cz = np.sqrt(c * c - cx * cx - cy * cy)
    return np.array([a, by, cz, bx, cx, cy]) / 10.0


def dcd_cells(path):
    data, cells, i = pathlib.Path(path).read_bytes(), [], 0
    while i < len(data):
        n = int.from_bytes(data[i:i + 4], "little")
        if n == 48:
            cells.append(lower(np.frombuffer(data[i + 4:i + 52], dtype=np.float64)))
        i += n + 8
    return np.array(cells)


def with_reporters(simulation, directory, checkpoint=True):
    simulation.reporters = [mdir.EnergyReporter(str(directory / "run.dat"), PERIOD),
                            mdir.TrajectoryReporter(str(directory / "run.dcd"), PERIOD)]
    if checkpoint:
        simulation.reporters.append(
            mdir.CheckpointReporter(str(directory / "run.h5"), STEPS // 2))


def same_files(directory, whole):
    for name in ("run.dat", "run.dcd"):
        assert (directory / name).read_bytes() == (whole / name).read_bytes(), (directory, name)


def axes(first, last, coupling):
    """The strains of the three axes, and whether they are those of the
    coupling: equal, x and y equal and z another, or all three apart."""
    strain = np.log(np.asarray(last.diagonal) / np.asarray(first.diagonal))
    assert np.abs(strain).max() > 1e-5, strain
    near = 1e-12
    xy, xz, yz = (abs(strain[0] - strain[1]), abs(strain[0] - strain[2]),
                  abs(strain[1] - strain[2]))
    if coupling == "ISOTROPIC":
        assert max(xy, xz, yz) < near, strain
    elif coupling == "SEMI_ISOTROPIC":
        assert xy < near and xz > 1e-7, strain
    else:
        assert min(xy, xz, yz) > 1e-7, strain
    return strain


def one_case(coupling, work_key):
    tag = f"{coupling.lower()}-{work_key.lower()}"
    case = f"{label} {scenario} {coupling} {work_key}"
    whole = work / tag / "whole"
    whole.mkdir(parents=True)
    run_cli(control(whole / "run.toml", coupling, work_key))
    reference = mdir.read_checkpoint(str(whole / "run.h5"))
    middle = mdir.read_checkpoint(str(whole / "run.h5.prev"))
    assert reference.step == STEPS and middle.step == STEPS // 2
    program = model(coupling, work_key)
    plan = program.plan["barostat"]
    assert plan == {"coupling": getattr(mdir.BarostatCoupling, COUPLINGS[coupling]),
                    "work": getattr(mdir.BarostatWork, WORKS[work_key])}, plan
    start = parts_of_model()[1].cell

    # Python in parts, with the checkpoint of `mdir run` at step 20.
    here = work / tag / "python"
    here.mkdir()
    simulation = mdir.Simulation(program)
    with_reporters(simulation, here)
    for part in (10, 30):
        simulation.run(part)
    state = simulation.state()
    simulation.close_reporters()
    compare(case, state, reference)
    same_files(here, whole)
    written = mdir.read_checkpoint(str(here / "run.h5"))
    assert written.step == STEPS and written.front_end == "python"
    assert written.fingerprint == reference.fingerprint, (
        set(written.fingerprint) ^ set(reference.fingerprint))
    cells = dcd_cells(here / "run.dcd")
    assert len(cells) == STEPS // PERIOD
    last = np.concatenate([state.cell.diagonal, state.cell.tilt])
    assert np.abs(cells[-1] - last).max() < 1e-12, (cells[-1], last)
    strain = axes(start, state.cell, coupling)

    # `mdir run` -> Python at step 20.
    part = work / tag / "cli-to-python"
    part.mkdir()
    shutil.copy(whole / "run.h5.prev", part / "run.h5")
    for name in ("run.dat", "run.dcd"):
        shutil.copy(whole / name, part / name)
    simulation = continued(program, part / "run.h5")
    assert simulation.step == STEPS // 2
    assert np.array_equal(simulation.state().cell.diagonal, middle.cell.diagonal)
    with_reporters(simulation, part, checkpoint=False)
    simulation.run(STEPS // 2)
    compare(case + ", mdir run -> Python", simulation.state(), reference)
    simulation.close_reporters()
    same_files(part, whole)

    # Python -> `mdir run --continue` at step 20.
    part = work / tag / "python-to-cli"
    part.mkdir()
    path = control(part / "run.toml", coupling, work_key)
    first = mdir.Simulation(program)
    with_reporters(first, part)
    first.run(STEPS // 2)
    first.close_reporters()
    taken = mdir.read_checkpoint(str(part / "run.h5"))
    assert taken.step == STEPS // 2 and taken.front_end == "python"
    assert np.array_equal(taken.cell.diagonal, middle.cell.diagonal)
    assert np.array_equal(taken.cell.tilt, middle.cell.tilt)
    # Another coupling or work is other coupling: neither front end
    # continues the run with it (D172, D223), and each names the key.
    if (coupling, work_key) == ("ISOTROPIC", "TROTTER"):
        shutil.copy(part / "run.h5", part / "other.h5")
        for other, other_work, key in (("ANISOTROPIC", "TROTTER", "[barostat] coupling"),
                                       ("ISOTROPIC", "EXACT", "[barostat] work")):
            expect(mdir.InputError,
                   lambda: mdir.Simulation(model(other, other_work),
                                           checkpoint=str(part / "run.h5")),
                   "other physics or coupling", key)
            refused = subprocess.run(
                [cli, "run", "--continue", str(control(part / "other.toml", other, other_work))],
                cwd=part, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            assert refused.returncode != 0 and "other physics or coupling" in refused.stdout \
                and key in refused.stdout, refused.stdout
        print(f"{case}: a continuation with another coupling or work is refused by both "
              f"front ends", flush=True)
    log = run_cli(path, "--continue")
    assert "other physics" not in log, log
    compare(case + ", Python -> mdir run", mdir.read_checkpoint(str(part / "run.h5")),
            reference)
    same_files(part, whole)
    print(f"{case}: the state, the cell, the energy file, and the trajectory of mdir run, "
          f"in parts and across a checkpoint in both directions", flush=True)
    print(f"{case}: strains of the axes {strain[0]:.3e} {strain[1]:.3e} {strain[2]:.3e}",
          file=sys.stderr, flush=True)


def cases():
    count = 0
    for coupling in chosen or list(COUPLINGS):
        for work_key in works_of(coupling):
            one_case(coupling, work_key)
            count += 1
    print(f"barostat {scenario} {label}: {count} cases passed")


def refusals():
    ensemble = mdir.Ensemble()
    assert ensemble.barostat_coupling == mdir.BarostatCoupling.Isotropic
    assert ensemble.barostat_work == mdir.BarostatWork.Trotter
    for name, wrong in (("barostat_coupling", "SEMI_ISOTROPIC"), ("barostat_coupling", 1),
                        ("barostat_coupling", None),
                        ("barostat_coupling", mdir.BarostatWork.Exact), ("barostat_work", "EXACT"),
                        ("barostat_work", 2),
                        ("barostat_work", mdir.BarostatCoupling.Anisotropic)):
        try:
            setattr(ensemble, name, wrong)
        except TypeError:
            continue
        raise AssertionError(f"Ensemble.{name} took {wrong!r}")
    # The words of the control file, with `model` for its path.
    words = ("model: 'work = \"FIRST_ORDER\"' counts the work from the trace of the virial of "
             "the step with twice the internal kinetic energy, which holds for the trace only; "
             "with 'coupling = \"SEMI_ISOTROPIC\"' or \"ANISOTROPIC\" use \"TROTTER\", "
             "\"TROTTER_FIRST_ORDER\", or \"EXACT\"")
    for coupling in ("SEMI_ISOTROPIC", "ANISOTROPIC"):
        expect(mdir.InputError, lambda: model(coupling, "FIRST_ORDER"), words)
    # The other keys of the couplings: their defaults, their types, and
    # the refusals of the control file.
    assert ensemble.compressibility == 4.5e-5 and ensemble.compressibility_z is None
    assert ensemble.surface_tension == 0.0 and ensemble.surfaces == 2
    ensemble.compressibility = (1e-5, 0.0, 2e-5)
    assert ensemble.compressibility == (1e-5, 0.0, 2e-5)
    ensemble.compressibility = 3e-5
    assert ensemble.compressibility == 3e-5
    expect(mdir.InputError, lambda: setattr(ensemble, "compressibility", (1e-5, 2e-5)),
           "expected one number, or three")
    expect(TypeError, lambda: setattr(ensemble, "compressibility", "4.5e-5"))
    axes = {"compressibility": (1e-5, 0.0, 2e-5)}
    for coupling in ("ISOTROPIC", "SEMI_ISOTROPIC"):
        expect(mdir.InputError, lambda: model(coupling, "TROTTER", settings=axes),
               "model: a 'compressibility' of each axis needs 'coupling = \"ANISOTROPIC\"'; "
               "give one number")
    expect(mdir.InputError,
           lambda: model("ANISOTROPIC", "TROTTER", settings={"compressibility": (0.0, 0.0, 0.0)}),
           "model: a barostat whose compressibilities are all 0 keeps the cell; give one "
           "that is not 0")
    expect(mdir.InputError,
           lambda: model("ANISOTROPIC", "TROTTER", settings={"compressibility": (1e-5, -1e-5, 0.0)}),
           "Ensemble.compressibility: expected one number, or three that are not negative")
    for name, value in (("compressibility_z", 0.0), ("surface_tension", 100.0), ("surfaces", 1)):
        for coupling in ("ISOTROPIC", "ANISOTROPIC"):
            expect(mdir.InputError, lambda: model(coupling, "TROTTER", settings={name: value}),
                   f"model: '{name}' needs 'coupling = \"SEMI_ISOTROPIC\"'")
    expect(mdir.InputError,
           lambda: model("SEMI_ISOTROPIC", "TROTTER", settings={"compressibility_z": -1e-5}),
           "model: expected 0 or a positive number for 'compressibility_z'")
    expect(mdir.InputError,
           lambda: model("SEMI_ISOTROPIC", "TROTTER", settings={"surfaces": 0}),
           "Ensemble.surfaces must be at least 1")
    # Without a barostat the two are not read, as `tau_p` is not.
    program = model("ANISOTROPIC", "FIRST_ORDER", kind="NVT")
    assert program.plan["barostat"] is None
    # The plan, and the defaults.
    assert model().plan["barostat"] == {"coupling": mdir.BarostatCoupling.Isotropic,
                                        "work": mdir.BarostatWork.Trotter}
    # The work of Trotter type at every step is refused by a simulation,
    # as before; the others scale at every step.
    for work_key in ("TROTTER", "TROTTER_FIRST_ORDER"):
        program = model("ANISOTROPIC", work_key, period=1)
        expect(mdir.UnsupportedError, lambda: mdir.Simulation(program),
               "a simulation with a barostat that scales the cell every step "
               "(coupling period 1) is not supported yet")
    print("barostat refusals passed")


ATM = 1.01325  # bar
# The keys of the control file (1/atm, dyn/cm) and the fields of the model
# (1/bar, bar nm) that give the same doubles; the last entry is the axis
# that does not move, or None.
KEYS = (
    ("ANISOTROPIC", "EXACT", "compressibility = [9.0e-5, 0.0, 3.0e-5]\n",
     {"compressibility": (9.0e-5 / ATM, 0.0, 3.0e-5 / ATM)}, 1),
    ("ANISOTROPIC", "TROTTER", "compressibility = [2.0e-5, 6.0e-5, 1.0e-4]\n",
     {"compressibility": (2.0e-5 / ATM, 6.0e-5 / ATM, 1.0e-4 / ATM)}, None),
    ("SEMI_ISOTROPIC", "TROTTER",
     "compressibility_z = 0.0\nsurface_tension = 30.0\nsurfaces = 1\n",
     {"compressibility_z": 0.0, "surface_tension": 300.0, "surfaces": 1}, 2),
    ("SEMI_ISOTROPIC", "EXACT",
     "compressibility = 6.0e-5\ncompressibility_z = 2.0e-5\nsurface_tension = 25.0\n",
     {"compressibility": 6.0e-5 / ATM, "compressibility_z": 2.0e-5 / ATM,
      "surface_tension": 250.0}, None),
)


def keys():
    for number, (coupling, work_key, extra, settings, still) in enumerate(KEYS):
        case = f"{label} keys {coupling} {work_key} {' '.join(settings)}"
        whole = work / f"case-{number}" / "whole"
        whole.mkdir(parents=True)
        run_cli(control(whole / "run.toml", coupling, work_key, extra=extra))
        reference = mdir.read_checkpoint(str(whole / "run.h5"))
        middle = mdir.read_checkpoint(str(whole / "run.h5.prev"))
        program = model(coupling, work_key, settings=settings)
        here = work / f"case-{number}" / "python"
        here.mkdir()
        simulation = mdir.Simulation(program)
        with_reporters(simulation, here)
        for part in (10, 30):
            simulation.run(part)
        state = simulation.state()
        simulation.close_reporters()
        compare(case, state, reference)
        same_files(here, whole)
        written = mdir.read_checkpoint(str(here / "run.h5"))
        assert written.fingerprint == reference.fingerprint, (
            set(written.fingerprint) ^ set(reference.fingerprint))
        start = parts_of_model()[1].cell
        strain = np.log(np.asarray(state.cell.diagonal) / np.asarray(start.diagonal))
        assert np.abs(strain).max() > 1e-5, strain
        if still is not None:
            assert strain[still] == 0.0, strain
        # Python -> `mdir run --continue` at step 20.
        part = work / f"case-{number}" / "python-to-cli"
        part.mkdir()
        path = control(part / "run.toml", coupling, work_key, extra=extra)
        first = mdir.Simulation(program)
        with_reporters(first, part)
        first.run(STEPS // 2)
        first.close_reporters()
        taken = mdir.read_checkpoint(str(part / "run.h5"))
        assert np.array_equal(taken.cell.diagonal, middle.cell.diagonal)
        log = run_cli(path, "--continue")
        assert "other physics" not in log, log
        compare(case + ", Python -> mdir run", mdir.read_checkpoint(str(part / "run.h5")),
                reference)
        same_files(part, whole)
        print(f"{case}: the state, the cell, and the files of mdir run", flush=True)
        print(f"{case}: strains of the axes {strain[0]:.3e} {strain[1]:.3e} {strain[2]:.3e}",
              file=sys.stderr, flush=True)
    print(f"barostat keys {label}: {len(KEYS)} cases passed")


def every_step():
    for coupling, work_key in EVERY_STEP:
        here = work / f"{coupling.lower()}-{work_key.lower()}"
        here.mkdir(parents=True)
        run_cli(control(here / "run.toml", coupling, work_key, period=1))
        reference = mdir.read_checkpoint(str(here / "run.h5"))
        assert reference.step == STEPS
        simulation = mdir.Simulation(model(coupling, work_key, period=1))
        simulation.reporters = [mdir.CheckpointReporter(str(here / "python.h5"), STEPS // 2)]
        for part in (13, 27):
            simulation.run(part)
        state = simulation.state()
        compare(f"{label} every step {coupling} {work_key}", state, reference)
        axes(parts_of_model()[1].cell, state.cell, coupling)
        print(f"{label} every step {coupling} {work_key}: the state and the cell of mdir run",
              flush=True)
    print(f"barostat every-step {label}: {len(EVERY_STEP)} cases passed")


if scenario == "refusals":
    refusals()
elif scenario == "every-step":
    every_step()
elif scenario == "keys":
    keys()
else:
    cases()
