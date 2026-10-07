"""Checkpoints of a Python simulation (D[python-checkpoints],
docs/python-checkpoints.md): `mdir run` -> Python and Python -> `mdir run
--continue` against the uninterrupted run, Python -> Python with tunables,
the same run against a changed stage, reporters across a continuation, and
corrupted files.

Usage: python_checkpoints.py ROOT TARGET PRECISION MDIR WORK SCENARIO. Each
scenario is one test file, so that the files run side by side."""
import pathlib
import shutil
import subprocess
import sys
import warnings

import numpy as np
import mdir

root = sys.argv[1]
target_name = sys.argv[2]
target = getattr(mdir.Target, target_name)
precision = sys.argv[3]
cli = sys.argv[4]
work = pathlib.Path(sys.argv[5])
scenario = sys.argv[6]
SEED = 2024
FIELDS = ("positions", "velocities", "forces")
KCAL_A2 = 4.184 / (0.1 * 0.1)  # kJ/mol/nm^2 per kcal/mol/A^2


def control(path, kind="NVT", steps=20, interval=10, pme=False, constraints=False,
            restraint=None, temperature=300.0, method="VELOCITY_VERLET", start=None,
            trajectory=True):
    """A control file that writes what the Python model of `model` gives."""
    name = path.stem
    thermostat = '[thermostat]\nmethod = "V-RESCALE"\ninterval = 10' if kind != "NVE" else ""
    barostat = '[barostat]\nmethod = "C-RESCALE"' if kind == "NPT" else ""
    com = "center_of_mass_interval = 10" if kind != "NVE" else ""
    tables = (f'[[restraints]]\nselection = "{restraint[0]}"\nforce_constant = {restraint[1]}\n'
              if restraint else "")
    constraint_table = ("[constraints]\nhydrogen_bonds = true\nrigid_water = true\n"
                        if constraints else "")
    checkpoint_input = f'checkpoint = "{start}"' if start else ""
    frames = (f'trajectory = "{name}.dcd"\ntrajectory_interval = {interval}'
              if trajectory else "")
    path.write_text(f"""[input]
topology = "{root}/dipeptide.prmtop"
coordinates = "{root}/dipeptide.inpcrd"
{checkpoint_input}
[energy]
cutoff = 8.0
pairlist_distance = 9.0
electrostatics = "{'PME' if pme else 'CUTOFF'}"
{constraint_table}
[dynamics]
integrator = "{method}"
time_step = 0.0005
steps = {steps}
seed = {SEED}
{com}
[output]
energy_interval = {interval}
energy = "{name}.dat"
{frames}
checkpoint = "{name}.h5"
checkpoint_interval = {interval}
[ensemble]
ensemble = "{kind}"
temperature = {temperature}
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


def model(kind="NVT", pme=False, constraints=False, restraint=None, temperature=300.0,
          method="VelocityVerlet", tunables=None, drawn=True, run_precision=None):
    """The Python model of `control`, setting the same options."""
    loaded = mdir.load_amber(root + "/dipeptide.prmtop", root + "/dipeptide.inpcrd")
    system, state = loaded.make_system(), loaded.make_state()
    system.cutoff, system.pairlist_distance = 0.8, 0.9
    system.truncation = mdir.Truncation.None_
    system.electrostatics = mdir.Electrostatics.PME if pme else mdir.Electrostatics.Cutoff
    system.periodic = True
    if constraints:
        system.rigid_hydrogen_bonds = system.rigid_water = True
    if restraint:
        system.restraints = [mdir.Restraint(restraint[0], restraint[1] * KCAL_A2)]
    if tunables:
        system.tunables = tunables
    if drawn:
        state = state.draw_velocities(system, temperature, SEED)
    integrator, ensemble, execution = mdir.Integrator(), mdir.Ensemble(), mdir.Execution()
    integrator.method = getattr(mdir.IntegratorMethod, method)
    integrator.timestep = 0.0005
    ensemble.seed = SEED
    ensemble.kind = getattr(mdir.EnsembleKind, kind)
    ensemble.temperature = temperature
    if kind != "NVE":
        ensemble.com_period = 10
        ensemble.coupling_period = 10
    execution.target = target
    execution.precision = getattr(mdir.Precision, run_precision or precision)
    execution.deterministic = True
    return mdir.compile(system, state, integrator, ensemble, execution, mdir.Schedule())


def run_cli(path, *options):
    result = subprocess.run([cli, "run", *options, str(path)], cwd=path.parent,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert result.returncode == 0, result.stdout
    return result.stdout


def continued(program, checkpoint, **options):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        simulation = mdir.Simulation(program, checkpoint=str(checkpoint), **options)
    return simulation, [str(w.message) for w in caught]


def expect(error, call, *texts):
    try:
        call()
    except error as exc:
        for text in texts:
            assert text in str(exc), str(exc)
        return str(exc)
    raise AssertionError(f"expected {error.__name__}")


def compare(label, state, reference):
    """The deterministic mode orders both front ends alike (#121, PR #173),
    so every comparison here is to the bit, in both precisions and on both
    targets."""
    line = [label]
    for field in FIELDS:
        mine, theirs = getattr(state, field), getattr(reference, field)
        differing = np.count_nonzero(mine.view(np.uint64) != theirs.view(np.uint64))
        assert differing == 0, (label, field, differing, np.abs(mine - theirs).max())
        line.append(f"{field} equal")
    print("; ".join(line))


def uninterrupted(directory, **settings):
    """`mdir run` of 20 steps with a checkpoint every 10: `<name>.h5` holds
    step 20 and `<name>.h5.prev` step 10."""
    directory.mkdir(parents=True, exist_ok=True)
    path = control(directory / "run.toml", **settings)
    run_cli(path)
    return path


def cli_to_python(kind, pme=False, constraints=False):
    # `mdir run` stops at step 10 (the checkpoint before the last); Python
    # continues it to step 20, appending to its energy file and trajectory.
    whole = work / f"cli-{kind}"
    uninterrupted(whole, kind=kind, pme=pme, constraints=constraints)
    reference = mdir.read_checkpoint(str(whole / "run.h5"))
    assert reference.step == 20 and reference.front_end == "mdir run"
    part = work / f"cli-{kind}-part"
    part.mkdir()
    shutil.copy(whole / "run.h5.prev", part / "run.h5")
    for name in ("run.dat", "run.dcd"):
        shutil.copy(whole / name, part / name)
    program = model(kind, pme, constraints)
    simulation, notes = continued(program, part / "run.h5")
    assert notes == [], notes
    assert simulation.step == 10 and abs(simulation.time - 10 * 0.0005) < 1e-15
    simulation.reporters = [mdir.EnergyReporter(str(part / "run.dat"), 10),
                            mdir.TrajectoryReporter(str(part / "run.dcd"), 10)]
    simulation.run(10)
    state = simulation.state()
    assert state.step == 20
    compare(f"mdir run -> Python {kind} {target_name} {precision}", state, reference)
    simulation.save_checkpoint(str(part / "run.h5"))
    simulation.close_reporters()
    mine = mdir.read_checkpoint(str(part / "run.h5"))
    assert mine.part == 2 and mine.first_step == 0 and mine.front_end == "python"
    assert mine.trajectory == "run.dcd" and mine.frames == 2
    assert mine.fingerprint == reference.fingerprint, \
        set(mine.fingerprint) ^ set(reference.fingerprint)
    assert mine.bath == reference.bath
    assert (part / "run.h5.prev").exists()
    rows = (part / "run.dat").read_text().splitlines()
    assert [int(r.split()[0]) for r in rows[2:]] == [0, 10, 20], rows
    assert (part / "run.dat").read_bytes() == (whole / "run.dat").read_bytes()
    assert (part / "run.dcd").read_bytes() == (whole / "run.dcd").read_bytes()
    print("the energy file and the trajectory equal those of mdir run byte for byte")


def python_to_cli(kind, pme=False, constraints=False):
    # Python takes 10 steps and writes the checkpoint; `mdir run --continue`
    # takes the 10 that remain, appending to the files of the reporters.
    # `mdir run` from the Python checkpoint equals a Python continuation of
    # the same file and the run of `mdir run` that did not stop, to the bit.
    whole = work / f"whole-{kind}"
    uninterrupted(whole, kind=kind, pme=pme, constraints=constraints)
    reference = mdir.read_checkpoint(str(whole / "run.h5"))
    part = work / f"python-{kind}"
    part.mkdir()
    path = control(part / "run.toml", kind=kind, pme=pme, constraints=constraints)
    simulation = mdir.Simulation(model(kind, pme, constraints))
    simulation.reporters = [mdir.EnergyReporter(str(part / "run.dat"), 10),
                            mdir.TrajectoryReporter(str(part / "run.dcd"), 10),
                            mdir.CheckpointReporter(str(part / "run.h5"), 10)]
    simulation.run(10)
    simulation.close_reporters()
    written = mdir.read_checkpoint(str(part / "run.h5"))
    assert written.step == 10 and written.part == 1 and written.frames == 1
    assert written.fingerprint == reference.fingerprint, \
        set(written.fingerprint) ^ set(reference.fingerprint)
    python, notes = continued(model(kind, pme, constraints), part / "run.h5")
    assert notes == [], notes
    python.run(10)
    log = run_cli(path, "--continue")
    assert "other physics" not in log, log
    state = mdir.read_checkpoint(str(part / "run.h5"))
    assert state.step == 20 and state.part == 2
    compare(f"Python -> mdir run {kind} {target_name} {precision}, against Python", state,
            python.state())
    compare(f"Python -> mdir run {kind} {target_name} {precision}, against mdir run", state,
            reference)
    rows = (part / "run.dat").read_text().splitlines()
    assert [int(r.split()[0]) for r in rows[2:]] == [0, 10, 20], rows
    assert (part / "run.dat").read_bytes() == (whole / "run.dat").read_bytes()
    assert (part / "run.dcd").read_bytes() == (whole / "run.dcd").read_bytes()
    print("the energy file and the trajectory equal those of mdir run byte for byte")
    code, text = mdir_checkpoint(part / "run.h5")
    assert code == 0 and "front end" not in text, text


def mdir_checkpoint(path, *options):
    result = subprocess.run([cli, "checkpoint", *options, str(path)], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    return result.returncode, result.stdout.strip()


def python_to_python():
    # In one front end, with tunables updated before the checkpoint: the
    # continuation equals the run that did not stop, to the bit.
    def tunable_model():
        loaded = mdir.load_amber(root + "/dipeptide.prmtop", root + "/dipeptide.inpcrd")
        count = loaded.make_system().particle_count
        tunables = [mdir.Tunable("q", "charge", map=np.arange(count))]
        return model("NVE", tunables=tunables)
    program = tunable_model()
    whole = mdir.Simulation(program)
    whole.run(10)
    q = whole.tunables["q"] * 0.99
    whole.tunables["q"] = q
    whole.run(5)
    # A simulation that writes a checkpoint goes on from the state written,
    # as one that continues the file does.
    whole.save_checkpoint(str(work / "whole.h5"))
    whole.run(5)
    reference = whole.state()

    first = mdir.Simulation(program)
    first.reporters = [mdir.EnergyReporter(str(work / "py.dat"), 5)]
    first.run(10)
    first.tunables["q"] = q
    first.run(5)
    first.save_checkpoint(str(work / "py.h5"))
    del first
    saved = mdir.read_checkpoint(str(work / "py.h5"))
    assert saved.tunables_version == 1 and saved.tunables_history == [(0, 0), (10, 1)]
    assert np.array_equal(saved.tunables["q"], q)
    assert saved.model_sha256 and saved.plan_sha256
    code, text = mdir_checkpoint(work / "py.h5")
    assert code == 0 and "front end:       python" in text and "tunables:" in text, text
    second, notes = continued(tunable_model(), work / "py.h5")
    assert notes == [], notes
    assert second.tunables.version == 1 and second.tunables.history == [(0, 0), (10, 1)]
    assert np.array_equal(second.tunables["q"], q)
    second.reporters = [mdir.EnergyReporter(str(work / "py.dat"), 5)]
    second.run(5)
    compare(f"Python -> Python NVE with tunables {target_name} {precision}", second.state(),
            reference)
    rows = (work / "py.dat").read_text().splitlines()[2:]
    assert [int(r.split()[0]) for r in rows] == [0, 5, 10, 15, 20], rows
    assert rows[-1].split()[-1] == "1", rows[-1]  # tunables_version
    # The same file again: a second continuation of step 15 cuts the rows
    # after it and goes on.
    third, _ = continued(tunable_model(), work / "py.h5")
    third.reporters = [mdir.EnergyReporter(str(work / "py.dat"), 5)]
    third.run(5)
    assert (work / "py.dat").read_text().splitlines()[2:] == rows
    # `mdir run` has no tunables: it refuses to continue the run, and
    # begins a stage of its own from the checkpoint.
    cli_dir = work / "cli"
    cli_dir.mkdir()
    shutil.copy(work / "py.h5", cli_dir / "run.h5")
    path = control(cli_dir / "run.toml", kind="NVE", steps=20, interval=5, trajectory=False)
    result = subprocess.run([cli, "run", "--continue", str(path)], cwd=cli_dir,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert result.returncode != 0 and "[python] tunables" in result.stdout, result.stdout
    stage_dir = work / "cli-stage"
    stage_dir.mkdir()
    path = control(stage_dir / "run.toml", kind="NVE", steps=5, interval=5, trajectory=False,
                   start=str(work / "py.h5"))
    assert "[python] tunables" in run_cli(path)
    # Other declarations: the same run is refused, a stage takes the
    # model's values.
    other = model("NVE", tunables=[mdir.Tunable("sigma", "sigma")])
    expect(mdir.InputError, lambda: mdir.Simulation(other, checkpoint=str(work / "py.h5")),
           "[python] tunables")
    stage, notes = continued(other, work / "py.h5", stage=True)
    assert any("evaluates the forces" in n for n in notes), notes
    assert stage.tunables.version == 0 and stage.step == 15
    print("Python -> Python with tunables: version, history, values, and energy rows restored")


def stages():
    # The same run against a changed stage, from the checkpoint of `mdir run`
    # at step 10. A stage of other physics evaluates its forces at its first
    # step; `mdir run` does the same from `[input] checkpoint`.
    directory = work / "stages"
    uninterrupted(directory, kind="NVT")
    start = directory / "run.h5.prev"

    def refused(program, *names):
        message = expect(mdir.InputError, lambda: mdir.Simulation(program, checkpoint=str(start)),
                         "other physics or coupling", *names)
        return message

    restraint = ("!:WAT & !@H*", 10.0)
    refused(model(restraint=restraint), "[[restraints]]", "the reference of the restraints")
    refused(model(temperature=310.0), "[ensemble] temperature")
    refused(model("NPT"), "[barostat] method", "[ensemble] ensemble")
    refused(model(pme=True), "[energy] electrostatics")
    # Masses: a copy of the topology with the mass of the first atom changed.
    text = pathlib.Path(root + "/dipeptide.prmtop").read_text().splitlines(keepends=True)
    at = next(i for i, line in enumerate(text) if line.startswith("%FLAG MASS")) + 2
    first = text[at][:16]
    text[at] = f"{float(first) + 1.0:16.8E}" + text[at][16:]
    heavy = directory / "heavy.prmtop"
    heavy.write_text("".join(text))
    loaded = mdir.load_amber(str(heavy), root + "/dipeptide.inpcrd")
    system, state = loaded.make_system(), loaded.make_state()
    system.cutoff, system.pairlist_distance = 0.8, 0.9
    system.truncation = mdir.Truncation.None_
    system.electrostatics = mdir.Electrostatics.Cutoff
    system.periodic = True
    integrator, ensemble, execution = mdir.Integrator(), mdir.Ensemble(), mdir.Execution()
    integrator.method, integrator.timestep = mdir.IntegratorMethod.VelocityVerlet, 0.0005
    ensemble.seed, ensemble.kind, ensemble.temperature = SEED, mdir.EnsembleKind.NVT, 300.0
    ensemble.com_period = ensemble.coupling_period = 10
    execution.target, execution.precision = target, getattr(mdir.Precision, precision)
    execution.deterministic = True
    heavy_program = mdir.compile(system, state, integrator, ensemble, execution, mdir.Schedule())
    refused(heavy_program, "the masses", "the files of the topology")
    # Another integrator is refused in both ways.
    for stage in (False, True):
        expect(mdir.InputError, lambda: mdir.Simulation(model(method="Leapfrog"),
                                                        checkpoint=str(start), stage=stage),
               "integrator VELOCITY_VERLET", "LEAPFROG")
    # Another precision is the same run, with a warning.
    other = "Mixed" if precision == "Double" else "Double"
    simulation, notes = continued(model(run_precision=other), start)
    assert len(notes) == 1 and "[execution] precision" in notes[0], notes
    simulation.run(10)
    # The same physics taken as a stage: the forces of the checkpoint, and
    # the run that did not stop.
    simulation, notes = continued(model(), start, stage=True)
    assert notes == [], notes
    simulation.run(10)
    compare(f"a stage of the same physics {target_name} {precision}", simulation.state(),
            mdir.read_checkpoint(str(directory / "run.h5")))
    assert mdir.read_checkpoint(str(start)).bath != 0.0

    # Stages of other physics against `mdir run` from `[input] checkpoint`:
    # restraints, another temperature, and the barostat.

    def cli_stage(stage_dir, start, **settings):
        stage_dir.mkdir()
        path = control(stage_dir / "run.toml", steps=10, start=str(start), **settings)
        log = run_cli(path)
        assert "other physics or coupling" in log, log
        return mdir.read_checkpoint(str(stage_dir / "run.h5"))

    for name, settings in (("restrained", dict(restraint=restraint)),
                           ("warmer", dict(temperature=310.0)),
                           ("npt", dict(kind="NPT"))):
        stage_dir = directory / name
        reference = cli_stage(stage_dir, start, **settings)
        simulation, notes = continued(model(**settings), start, stage=True)
        assert len(notes) == 1 and "evaluates the forces" in notes[0], notes
        simulation.run(10)
        compare(f"a stage {name} against mdir run {target_name} {precision}",
                simulation.state(), reference)
        saved = stage_dir / "python.h5"
        simulation.save_checkpoint(str(saved))
        mine = mdir.read_checkpoint(str(saved))
        assert mine.first_step == 10 and mine.part == 1
        assert mine.fingerprint == reference.fingerprint, \
            set(mine.fingerprint) ^ set(reference.fingerprint)
    # A stage of leapfrog from a leapfrog checkpoint of other physics takes
    # the velocities as they are, half a step behind (no second half kick
    # back), as `mdir run` does.
    leap = work / "leapfrog"
    uninterrupted(leap, method="LEAPFROG")
    reference = cli_stage(leap / "warmer", leap / "run.h5.prev", temperature=310.0,
                          method="LEAPFROG")
    simulation, notes = continued(model(method="Leapfrog", temperature=310.0),
                                  leap / "run.h5.prev", stage=True)
    assert len(notes) == 1, notes
    simulation.run(10)
    compare(f"a leapfrog stage against mdir run {target_name} {precision}", simulation.state(),
            reference)
    print("stages: refusals name the entries; stages equal mdir run from [input] checkpoint")


def corrupted():
    directory = work / "corrupted"
    directory.mkdir()
    simulation = mdir.Simulation(model("NVE"))
    simulation.run(10)
    good = directory / "good.h5"
    simulation.save_checkpoint(str(good))
    simulation.save_checkpoint(str(good))
    assert (directory / "good.h5.prev").exists()
    data = good.read_bytes()
    program = model("NVE")

    def refused(name, content, *texts):
        path = directory / name
        path.write_bytes(content)
        expect(mdir.InputError, lambda: mdir.Simulation(program, checkpoint=str(path)),
               *texts, ".prev' holds the checkpoint before it")
        expect(mdir.InputError, lambda: mdir.read_checkpoint(str(path)), *texts)
        code, text = mdir_checkpoint(path)
        assert code == 2, (code, text)
        return text

    # A value of the positions changed: the hash of the state.
    positions = simulation.state().positions.astype("<f8").tobytes()[:64]
    at = data.find(positions)
    assert at > 0
    changed = bytearray(data)
    changed[at + 3] ^= 0x10
    refused("state.h5", bytes(changed), "does not match the hash")
    # The model hash changed: the hash of the additional entries.
    model_hash = mdir.read_checkpoint(str(good)).model_sha256.encode()
    at = data.find(model_hash)
    assert at > 0
    changed = bytearray(data)
    changed[at] = ord("0") if changed[at] != ord("0") else ord("1")
    refused("model.h5", bytes(changed), "model, the plan, or the tunables")
    refused("truncated.h5", data[: len(data) // 2], "cannot read")
    refused("text.h5", b"not a checkpoint\n", "cannot read")
    expect(mdir.InputError, lambda: mdir.Simulation(program, checkpoint=str(directory / "none.h5")),
           "cannot read")
    expect(mdir.InputError, lambda: mdir.Simulation(program, stage=True), "give checkpoint=")
    expect(mdir.InputError, lambda: mdir.CheckpointReporter("x.h5", 0), "positive")
    simulation.reporters = [mdir.CheckpointReporter(str(directory / "a.h5"), 5),
                            mdir.CheckpointReporter(str(directory / "b.h5"), 5)]
    expect(mdir.InputError, lambda: simulation.run(5), "one CheckpointReporter")
    print("corrupted files: refused by the bindings and by mdir checkpoint")


SCENARIOS = {
    "cli-to-python": lambda: (cli_to_python("NVE"), cli_to_python("NVT"),
                              cli_to_python("NPT", pme=True, constraints=True)),
    "python-to-cli": lambda: (python_to_cli("NVE"), python_to_cli("NVT"),
                              python_to_cli("NPT", pme=True, constraints=True)),
    "python-to-python": python_to_python,
    "stages": stages,
    "corrupted": corrupted,
}
SCENARIOS[scenario]()
print(f"checkpoints {scenario} {target_name} {precision} passed")
