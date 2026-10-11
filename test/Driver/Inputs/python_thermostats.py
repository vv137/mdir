"""The thermostats of the control file and Brownian dynamics in a Python
simulation (D249, #278, docs/python-model.md) against
`mdir run` of the same control, to the bit in the deterministic mode.

Usage: python_thermostats.py INPUTS TARGET PRECISION MDIR WORK SCENARIO [CASE...]

  refusals  the fields, their defaults and types, the refusals with the
            words of the control file, and `Program.plan`; needs no device.
  cases     the dipeptide in water of `Inputs/dipeptide` (1,168 particles,
            PME on 32^3, SHAKE and SETTLE), for each case below (all, or
            those named).

The cases, each 40 steps with a row of energies and a frame every 10:

  langevin-nvt     Langevin dynamics, 1/ps, at constant volume: no period of
                   coupling, as the control file without `interval`
  langevin-npt     the same with stochastic cell rescaling every 10 steps
  langevin-com     at constant volume, 2.5/ps, with the motion of the center
                   of mass removed every 10 steps
  nose-hoover-nvt  a Nose-Hoover chain of 3 thermostats, 0.5 ps, every 10
  nose-hoover-npt  a chain of 5 with stochastic cell rescaling
  langevin-leapfrog  Langevin dynamics with the leapfrog integrator, the
                   period written as 0 in both front ends
  brownian         Brownian dynamics, 50/ps, at 0.1 fs

For each:

- a Python simulation in parts of 10 and 30 steps, with reporters, against
  `mdir run`: the positions, the velocities, the forces, and the cell to
  the bit, the energy file and the DCD trajectory byte for byte, the
  fingerprint, and the state of the chain to the bit;
- `mdir run` -> Python and Python -> `mdir run --continue` at step 20, to
  the same bits and bytes;
- without a period of coupling, parts of 7, 13, and 20 steps as well, also
  with leapfrog, whose step of energy leaves the state of its plain step
  (#285).

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
STEPS, PERIOD = 40, 10
FIELDS = ("positions", "velocities", "forces")
label = f"{target_name} {precision}"
T, I, B = mdir.ThermostatMethod, mdir.IntegratorMethod, mdir.EnsembleKind

# For each case: the kind of the ensemble, the keys of [dynamics], the
# [thermostat] table, the fields of the ensemble and of the integrator,
# whether there is a period of coupling, the length of the chain, and the
# thermostat of the plan.
CASES = {
    "langevin-nvt": dict(
        kind="NVT", thermostat='method = "LANGEVIN"\nfriction = 1.0\n',
        ensemble={"thermostat": T.Langevin, "friction": 1.0}, period=False,
        plan={"method": T.Langevin, "friction": 1.0}),
    "langevin-npt": dict(
        kind="NPT", thermostat='method = "LANGEVIN"\nfriction = 1.0\n',
        ensemble={"thermostat": T.Langevin, "friction": 1.0},
        plan={"method": T.Langevin, "friction": 1.0}),
    "langevin-com": dict(
        kind="NVT", dynamics="center_of_mass_interval = 10\n",
        thermostat='method = "LANGEVIN"\nfriction = 2.5\n',
        ensemble={"thermostat": T.Langevin, "friction": 2.5, "com_period": 10},
        plan={"method": T.Langevin, "friction": 2.5}),
    "nose-hoover-nvt": dict(
        kind="NVT", thermostat='method = "NOSE-HOOVER"\ntime_constant = 0.5\n',
        ensemble={"thermostat": T.NoseHoover, "tau_t": 0.5}, chain=3,
        plan={"method": T.NoseHoover, "chain_length": 3}),
    "nose-hoover-npt": dict(
        kind="NPT",
        thermostat='method = "NOSE-HOOVER"\ntime_constant = 0.5\nchain_length = 5\n'
                   'interval = 10\n',
        ensemble={"thermostat": T.NoseHoover, "tau_t": 0.5, "chain_length": 5,
                  "coupling_period": 10}, chain=5,
        plan={"method": T.NoseHoover, "chain_length": 5}),
    "langevin-leapfrog": dict(
        kind="NVT", method="LEAPFROG",
        thermostat='method = "LANGEVIN"\nfriction = 1.0\ninterval = 0\n',
        integrator={"method": I.Leapfrog},
        ensemble={"thermostat": T.Langevin, "friction": 1.0, "coupling_period": 0},
        period=False,
        plan={"method": T.Langevin, "friction": 1.0}),
    "brownian": dict(
        kind="NVT", method="BROWNIAN", timestep=0.0001, dynamics="friction = 50.0\n",
        integrator={"method": I.Brownian, "friction": 50.0}, period=False,
        plan=None),
}


def control(path, name):
    """The control file of the case `name`, with a checkpoint at half of
    its steps."""
    case = CASES[name]
    method = case.get("method", "VELOCITY_VERLET")
    thermostat = "[thermostat]\n" + case["thermostat"] if "thermostat" in case else ""
    barostat = ('[barostat]\nmethod = "C-RESCALE"\n' if case["kind"] == "NPT" else "")
    pressure = "pressure = 2000.0\n" if case["kind"] == "NPT" else ""
    path.write_text(f"""[input]
topology = "{inputs}/dipeptide/dipeptide.prmtop"
coordinates = "{inputs}/dipeptide/dipeptide.inpcrd"
[energy]
cutoff = 8.0
pairlist_distance = 9.0
electrostatics = "PME"
[pme]
grid = [32, 32, 32]
[constraints]
hydrogen_bonds = true
rigid_water = true
[dynamics]
integrator = "{method}"
time_step = {case.get("timestep", 0.002)}
steps = {STEPS}
seed = {SEED}
{case.get("dynamics", "")}[output]
energy_interval = {PERIOD}
energy = "run.dat"
trajectory = "run.dcd"
trajectory_interval = {PERIOD}
checkpoint = "run.h5"
checkpoint_interval = {STEPS // 2}
[ensemble]
ensemble = "{case["kind"]}"
temperature = 300.0
{pressure}{thermostat}{barostat}[boundary]
type = "PERIODIC"
[execution]
target = "{target_name}"
precision = "{precision.upper()}"
deterministic = true
""")
    return path


def parts_of_model(kind="NVT", ensemble=None, integrator=None, timestep=0.002):
    loaded = mdir.load_amber(inputs + "/dipeptide/dipeptide.prmtop",
                             inputs + "/dipeptide/dipeptide.inpcrd")
    system, state = loaded.make_system(), loaded.make_state()
    system.cutoff, system.pairlist_distance = 0.8, 0.9
    system.electrostatics = mdir.Electrostatics.PME
    system.pme_grid = [32] * 3
    system.periodic = True
    system.rigid_water = system.rigid_hydrogen_bonds = True
    # The state has no velocities: `mdir.compile` draws those of `mdir run`
    # at the temperature and with the seed of the ensemble (D248), for the
    # degrees of freedom of its thermostat. `State.draw_velocities` draws
    # for a system that keeps its momentum, which Langevin dynamics without
    # a removal and Brownian dynamics do not.
    assert state.velocities is None or len(state.velocities) == 0
    step, bath, execution = mdir.Integrator(), mdir.Ensemble(), mdir.Execution()
    step.method = I.VelocityVerlet
    step.timestep = timestep
    for name, value in (integrator or {}).items():
        setattr(step, name, value)
    bath.seed = SEED
    bath.kind = getattr(B, kind)
    bath.temperature = 300.0
    if kind == "NPT":
        bath.pressure = 2000.0 * 1.01325  # bar
    for name, value in (ensemble or {}).items():
        setattr(bath, name, value)
    execution.target = getattr(mdir.Target, target_name)
    execution.precision = getattr(mdir.Precision, precision)
    execution.deterministic = True
    return system, state, step, bath, execution


def compiled(*arguments, **keywords):
    schedule = mdir.Schedule()
    schedule.steps, schedule.energy_period = STEPS, PERIOD
    return mdir.compile(*parts_of_model(*arguments, **keywords), schedule)


def model(name):
    case = CASES[name]
    return compiled(case["kind"], case.get("ensemble"), case.get("integrator"),
                    case.get("timestep", 0.002))


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


def continued(program, checkpoint, **keywords):
    """The simulation from a checkpoint, and the notes it warned of."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        simulation = mdir.Simulation(program, checkpoint=str(checkpoint), **keywords)
    return simulation, [str(w.message) for w in caught]


def bits(values):
    return np.asarray(values, dtype=np.float64).view(np.uint64)


def compare(what, state, reference):
    """To the bit: the state and the cell with its tilts."""
    for field in FIELDS:
        mine, theirs = getattr(state, field), getattr(reference, field)
        differing = np.count_nonzero(bits(mine) != bits(theirs))
        assert differing == 0, (what, field, differing, np.abs(mine - theirs).max())
    for field in ("diagonal", "tilt"):
        mine, theirs = getattr(state.cell, field), getattr(reference.cell, field)
        assert np.array_equal(mine, theirs), (what, field, mine, theirs)


def same_chain(what, mine, theirs, length):
    assert len(theirs.thermostat_state) == 2 * length, (what, theirs.thermostat_state)
    assert np.array_equal(bits(mine.thermostat_state), bits(theirs.thermostat_state)), (
        what, mine.thermostat_state, theirs.thermostat_state)


def with_reporters(simulation, directory, checkpoint=True):
    simulation.reporters = [mdir.EnergyReporter(str(directory / "run.dat"), PERIOD),
                            mdir.TrajectoryReporter(str(directory / "run.dcd"), PERIOD)]
    if checkpoint:
        simulation.reporters.append(
            mdir.CheckpointReporter(str(directory / "run.h5"), STEPS // 2))


def same_files(directory, whole):
    for name in ("run.dat", "run.dcd"):
        assert (directory / name).read_bytes() == (whole / name).read_bytes(), (directory, name)


def one_case(name):
    case = CASES[name]
    tag = f"{label} {name}"
    chain = case.get("chain", 0)
    whole = work / name / "whole"
    whole.mkdir(parents=True)
    run_cli(control(whole / "run.toml", name))
    reference = mdir.read_checkpoint(str(whole / "run.h5"))
    middle = mdir.read_checkpoint(str(whole / "run.h5.prev"))
    assert reference.step == STEPS and middle.step == STEPS // 2
    program = model(name)
    assert program.plan["thermostat"] == case["plan"], program.plan["thermostat"]

    # Python in parts, with the checkpoint of `mdir run` at step 20.
    here = work / name / "python"
    here.mkdir()
    simulation = mdir.Simulation(program)
    with_reporters(simulation, here)
    for part in (10, 30):
        simulation.run(part)
    state = simulation.state()
    simulation.close_reporters()
    compare(tag, state, reference)
    same_files(here, whole)
    written = mdir.read_checkpoint(str(here / "run.h5"))
    assert written.step == STEPS and written.front_end == "python"
    assert written.fingerprint == reference.fingerprint, (
        set(written.fingerprint) ^ set(reference.fingerprint))
    same_chain(tag, written, reference, chain)
    if chain:
        # The chain has moved, and the checkpoint of the middle has it.
        assert np.abs(reference.thermostat_state).min() > 0.0, reference.thermostat_state
        same_chain(tag, mdir.read_checkpoint(str(here / "run.h5.prev")), middle, chain)

    # Without a period of coupling a part ends at any step.
    if not case.get("period", True):
        simulation = mdir.Simulation(program)
        simulation.reporters = [mdir.CheckpointReporter(str(here / "odd.h5"), STEPS // 2)]
        for part in (7, 13, 20):
            simulation.run(part)
        compare(tag + ", parts of 7, 13, and 20", simulation.state(), reference)

    # `mdir run` -> Python at step 20.
    part = work / name / "cli-to-python"
    part.mkdir()
    shutil.copy(whole / "run.h5.prev", part / "run.h5")
    for file in ("run.dat", "run.dcd"):
        shutil.copy(whole / file, part / file)
    simulation, notes = continued(program, part / "run.h5")
    assert not notes, notes
    assert simulation.step == STEPS // 2
    with_reporters(simulation, part, checkpoint=False)
    simulation.run(STEPS // 2)
    compare(tag + ", mdir run -> Python", simulation.state(), reference)
    simulation.close_reporters()
    same_files(part, whole)
    simulation.save_checkpoint(str(part / "end.h5"))
    same_chain(tag + ", mdir run -> Python", mdir.read_checkpoint(str(part / "end.h5")),
               reference, chain)

    # Python -> `mdir run --continue` at step 20.
    part = work / name / "python-to-cli"
    part.mkdir()
    path = control(part / "run.toml", name)
    first = mdir.Simulation(program)
    with_reporters(first, part)
    first.run(STEPS // 2)
    first.close_reporters()
    taken = mdir.read_checkpoint(str(part / "run.h5"))
    assert taken.step == STEPS // 2 and taken.front_end == "python"
    same_chain(tag + ", the middle", taken, middle, chain)
    shutil.copy(part / "run.h5", part / "middle.h5")
    log = run_cli(path, "--continue")
    assert "other physics" not in log, log
    ended = mdir.read_checkpoint(str(part / "run.h5"))
    compare(tag + ", Python -> mdir run", ended, reference)
    same_chain(tag + ", Python -> mdir run", ended, reference, chain)
    same_files(part, whole)
    print(f"{tag}: the state, the cell, the energy file, and the trajectory of mdir run, "
          f"in parts and across a checkpoint in both directions", flush=True)
    if chain:
        print(f"{tag}: the chain at step {STEPS}: "
              + " ".join(f"{value:.3e}" for value in reference.thermostat_state),
              file=sys.stderr, flush=True)

    # Another thermostat is other coupling: neither front end continues
    # the run with it (D172, D223), and each names the key. A stage begins
    # the chain at rest, and says so.
    if name == "nose-hoover-nvt":
        other = work / name / "other"
        other.mkdir()
        shutil.copy(part / "middle.h5", other / "run.h5")
        key = "[thermostat] method"
        expect(mdir.InputError,
               lambda: mdir.Simulation(model("langevin-nvt"), checkpoint=str(other / "run.h5")),
               "other physics or coupling", key)
        refused = subprocess.run(
            [cli, "run", "--continue", str(control(other / "run.toml", "langevin-nvt"))],
            cwd=other, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert refused.returncode != 0 and "other physics or coupling" in refused.stdout \
            and key in refused.stdout, refused.stdout
        longer = compiled("NVT", {"thermostat": T.NoseHoover, "tau_t": 0.5, "chain_length": 4})
        expect(mdir.InputError,
               lambda: mdir.Simulation(longer, checkpoint=str(other / "run.h5")),
               "other physics or coupling", "[thermostat] chain_length")
        staged, notes = continued(longer, other / "run.h5", stage=True)
        assert any("its chain begins at rest" in note for note in notes), notes
        staged.save_checkpoint(str(other / "staged.h5"))
        rest = mdir.read_checkpoint(str(other / "staged.h5")).thermostat_state
        assert len(rest) == 8 and not np.any(rest), rest
        print(f"{tag}: a continuation with another thermostat is refused by both front ends, "
              f"and a stage begins its chain at rest", flush=True)


def cases():
    names = chosen or list(CASES)
    for name in names:
        one_case(name)
    print(f"thermostats {label}: {len(names)} cases passed")


def refusals():
    ensemble, integrator = mdir.Ensemble(), mdir.Integrator()
    # The defaults and the types.
    assert ensemble.thermostat == T.VRescale and ensemble.friction is None
    assert ensemble.chain_length == 3 and ensemble.coupling_period is None
    assert integrator.friction is None
    assert list(T.__members__) == ["VRescale", "Langevin", "NoseHoover"]
    for target, name, wrong in ((ensemble, "thermostat", "LANGEVIN"), (ensemble, "thermostat", 1),
                                (ensemble, "thermostat", None),
                                (ensemble, "thermostat", I.Brownian),
                                (ensemble, "friction", "1.0"), (ensemble, "chain_length", 2.5),
                                (ensemble, "chain_length", None),
                                (integrator, "friction", "1.0")):
        try:
            setattr(target, name, wrong)
        except TypeError:
            continue
        raise AssertionError(f"{name} took {wrong!r}")
    ensemble.friction = 2.0
    assert ensemble.friction == 2.0
    ensemble.friction = None
    assert ensemble.friction is None
    ensemble.coupling_period = 5
    assert ensemble.coupling_period == 5
    ensemble.coupling_period = None
    assert ensemble.coupling_period is None

    # The refusals of the thermostat, with the words of the control file
    # and `model` for its path.
    def refuse(text, kind="NVT", ensemble=None, integrator=None):
        expect(mdir.InputError, lambda: compiled(kind, ensemble, integrator), text)

    refuse("model: expected 'friction' in [thermostat], in 1/ps",
           ensemble={"thermostat": T.Langevin})
    for method in (T.VRescale, T.NoseHoover):
        refuse("model: 'friction' is for \"LANGEVIN\"; this thermostat takes 'time_constant', "
               "in ps", ensemble={"thermostat": method, "friction": 1.0})
    refuse("model: 'time_constant' is for \"V-RESCALE\" and \"NOSE-HOOVER\"; Langevin dynamics "
           "takes 'friction', in 1/ps",
           ensemble={"thermostat": T.Langevin, "friction": 1.0, "tau_t": 0.5})
    for method, more in ((T.VRescale, {}), (T.Langevin, {"friction": 1.0})):
        refuse("model: 'chain_length' is for \"NOSE-HOOVER\"",
               ensemble={"thermostat": method, "chain_length": 5, **more})
    for friction in (0.0, -1.0, float("nan")):
        refuse("model: expected a positive number for 'friction'",
               ensemble={"thermostat": T.Langevin, "friction": friction})
    refuse("model: expected at least 1 for 'chain_length'",
           ensemble={"thermostat": T.NoseHoover, "chain_length": 0})
    # The period: 0 is no period, which Langevin dynamics at constant
    # volume has and the other thermostats and a barostat do not.
    for kind, fields in (("NVT", {}), ("NVT", {"thermostat": T.NoseHoover}),
                         ("NPT", {"thermostat": T.Langevin, "friction": 1.0})):
        refuse("model: 'interval' in [thermostat] is 0", kind=kind,
               ensemble={"coupling_period": 0, **fields})
    refuse("invalid step, thread or coupling count", ensemble={"coupling_period": -1})
    compiled("NVT", {"thermostat": T.Langevin, "friction": 1.0, "coupling_period": 0})

    # Brownian dynamics.
    brownian = {"method": I.Brownian, "friction": 50.0}
    refuse("model: expected 'friction' in [dynamics], in 1/ps, with 'integrator = \"BROWNIAN\"'",
           integrator={"method": I.Brownian})
    for method in (I.VelocityVerlet, I.Leapfrog):
        refuse("model: 'friction' in [dynamics] is for 'integrator = \"BROWNIAN\"'; Langevin "
               "dynamics takes 'friction' in [thermostat]",
               integrator={"method": method, "friction": 50.0})
    refuse("model: expected a positive number for 'friction'",
           integrator={"method": I.Brownian, "friction": 0.0})
    for fields in ({"thermostat": T.Langevin, "friction": 1.0}, {"thermostat": T.NoseHoover},
                   {"friction": 1.0}, {"chain_length": 4}, {"tau_t": 0.5}, {"coupling_period": 10}):
        refuse("model: 'integrator = \"BROWNIAN\"' takes no [thermostat]: Brownian dynamics is "
               "coupled to the bath by its 'friction' in [dynamics]",
               ensemble=fields, integrator=brownian)
    refuse("model: Brownian dynamics has no barostat", kind="NPT", integrator=brownian)
    refuse("model: 'integrator = \"BROWNIAN\"' needs 'ensemble = \"NVT\"' in [ensemble], whose "
           "'temperature' is that of the bath", kind="NVE", integrator=brownian)
    for period in (0, 10):
        refuse("model: Brownian dynamics has no momentum to remove: 'center_of_mass_interval' "
               "is not for 'integrator = \"BROWNIAN\"'",
               ensemble={"com_period": period}, integrator=brownian)

    # Without a thermostat its fields are not read, as `tau_t` is not.
    program = compiled("NVE", {"thermostat": T.Langevin, "chain_length": 7})
    assert program.plan["thermostat"] is None
    # The plan, and the default: stochastic velocity rescaling.
    assert compiled("NVT").plan["thermostat"] == {"method": T.VRescale}
    assert compiled("NVT").plan["integrator"] == {"method": I.VelocityVerlet}
    for name, case in CASES.items():
        plan = model(name).plan
        assert plan["thermostat"] == case["plan"], (name, plan["thermostat"])
        assert plan["integrator"] == (brownian if name == "brownian" else
                                      {"method": case.get("integrator", {}).get(
                                          "method", I.VelocityVerlet)}), (
            name, plan["integrator"])
    # A unit quantity, where openmm.unit is there.
    try:
        import openmm.unit as unit
    except ImportError:
        unit = None
    if unit is not None:
        ensemble.friction = 2.0 / unit.femtosecond
        assert abs(ensemble.friction - 2000.0) < 1e-9, ensemble.friction
        expect(mdir.InputError, lambda: setattr(integrator, "friction", 1.0 * unit.picosecond),
               "Integrator.friction: expected a quantity in 1/ps")
    print("thermostats refusals passed")


if scenario == "refusals":
    refusals()
else:
    cases()
