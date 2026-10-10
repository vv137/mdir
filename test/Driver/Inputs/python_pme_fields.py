"""The influence function and the order of PME and the analytic bonds in
the Python model (D[python-pme-fields], #279, docs/python-model.md) against
`mdir run` of the same control, to the bit in the deterministic mode: the
dipeptide in water with PME, SHAKE, and SETTLE.

Usage: python_pme_fields.py ROOT TARGET PRECISION MDIR WORK SCENARIO

  refusals  the defaults and the types of `System.pme_influence`,
            `System.analytic_bonds`, and `System.pme_order`, and the
            refusal of an order that is not 4, 6, or 8; needs no device.
  run       `influence = "OPTIMAL"`, `analytic_bonds = true`, `order = 6`,
            `order = 8`, and the settings of the Amber suite's script
            together (the optimal influence function with the analytic
            bonds): the energies of step 0 against the row of `mdir run`,
            20 steps against its checkpoint to the bit, and its
            fingerprint; and that each setting changes the run.
"""
import pathlib
import subprocess
import sys

import numpy as np
import mdir

root, target_name, precision, cli = sys.argv[1:5]
work = pathlib.Path(sys.argv[5])
scenario = sys.argv[6]
label = f"{target_name} {precision}"
SEED, STEPS, KCAL = 2718, 20, 4.184
FIELDS = ("positions", "velocities", "forces")
# The keys of [pme] and [constraints], and the fields that state them.
CASES = (
    ("influence OPTIMAL", 'influence = "OPTIMAL"\n', "",
     {"pme_influence": mdir.PMEInfluence.Optimal}),
    ("analytic_bonds", "", "analytic_bonds = true\n", {"analytic_bonds": True}),
    ("order 6", "order = 6\n", "", {"pme_order": 6}),
    ("order 8", "order = 8\n", "", {"pme_order": 8}),
    ("the suite's settings", 'influence = "OPTIMAL"\n', "analytic_bonds = true\n",
     {"pme_influence": mdir.PMEInfluence.Optimal, "analytic_bonds": True}),
)


def model(settings):
    loaded = mdir.load_amber(root + "/dipeptide.prmtop", root + "/dipeptide.inpcrd")
    system, state = loaded.make_system(), loaded.make_state()
    system.cutoff, system.pairlist_distance = 0.8, 0.9
    system.truncation = mdir.Truncation.None_
    system.electrostatics = mdir.Electrostatics.PME
    system.pme_grid = [32, 32, 32]
    system.periodic = True
    system.rigid_hydrogen_bonds = system.rigid_water = True
    for name, value in settings.items():
        setattr(system, name, value)
    state = state.draw_velocities(system, 300.0, SEED)
    integrator, ensemble, execution = mdir.Integrator(), mdir.Ensemble(), mdir.Execution()
    integrator.timestep = 0.002
    ensemble.kind = mdir.EnsembleKind.NVE
    ensemble.temperature, ensemble.seed = 300.0, SEED
    execution.target = getattr(mdir.Target, target_name)
    execution.precision = getattr(mdir.Precision, precision)
    execution.deterministic = True
    schedule = mdir.Schedule()
    schedule.steps = schedule.energy_period = STEPS
    return mdir.compile(system, state, integrator, ensemble, execution, schedule)


def expect(error, call, *texts):
    try:
        call()
    except error as exc:
        for text in texts:
            assert text in str(exc), str(exc)
        return
    raise AssertionError(f"expected {error.__name__}")


def refusals():
    system = mdir.System()
    assert system.pme_influence == mdir.PMEInfluence.SPME
    assert system.analytic_bonds is False and system.pme_order == 4
    for wrong in ("OPTIMAL", 1, None, True):
        try:
            system.pme_influence = wrong
        except TypeError:
            continue
        raise AssertionError(f"pme_influence took {wrong!r}")
    for order in (5, 2, 10, 0):
        expect(mdir.InputError, lambda: model({"pme_order": order}),
               "model: expected 4, 6, or 8 for 'order'")
    print("pme fields refusals passed")


def run():
    # The run without the settings, which each of them must change.
    plain = mdir.Simulation(model({}))
    plain.run(0, energy=True)
    plain_potential = plain.state().energies["potential"]
    plain.run(STEPS)
    plain_end = np.asarray(plain.state().positions)
    for number, (name, pme, constraints, settings) in enumerate(CASES):
        here = work / f"case-{number}"
        here.mkdir(parents=True)
        (here / "run.toml").write_text(f"""[input]
topology = "{root}/dipeptide.prmtop"
coordinates = "{root}/dipeptide.inpcrd"
[energy]
cutoff = 8.0
pairlist_distance = 9.0
electrostatics = "PME"
[pme]
grid = [32, 32, 32]
{pme}[constraints]
hydrogen_bonds = true
rigid_water = true
{constraints}[dynamics]
time_step = 0.002
steps = {STEPS}
seed = {SEED}
[output]
energy = "run.dat"
energy_interval = {STEPS}
checkpoint = "run.h5"
checkpoint_interval = {STEPS}
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
        done = subprocess.run([cli, "run", "run.toml"], cwd=here, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True)
        assert done.returncode == 0, done.stdout
        reference = mdir.read_checkpoint(str(here / "run.h5"))
        lines = (here / "run.dat").read_text().splitlines()
        names = lines[0].split()[1:]
        row = dict(zip(names, (float(x) for x in next(
            line for line in lines[1:] if not line.startswith("#")).split())))
        program = model(settings)
        simulation = mdir.Simulation(program)
        simulation.run(0, energy=True)
        potential = simulation.state().energies["potential"]
        worst = abs(potential / KCAL - row["potential"])
        assert worst < 1e-6, (name, worst)
        simulation.run(STEPS, energy=True)
        simulation.save_checkpoint(str(here / "python.h5"))
        end = simulation.state()
        changed = max(abs(potential - plain_potential),
                      float(np.abs(np.asarray(end.positions) - plain_end).max()))
        assert changed > 0.0, f"{name} leaves the run as it was"
        for field in FIELDS:
            mine, theirs = getattr(end, field), getattr(reference, field)
            differing = np.count_nonzero(mine.view(np.uint64) != theirs.view(np.uint64))
            assert differing == 0, (name, field, differing, np.abs(mine - theirs).max())
        written = mdir.read_checkpoint(str(here / "python.h5"))
        assert written.fingerprint == reference.fingerprint, (
            name, set(written.fingerprint) ^ set(reference.fingerprint))
        print(f"{label} {name}: the potential of step 0 within {worst:.1e} kcal/mol of the row "
              f"of mdir run, and {STEPS} steps to its bits, with its fingerprint", flush=True)
    print(f"pme fields run {label}: {len(CASES)} cases passed")


{"refusals": refusals, "run": run}[scenario]()
