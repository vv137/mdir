"""The defaults of the Python model are those of the control file
(D[python-defaults], #281, docs/python-model.md): a system given to both
front ends with nothing beyond its inputs is one model.

Usage: python_defaults.py ROOT TARGET PRECISION MDIR WORK SCENARIO

  values   the defaults as the fields give them, and what an assignment
           and `None` do; needs no run.
  run      the dipeptide in water of ROOT with no setting but the files,
           the ensemble (NVE, NVT, NPT), the periodic boundary, and, for
           the comparison to the bit, the deterministic mode: the
           energies of step 0 against the row of `mdir run`, and 20 steps
           against its checkpoint, the positions, the velocities, the
           forces, and the cell to the bit; the fingerprint of the
           checkpoint against that of `mdir run`.
  old      the model of before, stated: `Truncation.Switch` from 1.0 nm,
           a pairlist distance of 1.35 nm, no removal of the motion of the
           center of mass, and velocities of zero. It compiles and runs,
           with the warning `dispersion_switched`, and differs from the
           default model.
"""
import pathlib
import subprocess
import sys
import warnings

import numpy as np
import mdir

root, target_name, precision, cli = sys.argv[1:5]
work = pathlib.Path(sys.argv[5])
scenario = sys.argv[6]
label = f"{target_name} {precision}"
KCAL = 4.184
STEPS = 20
FIELDS = ("positions", "velocities", "forces")


def loaded():
    data = mdir.load_amber(root + "/dipeptide.prmtop", root + "/dipeptide.inpcrd")
    return data.make_system(), data.make_state()


def execution():
    result = mdir.Execution()
    result.target = getattr(mdir.Target, target_name)
    result.precision = getattr(mdir.Precision, precision)
    result.deterministic = True
    return result


def values():
    system, state = loaded()
    assert system.truncation == mdir.Truncation.None_
    assert system.cutoff == 1.2
    assert system.switch_distance == 1.2
    assert abs(system.pairlist_distance - 1.35) < 1e-15
    assert system.dispersion == mdir.DispersionCorrection.EnergyPressure
    assert not system.dispersion_given
    # The two follow the cutoff until they are set; None restores that.
    system.cutoff = 0.9
    assert system.switch_distance == 0.9 and abs(system.pairlist_distance - 1.05) < 1e-15
    system.pairlist_distance, system.switch_distance = 1.0, 0.8
    system.cutoff = 0.95
    assert system.pairlist_distance == 1.0 and system.switch_distance == 0.8
    system.pairlist_distance = system.switch_distance = None
    assert system.switch_distance == 0.95 and abs(system.pairlist_distance - 1.1) < 1e-15
    ensemble = mdir.Ensemble()
    assert ensemble.com_period is None
    ensemble.com_period = 0
    assert ensemble.com_period == 0
    ensemble.com_period = None
    assert ensemble.com_period is None
    # The default model compiles without a warning: a plain cutoff with the
    # correction for the dispersion.
    system, state = loaded()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        program = mdir.compile(system, state, mdir.Integrator(), mdir.Ensemble(),
                               mdir.Execution(), mdir.Schedule())
    assert program.plan["dispersion"]["correction"] == mdir.DispersionCorrection.EnergyPressure
    print("defaults values passed")


def run():
    for kind in ("NVE", "NVT", "NPT"):
        here = work / kind
        here.mkdir(parents=True)
        baths = '[thermostat]\nmethod = "V-RESCALE"\n' if kind != "NVE" else ""
        baths += '[barostat]\nmethod = "C-RESCALE"\n' if kind == "NPT" else ""
        (here / "run.toml").write_text(f"""[input]
topology = "{root}/dipeptide.prmtop"
coordinates = "{root}/dipeptide.inpcrd"
[energy]
[dynamics]
steps = {STEPS}
[output]
energy = "run.dat"
energy_interval = {STEPS}
checkpoint = "run.h5"
checkpoint_interval = {STEPS}
[ensemble]
ensemble = "{kind}"
{baths}[boundary]
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

        system, state = loaded()
        system.periodic = True
        ensemble = mdir.Ensemble()
        ensemble.kind = getattr(mdir.EnsembleKind, kind)
        schedule = mdir.Schedule()
        schedule.steps = schedule.energy_period = STEPS
        simulation = mdir.Simulation(mdir.compile(system, state, mdir.Integrator(), ensemble,
                                                  execution(), schedule))
        simulation.run(0, energy=True)
        energies = simulation.state().energies
        # The row has 6 decimals of kcal/mol.
        worst = 0.0
        for name in ("potential", "kinetic", "total"):
            if name in row and name in energies:
                worst = max(worst, abs(energies[name] / KCAL - row[name]))
        assert "potential" in row and worst < 1e-6, (worst, row, energies)
        simulation.run(STEPS, energy=True)
        simulation.save_checkpoint(str(here / "python.h5"))
        end = simulation.state()
        for field in FIELDS:
            mine, theirs = getattr(end, field), getattr(reference, field)
            differing = np.count_nonzero(mine.view(np.uint64) != theirs.view(np.uint64))
            assert differing == 0, (kind, field, differing, np.abs(mine - theirs).max())
        assert np.array_equal(end.cell.diagonal, reference.cell.diagonal), kind
        written = mdir.read_checkpoint(str(here / "python.h5"))
        assert written.fingerprint == reference.fingerprint, (
            kind, set(written.fingerprint) ^ set(reference.fingerprint))
        assert np.abs(np.asarray(end.velocities)).max() > 0.1, "the run began at rest"
        print(f"{label} {kind}: the energies of step 0 within {worst:.1e} kcal/mol of the row of "
              f"mdir run, and {STEPS} steps to its bits, with its fingerprint", flush=True)
    print(f"defaults run {label} passed")


def old():
    def energies(change):
        system, state = loaded()
        system.periodic = True
        ensemble = mdir.Ensemble()
        ensemble.kind = mdir.EnsembleKind.NVT
        change(system, state, ensemble)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            program = mdir.compile(system, state, mdir.Integrator(), ensemble, execution(),
                                   mdir.Schedule())
        simulation = mdir.Simulation(program)
        simulation.run(0, energy=True)
        return simulation.state(), [str(w.message) for w in caught], program

    def before(system, state, ensemble):
        system.truncation = mdir.Truncation.Switch
        system.switch_distance = 1.0
        system.pairlist_distance = 1.35
        ensemble.com_period = 0
        state.velocities = np.zeros((system.particle_count, 3))

    new, notes, _ = energies(lambda *_: None)
    assert not notes, notes
    stated, notes, program = energies(before)
    assert len(notes) == 1 and "turns the correction for the dispersion off" in notes[0], notes
    assert program.plan["dispersion"]["correction"] == mdir.DispersionCorrection.None_
    assert np.abs(np.asarray(stated.velocities)).max() == 0.0
    assert np.abs(np.asarray(new.velocities)).max() > 0.1
    difference = abs(new.energies["potential"] - stated.energies["potential"])
    assert difference > 1.0, difference
    print(f"{label}: the model of before, stated, begins at rest with the switch and without the "
          f"correction; its potential differs from the default model's by {difference:.1f} kJ/mol")
    print(f"defaults old {label} passed")


{"values": values, "run": run, "old": old}[scenario]()
