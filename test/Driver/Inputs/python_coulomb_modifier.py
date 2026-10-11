"""The Coulomb modifier of the Python model (D205): PME
with the real-space term shifted at the cutoff, against `mdir run` with
`coulomb_modifier = "POTENTIAL_SHIFT"`, and the refusal without PME."""
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
COLUMNS = {"total": 4.184, "potential": 4.184, "kinetic": 4.184, "temperature": 1.0}


def expect(error, call, text=""):
    try:
        call()
    except error as exc:
        assert text in str(exc), str(exc)
        return
    raise AssertionError(f"expected {error.__name__}")


def cli_run(name, precision, modifier):
    path = work / f"{name}.toml"
    path.write_text(f"""[input]
topology = "{root}/dipeptide.prmtop"
coordinates = "{root}/dipeptide.inpcrd"
[energy]
cutoff = 8.0
pairlist_distance = 9.0
electrostatics = "PME"
coulomb_modifier = "{modifier}"
[constraints]
hydrogen_bonds = true
rigid_water = true
[dynamics]
time_step = 0.0005
steps = 10
seed = {SEED}
[output]
energy_interval = 10
energy = "{name}.dat"
checkpoint = "{name}.h5"
checkpoint_interval = 10
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
    subprocess.run([cli, "run", str(path)], cwd=work, check=True, stdout=subprocess.DEVNULL)
    text = [r for r in (work / f"{name}.dat").read_text().splitlines() if not r.startswith("#")]
    names = (work / f"{name}.dat").read_text().splitlines()[0].lstrip("#").split()
    rows = {int(r.split()[0]): dict(zip(names, r.split())) for r in text}
    state = {}
    for field in ("positions", "velocities", "forces"):
        out = subprocess.run([cli, "checkpoint", f"--print={field}", f"{name}.h5"], cwd=work,
                             check=True, stdout=subprocess.PIPE, text=True).stdout
        state[field] = np.array([[float(x) for x in line.split()[2:]] for line in out.splitlines()])
    return rows, state


def python_run(precision, modifier):
    loaded = mdir.load_amber(root + "/dipeptide.prmtop", root + "/dipeptide.inpcrd")
    system, state = loaded.make_system(), loaded.make_state()
    system.cutoff, system.pairlist_distance = 0.8, 0.9
    system.truncation = mdir.Truncation.None_
    system.electrostatics = mdir.Electrostatics.PME
    system.coulomb_modifier = modifier
    system.rigid_hydrogen_bonds = system.rigid_water = True
    state = state.draw_velocities(system, 300.0, SEED)
    integrator, ensemble, execution = mdir.Integrator(), mdir.Ensemble(), mdir.Execution()
    integrator.timestep, ensemble.temperature = 0.0005, 300.0
    execution.target, execution.precision = target, getattr(mdir.Precision, precision)
    execution.deterministic = True
    simulation = mdir.Simulation(mdir.compile(system, state, integrator, ensemble, execution,
                                              mdir.Schedule()))
    simulation.run(10, energy=True)
    return simulation.state()


# The default is that of the control file: no shift.
assert mdir.System().coulomb_modifier == mdir.CoulombModifier.None_
shift = mdir.CoulombModifier.PotentialShift
for precision in ("Double", "Mixed"):
    rows, reference = cli_run(f"shift-{precision}".lower(), precision, "POTENTIAL_SHIFT")
    state = python_run(precision, shift)
    mine = {c: "%.6f" % (state.energies[c] / s) for c, s in COLUMNS.items()}
    theirs = {c: rows[10][c] for c in COLUMNS}
    if precision == "Double":
        fields = ("positions", "velocities", "forces")
        differences = [float(np.abs(getattr(state, f) - reference[f]).max()) for f in fields]
        assert mine == theirs, (mine, theirs)
        if target_name == "CPU":
            # The same state to the bit, so the same energies.
            assert differences == [0.0, 0.0, 0.0], differences
            print(f"{target_name} Double: the state at step 10 equals that of mdir run to the "
                  f"bit; rows {mine}")
        else:
            # On a GPU the two front ends differ in the last bits of PME in
            # double precision with or without the shift (#105): the shift
            # adds no difference of its own.
            _, unshifted = cli_run("noshift-double", "Double", "NONE")
            plain = python_run("Double", mdir.CoulombModifier.None_)
            without = [float(np.abs(getattr(plain, f) - unshifted[f]).max()) for f in fields]
            assert differences == without, (differences, without)
            assert differences[0] < 1e-14 and differences[2] < 1e-9, differences
            print(f"{target_name} Double: rows {mine} equal to mdir run's; the state at step 10 "
                  f"within {differences} (positions, velocities, forces), the same as without "
                  f"the shift")
    else:
        # With PME in mixed precision the model and `mdir run` differ in the
        # last digits from the first evaluation, with or without the shift
        # (#105).
        worst = max(abs(float(mine[c]) - float(theirs[c])) / abs(float(theirs[c])) for c in COLUMNS)
        assert worst < 1e-6, (mine, theirs)
        print(f"{target_name} Mixed: rows within {worst:.1e} of mdir run (#105)")
    # The shift changes the energy: without it, the potential differs.
    plain = python_run(precision, mdir.CoulombModifier.None_)
    difference = abs(plain.energies["potential"] - state.energies["potential"])
    assert difference > 0.01, difference
    print(f"{target_name} {precision}: the shift changes the potential by {difference:.3f} kJ/mol")

# As the control file: the modifier is of PME's real-space term.
loaded = mdir.load_amber(root + "/dipeptide.prmtop", root + "/dipeptide.inpcrd")
system, state = loaded.make_system(), loaded.make_state()
# The defaults of before D[python-defaults], with which this was written.
system.truncation = mdir.Truncation.Switch
system.cutoff, system.pairlist_distance, system.switch_distance = 0.8, 0.9, 0.7
system.coulomb_modifier = shift
expect(mdir.InputError, lambda: mdir.compile(system, state, mdir.Integrator(), mdir.Ensemble(),
                                             mdir.Execution(), mdir.Schedule()), "for PME")
print("coulomb modifier passed")
