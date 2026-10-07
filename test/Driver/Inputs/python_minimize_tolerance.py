"""A force tolerance for the minimizer (D[minimize-tolerance]): `[minimize]
force_tolerance` of `mdir run` and `Simulation.minimize(tolerance=...)`
against a NumPy reference of the criterion, on the dipeptide in water.

The reference is `mdir run` without the key, its rows every ENERGY steps:
the run with the key must stop at the first of those rows whose largest
force is below the tolerance, with that row, and equal a run without the
key of that many steps (log, trajectory, checkpoint). Python, which checks
every `Schedule.energy_period` steps between its parts, must stop at the
same step with the same row.

Usage: python_minimize_tolerance.py ROOT TARGET MDIR WORK PRECISION."""
import pathlib
import re
import subprocess
import sys

import numpy as np
import mdir

root, target_name, cli, work, precision = sys.argv[1:6]
work = pathlib.Path(work)
target = getattr(mdir.Target, target_name)
STEPS = 400
ENERGY = 10
# kcal/mol/A. The largest force of the rows every ENERGY steps falls below
# it first at step 100 (9.6 kcal/mol/A there; 60.9 at step 90 and at least
# 25 at every row before), in either precision, on the CPU and a GPU.
TOLERANCE = 15.0
KCAL_A = 41.84  # kJ/mol/nm per kcal/mol/A
KCAL_A2 = 4.184 / (0.1 * 0.1)


def expect(error, call, text=""):
    try:
        call()
    except error as exc:
        assert text in str(exc), str(exc)
        return
    raise AssertionError(f"expected {error.__name__}")


def control(name, steps, tolerance=None, electrostatics="PME", constraints=True):
    key = f"force_tolerance = {tolerance}\n" if tolerance is not None else ""
    path = work / f"{name}.toml"
    path.write_text(f"""[input]
topology = "{root}/dipeptide.prmtop"
coordinates = "{root}/dipeptide.inpcrd"
[energy]
cutoff = 8.0
pairlist_distance = 9.0
electrostatics = "{electrostatics}"
[constraints]
hydrogen_bonds = {str(constraints).lower()}
rigid_water = {str(constraints).lower()}
[output]
energy_interval = {ENERGY}
energy = "{name}.dat"
checkpoint = "{name}.h5"
trajectory = "{name}.dcd"
trajectory_interval = {ENERGY}
[minimize]
method = "STEEPEST_DESCENT"
steps = {steps}
{key}[[restraints]]
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


def cli_run(name, steps, tolerance=None):
    path = control(name, steps, tolerance)
    log = subprocess.run([cli, "run", str(path)], cwd=work, check=True,
                         stdout=subprocess.PIPE, text=True).stdout
    lines = (work / f"{name}.dat").read_text().splitlines()
    names = lines[0].lstrip("#").split()
    rows = {int(row.split()[0]): dict(zip(names, row.split())) for row in lines[2:]}
    checkpoint = subprocess.run([cli, "checkpoint", f"{name}.h5"], cwd=work, check=True,
                                stdout=subprocess.PIPE, text=True).stdout
    step = int(re.search(r"step:\s+(\d+)", checkpoint).group(1))
    positions = subprocess.run([cli, "checkpoint", "--print=positions", f"{name}.h5"],
                               cwd=work, check=True, stdout=subprocess.PIPE, text=True).stdout
    return log, rows, step, positions


def first_below(rows, tolerance):
    """The criterion in NumPy: the first row whose largest force is below
    the tolerance, or none."""
    steps = np.array(sorted(rows))
    largest = np.array([float(rows[s]["max_force"]) for s in steps])
    below = np.nonzero(largest < tolerance)[0]
    return int(steps[below[0]]) if below.size else None


label = f"{target_name} {precision}"

# mdir run: the reference, the criterion, and a run that stops.
_, reference, _, _ = cli_run("reference", STEPS)
stop = first_below(reference, TOLERANCE)
assert stop == 100, stop
log, rows, step, positions = cli_run("tolerance", STEPS, TOLERANCE)
assert max(rows) == stop and step == stop, (max(rows), step)
assert rows == {s: r for s, r in reference.items() if s <= stop}
match = re.search(r"MDIR: converged at step (\d+): the largest force, ([0-9.]+) kcal/mol/Å, "
                  r"is below 15", log)
assert match and int(match.group(1)) == stop, log[-400:]
assert "at most 400 steps of steepest descent, until the largest force is below 15 kcal/mol/Å" in log
# A run of that many steps without the key writes the same files.
_, _, short_step, short_positions = cli_run("short", stop)
assert short_step == stop and short_positions == positions
for suffix in (".dat", ".dcd"):
    assert (work / f"tolerance{suffix}").read_bytes() == (work / f"short{suffix}").read_bytes(), suffix
row = rows[stop]
print(f"{label}: mdir run stops at step {stop} (NumPy reference {stop}), "
      f"largest force {row['max_force']} < {TOLERANCE} kcal/mol/A, "
      f"energy {row['potential']} kcal/mol (reference {reference[stop]['potential']})")

# Not converged in the steps given, and converged at the start.
log, rows, step, _ = cli_run("unconverged", 20, 1.0)
assert step == 20 and max(rows) == 20
assert "MDIR: not converged in 20 steps: the largest force, " in log, log[-400:]
log, rows, step, _ = cli_run("start", STEPS, 30.0)
assert step == 0 and list(rows) == [0] and "MDIR: converged at step 0:" in log
print(f"{label}: not converged after 20 steps at 1 kcal/mol/A; converged at step 0 at 30")

# Python: the same steps, checked every energy period between the parts.
loaded = mdir.load_amber(root + "/dipeptide.prmtop", root + "/dipeptide.inpcrd")
system, state = loaded.make_system(), loaded.make_state()
system.cutoff, system.pairlist_distance = 0.8, 0.9
system.truncation = mdir.Truncation.None_
system.electrostatics = mdir.Electrostatics.PME
system.rigid_hydrogen_bonds = system.rigid_water = True
system.restraints = [mdir.Restraint("!:WAT & !@H*", 10.0 * KCAL_A2)]
execution = mdir.Execution()
execution.target, execution.precision = target, getattr(mdir.Precision, precision)
execution.deterministic = True
integrator, schedule = mdir.Integrator(), mdir.Schedule()
integrator.minimize, schedule.steps, schedule.energy_period = True, STEPS, ENERGY
program = mdir.compile(system, state, integrator, mdir.Ensemble(), execution, schedule)
# A fresh simulation converged at its start takes no step and has the row
# of step 0.
fresh = mdir.Simulation(program)
assert fresh.minimize(tolerance=30.0 * KCAL_A) == 0 and fresh.step == 0
assert fresh.state().minimization["converged"] is True
assert "%.6f" % (fresh.state().minimization["max_force"] / KCAL_A) == reference[0]["max_force"] \
    or precision != "Double"
simulation = mdir.Simulation(program)
expect(mdir.InputError, lambda: simulation.minimize(tolerance=-1.0), "positive, finite tolerance")
taken = simulation.minimize(tolerance=TOLERANCE * KCAL_A)
assert taken == stop and simulation.step == stop, taken
mine = simulation.state().minimization
assert mine["converged"] is True
scales = {"potential": ("energy", 4.184), "rms_force": ("rms_force", KCAL_A),
          "max_force": ("max_force", KCAL_A), "step_size": ("step_size", 0.1)}
worst, exact = 0.0, True
for column, (key, scale) in scales.items():
    value, written = mine[key] / scale, float(reference[stop][column])
    if precision == "Double":
        assert "%.6f" % value == reference[stop][column], (column, value, written)
    else:
        # The rounding of f32 forces (D202): 1e-4 of the value.
        assert abs(value - written) <= 1e-6 + 1e-4 * abs(written), (column, value, written)
    worst = max(worst, abs(value - written) / abs(written))
    exact = exact and "%.6f" % value == reference[stop][column]
assert mine["max_force_particle"] + 1 == int(reference[stop]["max_atom"])
agreement = "every printed digit" if exact else f"within {worst:.1e} relative"
print(f"{label}: Python stops at step {taken}, converged, row of mdir run's to {agreement}")
# Converged already: no step; a tolerance it does not meet: steps taken.
assert simulation.minimize(tolerance=TOLERANCE * KCAL_A) == 0 and simulation.step == stop
assert simulation.minimize(20, tolerance=0.01) == 20
assert simulation.state().minimization["converged"] is False
assert simulation.minimize(10) == 10
assert simulation.state().minimization["converged"] is None
try:
    import openmm.unit as unit
    assert simulation.minimize(10, tolerance=1.0 * unit.kilocalorie_per_mole / unit.angstrom) == 10
    assert simulation.state().minimization["converged"] is False
except ImportError:
    pass
print(f"minimization tolerance {precision.lower()} passed")
