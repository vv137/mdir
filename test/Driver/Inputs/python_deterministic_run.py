"""A Python simulation against `mdir run` in the deterministic mode (#105):
the dipeptide in water with SHAKE and SETTLE, NVE from the velocities that
both draw, 20 steps in one part with a step of energy at its end, against
`mdir run` with a row at the end. Prints whether the positions, the
velocities, and the forces are the same bit for bit.

  python_deterministic_run.py ROOT TARGET MDIR WORK PRECISION ELECTROSTATICS
"""
import pathlib
import subprocess
import sys

import numpy as np
import mdir

root, target_name, cli, work, precision, electrostatics = sys.argv[1:7]
work = pathlib.Path(work) / f"{target_name}-{precision}-{electrostatics}"
work.mkdir(parents=True, exist_ok=True)
SEED = 271828
(work / "run.toml").write_text(f"""[input]
topology = "{root}/dipeptide.prmtop"
coordinates = "{root}/dipeptide.inpcrd"
[energy]
cutoff = 8.0
pairlist_distance = 9.0
electrostatics = "{electrostatics}"
[constraints]
hydrogen_bonds = true
rigid_water = true
[dynamics]
time_step = 0.0005
steps = 20
seed = {SEED}
[output]
energy_interval = 20
checkpoint = "run.h5"
checkpoint_interval = 20
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
subprocess.run([cli, "run", "run.toml"], cwd=work, check=True,
               stdout=subprocess.DEVNULL)


def read(field):
    out = subprocess.run([cli, "checkpoint", f"--print={field}", "run.h5"],
                         cwd=work, check=True, stdout=subprocess.PIPE,
                         text=True).stdout
    return np.array([[float(x) for x in line.split()[2:]]
                     for line in out.splitlines()])


loaded = mdir.load_amber(root + "/dipeptide.prmtop", root + "/dipeptide.inpcrd")
system, state = loaded.make_system(), loaded.make_state()
system.cutoff, system.pairlist_distance = 0.8, 0.9
system.truncation = mdir.Truncation.None_
system.electrostatics = {"PME": mdir.Electrostatics.PME,
                         "CUTOFF": mdir.Electrostatics.Cutoff}[electrostatics]
system.rigid_hydrogen_bonds = system.rigid_water = True
start = state.draw_velocities(system, 300.0, SEED)
integrator, ensemble, execution = mdir.Integrator(), mdir.Ensemble(), mdir.Execution()
integrator.timestep, ensemble.temperature, ensemble.seed = 0.0005, 300.0, SEED
execution.target = getattr(mdir.Target, target_name)
execution.precision = getattr(mdir.Precision, precision)
execution.deterministic = True
simulation = mdir.Simulation(mdir.compile(system, start, integrator, ensemble,
                                          execution, mdir.Schedule()))
simulation.run(20, energy=True)
result = simulation.state()
line = [f"{target_name} {precision} {electrostatics}:"]
for name in ("positions", "velocities", "forces"):
    theirs = read(name)
    mine = np.asarray(getattr(result, name)).reshape(theirs.shape)
    # The printout of `mdir checkpoint` has 17 significant digits, which
    # identify a double.
    largest = np.abs(mine - theirs).max()
    line.append(f"{name} {'same' if largest == 0.0 else 'differ by %.3e' % largest}")
print(" ".join(line))
