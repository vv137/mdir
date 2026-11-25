"""Exercise force-only work units with a Q18 component outside signed int32."""
from pathlib import Path
import sys

import mdir
import numpy as np

root = Path(sys.argv[1])
top = mdir.load_amber(str(root / "dipeptide.prmtop"), str(root / "dipeptide.inpcrd"))
model, state = top.make_system(), top.make_state()
model.electrostatics = mdir.Electrostatics.Cutoff
model.cutoff, model.pairlist_distance = 0.8, 1.0
model.analytic_bonds = model.rigid_water = model.rigid_hydrogen_bonds = False
# Stretch the first ACE C-H bond along its axis. Its spring constant is
# 340 kcal/(mol A^2), so 0.06 nm adds over 16,000 kJ/(mol nm).
positions = np.array(state.positions)
delta = positions[0] - positions[1]
positions[0] += 0.06 * delta / np.linalg.norm(delta)
state.positions = positions
integrator = mdir.Integrator()
integrator.timestep = 1e-7
run = mdir.Execution()
run.target = mdir.Target.GPU
run.precision = mdir.Precision.Mixed
program = mdir.compile(model, state, integrator, mdir.Ensemble(), run, mdir.Schedule())
sim = mdir.Simulation(program)
assert sim.run(3, energy=False) == 3
forces = sim.state().forces
assert np.isfinite(forces).all() and np.max(np.abs(forces)) > 8192
print("wide force-only steps passed")
