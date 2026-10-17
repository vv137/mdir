"""`mdir.Auto` for the reaches of a Python model (D[skin-auto]): the fields
take it and give it back, `Program.plan` names where each reach came from,
and the fingerprint of a checkpoint records "AUTO" as the control file
does. A Python simulation measures no candidates: the reaches are the
defaults, or those an earlier run of the same key cached.

Usage: python_skin_auto.py ROOT TARGET
"""
import sys

import mdir

root, target = sys.argv[1:3]
loaded = mdir.load_amber(root + "/dipeptide.prmtop", root + "/dipeptide.inpcrd")
system, state = loaded.make_system(), loaded.make_state()
system.cutoff = 0.8
system.electrostatics = mdir.Electrostatics.PME
system.periodic = True
system.pme_grid = [32, 32, 32]
system.rigid_hydrogen_bonds = system.rigid_water = True

assert repr(mdir.Auto) == "mdir.Auto"
assert mdir.Auto == mdir.Auto and mdir.Auto != 1.0
system.pairlist_distance = mdir.Auto
system.pruned_distance = mdir.Auto
assert system.pairlist_distance == mdir.Auto, system.pairlist_distance
assert system.pruned_distance == mdir.Auto, system.pruned_distance

state = state.draw_velocities(system, 300.0, 2718)
integrator, ensemble, execution = mdir.Integrator(), mdir.Ensemble(), mdir.Execution()
integrator.timestep = 0.001
execution.target = getattr(mdir.Target, target)
execution.precision = mdir.Precision.Mixed if target == "GPU" else mdir.Precision.Double
if target == "GPU":
    execution.neighbor_structure = mdir.NeighborStructure.Groups
program = mdir.compile(system, state, integrator, ensemble, execution, mdir.Schedule())
plan = program.plan
print("structure", plan["neighbor_structure"])
print("pairlist", round(plan["pairlist_distance"], 6), plan["pairlist_distance_source"])
print("pruned", round(plan["pruned_distance"], 6), plan["pruned_distance_source"])
print("reason", plan["skin_reason"])

# A number is fixed, as before; None is the default of a key not given.
system.pairlist_distance = 1.1
assert abs(system.pairlist_distance - 1.1) < 1e-12
system.pruned_distance = 0.0
program = mdir.compile(system, state, integrator, ensemble, execution, mdir.Schedule())
print("given", program.plan["pairlist_distance_source"], program.plan["pruned_distance_source"])
system.pairlist_distance = None
assert abs(system.pairlist_distance - 0.95) < 1e-12, system.pairlist_distance
print("done")
