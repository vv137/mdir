"""The program of a model does not depend on what the process built before
(#152).

  python_build_history.py ROOT TARGET

The dipeptide in water at constant pressure, whose barostat of Trotter type
scales the cell every step and stores the state of the scaling (D92), is
compiled, then another model (a different temperature), then the first
again, all in one process. The text of the two programs of the first model
must be equal, and so must their lowered text, from which the keys of the
compile caches of host objects and GPU modules are taken (D212, D214).
"""
import sys

import mdir

root, target = sys.argv[1:3]


def build(temperature):
    loaded = mdir.load_amber(root + "/dipeptide.prmtop", root + "/dipeptide.inpcrd")
    system, state = loaded.make_system(), loaded.make_state()
    system.cutoff, system.pairlist_distance = 0.8, 0.9
    system.truncation = mdir.Truncation.None_
    system.electrostatics = mdir.Electrostatics.PME
    system.rigid_hydrogen_bonds = system.rigid_water = True
    integrator, ensemble, execution = mdir.Integrator(), mdir.Ensemble(), mdir.Execution()
    ensemble.com_period = 0  # the default of before D[python-defaults]
    integrator.timestep = 0.002
    ensemble.kind = mdir.EnsembleKind.NPT
    ensemble.temperature, ensemble.pressure, ensemble.seed = temperature, 1.0, 1
    ensemble.coupling_period = 1
    execution.target = getattr(mdir.Target, target)
    execution.deterministic = True
    return mdir.compile(system, state, integrator, ensemble, execution, mdir.Schedule())


first = build(300.0)
assert "@mdrtSetBarostatState" in first.ir, "the model stores no Trotter state"
other = build(310.0)
assert other.ir != first.ir
again = build(300.0)
assert again.ir == first.ir, "the IR of a model depends on what was built before"
assert again.lowered_ir == first.lowered_ir, "the lowered IR depends on what was built before"
print("program independent of build history passed")
