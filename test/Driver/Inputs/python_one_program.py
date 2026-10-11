"""A simulation compiles one program (D211): runs
of dynamics in several calls of the entry, and a minimization in several,
load the kernels of one module each, which the runtime traces with
MDRT_TRACE.

Usage: python_one_program.py ROOT"""
import sys

import mdir

root = sys.argv[1]


def compile_program(minimize):
    loaded = mdir.load_amber(root + "/dipeptide.prmtop", root + "/dipeptide.inpcrd")
    system, state = loaded.make_system(), loaded.make_state()
    # The defaults of before D[python-defaults], with which this was written.
    system.truncation = mdir.Truncation.Switch
    system.cutoff, system.pairlist_distance, system.switch_distance = 0.8, 0.9, 0.7
    system.electrostatics = mdir.Electrostatics.PME
    integrator, ensemble = mdir.Integrator(), mdir.Ensemble()
    ensemble.com_period = 0  # the default of before D[python-defaults]
    integrator.timestep = 0.001
    integrator.minimize = minimize
    if not minimize:
        ensemble.kind = mdir.EnsembleKind.NVT
        ensemble.temperature, ensemble.coupling_period = 300, 10
    execution = mdir.Execution()
    execution.target, execution.precision = mdir.Target.GPU, mdir.Precision.Mixed
    schedule = mdir.Schedule()
    schedule.steps = 30
    return mdir.compile(system, state, integrator, ensemble, execution, schedule)


kind = sys.argv[2]
simulation = mdir.Simulation(compile_program(kind == "minimize"))
print(f"created {kind}", file=sys.stderr, flush=True)
for steps in (1, 7, 13):
    if kind == "minimize":
        assert simulation.minimize(steps) == steps
    else:
        assert simulation.run(steps) == steps
assert simulation.step == 21
print(f"ran {kind}", file=sys.stderr, flush=True)
