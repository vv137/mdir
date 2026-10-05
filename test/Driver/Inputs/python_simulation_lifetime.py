"""Exceptions after interleaved JIT destruction (D[python-segments])."""
import faulthandler
import pathlib
import sys

# Exercise the runtime already selected by an array library, as applications do.
import numpy as np
import mdir

faulthandler.dump_traceback_later(900, exit=True)
work = pathlib.Path(sys.argv[2])
work.mkdir(parents=True, exist_ok=True)
topology = work / "pair.top"
coordinates = work / "pair.gro"
topology.write_text("""[ defaults ]
1 2 no 1 1
[ atomtypes ]
X 1.0 0.0 A 0.3 0.1
[ moleculetype ]
PAIR 0
[ atoms ]
1 X 1 PAIR X1 1 0.0 1.0
2 X 1 PAIR X2 2 0.0 1.0
[ system ]
Two particles
[ molecules ]
PAIR 1
""")
coordinates.write_text("""Two particles
2
    1PAIR    X1    1   0.500   0.500   0.500
    1PAIR    X2    2   0.900   0.500   0.500
   3.00000   3.00000   3.00000
""")
loaded = mdir.load_gromacs(str(topology), str(coordinates))
system, state = loaded.make_system(), loaded.make_state()
system.cutoff, system.pairlist_distance, system.switch_distance = 0.8, 0.9, 0.7
integrator, ensemble = mdir.Integrator(), mdir.Ensemble()
programs = []
for precision in (mdir.Precision.Double, mdir.Precision.Mixed):
    execution = mdir.Execution()
    execution.target, execution.precision = getattr(mdir.Target, sys.argv[1]), precision
    execution.reorder, execution.deterministic = False, True
    programs.append(mdir.compile(system, state, integrator, ensemble, execution,
                                 mdir.Schedule()))

# Keep engines alive while others are destroyed and their mappings are reused.
# GPU constructors/destructors used to put one module's host code in disjoint
# sections. Their bounding unwind ranges overlapped other modules, leaving
# freed libgcc registration objects reachable at the first subsequent throw.
# Defer that throw until after all cycles: early throws can hide the defect.
live, references = [], {}
for cycle in range(24):
    precision = cycle % 2
    simulation = mdir.Simulation(programs[precision])
    assert simulation.run(1) == 1
    assert simulation.run(1) == 1
    current = simulation.state()
    if precision not in references:
        references[precision] = current
    for quantity in ("positions", "velocities", "forces"):
        assert np.array_equal(getattr(current, quantity),
                              getattr(references[precision], quantity)), quantity
    live.append(simulation)
    if len(live) > 3:
        live.pop(0)

try:
    simulation.run(-1)
except mdir.InputError as error:
    assert "nonnegative" in str(error)
else:
    raise AssertionError("negative step count did not raise InputError")
for simulation in live:
    assert simulation.run(1) == 1
    assert simulation.step == 3
live.clear()
del simulation
# Also throw after the last engine has been destroyed.
try:
    mdir.load_amber(str(work / "missing.prmtop"), str(work / "missing.inpcrd"))
except mdir.InputError:
    pass
else:
    raise AssertionError("missing input did not raise InputError")
print("simulation lifetime: double and mixed passed")
faulthandler.cancel_dump_traceback_later()
