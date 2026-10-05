"""Exceptions after interleaved JIT destruction (D196)."""
import faulthandler
import pathlib
import sys
import random
import ctypes
import concurrent.futures

# Each order runs in a fresh process with ASLR unchanged.
order = sys.argv[3] if len(sys.argv) > 3 else "numpy-first"
if order == "runtime-first":
    ctypes.CDLL(sys.argv[4], mode=ctypes.RTLD_GLOBAL)
if order == "mdir-first":
    import mdir
    import numpy as np
else:
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

# Independent analytic Lennard-Jones force at the returned coordinates.
# GROMACS fixture: epsilon=0.1 kJ/mol and sigma=0.3 nm, below switching.
for precision, program in enumerate(programs):
    current = mdir.Simulation(program)
    current.run(2)
    result = current.state()
    delta = result.positions[0] - result.positions[1]
    r = np.linalg.norm(delta)
    ratio6 = (0.3 / r) ** 6
    oracle = 24 * 0.1 * (2 * ratio6**2 - ratio6) / r**2 * delta
    expected = np.stack((oracle, -oracle))
    difference = np.max(np.abs(result.forces - expected))
    tolerance = (256 * np.finfo(np.float64).eps if precision == 0 else
                 64 * np.finfo(np.float32).eps) * np.max(np.abs(expected))
    assert difference <= tolerance, (precision, difference, tolerance)
    print(f"force oracle target={sys.argv[1]} precision={precision} "
          f"reference={oracle[0]:.12e} difference={difference:.12e} "
          f"tolerance={tolerance:.12e} kJ/mol/nm")
    del current

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
# Seeded operation sequences retain several live engines and report the
# entire sequence on failure. Each seed is run once; failures are never retried.
for seed in (89, 196, 20261006):
    randomizer = random.Random(seed)
    trace, engines = [], []
    try:
        for operation in range(32):
            action = randomizer.choice(("create", "run", "destroy", "error"))
            if not engines:
                action = "create"
            if len(engines) >= 5 and action == "create":
                action = "destroy"
            index = randomizer.randrange(len(engines)) if engines else 0
            precision = randomizer.randrange(2)
            trace.append((operation, action, index, precision))
            if action == "create":
                current = mdir.Simulation(programs[precision])
                current.run(1)
                current.run(1)  # Materialize the continued engine, too.
                engines.append((current, precision))
                for quantity in ("positions", "velocities", "forces"):
                    assert np.array_equal(getattr(current.state(), quantity),
                                          getattr(references[precision], quantity))
                del current
            elif action == "destroy":
                del engines[index]
            elif action == "run":
                engines[index][0].run(randomizer.randrange(1, 4))
            else:
                try:
                    engines[index][0].run(-1)
                except mdir.InputError:
                    pass
                else:
                    raise AssertionError("missing delayed exception")
        engines.clear()
        try:
            mdir.load_amber(str(work / "missing.prmtop"), str(work / "missing.inpcrd"))
        except mdir.InputError:
            pass
        else:
            raise AssertionError("missing post-destruction exception")
    except BaseException:
        print(f"seed={seed} order={order} operations={trace}", file=sys.stderr)
        raise
    print(f"lifecycle seed={seed} order={order}: 32 operations, 0 failures")

# Creation and destruction must synchronize with runs of another simulation.
# The extension releases the GIL for execution; workers retain their own
# simulations, so no object is destroyed while one of its methods is active.
def worker(index):
    current = mdir.Simulation(programs[index % 2])
    current.run(2)
    current.run(1)
    return current.step
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    assert list(pool.map(worker, range(6))) == [3] * 6
print("simulation lifetime: double and mixed passed")
faulthandler.cancel_dump_traceback_later()
