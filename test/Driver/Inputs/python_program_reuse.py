"""The code that a Program keeps for its simulations (D236):
the dipeptide in water with PME, SHAKE, and SETTLE, in the deterministic
mode.

  python_program_reuse.py ROOT TARGET PRECISION [CACHE_DIRECTORY]

The first simulation of a program lowers and generates code; a second takes
the code that the first left and gives the same positions and velocities
bit for bit after the same steps; one with cache=False generates everything
anew, leaves nothing, and gives the same bits; a program of its own, of the
same inputs, shares nothing with the first (with a compile cache directory
its first simulation hits the disk and its second takes what that left);
the simulations of a program compiled with cache=False never take kept
code; and two threads that make simulations of one new program at once both
get one that runs.
"""
import os
import sys
import threading

import numpy as np

root, target, precision = sys.argv[1:4]
if len(sys.argv) > 4:
    os.environ["MDIR_COMPILE_CACHE_DIR"] = sys.argv[4]
else:
    os.environ.pop("MDIR_COMPILE_CACHE_DIR", None)

import mdir

SEED, STEPS = 271828, 20
loaded = mdir.load_amber(root + "/dipeptide.prmtop", root + "/dipeptide.inpcrd")
system, state = loaded.make_system(), loaded.make_state()
system.cutoff, system.pairlist_distance = 0.8, 0.9
system.truncation = mdir.Truncation.None_
system.electrostatics = mdir.Electrostatics.PME
system.rigid_hydrogen_bonds = system.rigid_water = True
start = state.draw_velocities(system, 300.0, SEED)
integrator, ensemble, execution = mdir.Integrator(), mdir.Ensemble(), mdir.Execution()
integrator.timestep, ensemble.temperature, ensemble.seed = 0.0005, 300.0, SEED
execution.target = getattr(mdir.Target, target)
execution.precision = getattr(mdir.Precision, precision)
execution.deterministic = True


def compile_program(**keywords):
    return mdir.compile(system, start, integrator, ensemble, execution,
                        mdir.Schedule(), **keywords)


def trajectory(simulation):
    """The state after STEPS steps in two runs of one part each."""
    simulation.part_seconds = 1e9
    simulation.run(STEPS // 2)
    simulation.run(STEPS - STEPS // 2, energy=True)
    result = simulation.state()
    return (np.asarray(result.positions).tobytes(),
            np.asarray(result.velocities).tobytes())


reference = None


def report(name, simulation):
    global reference
    stats = simulation.compile_stats
    result = trajectory(simulation)
    if reference is None:
        reference = result
    # A simulation that took kept code lowered, generated, and looked up
    # nothing.
    if stats["program_reused"]:
        quiet = all(stats[key] == 0 for key in stats if key not in (
            "programs", "engine_seconds", "program_reused",
            "reuse_saved_seconds", "cache_bypassed"))
        assert quiet and stats["reuse_saved_seconds"] > 0, stats
    else:
        assert stats["pipeline_seconds"] > 0, stats
        assert stats["reuse_saved_seconds"] == 0, stats
    print(f"{name}: reused {stats['program_reused']} pipeline "
          f"{'run' if stats['pipeline_seconds'] > 0 else 'none'} host compiled "
          f"{stats['host_compiled']} disk hits {stats['cache_hits']} bypassed "
          f"{stats['cache_bypassed']} state "
          f"{'same' if result == reference else 'DIFFERENT'}")


program = compile_program()
first = mdir.Simulation(program)
report("first", first)
# While the first lives, and after it has gone.
report("second", mdir.Simulation(program))
del first
report("third", mdir.Simulation(program))
report("bypass", mdir.Simulation(program, cache=False))
report("after-bypass", mdir.Simulation(program))

other = compile_program()
report("other-program", mdir.Simulation(other))
report("other-program-second", mdir.Simulation(other))

never = compile_program(cache=False)
report("bypass-program", mdir.Simulation(never))
report("bypass-program-second", mdir.Simulation(never))
# The choice of the simulation over that of its program: the first with the
# cache leaves code, the second takes it.
report("bypass-program-with-cache", mdir.Simulation(never, cache=True))
report("bypass-program-with-cache-second", mdir.Simulation(never, cache=True))

# Two threads at once on a program without code.
fresh = compile_program()
made, errors = [None, None], []


def make(k):
    try:
        made[k] = mdir.Simulation(fresh)
    except Exception as error:  # reported below
        errors.append(error)


threads = [threading.Thread(target=make, args=(k,)) for k in range(2)]
for thread in threads:
    thread.start()
for thread in threads:
    thread.join()
assert not errors, errors
reused = sorted(s.compile_stats["program_reused"] for s in made)
same = all(trajectory(s) == reference for s in made)
print(f"threads: reused {reused[0]} and {reused[1]} state "
      f"{'same' if same else 'DIFFERENT'}")
report("after-threads", mdir.Simulation(fresh))
