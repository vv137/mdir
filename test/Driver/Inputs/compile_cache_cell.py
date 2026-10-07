"""Programs that differ only in the cell they start from share the compile
cache (D227, docs/compile-cache.md).

  compile_cache_cell.py ROOT TARGET PRECISION WORK

The dipeptide in water with PME, SHAKE, and SETTLE, at constant pressure
(stochastic cell rescaling) with the heavy atoms restrained, deterministic.
A start is the file's with its cell and positions scaled, as an
equilibrated cell differs from the cell of the file. The PME grid is the
same for every start here. Checks that

- the start scaled by 1.01 has the program text and the pipeline of the
  file's start: the values of the start (the barostat's constants and the
  edges of the restraints' cell) are arguments of the entry, not constants,
  and the neighbor capacity, whose estimate is rounded up to four
  significant bits, is 448 for both (rounded to a multiple of 8 it was 432
  and 424);
- that start hits the entry that the first stored, and its 20 steps equal
  those of a compile without the cache, bit for bit;
- the start scaled by 1.02, whose estimate is 416, misses, and hits with
  `Execution.neighbor_capacity` set to the capacity of the first
  (`Program.plan["neighbor_capacity"]`); its 20 steps with that capacity
  equal those with its own estimate, bit for bit;
- a negative capacity is refused.
"""
import os
import pathlib
import sys

import numpy as np
import mdir

root, target, precision, work = sys.argv[1], sys.argv[2], sys.argv[3], pathlib.Path(sys.argv[4])
os.environ.pop("MDIR_COMPILE_CACHE", None)
os.environ["MDIR_COMPILE_CACHE_DIR"] = str(work / "cache")

loaded = mdir.load_amber(root + "/dipeptide.prmtop", root + "/dipeptide.inpcrd")
system, initial = loaded.make_system(), loaded.make_state()
system.cutoff, system.pairlist_distance = 0.8, 0.9
system.truncation = mdir.Truncation.None_
system.electrostatics = mdir.Electrostatics.PME
system.rigid_hydrogen_bonds = system.rigid_water = True
system.restraint_reference = initial.positions
system.restraints = [mdir.Restraint("!:WAT & !@H*", 400.0)]
integrator, ensemble = mdir.Integrator(), mdir.Ensemble()
integrator.method = mdir.IntegratorMethod.VelocityVerlet
integrator.timestep = 0.001
ensemble.kind = mdir.EnsembleKind.NPT
ensemble.temperature, ensemble.pressure = 300.0, 1.0
ensemble.tau_t, ensemble.tau_p = 0.5, 1.0
ensemble.coupling_period = 5
ensemble.seed = 271828
schedule = mdir.Schedule()


def start(scale):
    state = loaded.make_state()
    cell = state.cell
    cell.diagonal = np.asarray(cell.diagonal) * scale
    state.cell = cell
    state.positions = np.asarray(state.positions) * scale
    return state.draw_velocities(system, 300.0, 271828)


def compile_(scale, capacity=0):
    execution = mdir.Execution()
    execution.target = getattr(mdir.Target, target)
    execution.precision = getattr(mdir.Precision, precision)
    execution.deterministic = True
    execution.neighbor_capacity = capacity
    return mdir.compile(system, start(scale), integrator, ensemble, execution, schedule)


def uncached(program):
    os.environ["MDIR_COMPILE_CACHE"] = "off"
    simulation = mdir.Simulation(program)
    del os.environ["MDIR_COMPILE_CACHE"]
    return simulation


def same(a, b):
    for simulation in (a, b):
        simulation.run(20, energy=True)
    x, y = a.state(), b.state()
    return all(np.array_equal(np.asarray(getattr(x, k)), np.asarray(getattr(y, k)))
               for k in ("positions", "velocities")) and \
        np.array_equal(x.cell.diagonal, y.cell.diagonal) and x.energies == y.energies


def stats(name, simulation):
    s = simulation.compile_stats
    print(f"{name}: compiled {s['host_compiled']} hits {s['cache_hits']} stored {s['cache_stored']}")


first, second = compile_(1.0), compile_(1.01)
capacity = first.plan["neighbor_capacity"]
assert first.ir == second.ir, "the starting cell changes the program text"
assert first.pipeline == second.pipeline, (first.pipeline, second.pipeline)
assert f"width={capacity}" in first.pipeline, first.pipeline
for name in ("baro_constant", "baro_energy_constant", "rest_edge0", "rest_edge1", "rest_edge2"):
    assert f"%{name}: f64" in first.ir and f"%{name} = arith.constant" not in first.ir, name
print(f"{target} {precision}: the starting cell leaves the text and the pipeline as they are, "
      f"capacity {capacity}")

stats("first", mdir.Simulation(first))
cached = mdir.Simulation(second)
stats("second", cached)
print(f"second, 20 steps: {'same as without the cache' if same(cached, uncached(compile_(1.01))) else 'DIFFERENT'}")

# Another estimate is another program; the capacity of the first makes it
# the same one, and does not change the run.
third = compile_(1.02)
print(f"third: capacity {third.plan['neighbor_capacity']}")
assert third.ir == first.ir and third.pipeline != first.pipeline
stats("third", mdir.Simulation(third))
pinned = compile_(1.02, capacity)
assert pinned.plan["neighbor_capacity"] == capacity and pinned.pipeline == first.pipeline
cached = mdir.Simulation(pinned)
stats("third with the capacity of the first", cached)
print(f"third, 20 steps: {'same with either capacity' if same(cached, uncached(compile_(1.02))) else 'DIFFERENT'}")

try:
    compile_(1.0, -1)
except mdir.InputError as error:
    print(f"refused: {error}")
