"""Programs that differ only in the cell they start from share the compile
cache (D[cell-runtime-constants], docs/compile-cache.md).

  compile_cache_cell.py ROOT TARGET PRECISION WORK

The dipeptide in water with PME, SHAKE, and SETTLE, at constant pressure
(stochastic cell rescaling) with the heavy atoms restrained, deterministic.
The second start is the first with its cell and positions scaled by
1.001, as an equilibrated cell differs from the cell of the file: the PME
grid and the neighbor capacity stay. Checks that

- the program texts of the two starts are the same, and the values of the
  start (the barostat's constants and the edges of the restraints' cell)
  are arguments of the entry, not constants;
- the second start hits the entry that the first stored;
- 20 steps from the second start with the cached object equal those of a
  compile without the cache, bit for bit.
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
integrator, ensemble, execution = mdir.Integrator(), mdir.Ensemble(), mdir.Execution()
integrator.method = mdir.IntegratorMethod.VelocityVerlet
integrator.timestep = 0.001
ensemble.kind = mdir.EnsembleKind.NPT
ensemble.temperature, ensemble.pressure = 300.0, 1.0
ensemble.tau_t, ensemble.tau_p = 0.5, 1.0
ensemble.coupling_period = 5
ensemble.seed = 271828
execution.target = getattr(mdir.Target, target)
execution.precision = getattr(mdir.Precision, precision)
execution.deterministic = True
schedule = mdir.Schedule()


def start(scale):
    state = loaded.make_state()
    cell = state.cell
    cell.diagonal = np.asarray(cell.diagonal) * scale
    state.cell = cell
    state.positions = np.asarray(state.positions) * scale
    return state.draw_velocities(system, 300.0, 271828)


def compile_(scale):
    return mdir.compile(system, start(scale), integrator, ensemble, execution, schedule)


first, second = compile_(1.0), compile_(1.001)
assert first.ir == second.ir, "the starting cell changes the program text"
assert first.pipeline == second.pipeline, (first.pipeline, second.pipeline)
for name in ("baro_constant", "baro_energy_constant", "rest_edge0", "rest_edge1", "rest_edge2"):
    assert f"%{name}: f64" in first.ir and f"%{name} = arith.constant" not in first.ir, name
print(f"{target} {precision}: the starting cell leaves the text and the pipeline as they are")

stats = mdir.Simulation(first).compile_stats
print(f"first: compiled {stats['host_compiled']} hits {stats['cache_hits']} stored {stats['cache_stored']}")
cached = mdir.Simulation(second)
stats = cached.compile_stats
print(f"second: compiled {stats['host_compiled']} hits {stats['cache_hits']} stored {stats['cache_stored']}")
os.environ["MDIR_COMPILE_CACHE"] = "off"
plain = mdir.Simulation(compile_(1.001))
del os.environ["MDIR_COMPILE_CACHE"]
for simulation in (cached, plain):
    simulation.run(20, energy=True)
a, b = cached.state(), plain.state()
same = all(np.array_equal(np.asarray(getattr(a, k)), np.asarray(getattr(b, k)))
           for k in ("positions", "velocities")) and \
    np.array_equal(a.cell.diagonal, b.cell.diagonal) and a.energies == b.energies
print(f"second, 20 steps: {'same as without the cache' if same else 'DIFFERENT'}")
