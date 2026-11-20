"""Compare the initial GROUPS evaluation with the CPU matrix on a small LJ lattice."""
import sys
from pathlib import Path

import mdir
import numpy as np

work = Path(sys.argv[1])
precision = sys.argv[2]


def evaluate(target, layout):
    loaded = mdir.load_gromacs(str(work / "plain.top"), str(work / "system.gro"))
    system, state = loaded.make_system(), loaded.make_state()
    system.periodic = True
    system.cutoff = 0.72
    system.pairlist_distance = 0.97
    system.truncation = mdir.Truncation.None_
    system.electrostatics = mdir.Electrostatics.Cutoff
    execution = mdir.Execution()
    execution.target = target
    execution.precision = getattr(mdir.Precision, precision)
    execution.neighbor_structure = layout
    integrator = mdir.Integrator()
    integrator.timestep = 0.001
    ensemble = mdir.Ensemble()
    ensemble.kind = mdir.EnsembleKind.NVE
    simulation = mdir.Simulation(mdir.compile(
        system, state, integrator, ensemble, execution, mdir.Schedule()))
    simulation.run(0, energy=True)
    result = simulation.state()
    return (np.array(result.forces), result.energies["potential"], result.energies["virial"])


reference = evaluate(mdir.Target.CPU, mdir.NeighborStructure.Matrix)
actual = evaluate(mdir.Target.GPU, mdir.NeighborStructure.Groups)
for name, got, expected in zip(("force", "energy", "virial"), actual, reference):
    peak = np.max(np.abs(expected))
    tolerance = (2e-5 if precision == "Mixed" else 1e-10) * peak
    error = np.max(np.abs(got - expected))
    print(f"{precision} {name}: reference peak {peak:.12g}, max difference "
          f"{error:.12g}, tolerance {tolerance:.12g}", flush=True)
    assert peak > 0 and error <= tolerance, (name, error, tolerance)
