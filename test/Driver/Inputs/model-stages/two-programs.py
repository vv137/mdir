"""Compiles and runs the models in stages named on the command line one
after the other in one process, each program released before the next is
compiled, and prints the potential energy at the start of each, in kJ/mol
(D302: the device's copies of a body's weights belong to the
program that made them, not to the host addresses of the weights, which a
later program may reuse).

usage: two-programs.py DIRECTORY MODEL.mlir...
"""
import gc
import os
import sys

import mdir


def energy(directory, path):
    loaded = mdir.load_gromacs(os.path.join(directory, "system.top"),
                               os.path.join(directory, "system.gro"))
    system, state = loaded.make_system(), loaded.make_state()
    system.cutoff = 0.2
    system.pairlist_distance = 0.25
    system.models = [mdir.ModelTerm("painn", path, backend="STAGES")]
    integrator, ensemble, execution = mdir.Integrator(), mdir.Ensemble(), mdir.Execution()
    ensemble.kind = mdir.EnsembleKind.NVE
    execution.target = mdir.Target.GPU
    execution.precision = mdir.Precision.Double
    program = mdir.compile(system, state, integrator, ensemble, execution, mdir.Schedule())
    sim = mdir.Simulation(program)
    sim.run(0, energy=True)
    value = sim.state().energies["potential"]
    del sim, program
    gc.collect()
    return value


directory = sys.argv[1]
for path in sys.argv[2:]:
    print(f"{os.path.basename(path)} {energy(directory, path):.6f}")
