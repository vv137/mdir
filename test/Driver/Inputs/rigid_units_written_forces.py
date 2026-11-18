"""rigid_units_written_forces.py WORK: frames that write their forces under
`rigid_units = ON` (the maintainer's ruling on #915; #916). Alanine
dipeptide in OPC (a taken kind with a site) refuses an H5MDReporter with
forces; without forces it runs; in TIP3P (no site) it runs with them."""
import sys
from pathlib import Path

import mdir

work = Path(sys.argv[1])
inputs = Path(__file__).resolve().parent


def simulation(files):
    loaded = mdir.load_amber(str(inputs / f"{files}.prmtop"), str(inputs / f"{files}.inpcrd"))
    system, state = loaded.make_system(), loaded.make_state()
    system.cutoff = 0.8
    system.pairlist_distance = 1.0
    system.electrostatics = mdir.Electrostatics.PME
    system.periodic = True
    system.rigid_water = True
    system.rigid_hydrogen_bonds = True
    i, e, x = mdir.Integrator(), mdir.Ensemble(), mdir.Execution()
    i.timestep = 0.002
    e.kind = mdir.EnsembleKind.NVE
    x.target = mdir.Target.GPU
    x.precision = mdir.Precision.Mixed
    x.rigid_units = mdir.RigidUnits.On
    return mdir.Simulation(mdir.compile(system, state, i, e, x, mdir.Schedule()))


for files, forces in (("opc/opc", True), ("opc/opc", False), ("dipeptide/dipeptide", True)):
    sim = simulation(files)
    sim.reporters.append(mdir.H5MDReporter(str(work / f"{files.split('/')[1]}-{forces}.h5md"), 5,
                                           forces=forces))
    try:
        sim.run(10)
        print(f"{files} forces={forces}: ran")
    except Exception as error:  # noqa: BLE001
        print(f"{files} forces={forces}: {error}")
    sim.close_reporters()
