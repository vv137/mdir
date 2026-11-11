"""The one wait of a Python simulation for the frames on their own thread
(D400, #113), at the end of each part.

Usage: python_frame_writer.py ROOT WORK

ROOT holds water.top and dodecahedron.gro of Inputs/triclinic: waters in a
rhombic dodecahedron at constant pressure, on the CPU, in the deterministic
mode, whose barostat changes the cell and its tilts. The end of each part
waits for the frames that the thread holds, and the entry gives the host
control only there, so the file holds every frame of a part when the part
returns (it is read there, with its reporter open), and the calls between
parts find nothing queued: a checkpoint, which counts the frames, a change
of the reporters, a frame of the state with its velocities, a commit of a
cell, the close of the reporters, and the end of a simulation. The script
runs one simulation through each of them. Run once with
MDIR_FRAME_WRITE_DELAY in the environment and once without, its files are
compared by the test."""
import gc
import pathlib
import sys

import numpy as np
import mdir

root, work = sys.argv[1], pathlib.Path(sys.argv[2])
work.mkdir(parents=True, exist_ok=True)
PERIOD = 10


def model():
    loaded = mdir.load_gromacs(root + "/water.top", root + "/dodecahedron.gro")
    system, state = loaded.make_system(), loaded.make_state()
    system.cutoff, system.pairlist_distance = 0.8, 0.9
    system.truncation = mdir.Truncation.None_
    system.electrostatics = mdir.Electrostatics.PME
    system.pme_grid = [28] * 3
    system.periodic = True
    system.rigid_water = True
    state = state.draw_velocities(system, 300.0, 2024)
    integrator, ensemble, execution = mdir.Integrator(), mdir.Ensemble(), mdir.Execution()
    integrator.method = mdir.IntegratorMethod.VelocityVerlet
    integrator.timestep = 0.002
    ensemble.seed = 2024
    ensemble.kind = mdir.EnsembleKind.NPT
    ensemble.temperature = 300.0
    ensemble.com_period = ensemble.coupling_period = PERIOD
    execution.target = mdir.Target.CPU
    execution.deterministic = True
    return mdir.compile(system, state, integrator, ensemble, execution, mdir.Schedule())


def dcd_frames(path):
    """The number of frames of a DCD file, from its records."""
    data, i, records = path.read_bytes(), 0, 0
    while i < len(data):
        i += int.from_bytes(data[i:i + 4], "little") + 8
        records += 1
    # The header, the title, and the number of particles; then the cell
    # and the three components of each frame.
    return (records - 3) // 4


def commit_cell(sim, scale):
    """Commits the cell scaled by `scale`, a small strain, with the
    positions as they are: the borrow reads and writes the cell alone."""
    with sim.borrow() as borrow:
        cell, tilt = np.from_dlpack(borrow.cell), np.from_dlpack(borrow.tilt)
        before = cell.copy(), tilt.copy()
        cell[...] = before[0] * scale
        tilt[...] = before[1] * scale
        del cell, tilt
        gc.collect()
        borrow.commit()
    now = sim.state().cell
    assert np.array_equal(now.diagonal, before[0] * scale), (now.diagonal, before[0])
    assert np.array_equal(now.tilt, before[1] * scale), (now.tilt, before[1])
    print(f"committed a cell {scale} times the one of step {sim.step}")


program = model()
sim = mdir.Simulation(program)
sim.reporters = [mdir.EnergyReporter(str(work / "run.dat"), PERIOD),
                 mdir.TrajectoryReporter(str(work / "run.dcd"), PERIOD)]
# Parts that end between frames and at a frame: after each, the file holds
# every frame of its steps, with the reporter open.
for steps in (35, 25, 40):
    sim.run(steps)
    print("frames after step", sim.step, dcd_frames(work / "run.dcd"))
sim.save_checkpoint(str(work / "first.h5"))
sim.run(20)
# A change of the reporters closes the trajectory.
sim.reporters = [mdir.EnergyReporter(str(work / "run.dat"), PERIOD),
                 mdir.H5MDReporter(str(work / "state.h5md"), PERIOD, velocities=True)]
sim.run(30)
sim.reporters = [mdir.EnergyReporter(str(work / "run.dat"), PERIOD),
                 mdir.TrajectoryReporter(str(work / "moved.dcd"), PERIOD)]
sim.run(25)
commit_cell(sim, 1.001)
sim.run(25)
sim.save_checkpoint(str(work / "second.h5"))
sim.run(20)
sim.close_reporters()
# The end of a simulation closes its files.
last = mdir.Simulation(program)
last.reporters = [mdir.H5MDReporter(str(work / "last.h5md"), PERIOD)]
last.run(40)
del last
gc.collect()
print("done at step", sim.step)
