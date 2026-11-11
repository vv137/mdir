"""Failures of the frames of a Python simulation (D[async-output], #113).

Usage: python_frame_failures.py ROOT WORK MODE TARGET

ROOT holds water.top and dodecahedron.gro of Inputs/triclinic. MODE:

  write       with MDIR_FRAME_WRITE_FAIL in the environment, the write of a
              frame to the trajectory in H5MD fails on the thread of the
              frames: the part raises the frame's error, the frames before
              it are in the file, and the waits after the part (a
              checkpoint, the close of the reporters, a second
              simulation's run and end) go on.
  incomplete  with MDIR_STAGING_INCOMPLETE in the environment, on a GPU,
              the call of each frame finds the staging functions
              incomplete and gives its slot back: a part of 40 steps, whose
              segment goes on to its end after the failure (D196), calls 4
              frames, one more than the ring has slots, and raises the
              error; so does the part of a second simulation, while the
              first is kept."""
import pathlib
import sys

import numpy as np
import mdir

root, work, mode, target = sys.argv[1], pathlib.Path(sys.argv[2]), sys.argv[3], sys.argv[4]
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
    ensemble.kind = mdir.EnsembleKind.NVE
    execution.target = getattr(mdir.Target, target)
    execution.deterministic = True
    return mdir.compile(system, state, integrator, ensemble, execution, mdir.Schedule())


def attempt(what, call):
    try:
        call()
        print(what, "returned")
    except Exception as error:  # the type is printed
        print(what, "raised", type(error).__name__ + ":", str(error).strip().splitlines()[-1])


program = model()
if mode == "write":
    sim = mdir.Simulation(program)
    sim.reporters = [mdir.H5MDReporter(str(work / "run.h5md"), PERIOD)]
    attempt("the part of 100 steps", lambda: sim.run(100))
    print("step", sim.step)
    print("frames", *[int(f.step) for f in mdir.read_h5md(str(work / "run.h5md"))])
    attempt("a checkpoint", lambda: sim.save_checkpoint(str(work / "after.h5")))
    attempt("the close of the reporters", sim.close_reporters)
    other = mdir.Simulation(program)
    other.reporters = [mdir.TrajectoryReporter(str(work / "other.dcd"), PERIOD)]
    attempt("a second simulation of 30 steps", lambda: other.run(30))
    del other
    print("alive")
elif mode == "incomplete":
    kept = []
    for n in (1, 2):
        sim = mdir.Simulation(program)
        sim.reporters = [mdir.TrajectoryReporter(str(work / f"run{n}.dcd"), PERIOD)]
        attempt(f"simulation {n}, 40 steps", lambda: sim.run(4 * PERIOD))
        kept.append(sim)
    del kept
    print("alive")
