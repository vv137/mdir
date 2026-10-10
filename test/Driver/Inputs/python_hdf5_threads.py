"""The one mutex of the process around the HDF5 library (#262,
docs/python-checkpoints.md): threads that write checkpoints of two
simulations, write frames in H5MD inside a run, read a checkpoint, and
read a trajectory in H5MD, all at once.

Usage: python_hdf5_threads.py ROOT TARGET SCRATCH [ITERATIONS]

The library of the reference build is not thread-safe, and
`save_checkpoint` writes outside the mutex of the runs and without the
GIL, so without the mutex two of these threads are inside the library
together. The test cannot prove that they never are; it exercises the
paths, and checks that every file read while others are written holds what
was written, and that the files written hold the states.
"""
import pathlib
import sys
import threading
import time

import numpy as np
import mdir

root = pathlib.Path(sys.argv[1])
target_name = sys.argv[2]
target = getattr(mdir.Target, target_name)
work = pathlib.Path(sys.argv[3])
iterations = int(sys.argv[4]) if len(sys.argv) > 4 else 120
FRAMES = 4


def bits(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def program():
    loaded = mdir.load_amber(str(root / "dipeptide/dipeptide.prmtop"),
                             str(root / "dipeptide/dipeptide.inpcrd"))
    system, state = loaded.make_system(), loaded.make_state()
    system.cutoff, system.pairlist_distance = 0.8, 0.9
    system.truncation = mdir.Truncation.None_
    system.electrostatics = mdir.Electrostatics.Cutoff
    state = state.draw_velocities(system, 300.0, 271828)
    integrator, ensemble, execution = mdir.Integrator(), mdir.Ensemble(), mdir.Execution()
    integrator.timestep, ensemble.temperature = 0.0005, 300.0
    ensemble.kind = mdir.EnsembleKind.NVE
    execution.target, execution.precision = target, mdir.Precision.Double
    execution.deterministic = True
    return mdir.compile(system, state, integrator, ensemble, execution, mdir.Schedule())


compiled = program()

# The files that the readers read while the others write: a checkpoint and
# a trajectory of a simulation that has ended.
states = []
reference = mdir.Simulation(compiled)
reference.reporters = [mdir.H5MDReporter(str(work / "fixed.h5md"), 1, velocities=True),
                       mdir.CallbackReporter(lambda s, state: states.append(state), 1)]
reference.run(FRAMES)
reference.close_reporters()
reference.save_checkpoint(str(work / "fixed.h5"))
fixed = reference.state()
del reference

# Two simulations whose checkpoints are written in two threads; the second
# also writes frames in H5MD inside its runs.
first, second = mdir.Simulation(compiled), mdir.Simulation(compiled)
first.run(3)
second.reporters = [mdir.H5MDReporter(str(work / "second.h5md"), 1)]
second.run(1)
start = threading.Barrier(4)
failures, counts = [], {}
RUNS = 6


def worker(name, body):
    def run():
        try:
            start.wait()
            done = 0
            for k in range(iterations):
                body(k)
                done += 1
            counts[name] = done
        except BaseException as exc:  # an assertion or an error of a thread
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
    return threading.Thread(target=run, name=name)


def save_first(k):
    first.save_checkpoint(str(work / "first.h5"))


def save_second(k):
    # A run of one step now and then: a frame written inside a part, under
    # the mutex of the runs, while the first simulation writes a checkpoint.
    if k % (iterations // RUNS) == 0:
        second.run(1)
    second.save_checkpoint(str(work / "second.h5"))


def read_checkpoint(k):
    read = mdir.read_checkpoint(str(work / "fixed.h5"))
    assert read.step == FRAMES and bits(read.positions, fixed.positions)
    assert bits(read.velocities, fixed.velocities)


def read_frames(k):
    with mdir.read_h5md(str(work / "fixed.h5md")) as frames:
        assert len(frames) == FRAMES
        for frame, state in zip(frames, states):
            assert frame.step == state.step
            assert bits(frame.positions, state.positions)
            assert bits(frame.velocities, state.velocities)


threads = [worker("checkpoints of the first simulation", save_first),
           worker("checkpoints and frames of the second", save_second),
           worker("reads of a checkpoint", read_checkpoint),
           worker("reads of a trajectory", read_frames)]
began = time.perf_counter()
for thread in threads:
    thread.start()
for thread in threads:
    thread.join()
elapsed = time.perf_counter() - began
assert not failures, failures
assert all(counts.get(thread.name) == iterations for thread in threads), counts

# What was written while the others wrote and read holds the states.
for sim, name in ((first, "first"), (second, "second")):
    read, state = mdir.read_checkpoint(str(work / f"{name}.h5")), sim.state()
    assert read.step == state.step and bits(read.positions, state.positions), name
    assert bits(read.velocities, state.velocities), name
second.close_reporters()
with mdir.read_h5md(str(work / "second.h5md")) as frames:
    assert len(frames) == 1 + RUNS and list(frames.steps) == list(range(1, 2 + RUNS))
    last = frames[len(frames) - 1]
    assert bits(last.positions, second.state().positions)
print(f"{4 * iterations} operations in {elapsed:.2f} s", file=sys.stderr)
print(f"{target_name}: {iterations} checkpoints of each of two simulations, {RUNS} frames in H5MD, "
      f"{iterations} reads of a checkpoint and {iterations} of a trajectory in four threads: "
      "every file holds what was written")
print("hdf5 threads passed")
