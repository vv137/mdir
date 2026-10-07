"""Checkpoints of a Python simulation in a build without HDF5
(D[python-checkpoints]): each use raises UnsupportedError.

Usage: python_checkpoints_no_hdf5.py ROOT WORK."""
import pathlib
import sys

import mdir

root = sys.argv[1]
work = pathlib.Path(sys.argv[2])
loaded = mdir.load_amber(root + "/dipeptide.prmtop", root + "/dipeptide.inpcrd")
program = mdir.compile(loaded.make_system(), loaded.make_state(), mdir.Integrator(),
                       mdir.Ensemble(), mdir.Execution(), mdir.Schedule())


def expect_unsupported(call):
    try:
        call()
    except mdir.UnsupportedError as exc:
        assert "HDF5" in str(exc), str(exc)
        return
    raise AssertionError("expected UnsupportedError")


simulation = mdir.Simulation(program)
expect_unsupported(lambda: simulation.save_checkpoint(str(work / "a.h5")))
expect_unsupported(lambda: mdir.read_checkpoint(str(work / "a.h5")))
expect_unsupported(lambda: mdir.Simulation(program, checkpoint=str(work / "a.h5")))
simulation.reporters = [mdir.CheckpointReporter(str(work / "a.h5"), 1)]
expect_unsupported(lambda: simulation.run(1))
print("checkpoints without HDF5 raise UnsupportedError")
