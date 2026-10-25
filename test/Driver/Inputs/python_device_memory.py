"""A long Python process that alternates simulations of other sizes, grids,
and precisions on one device (#622): the idle blocks of the pool stay
within its budget, and `mdir.trim_device_memory()` gives them and the
plans of cuFFT back, so that the blocks the driver holds for the pool are
none once every simulation is gone.

Usage: python_device_memory.py ROOT LIBMDRT_CUDA

The budget of the pool is MDIR_DEVICE_POOL_BYTES of the test.
"""
import ctypes
import gc
import os
import sys

import mdir

root, library = sys.argv[1], sys.argv[2]
budget = int(os.environ["MDIR_DEVICE_POOL_BYTES"])
NAMES = ("live", "live bytes", "idle", "idle bytes", "driver blocks", "plans", "plan bytes", "free")


def counts():
    values = (ctypes.c_int64 * 8)()
    ctypes.CDLL(library).mdrtDeviceMemoryCounts(values)
    return dict(zip(NAMES, values))


def simulate(system_name, grid, precision):
    data = mdir.load_amber(f"{root}/{system_name}/{system_name}.prmtop",
                           f"{root}/{system_name}/{system_name}.inpcrd")
    system, state = data.make_system(), data.make_state()
    system.periodic = True
    system.electrostatics = mdir.Electrostatics.PME
    system.pme_grid = [grid, grid, grid]
    execution = mdir.Execution()
    execution.target = mdir.Target.GPU
    execution.precision = getattr(mdir.Precision, precision)
    sim = mdir.Simulation(mdir.compile(system, state, mdir.Integrator(), mdir.Ensemble(),
                                       execution, mdir.Schedule()))
    sim.run(10)
    del sim
    gc.collect()


most = 0
for system_name, grid, precision in [("dipeptide", 24, "Double"), ("opc", 32, "Mixed"),
                                     ("dipeptide", 36, "Mixed"), ("opc", 40, "Double"),
                                     ("dipeptide", 24, "Double")]:
    simulate(system_name, grid, precision)
    c = counts()
    assert c["live"] == 0, c
    assert c["idle bytes"] <= budget, (c, budget)
    most = max(most, c["idle bytes"])
    print(f"{system_name} grid {grid} {precision}: no block in use, idle within the budget")
print("the most idle at once within the budget:", most <= budget)
released = mdir.trim_device_memory()
c = counts()
assert released > 0 and c["idle"] == 0 and c["plans"] == 0 and c["driver blocks"] == 0, c
print("after the trim: no idle block, no plan, no block of the pool held by the driver")
