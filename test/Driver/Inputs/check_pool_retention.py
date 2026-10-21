"""Compares the runtime's blocks of device memory of two runs of one input
that differ only in their number of steps (MDIR_DEVICE_MEMORY=2, D311),
and checks that the longer run holds no more blocks of the driver than the
shorter one: the blocks taken from the driver less those returned to it
(D[pool-retention], #471). Once the first steps have run, every block that
a step or a build of the neighbor structure asks for comes from the pool;
a buffer of the runtime that a growth replaces goes back to the driver.
It prints the blocks that each run holds and the free bytes of the pool at
its peak, which the report gives in MiB to one decimal.

Usage: check_pool_retention.py <short run's log> <long run's log>
"""
import re
import sys


def read(path):
    text = open(path).read()
    counts = re.search(
        r"mdrt: device memory: (\d+) allocations of blocks, (\d+) from the "
        r"pool, (\d+) returned to the driver", text)
    peaks = re.search(
        r"peak ([0-9.]+) MiB allocated, ([0-9.]+) MiB with the pool", text)
    if not counts or not peaks:
        sys.exit(f"{path}: no report of the device memory")
    allocations, reused, returned = (int(g) for g in counts.groups())
    held = allocations - reused - returned
    return held, float(peaks.group(2)) - float(peaks.group(1))


short_held, short_free = read(sys.argv[1])
long_held, long_free = read(sys.argv[2])
print(f"short run: {short_held} blocks of the driver, "
      f"{short_free:.1f} MiB free in the pool at its peak")
print(f"long run: {long_held} blocks of the driver, "
      f"{long_free:.1f} MiB free in the pool at its peak")
if long_held > short_held:
    print(f"the pool grew with the length of the run: "
          f"{long_held - short_held} more blocks of the driver")
    sys.exit(1)
print("the pool did not grow with the length of the run")
