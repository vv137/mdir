"""Compares the runtime's blocks of device memory of two runs of one input
that differ only in their number of steps (MDIR_DEVICE_MEMORY=2, D311),
and checks that the longer run holds at most SLACK blocks of the driver
more than the shorter one: the blocks taken from the driver less those
returned to it (D[pool-retention], #471). Once the first steps have run,
a step or a build of the neighbor structure takes its blocks from the
pool, and a buffer of the runtime that a growth replaces goes back to the
driver. The slack allows for a step that has more blocks in use at once
than the others and takes a block or two of the driver once: in 1 of 21
runs of 1,000 steps on an RTX 4090 the branch held 2 blocks more than the
run of 100 steps, with a higher peak in use (the step was not isolated);
in the other 20 it held the same. It prints the blocks that each run holds and the free
bytes of the pool at its peak, which the report gives in MiB to one
decimal.

Usage: check_pool_retention.py <short run's log> <long run's log> SLACK
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
slack = int(sys.argv[3])
print(f"short run: {short_held} blocks of the driver, "
      f"{short_free:.1f} MiB free in the pool at its peak")
print(f"long run: {long_held} blocks of the driver, "
      f"{long_free:.1f} MiB free in the pool at its peak")
if long_held > short_held + slack:
    print(f"the pool grew with the length of the run: "
          f"{long_held - short_held} more blocks of the driver")
    sys.exit(1)
print("the pool did not grow with the length of the run")
