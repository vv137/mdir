"""Checks the intervals that the tests of Inputs/serial recorded: the tests
that need CUDA never overlap, and the others overlap at least once."""
import itertools
import pathlib
import sys

intervals = {}
for path in pathlib.Path(sys.argv[1]).glob("*.interval"):
    start, end = map(float, path.read_text().split())
    intervals[path.stem] = (start, end)
gpu = sorted(v for k, v in intervals.items() if k.startswith("gpu-"))
cpu = sorted(v for k, v in intervals.items() if k.startswith("cpu-"))
assert len(gpu) == 4 and len(cpu) == 3, sorted(intervals)


def overlap(a, b):
    return a[0] < b[1] and b[0] < a[1]


for a, b in itertools.combinations(gpu, 2):
    assert not overlap(a, b), f"GPU tests overlap: {a} {b}"
assert any(overlap(a, b) for a, b in itertools.combinations(cpu, 2)), \
    f"no CPU tests overlap: {cpu}"
print("GPU tests one at a time, CPU tests side by side")
