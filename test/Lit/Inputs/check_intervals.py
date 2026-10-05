"""Checks the intervals that the tests of Inputs/serial recorded: at most
<limit> tests that need CUDA run at once (and, for a limit above 1, some of
them do run together), and the tests that do not need it overlap."""
import itertools
import pathlib
import sys

limit = int(sys.argv[2])
intervals = {}
for path in pathlib.Path(sys.argv[1]).glob("*.interval"):
    start, end = map(float, path.read_text().split())
    intervals[path.stem] = (start, end)
gpu = sorted(v for k, v in intervals.items() if k.startswith("gpu-"))
cpu = sorted(v for k, v in intervals.items() if k.startswith("cpu-"))
assert len(gpu) == 4 and len(cpu) == 3, sorted(intervals)


def overlap(a, b):
    return a[0] < b[1] and b[0] < a[1]


# The most GPU tests running at one instant, which is the start of a test.
most = max(sum(a <= s < b for a, b in gpu) for s, _ in gpu)
assert most <= limit, f"{most} GPU tests ran at once, more than {limit}: {gpu}"
if limit > 1:
    assert most > 1, f"GPU tests never shared the device: {gpu}"
assert any(overlap(a, b) for a, b in itertools.combinations(cpu, 2)), \
    f"no CPU tests overlap: {cpu}"
print(f"at most {limit} GPU tests at once, CPU tests side by side")
