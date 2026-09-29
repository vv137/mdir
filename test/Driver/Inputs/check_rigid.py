"""Checks that the rigid waters of a trajectory keep their shape: the
distances O-H1, O-H2, and H1-H2 of the waters that begin at the given
atoms, in every frame of a DCD file, against the given lengths in Å.

    check_rigid.py trajectory.dcd OH HH first-oxygen [more oxygens...]

Positions are 32-bit numbers, so the distances agree to about 1e-5 Å."""

import math
import struct
import sys


def records(data):
    at = 0
    while at < len(data):
        (length,) = struct.unpack_from("<i", data, at)
        yield data[at + 4 : at + 4 + length]
        at += 8 + length


path, oh, hh = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
oxygens = [int(a) for a in sys.argv[4:]]
blocks = list(records(open(path, "rb").read()))
header = blocks[0]
has_cell = struct.unpack_from("<i", header, 4 + 10 * 4)[0] != 0
count = struct.unpack_from("<i", blocks[2])[0]
frames = blocks[3:]
step = 4 if has_cell else 3
worst = 0.0
for start in range(0, len(frames), step):
    xyz = frames[start + (1 if has_cell else 0) : start + step]
    coordinate = [struct.unpack(f"<{count}f", block) for block in xyz]

    def distance(i, j):
        return math.sqrt(sum((c[i] - c[j]) ** 2 for c in coordinate))

    for o in oxygens:
        for (i, j), length in (((o, o + 1), oh), ((o, o + 2), oh), ((o + 1, o + 2), hh)):
            worst = max(worst, abs(distance(i, j) - length))
print(f"frames {len(frames) // step}, largest deviation {worst:.1e} Å")
print("rigid " + ("ok" if worst < 5e-5 else "FAILED"))
