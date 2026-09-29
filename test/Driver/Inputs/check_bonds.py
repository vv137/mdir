"""Checks that the bonds of hydrogen of an Amber topology keep their
lengths in every frame of a DCD trajectory: BONDS_INC_HYDROGEN against
BOND_EQUIL_VALUE, in Å. Positions are 32-bit numbers, so the lengths agree
to about 1e-5 Å.

    check_bonds.py topology.prmtop trajectory.dcd"""

import math
import struct
import sys


def read_prmtop(path):
    sections, flag, width = {}, None, None
    for line in open(path):
        if line.startswith("%FLAG"):
            flag = line.split()[1]
            sections[flag] = []
        elif line.startswith("%FORMAT"):
            spec = line[line.index("(") + 1 : line.index(")")]
            letter = next(c for c in spec if c.isalpha())
            width = int(spec.split(letter)[1].split(".")[0])
        elif line.startswith("%"):
            continue
        elif flag:
            text = line.rstrip("\n")
            sections[flag] += [text[i : i + width].strip() for i in range(0, len(text), width) if text[i : i + width].strip()]
    return sections


def records(data):
    at = 0
    while at < len(data):
        (length,) = struct.unpack_from("<i", data, at)
        yield data[at + 4 : at + 4 + length]
        at += 8 + length


top = read_prmtop(sys.argv[1])
bonds = [int(v) for v in top["BONDS_INC_HYDROGEN"]]
lengths = [float(v) for v in top["BOND_EQUIL_VALUE"]]
pairs = [(bonds[n] // 3, bonds[n + 1] // 3, lengths[bonds[n + 2] - 1]) for n in range(0, len(bonds), 3)]
blocks = list(records(open(sys.argv[2], "rb").read()))
has_cell = struct.unpack_from("<i", blocks[0], 4 + 10 * 4)[0] != 0
count = struct.unpack_from("<i", blocks[2])[0]
frames = blocks[3:]
step = 4 if has_cell else 3
worst = 0.0
for start in range(0, len(frames), step):
    xyz = frames[start + (1 if has_cell else 0) : start + step]
    c = [struct.unpack(f"<{count}f", block) for block in xyz]
    for i, j, length in pairs:
        r = math.sqrt(sum((c[k][i] - c[k][j]) ** 2 for k in range(3)))
        worst = max(worst, abs(r - length))
print(f"bonds {len(pairs)}, frames {len(frames) // step}, largest deviation {worst:.1e} Å")
print("bonds " + ("ok" if worst < 1e-4 else "FAILED"))
