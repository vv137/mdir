"""Writes a file of coordinates of Amber with velocities: zero for every
particle but one, which takes the velocity given.

    kick_atom.py input.inpcrd output.rst7 atom vx vy vz

The atom is numbered from 1, as in the topology; the velocity is in Å/ps.
The file holds velocities in Å per 1/20.455 ps."""

import sys

source, target = sys.argv[1], sys.argv[2]
atom = int(sys.argv[3]) - 1
kick = [float(value) / 20.455 for value in sys.argv[4:7]]

with open(source) as file:
    lines = file.read().splitlines()
title = lines[0]
count = int(lines[1].split()[0])
numbers = []
for line in lines[2:]:
    numbers += [float(line[i : i + 12]) for i in range(0, len(line), 12)
                if line[i : i + 12].strip()]
positions, box = numbers[: 3 * count], numbers[3 * count :]
velocities = [0.0] * (3 * count)
velocities[3 * atom : 3 * atom + 3] = kick


def write_block(file, values):
    for start in range(0, len(values), 6):
        file.write("".join(f"{v:12.7f}" for v in values[start : start + 6]))
        file.write("\n")


with open(target, "w") as file:
    file.write(title + "\n")
    file.write(f"{count:6d}{0.0:15.7e}\n")
    write_block(file, positions)
    write_block(file, velocities)
    write_block(file, box)
