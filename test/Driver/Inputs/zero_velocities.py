"""Writes the coordinates of an Amber file of coordinates (inpcrd) as a
restart whose velocities are all 0.

    zero_velocities.py COORDINATES > RESTART"""

import sys

lines = open(sys.argv[1]).read().split("\n")
count = int(lines[1].split()[0])
rows = (3 * count + 5) // 6
coordinates, box = lines[2:2 + rows], lines[2 + rows]
zeros = ["".join(f"{0.0:12.7f}" for _ in range(min(6, 3 * count - i)))
         for i in range(0, 3 * count, 6)]
print("\n".join([lines[0], f"{count:6d}  0.0000000E+00"] + coordinates +
                zeros + [box]))
