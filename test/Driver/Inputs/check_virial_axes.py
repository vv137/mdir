"""Checks each diagonal component of the virial of a run from an Amber
topology against the derivative of the energy under a strain of that axis
alone, x_a → (1 + ε) x_a with the cell's edge a: W_aa = −dU/dε_aa at
ε = 0, with virtual sites placed from their strained atoms. The
anisotropic counterpart of check_virial.py: it holds the terms m_a m_a / m²
of a reciprocal virial, which a uniform scaling sums away
(D266).

    check_virial_axes.py mdir run.toml coordinates.inpcrd [step tolerance]

The derivative is a central difference at the steps `step` and `step / 2`,
extrapolated to a step of 0 (Richardson), relative to the largest of the
three components."""

import os
import re
import subprocess
import sys

mdir, control, coordinates = sys.argv[1:4]
step = float(sys.argv[4]) if len(sys.argv) > 4 else 2e-3
tolerance = float(sys.argv[5]) if len(sys.argv) > 5 else 2e-5
lines = open(coordinates).read().splitlines()
count = int(lines[1].split()[0])
numbers = []
for line in lines[2:]:
    numbers += [
        float(line[i : i + 12])
        for i in range(0, len(line), 12)
        if line[i : i + 12].strip()
    ]
positions, cell = numbers[: 3 * count], numbers[3 * count :]
directory = os.path.dirname(os.path.abspath(control))
scaled = os.path.join(directory, "strained.inpcrd")
text = open(control).read().replace(os.path.basename(coordinates), "strained.inpcrd")
scaled_control = os.path.join(directory, "strained.toml")
open(scaled_control, "w").write(text)


def run(axis, factor):
    with open(scaled, "w") as file:
        file.write(lines[0] + "\n" + f"{count:6d}\n")
        values = [v * (factor if k % 3 == axis else 1.0) for k, v in enumerate(positions)]
        for i in range(0, len(values), 6):
            file.write("".join(f"{v:12.7f}" for v in values[i : i + 6]) + "\n")
        edges = [v * (factor if k == axis else 1.0) for k, v in enumerate(cell[:3])] + cell[3:]
        file.write("".join(f"{v:12.7f}" for v in edges) + "\n")
    log = subprocess.run(
        [mdir, "run", scaled_control], capture_output=True, text=True, check=True
    ).stdout
    total = float(re.search(r"MDIR:   total +(\S+)", log).group(1))
    diagonal = re.search(r"the diagonal of the virial at the start.*\nMDIR: +(\S+) +(\S+) +(\S+)", log)
    return total, [float(diagonal.group(k)) for k in (1, 2, 3)]


def difference(axis, h):
    return -(run(axis, 1 + h)[0] - run(axis, 1 - h)[0]) / (2 * h)


virial = run(0, 1.0)[1]
largest = max(abs(v) for v in virial)
worst = 0.0
for axis in range(3):
    coarse, fine = difference(axis, step), difference(axis, step / 2)
    derivative = fine + (fine - coarse) / 3
    error = abs(derivative - virial[axis]) / largest
    worst = max(worst, error)
    print(f"axis {'xyz'[axis]}: -dU/deps {derivative:.4f}, W {virial[axis]:.4f}, {error:.1e}")
print("virial of the axes " + ("ok" if worst < tolerance else "FAILED") + f": {worst:.1e}")
