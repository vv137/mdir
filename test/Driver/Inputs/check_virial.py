"""Checks the trace of the virial of a run from an Amber topology against
the derivative of the energy under a uniform scaling of the positions and
the cell by λ: tr W = −dU/dλ at λ = 1, with virtual sites placed from their
scaled atoms.

    check_virial.py mdir run.toml coordinates.inpcrd

The derivative is a central difference at two steps, extrapolated to a step
of 0 (Richardson), from the total energy of the terms at the start."""

import os
import re
import subprocess
import sys

mdir, control, coordinates = sys.argv[1:4]
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
scaled = os.path.join(directory, "scaled.inpcrd")
text = open(control).read().replace(os.path.basename(coordinates), "scaled.inpcrd")
scaled_control = os.path.join(directory, "scaled.toml")
open(scaled_control, "w").write(text)


def run(factor):
    with open(scaled, "w") as file:
        file.write(lines[0] + "\n" + f"{count:6d}\n")
        values = [v * factor for v in positions]
        for i in range(0, len(values), 6):
            file.write("".join(f"{v:12.7f}" for v in values[i : i + 6]) + "\n")
        edges = [v * factor for v in cell[:3]] + cell[3:]
        file.write("".join(f"{v:12.7f}" for v in edges) + "\n")
    log = subprocess.run(
        [mdir, "run", scaled_control], capture_output=True, text=True, check=True
    ).stdout
    total = float(re.search(r"MDIR:   total +(\S+)", log).group(1))
    row = re.search(r"INFO: +0 .*", log).group(0).split()
    return total, float(row[7])


def difference(step):
    return -(run(1 + step)[0] - run(1 - step)[0]) / (2 * step)


coarse, fine = difference(2e-3), difference(1e-3)
derivative = fine + (fine - coarse) / 3
trace = run(1.0)[1]
error = abs(derivative - trace) / abs(trace)
print(f"-dU/dlambda {derivative:.4f}, trace of W {trace:.4f}")
print("virial " + ("ok" if error < 2e-5 else "FAILED") + f": {error:.1e}")
