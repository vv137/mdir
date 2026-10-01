"""Writes scrambled.top and scrambled.gro from system.top and generate.py:
propane with its carbons first and its hydrogens after them, so that the
members of a group of SHAKE (a carbon and its hydrogens) do not follow one
another, and many molecules of it among water, in a box of 3 nm.

Usage: python3 scrambled.py DIRECTORY
"""
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate  # noqa: E402

# The atoms of propane in the order of system.top (1 to 11) by their new
# numbers: C1 C2 C3, then H11 H12 H13 H21 H22 H31 H32 H33.
ORDER = [1, 5, 8, 2, 3, 4, 6, 7, 9, 10, 11]
NEW = {old: new for new, old in enumerate(ORDER, start=1)}
PROPANES, WATERS, EDGE, SIDE = 60, 65, 3.0, 5


def unit(v):
    norm = math.sqrt(sum(c * c for c in v))
    return [c / norm for c in v]


def cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]]


def propane():
    """Positions in nm of C1 C2 C3 H11 H12 H13 H21 H22 H31 H32 H33, with
    tetrahedral angles and bonds of 0.1526 and 0.109 nm."""
    cc, ch = 0.1526, 0.109
    half = math.radians(109.47) / 2.0
    c1 = [-cc * math.sin(half), -cc * math.cos(half), 0.0]
    c2 = [0.0, 0.0, 0.0]
    c3 = [cc * math.sin(half), -cc * math.cos(half), 0.0]
    tilt = math.radians(180.0 - 109.47)

    def methyl(carbon, neighbor):
        axis = unit([a - b for a, b in zip(carbon, neighbor)])
        side = unit(cross(axis, [0.0, 0.0, 1.0]))
        other = cross(axis, side)
        return [[carbon[k] + ch * (math.cos(tilt) * axis[k] + math.sin(tilt) * (
                    math.cos(phi) * side[k] + math.sin(phi) * other[k]))
                 for k in range(3)]
                for phi in (0.0, 2.0 * math.pi / 3.0, 4.0 * math.pi / 3.0)]

    bisector = unit([-(a + b) for a, b in zip(c1, c3)])
    normal = [0.0, 0.0, 1.0]
    open_ = math.radians(109.47) / 2.0
    middle = [[c2[k] + ch * (math.cos(open_) * bisector[k] + sign * math.sin(open_) * normal[k])
               for k in range(3)] for sign in (1.0, -1.0)]
    positions = [c1, c2, c3] + methyl(c1, c2) + middle + methyl(c3, c2)
    names = ["C1", "C2", "C3", "H11", "H12", "H13", "H21", "H22", "H31",
             "H32", "H33"]
    return "PRO", list(zip(names, positions))


def scrambled_top(text):
    """The topology with the atoms of propane renumbered and its count."""
    start = text.index("[ moleculetype ]")
    end = text.index('#include "water.itp"')
    section, out = None, []
    atoms = []
    for line in text[start:end].splitlines():
        header = re.match(r"\s*\[\s*(\w+)\s*\]", line)
        if header:
            if atoms:
                out += [a for _, a in sorted(atoms)]
                atoms = []
            section = header.group(1)
            out.append(line)
            continue
        fields = line.split()
        if not fields or line.lstrip().startswith(";") or section == "moleculetype":
            out.append(line)
            continue
        if section == "atoms":
            new = NEW[int(fields[0])]
            atoms.append((new, "  %2d" % new + line[len(line) - len(line.lstrip()) + len(fields[0]):]))
            continue
        out.append("  " + " ".join(str(NEW[int(f)]) for f in fields))
    if atoms:
        out += [a for _, a in sorted(atoms)]
    body = "\n".join(out) + "\n"
    tail = text[end:].replace("PRO   4", "PRO  %d" % PROPANES).replace(
        "SOL  60", "SOL  %d" % WATERS)
    return text[:start] + body + tail


def main():
    directory = sys.argv[1]
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "system.top")) as file:
        top = scrambled_top(file.read())
    with open(os.path.join(directory, "scrambled.top"), "w") as file:
        file.write(top)
    molecules = ([propane() for _ in range(PROPANES)] +
                 [generate.water() for _ in range(WATERS)])
    spacing = EDGE / SIDE
    lines, atom = [], 0
    for index, (residue, members) in enumerate(molecules):
        cell = [index % SIDE, (index // SIDE) % SIDE, index // (SIDE * SIDE)]
        centre = [(c + 0.5) * spacing for c in cell]
        matrix = generate.rotation()
        for name, point in members:
            atom += 1
            x = [centre[k] + v for k, v in enumerate(generate.turn(matrix, point))]
            lines.append("%5d%-5s%5s%5d%12.6f%12.6f%12.6f" % (
                index + 1, residue, name, atom, x[0], x[1], x[2]))
    with open(os.path.join(directory, "scrambled.gro"), "w") as out:
        out.write("Propane with its hydrogens last, and water\n%5d\n" % atom)
        out.write("\n".join(lines) + "\n")
        out.write("%10.5f%10.5f%10.5f\n" % (EDGE, EDGE, EDGE))


if __name__ == "__main__":
    main()
