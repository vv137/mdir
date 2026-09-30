"""Writes an Amber topology with the masses of the hydrogens of its waters
(residue WAT) repartitioned: 3.024 amu each, taken from the oxygen.

    repartition.py topology.prmtop > heavy.prmtop"""

import sys

lines = open(sys.argv[1]).read().split("\n")
sections, order, flag = {}, [], None
for number, line in enumerate(lines):
    if line.startswith("%FLAG"):
        flag = line.split()[1]
        order.append(flag)
        sections[flag] = [number, None, number]
    elif line.startswith("%FORMAT") and flag:
        sections[flag][1] = number
    if flag:
        sections[flag][2] = number


def values(flag, width):
    start, form, end = sections[flag]
    text = "".join(line for line in lines[form + 1 : end + 1] if not line.startswith("%"))
    return [text[i : i + width] for i in range(0, len(text), width) if text[i : i + width].strip()]


names = [v.strip() for v in values("ATOM_NAME", 4)]
masses = [float(v) for v in values("MASS", 16)]
labels = [v.strip() for v in values("RESIDUE_LABEL", 4)]
pointers = [int(v) - 1 for v in values("RESIDUE_POINTER", 8)] + [len(masses)]
for r, label in enumerate(labels):
    if label != "WAT":
        continue
    oxygen = pointers[r]
    for atom in range(pointers[r], pointers[r + 1]):
        if names[atom].startswith("H"):
            masses[oxygen] -= 3.024 - masses[atom]
            masses[atom] = 3.024
start, form, end = sections["MASS"]
body = ["".join("%16.8E" % m for m in masses[i : i + 5]) for i in range(0, len(masses), 5)]
lines[form + 1 : end + 1] = body
print("\n".join(lines), end="")
