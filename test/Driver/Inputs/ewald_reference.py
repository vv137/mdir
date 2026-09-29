"""The Coulomb energy of a box of four-site waters by the Ewald sum, in pure
Python: the direct sum with erfc(β r) / r over the pairs within the cutoff
that are not excluded, the excluded pairs with −erf(β r) / r, the self term,
the background of a net charge, and the reciprocal sum over many wave
vectors (docs/pme-m1.md, Section 1).

    ewald_reference.py topology.prmtop coordinates.inpcrd cutoff-Å beta-per-nm

The topology holds waters only: the pairs of a residue are excluded. The
extra point of each water is placed as sander places it, from the bond that
gives its distance. Prints each term in kcal/mol."""

import math
import sys

F = 138.935457644  # kJ mol⁻¹ nm e⁻², CODATA 2018 (docs/conventions.md)
KCAL = 4.184


def read_prmtop(path):
    sections, flag, fmt = {}, None, None
    for line in open(path):
        if line.startswith("%FLAG"):
            flag = line.split()[1]
            sections[flag] = []
        elif line.startswith("%FORMAT"):
            fmt = line[line.index("(") + 1 : line.index(")")]
        elif line.startswith("%"):
            continue
        elif flag:
            width = int("".join(c for c in fmt.split("a")[-1].split("I")[-1].split("E")[-1].split(".")[0] if c.isdigit()))
            for i in range(0, len(line.rstrip("\n")), width):
                field = line[i : i + width]
                if field.strip():
                    sections[flag].append(field.strip())
    return sections


top = read_prmtop(sys.argv[1])
charges = [float(q) / 18.2223 for q in top["CHARGE"]]
count = len(charges)
starts = [int(s) - 1 for s in top["RESIDUE_POINTER"]] + [count]
types = top["AMBER_ATOM_TYPE"]
lines = open(sys.argv[2]).read().splitlines()
numbers = []
for line in lines[2:]:
    numbers += [float(line[i : i + 12]) for i in range(0, len(line), 12) if line[i : i + 12].strip()]
x = [[0.1 * numbers[3 * i + k] for k in range(3)] for i in range(count)]
box = [0.1 * numbers[3 * count + k] for k in range(3)]
rc = 0.1 * float(sys.argv[3])
beta = float(sys.argv[4])

# The extra points, at the distance of their bond along the bisector.
bonds = [int(v) for v in top["BONDS_WITHOUT_HYDROGEN"]]
lengths = [0.1 * float(v) for v in top["BOND_EQUIL_VALUE"]]
for n in range(0, len(bonds), 3):
    i, j, t = bonds[n] // 3, bonds[n + 1] // 3, bonds[n + 2] - 1
    site, owner = (i, j) if types[i] == "EP" else (j, i)
    if types[site] != "EP":
        continue
    h = [a for a in range(owner + 1, owner + 3)]
    units = []
    for a in h:
        d = [x[a][k] - x[owner][k] for k in range(3)]
        norm = math.sqrt(sum(c * c for c in d))
        units.append([c / norm for c in d])
    s = [units[0][k] + units[1][k] for k in range(3)]
    norm = math.sqrt(sum(c * c for c in s))
    x[site] = [x[owner][k] + lengths[t] * s[k] / norm for k in range(3)]

residue = [0] * count
for r in range(len(starts) - 1):
    for i in range(starts[r], starts[r + 1]):
        residue[i] = r


def image(d, k):
    return d - box[k] * round(d / box[k])


direct = excluded = 0.0
for i in range(count):
    for j in range(i + 1, count):
        d = [image(x[i][k] - x[j][k], k) for k in range(3)]
        r = math.sqrt(sum(c * c for c in d))
        qq = F * charges[i] * charges[j]
        if residue[i] == residue[j]:
            excluded -= qq * math.erf(beta * r) / r
        elif r < rc:
            direct += qq * math.erfc(beta * r) / r
volume = box[0] * box[1] * box[2]
self = -F * beta / math.sqrt(math.pi) * sum(q * q for q in charges)
net = sum(charges)
background = -F * math.pi * net * net / (2 * volume * beta * beta)

# The reciprocal sum, over wave vectors until the Gaussian is 1e-18.
reciprocal = 0.0
limit = beta * math.sqrt(18 * math.log(10)) / math.pi
kmax = [int(limit * box[k]) + 1 for k in range(3)]
for a in range(0, kmax[0] + 1):
    for b in range(-kmax[1], kmax[1] + 1):
        for c in range(-kmax[2], kmax[2] + 1):
            if a == 0 and (b < 0 or (b == 0 and c <= 0)):
                continue  # each ±m once, and m = 0 never
            m = (a / box[0], b / box[1], c / box[2])
            m2 = m[0] * m[0] + m[1] * m[1] + m[2] * m[2]
            if m2 > limit * limit:
                continue
            re = im = 0.0
            for i in range(count):
                phase = 2 * math.pi * (m[0] * x[i][0] + m[1] * x[i][1] + m[2] * x[i][2])
                re += charges[i] * math.cos(phase)
                im += charges[i] * math.sin(phase)
            reciprocal += 2 * F / (2 * math.pi * volume) * math.exp(-math.pi ** 2 * m2 / beta ** 2) / m2 * (re * re + im * im)

for name, value in (("direct", direct), ("excluded", excluded),
                    ("reciprocal", reciprocal), ("self", self + background),
                    ("total", direct + excluded + reciprocal + self + background)):
    print(f"{name:12s} {value / KCAL:.9f}")
