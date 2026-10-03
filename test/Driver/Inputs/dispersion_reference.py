"""The Lennard-Jones of a small topology of GROMACS with the dispersion by the
Ewald sum (D162), in pure Python: the grid sums −c_i c_j / r⁶ over all pairs
and images with c_i = 2 √ε σ³ of the type of i with itself, and

  direct      the Lennard-Jones of each pair within the cutoff that is not
              excluded, with its own σ and ε (the rule, or [ nonbond_params ]),
              plus c_i c_j (1 − g(β r)) / r⁶, both shifted to 0 at the cutoff
  excluded    c_i c_j (1 − g(β r)) / r⁶ over the excluded pairs
  reciprocal  −π^{3/2} β³ / (2V) Σ_m F(π |m| / β) / 3 |S(m)|², summed over
              the wave vectors instead of a grid, the term m = 0 included,
              F(b) = (1 − 2b²) exp(−b²) + 2 √π b³ erfc(b)
  self        β⁶ / 12 Σ c_i²

with g(x) = exp(−x²) (1 + x² + x⁴/2) [Essmann1995]. Prints each in kcal/mol.

    dispersion_reference.py system.top system.gro cutoff-Å beta-per-Å

The topology has [ defaults ] with the rule of Lorentz and Berthelot (2),
[ atomtypes ] with σ and ε, [ nonbond_params ], and molecules of [ atoms ]
and [ bonds ], whose pairs within nrexcl bonds are excluded."""

import math
import sys

KCAL = 4.184


def sections(path):
    current, result = None, []
    for raw in open(path):
        line = raw.split(";")[0].strip()
        if not line:
            continue
        if line.startswith("["):
            current = line.strip("[] ")
            result.append((current, []))
        else:
            result[-1][1].append(line.split())
    return result


types, overrides, molecules, order = {}, {}, {}, []
molecule = None
for name, rows in sections(sys.argv[1]):
    if name == "atomtypes":
        for row in rows:
            types[row[0]] = (float(row[-2]), float(row[-1]))
    elif name == "nonbond_params":
        for row in rows:
            overrides[frozenset((row[0], row[1]))] = (float(row[3]), float(row[4]))
    elif name == "moleculetype":
        molecule = {"name": rows[0][0], "nrexcl": int(rows[0][1]),
                    "atoms": [], "bonds": []}
        molecules[molecule["name"]] = molecule
    elif name == "atoms":
        molecule["atoms"] = [row[1] for row in rows]
    elif name == "bonds":
        molecule["bonds"] = [(int(row[0]) - 1, int(row[1]) - 1) for row in rows]
    elif name == "molecules":
        order = [(row[0], int(row[1])) for row in rows]

lines = open(sys.argv[2]).read().splitlines()
count = int(lines[1])
x = [[float(v) for v in line[20:].split()[:3]] for line in lines[2 : 2 + count]]
box = [float(v) for v in lines[2 + count].split()[:3]]
rc = 0.1 * float(sys.argv[3])
beta = 10.0 * float(sys.argv[4])

# The particles, their types, and the excluded pairs.
atom_types, excluded = [], set()
for name, copies in order:
    molecule = molecules[name]
    size = len(molecule["atoms"])
    neighbors = [set() for _ in range(size)]
    for i, j in molecule["bonds"]:
        neighbors[i].add(j)
        neighbors[j].add(i)
    for _ in range(copies):
        first = len(atom_types)
        atom_types += molecule["atoms"]
        for i in range(size):
            reached, frontier = {i}, {i}
            for _ in range(molecule["nrexcl"]):
                frontier = {k for f in frontier for k in neighbors[f]} - reached
                reached |= frontier
            for j in reached:
                if j > i:
                    excluded.add((first + i, first + j))


def pair_parameters(a, b):
    if frozenset((a, b)) in overrides:
        return overrides[frozenset((a, b))]
    (sa, ea), (sb, eb) = types[a], types[b]
    return 0.5 * (sa + sb), math.sqrt(ea * eb)


def coefficient(a):
    sigma, epsilon = pair_parameters(a, a)
    return 2.0 * math.sqrt(epsilon) * sigma ** 3


def kept(r):
    """1 − g(β r)."""
    y = (beta * r) ** 2
    return 1.0 - math.exp(-y) * (1.0 + y + 0.5 * y * y)


def lennard_jones(sigma, epsilon, r):
    s6 = (sigma / r) ** 6
    return 4.0 * epsilon * (s6 * s6 - s6)


c = [coefficient(t) for t in atom_types]
direct = excluded_energy = 0.0
for i in range(count):
    for j in range(i + 1, count):
        d = [x[i][k] - x[j][k] - box[k] * round((x[i][k] - x[j][k]) / box[k])
             for k in range(3)]
        r = math.sqrt(sum(v * v for v in d))
        cc = c[i] * c[j]
        if (i, j) in excluded:
            excluded_energy += cc * kept(r) / r ** 6
            continue
        if r < rc:
            sigma, epsilon = pair_parameters(atom_types[i], atom_types[j])
            direct += (lennard_jones(sigma, epsilon, r)
                       - lennard_jones(sigma, epsilon, rc)
                       + cc * (kept(r) / r ** 6 - kept(rc) / rc ** 6))

volume = box[0] * box[1] * box[2]
reach = [int(math.ceil(6.0 * beta / math.pi * box[k])) for k in range(3)]
reciprocal = 0.0
for a in range(-reach[0], reach[0] + 1):
    for b in range(-reach[1], reach[1] + 1):
        for e in range(-reach[2], reach[2] + 1):
            m = (a / box[0], b / box[1], e / box[2])
            size = math.sqrt(sum(v * v for v in m))
            u = math.pi * size / beta
            if u > 6.0:
                continue
            f = ((1.0 - 2.0 * u * u) * math.exp(-u * u)
                 + 2.0 * math.sqrt(math.pi) * u ** 3 * math.erfc(u)) / 3.0
            re = im = 0.0
            for i in range(count):
                phase = 2.0 * math.pi * sum(m[k] * x[i][k] for k in range(3))
                re += c[i] * math.cos(phase)
                im += c[i] * math.sin(phase)
            reciprocal += f * (re * re + im * im)
reciprocal *= -math.pi ** 1.5 * beta ** 3 / (2.0 * volume)
self_energy = beta ** 6 / 12.0 * sum(v * v for v in c)

print("direct       %.7f" % (direct / KCAL))
print("excluded     %.7f" % (excluded_energy / KCAL))
print("reciprocal   %.7f" % (reciprocal / KCAL))
print("self         %.7f" % (self_energy / KCAL))
