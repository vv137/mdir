"""The pair terms of pair-terms.test, computed from the topology and the
coordinates of the dipeptide: the energy of each, and the forces that they
add on a few particles, against central differences of the energies of the
pairs of those particles (kcal/mol/Å).

    check_pair_terms.py TOPOLOGY COORDINATES PLAIN_FORCES FORCES

PLAIN_FORCES and FORCES are `mdir checkpoint --print=forces` of the runs
without and with the terms."""
import math, sys

def sections(path):
    out = {}; name = None; fmt = None
    for line in open(path):
        if line.startswith('%FLAG'):
            name = line.split()[1]; out[name] = []
        elif line.startswith('%FORMAT'):
            fmt = line[line.index('(') + 1:line.index(')')]
            width = int(fmt.split('a' if 'a' in fmt else ('I' if 'I' in fmt else 'E'))[1].split('.')[0])
            out[name + '#w'] = width
        elif name and not line.startswith('%'):
            w = out[name + '#w']; s = line.rstrip('\n')
            out[name] += [s[k:k + w] for k in range(0, len(s), w) if s[k:k + w].strip()]
    return out

top = sections(sys.argv[1])
n = int(top['POINTERS'][0]); ntypes = int(top['POINTERS'][1])
q = [float(v) / 18.2223 for v in top['CHARGE']]
t = [int(v) - 1 for v in top['ATOM_TYPE_INDEX']]
index = [int(v) for v in top['NONBONDED_PARM_INDEX']]
A = [float(v) for v in top['LENNARD_JONES_ACOEF']]
B = [float(v) for v in top['LENNARD_JONES_BCOEF']]
residues = [int(v) - 1 for v in top['RESIDUE_POINTER']]
counts = [int(v) for v in top['NUMBER_EXCLUDED_ATOMS']]
listed = [int(v) - 1 for v in top['EXCLUDED_ATOMS_LIST']]
excluded = set(); k = 0
for i in range(n):
    for j in listed[k:k + counts[i]]:
        if j >= 0:
            excluded.add((min(i, j), max(i, j)))
    k += counts[i]

def lj(a, b):
    e = index[ntypes * t[a] + t[b]] - 1
    if A[e] == 0.0 and B[e] == 0.0:
        return 0.0, 0.0
    return (A[e] / B[e]) ** (1 / 6), B[e] * B[e] / (4 * A[e])

lines = open(sys.argv[2]).read().split('\n')
vals = [float(lines[2 + i // 6][12 * (i % 6):12 * (i % 6) + 12]) for i in range(3 * n)]
X = [vals[3 * i:3 * i + 3] for i in range(n)]
box = [float(v) for v in [l for l in lines if l.strip()][-1].split()[:3]]
cutoff = 9.0
coulomb = 332.06371329919205

def residue(i):
    return max(r for r in range(len(residues)) if residues[r] <= i) + 1
# groups = [":1-3", ":1-20"]
first = [residue(i) <= 3 for i in range(n)]
second = [residue(i) <= 20 for i in range(n)]

def distance(x, i, j):
    d = [x[i][c] - x[j][c] for c in range(3)]
    d = [d[c] - box[c] * round(d[c] / box[c]) for c in range(3)]
    return math.sqrt(sum(v * v for v in d))

def copy(x, i, j, r):
    s, e = lj(i, j)
    return 4 * e * ((s / r) ** 12 - (s / r) ** 6)

def grouped(x, i, j, r):
    if not ((first[i] and second[j]) or (first[j] and second[i])):
        return 0.0
    s1, e1 = lj(i, i); s2, e2 = lj(j, j); s, e = lj(i, j)
    s12 = 0.5 * (s1 + s2); e12 = math.sqrt(e1 * e2)
    return (0.5 * coulomb * q[i] * q[j] / r +
            4 * e12 * ((s12 / r) ** 12 - (s12 / r) ** 6) + 0.3 * e * (s / r) ** 6)

def pair(x, i, j, terms):
    if (min(i, j), max(i, j)) in excluded:
        return 0.0
    r = distance(x, i, j)
    if r >= cutoff:
        return 0.0
    return sum(f(x, i, j, r) for f in terms)

members = [i for i in range(n) if first[i] or second[i]]
print('copy %.6f' % sum(pair(X, i, j, [copy]) for i in range(n) for j in range(i + 1, n)))
print('grouped %.6f' % sum(pair(X, i, j, [grouped]) for a, i in enumerate(members)
                           for j in members[a + 1:]))

# The forces on a few particles: in the dipeptide, in water of the second
# group, and in water of neither.
h = 1e-5; worst = 0; ref = 0
F = {int(l.split()[0]): [float(v) for v in l.split()[2:5]] for l in open(sys.argv[4])}
P = {int(l.split()[0]): [float(v) for v in l.split()[2:5]] for l in open(sys.argv[3])}
for a in [0, 4, 8, 14, 21, 22, 40, 61, 100, 500]:
    for c in range(3):
        energies = []
        for step in (h, -h):
            x = [row[:] for row in X]; x[a][c] += step
            energies.append(sum(pair(x, a, j, [copy, grouped]) for j in range(n) if j != a))
        force = -(energies[0] - energies[1]) / (2 * h)
        mine = (F[a][c] - P[a][c]) / 4.184 / 10   # kJ/mol/nm -> kcal/mol/Å
        worst = max(worst, abs(mine - force)); ref = max(ref, abs(force))
print("forces against differences of the energy: %s" % ("ok" if worst < 1e-6 * ref else "FAILED %.2e" % worst))
