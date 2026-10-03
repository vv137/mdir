"""The terms of particle-parameters.test with parameters of each particle
(D165), computed from the topology and the coordinates of the dipeptide:
the energy of each term (kcal/mol), and the forces that the four add on a
few particles against central differences of their energies (kcal/mol/Å).

    check_particle_parameters.py TOPOLOGY COORDINATES PLAIN_FORCES FORCES TOLERANCE

PLAIN_FORCES and FORCES are `mdir checkpoint --print=forces` of the runs
without and with the terms (kJ/mol/nm). The forces must agree within
TOLERANCE of the largest of them."""
import math, sys

def sections(path):
    out = {}; name = None
    for line in open(path):
        if line.startswith('%FLAG'):
            name = line.split()[1]; out[name] = []
        elif line.startswith('%FORMAT'):
            fmt = line[line.index('(') + 1:line.index(')')]
            out[name + '#w'] = int(fmt.split('a' if 'a' in fmt else ('I' if 'I' in fmt else 'E'))[1].split('.')[0])
        elif name and not line.startswith('%'):
            w = out[name + '#w']; s = line.rstrip('\n')
            out[name] += [s[k:k + w] for k in range(0, len(s), w) if s[k:k + w].strip()]
    return out

top = sections(sys.argv[1])
n = int(top['POINTERS'][0])
q = [float(v) / 18.2223 for v in top['CHARGE']]
names = [v.strip() for v in top['ATOM_NAME']]
residues = [int(v) - 1 for v in top['RESIDUE_POINTER']]
counts = [int(v) for v in top['NUMBER_EXCLUDED_ATOMS']]
listed = [int(v) - 1 for v in top['EXCLUDED_ATOMS_LIST']]
excluded = set(); k = 0
for i in range(n):
    for j in listed[k:k + counts[i]]:
        if j >= 0:
            excluded.add((min(i, j), max(i, j)))
    k += counts[i]

lines = open(sys.argv[2]).read().split('\n')
vals = [float(lines[2 + i // 6][12 * (i % 6):12 * (i % 6) + 12]) for i in range(3 * n)]
X = [vals[3 * i:3 * i + 3] for i in range(n)]
box = [float(v) for v in [l for l in lines if l.strip()][-1].split()[:3]]
cutoff = 9.0

def residue(i):
    return max(r for r in range(len(residues)) if residues[r] <= i) + 1

# The parameters, entry by entry as the control file gives them.
w = [1.0] * n
for i in range(n):
    if residue(i) <= 3:
        w[i] = 0.5
for p in (5, 6, 30):
    w[p - 1] = 2.0
s = [-0.75 if names[i].startswith('O') else 0.25 for i in range(n)]

def vector(x, i, j):
    d = [x[i][c] - x[j][c] for c in range(3)]
    return [d[c] - box[c] * round(d[c] / box[c]) for c in range(3)]

def norm(d):
    return math.sqrt(sum(v * v for v in d))

def pair(x, i, j):
    if (min(i, j), max(i, j)) in excluded:
        return 0.0
    r = norm(vector(x, i, j))
    if r >= cutoff:
        return 0.0
    return 2.0 * w[i] * w[j] * math.exp(-r / 1.5) + 0.3 * (s[i] + s[j]) * q[i] * q[j] / r

bonds = [(0, 4), (1, 6), (29, 30)]
angles = [(0, 4, 5), (6, 8, 10)]
selected = [i for i in range(n) if residue(i) <= 3]

def bond(x, b):
    i, j = b
    return 3.0 * w[i] * w[j] * (norm(vector(x, i, j)) - 1.2) ** 2 + s[i] * s[j]

def angle(x, a):
    i, j, k = a
    u, v = vector(x, i, j), vector(x, k, j)
    theta = math.acos(sum(p * r for p, r in zip(u, v)) / (norm(u) * norm(v)))
    return 1.5 * (w[i] + 2 * w[j] + 3 * w[k]) * (theta - 2) ** 2

def external(x, i):
    return 0.2 * w[i] * (x[i][0] - 10) ** 2 + s[i] * x[i][2]

print('wbond %.6f' % sum(bond(X, b) for b in bonds))
print('wangle %.6f' % sum(angle(X, a) for a in angles))
print('screened %.6f' % sum(pair(X, i, j) for i in range(n) for j in range(i + 1, n)))
print('wext %.6f' % sum(external(X, i) for i in selected))

def local(x, a):
    """The energy of the terms that particle a takes part in."""
    e = sum(pair(x, a, j) for j in range(n) if j != a)
    e += sum(bond(x, b) for b in bonds if a in b)
    e += sum(angle(x, t) for t in angles if a in t)
    if a in selected:
        e += external(x, a)
    return e

h = 1e-5; worst = 0; ref = 0
F = {int(l.split()[0]): [float(v) for v in l.split()[2:5]] for l in open(sys.argv[4])}
P = {int(l.split()[0]): [float(v) for v in l.split()[2:5]] for l in open(sys.argv[3])}
for a in [0, 4, 5, 6, 8, 10, 21, 29, 30, 61, 500]:
    for c in range(3):
        energies = []
        for step in (h, -h):
            x = [row[:] for row in X]; x[a][c] += step
            energies.append(local(x, a))
        force = -(energies[0] - energies[1]) / (2 * h)
        mine = (F[a][c] - P[a][c]) / 4.184 / 10   # kJ/mol/nm -> kcal/mol/Å
        worst = max(worst, abs(mine - force)); ref = max(ref, abs(force))
tolerance = float(sys.argv[5])
print('forces against differences of the energy: %s' %
      ('ok' if worst < tolerance * ref else 'FAILED %.2e of %.2e' % (worst, ref)))
