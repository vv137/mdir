"""The Coulomb terms of reaction-field.test, computed from the topology and
the coordinates of the dipeptide: the pairs within the cutoff that are not
excluded, f q_i q_j (1/r + k r² − c); the excluded pairs within the cutoff,
f q_i q_j (k r² − c); and the self term, −c f Σ q² / 2 (kcal/mol), with
the forces that they add on a few particles against central differences of
those energies (kcal/mol/Å).

    check_reaction_field.py TOPOLOGY COORDINATES EPSILON PLAIN_FORCES FORCES

PLAIN_FORCES and FORCES are `mdir checkpoint --print=forces` of the runs
with a plain cutoff and with the reaction field."""
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

f = 332.06371329919205
rc = 9.0
eps = float(sys.argv[3])
krf = 1 / (2 * rc ** 3) if eps == 0 else (eps - 1) / ((2 * eps + 1) * rc ** 3)
crf = 1 / rc + krf * rc ** 2

def distance(x, i, j):
    d = [x[i][c] - x[j][c] for c in range(3)]
    d = [d[c] - box[c] * round(d[c] / box[c]) for c in range(3)]
    return math.sqrt(sum(v * v for v in d))

def pair(x, i, j):
    r = distance(x, i, j)
    if r >= rc:
        return 0.0, 0.0
    if (min(i, j), max(i, j)) in excluded:
        return 0.0, f * q[i] * q[j] * (krf * r * r - crf)
    return f * q[i] * q[j] * (1 / r + krf * r * r - crf), 0.0

near = far = 0.0
for i in range(n):
    for j in range(i + 1, n):
        a, b = pair(X, i, j); near += a; far += b
print('Coulomb %.6f' % near)
print('Coulomb excluded %.6f' % far)
print('Coulomb self %.6f' % (-0.5 * crf * f * sum(v * v for v in q)))

def plain(x, i, j):
    if (min(i, j), max(i, j)) in excluded:
        return 0.0
    r = distance(x, i, j)
    return f * q[i] * q[j] / r if r < rc else 0.0

# The forces that the field adds: those of the run with it less those of
# the run with a plain cutoff.
h = 1e-5; worst = 0; ref = 0
F = {int(l.split()[0]): [float(v) for v in l.split()[2:5]] for l in open(sys.argv[5])}
P = {int(l.split()[0]): [float(v) for v in l.split()[2:5]] for l in open(sys.argv[4])}
for a in [0, 4, 8, 14, 21, 22, 40, 100, 500]:
    for c in range(3):
        e = []
        for step in (h, -h):
            x = [row[:] for row in X]; x[a][c] += step
            e.append(sum(sum(pair(x, a, j)) - plain(x, a, j) for j in range(n) if j != a))
        force = -(e[0] - e[1]) / (2 * h)
        mine = (F[a][c] - P[a][c]) / 4.184 / 10   # kJ/mol/nm -> kcal/mol/Å
        worst = max(worst, abs(mine - force)); ref = max(ref, abs(force))
print("forces against differences of the energy: %s" % ("ok" if worst < 1e-6 * ref else "FAILED %.2e" % worst))
