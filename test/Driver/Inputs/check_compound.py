"""The compound terms of compound-terms.test (D165), computed from the
topology and the coordinates of the dipeptide: the energy of each term
(kcal/mol), and the forces that they add on their particles against
central differences of their energies (kcal/mol/Å).

    check_compound.py TOPOLOGY COORDINATES PLAIN_FORCES FORCES TOLERANCE

PLAIN_FORCES and FORCES are `mdir checkpoint --print=forces` of the runs
without and with the terms (kJ/mol/nm)."""
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
names = [v.strip() for v in top['ATOM_NAME']]
lines = open(sys.argv[2]).read().split('\n')
vals = [float(lines[2 + i // 6][12 * (i % 6):12 * (i % 6) + 12]) for i in range(3 * n)]
X = [vals[3 * i:3 * i + 3] for i in range(n)]
box = [float(v) for v in [l for l in lines if l.strip()][-1].split()[:3]]
w = [1.5 if names[i][0] in 'NO' else 1.0 for i in range(n)]

def vector(x, i, j):
    d = [x[i][c] - x[j][c] for c in range(3)]
    return [d[c] - box[c] * round(d[c] / box[c]) for c in range(3)]

def dot(a, b):
    return sum(p * q for p, q in zip(a, b))

def norm(a):
    return math.sqrt(dot(a, a))

def cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]

def distance(x, i, j):
    return norm(vector(x, i, j))

def angle(x, i, j, k):
    u, v = vector(x, i, j), vector(x, k, j)
    return math.acos(max(-1.0, min(1.0, dot(u, v) / (norm(u) * norm(v)))))

def dihedral(x, i, j, k, l):
    b1, b2, b3 = vector(x, j, i), vector(x, k, j), vector(x, l, k)
    n1, n2 = cross(b1, b2), cross(b2, b3)
    return math.atan2(norm(b2) * dot(b1, n2), dot(n1, n2))

hbonds = [((4, 6, 8, 10), 1.5), ((8, 10, 14, 16), 2.0), ((1, 4, 6, 21), 0.5)]
fives = [(1, 4, 6, 8, 10), (6, 8, 10, 14, 16)]

def hbond(x, t, k):
    p1, p2, p3, p4 = t
    d = distance(x, p1, p3); a = angle(x, p1, p2, p3); phi = dihedral(x, p1, p2, p3, p4)
    return (k * (d - 2.9) ** 2 + 0.8 * w[p1] * w[p4] * (math.cos(a) - math.cos(2.6)) ** 2
            + 0.3 * (1 + math.cos(2 * phi - 0.4)))

def five(x, t):
    p1, p2, p3, p4, p5 = t
    return (0.05 * w[p2] * distance(x, p1, p5) * math.sin(dihedral(x, p2, p3, p4, p5))
            + 0.7 * (angle(x, p5, p3, p1) - 1.9) ** 2)

def total(x):
    return sum(hbond(x, t, k) for t, k in hbonds), sum(five(x, t) for t in fives)

e1, e2 = total(X)
print('hbond %.6f' % e1)
print('five %.6f' % e2)

h = 1e-5; worst = 0; ref = 0
F = {int(l.split()[0]): [float(v) for v in l.split()[2:5]] for l in open(sys.argv[4])}
P = {int(l.split()[0]): [float(v) for v in l.split()[2:5]] for l in open(sys.argv[3])}
members = sorted({i for t, k in hbonds for i in t} | {i for t in fives for i in t})
for a in members:
    for c in range(3):
        energies = []
        for step in (h, -h):
            x = [row[:] for row in X]; x[a][c] += step
            energies.append(sum(total(x)))
        force = -(energies[0] - energies[1]) / (2 * h)
        mine = (F[a][c] - P[a][c]) / 4.184 / 10   # kJ/mol/nm -> kcal/mol/Å
        worst = max(worst, abs(mine - force)); ref = max(ref, abs(force))
tolerance = float(sys.argv[5])
print('forces against differences of the energy: %s' %
      ('ok' if worst < tolerance * ref else 'FAILED %.2e of %.2e' % (worst, ref)))
