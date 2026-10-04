"""The tabulated functions of tabulated-functions.test on the dipeptide:
a cubic spline through the values of each, natural or periodic, solved here
on its own,
the energies of the pair term and the dihedral term that call them, and the
forces that they add on a few particles, against central differences of
those energies (kcal/mol/Å).

    check_tabulated.py TOPOLOGY COORDINATES PLAIN_FORCES FORCES [TOLERANCE]

PLAIN_FORCES and FORCES are `mdir checkpoint --print=forces` of the runs
without and with the terms."""
import math, sys

def solve(A, d):
    """Gaussian elimination with partial pivoting."""
    n = len(d); A = [row[:] + [d[i]] for i, row in enumerate(A)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(A[r][c])); A[c], A[p] = A[p], A[c]
        for r in range(c + 1, n):
            f = A[r][c] / A[c][c]
            A[r] = [A[r][k] - f * A[c][k] for k in range(n + 1)]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        x[r] = (A[r][n] - sum(A[r][k] * x[k] for k in range(r + 1, n))) / A[r][r]
    return x

def spline(values, lo, hi, periodic=False):
    """The cubic spline through `values` at evenly spaced points from lo to
    hi: its second derivatives m from h/6 m[i-1] + 2h/3 m[i] + h/6 m[i+1] =
    (y[i+1] - y[i])/h - (y[i] - y[i-1])/h, solved as a dense system; natural
    (m = 0 at both ends, zero outside) or periodic (the indices modulo the
    number of intervals, the argument modulo hi - lo)."""
    n = len(values); h = (hi - lo) / (n - 1)
    N = n - 1 if periodic else n
    A = [[0.0] * N for _ in range(N)]; d = [0.0] * N
    for i in range(N):
        if not periodic and (i == 0 or i == n - 1):
            A[i][i] = 1.0
            continue
        A[i][(i - 1) % N] += h / 6; A[i][i] += 2 * h / 3; A[i][(i + 1) % N] += h / 6
        d[i] = (values[i + 1] - 2 * values[i] + values[(i - 1) % N]) / h
    m = solve(A, d)
    if periodic:
        m.append(m[0])
    def f(x):
        if periodic:
            x = lo + (x - lo) % (hi - lo)
        elif x < lo or x > hi:
            return 0.0
        i = min(int((x - lo) / h), n - 2)
        x0 = lo + i * h; x1 = x0 + h
        A = (x1 - x) / h; B = (x - x0) / h
        return (A * values[i] + B * values[i + 1] +
                ((A ** 3 - A) * m[i] + (B ** 3 - B) * m[i + 1]) * h * h / 6)
    return f

def table(f, n, lo, hi):
    return [float('%.6f' % f(lo + (hi - lo) * i / (n - 1))) for i in range(n)]

wave = spline(table(lambda x: math.exp(-x / 3) * math.cos(1.3 * x), 26, 1.5, 7.0), 1.5, 7.0)
torsion = spline(table(lambda x: 1 + math.cos(2 * x - 0.3), 37, -math.pi, math.pi), -math.pi, math.pi, periodic=True)

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

def residue(i):
    return max(r for r in range(len(residues)) if residues[r] <= i) + 1
# groups = [":1-3", ":1-20"]
first = [residue(i) <= 3 for i in range(n)]
second = [residue(i) <= 20 for i in range(n)]
members = [i for i in range(n) if first[i] or second[i]]

def sub(a, b): return [a[k] - b[k] for k in range(3)]
def dot(a, b): return sum(a[k] * b[k] for k in range(3))
def cross(a, b): return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
def dihedral(x, i, j, k, l):
    b0 = sub(x[i], x[j]); b1 = sub(x[k], x[j]); b2 = sub(x[l], x[k])
    n1 = math.sqrt(dot(b1, b1)); b1n = [c / n1 for c in b1]
    v = [b0[c] - dot(b0, b1n) * b1n[c] for c in range(3)]
    w = [b2[c] - dot(b2, b1n) * b1n[c] for c in range(3)]
    return math.atan2(dot(cross(b1n, v), w), dot(v, w))

def pair(x, i, j):
    if (min(i, j), max(i, j)) in excluded:
        return 0.0
    if not ((first[i] and second[j]) or (first[j] and second[i])):
        return 0.0
    d = sub(x[i], x[j]); d = [d[c] - box[c] * round(d[c] / box[c]) for c in range(3)]
    r = math.sqrt(dot(d, d))
    return 2.0 * wave(r) if r < 9.0 else 0.0

tuples = [(1, 4, 6, 8), (4, 6, 8, 10)]
def twist(x):
    return sum(1.5 * torsion(dihedral(x, *t)) for t in tuples)

print('tabulated %.6f' % sum(pair(X, i, j) for a, i in enumerate(members) for j in members[a + 1:]))
print('twist %.6f' % twist(X))

h = 1e-5; worst = 0; ref = 0
F = {int(l.split()[0]): [float(v) for v in l.split()[2:5]] for l in open(sys.argv[4])}
P = {int(l.split()[0]): [float(v) for v in l.split()[2:5]] for l in open(sys.argv[3])}
for a in [1, 4, 6, 8, 10, 14, 22, 40, 61]:
    for c in range(3):
        e = []
        for step in (h, -h):
            x = [row[:] for row in X]; x[a][c] += step
            e.append(sum(pair(x, a, j) for j in members if j != a) + twist(x))
        force = -(e[0] - e[1]) / (2 * h)
        mine = (F[a][c] - P[a][c]) / 4.184 / 10   # kJ/mol/nm -> kcal/mol/Å
        worst = max(worst, abs(mine - force)); ref = max(ref, abs(force))
tolerance = float(sys.argv[5]) if len(sys.argv) > 5 else 1e-6
print("forces against differences of the energy: %s" % ("ok" if worst < tolerance * ref else "FAILED %.2e" % worst))
print("maximum force difference %.9g; reference scale %.9g; tolerance %.9g kcal/mol/Angstrom" % (worst, ref, tolerance * ref))
