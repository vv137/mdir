"""The terms of tabulated-functions-nd.test (D165) computed on their own from
the control file, the topology, and the coordinates of the dipeptide: a
pair term in a function of two arguments and a discrete one, a term of the
positions in a function of three, and a dihedral term in a periodic
function of two. It fits the splines along the axes, takes the derivatives
at the points from them and the mixed ones from splines of those, builds
each patch from the cubics of Hermite, and prints the energy of each term
(kcal/mol); then the forces that the terms add on a few particles against
central differences of their energies (kcal/mol/Å).

    check_functions.py CONTROL TOPOLOGY COORDINATES PLAIN_FORCES FORCES TOLERANCE

PLAIN_FORCES and FORCES are `mdir checkpoint --print=forces` of the runs
without and with the terms (kJ/mol/nm)."""
import ast, math, sys

control, prmtop, inpcrd, plain, terms, tolerance = sys.argv[1:7]
tolerance = float(tolerance)

# The functions of the control file: each [[energy.function]] up to the
# next table, its keys as Python literals.
functions = {}
current = None
for line in open(control):
    line = line.strip()
    if line.startswith('['):
        current = {} if line == '[[energy.function]]' else None
        continue
    if current is None or '=' not in line:
        continue
    key, value = (part.strip() for part in line.split('=', 1))
    value = ast.literal_eval(value.replace('true', 'True').replace('false', 'False'))
    current[key] = value
    if key == 'name':
        functions[value] = current

def shape(values):
    sizes = []
    while isinstance(values, list):
        sizes.append(len(values))
        values = values[0]
    return sizes

def flat(values, sizes):
    """The values with the first index fastest."""
    out = [0.0] * math.prod(sizes)
    stride = [math.prod(sizes[:k]) for k in range(len(sizes))]
    def walk(v, depth, offset):
        if depth == len(sizes):
            out[offset] = float(v); return
        for i, w in enumerate(v):
            walk(w, depth + 1, offset + i * stride[depth])
    walk(values, 0, 0)
    return out

def curvatures(y, h, periodic):
    """Second derivatives of the natural or periodic cubic spline."""
    n = len(y)
    if not periodic:
        m = [0.0] * n
        if n > 2:
            # Thomas algorithm on m[i-1] + 4 m[i] + m[i+1] = 6 Δ²y / h².
            a = [1.0] * (n - 2); b = [4.0] * (n - 2); c = [1.0] * (n - 2)
            d = [6 * (y[i - 1] - 2 * y[i] + y[i + 1]) / h**2 for i in range(1, n - 1)]
            for i in range(1, n - 2):
                w = a[i] / b[i - 1]; b[i] -= w * c[i - 1]; d[i] -= w * d[i - 1]
            x = [0.0] * (n - 2)
            x[-1] = d[-1] / b[-1]
            for i in range(n - 4, -1, -1):
                x[i] = (d[i] - c[i] * x[i + 1]) / b[i]
            m[1:n - 1] = x
        return m
    # Periodic: the cyclic system of the n - 1 distinct points, by dense
    # Gaussian elimination.
    k = n - 1
    A = [[0.0] * k for _ in range(k)]; r = [0.0] * k
    for i in range(k):
        A[i][i] = 4.0; A[i][(i - 1) % k] += 1.0; A[i][(i + 1) % k] += 1.0
        r[i] = 6 * (y[(i - 1) % k] - 2 * y[i] + y[i + 1]) / h**2
    for col in range(k):
        piv = max(range(col, k), key=lambda q: abs(A[q][col]))
        A[col], A[piv] = A[piv], A[col]; r[col], r[piv] = r[piv], r[col]
        for row in range(col + 1, k):
            f = A[row][col] / A[col][col]
            for j in range(col, k):
                A[row][j] -= f * A[col][j]
            r[row] -= f * r[col]
    x = [0.0] * k
    for i in range(k - 1, -1, -1):
        x[i] = (r[i] - sum(A[i][j] * x[j] for j in range(i + 1, k))) / A[i][i]
    return x + [x[0]]

def slopes(y, h, periodic):
    m = curvatures(y, h, periodic); n = len(y)
    s = [(y[i + 1] - y[i]) / h - h * (2 * m[i] + m[i + 1]) / 6 for i in range(n - 1)]
    return s + [(y[n - 1] - y[n - 2]) / h + h * (m[n - 2] + 2 * m[n - 1]) / 6]

class Function:
    def __init__(self, spec):
        self.discrete = spec.get('discrete', False)
        self.sizes = shape(spec['values'])
        self.values = flat(spec['values'], self.sizes)
        self.dim = len(self.sizes)
        self.stride = [math.prod(self.sizes[:k]) for k in range(self.dim)]
        if self.discrete:
            return
        lo, hi = spec['min'], spec['max']
        self.lo = lo if isinstance(lo, list) else [lo]
        self.hi = hi if isinstance(hi, list) else [hi]
        self.periodic = spec.get('periodic', False)
        self.h = [(self.hi[k] - self.lo[k]) / (self.sizes[k] - 1) for k in range(self.dim)]
        # The derivatives along sets of axes, by a bit for each, in the
        # order of the continuous functions of OpenMM.
        self.d = {0: self.values}
        for mask, axis, source in ((1, 0, 0), (2, 1, 0), (4, 2, 0), (3, 0, 2), (5, 2, 1), (6, 1, 4), (7, 0, 6)):
            if mask < 2 ** self.dim:
                self.d[mask] = self.along(self.d[source], axis)

    def along(self, data, axis):
        out = [0.0] * len(data)
        st, n = self.stride[axis], self.sizes[axis]
        for start in range(len(data)):
            if (start // st) % n:
                continue
            line = [data[start + i * st] for i in range(n)]
            for i, v in enumerate(slopes(line, self.h[axis], self.periodic)):
                out[start + i * st] = v
        return out

    def __call__(self, *x):
        if self.discrete:
            idx = [min(max(int(math.floor(x[k] + 0.5)), 0), self.sizes[k] - 1) for k in range(self.dim)]
            return self.values[sum(i * s for i, s in zip(idx, self.stride))]
        cell, t = [], []
        for k in range(self.dim):
            v = x[k]
            if self.periodic:
                period = self.hi[k] - self.lo[k]
                v -= period * math.floor((v - self.lo[k]) / period)
            elif v < self.lo[k] or v > self.hi[k]:
                return 0.0
            u = (v - self.lo[k]) / self.h[k]
            c = min(max(int(math.floor(u)), 0), self.sizes[k] - 2)
            cell.append(c); t.append(u - c)
        # Hermite basis along each axis: p0, p1, h d0, h d1.
        def basis(s):
            return [1 - 3 * s * s + 2 * s**3, 3 * s * s - 2 * s**3, s - 2 * s * s + s**3, -s * s + s**3]
        B = [basis(s) for s in t]
        total = 0.0
        for q in range(4 ** self.dim):
            qs = [(q >> (2 * k)) & 3 for k in range(self.dim)]
            w = 1.0; mask = 0; point = 0
            for k, qk in enumerate(qs):
                w *= B[k][qk]
                if qk >= 2:
                    mask |= 1 << k; w *= self.h[k]
                point += (cell[k] + (qk & 1)) * self.stride[k]
            total += w * self.d[mask][point]
        return total

F = {name: Function(spec) for name, spec in functions.items()}

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

top = sections(prmtop)
n = int(top['POINTERS'][0])
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
lines = open(inpcrd).read().split('\n')
vals = [float(lines[2 + i // 6][12 * (i % 6):12 * (i % 6) + 12]) for i in range(3 * n)]
X = [vals[3 * i:3 * i + 3] for i in range(n)]
box = [float(v) for v in [l for l in lines if l.strip()][-1].split()[:3]]

def residue(i):
    return max(r for r in range(len(residues)) if residues[r] <= i) + 1

s = [-0.75 if names[i].startswith('O') else 0.25 for i in range(n)]
kind = [2 if residue(i) <= 3 else (1 if names[i].startswith('H') else 0) for i in range(n)]
solute = [i for i in range(n) if residue(i) <= 3]
dihedrals = [(1, 4, 6, 8), (4, 6, 8, 10), (6, 8, 10, 12)]

def vector(x, i, j):
    d = [x[i][c] - x[j][c] for c in range(3)]
    return [d[c] - box[c] * round(d[c] / box[c]) for c in range(3)]

def norm(d):
    return math.sqrt(sum(v * v for v in d))

def pair(x, i, j):
    if (min(i, j), max(i, j)) in excluded:
        return 0.0
    r = norm(vector(x, i, j))
    if r >= 9.0:
        return 0.0
    return 2.0 * F['surf'](r, s[i] + s[j]) + 1.5 * F['pairkind'](kind[i], kind[j]) * math.exp(-r)

def field(x, i):
    return 0.8 * F['vol'](*x[i])

def cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]

def dihedral(x, t):
    i, j, k, l = t
    b1, b2, b3 = vector(x, j, i), vector(x, k, j), vector(x, l, k)
    n1, n2 = cross(b1, b2), cross(b2, b3)
    # The angle of IUPAC, positive when the far bond turns clockwise seen
    # along the middle one: atan2(|b2| b1 . (b2 x b3), n1 . n2).
    y = norm(b2) * sum(p * q for p, q in zip(b1, n2))
    return math.atan2(y, sum(p * q for p, q in zip(n1, n2)))

def twist(x, t):
    theta = dihedral(x, t)
    return 1.2 * F['per2'](theta, 2 * theta + 0.3)

print('twist2 %.6f' % sum(twist(X, t) for t in dihedrals))
print('surface %.6f' % sum(pair(X, i, j) for i in range(n) for j in range(i + 1, n)))
print('field %.6f' % sum(field(X, i) for i in solute))

def local(x, a):
    e = sum(pair(x, a, j) for j in range(n) if j != a)
    e += sum(twist(x, t) for t in dihedrals if a in t)
    if a in solute:
        e += field(x, a)
    return e

h = 1e-5; worst = 0; ref = 0
Fm = {int(l.split()[0]): [float(v) for v in l.split()[2:5]] for l in open(terms)}
P = {int(l.split()[0]): [float(v) for v in l.split()[2:5]] for l in open(plain)}
for a in [1, 4, 6, 8, 10, 12, 21, 30, 61, 500]:
    for c in range(3):
        energies = []
        for step in (h, -h):
            x = [row[:] for row in X]; x[a][c] += step
            energies.append(local(x, a))
        force = -(energies[0] - energies[1]) / (2 * h)
        mine = (Fm[a][c] - P[a][c]) / 4.184 / 10   # kJ/mol/nm -> kcal/mol/Å
        worst = max(worst, abs(mine - force)); ref = max(ref, abs(force))
print('forces against differences of the energy: %s' %
      ('ok' if worst < tolerance * ref else 'FAILED %.2e of %.2e' % (worst, ref)))
