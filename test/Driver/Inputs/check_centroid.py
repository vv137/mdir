"""The terms over the centers of groups of centroid-terms.test, computed
from the topology and the coordinates of the dipeptide: the centers in the
minimum image from the particle in the middle of each group, the energy of
each term, the forces that they add on every particle of a group against
central differences of the energies (kcal/mol/Å), and the diagonal of their
virial, sum of x ⊗ F over the particles in the images of the centers.

    check_centroid.py TOPOLOGY COORDINATES PLAIN_FORCES FORCES PLAIN_LOG LOG
    check_centroid.py --pull TOPOLOGY COORDINATES PULL

PLAIN_FORCES and FORCES are `mdir checkpoint --print=forces`, and PLAIN_LOG
and LOG the logs, of the runs without and with the terms. With --pull, the
first line of the file of `pull` (D145, D149): the coordinates of
each term, its energy, and its force, against central differences: on the
second center of the bond, and along the angle of the others."""
import math, re, sys

pull = sys.argv[1] == '--pull'
if pull:
    del sys.argv[1]

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
masses = [float(v) for v in top['MASS']]
starts = [int(v) - 1 for v in top['RESIDUE_POINTER']] + [n]
lines = open(sys.argv[2]).read().split('\n')
vals = [float(lines[2 + i // 6][12 * (i % 6):12 * (i % 6) + 12]) for i in range(3 * n)]
X = [vals[3 * i:3 * i + 3] for i in range(n)]
box = [float(v) for v in [l for l in lines if l.strip()][-1].split()[:3]]

def residues(first, last):
    return list(range(starts[first - 1], starts[last]))

def sub(a, b): return [a[k] - b[k] for k in range(3)]
def add(a, b): return [a[k] + b[k] for k in range(3)]
def dot(a, b): return sum(a[k] * b[k] for k in range(3))
def cross(a, b): return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
def norm(a): return math.sqrt(dot(a, a))
def image(d): return [d[k] - box[k] * round(d[k] / box[k]) for k in range(3)]

# The terms of the test: groups of residues, whether mass weighted, and the
# energy of the coordinate.
terms = [
    ('pull', [residues(1, 3), residues(4, 4)], True),
    ('bend', [residues(1, 1), residues(2, 2), residues(3, 3)], False),
    ('twist', [residues(1, 1), residues(2, 2), residues(3, 3), residues(5, 5)], True),
]

def centers(x, groups, weighted):
    """The centers, in the images from the reference of the first group, and
    the image of every member in which it enters its center."""
    out = []; images = {}
    first = groups[0][(len(groups[0]) - 1) // 2]
    for group in groups:
        ref = group[(len(group) - 1) // 2]
        origin = add(x[first], image(sub(x[ref], x[first])))
        w = [masses[i] if weighted else 1.0 for i in group]; total = sum(w)
        c = [0.0, 0.0, 0.0]
        for i, wi in zip(group, w):
            xi = add(origin, image(sub(x[i], x[ref])))
            images.setdefault(i, xi)
            c = add(c, [wi / total * v for v in xi])
        out.append(c)
    return out, images

def coordinates(name, c):
    if name == 'pull':
        d = sub(c[1], c[0])
        return [norm(d)] + d
    if name == 'bend':
        u = sub(c[0], c[1]); v = sub(c[2], c[1])
        return [math.atan2(norm(cross(u, v)), dot(u, v))]
    b0 = sub(c[0], c[1]); b1 = sub(c[2], c[1]); b2 = sub(c[3], c[2])
    a = [v / norm(b1) for v in b1]
    v = sub(b0, [dot(b0, a) * k for k in a]); w = sub(b2, [dot(b2, a) * k for k in a])
    return [math.atan2(dot(cross(a, v), w), dot(v, w))]

def value(name, q):
    if name == 'pull':
        return 2.0 * (q[0] - 6.0) ** 2 + 0.3 * q[3]
    if name == 'bend':
        return 5.0 * (q[0] - 2.0) ** 2
    return 1.0 * (1 + math.cos(q[0] - 0.5))

def energy(name, c):
    return value(name, coordinates(name, c))

if pull:
    got = [float(v) for v in [l for l in open(sys.argv[3]) if not l.startswith('#')][0].split()[2:]]
    want = []; h = 1e-5
    for name, groups, weighted in terms:
        c = centers(X, groups, weighted)[0]
        q = coordinates(name, c)
        want += q + [value(name, q)]
        if name == 'pull':
            force = []
            for k in range(3):
                e = []
                for step in (h, -h):
                    moved = [p[:] for p in c]; moved[1][k] += step
                    e.append(energy(name, moved))
                force.append(-(e[0] - e[1]) / (2 * h))
            want += [dot(force, q[1:]) / q[0]] + force
        else:
            want += [-(value(name, [q[0] + h]) - value(name, [q[0] - h])) / (2 * h)]
    worst = max(abs(a - b) for a, b in zip(got, want))
    print('%d columns: %s' % (len(got), 'ok' if len(got) == len(want) and worst < 2e-6 else 'FAILED %.2e %s %s' % (worst, got, want)))
    sys.exit()


for name, groups, weighted in terms:
    print('%s %.6f' % (name, energy(name, centers(X, groups, weighted)[0])))

# The forces on every particle of a group, term by term, and the virial
# from them, each particle in the image in which the term takes it.
F = {int(l.split()[0]): [float(v) for v in l.split()[2:5]] for l in open(sys.argv[4])}
P = {int(l.split()[0]): [float(v) for v in l.split()[2:5]] for l in open(sys.argv[3])}
h = 1e-5; forces = {}; virial = [0.0, 0.0, 0.0]
for name, groups, weighted in terms:
    images = centers(X, groups, weighted)[1]
    for i, xi in images.items():
        for c in range(3):
            e = []
            for step in (h, -h):
                x = [row[:] for row in X]; x[i][c] += step
                e.append(energy(name, centers(x, groups, weighted)[0]))
            force = -(e[0] - e[1]) / (2 * h)
            forces.setdefault(i, [0.0, 0.0, 0.0])[c] += force
            virial[c] += xi[c] * force
worst = 0; ref = 0
for i, force in forces.items():
    for c in range(3):
        mine = (F[i][c] - P[i][c]) / 4.184 / 10   # kJ/mol/nm -> kcal/mol/Å
        worst = max(worst, abs(mine - force[c])); ref = max(ref, abs(force[c]))
print("forces against differences of the energy: %s" % ("ok" if worst < 1e-6 * ref else "FAILED %.2e" % worst))

def logged(path):
    text = open(path).read()
    return [float(v) for v in re.search(r'virial at the start.*\n.*?MDIR:\s+(\S+)\s+(\S+)\s+(\S+)', text).groups()]
mdir = [a - b for a, b in zip(logged(sys.argv[6]), logged(sys.argv[5]))]
ok = all(abs(m - v) < 1e-5 * max(1.0, max(abs(u) for u in virial)) for m, v in zip(mdir, virial))
print("virial: %s" % ("ok" if ok else "FAILED %s against %s" % (mdir, virial)))
