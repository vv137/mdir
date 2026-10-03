"""The terms of the absolute positions of external-terms.test, computed
from the topology and the coordinates of the dipeptide: the energy of each
term and the forces that the three add on every particle, against those
that MDIR adds, the difference of the forces of runs with and without them
(kcal/mol/Å), within TOLERANCE of the largest of them (1e-9 by default;
in mixed precision the kernel takes the positions in f32).

    check_external.py TOPOLOGY COORDINATES PLAIN_FORCES FORCES [TOLERANCE]
        [PLAIN_LOG LOG]

With the logs of the runs, the diagonal of the virial that the terms add,
fixed in space (D154): -dU/de_a when the positions scale by 1 + e along
axis a about the origin, the sum of x_a F_a (kcal/mol)."""
import sys

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
charges = [float(v) / 18.2223 for v in top['CHARGE']]
starts = [int(v) - 1 for v in top['RESIDUE_POINTER']] + [n]
labels = [v.strip() for v in top['RESIDUE_LABEL']]
lines = open(sys.argv[2]).read().split('\n')
vals = [float(lines[2 + i // 6][12 * (i % 6):12 * (i % 6) + 12]) for i in range(3 * n)]
X = [vals[3 * i:3 * i + 3] for i in range(n)]

def residues(first, last):
    return list(range(starts[first - 1], starts[last]))

waters = [i for r, label in enumerate(labels) if label == 'WAT'
          for i in range(starts[r], starts[r + 1])]

# Each term: its particles, and for each its energy and its gradient.
def wall(i):
    d = max(0.0, X[i][2] - 13.0)
    return 2.0 * d * d, [0.0, 0.0, 4.0 * d]

def field(i):
    return -charges[i] * 0.5 * X[i][0], [-charges[i] * 0.5, 0.0, 0.0]

well_k = {0: 1.0, 4: 2.0, 8: 3.0}
well_x = {0: 11.0, 4: 13.5, 8: 14.0}
well_y = {0: 11.5, 4: 12.0, 8: 15.0}
def well(i):
    k, dx, dy = well_k[i], X[i][0] - well_x[i], X[i][1] - well_y[i]
    return k * (dx * dx + dy * dy), [2 * k * dx, 2 * k * dy, 0.0]

terms = [('wall', residues(1, 3), wall), ('field', waters, field),
         ('well', [0, 4, 8], well)]
forces = [[0.0, 0.0, 0.0] for _ in range(n)]
for name, particles, function in terms:
    total = 0.0
    for i in particles:
        e, g = function(i)
        total += e
        for c in range(3):
            forces[i][c] -= g[c]
    print('%s %.6f' % (name, total))

F = {int(l.split()[0]): [float(v) for v in l.split()[2:5]] for l in open(sys.argv[4])}
P = {int(l.split()[0]): [float(v) for v in l.split()[2:5]] for l in open(sys.argv[3])}
worst = 0.0; largest = 0.0
for i in range(n):
    for c in range(3):
        mine = (F[i][c] - P[i][c]) / 4.184 / 10   # kJ/mol/nm -> kcal/mol/Å
        worst = max(worst, abs(mine - forces[i][c]))
        largest = max(largest, abs(forces[i][c]))
tolerance = float(sys.argv[5]) if len(sys.argv) > 5 else 1e-9
print('forces: %s' % ('ok' if worst < tolerance * max(1.0, largest) else 'FAILED %.2e' % worst))
if len(sys.argv) > 7:
    import re
    def logged(path):
        text = open(path).read()
        return [float(v) for v in re.search(r'virial at the start.*\n.*?MDIR:\s+(\S+)\s+(\S+)\s+(\S+)', text).groups()]
    mdir = [a - b for a, b in zip(logged(sys.argv[7]), logged(sys.argv[6]))]
    virial = [sum(X[i][c] * forces[i][c] for i in range(n)) for c in range(3)]
    ok = all(abs(m - v) < 2e-6 * max(1.0, max(abs(u) for u in virial)) for m, v in zip(mdir, virial))
    print('virial: %s' % ('ok' if ok else 'FAILED %s against %s' % (mdir, virial)))
