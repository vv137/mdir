"""The columns of `[output] observe` of observables.test at the first row,
computed from the topology and the coordinates of the dipeptide: the
energies of the terms and their derivatives in their constants, written
out by hand, against those that MDIR writes, within the six decimals of
the file and TOLERANCE relative to each value (1e-9 by default). The pair term, linear in
its constant `k`, is checked against its energy in the log, k dU/dk = U.

    check_observables.py TOPOLOGY COORDINATES OBSERVABLES LOG [TOLERANCE]
"""
import math, re, sys

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
labels = [v.strip() for v in top['RESIDUE_LABEL']]
lines = open(sys.argv[2]).read().split('\n')
vals = [float(lines[2 + i // 6][12 * (i % 6):12 * (i % 6) + 12]) for i in range(3 * n)]
X = [vals[3 * i:3 * i + 3] for i in range(n)]
waters = [i for r, label in enumerate(labels) if label == 'WAT'
          for i in range(starts[r], starts[r + 1])]

def distance(a, b):
    return math.sqrt(sum((a[c] - b[c]) ** 2 for c in range(3)))

def center(first, last):
    members = range(starts[first - 1], starts[last])
    m = sum(masses[i] for i in members)
    return [sum(masses[i] * X[i][c] for i in members) / m for c in range(3)]

below = [max(0.0, 6.0 - X[i][2]) for i in waters]
above = [max(0.0, X[i][2] - 20.0) for i in waters]
flat = [max(0.0, distance(X[a - 1], X[b - 1]) - 4.0) for a, b in [(2, 19), (5, 15)]]
pull_r = distance(center(1, 2), center(3, 3))
comp_d = distance(X[1], X[6])
expected = {
    'lower.energy': sum(0.5 * 10.0 * d * d for d in below),
    'lower.d_z0': sum(10.0 * d for d in below),
    'upper.d_z0': -sum(10.0 * d for d in above),
    'upper.d_k': sum(0.5 * d * d for d in above),
    'flat.energy': sum(10.0 * d * d for d in flat),
    'flat.d_r0': -sum(2 * 10.0 * d for d in flat),
    'pull.d_r0': -2 * 2.0 * (pull_r - 3.0),
    'upper.energy': sum(0.5 * 10.0 * d * d for d in above),
    'pull.energy': 2.0 * (pull_r - 3.0) ** 2,
    'comp.energy': 1.5 * (comp_d - 3.0) ** 2,
    'comp.d_k': (comp_d - 3.0) ** 2,
    'comp.d_m': 0.0,
}
log = open(sys.argv[4]).read()
soft = float(re.search(r'MDIR:\s+soft\s+(\S+)', log).group(1))
expected['soft.energy'] = soft
expected['soft.d_k'] = soft / 0.5

rows = [l.split() for l in open(sys.argv[3]) if not l.startswith('#')]
header = open(sys.argv[3]).readline().split()[1:]
first = dict(zip(header, (float(v) for v in rows[0])))
tolerance = float(sys.argv[5]) if len(sys.argv) > 5 else 1e-9
if sorted(header[2:]) != sorted(expected):
    print('columns: FAILED %s' % header[2:])
for name, value in expected.items():
    mine = first[name]
    # The file prints six decimals.
    ok = abs(mine - value) <= 5.1e-7 + tolerance * abs(value)
    print('%s %s' % (name, 'ok' if ok else 'FAILED %.12g against %.12g' % (mine, value)))
