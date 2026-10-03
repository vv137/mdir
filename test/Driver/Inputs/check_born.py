"""Generalized Born of generalized-born.test, computed from the topology and
the coordinates of the peptide: the Born radii of OBC II from the integral
of the descreening over all pairs [Hawkins1996, Onufriev2004], the energy
-tau f (sum over pairs of q q / f_GB + sum of q^2 / 2B), the nonpolar term
4 pi gamma (rho + 0.14 nm)^2 (rho / B)^6, and the forces that they add on a
few atoms against central differences of the energy (kJ/mol/nm).

    check_born.py TOPOLOGY COORDINATES PLAIN_FORCES FORCES [--model M]
        [--salt C --temperature T] [--radius-cutoff R] [--mbondi2]

PLAIN_FORCES and FORCES are `mdir checkpoint --print=forces` of the runs
without and with generalized Born. The options follow D152: the model HCT,
OBC1, or OBC2 (the default); a salt of C mol/L at T K, which screens the
solvent with exp(-kappa f_GB), kappa scaled by 0.73; the integral of the
descreening cut at R Å; and the radii and screening of mbondi2 by element
in place of those of the topology."""
import argparse, math, re, sys

parser = argparse.ArgumentParser()
for name in ('topology', 'coordinates', 'plain', 'forces'):
    parser.add_argument(name)
parser.add_argument('--model', default='OBC2')
parser.add_argument('--salt', type=float, default=0.0)
parser.add_argument('--temperature', type=float, default=300.0)
parser.add_argument('--radius-cutoff', type=float, default=0.0)
parser.add_argument('--mbondi2', action='store_true')
args = parser.parse_args()
sys.argv = [sys.argv[0], args.topology, args.coordinates, args.plain, args.forces]

def sections(path):
    out = {}; name = None
    for line in open(path):
        if line.startswith('%FLAG'):
            name = line.split()[1]; out[name] = []
        elif line.startswith('%FORMAT'):
            out[name + '#w'] = int(re.match(r'\(\d*[aAiIeEfF](\d+)', line[line.index('('):]).group(1))
        elif name and not line.startswith('%'):
            w = out[name + '#w']; s = line.rstrip('\n')
            out[name] += [s[k:k + w] for k in range(0, len(s), w) if s[k:k + w].strip()]
    return out

top = sections(sys.argv[1])
n = int(top['POINTERS'][0])
q = [float(v) / 18.2223 for v in top['CHARGE']]
rho = [float(v) * 0.1 for v in top['RADII']]
screen = [float(v) for v in top['SCREEN']]
if args.mbondi2:
    # Bondi's radii, 1.3 Å for a hydrogen bonded to a nitrogen, and the
    # screening factors of each element.
    numbers = [int(v) for v in top['ATOMIC_NUMBER']]
    partner = [-1] * n
    for k in range(0, len(top['BONDS_INC_HYDROGEN']), 3):
        a = int(top['BONDS_INC_HYDROGEN'][k]) // 3; b = int(top['BONDS_INC_HYDROGEN'][k + 1]) // 3
        partner[a] = b if partner[a] < 0 else partner[a]
        partner[b] = a if partner[b] < 0 else partner[b]
    table = {6: (1.7, 0.72), 7: (1.55, 0.79), 8: (1.5, 0.85), 9: (1.5, 0.88), 14: (2.1, 0.8),
             15: (1.85, 0.86), 16: (1.8, 0.96), 17: (1.7, 0.8)}
    rho = []; screen = []
    for i in range(n):
        if numbers[i] == 1:
            radius = 1.3 if partner[i] >= 0 and numbers[partner[i]] == 7 else 1.2
            rho.append(radius * 0.1); screen.append(0.85)
        else:
            radius, factor = table.get(numbers[i], (1.5, 0.8))
            rho.append(radius * 0.1); screen.append(factor)
cut = args.radius_cutoff * 0.1
kappa = 0.0
if args.salt > 0:
    kappa = 0.73 * math.sqrt(2 * 6.02214076e23 * 1.602176634e-19 ** 2 * args.salt * 1000 /
                             (8.8541878128e-12 * 78.5 * 1.380649e-23 * args.temperature)) * 1e-9
lines = open(sys.argv[2]).read().split('\n')
vals = [float(lines[2 + i // 6][12 * (i % 6):12 * (i % 6) + 12]) for i in range(3 * n)]
X = [[v * 0.1 for v in vals[3 * i:3 * i + 3]] for i in range(n)]
f = 138.935457644; tau = 1 / 1.0 - 1 / 78.5; gamma = 0.0054 * 4.184 * 100
alpha, beta, gamma3 = {'OBC1': (0.8, 0.0, 2.909125), 'OBC2': (1.0, 0.8, 4.85)}.get(args.model, (0, 0, 0))

def screened(distance):
    return 1 / 1.0 - math.exp(-kappa * distance) / 78.5

def energies(x):
    d = lambda i, j: math.dist(x[i], x[j])
    B = []
    for i in range(n):
        ri = rho[i] - 0.009; s = 0.0
        for j in range(n):
            if j == i:
                continue
            r = d(i, j); sj = screen[j] * (rho[j] - 0.009)
            L = max(ri, abs(r - sj)); U = r + sj
            if cut > 0:
                U = min(U, cut)
            if L < U:
                l = 1 / L; u = 1 / U
                t = l - u + 0.25 * r * (u * u - l * l) + 0.5 / r * math.log(u / l) + 0.25 * sj * sj / r * (l * l - u * u)
                if ri < sj - r:
                    t += 2 * (1 / ri - l)
                s += t
        if args.model == 'HCT':
            B.append(1 / (1 / ri - 0.5 * s))
            continue
        psi = 0.5 * s * ri
        B.append(1 / (1 / ri - math.tanh(alpha * psi - beta * psi ** 2 + gamma3 * psi ** 3) / rho[i]))
    born = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            fgb = math.sqrt(d(i, j) ** 2 + B[i] * B[j] * math.exp(-d(i, j) ** 2 / (4 * B[i] * B[j])))
            born += -screened(fgb) * f * q[i] * q[j] / fgb
    born += sum(-0.5 * screened(B[i]) * f * q[i] ** 2 / B[i] for i in range(n))
    surface = sum(4 * math.pi * gamma * (rho[i] + 0.14) ** 2 * (rho[i] / B[i]) ** 6 for i in range(n))
    return born, surface

born, surface = energies(X)
print('generalized Born %.6f' % (born / 4.184))
print('nonpolar surface %.6f' % (surface / 4.184))
F = {int(l.split()[0]): [float(v) for v in l.split()[2:5]] for l in open(sys.argv[4])}
P = {int(l.split()[0]): [float(v) for v in l.split()[2:5]] for l in open(sys.argv[3])}
h = 1e-6; worst = 0; ref = 0
for a in [0, 5, 20, 41, 66, 90]:
    for c in range(3):
        e = []
        for step in (h, -h):
            x = [r[:] for r in X]; x[a][c] += step
            e.append(sum(energies(x)))
        force = -(e[0] - e[1]) / (2 * h)
        mine = F[a][c] - P[a][c]
        worst = max(worst, abs(mine - force)); ref = max(ref, abs(force))
print("forces against differences of the energy: %s" % ("ok" if worst < 1e-6 * ref else "FAILED %.2e" % worst))
