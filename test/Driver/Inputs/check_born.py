"""Generalized Born of generalized-born.test, computed from the topology and
the coordinates of the peptide: the Born radii of OBC II from the integral
of the descreening over all pairs [Hawkins1996, Onufriev2004], the energy
-tau f (sum over pairs of q q / f_GB + sum of q^2 / 2B), the nonpolar term
4 pi gamma (rho + 0.14 nm)^2 (rho / B)^6, and the forces that they add on a
few atoms against central differences of the energy (kJ/mol/nm).

    check_born.py TOPOLOGY COORDINATES PLAIN_FORCES FORCES

PLAIN_FORCES and FORCES are `mdir checkpoint --print=forces` of the runs
without and with generalized Born."""
import math, re, sys

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
lines = open(sys.argv[2]).read().split('\n')
vals = [float(lines[2 + i // 6][12 * (i % 6):12 * (i % 6) + 12]) for i in range(3 * n)]
X = [[v * 0.1 for v in vals[3 * i:3 * i + 3]] for i in range(n)]
f = 138.935457644; tau = 1 / 1.0 - 1 / 78.5; gamma = 0.0054 * 4.184 * 100

def energies(x):
    d = lambda i, j: math.dist(x[i], x[j])
    B = []
    for i in range(n):
        ri = rho[i] - 0.009; s = 0.0
        for j in range(n):
            if j == i:
                continue
            r = d(i, j); sj = screen[j] * (rho[j] - 0.009)
            if ri < r + sj:
                L = max(ri, abs(r - sj)); U = r + sj; l = 1 / L; u = 1 / U
                t = l - u + 0.25 * r * (u * u - l * l) + 0.5 / r * math.log(u / l) + 0.25 * sj * sj / r * (l * l - u * u)
                if ri < sj - r:
                    t += 2 * (1 / ri - l)
                s += t
        psi = 0.5 * s * ri
        B.append(1 / (1 / ri - math.tanh(psi - 0.8 * psi ** 2 + 4.85 * psi ** 3) / rho[i]))
    born = sum(-tau * f * q[i] * q[j] / math.sqrt(d(i, j) ** 2 + B[i] * B[j] * math.exp(-d(i, j) ** 2 / (4 * B[i] * B[j])))
               for i in range(n) for j in range(i + 1, n))
    born += sum(-0.5 * tau * f * q[i] ** 2 / B[i] for i in range(n))
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
