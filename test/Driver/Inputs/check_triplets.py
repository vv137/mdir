"""The terms of triplet-terms.test on the mW water of Molinero and Moore
[Molinero2009], computed from the coordinates: the two-body term over the
pairs and the three-body term over each center and each unordered pair of
its neighbors within a sigma [StillingerWeber1985], their energies, the
forces of each against those that MDIR gives (kcal/mol/Å) to TOLERANCE of
the largest, and the virial of the three-body term, the trace of the sum
over the triplets of d (x) F, against the change of the virial of the log.

    check_triplets.py PDB EDGE PLAIN_FORCES FORCES PLAIN_LOG LOG [TOLERANCE]

PLAIN_FORCES and FORCES are `mdir checkpoint --print=forces` (kJ/mol/nm) of the runs
without and with the three-body term, PLAIN_LOG and LOG their logs."""
import math
import re
import sys

A, B = 7.049556277, 0.6022245584
LAMBDA, EPSILON, SIGMA, GAMMA, CUT = 23.15, 6.189, 2.3925, 1.2, 1.8
RC = CUT * SIGMA

LINES = [line for line in open(sys.argv[1]) if line.startswith('HETATM')]
X = [[float(line[30 + 8 * c:38 + 8 * c]) for c in range(3)]
     for line in LINES]
# The type WU of the second run of the test has lambda = 30 where WT has
# 23.15, and its expression takes lambda1 sqrt(lambda2 lambda3) / 23.15 for
# lambda: the parameters of the center and of the ends by their places.
LAMBDAS = [30.0 if line[12:16].strip() == 'WU' else LAMBDA for line in LINES]
N = len(X)
EDGE = float(sys.argv[2])
TOLERANCE = float(sys.argv[7]) if len(sys.argv) > 7 else 1e-9


def image(i, j):
    """x_j - x_i in the minimum image, and its length."""
    d = [X[j][c] - X[i][c] for c in range(3)]
    d = [v - EDGE * round(v / EDGE) for v in d]
    return d, math.sqrt(sum(v * v for v in d))


def two_body():
    energy = 0.0
    forces = [[0.0] * 3 for _ in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            d, r = image(i, j)
            if r >= RC:
                continue
            s4 = (SIGMA / r) ** 4
            f = math.exp(SIGMA / (r - RC))
            energy += A * EPSILON * (B * s4 - 1) * f
            slope = A * EPSILON * (-4 * B * s4 / r * f
                                   - (B * s4 - 1) * f * SIGMA / (r - RC) ** 2)
            for c in range(3):
                forces[j][c] -= slope * d[c] / r
                forces[i][c] += slope * d[c] / r
    return energy, forces


def three_body():
    energy = 0.0
    forces = [[0.0] * 3 for _ in range(N)]
    trace = 0.0
    for i in range(N):
        legs = [(j,) + tuple(image(i, j)) for j in range(N) if j != i]
        legs = [leg for leg in legs if leg[2] < RC]
        for p in range(len(legs)):
            for q in range(p + 1, len(legs)):
                j, u, ru = legs[p]
                k, v, rv = legs[q]
                c = sum(a * b for a, b in zip(u, v)) / (ru * rv)
                fu = math.exp(GAMMA * SIGMA / (ru - RC))
                fv = math.exp(GAMMA * SIGMA / (rv - RC))
                weight = LAMBDAS[i] * math.sqrt(LAMBDAS[j] * LAMBDAS[k]) / LAMBDA
                h = weight * EPSILON * (c + 1 / 3) ** 2
                energy += h * fu * fv
                dh = 2 * weight * EPSILON * (c + 1 / 3)
                dfu = -fu * GAMMA * SIGMA / (ru - RC) ** 2
                dfv = -fv * GAMMA * SIGMA / (rv - RC) ** 2
                for a in range(3):
                    gu = (dh * (v[a] / (ru * rv) - c * u[a] / ru ** 2) * fu * fv
                          + h * dfu * fv * u[a] / ru)
                    gv = (dh * (u[a] / (ru * rv) - c * v[a] / rv ** 2) * fu * fv
                          + h * fu * dfv * v[a] / rv)
                    forces[j][a] -= gu
                    forces[k][a] -= gv
                    forces[i][a] += gu + gv
                    trace += -u[a] * gu - v[a] * gv
    return energy, forces, trace


def read_forces(path):
    """The forces of a checkpoint, in kJ/mol/nm, in kcal/mol/Å."""
    rows = [line.split() for line in open(path) if line.strip()]
    return [[float(v) / 41.84 for v in row[2:5]] for row in rows]


def logged(path):
    """The potential energy and the virial at step 0 of the log."""
    text = open(path).read()
    header = re.search(r'INFO:\s+STEP.*', text).group(0).split()[1:]
    row = re.search(r'INFO:\s+0\s.*', text).group(0).split()[1:]
    values = dict(zip(header, row))
    return float(values['POTENTIAL_ENE']), float(values['VIRIAL'])


def compare(name, mdir, reference):
    largest = max(abs(f) for row in reference for f in row)
    worst = max(abs(a - b) for ra, rb in zip(mdir, reference)
                for a, b in zip(ra, rb))
    print('forces of the %s term: %s' % (
        name, 'ok' if worst <= TOLERANCE * largest
        else 'FAILED by %.3e of %.3e' % (worst, largest)))


e2, f2 = two_body()
e3, f3, w3 = three_body()
print('two-body %.4f' % e2)
print('three-body %.4f' % e3)
plain = read_forces(sys.argv[3])
terms = read_forces(sys.argv[4])
compare('two-body', plain, f2)
compare('three-body', [[a - b for a, b in zip(ra, rb)]
                       for ra, rb in zip(terms, plain)], f3)
u_plain, w_plain = logged(sys.argv[5])
u_terms, w_terms = logged(sys.argv[6])
# The log prints four decimals.
print('energies of the log: %s' % (
    'ok' if abs(u_plain - e2) < 1e-4 and abs(u_terms - u_plain - e3) < 2e-4
    else 'FAILED, %.4f and %.4f' % (u_plain, u_terms - u_plain)))
print('virial of the log: %s' % (
    'ok' if abs(w_terms - w_plain - w3) < 2e-4
    else 'FAILED, %.4f against %.4f' % (w_terms - w_plain, w3)))
