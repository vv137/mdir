"""A Nose-Hoover chain on identical harmonic wells (#117).

  nose_hoover_wells.py LOG N_F STEPS

The atoms of brownian_wells.py start at the centers of their wells, so
under velocity Verlet every atom moves as x_i(t) = a(t) v_i(0), v_i(t) =
b(t) v_i(0) with one a and b for all: the run is one oscillator of
frequency omega = sqrt(k/m), whose kinetic energy starts at N_f k_B T / 2.
This script moves that oscillator and the chain of Martyna, Klein, and
Tuckerman (1992), with Q_1 = N_f k_B T / w^2 and Q_j = k_B T / w^2, w =
2 pi / tau, acting at the end of each period of 10 steps; its action over
the period is factorized as in Martyna, Tuckerman, Tobias, and Klein
(1996), Suzuki-Yoshida weights of order 6, in 400 equal parts, enough that
the result does not depend on their number. It compares the potential
energy of each row of LOG with that of the oscillator.
"""
import math
import sys

log, nf, steps = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
kB = 0.0083144626181532
kT = kB * 150.0
kcal = 4.184
omega2 = 2.0 * kcal * 100 / 39.948
dt, period, tau, m, parts = 0.10925, 10, 5.0, 3, 400
w = 2 * math.pi / tau
Q = [kT / w ** 2] * m
Q[0] *= nf
h = period * dt
weights = [0.784513610477560, 0.235573213359357, -1.17767998417887,
           1.31518632068391, -1.17767998417887, 0.235573213359357,
           0.784513610477560]
v = [0.0] * m


def force(j, k):
    if j == 0:
        return (2 * k - nf * kT) / Q[0]
    return (Q[j - 1] * v[j - 1] ** 2 - kT) / Q[j]


def chain(k):
    scale = 1.0
    for _ in range(parts):
        for weight in weights:
            s = weight * h / parts
            v[m - 1] += 0.5 * s * force(m - 1, k)
            for j in range(m - 2, -1, -1):
                d = math.exp(-0.25 * s * v[j + 1])
                v[j] = (v[j] * d + 0.5 * s * force(j, k)) * d
            f = math.exp(-s * v[0])
            scale *= f
            k *= f * f
            for j in range(m - 1):
                d = math.exp(-0.25 * s * v[j + 1])
                v[j] = (v[j] * d + 0.5 * s * force(j, k)) * d
            v[m - 1] += 0.5 * s * force(m - 1, k)
    return scale


k0 = 0.5 * nf * kT
x, b = 0.0, 1.0
reference = {0: 0.0}
for step in range(1, steps + 1):
    b -= 0.5 * dt * omega2 * x
    x += dt * b
    b -= 0.5 * dt * omega2 * x
    if step % period == 0:
        b *= chain(k0 * b * b)
        reference[step] = k0 * omega2 * x * x / kcal

rows, column, worst = 0, None, 0.0
for line in open(log):
    words = line.split()
    if words[:2] == ["INFO:", "STEP"]:
        column = words.index("POTENTIAL_ENE")
    elif len(words) > 2 and words[0] == "INFO:" and words[1].isdigit():
        step = int(words[1])
        value = float(words[column])
        rows += 1
        difference = abs(value - reference[step])
        if not math.isfinite(difference):
            worst = math.inf
        worst = max(worst, difference)
print("%d rows, the largest difference of the potential energy %.2g "
      "kcal/mol" % (rows, worst))
print("within 0.01 kcal/mol: %s" % ("yes" if worst <= 0.01 else "no"))
