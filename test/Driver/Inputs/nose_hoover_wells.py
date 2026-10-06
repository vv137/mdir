"""A Nose-Hoover chain on identical harmonic wells (#117).

  nose_hoover_wells.py LOG N_F STEPS
  nose_hoover_wells.py hot GRO OUT SPEED
                              writes OUT, the coordinates of GRO with the
                              velocities +-SPEED nm/ps along x

The atoms of brownian_wells.py start at the centers of their wells, so
under velocity Verlet every atom moves as x_i(t) = a(t) v_i(0), v_i(t) =
b(t) v_i(0) with one a and b for all: the run is one oscillator of
frequency omega = sqrt(k/m), whose kinetic energy starts at N_f k_B T / 2.
This script moves that oscillator and the chain of Martyna, Klein, and
Tuckerman (1992), with Q_1 = N_f k_B T / w^2 and Q_j = k_B T / w^2, w =
2 pi / tau, acting at the end of each period of 10 steps; its action over
the period is factorized as in Martyna, Tuckerman, Tobias, and Klein
(1996), Suzuki-Yoshida weights of order 6, in n_c = ceil(50 h / tau) = 11
equal parts. It finds the first action in which some part s meets
|s v_j| > 3 (D206), and compares the potential energy of each row of
LOG, all before that action, with that of the oscillator.
"""
import math
import sys

if sys.argv[1] == "hot":
    lines = open(sys.argv[2]).read().splitlines()
    count, speed = int(lines[1]), float(sys.argv[4])
    with open(sys.argv[3], "w") as f:
        f.write("\n".join(lines[:2]) + "\n")
        for i in range(count):
            f.write("%s%8.1f%8.1f%8.1f\n" % (
                lines[2 + i], speed if i % 2 == 0 else -speed, 0.0, 0.0))
        f.write("\n".join(lines[2 + count:]) + "\n")
    sys.exit(0)

log, nf, steps = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
kB = 0.0083144626181532
kT = kB * 150.0
kcal = 4.184
omega2 = 2.0 * kcal * 100 / 39.948
dt, period, tau, m = 0.10925, 10, 5.0, 3
h = period * dt
parts = math.ceil(50 * h / tau)
w = 2 * math.pi / tau
Q = [kT / w ** 2] * m
Q[0] *= nf
weights = [0.784513610477560, 0.235573213359357, -1.17767998417887,
           1.31518632068391, -1.17767998417887, 0.235573213359357,
           0.784513610477560]
v = [0.0] * m


def force(j, k):
    if j == 0:
        return (2 * k - nf * kT) / Q[0]
    return (Q[j - 1] * v[j - 1] ** 2 - kT) / Q[j]


def chain(k):
    """The factor of the velocities and the largest |s v_j| met."""
    scale, largest = 1.0, 0.0
    for _ in range(parts):
        for weight in weights:
            s = weight * h / parts
            largest = max([largest] + [abs(s * q) for q in v])
            v[m - 1] += 0.5 * s * force(m - 1, k)
            for j in range(m - 2, -1, -1):
                d = math.exp(-0.25 * s * v[j + 1])
                v[j] = (v[j] * d + 0.5 * s * force(j, k)) * d
            largest = max([largest] + [abs(s * q) for q in v])
            f = math.exp(-s * v[0])
            scale *= f
            k *= f * f
            for j in range(m - 1):
                d = math.exp(-0.25 * s * v[j + 1])
                v[j] = (v[j] * d + 0.5 * s * force(j, k)) * d
            v[m - 1] += 0.5 * s * force(m - 1, k)
            largest = max([largest] + [abs(s * q) for q in v])
    return scale, largest


k0 = 0.5 * nf * kT
x, b = 0.0, 1.0
reference = {0: 0.0}
stop = None
for step in range(1, steps + 1):
    b -= 0.5 * dt * omega2 * x
    x += dt * b
    b -= 0.5 * dt * omega2 * x
    if step % period == 0:
        # The row of the step is written before the chain acts.
        reference[step] = k0 * omega2 * x * x / kcal
        alpha, largest = chain(k0 * b * b)
        if largest > 3.0:
            stop = step
            break
        b *= alpha
print("%d parts; the reference stops at step %s, where |s v_j| reaches %.3g"
      % (parts, stop, largest))

rows, last, column, worst = 0, None, None, 0.0
for line in open(log):
    words = line.split()
    if words[:2] == ["INFO:", "STEP"]:
        column = words.index("POTENTIAL_ENE")
    elif len(words) > 2 and words[0] == "INFO:" and words[1].isdigit():
        step = int(words[1])
        rows += 1
        last = step
        difference = abs(float(words[column]) - reference.get(step, math.nan))
        if not math.isfinite(difference):
            worst = math.inf
        worst = max(worst, difference)
print("%d rows, the last at step %s; the largest difference of the potential "
      "energy %.2g kcal/mol" % (rows, last, worst))
print("within 0.001 kcal/mol: %s" % ("yes" if worst <= 0.001 else "no"))
