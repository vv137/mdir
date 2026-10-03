"""The steered spring of pull-coordinates.test: two argon atoms joined by
E = k (r - (r0 + v t))^2 in kcal/mol and Å, whose length grows with the
time t in ps. Velocity Verlet with the forces at the time of the end of
each step, as MDIR computes them (D145), gives the positions after 100
steps of 1 fs; the file of pulling holds at each energy the coordinates,
the energy, the force along the distance, and the force on the second
center, which follow from its own coordinates and the time.

    check_steer.py PULL POSITIONS...

POSITIONS are `mdir checkpoint --print=positions`."""
import math, sys

k, r0, vel = 5.0, 5.2, 20.0

lines = [l.split() for l in open(sys.argv[1]) if not l.startswith('#')]
worst = 0.0
for line in lines:
    step, t, r, dx, dy, dz, e, fr, fx, fy, fz = [float(v) for v in line]
    d = [dx, dy, dz]
    g = 2 * k * (r - (r0 + vel * t))
    want = [math.sqrt(dx * dx + dy * dy + dz * dz), k * (r - (r0 + vel * t)) ** 2, -g]
    want += [-g * c / r for c in d]
    got = [r, e, fr, fx, fy, fz]
    worst = max(worst, max(abs(a - b) for a, b in zip(got, want)))
print('%d lines, %s' % (len(lines), 'ok' if worst < 2e-5 else 'FAILED %.2e' % worst))

x = [[1.0, 1.0, 1.0], [1.5, 1.1, 0.9]]
v = [[0.0] * 3, [0.0] * 3]
m = 39.948; dt = 0.001

def forces(x, t):
    d = [x[1][c] - x[0][c] for c in range(3)]
    r = math.sqrt(sum(c * c for c in d))
    g = 2 * k * (10 * r - (r0 + vel * t)) * 10 * 4.184   # kJ/mol/nm
    return [[g * c / r for c in d], [-g * c / r for c in d]]

f = forces(x, 0.0)
for n in range(1, 101):
    for i in range(2):
        for c in range(3):
            v[i][c] += f[i][c] / m * dt / 2; x[i][c] += v[i][c] * dt
    f = forces(x, n * dt)
    for i in range(2):
        for c in range(3):
            v[i][c] += f[i][c] / m * dt / 2
for path in sys.argv[2:]:
    mine = [[float(v) for v in l.split()[2:5]] for l in open(path)]
    print('positions: %s' % ('equal' if mine == x else 'DIFFER by %.2e' % max(
        abs(a - b) for p, q in zip(mine, x) for a, b in zip(p, q))))
