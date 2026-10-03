"""The moving trap of external-terms.test: two argon atoms, each in a
harmonic well of its own, E = k ((x - (x0 + v t))^2 + (y - y0)^2 +
(z - z0)^2) in kcal/mol and Å, whose center moves along x at v Å/ps.
Velocity Verlet with the forces at the time of the end of each step, as
MDIR computes them (D145), gives the positions after 100 steps of 1 fs.

    check_trap.py POSITIONS...

POSITIONS are `mdir checkpoint --print=positions`."""
import sys

x = [[1.0, 1.0, 1.0], [1.5, 1.1, 0.9]]
v = [[0.0] * 3, [0.0] * 3]
m = 39.948; dt = 0.001
k = [2.0, 3.0]; x0 = [10.0, 15.0]; y0 = [10.5, 11.0]; z0 = [9.5, 9.0]
vel = 30.0

def forces(x, t):
    f = []
    for i in range(2):
        ref = [x0[i] + vel * t, y0[i], z0[i]]
        f.append([-2 * k[i] * (10 * x[i][c] - ref[c]) * 10 * 4.184   # kJ/mol/nm
                  for c in range(3)])
    return f

f = forces(x, 0.0)
for n in range(1, 101):
    for i in range(2):
        for c in range(3):
            v[i][c] += f[i][c] / m * dt / 2; x[i][c] += v[i][c] * dt
    f = forces(x, n * dt)
    for i in range(2):
        for c in range(3):
            v[i][c] += f[i][c] / m * dt / 2
for path in sys.argv[1:]:
    mine = [[float(v) for v in l.split()[2:5]] for l in open(path)]
    print('positions: %s' % ('equal' if mine == x else 'DIFFER by %.2e' % max(
        abs(a - b) for p, q in zip(mine, x) for a, b in zip(p, q))))
