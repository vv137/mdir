"""Reference values for the integration tests of the Lennard-Jones system.

Evaluates the system of lj-forces.mlir and lj-dynamics.mlir over all pairs,
in double precision, with nothing but the definitions: the minimum image, the
switched Lennard-Jones potential, and its closed-form derivative.

Usage: python3 lj_reference.py
"""

import math

COUNT = 64
SPACING = 1.2
EDGE = 4.8
AMPLITUDE = 0.2
EPS = 1.0
SIGMA = 1.0
CUTOFF = 2.0
SWITCH_FROM = 1.6


def place():
    """The positions, from the generator that the tests use."""
    state = 42
    positions = []
    for i in range(COUNT):
        digits = (i % 4, (i // 4) % 4, (i // 16) % 4)
        point = []
        for k in range(3):
            state = (state * 1103515245 + 12345) % 2147483648
            fraction = state / 2147483648.0
            point.append(digits[k] * SPACING + (fraction - 0.5) * AMPLITUDE)
        positions.append(point)
    return positions


def switch(r):
    """The switching function and its derivative."""
    if r <= SWITCH_FROM:
        return 1.0, 0.0
    width = CUTOFF - SWITCH_FROM
    t = (r - SWITCH_FROM) / width
    value = 1.0 - 10.0 * t**3 + 15.0 * t**4 - 6.0 * t**5
    slope = (-30.0 * t**2 + 60.0 * t**3 - 30.0 * t**4) / width
    return value, slope


def pair(r):
    """The pair energy and its derivative with respect to the distance."""
    s6 = (SIGMA / r) ** 6
    u = 4.0 * EPS * (s6 * s6 - s6)
    du = -(24.0 * EPS / r) * (2.0 * s6 * s6 - s6)
    s, ds = switch(r)
    return u * s, du * s + u * ds


def virial(positions):
    """The virial W = sum over the pairs of d (x) K, with the displacement d
    from j to i and the force K on i due to j. Row a, column b is
    d_a K_b."""
    total = [[0.0, 0.0, 0.0] for _ in range(3)]
    for i in range(COUNT):
        for j in range(i + 1, COUNT):
            d = []
            for k in range(3):
                component = positions[i][k] - positions[j][k]
                component -= EDGE * round(component / EDGE)
                d.append(component)
            r = math.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2)
            if r >= CUTOFF:
                continue
            _, du = pair(r)
            for a in range(3):
                for b in range(3):
                    total[a][b] += d[a] * (-du * d[b] / r)
    return total


def evaluate(positions):
    """The energy and the forces, over all pairs."""
    energy = 0.0
    forces = [[0.0, 0.0, 0.0] for _ in positions]
    for i in range(COUNT):
        for j in range(i + 1, COUNT):
            d = []
            for k in range(3):
                component = positions[i][k] - positions[j][k]
                component -= EDGE * round(component / EDGE)
                d.append(component)
            r = math.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2)
            if r >= CUTOFF:
                continue
            u, du = pair(r)
            energy += u
            for k in range(3):
                force = -du * d[k] / r
                forces[i][k] += force
                forces[j][k] -= force
    return energy, forces


def kinetic(velocities, mass):
    return sum(0.5 * mass * sum(c * c for c in v) for v in velocities)


def velocity_verlet(positions, velocities, mass, dt, steps):
    """Integrates in place and returns the forces at the end."""
    _, forces = evaluate(positions)
    for _ in range(steps):
        for i in range(COUNT):
            for k in range(3):
                velocities[i][k] += 0.5 * dt * forces[i][k] / mass
                positions[i][k] += dt * velocities[i][k]
        _, forces = evaluate(positions)
        for i in range(COUNT):
            for k in range(3):
                velocities[i][k] += 0.5 * dt * forces[i][k] / mass
    return forces


def leapfrog(positions, velocities, mass, dt, steps):
    """Integrates in place. The velocities are half a step behind."""
    for _ in range(steps):
        _, forces = evaluate(positions)
        for i in range(COUNT):
            for k in range(3):
                velocities[i][k] += dt * forces[i][k] / mass
                positions[i][k] += dt * velocities[i][k]


def main():
    positions = place()
    energy, forces = evaluate(positions)
    print("lj-forces.mlir")
    print("  energy          ", repr(energy))
    print("  force on 0      ", [repr(c) for c in forces[0]])
    print("  force on 17, x  ", repr(forces[17][0]))
    print("  sum of |F|^2    ", repr(sum(c * c for f in forces for c in f)))
    w = virial(positions)
    print("  virial, trace   ", repr(w[0][0] + w[1][1] + w[2][2]))
    for a in range(3):
        print("  virial, row %d   " % a, [repr(c) for c in w[a]])

    mass, dt, steps = 1.0, 0.004, 200

    positions = place()
    velocities = [[0.0, 0.0, 0.0] for _ in range(COUNT)]
    start, _ = evaluate(positions)
    velocity_verlet(positions, velocities, mass, dt, steps)
    potential, _ = evaluate(positions)
    print("lj-dynamics.mlir, velocity Verlet, %d steps of %g" % (steps, dt))
    print("  energy at start ", repr(start))
    print("  potential at end", repr(potential))
    print("  kinetic at end  ", repr(kinetic(velocities, mass)))
    print("  total at end    ", repr(potential + kinetic(velocities, mass)))
    print("  x of particle 5 ", repr(positions[5][0]))
    verlet_positions = [list(p) for p in positions]

    # Leapfrog from the same physical state: the stored velocity is half a
    # step behind, v(-dt/2) = v(0) - (dt/2) F(0) / m.
    positions = place()
    _, forces = evaluate(positions)
    velocities = [[-0.5 * dt * c / mass for c in f] for f in forces]
    leapfrog(positions, velocities, mass, dt, steps)
    potential, _ = evaluate(positions)
    print("lj-dynamics.mlir, leapfrog, %d steps of %g" % (steps, dt))
    print("  potential at end", repr(potential))
    print("  x of particle 5 ", repr(positions[5][0]))
    print("  largest distance from the velocity Verlet positions",
          max(abs(a - b) for p, q in zip(positions, verlet_positions)
              for a, b in zip(p, q)))


if __name__ == "__main__":
    main()
