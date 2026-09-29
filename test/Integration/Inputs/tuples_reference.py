"""Reference values for the tests of terms over tuples.

Evaluates bonds, angles, and dihedrals of a few chains of particles from
the definitions of the internal coordinates, and their forces from the
closed forms of the derivatives. The closed forms are compared with finite
differences of the energy before anything is printed.

Usage: python3 tuples_reference.py
"""

import math

EDGE = 6.0
CHAINS = 8
LENGTH = 6
COUNT = CHAINS * LENGTH
STEPS = 200
DT = 0.002


def sub(a, b):
    return [a[k] - b[k] for k in range(3)]


def add(a, b):
    return [a[k] + b[k] for k in range(3)]


def scale(s, a):
    return [s * a[k] for k in range(3)]


def dot(a, b):
    return sum(a[k] * b[k] for k in range(3))


def cross(a, b):
    return [a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]]


def norm(a):
    return math.sqrt(dot(a, a))


def displacement(a, b):
    """x_a - x_b in the minimum image."""
    d = sub(a, b)
    return [c - EDGE * round(c / EDGE) for c in d]


#===------------------------------------------------------------------------===
# Internal coordinates and their derivatives
#===------------------------------------------------------------------------===

def distance(x, members):
    """The distance of two members and its derivative for each."""
    a, b = members
    d = displacement(x[a], x[b])
    r = norm(d)
    return r, [scale(1.0 / r, d), scale(-1.0 / r, d)]


def cosine(x, members):
    """The cosine of the angle at the second member."""
    a, b, c = members
    u = displacement(x[a], x[b])
    v = displacement(x[c], x[b])
    nu, nv = norm(u), norm(v)
    value = dot(u, v) / (nu * nv)
    ga = scale(1.0 / nu, sub(scale(1.0 / nv, v), scale(value / nu, u)))
    gc = scale(1.0 / nv, sub(scale(1.0 / nu, u), scale(value / nv, v)))
    gb = scale(-1.0, add(ga, gc))
    return value, [ga, gb, gc]


def angle(x, members):
    """The angle at the second member, from 0 to pi."""
    value, gradients = cosine(x, members)
    sine = math.sqrt(1.0 - value * value)
    return math.acos(value), [scale(-1.0 / sine, g) for g in gradients]


def dihedral(x, members):
    """The angle between the planes of the first three and of the last three
    members, from -pi to pi, with the derivatives of Blondel and Karplus
    (J. Comput. Chem. 17, 1132 (1996)), which have no singularity where
    three members are in line."""
    a, b, c, d = members
    f = displacement(x[a], x[b])
    g = displacement(x[b], x[c])
    h = displacement(x[d], x[c])
    p = cross(f, g)
    q = cross(h, g)
    ng = norm(g)
    p2, q2 = dot(p, p), dot(q, q)
    value = math.atan2(dot(cross(q, p), g) / ng, dot(p, q))

    fg, hg = dot(f, g), dot(h, g)
    ga = scale(-ng / p2, p)
    gd = scale(ng / q2, q)
    gb = add(scale(ng / p2 + fg / (p2 * ng), p), scale(-hg / (q2 * ng), q))
    gc = add(scale(-ng / q2 + hg / (q2 * ng), q), scale(-fg / (p2 * ng), p))
    return value, [ga, gb, gc, gd]


#===------------------------------------------------------------------------===
# The system
#===------------------------------------------------------------------------===

def place():
    """Chains of LENGTH particles. A chain starts at a point of the cell and
    goes on in steps of about 0.5, each in a direction of its own."""
    state = 42

    def draw():
        nonlocal state
        state = (state * 1103515245 + 12345) % 2147483648
        return state / 2147483648.0

    positions = []
    for _ in range(CHAINS):
        point = [draw() * EDGE for _ in range(3)]
        positions.append(point)
        for _ in range(LENGTH - 1):
            step = [draw() - 0.5 for _ in range(3)]
            step = scale((0.4 + 0.2 * draw()) / norm(step), step)
            point = add(point, step)
            positions.append(point)
    return positions


def topology():
    """The tuples and their parameters: bonds (k, r0), angles (k, c0), and
    dihedrals (k, phi0), the last with a multiplicity of 2."""
    bonds, angles, dihedrals = [], [], []
    for chain in range(CHAINS):
        first = chain * LENGTH
        for i in range(LENGTH - 1):
            bonds.append(((first + i, first + i + 1),
                          (100.0 + 10.0 * i, 0.45 + 0.01 * i)))
        for i in range(LENGTH - 2):
            angles.append(((first + i, first + i + 1, first + i + 2),
                           (20.0 + i, -0.5 + 0.1 * i)))
        for i in range(LENGTH - 3):
            dihedrals.append(((first + i, first + i + 1, first + i + 2,
                               first + i + 3),
                              (3.0 + i, 0.3 * i)))
    return bonds, angles, dihedrals


def bond(r, k, r0):
    return 0.5 * k * (r - r0) ** 2, k * (r - r0)


def bend(c, k, c0):
    return 0.5 * k * (c - c0) ** 2, k * (c - c0)


def twist(phi, k, phi0):
    return (k * (1.0 + math.cos(2.0 * phi - phi0)),
            -2.0 * k * math.sin(2.0 * phi - phi0))


TERMS = [(distance, bond), (cosine, bend), (dihedral, twist)]


def evaluate(x, tuples):
    """The energy, the forces, and the virial of the three kinds of terms.
    The virial is the sum over the tuples and their members m of
    d_m0 (x) F_m, with the displacement of m from the first member."""
    energy = [0.0, 0.0, 0.0]
    forces = [[0.0, 0.0, 0.0] for _ in x]
    virial = [[0.0, 0.0, 0.0] for _ in range(3)]
    for kind, (coordinate, term) in enumerate(TERMS):
        for members, parameters in tuples[kind]:
            value, gradients = coordinate(x, members)
            u, slope = term(value, *parameters)
            energy[kind] += u
            for m, gradient in zip(members, gradients):
                force = scale(-slope, gradient)
                forces[m] = add(forces[m], force)
                d = displacement(x[m], x[members[0]])
                for a in range(3):
                    for b in range(3):
                        virial[a][b] += d[a] * force[b]
    return energy, forces, virial


def check(x, tuples):
    """Compares the forces with finite differences of the energy."""
    _, forces, _ = evaluate(x, tuples)
    worst = 0.0
    h = 1.0e-6
    for i in range(COUNT):
        for k in range(3):
            saved = x[i][k]
            x[i][k] = saved + h
            above = sum(evaluate(x, tuples)[0])
            x[i][k] = saved - h
            below = sum(evaluate(x, tuples)[0])
            x[i][k] = saved
            worst = max(worst,
                        abs(-(above - below) / (2.0 * h) - forces[i][k]))
    return worst


def single():
    """One tuple of each kind, without a cell, for the test of the kernels
    that differentiation generates."""
    global EDGE
    saved = EDGE
    EDGE = 1.0e6
    x = [[0.1, 0.2, -0.3], [1.0, 0.4, 0.2], [1.3, 1.5, -0.1],
         [2.4, 1.7, 0.9]]
    cases = [("bond     ", distance, bond, (0, 1), (300.0, 0.8)),
             ("cosine   ", cosine, bend, (0, 1, 2), (40.0, -0.4)),
             ("angle    ", angle, bond, (0, 1, 2), (50.0, 1.9)),
             ("dihedral ", dihedral, twist, (0, 1, 2, 3), (2.5, 0.7))]
    for name, coordinate, term, members, parameters in cases:
        value, gradients = coordinate(x, members)
        u, slope = term(value, *parameters)
        print(name, "coordinate", repr(value), "energy", repr(u))
        virial = [[0.0, 0.0, 0.0] for _ in range(3)]
        for m, gradient in zip(members, gradients):
            force = scale(-slope, gradient)
            print(name, "force on", m, [repr(c) for c in force])
            d = sub(x[m], x[members[0]])
            for a in range(3):
                for b in range(3):
                    virial[a][b] += d[a] * force[b]
        print(name, "virial", [repr(c) for row in virial for c in row])
    EDGE = saved


#===------------------------------------------------------------------------===
# Nonbonded terms with exclusions
#===------------------------------------------------------------------------===

EPSILON = 0.5
SIGMA = 0.25
CUTOFF = 1.5
SCALE14 = 0.5


def excluded_pairs(tuples):
    """The pairs that the neighborhood leaves out: those one, two, and three
    bonds apart. The last come back, scaled, as pairs14()."""
    bonds, angles, dihedrals = tuples
    pairs = set()
    for members, _ in bonds:
        pairs.add(tuple(sorted(members)))
    for members, _ in angles:
        pairs.add(tuple(sorted((members[0], members[2]))))
    for members, _ in dihedrals:
        pairs.add(tuple(sorted((members[0], members[3]))))
    return sorted(pairs)


def pairs14(tuples):
    """The pairs three bonds apart: the ends of the dihedrals."""
    return sorted(set(tuple(sorted((m[0], m[3]))) for m, _ in tuples[2]))


def lennard_jones(r):
    s6 = (SIGMA / r) ** 6
    return 4.0 * EPSILON * (s6 * s6 - s6), -24.0 * EPSILON * (2.0 * s6 * s6 - s6) / r


def nonbonded(x, tuples):
    """Lennard-Jones between the pairs within CUTOFF that are not excluded,
    and the pairs three bonds apart, scaled by SCALE14, at any distance.
    Returns the energies of the two, the forces, and the virial."""
    excluded = set(excluded_pairs(tuples))
    energy = [0.0, 0.0]
    forces = [[0.0, 0.0, 0.0] for _ in x]
    virial = [[0.0, 0.0, 0.0] for _ in range(3)]

    def add_pair(i, j, weight, kind):
        d = displacement(x[i], x[j])
        r = norm(d)
        u, slope = lennard_jones(r)
        energy[kind] += weight * u
        force = [-weight * slope / r * c for c in d]
        forces[i] = add(forces[i], force)
        forces[j] = sub(forces[j], force)
        for a in range(3):
            for b in range(3):
                virial[a][b] += d[a] * force[b]

    for i in range(COUNT):
        for j in range(i + 1, COUNT):
            if (i, j) in excluded:
                continue
            if norm(displacement(x[i], x[j])) < CUTOFF:
                add_pair(i, j, 1.0, 0)
    for i, j in pairs14(tuples):
        add_pair(i, j, SCALE14, 1)
    return energy, forces, virial


def check_nonbonded(x, tuples):
    """Compares the forces with finite differences of the energy."""
    _, forces, _ = nonbonded(x, tuples)
    worst = 0.0
    h = 1.0e-6
    for i in range(COUNT):
        for k in range(3):
            saved = x[i][k]
            x[i][k] = saved + h
            above = sum(nonbonded(x, tuples)[0])
            x[i][k] = saved - h
            below = sum(nonbonded(x, tuples)[0])
            x[i][k] = saved
            worst = max(worst, abs(-(above - below) / (2.0 * h) - forces[i][k])
                        / max(1.0, abs(forces[i][k])))
    return worst


def masses():
    """The masses of the particles: 1, 1.5, and 2 in turn."""
    return [1.0 + 0.5 * (i % 3) for i in range(COUNT)]


def velocities():
    """Velocities from a linear congruential generator, from -0.5 to 0.5
    along each axis, with the momentum of the whole removed."""
    state = 7
    m = masses()
    v = []
    for _ in range(COUNT):
        row = []
        for _ in range(3):
            state = (state * 1103515245 + 12345) % 2147483648
            row.append(state / 2147483648.0 - 0.5)
        v.append(row)
    total = sum(m)
    for k in range(3):
        p = sum(m[i] * v[i][k] for i in range(COUNT))
        for i in range(COUNT):
            v[i][k] -= p / total
    return v


def integrate(steps, dt):
    """Velocity Verlet from the positions of place() and the velocities of
    velocities(). Returns the positions, the velocities, and the potential
    and kinetic energies after `steps` steps, and the largest deviation of
    the total energy from its start along the way."""
    x = place()
    v = velocities()
    m = masses()
    tuples = topology()
    energy, f, _ = evaluate(x, tuples)

    def kinetic():
        return sum(0.5 * m[i] * dot(v[i], v[i]) for i in range(COUNT))

    start = sum(energy) + kinetic()
    worst = 0.0
    for _ in range(steps):
        for i in range(COUNT):
            v[i] = add(v[i], scale(0.5 * dt / m[i], f[i]))
            x[i] = add(x[i], scale(dt, v[i]))
        energy, f, _ = evaluate(x, tuples)
        for i in range(COUNT):
            v[i] = add(v[i], scale(0.5 * dt / m[i], f[i]))
        worst = max(worst, abs(sum(energy) + kinetic() - start))
    return x, v, sum(energy), kinetic(), start, worst


def main():
    single()
    x = place()
    tuples = topology()
    worst = check(x, tuples)
    assert worst < 1.0e-6, worst
    print("largest difference from finite differences", "%.2e" % worst)

    energy, forces, virial = evaluate(x, tuples)
    print("tuples           ", [len(t) for t in tuples])
    print("energy of bonds    ", repr(energy[0]))
    print("energy of angles   ", repr(energy[1]))
    print("energy of dihedrals", repr(energy[2]))
    print("energy             ", repr(sum(energy)))
    print("force on 0         ", [repr(c) for c in forces[0]])
    print("force on 20, y     ", repr(forces[20][1]))
    print("sum of |F|^2       ", repr(sum(c * c for f in forces for c in f)))
    print("sum of F           ",
          ["%.1e" % sum(f[k] for f in forces) for k in range(3)])
    print("virial, trace      ",
          repr(virial[0][0] + virial[1][1] + virial[2][2]))
    for a in range(3):
        print("virial, row %d      " % a, [repr(c) for c in virial[a]])

    x = place()
    worst = check_nonbonded(x, tuples)
    assert worst < 1.0e-6, worst
    energy, forces, virial = nonbonded(x, tuples)
    print("nonbonded, largest relative difference from finite differences",
          "%.2e" % worst)
    print("excluded pairs     ", len(excluded_pairs(tuples)),
          "pairs 1-4", len(pairs14(tuples)))
    print("Lennard-Jones      ", repr(energy[0]))
    print("pairs 1-4          ", repr(energy[1]))
    print("force on 0         ", [repr(c) for c in forces[0]])
    print("virial, trace      ",
          repr(virial[0][0] + virial[1][1] + virial[2][2]))

    x, v, u, k, start, worst = integrate(STEPS, DT)
    print("after", STEPS, "steps of", DT)
    print("potential energy   ", repr(u))
    print("kinetic energy     ", repr(k))
    print("position of 0      ", [repr(c) for c in x[0]])
    print("velocity of 20     ", [repr(c) for c in v[20]])
    print("total energy, start", repr(start))
    print("largest deviation  ", "%.2e" % worst)


if __name__ == "__main__":
    main()
