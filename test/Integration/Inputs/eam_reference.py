"""Reference values for the EAM test on domains (dist-eam.mlir, D[ml2-eam]).

Evaluates the 64 particles of lj-forces.mlir with an embedded-atom potential
of the analytic form of Finnis and Sinclair (Philos. Mag. A 50, 45 (1984)),
over all pairs in the minimum image, in double precision, from nothing but
the definitions and their closed-form derivatives:

    U = sum_i F(rho_i) + sum_{i<j} phi(r_ij),  rho_i = sum_{j != i} f(r_ij),
    f(r) = (r - D)^2,  F(rho) = -A sqrt(rho),
    phi(r) = (r - C)^2 (C0 + C1 r + C2 r^2),   for r < D = C, else 0;
    F_i = -sum_j [(F'(rho_i) + F'(rho_j)) f'(r) + phi'(r)] d_ij / r,
    W = sum_{i<j} d_ij (x) K_ij, K_ij the term of the pair in F_i.

Usage: python3 eam_reference.py
"""

import math

COUNT = 64
SPACING = 1.2
EDGE = 4.8
AMPLITUDE = 0.2
A = 1.0
D = 2.0
C = 2.0
C0 = 3.0
C1 = -2.0
C2 = 0.4


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


def density(r):
    """f and f'."""
    return (r - D) ** 2, 2.0 * (r - D)


def pair(r):
    """phi and phi'."""
    p = C0 + C1 * r + C2 * r * r
    dp = C1 + 2.0 * C2 * r
    return (r - C) ** 2 * p, 2.0 * (r - C) * p + (r - C) ** 2 * dp


def pairs(positions):
    """(i, j, d, r) of every pair i < j within the cutoff."""
    for i in range(COUNT):
        for j in range(i + 1, COUNT):
            d = []
            for k in range(3):
                component = positions[i][k] - positions[j][k]
                component -= EDGE * round(component / EDGE)
                d.append(component)
            r = math.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2)
            if r < D:
                yield i, j, d, r


def evaluate(positions):
    """The energy, the forces, and the virial."""
    rho = [0.0] * COUNT
    energy = 0.0
    found = list(pairs(positions))
    for i, j, _, r in found:
        f, _ = density(r)
        rho[i] += f
        rho[j] += f
        energy += pair(r)[0]
    energy += sum(-A * math.sqrt(value) for value in rho)
    embedding = [-A / (2.0 * math.sqrt(value)) for value in rho]
    forces = [[0.0, 0.0, 0.0] for _ in range(COUNT)]
    virial = [[0.0, 0.0, 0.0] for _ in range(3)]
    for i, j, d, r in found:
        _, df = density(r)
        _, dphi = pair(r)
        factor = -((embedding[i] + embedding[j]) * df + dphi) / r
        for k in range(3):
            forces[i][k] += factor * d[k]
            forces[j][k] -= factor * d[k]
        for a in range(3):
            for b in range(3):
                virial[a][b] += d[a] * factor * d[b]
    return energy, forces, virial


def main():
    energy, forces, virial = evaluate(place())
    largest = max(abs(c) for f in forces for c in f)
    print(f"energy {energy!r}")
    print(f"largest force component {largest!r}")
    for i, k in ((0, 0), (17, 1), (30, 2), (45, 0), (63, 1), (63, 2)):
        print(f"force[{i}][{k}] {forces[i][k]!r}")
    trace = virial[0][0] + virial[1][1] + virial[2][2]
    print(f"virial diagonal {virial[0][0]!r} {virial[1][1]!r} "
          f"{virial[2][2]!r}")
    print(f"virial trace {trace!r}")
    print(f"virial xy {virial[0][1]!r}")


if __name__ == "__main__":
    main()
