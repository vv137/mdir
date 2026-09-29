"""Reference values for neighbor-reuse.mlir.

Moves particles at constant velocity and, at every step, counts the pairs
within the cutoff by testing all pairs. Also counts how often a neighbor
structure with the given skin has to be rebuilt.

Usage: python3 neighbor_reuse_reference.py
"""

COUNT = 200
EDGE = 6.0
CUTOFF = 1.5
SKIN = 0.3
DT = 0.01
STEPS = 100


def generate():
    """Positions in the cell and velocities in [-1, 1), from the generator
    that the test uses."""
    state = 42

    def draw():
        nonlocal state
        state = (state * 1103515245 + 12345) % 2147483648
        return state / 2147483648.0

    positions = [[draw() * EDGE for _ in range(3)] for _ in range(COUNT)]
    velocities = [[2.0 * draw() - 1.0 for _ in range(3)]
                  for _ in range(COUNT)]
    return positions, velocities


def pairs_within(positions):
    found = 0
    for i in range(COUNT):
        for j in range(i + 1, COUNT):
            r2 = 0.0
            for k in range(3):
                d = positions[i][k] - positions[j][k]
                d -= EDGE * round(d / EDGE)
                r2 += d * d
            if r2 < CUTOFF * CUTOFF:
                found += 1
    return found


def main():
    positions, velocities = generate()
    reference = None
    builds = 0
    total = 0
    for _ in range(STEPS):
        for i in range(COUNT):
            for k in range(3):
                positions[i][k] += DT * velocities[i][k]

        stale = reference is None
        if not stale:
            farthest = max(
                sum((a - b) ** 2 for a, b in zip(p, q))
                for p, q in zip(positions, reference))
            stale = not farthest <= 0.25 * SKIN * SKIN
        if stale:
            reference = [list(p) for p in positions]
            builds += 1

        total += pairs_within(positions)

    print("pairs within the cutoff, summed over the steps:", total)
    print("builds of the neighbor structure:", builds)


if __name__ == "__main__":
    main()
