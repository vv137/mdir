"""Reference values for the tests of the templates in this directory.

Counts, by testing all pairs, the entries of a neighbor matrix: the pairs
within the reach, and the pairs that the search takes as well because they
are within its margin. Orders the particles of spatial-order.mlir by cell.

Usage: python3 neighbors_reference.py
"""

import math

# The margin of the search, as a fraction of the sum of the edge lengths.
MARGIN = 3.0e-6

# The number of particles, the edge length, and the reach of each case.
CASES = [
    (200, 6.0, 1.5),
    (200, 3.5, 1.5),
    (200, 2.9, 1.4),
    (200, 5.0, 1.5),
    (2000, 12.0, 1.5),
    (2000, 7.0, 1.0),
    (2000, 5.8, 0.8),
    (2000, 10.0, 1.5),
]


def generate(count, edge):
    """Positions in the cell, from the generator that the tests use."""
    state = 42
    positions = []
    for _ in range(count):
        position = []
        for _ in range(3):
            state = (state * 1103515245 + 12345) % 2147483648
            position.append(state / 2147483648.0 * edge)
        positions.append(position)
    return positions


def distance(a, b, edge):
    total = 0.0
    for k in range(3):
        d = a[k] - b[k]
        d -= edge * round(d / edge)
        total += d * d
    return math.sqrt(total)


def order():
    """The order of the particles of spatial-order.mlir."""
    count, edge, width = 2000, 12.0, 1.4
    positions = generate(count, edge)
    for i in range(count):
        for k in range(3):
            positions[i][k] += (i % 5 - 2.0) * edge
    ids = [i * 7919 % count for i in range(count)]

    cells = max(int(math.floor(edge / (width * 1.0001))), 1)
    inverse = 1.0 / edge

    def cell(position):
        coordinates = []
        for x in position:
            wrapped = x - math.floor(x * inverse) * edge
            coordinate = int(wrapped * inverse * cells)
            coordinates.append(min(max(coordinate, 0), cells - 1))
        return (coordinates[2] * cells + coordinates[1]) * cells \
            + coordinates[0]

    places = sorted(range(count), key=lambda i: (cell(positions[i]), ids[i]))
    total = sum((k + 1) * (i + 1) for k, i in enumerate(places))
    print(f"{count} particles, edge {edge}, cells of {width}: "
          f"{cells} cells along an edge, sum {total}")


def main():
    order()
    for count, edge, reach in CASES:
        positions = generate(count, edge)
        limit = reach + MARGIN * 3.0 * edge
        within = 0
        entries = 0
        checksum = 0
        rows = [0] * count
        for i in range(count):
            for j in range(count):
                if i == j:
                    continue
                r = distance(positions[i], positions[j], edge)
                if r < reach:
                    within += 1
                if r < limit:
                    entries += 1
                    checksum += (i + 1) * (j + 1)
                    rows[i] += 1
        print(f"{count} particles, edge {edge}, reach {reach}: "
              f"{within} within the reach, {entries} entries, "
              f"sum {checksum}, largest row {max(rows)}")


if __name__ == "__main__":
    main()
