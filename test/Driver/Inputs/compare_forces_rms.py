"""Compares the forces of two outputs of `mdir checkpoint --print=forces`:
the root mean square of their difference relative to that of the second,
sqrt(sum |F1 - F2|^2 / sum |F2|^2), against a tolerance.

    compare_forces_rms.py FIRST SECOND TOLERANCE [LABEL]

Prints the label, the relative difference, and 'ok' or 'differ'."""
import math
import sys


def read(path):
    return [[float(v) for v in line.split()[2:5]] for line in open(path)
            if line.strip()]


first, second = read(sys.argv[1]), read(sys.argv[2])
tolerance = float(sys.argv[3])
label = sys.argv[4] if len(sys.argv) > 4 else 'forces'
difference = sum((a - b) ** 2 for p, q in zip(first, second) for a, b in zip(p, q))
norm = sum(b * b for q in second for b in q)
relative = math.sqrt(difference / norm) if norm else math.inf
ok = len(first) == len(second) and relative <= tolerance
print('%s: relative difference %.2e %s' % (label, relative, 'ok' if ok else 'differ'))
