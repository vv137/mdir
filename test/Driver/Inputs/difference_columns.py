"""The difference of two outputs of `mdir checkpoint --print`, a line for
each particle: its number, its mass, and the first less the second, the
forces that a term adds, say.

    difference_columns.py FIRST SECOND"""
import sys

for a, b in zip(open(sys.argv[1]), open(sys.argv[2])):
    p, q = a.split(), b.split()
    print(' '.join(p[:2] + ['%.17g' % (float(x) - float(y)) for x, y in zip(p[2:], q[2:])]))
