"""Compares two outputs of `mdir checkpoint --print`, a line for each
particle: the numbers of each line agree within the tolerance, relative to
the largest of them in the file.

    compare_columns.py FIRST SECOND TOLERANCE"""
import sys

first = [[float(v) for v in line.split()[1:]] for line in open(sys.argv[1])]
second = [[float(v) for v in line.split()[1:]] for line in open(sys.argv[2])]
tolerance = float(sys.argv[3])
largest = max(abs(v) for row in first for v in row) or 1.0
worst = max(abs(a - b) for p, q in zip(first, second) for a, b in zip(p, q))
if len(first) == len(second) and worst <= tolerance * largest:
    print('the columns agree')
else:
    print('the columns differ: %.3e of %.3e' % (worst, largest))
