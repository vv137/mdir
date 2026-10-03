"""A file of energies of `[output] energy` (D149) against the rows of the
log of the same run: the same steps, and each value as the log prints it,
to its four decimals.

    check_columns.py LOG ENERGY"""
import sys

rows = []
for line in open(sys.argv[1]):
    fields = line.split()
    if line.startswith('INFO:') and fields[1].isdigit():
        rows.append([float(v) for v in fields[1:]])
lines = open(sys.argv[2]).read().splitlines()
names, units = lines[0].split()[1:], lines[1].split()[1:]
values = [[float(v) for v in line.split()] for line in lines[2:]]
assert len(names) == len(units), (names, units)
worst = 0.0
for row, value in zip(rows, values):
    assert len(row) == len(value) == len(names), (row, value, names)
    assert row[0] == value[0], (row, value)
    worst = max(worst, max(abs(a - b) for a, b in zip(row[1:], value[1:])))
ok = len(rows) == len(values) and worst <= 5.0001e-5
print('%d rows of %d columns, those of the log: %s'
      % (len(values), len(names), 'ok' if ok else
         'FAILED, %d rows in the log, %.2e' % (len(rows), worst)))
