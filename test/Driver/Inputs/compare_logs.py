"""Compares the rows of energies of two logs of mdir, column by column.

    compare_logs.py first.log second.log tolerance [column...]

Each number of the rows that begin with INFO: and a step must agree within
`tolerance`, relative to the largest magnitude of its column in the first
log. Without columns, all columns but STEP and TIME are compared. Prints
the largest difference of each column, and 'logs agree' or the column that
differs."""

import sys


def read(path):
    header, rows = None, []
    with open(path) as file:
        for line in file:
            fields = line.split()
            if not fields or fields[0] != "INFO:":
                continue
            if fields[1] == "STEP":
                header = fields[1:]
            else:
                rows.append([float(value) for value in fields[1:]])
    return header, rows


first_path, second_path, tolerance = sys.argv[1:4]
tolerance = float(tolerance)
header, first = read(first_path)
other, second = read(second_path)
if header != other or len(first) != len(second) or not first:
    print("the logs have other columns or rows")
    sys.exit(1)
columns = sys.argv[4:] or [name for name in header
                           if name not in ("STEP", "TIME")]
failed = False
for name in columns:
    k = header.index(name)
    scale = max(abs(row[k]) for row in first) or 1.0
    difference = max(abs(a[k] - b[k]) for a, b in zip(first, second))
    print(f"{name}: {difference / scale:.3e}")
    if difference > tolerance * scale:
        print(f"{name} differs by more than {tolerance:g}")
        failed = True
if not failed:
    print("logs agree")
sys.exit(1 if failed else 0)
