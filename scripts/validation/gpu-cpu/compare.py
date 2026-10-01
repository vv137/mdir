#!/usr/bin/env python3
"""Compares two logs of MDIR of the same run on the CPU and on the GPU:
the largest difference of the total energy between the rows of the two up
to each time, the means of the potential energy and the temperature with
their standard errors (from blocks of a tenth of the rows), and what each
log says of its conservation.

    scripts/validation/gpu-cpu/compare.py CPU.log GPU.log
"""
import math
import re
import statistics
import sys


def rows(path):
    out = []
    for line in open(path):
        if line.startswith("INFO:") and "STEP" not in line:
            out.append([float(v) for v in line.split()[1:]])
    return out


def blocks(values, n=10):
    size = len(values) // n
    means = [statistics.mean(values[k * size:(k + 1) * size]) for k in range(n)]
    return statistics.mean(values), statistics.stdev(means) / math.sqrt(n)


def main():
    a, b = rows(sys.argv[1]), rows(sys.argv[2])
    n = min(len(a), len(b))
    print("step      |E_cpu - E_gpu| (kcal/mol), largest so far")
    worst = 0.0
    for k in range(n):
        worst = max(worst, abs(a[k][2] - b[k][2]))
        if k in (1, 2, 5, 10, 20, 50, 100, 200) or k == n - 1:
            print(f"{int(a[k][0]):8d}  {worst:.3e}")
    for name, column in (("potential energy", 3), ("temperature", 5)):
        half = n // 5
        ma, ea = blocks([r[column] for r in a[half:n]])
        mb, eb = blocks([r[column] for r in b[half:n]])
        print(f"{name}: CPU {ma:.4f} +- {ea:.4f}, GPU {mb:.4f} +- {eb:.4f}, "
              f"difference {(ma - mb) / math.hypot(ea, eb):+.2f} standard errors")
    for path in sys.argv[1:3]:
        text = open(path).read()
        m = re.search(r"energy changed by ([0-9.e+-]+)", text)
        r = re.search(r"([0-9.]+) ns per day", text)
        print(f"{path}: changed by {m.group(1) if m else '?'}, "
              f"{r.group(1) if r else '?'} ns/day")


if __name__ == "__main__":
    main()
