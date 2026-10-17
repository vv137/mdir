#!/usr/bin/env python3
"""The kernels of the reciprocal sums of PME in a report of Nsight Systems
(D[pme-tiles]): the mean time of each launch, the launches of one
function told apart by their place among the consecutive launches of its
name (the kernels of a function of the templates share its name: the
binning is #0, the zeroing of the grid, and #1, the binning proper).

    trace.py REPORT.nsys-rep

The first tenth of the launches (the warm-up) is left out. `nsys` is
taken from the path."""
import csv
import io
import shutil
import subprocess
import sys
from collections import defaultdict

nsys = shutil.which("nsys")
text = subprocess.run([nsys, "stats", "--report", "cuda_gpu_trace", "--format", "csv",
                       "--force-export=true", sys.argv[1]], capture_output=True, text=True).stdout
rows = list(csv.DictReader(io.StringIO(text[text.find("Start (ns)"):])))
times = defaultdict(list)
previous, index = None, 0
for row in rows:
    name = row.get("Name", "")
    if "pme" not in name:
        previous = None
        continue
    index = index + 1 if name == previous else 0
    previous = name
    times[(name, index)].append(float(row["Duration (ns)"]) / 1000.0)
for (name, index), t in sorted(times.items()):
    t = t[len(t) // 10:]
    print(f"{name[:64]:64s} #{index} {sum(t) / len(t):9.2f} us x{len(t)}")
