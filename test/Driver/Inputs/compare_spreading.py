"""Runs one control file with each layout of the spreading of PME on a
device (MDIR_PME_SPREAD, D282) and compares each run after the
first with the first: the forces of the checkpoint `run.h5`, by the root
mean square of their difference relative to that of the first,
sqrt(sum |F - F1|^2 / sum |F1|^2), and the rows of energies, each number
relative to the largest magnitude of its column in the first log.

    compare_spreading.py MDIR DIRECTORY CASE FORCES ENERGIES LAYOUT...

Runs DIRECTORY/CASE.toml in DIRECTORY with the program MDIR; prints, for each layout after the
first, 'LAYOUT: forces ok, energies ok' (or 'differ') with the numbers."""

import math
import os
import subprocess
import sys


def run(mdir, directory, case, layout):
    env = dict(os.environ, MDIR_PME_SPREAD=layout)
    log = subprocess.run([mdir, "run", case + ".toml"], cwd=directory,
                         env=env, capture_output=True, text=True, check=True)
    checkpoint = os.path.join(directory, "%s-%s.h5" % (case, layout))
    os.replace(os.path.join(directory, "run.h5"), checkpoint)
    forces = subprocess.run([mdir, "checkpoint", "--print=forces",
                             checkpoint], capture_output=True, text=True,
                            check=True).stdout
    values = [[float(v) for v in line.split()[2:5]]
              for line in forces.splitlines() if line.strip()]
    rows = []
    for line in log.stdout.splitlines():
        fields = line.split()
        if len(fields) > 2 and fields[0] == "INFO:" and fields[1].isdigit():
            rows.append([float(v) for v in fields[3:]])
    return values, rows


mdir, directory, case = sys.argv[1:4]
force_tolerance, energy_tolerance = float(sys.argv[4]), float(sys.argv[5])
layouts = sys.argv[6:]
first_forces, first_rows = run(mdir, directory, case, layouts[0])
norm = sum(f * f for p in first_forces for f in p)
scale = [max(abs(row[c]) for row in first_rows) or 1.0
         for c in range(len(first_rows[0]))]
for layout in layouts[1:]:
    forces, rows = run(mdir, directory, case, layout)
    difference = sum((a - b) ** 2 for p, q in zip(forces, first_forces)
                     for a, b in zip(p, q))
    relative = math.sqrt(difference / norm) if norm else math.inf
    forces_ok = len(forces) == len(first_forces) and relative <= force_tolerance
    worst = max(abs(a - b) / s for row, first in zip(rows, first_rows)
                for a, b, s in zip(row, first, scale)) if rows else math.inf
    energies_ok = len(rows) == len(first_rows) and worst <= energy_tolerance
    print("%s: forces %s, energies %s (relative rms of the forces %.2e, "
          "largest difference of the energies %.2e)"
          % (layout, "ok" if forces_ok else "differ",
             "ok" if energies_ok else "differ", relative, worst))
