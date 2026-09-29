"""Compares the terms of MDIR at the start with those of GROMACS.

Usage: compare.py <directory of one force field>, which holds energy.xvg
of GROMACS and mdir.toml.
"""
import os
import re
import subprocess
import sys

d = os.path.abspath(sys.argv[1])
ff = os.path.basename(d)
rows = [l.split() for l in open(os.path.join(d, "energy.xvg")) if not l.startswith(("#", "@"))]
legends = [re.search(r'"(.*)"', l).group(1) for l in open(os.path.join(d, "energy.xvg")) if l.startswith("@ s") and "legend" in l]
g = dict(zip(legends, map(float, rows[0][1:])))
mdir = os.environ.get("MDIR", os.path.expanduser("~/build/mdir/bin/mdir"))
out = subprocess.run([mdir, "run", "mdir.toml"], cwd=d, capture_output=True,
                     text=True)
if out.returncode != 0:
    print(f"{ff:16s} MDIR: {out.stderr.strip().splitlines()[-1]}")
    sys.exit(0)
m = {}
for line in out.stdout.splitlines():
    x = re.match(r"MDIR:   (.+?)\s+(-?[0-9.]+)$", line)
    if x:
        m[x.group(1)] = float(x.group(2)) * 4.184
pairs = [("bonds", ["Bond"]), ("angles", ["Angle"]),
         ("dihedrals", ["Proper Dih.", "Per. Imp. Dih.", "Improper Dih."]),
         ("Lennard-Jones 1-4", ["LJ-14"]), ("Coulomb 1-4", ["Coulomb-14"]),
         ("Lennard-Jones", ["LJ (SR)"]), ("dispersion", ["Disper. corr."]),
         ("CMAP", ["CMAP Dih."])]
print(f"{ff:16s} {'term':20s} {'MDIR kJ/mol':>16s} {'GROMACS':>16s} {'relative':>10s}")
for mine, theirs in pairs:
    if mine not in m:
        continue
    ref = sum(g.get(t, 0.0) for t in theirs)
    val = m[mine]
    rel = abs(val - ref) / max(abs(ref), 1e-12)
    print(f"{'':16s} {mine:20s} {val:16.4f} {ref:16.4f} {rel:10.1e}")
