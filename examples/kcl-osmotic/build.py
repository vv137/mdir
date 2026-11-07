#!/usr/bin/env python3
"""Builds KCl in TIP3P water between two walls on the ions (#140).

    build.py <directory> [--pairs 126] [--waters 6858] [--seed 1]

Places the ions and the waters on a cubic lattice of 16 x 16 x 32 sites in
a cell of 48 x 48 x 96 A: the ions on random sites of the middle half of
the cell in z, 2 A inside the walls at L_z/4 and 3 L_z/4, and the waters,
randomly oriented, on random sites of the rest, so that the counts are
exact. The lattice is less dense than the liquid; the minimization and
the equilibration at constant pressure close it. tleap of AmberTools
(on PATH) writes system.prmtop and system.inpcrd with the ions of Joung
and Cheatham for TIP3P (frcmod.ionsjc_tip3p, which leaprc.water.tip3p
loads). The two systems of the study:

    about 2 m, the training point:  --pairs 126 --waters 6858
    about 1 m, the held-out point:  --pairs 65  --waters 7108
"""
import argparse
import math
import os
import random
import subprocess

N_XY, N_Z, EDGE = 16, 32, 48.0          # sites and the edge of the cell in x and y, A
SPACING = EDGE / N_XY                    # 3 A
R_OH, ANGLE = 0.9572, math.radians(104.52)   # TIP3P


def rotation(rng):
    """A uniformly random rotation matrix (from a random unit quaternion)."""
    u1, u2, u3 = rng.random(), rng.random(), rng.random()
    a, b = math.sqrt(1 - u1), math.sqrt(u1)
    w, x = a * math.sin(2 * math.pi * u2), a * math.cos(2 * math.pi * u2)
    y, z = b * math.sin(2 * math.pi * u3), b * math.cos(2 * math.pi * u3)
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("directory")
    ap.add_argument("--pairs", type=int, default=126)
    ap.add_argument("--waters", type=int, default=6858)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    sites = [(i, j, k) for i in range(N_XY) for j in range(N_XY) for k in range(N_Z)]
    lz = N_Z * SPACING
    inner = [s for s in sites if lz / 4 + 2 < (s[2] + 0.5) * SPACING < 3 * lz / 4 - 2]
    if 2 * a.pairs > len(inner) or 2 * a.pairs + a.waters > len(sites):
        raise SystemExit("too many molecules for the lattice")
    ions = rng.sample(inner, 2 * a.pairs)
    taken = set(ions)
    rest = [s for s in sites if s not in taken]
    waters = rng.sample(rest, a.waters)
    # local frame of a water: O at the origin, H in the xy plane
    h1 = (R_OH, 0.0, 0.0)
    h2 = (R_OH * math.cos(ANGLE), R_OH * math.sin(ANGLE), 0.0)
    lines, serial, res = [], 0, 0

    def atom(name, resname, x):
        nonlocal serial
        serial += 1
        lines.append("ATOM  %5d %-4s %-3s  %4d    %8.3f%8.3f%8.3f  1.00  0.00"
                     % (serial % 100000, name, resname, res % 10000, x[0], x[1], x[2]))

    center = lambda s: [(c + 0.5) * SPACING for c in s]
    for n, s in enumerate(ions):
        res += 1
        atom("K+" if n % 2 == 0 else "Cl-", "K+" if n % 2 == 0 else "Cl-", center(s))
        lines.append("TER")
    for s in waters:
        res += 1
        o, m = center(s), rotation(rng)
        rot = lambda v: [o[i] + sum(m[i][j] * v[j] for j in range(3)) for i in range(3)]
        atom("O", "WAT", o)
        atom("H1", "WAT", rot(h1))
        atom("H2", "WAT", rot(h2))
        lines.append("TER")
    os.makedirs(a.directory, exist_ok=True)
    with open(os.path.join(a.directory, "lattice.pdb"), "w") as f:
        f.write("\n".join(lines) + "\nEND\n")
    with open(os.path.join(a.directory, "system.leap"), "w") as f:
        f.write("source leaprc.water.tip3p\n"
                "s = loadpdb lattice.pdb\n"
                "setBox s \"centers\"\n"
                f"set s box {{{EDGE} {EDGE} {lz}}}\n"
                "saveamberparm s system.prmtop system.inpcrd\n"
                "quit\n")
    with open(os.path.join(a.directory, "leap.log"), "w") as log:
        subprocess.run(["tleap", "-f", "system.leap"], cwd=a.directory, check=True, stdout=log, stderr=log)
    text = open(os.path.join(a.directory, "leap.log")).read()
    if "Errors = 0" not in text:
        raise SystemExit("tleap reported errors; see leap.log")
    print(f"built {a.pairs} KCl and {a.waters} waters, {2 * a.pairs + 3 * a.waters} atoms, "
          f"in {EDGE:g} x {EDGE:g} x {lz:g} A: {a.directory}/system.prmtop")


if __name__ == "__main__":
    main()
