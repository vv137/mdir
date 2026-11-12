"""Writes argon dimers to the PDB file OUT: n per edge on a simple cubic
lattice of spacing 10 Å, each dimer two atoms 3.8 Å apart along x, in a
cubic cell of 10 n Å. With a cutoff of 5 Å each atom has its partner alone
within the cutoff (the nearest atom of another dimer is 6.2 Å away), and,
with a reach of 12 Å, some twenty more candidates beyond it.

    argon_dimers.py N OUT"""
import sys

n = int(sys.argv[1])
a, bond = 10.0, 3.8
edge = n * a
lines = ["CRYST1%9.3f%9.3f%9.3f  90.00  90.00  90.00 P 1           1" % (edge, edge, edge)]
k = 0
for i in range(n):
    for j in range(n):
        for m in range(n):
            for dx in (0.0, bond):
                k += 1
                x, y, z = (i + 0.3) * a + dx, (j + 0.5) * a, (m + 0.5) * a
                lines.append("HETATM%5d  AR   AR A   1    %8.3f%8.3f%8.3f  1.00  0.00          AR"
                             % (k % 100000, x, y, z))
open(sys.argv[2], "w").write("\n".join(lines) + "\nEND\n")
