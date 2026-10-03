"""Writes mW particles on a simple cubic lattice at 0.0334 per cubic Å to
the PDB file OUT, n per edge, and prints the edge of the cell in Å.

    mw_lattice.py N OUT"""
import sys

n = int(sys.argv[1])
edge = (n ** 3 / 0.0334) ** (1 / 3)
lines = ["CRYST1%9.3f%9.3f%9.3f  90.00  90.00  90.00 P 1           1" % (edge, edge, edge)]
k = 0
for i in range(n):
    for j in range(n):
        for m in range(n):
            k += 1
            x, y, z = [(q + 0.5) * edge / n for q in (i, j, m)]
            lines.append("HETATM%5d  WT  WT  A   1    %8.3f%8.3f%8.3f  1.00  0.00          WT"
                         % (k, x, y, z))
open(sys.argv[2], "w").write("\n".join(lines) + "\nEND\n")
print("%.4f" % edge)
