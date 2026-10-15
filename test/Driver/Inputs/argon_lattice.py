"""Writes argon atoms on a simple cubic lattice to the PDB file OUT, n per
edge at the spacing a (Å, 3.8 by default), in a cubic cell of n a.

    argon_lattice.py N OUT [A]"""
import sys

n = int(sys.argv[1])
a = float(sys.argv[3]) if len(sys.argv) > 3 else 3.8
edge = n * a
lines = ["CRYST1%9.3f%9.3f%9.3f  90.00  90.00  90.00 P 1           1" % (edge, edge, edge)]
k = 0
for i in range(n):
    for j in range(n):
        for m in range(n):
            k += 1
            x, y, z = [(q + 0.5) * a for q in (i, j, m)]
            lines.append("HETATM%5d  AR   AR A   1    %8.3f%8.3f%8.3f  1.00  0.00          AR"
                         % (k % 100000, x, y, z))
open(sys.argv[2], "w").write("\n".join(lines) + "\nEND\n")
