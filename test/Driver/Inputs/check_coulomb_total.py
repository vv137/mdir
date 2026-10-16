"""Sums the Coulomb terms of particle mesh Ewald that a log of `mdir run`
prints at the start (the direct sum, the excluded pairs, the reciprocal
sum, the self term, and the background of a net charge) and compares the
sum with a reference, which no single term equals when the splitting
changes (D[prolate-ewald]).

    check_coulomb_total.py LOG REFERENCE TOLERANCE

Prints the sum, the reference, the difference relative to the size of the
reference, and "ok" or "FAIL"; exits non-zero on FAIL."""

import re
import sys

log, reference, tolerance = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
names = {"Coulomb", "Coulomb excluded", "Coulomb reciprocal", "Coulomb self",
         "Coulomb background"}
total, seen = 0.0, set()
for line in open(log):
    m = re.match(r"MDIR:   (Coulomb(?: [a-z]+)?)\s+(-?[0-9.]+)\s*$", line)
    if m and m.group(1) in names and m.group(1) not in seen:
        seen.add(m.group(1))
        total += float(m.group(2))
difference = abs(total - reference) / abs(reference)
ok = difference <= tolerance and {"Coulomb", "Coulomb excluded",
                                  "Coulomb reciprocal", "Coulomb self"} <= seen
print(f"total {total:.7f} reference {reference:.7f} difference {difference:.1e} "
      f"{'ok' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
