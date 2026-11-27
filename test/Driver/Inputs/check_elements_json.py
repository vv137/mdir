"""Prints the elements, their sources, the constraints by kind, and the
counts of the warnings of the elements of `mdir check --json` (#469), one
line each, for FileCheck."""

import collections
import json
import sys

report = json.load(sys.stdin)
elements = report["system"]["topology"]["elements"]
print("elements " + " ".join("%s:%s:%d" % (c["element"], c["source"],
                                          c["particles"])
                             for c in elements["counts"]))
sources = elements["sources"]
print("sources " + " ".join("%s=%d" % (k, sources[k]) for k in
                            ("file", "name_and_mass", "mass", "none")) +
      " fallback=%d" % elements["fallback"])
constraints = report["run"]["constraints"]
print("constraints hydrogen_bonds=%d groups=%d rigid_waters=%d" %
      (constraints["constrained_hydrogen_bonds"],
       constraints["hydrogen_bond_groups"], constraints["rigid_waters"]))
codes = collections.Counter(w["code"] for w in report["warnings"]
                            if w["code"].startswith("element_"))
print("warnings " + " ".join("%s=%d" % (k, codes[k]) for k in sorted(codes)))
