"""Checks two-programs.py's energies against the JAX references of their
weights (model-stages-programs-gpu.test), within 0.013 kJ/mol (the bodies
compute in f32), and that the two models'
energies differ by more than 100 times that, so that a program reading the
other's weights would fail.

usage: check-energies.py ENERGIES.txt A/reference.npz B/reference.npz"""
import sys

import numpy as np

# The fixed second model differs from its CPU JAX f32 reference by 0.011200
# kJ/mol and from an f64-body composition by 0.010218 kJ/mol. This bound is
# approximately 1e-5 of that model's 1291.75 kJ/mol energy, as used for
# the independent batched-stage composition test, while keeping the large
# first model's comparison equally strict. D305.
TOLERANCE = 0.013  # kJ/mol

lines = [l.split() for l in open(sys.argv[1]).read().splitlines()]
references = {"a.mlir": float(np.load(sys.argv[2])["energy"]),
              "b.mlir": float(np.load(sys.argv[3])["energy"])}
for name, value in lines:
    reference = references[name]
    difference = abs(float(value) - reference)
    verdict = "ok" if difference <= TOLERANCE else "FAILED"
    print(f"{name} {verdict}: {float(value):.6f} against {reference:.6f}")
gap = abs(references["a.mlir"] - references["b.mlir"])
print(f"the weights differ: {'ok' if gap > 100 * TOLERANCE else 'FAILED'}"
      f" ({gap:.6g} kJ/mol)")
