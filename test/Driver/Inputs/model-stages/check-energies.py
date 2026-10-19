"""Checks two-programs.py's energies against the JAX references of their
weights (model-stages-programs-gpu.test), within 2e-3 kcal/mol as
model-stages-gpu.test (the bodies compute in f32), and that the two models'
energies differ by more than 100 times that, so that a program reading the
other's weights would fail.

usage: check-energies.py ENERGIES.txt A/reference.npz B/reference.npz"""
import sys

import numpy as np

TOLERANCE = 2e-3 * 4.184  # kJ/mol

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
