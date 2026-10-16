"""Compares an evaluation of the model term (D[metatomic-plugin]) with the
reference of NequIP's calculator, within the tolerances of
docs/metatomic-ml1.md, Section 7 (float64 inputs; ten times them for
float32), in kJ/mol and nm.

usage: compare.py json SCALE < check-model-term.py's output
       compare.py log REFERENCE.npz < the log of mdir run
"""
import json
import re
import sys

import numpy as np

EV = 96.48533212331  # kJ/mol per eV
mode = sys.argv[1]
if mode == "json":
    scale = float(sys.argv[2])
    r = json.loads(sys.stdin.read().strip().splitlines()[-1])
    checks = {"energy": (r["energy_difference"], 5e-5 * EV),
              "forces": (r["force_difference"], 2e-5 * EV * 10),
              "virial": (r["virial_difference"], 5e-5 * EV)}
    for name, (value, tolerance) in checks.items():
        verdict = "ok" if value <= scale * tolerance else "FAILED"
        print(f"{name} {verdict}: {value:.3g} (tolerance {scale * tolerance:.3g})")
else:
    reference = float(np.load(sys.argv[2])["energy"]) / 4.184
    for line in sys.stdin:
        if line.startswith("INFO:") and line.split()[1] == "0":
            potential = float(line.split()[4])
            # The log prints 4 decimals of kcal/mol.
            verdict = "ok" if abs(potential - reference) <= 2e-3 else "FAILED"
            print(f"mdir run energy {verdict}: {potential:.4f} against {reference:.4f}")
