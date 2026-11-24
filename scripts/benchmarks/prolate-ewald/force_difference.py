"""Estimate a projected force from repeated pairs of displaced energies.

Read JSON lines from stdin. Each record contains a projected force under
"force" and [h, E(+h), E(-h)] triples under "samples". Other fields identify
the case and are passed through. A line prefixed with RESULT and a space is
also accepted, for use with redirected batch output.

The estimated numerical uncertainty combines the spread of repeated
differences, a subtraction roundoff allowance, and the change on doubling
and halving h. It is a displacement sensitivity estimate, not a confidence
interval. The selection of h uses only the energies.
"""

import json
import math
import statistics
import sys


def compare(record):
    differences = {}
    rounding = {}
    for step, plus, minus in record["samples"]:
        if not all(map(math.isfinite, (step, plus, minus))) or step <= 0:
            raise ValueError("expected a positive displacement and finite energies")
        differences.setdefault(step, []).append((minus - plus) / (2 * step))
        allowance = (math.ulp(plus) + math.ulp(minus)) / (2 * step)
        rounding[step] = max(allowance, rounding.get(step, 0))
    means = {h: statistics.mean(values) for h, values in differences.items()}
    options = []
    for h, force in means.items():
        neighbors = (h / 2, h, 2 * h)
        if all(k in means for k in neighbors):
            spread = max(max(abs(v - means[k]) for v in differences[k]) + rounding[k]
                         for k in neighbors)
            variation = max(abs(means[k] - force) for k in neighbors)
            options.append((spread + variation, h, force))
    if not options:
        raise ValueError("need three displacements in a ratio of 1:2:4")
    uncertainty, h, estimate = min(options)
    return {**{k: v for k, v in record.items() if k != "samples"},
            "step": h, "difference_force": estimate, "uncertainty": uncertainty,
            "residual": record["force"] - estimate}


if __name__ == "__main__":
    for line in sys.stdin:
        line = line.removeprefix("RESULT ").strip()
        if line.startswith("{"):
            print(json.dumps(compare(json.loads(line)), allow_nan=False))
