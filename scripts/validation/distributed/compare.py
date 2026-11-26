"""Compare a distributed checkpoint and its energy history against a MATRIX run.

For mixed CPU comparisons, each --term NPZ contains the oracle's potential
(energy) and virial histories for one interaction term, the matching step
numbers, and the final force array in checkpoint units. Terms are evaluated
at the reference coordinates. Supply all active terms for mixed CPU comparisons.
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


FIELDS = ("position", "velocity", "force", "energy", "virial")
FLOORS = {
    ("liquid", "double"): (1e-9, 1e-8, 1e-6, 2e-5, 2e-5),
    ("liquid", "mixed"): (1e-7, 1e-4, .025, .05, .1),
    ("jac", "double"): (1e-8, 1e-6, 5e-4, 2e-4, 2e-3),
    ("jac", "mixed"): (3e-7, 1e-4, .5, .05, .1),
}


def compare_field(candidate, reference, floor, parts=(), box=None):
    if candidate.shape != reference.shape or reference.size == 0:
        raise ValueError("empty comparison or different array shapes")
    delta = candidate.astype(np.float64) - reference
    if box is not None:
        delta -= box * np.rint(delta / box)
    error = np.abs(delta)
    scale = np.zeros_like(error)
    total = np.zeros_like(error) if parts else None
    for part in parts:
        if part.shape != reference.shape:
            raise ValueError("interaction component shape differs from reference")
        total += part
        scale += np.abs(part.astype(np.float64))
    bound = np.maximum(floor, (4 * 2**-23) * scale)
    if not (np.isfinite(error).all() and np.isfinite(bound).all()):
        raise ValueError("nonfinite comparison")
    if parts and np.any(np.abs(total - reference) > bound):
        raise ValueError("reference contributions do not reproduce the observable")
    ratio = error / bound
    index = np.unravel_index(np.argmax(ratio), ratio.shape)
    return dict(max_abs=float(error.max()), index=list(map(int, index)),
                reference=float(reference[index]), difference=float(error[index]),
                bound=float(bound[index]), ratio=float(ratio[index]),
                passed=bool(ratio[index] <= 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--system", choices=("liquid", "jac"), required=True)
    parser.add_argument("--precision", choices=("mixed", "double"), required=True)
    parser.add_argument("--reference-backend", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--term", action="append", type=Path, default=[])
    args = parser.parse_args()
    relative = args.precision == "mixed" and args.reference_backend == "cpu"
    if bool(args.term) != relative:
        parser.error("supply the reference interaction terms for mixed CPU comparisons only")
    floors = dict(zip(FIELDS, FLOORS[args.system, args.precision]))
    energy, oracle = (np.loadtxt(p / "energy.dat", ndmin=2)
                      for p in (args.candidate, args.reference))
    if not np.array_equal(energy[:, 0], oracle[:, 0]):
        raise ValueError("energy histories have different steps")
    components = {name: [] for name in ("force", "energy", "virial")}
    for path in args.term:
        with np.load(path, allow_pickle=False) as term:
            if not np.array_equal(term["step"], oracle[:, 0]):
                raise ValueError(f"{path}: component history has different steps")
            for name in components:
                components[name].append(term[name])
    results = {}
    with h5py.File(args.candidate / "state.h5") as a, h5py.File(args.reference / "state.h5") as b:
        group = "particles/all/"
        box = b[group + "box/edges"][:]
        if not np.array_equal(box, a[group + "box/edges"][:]) or not np.all(box > 0):
            raise ValueError("periodic boxes must match and have positive lengths")
        for field in FIELDS[:3]:
            key = group + field
            for checkpoint in (a, b):
                if (not np.array_equal(checkpoint[key + "/step"][:], oracle[-1:, 0])
                        or checkpoint[key + "/value"].shape[0] != 1):
                    raise ValueError(f"{field}: expected a single final-step checkpoint")
            x, y = a[key + "/value"][0], b[key + "/value"][0]
            parts = components["force"] if field == "force" else ([y] if relative else [])
            results[field] = compare_field(x, y, floors[field], parts,
                                           box if field == "position" else None)
    for field, col in (("energy", 3), ("virial", 6)):
        results[field] = compare_field(energy[:, col], oracle[:, col], floors[field],
                                       components[field])
    print(json.dumps(results, sort_keys=True))
    return 0 if all(row["passed"] for row in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
