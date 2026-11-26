"""Gate two mixed 2 fs energy histories against ordinary repetitions.

Use --system for the calibrated protein gate with its recorded ordinary
histories. To calibrate another input, supply two ordinary histories with
--reference and 19 ordinary pairs with --control. Each history contains
the initial frame and 204 steps.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


QUANTITIES = ("potential", "virial")
CALIBRATIONS = {
    "factorix": (2.0520896338131216, (
        "4b1fc3c2f3b305d4c8778682d38277a77ff28950c55bb68dda33a21bc141701c",
        "5931e49908ee06df84d99bb11de467b91103c0a6352d479b898c23d24a5d7bfe")),
    "jac-rf": (1.7427031882513417, (
        "72625ddfdf5c643df5268555d9dab40a22cf3c04db33d542027843f4caad0743",
        "bd085fe9b5c7f7a19fddb8fbd8d90ed043860734bb1313d067c6b6a40f8816e1")),
}


def trajectory(path):
    values = np.loadtxt(path, ndmin=2)
    steps = np.arange(205)
    if (values.shape != (205, 8) or not np.isfinite(values).all()
            or not np.array_equal(values[:, 0], steps)
            or not np.allclose(values[:, 1], steps / 500, rtol=0, atol=1e-10)):
        raise ValueError(f"{path}: expected frames 0 through 204 at 2 fs with eight finite columns")
    return values[:, [3, 6]]


def ordinary_spread(reference):
    spread = np.abs(reference[0] - reference[1]).max(axis=0)
    if np.any(spread == 0):
        raise ValueError("a zero ordinary spread cannot calibrate a relative trajectory bound")
    return spread


def evaluate(reference, controls, candidate):
    if len(controls) != 19:
        raise ValueError("expected 19 ordinary calibration pairs")
    spread = ordinary_spread(reference)
    scores = [float((np.abs(pair - reference) / spread).max()) for pair in controls]
    return dict(check(reference, candidate, max(scores)), control_scores=scores)


def check(reference, candidate, multiplier):
    spread = ordinary_spread(reference)
    bounds = spread * multiplier
    error = np.abs(candidate - reference)
    peaks = {}
    for i, name in enumerate(QUANTITIES):
        diff = error[:, :, i]
        repetition, frame = np.unravel_index(np.argmax(diff), diff.shape)
        peak = float(diff[repetition, frame])
        peaks[name] = dict(repetition=int(repetition + 1), frame=int(frame),
                           max_abs=peak, ordinary_spread=float(spread[i]),
                           bound=float(bounds[i]), passed=bool(peak / spread[i] <= multiplier))
    return dict(multiplier=multiplier, quantities=peaks,
                passed=all(item["passed"] for item in peaks.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, nargs=2, default=[])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--control", type=Path, nargs=2, action="append", default=[])
    mode.add_argument("--system", choices=CALIBRATIONS)
    parser.add_argument("candidate", type=Path, nargs=2)
    args = parser.parse_args()
    paths = [p.resolve() for pair in [args.reference, *args.control, args.candidate] for p in pair]
    if len(set(paths)) != len(paths):
        parser.error("each reference, control and candidate repetition needs a distinct file")

    def read_pair(paths):
        return np.stack([trajectory(path) for path in paths])

    if args.system:
        multiplier, expected = CALIBRATIONS[args.system]
        if args.reference:
            actual = tuple(hashlib.sha256(path.read_bytes()).hexdigest() for path in args.reference)
            if actual != expected:
                parser.error("the two reference files must match the frozen ordinary pair in order")
            reference = read_pair(args.reference)
        else:
            path = Path(__file__).with_name(f"{args.system}-ordinary-mixed.dat")
            frames = np.loadtxt(path)
            if (frames.shape != (205, 5) or not np.isfinite(frames).all()
                    or not np.array_equal(frames[:, 0], np.arange(205))):
                raise ValueError(f"{path}: invalid ordinary trajectory pair")
            reference = np.stack((frames[:, 1:3], frames[:, 3:5]))
        result = check(reference, read_pair(args.candidate), multiplier)
    else:
        if not args.reference:
            parser.error("--control requires two --reference files")
        result = evaluate(read_pair(args.reference), [read_pair(pair) for pair in args.control],
                          read_pair(args.candidate))
    print(json.dumps(result, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
