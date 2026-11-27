"""Gate two mixed RF histories at 2 fs against frozen ordinary reference runs.

Each history contains the initial frame and 204 steps. The frozen data give
separate potential and virial bounds derived across ordinary MATRIX orders.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


QUANTITIES = ("potential", "virial")
REFERENCE_HASHES = {
    "factorix": (
        "4b1fc3c2f3b305d4c8778682d38277a77ff28950c55bb68dda33a21bc141701c",
        "5931e49908ee06df84d99bb11de467b91103c0a6352d479b898c23d24a5d7bfe"),
    "jac-rf": (
        "72625ddfdf5c643df5268555d9dab40a22cf3c04db33d542027843f4caad0743",
        "bd085fe9b5c7f7a19fddb8fbd8d90ed043860734bb1313d067c6b6a40f8816e1"),
}


def trajectory(path):
    values = np.loadtxt(path, ndmin=2)
    steps = np.arange(205)
    if (values.shape != (205, 8) or not np.isfinite(values).all()
            or not np.array_equal(values[:, 0], steps)
            or not np.allclose(values[:, 1], steps / 500, rtol=0, atol=1e-10)):
        raise ValueError(f"{path}: expected frames 0 through 204 at 2 fs with eight finite columns")
    return values[:, [3, 6]]


def check(reference, candidate, bounds):
    error = np.abs(candidate - reference)
    peaks = {}
    for i, name in enumerate(QUANTITIES):
        diff = error[:, :, i]
        repetition, frame = np.unravel_index(np.argmax(diff), diff.shape)
        peak = float(diff[repetition, frame])
        bound = bounds[name]
        peaks[name] = dict(repetition=int(repetition + 1), frame=int(frame),
                           max_abs=peak, bound=bound, passed=peak <= bound)
    return dict(quantities=peaks, passed=all(item["passed"] for item in peaks.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, nargs=2)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--system", choices=REFERENCE_HASHES)
    mode.add_argument("--calibration", type=Path)
    parser.add_argument("candidate", type=Path, nargs=2)
    args = parser.parse_args()
    paths = [p.resolve() for p in [*(args.reference or []), *args.candidate]]
    if len(set(paths)) != len(paths):
        parser.error("each reference and candidate repetition needs a distinct file")
    if args.calibration and not args.reference:
        parser.error("--calibration requires two --reference files")

    if args.system:
        calibration = Path(__file__).with_name(f"{args.system}-mixed-orders.json")
        if args.reference:
            actual = tuple(hashlib.sha256(path.read_bytes()).hexdigest() for path in args.reference)
            if actual != REFERENCE_HASHES[args.system]:
                parser.error("the two reference files must match the frozen ordinary pair in order")
    else:
        calibration = args.calibration
    quantities = json.loads(calibration.read_text())["quantities"]
    bounds = {name: float(quantities[name]["bound"]) for name in QUANTITIES}
    if any(not np.isfinite(bound) or bound <= 0 for bound in bounds.values()):
        raise ValueError(f"{calibration}: expected positive finite bounds")

    if args.reference:
        reference = np.stack([trajectory(path) for path in args.reference])
    else:
        path = Path(__file__).with_name(f"{args.system}-ordinary-mixed.dat")
        frames = np.loadtxt(path)
        if (frames.shape != (205, 5) or not np.isfinite(frames).all()
                or not np.array_equal(frames[:, 0], np.arange(205))):
            raise ValueError(f"{path}: invalid ordinary trajectory pair")
        reference = np.stack((frames[:, 1:3], frames[:, 3:5]))
    result = check(reference, np.stack([trajectory(path) for path in args.candidate]), bounds)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
