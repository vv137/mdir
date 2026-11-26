"""Check cancellation, absolute floors, and per-frame comparison."""
import sys

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, sys.argv[1])
from compare import compare_field

reference = np.array([-1202027.032770, 0.0])
parts = [np.array([-4297730.897035, 1.0]), np.array([3095703.864428, -1.0])]
candidate = reference + [-.521918, .099]
assert compare_field(candidate, reference, .1, parts)["passed"]
assert not compare_field(candidate, reference, .1)["passed"]
candidate[1] = .101
failed = compare_field(candidate, reference, .1, parts)
assert not failed["passed"] and failed["index"] == [1]
assert failed["bound"] == .1 and failed["max_abs"] > .52

try:
    compare_field(candidate, reference, .1, parts[:1])
except ValueError as error:
    assert "do not reproduce" in str(error)
else:
    raise AssertionError("incomplete reference decomposition passed")
