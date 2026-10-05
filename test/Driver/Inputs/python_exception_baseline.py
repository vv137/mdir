"""Exercise native-runtime import and C++ exceptions without JIT or CUDA."""
import pathlib
import sys
import mdir
for index in range(32):
    try:
        mdir.load_amber(str(pathlib.Path(sys.argv[1]) / "missing.prmtop"),
                        str(pathlib.Path(sys.argv[1]) / "missing.inpcrd"))
    except mdir.InputError:
        pass
    else:
        raise AssertionError(f"missing exception at repetition {index}")
print("native exception baseline: 32 missing-input errors, 0 failures")
