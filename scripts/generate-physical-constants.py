#!/usr/bin/env python3
"""Generate the shared Python constants by compiling the public C++ header.

The checked-in module serves standalone scripts. CMake copies the same
module into mdir._physical_constants for Python package consumers. No
physical formula or header-expression parser is maintained in Python.
"""
import argparse
import hashlib
from pathlib import Path
import subprocess
import tempfile


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", default="c++", help="C++17 compiler executable")
    parser.add_argument("--output", type=Path,
                        default=root / "scripts/mdir_physical_constants.py")
    parser.add_argument("--scratch", type=Path, help="parent of the private build directory")
    parser.add_argument("--check", action="store_true", help="fail if the module is stale")
    args = parser.parse_args()
    header = root / "include/mdir/Support/PhysicalConstants.h"
    digest = hashlib.sha256(header.read_bytes()).hexdigest()
    with tempfile.TemporaryDirectory(prefix="mdir-physical-constants-", dir=args.scratch) as work:
        executable = Path(work) / "generate-physical-constants"
        subprocess.run([args.cxx, "-std=c++17", "-I", str(root / "include"),
                        str(root / "scripts/generate-physical-constants.cpp"),
                        "-o", str(executable)], check=True)
        values = subprocess.check_output([str(executable)], text=True)
    text = ('"""Generated from include/mdir/Support/PhysicalConstants.h; do not edit.\n\n'
            'Regenerate with scripts/generate-physical-constants.py.\n'
            'Standalone scripts and mdir._physical_constants use this same module.\n'
            '"""\n'
            f'_HEADER_SHA256 = "{digest}"\n\n' + values)
    current = args.output.read_text() if args.output.exists() else None
    if args.check:
        if current != text:
            parser.exit(1, "generated physical constants are stale; regenerate the module\n")
        print("generated physical constants: current")
    elif current != text:
        args.output.write_text(text)


if __name__ == "__main__":
    main()
