#!/usr/bin/env python3
"""No tracked file but include/mdir/Support/PhysicalConstants.h writes a
physical constant of MDIR (#805, D394).

The header is the one definition; C and C++ include it, and Python imports
scripts/mdir_physical_constants.py, generated from it (as hexadecimal
floats, which no pattern below matches), or mdir._physical_constants, its
copy in the package. The patterns are the leading digits of each defining
constant and of each value derived from them, in both unit systems, and of
the values that MDIR used before D394; a truncation such as 138.935 or
332.0637 is a sample value and passes. Markdown is not checked: the
documents state the values, and docs/conventions.md, Section 1, is their
reference. The constants of other engines (Amber's 332.0522, CHARMM's
332.0716) match no pattern; a script names them as that engine's, next to
their source. ALLOWED lists the other literals that stay, each with its
reason. Prints file:line: number of each other one; exit status 1 if any.
"""
import fnmatch
import subprocess
import sys

PREFIXES = [
    r"138\.93545", r"138\.935458",            # Coulomb, kJ nm mol^-1 e^-2
    r"332\.06371",                            # Coulomb, kcal Å mol^-1 e^-2
    r"1\.380649", r"6\.02214076",             # k, N_A
    r"1\.602176634", r"8\.8541878",           # e, epsilon_0
    r"8\.31446261", r"0\.00831446261",        # k N_A, J and kJ per mol K
    r"1\.98720425", r"0\.00198720425",        # k N_A, kcal per mol K
    r"16\.60539", r"16\.38824", r"69476\.9",  # pressure: bar, atm, per kcal/mol/Å^3
    r"1\.66053906", r"0\.00166053906",        # density of g/mol per nm^3
    r"252912\.1",                             # Debye screening coefficient
]
PATTERN = r"(?<![0-9.])(?:" + "|".join(PREFIXES) + r")[0-9]*"

# (path, or a pattern of paths, the number as written, the reason)
COUPLING = ("the coupling of an IR test, a parameter of the op that the "
            "test and its reference share, not the engine's constant")
ALLOWED = [
    ("test/Conversion/MDExecToGPU/*.mlir", "138.935457644", COUPLING),
    ("test/Dialect/MDExec/Transforms/assign-streams.mlir", "138.935457644", COUPLING),
    ("test/Integration/tables.mlir", "138.935457644", COUPLING),
    ("test/Integration/Inputs/tuples_reference.py", "138.935457644", COUPLING),
    ("test/Integration/Inputs/obc_reference.py", "138.935457644", COUPLING),
    ("test/Driver/charged.toml", "332.0637130741",
     "a pattern of FileCheck that checks the constant MDIR writes"),
    ("scripts/validation/charmm/run.py", "138.935458",
     "the constant of the topology of GROMACS, named as such"),
    ("examples/ala3/run.py", "6.02214076",
     "N_A for the wheel that the example pins if it lacks "
     "mdir._physical_constants; remove it once the pin has the module"),
]
EXCLUDED = [":!*.md", ":!include/mdir/Support/PhysicalConstants.h",
            ":!scripts/check-physical-constants.py"]


def main():
    done = subprocess.run(["git", "grep", "-n", "-I", "-o", "-P", PATTERN,
                           "--", "."] + EXCLUDED,
                          capture_output=True, text=True)
    if done.returncode not in (0, 1):
        sys.stderr.write(done.stderr)
        return 2
    found = 0
    for hit in done.stdout.splitlines():
        path, line, number = hit.split(":", 2)
        if any(fnmatch.fnmatch(path, where) and number == value
               for where, value, _ in ALLOWED):
            continue
        print(f"{path}:{line}: {number}")
        found += 1
    if found:
        print(f"{found} physical constant(s) written outside "
              "include/mdir/Support/PhysicalConstants.h: import them from "
              "scripts/mdir_physical_constants.py (docs/conventions.md, "
              "Section 1)")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
