"""Compares the terms that `mdir run` writes at the start with reference
values in kJ/mol, and prints one line for each term: the name and 1 if the
two agree to the relative tolerance, or 0 if they do not.

Usage: mdir run control.toml | python3 check_terms.py reference tolerance
           [absolute]

A term agrees if it is within the relative tolerance, or within the
absolute tolerance in kJ/mol if one is given, for terms near zero.
"""
import re
import sys

KCAL = 4.184


def main():
    reference = {}
    for line in open(sys.argv[1]):
        line = line.split("#")[0].strip()
        if not line:
            continue
        name, value = line.rsplit(None, 1)
        reference[name] = float(value)
    tolerance = float(sys.argv[2])
    absolute = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0

    found = {}
    for line in sys.stdin:
        match = re.match(r"MDIR:   (.+?)\s+(-?[0-9.]+)$", line.rstrip())
        if match:
            found[match.group(1)] = float(match.group(2)) * KCAL
    for name, value in reference.items():
        mine = found.get(name)
        agrees = mine is not None and abs(mine - value) <= max(
            tolerance * abs(value), absolute)
        print("%s %d" % (name, 1 if agrees else 0))


if __name__ == "__main__":
    main()
