"""Variants of a GROMACS topology with repartitioned hydrogen masses (#469).

usage: hmr_topology.py INPUT.top OUTPUT_DIR

Reads a topology whose [ atomtypes ] give atomic numbers (the dipeptide of
Inputs/dipeptide, as ParmEd writes it) and writes into OUTPUT_DIR:

  given.top    each hydrogen 3.024 amu, each heavy atom lighter by 2.016 amu
               for each hydrogen bonded to it (waters included), the at.num
               column unchanged;
  guessed.top  the same masses, and the at.num of each type guessed from the
               mass of its first atom, the nearest standard atomic weight, as
               a conversion from a prmtop without ATOMIC_NUMBER wrote them
               (H as He, CT as Li, N as C);
  absent.top   the same masses without the at.num column;
  plain.top    the masses of the input without the at.num column;
  names.top    given.top with the type CX named CA (carbon, whose two letters
               are the symbol of calcium) and a type C0 of carbon (the name
               of Amber's calcium), which agree with their elements.
"""

import re
import sys

WEIGHTS = [(1, 1.008), (2, 4.0026), (3, 6.94), (4, 9.0122), (5, 10.81),
           (6, 12.011), (7, 14.007), (8, 15.999), (9, 18.998), (11, 22.990),
           (15, 30.974), (16, 32.06), (17, 35.45)]


def sections(lines):
    """Yields (section, line) for each line, the section in lower case."""
    section = None
    for line in lines:
        m = re.match(r"\s*\[\s*(\w+)\s*\]", line)
        if m:
            section = m.group(1).lower()
        yield section, line


def fields(line):
    """The fields of a line of data, or none for a header or a directive."""
    f = line.split(";")[0].split()
    return [] if f and (f[0].startswith("[") or f[0].startswith("#")) else f


def main():
    source, out = sys.argv[1], sys.argv[2]
    lines = open(source).read().splitlines()

    numbers = {}
    for section, line in sections(lines):
        f = fields(line)
        if section == "atomtypes" and len(f) >= 7:
            numbers[f[0]] = int(f[1])

    # The atoms and bonds of each molecule type, by the index of the line.
    hydrogen_count = {}  # (molecule, atom number) -> bonded hydrogens
    pairs = set()  # (heavy atom, hydrogen)
    atom_type = {}
    molecule = None
    for section, line in sections(lines):
        f = fields(line)
        if section == "moleculetype" and f:
            molecule = f[0]
        elif section == "atoms" and len(f) >= 8:
            atom_type[(molecule, f[0])] = f[1]
        elif section == "bonds" and len(f) >= 3:
            a, b = (molecule, f[0]), (molecule, f[1])
            for x, y in ((a, b), (b, a)):
                if numbers[atom_type[y]] == 1 and numbers[atom_type[x]] != 1:
                    pairs.add((x, y))
        elif section == "settles" and len(f) >= 4:
            o = int(f[0])
            for h in (o + 1, o + 2):
                pairs.add(((molecule, f[0]), (molecule, str(h))))
    # The water's bonds (with -DFLEXIBLE) and its [ settles ] name the same
    # pairs.
    for x, _ in pairs:
        hydrogen_count[x] = hydrogen_count.get(x, 0) + 1

    def repartitioned():
        molecule = None
        result = []
        for section, line in sections(lines):
            f = fields(line)
            if section == "moleculetype" and f:
                molecule = f[0]
            if section == "atoms" and len(f) >= 8:
                key = (molecule, f[0])
                mass = float(f[7])
                if numbers[f[1]] == 1:
                    mass = 3.024
                else:
                    mass -= 2.016 * hydrogen_count.get(key, 0)
                f[7] = "%.6f" % mass
                line = " ".join(f[:8])
            result.append(line)
        return result

    def first_masses(text):
        masses = {}
        for section, line in sections(text):
            f = fields(line)
            if section == "atoms" and len(f) >= 8:
                masses.setdefault(f[1], float(f[7]))
        return masses

    def with_numbers(text, number_of):
        result = []
        for section, line in sections(text):
            f = fields(line)
            if section == "atomtypes" and len(f) >= 7:
                number = number_of(f[0])
                if number is None:
                    del f[1]
                else:
                    f[1] = str(number)
                line = " ".join(f)
            elif section == "atomtypes" and line.lstrip().startswith(";"):
                line = "; name at.num mass charge ptype sigma epsilon"
            result.append(line)
        return result

    def write(name, text):
        with open("%s/%s" % (out, name), "w") as f:
            f.write("\n".join(text) + "\n")

    hmr = repartitioned()
    write("given.top", hmr)
    masses = first_masses(hmr)
    guess = lambda t: min(WEIGHTS, key=lambda w: abs(w[1] - masses[t]))[0]
    write("guessed.top", with_numbers(hmr, guess))
    write("absent.top", with_numbers(hmr, lambda t: None))
    write("plain.top", with_numbers(lines, lambda t: None))

    # CX renamed CA, and the CT of the methyl of NME named C0, both carbon.
    renamed = []
    for section, line in sections(hmr):
        f = fields(line)
        if section == "atomtypes" and f and f[0] == "CX":
            renamed.append(line.replace("CX", "CA", 1))
            renamed.append(line.replace("CX", "C0", 1))
            continue
        if section == "atoms" and len(f) >= 8:
            if f[1] == "CX":
                f[1] = "CA"
            elif f[1] == "CT" and f[3] == "NME":
                f[1] = "C0"
            line = " ".join(f)
        renamed.append(line)
    write("names.top", renamed)


if __name__ == "__main__":
    main()
