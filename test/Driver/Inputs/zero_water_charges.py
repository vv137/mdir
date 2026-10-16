"""Writes an Amber topology whose every other water (residue WAT) has no
charge on any of its sites, for a test of the reciprocal sum over the
particles of a nonzero charge (D262) where at most half the particles are
listed. The other sections are copied as they are.

    python zero_water_charges.py in.prmtop out.prmtop
"""

import sys


def sections(lines):
    """The lines of each %FLAG, by name, with its %FORMAT line."""
    found, name = {}, None
    for number, line in enumerate(lines):
        if line.startswith('%FLAG'):
            name = line.split()[1]
            found[name] = [number + 1, number + 1]
        elif name is not None:
            found[name][1] = number + 1
    return found


def read(lines, span, width, convert):
    start, end = span
    values = []
    for line in lines[start + 1:end]:
        text = line.rstrip('\n')
        values += [convert(text[k:k + width]) for k in range(0, len(text), width)
                   if text[k:k + width].strip()]
    return values


def main():
    source, target = sys.argv[1], sys.argv[2]
    with open(source) as file:
        lines = file.readlines()
    found = sections(lines)
    labels = read(lines, found['RESIDUE_LABEL'], 4, str.strip)
    pointers = read(lines, found['RESIDUE_POINTER'], 8, int)
    charges = read(lines, found['CHARGE'], 16, float)
    ends = pointers[1:] + [len(charges) + 1]
    waters = 0
    for label, first, after in zip(labels, pointers, ends):
        if label != 'WAT':
            continue
        if waters % 2 == 0:
            for atom in range(first - 1, after - 1):
                charges[atom] = 0.0
        waters += 1
    start, end = found['CHARGE']
    text = [''.join('%16.8E' % q for q in charges[k:k + 5]) + '\n'
            for k in range(0, len(charges), 5)]
    lines[start + 1:end] = text
    with open(target, 'w') as file:
        file.writelines(lines)
    listed = sum(1 for q in charges if q != 0.0)
    print(f'{listed} of {len(charges)} particles charged')


main()
