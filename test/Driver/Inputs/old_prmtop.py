"""Writes three topologies of Amber from one, each with the same dihedral of
two terms, the second term (k = 50 kcal/mol, n = 2, phase π) added to the
second dihedral without hydrogen (C-N-CA-CB of the dipeptide, at 60°, which
carries a 1-4 pair):

  chained.prmtop  the term given by the convention of a negative
                  periodicity: the first term's type, its periodicity
                  negated and its SCEE 99, followed by the second term's
                  type, whose phase is written 3.141594, as older files
                  write π, and whose SCEE is the first term's;
  old.prmtop      the same in the format before Amber 7, without %FLAG
                  lines, without the sections that format lacks;
  explicit.prmtop the second term as an entry of its own, whose negative
                  third index leaves its 1-4 pair out, with the phase π to
                  the digits of the format.

The three agree if a phase within 1e-3 of π is π and a chain takes the
factors of its 1-4 pair from its last type.

    old_prmtop.py topology.prmtop DIRECTORY"""

import math
import os
import sys

lines = open(sys.argv[1]).read().split("\n")
out = sys.argv[2]
order, sections, formats, flag = [], {}, {}, None
for line in lines:
    if line.startswith("%FLAG"):
        flag = line.split()[1]
        order.append(flag)
        sections[flag] = []
    elif line.startswith("%FORMAT"):
        formats[flag] = line[len("%FORMAT("):].rstrip().rstrip(")")
    elif flag and not line.startswith("%"):
        sections[flag].append(line)


def width(form):
    letters = form.lstrip("0123456789").upper()
    return int(letters[1:].split(".")[0])


def values(flag):
    w = width(formats[flag])
    text = "".join(line.ljust(len(line) + (-len(line)) % w) for line in sections[flag])
    items = [text[i:i + w] for i in range(0, len(text), w)]
    if formats[flag].upper().lstrip("0123456789").startswith("A"):
        return [v for v in items]
    return [v for v in items if v.strip()]


data = {f: values(f) for f in order}
ints = lambda f: [int(v) for v in data[f]]
reals = lambda f: [float(v) for v in data[f]]
p = ints("POINTERS")
k, n, phase = (reals(f) for f in ("DIHEDRAL_FORCE_CONSTANT",
                                  "DIHEDRAL_PERIODICITY", "DIHEDRAL_PHASE"))
scee, scnb = reals("SCEE_SCALE_FACTOR"), reals("SCNB_SCALE_FACTOR")
dihedrals = ints("DIHEDRALS_WITHOUT_HYDROGEN")
ENTRY = 5 * 1
t = dihedrals[ENTRY + 4]


def write_new(path, p, k, n, phase, scee, scnb, dihedrals):
    new = dict(data)
    new["POINTERS"] = p
    new["DIHEDRAL_FORCE_CONSTANT"], new["DIHEDRAL_PERIODICITY"] = k, n
    new["DIHEDRAL_PHASE"] = phase
    new["SCEE_SCALE_FACTOR"], new["SCNB_SCALE_FACTOR"] = scee, scnb
    new["DIHEDRALS_WITHOUT_HYDROGEN"] = dihedrals
    with open(path, "w") as f:
        f.write(lines[0] + "\n")
        for flag in order:
            f.write(f"%FLAG {flag}\n%FORMAT({formats[flag]})\n")
            f.write(block(new[flag], formats[flag]))


def block(items, form):
    letters = form.lstrip("0123456789").upper()
    count = int(form[:len(form) - len(letters)])
    w = width(form)
    if letters.startswith("A"):
        cells = [str(v).ljust(w)[:w] for v in items]
    elif letters.startswith("I"):
        cells = [f"{int(v):{w}d}" for v in items]
    else:
        cells = [f"{float(v):{w}.8E}" for v in items]
    rows = ["".join(cells[i:i + count]) for i in range(0, len(cells), count)]
    return "\n".join(rows or [""]) + "\n"


# Chained: the entry takes a copy of its type with the periodicity negated,
# followed by the second term.
pc = list(p)
pc[17] += 2
kc, nc = k + [k[t - 1], 50.0], n + [-n[t - 1], 2.0]
phc = phase + [phase[t - 1], 3.141594]
sc, sn = scee + [99.0, scee[t - 1]], scnb + [scnb[t - 1]] * 2
dc = list(dihedrals)
dc[ENTRY + 4] = len(k) + 1
write_new(os.path.join(out, "chained.prmtop"), pc, kc, nc, phc, sc, sn, dc)

# Explicit: the second term as its own entry, without a 1-4 pair.
pe = list(p)
pe[17] += 1
pe[7] += 1
pe[14] += 1
de = list(dihedrals) + [dihedrals[ENTRY], dihedrals[ENTRY + 1],
                        -abs(dihedrals[ENTRY + 2]), dihedrals[ENTRY + 3],
                        len(k) + 1]
write_new(os.path.join(out, "explicit.prmtop"), pe, k + [50.0], n + [2.0],
          phase + [math.pi], scee + [scee[t - 1]], scnb + [scnb[t - 1]], de)

# The format before Amber 7: a title, 30 pointers in lines of 12, and the
# sections in their fixed order, from the chained topology.
old = dict(data)
old.update({"DIHEDRAL_FORCE_CONSTANT": kc, "DIHEDRAL_PERIODICITY": nc,
            "DIHEDRAL_PHASE": phc, "DIHEDRALS_WITHOUT_HYDROGEN": dc})
sequence = [("ATOM_NAME", "20a4"), ("CHARGE", "5E16.8"), ("MASS", "5E16.8"),
            ("ATOM_TYPE_INDEX", "12I6"), ("NUMBER_EXCLUDED_ATOMS", "12I6"),
            ("NONBONDED_PARM_INDEX", "12I6"), ("RESIDUE_LABEL", "20a4"),
            ("RESIDUE_POINTER", "12I6"), ("BOND_FORCE_CONSTANT", "5E16.8"),
            ("BOND_EQUIL_VALUE", "5E16.8"), ("ANGLE_FORCE_CONSTANT", "5E16.8"),
            ("ANGLE_EQUIL_VALUE", "5E16.8"),
            ("DIHEDRAL_FORCE_CONSTANT", "5E16.8"),
            ("DIHEDRAL_PERIODICITY", "5E16.8"), ("DIHEDRAL_PHASE", "5E16.8"),
            ("SOLTY", "5E16.8"), ("LENNARD_JONES_ACOEF", "5E16.8"),
            ("LENNARD_JONES_BCOEF", "5E16.8"), ("BONDS_INC_HYDROGEN", "12I6"),
            ("BONDS_WITHOUT_HYDROGEN", "12I6"),
            ("ANGLES_INC_HYDROGEN", "12I6"),
            ("ANGLES_WITHOUT_HYDROGEN", "12I6"),
            ("DIHEDRALS_INC_HYDROGEN", "12I6"),
            ("DIHEDRALS_WITHOUT_HYDROGEN", "12I6"),
            ("EXCLUDED_ATOMS_LIST", "12I6"), ("HBOND_ACOEF", "5E16.8"),
            ("HBOND_BCOEF", "5E16.8"), ("HBCUT", "5E16.8"),
            ("AMBER_ATOM_TYPE", "20a4"),
            ("TREE_CHAIN_CLASSIFICATION", "20a4"), ("JOIN_ARRAY", "12I6"),
            ("IROTAT", "12I6"), ("SOLVENT_POINTERS", "12I6"),
            ("ATOMS_PER_MOLECULE", "12I6"), ("BOX_DIMENSIONS", "5E16.8")]
counts = {"ATOM_NAME": p[0], "RESIDUE_LABEL": p[11], "AMBER_ATOM_TYPE": p[0],
          "TREE_CHAIN_CLASSIFICATION": p[0]}
with open(os.path.join(out, "old.prmtop"), "w") as f:
    f.write("dipeptide in the format before Amber 7\n")
    f.write(block(pc[:30], "12I6"))
    for flag, form in sequence:
        items = old[flag]
        if flag in counts:
            items = items[:counts[flag]]
        f.write(block(items, form))
