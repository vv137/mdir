"""The energies, forces, virials, and descriptors of a NEP model from
calorine's CPUNEP, in double precision (Lindgren et al., J. Open Source
Softw. 9, 6264, 2024), for docs/nep.md (D356, #583).

Usage: calorine.py MODEL XYZ NFRAMES OUT.npz
Reads the first NFRAMES frames of an extended XYZ file. The virial is
-stress * volume in eV, as GPUMD writes it.
"""
import sys

import numpy as np
from ase.io import read
from calorine.calculators import CPUNEP

model, xyz, nframes, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
frames = read(xyz, index=f":{nframes}")
res = {}
for f, atoms in enumerate(frames):
    calc = CPUNEP(model)
    atoms.calc = calc
    res[f"pos{f}"] = atoms.get_positions()
    res[f"cell{f}"] = np.array(atoms.get_cell())
    res[f"sym{f}"] = np.array(atoms.get_chemical_symbols())
    res[f"energy{f}"] = atoms.get_potential_energy()
    res[f"forces{f}"] = atoms.get_forces()
    res[f"virial{f}"] = -atoms.get_stress(voigt=False) * atoms.get_volume()
    res[f"desc{f}"] = calc.get_descriptors()
np.savez(out, nframes=len(frames), **res)
print(f"{len(frames)} frames of {len(frames[0])} atoms")
