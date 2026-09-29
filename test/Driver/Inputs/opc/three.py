"""Writes three.prmtop and three.inpcrd: the three waters of opc.prmtop
nearest to the first, centered in a cell of 40 Å, with ParmEd of
AmberTools 26:

    python three.py

Far from their images and within the cutoff of each other, no pair crosses
the cutoff, so the energy is conserved as the time step allows."""

import numpy as np
import parmed

system = parmed.load_file("opc.prmtop", "opc.inpcrd")
waters = [r for r in system.residues if r.name == "WAT"]
oxygens = np.array(
    [[a.xx, a.xy, a.xz] for r in waters for a in r.atoms if a.name == "O"]
)
distances = np.linalg.norm(oxygens - oxygens[0], axis=1)
keep = [waters[i].idx for i in np.argsort(distances)[:3]]
system.strip("!:" + ",".join(str(k + 1) for k in keep))
system.box = [40.0, 40.0, 40.0, 90.0, 90.0, 90.0]
system.coordinates = system.coordinates - system.coordinates.mean(axis=0) + 20.0
system.save("three.prmtop", overwrite=True)
system.save("three.inpcrd", overwrite=True)
