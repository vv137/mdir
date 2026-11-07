### Added

- **Linear virtual sites in the products of force matching** (D[force-matching-sites], #771 part 2): `FrameEnergies.vjp` gives the product in the tunables (D381) and the derivative in the positions (D383) for programs with linear sites (`[ virtual_sites3 ]`, function 1, of GROMACS, as in its OPC and TIP4P), placing the tangent and spreading the forces to the atoms; the extra points of Amber, placed by the directions of their bonds, stay refused by name ([docs/python-frames.md](docs/python-frames.md)).
