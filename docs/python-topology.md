# Topology views and mask selection in the Python model (D[python-topology])

Issue #120, an M2a item after reporters (D207, #109).
Status: implemented; the choices put to the maintainer on the pull request
are marked below.

The Python model loads a topology (D191) and compiles it (D192), but before
this item it exposed little of it: a Python user could not list residues or
bonds, read masses, or find the particles that an Amber mask selects. Group
temperatures, centers of mass, and analysis need masses and residue
membership, and a restraint mask (D198) could not be checked before a run.

## Interface

```python
loaded = mdir.load_amber("system.prmtop", "system.inpcrd")
top = loaded.topology                  # mdir.Topology, a read-only copy
top.particle_count, top.residue_count
top.masses                             # (N,) float64, amu
top.select("!:WAT & !@H*")            # (k,) int64, from 0, ascending
system = loaded.make_system()
system.topology                        # the same view of a System
program = mdir.compile(system, state, integrator, ensemble, execution, schedule)
program.topology.constraints           # (n, 2) int64, after preparation
program.topology.rigid_waters          # (n, 3) int64
```

`LoadedData.topology`, `System.topology`, and `Program.topology` each return
an `mdir.Topology`. The view holds a copy of the topology taken when the
attribute is read, without positions or velocities, which belong to the
initial state (D191); every attribute returns a new read-only NumPy array or
a new list of strings, following D193. A view is not affected by anything
done later to the object it came from (a later setting of the system, a
recompile, an update of tunable values in a simulation), and changing an array copied out of it
(after `np.array(...)`) changes neither the view nor the model. Imported
topology stays immutable (D192): the view has no setters, and building or
editing a topology from Python is outside this item. Each read of
`.topology` copies the whole topology, its exclusions and parameter tables
included: hold the view in a variable (`top = system.topology`) rather than
reading `system.topology.masses` in a loop.

### Atoms

| Attribute | Shape and type | Meaning |
|---|---|---|
| `particle_count` | `int` | $N$, the particles, virtual sites included |
| `atom_names` | list of $N$ `str` | As in the file |
| `atomic_numbers` | $(N,)$ int64 | As the file gives them (Amber `ATOMIC_NUMBER`, GROMACS atom types, CHARMM `MASS` lines), or inferred from the mass for an Amber file without them; 0 or −1 where the file gives none |
| `masses` | $(N,)$ float64 | amu; 0 for a virtual site |
| `charges` | $(N,)$ float64 | e (an Amber `CHARGE` divided by 18.2223) |
| `particle_types` | $(N,)$ int64 | The Lennard-Jones type of each particle, from 0 |
| `type_names` | list of `str` | The name of each type |
| `residue_indices` | $(N,)$ int64 | The residue of each particle, from 0 |

These are the names of the read-only arrays that D213 gave `System`
(`System.charges`, `particle_types`, `type_names`, `atom_names`,
`residue_indices`), with the same values.

### Residues

| Attribute | Shape and type | Meaning |
|---|---|---|
| `residue_count` | `int` | $R$ |
| `residue_names` | list of $R$ `str` | One per residue |
| `residue_starts` | $(R,)$ int64 | The first particle of each residue |

`residue_names` here has one entry per residue; `System.residue_names`
(D213) has one per particle and equals
`[top.residue_names[r] for r in top.residue_indices]`. Which of the two the
name should keep is a choice put to the maintainer (below).

### Bonded tuples

| Attribute | Shape and type | Meaning |
|---|---|---|
| `bonds` | $(n, 2)$ int64 | The harmonic bonds |
| `angles` | $(n, 3)$ int64 | The harmonic angles, the vertex in the middle |
| `dihedrals` | $(n, 4)$ int64 | The periodic torsions, proper and improper, one row for each term |
| `improper_dihedrals` | $(n,)$ bool | Whether each row of `dihedrals` is an improper torsion |
| `harmonic_impropers` | $(n, 4)$ int64 | The harmonic impropers of CHARMM and of GROMACS function 2 |
| `virtual_sites` | $(n, 4)$ int64 | Each site and the three particles that place it |

The rows are those of the terms of the potential, in the order of the
reader. A quadruple with a torsion of several terms (a dihedral type of two
terms in CHARMM or GROMACS, the repeated entries of an Amber `prmtop`) has a
row for each term; `np.unique(top.dihedrals, axis=0)` gives the quadruples.
Amber's `prmtop` lists the H–H "bond" of a three-site water among its
bonds, and so does the view. The bonds, angles, and torsions of an Amber
extra point carry no energy and are dropped, as sander drops them;
`virtual_sites` gives each extra point with its owner first.

### Constraints after preparation

| Attribute | Shape and type | Meaning |
|---|---|---|
| `prepared` | `bool` | True for `Program.topology` |
| `constraints` | $(n, 2)$ int64 or `None` | The bonds that SHAKE and RATTLE hold, heavy atom first |
| `rigid_waters` | $(n, 3)$ int64 or `None` | The waters that SETTLE holds rigid, oxygen first |

Which bonds are constrained and which waters are rigid depends on the
settings of the `System` (`rigid_hydrogen_bonds`, `rigid_water`,
`water_residues`, D63) and is resolved when a program is compiled. A view of
`LoadedData` or `System` therefore gives `None` for both, which tells "not
resolved" from "no constraint"; `Program.topology` gives the arrays, empty
where nothing is constrained. The number of degrees of freedom of a group of
particles follows from them (#112): three for each particle with a mass in
the group, less one for each row of `constraints` and three for each row of
`rigid_waters` within it.

`Program.topology` is the topology as it was compiled: a bond that a
constraint holds is not a term of the potential and is not among its
`bonds`, and the bonds and angles of rigid waters are gone. The chemical
bonds are those of `LoadedData.topology` and `System.topology`.

### `select(mask)`

`top.select(mask)` returns the particles that the Amber mask `mask` selects,
an int64 array of indices from 0 in ascending order, read-only. It calls the
parser of the control file, `driver::selectParticles`
(`lib/Driver/Selection.cpp`), which `[[restraints]]`, `[[energy.external]]`,
the groups of pair terms and of centroid terms, the parameters of particles
(D165), and
`couple` of `[free_energy]` use, and `mdir.Restraint` through `compile`; its
forms are listed in `include/mdir/Driver/Selection.h` (`:` residues by
number or name, `@` atoms by number or name, `*` and `?` in names, `!`, `&`,
`|`, and parentheses).

- A mask that selects nothing returns an empty array of shape $(0,)$. It is
  not an error: what is wrong with an empty selection depends on its use, and
  the users of a mask say so as before (a restraint or an external term that
  selects nothing is a warning, `couple` that selects nothing an error).
- A mask that does not parse raises `mdir.InputError` with the control file's
  diagnostic, for example `in the mask ':1-', at 4: expected a number or a
  range such as 1-10, not '1-'`.
- A restraint restrains the particles of its selection that have a mass:
  the particles that `mdir.Restraint(mask, k)` restrains are
  `top.select(mask)` without the virtual sites, `masses > 0`.

## Interoperability (choice put to the maintainer)

The issue leaves open whether the view also iterates in the style of
`openmm.app.Topology` (`atoms()`, `residues()`, `bonds()`) or converts to an
OpenMM `Topology` when OpenMM is importable, without a dependency (as in
D200). Neither is in this item until the maintainer decides.

## Validation

`test/Driver/python-topology.test` (CPU only; no device is touched) checks:

- the atoms, residues, bonds, angles, dihedrals, and impropers of the Amber
  dipeptide, the GROMACS propane and water (with and without `FLEXIBLE`), and
  the CHARMM toy system against parsers of the files written in the test
  itself;
- `select(mask)` against the selections of `mdir run` for the same masks,
  printed by `mdir-model-test --selections` from the control file's path:
  the particles that `[[restraints]]` restrain, those of each
  `[[energy.external]]` term, and those that `couple` decouples, including
  masks that select nothing, and the diagnostic of a mask that does not
  parse;
- the constraints of `Program.topology` against the bonds of hydrogen and
  the water residues of the files;
- dtypes, shapes, read-only flags, fresh copies, and views that do not change
  when the system, the loaded data, or a recompiled program changes.

The views were also compared once, outside the suite, with ParmEd 4.3.1
(`parmed.load_file`, `GromacsTopologyFile`, `CharmmPsfFile`, and
`AmberMask`) on the dipeptide, the four-site OPC water box, the ethanol of
the free-energy tests, the GROMACS propane and water, and the CHARMM toy
system: names, charges and masses (to the bit), atomic numbers, residues,
bonds, angles, torsions, impropers, and the extra points with their owners
are equal, and ten masks select the same particles in each. ParmEd keeps
the 387 bonds of the OPC extra points, which MDIR drops as sander does;
without them the 1182 bonds are equal. ParmEd refuses `!*`, which MDIR's
parser takes as the empty selection.
