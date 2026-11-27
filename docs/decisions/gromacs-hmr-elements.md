# D[gromacs-hmr-elements]

Recorded 2026-11-27.

**`mdir check` reports the elements of a topology, where each comes from,
and the constraints by kind, and warns where other evidence contradicts an
element; the readers and `mdir run` are unchanged.** Issue #469, the
maintainer's ruling of 2026-10-21 (items 2 and 3; item 1, `bench.py
prepare`, was #697).

A topology of GROMACS converted from a prmtop with repartitioned masses
of hydrogen (3.024 amu) but without `ATOMIC_NUMBER` carries at.num guessed
from the masses: H as He, CT as Li or B, N as C. MDIR takes a hydrogen by
its atomic number, so `hydrogen_bonds = true` constrains none of these
bonds, and a run at 4 fs fails (STMV failed SETTLE at step 851). Nothing
said so before the run.

*The report.* Each particle's element has a source, recorded by its
reader: `file` (Amber's `ATOMIC_NUMBER`, the at.num column of
`[ atomtypes ]`, CHARMM's element of `MASS`), `name_and_mass` (Amber
without `ATOMIC_NUMBER`), `mass` (CHARMM without an element), or `none`
(GROMACS without the at.num column, where the run takes a hydrogen as
below). `mdir check` prints the count of each element and of
each source, with the count of fallbacks (all but `file`), and the
constraints by kind: the bonds of hydrogen of SHAKE in their groups, and
the rigid waters of SETTLE. The JSON report has them in
`system.topology.elements` and `run.constraints`. A protein whose bonds
of hydrogen are not constrained is visible as 0 before a run.

*The warnings.* One for each atom type and kind, at most 8 of a kind in
the text:

- `element_group_mass`: repartitioning keeps the mass of a heavy atom and
  its hydrogens, so the group's mass is compared with the standard atomic
  weights of its elements (IUPAC 2021); a difference over 0.1 amu is
  flagged, except the one to three hydrogens of a united atom of C, N, O,
  or S. It flags H read as He (−0.979 amu) and C read as B or Li.
- `element_light_atom`: a particle with one bond, below 4.1 amu, that the
  run does not take for a hydrogen (the bound of the Amber reader's
  inference); not the waters of SETTLE.
- `element_type_name`: an element given by the file that is neither symbol
  the type name begins with (one or two letters, after digits; the bonded
  type of GROMACS), nor that of a known ion type (CHARMM's `SOD`, `POT`,
  `CLA`, ...; Amber's `C0`). It flags N read as C, which the mass of
  11.994 amu does not tell.

Each names the type and where its element is corrected. The exit status
and the run are unchanged.

*The bonds of hydrogen without the at.num column.* A GROMACS topology
without the at.num column gives no atomic number. The reader took a
hydrogen there by a mass below 1.2 amu, which a repartitioned hydrogen is
not, so its bond was left free, with no wrong number in the file to warn
about. It now also takes a particle named with an H first and below 4.1
amu (`isHydrogenWithoutNumber`, `include/mdir/Driver/Topology.h`). grompp
takes the hydrogens of `constraints = h-bonds` by the first letter of the
name alone; the bound of 4.1 amu, that of the Amber reader's inference,
keeps a heavy atom named H (mercury, HG) out. The atomic numbers stay
absent; only the bonds of hydrogen and their constraints change, and only
for particles named H of 1.2 to 4.1 amu: repartitioned hydrogens and
deuterium. This changes the results of such inputs: their bonds of
hydrogen are constrained under `hydrogen_bonds = true`, as grompp
constrains them; an input with the at.num column, or without it and
without such particles, runs as before. It departs from the ruling, under which `mdir run`
keeps "constraining ... every topology exactly as [it does] today"; the PR
asks the maintainer to confirm it, and it is a commit of its own.

*Checked.* `check-elements.test` generates from the dipeptide of
`gromacs-dipeptide.test` (`Inputs/hmr_topology.py`) the repartitioned
topology with the at.num given, guessed from the masses as ParmEd 4.3.1
writes them from a prmtop without `ATOMIC_NUMBER` (the same at.num for
every type), and absent, and one with type names CA and C0 of carbon; and
an Amber topology without `ATOMIC_NUMBER`. Without the at.num column, 12
bonds of hydrogen and 2343 degrees of freedom, as grompp gives (main: 0
and 2355); with the first hydrogen named X1, 11 and 2344, as grompp
gives, and the warning `element_light_atom` for it. With the at.num given, the 12
bonds of hydrogen and the 382 waters are constrained, 2343 degrees of
freedom, as grompp of GROMACS 2026.3 gives for the same system, and no
warning. With them guessed, 0 bonds are constrained, 2355 degrees of
freedom (as on main), and the three kinds of warning name every wrong
type. On the systems of the Amber suite, read from the prmtop and from the
GROMACS topology of `bench.py prepare`, the degrees of freedom and the
constraints are those of main; the files written before #697 (JAC at 4 fs,
STMV) get the warnings, and the others none.

*Not taken.* A refusal, or a correction of the readers' atomic numbers
from these warnings: the ruling keeps every input running as before. A
name rule limited to hydrogens: it would not flag N read as C.
