# CHARMM force fields in M1

Status: 2026-10-02. What a run of a CHARMM force field needs, what CHARMM
computes where its documentation leaves a choice open, what MDIR does, and
how the two were compared. CHARMM 51b1 served as a reference program only:
its energies were measured, as those of sander and GROMACS are (D58), and
nothing of it is in this repository. The formulas are those of the papers
cited.

## 1. What runs

A CHARMM system runs from its own files, a PSF with the files of its
parameters and a CRD (Section 4), or from a topology of GROMACS: the one
that CHARMM-GUI writes, a port of a force field such as `charmm27.ff` of
GROMACS, or a conversion of a PSF by ParmEd with the correction of
Section 4.

| Term | In the topology of GROMACS | In MDIR |
|---|---|---|
| Bonds | `[ bonds ]`, function 1 | `bonds` |
| Angles with a Urey–Bradley term between the outer atoms | `[ angles ]`, function 5: θ0, kθ, r13, k_UB | `angles` and `Urey-Bradley`, ½ k_UB (r₁₃ − r13)²; a term with k_UB = 0 is dropped |
| Proper dihedrals, several terms on one set of atoms | `[ dihedrals ]`, function 9 | `dihedrals` |
| Harmonic impropers | `[ dihedrals ]`, function 2: ξ0, k | `harmonic impropers`, ½ k (ξ − ξ0)², the difference taken in [−π, π) |
| CMAP | `[ cmaptypes ]`, `[ cmap ]` | `CMAP` [[MacKerell2004]](references.md#mackerell2004) (Section 20 of design-m1.md) |
| Special Lennard-Jones of pairs three bonds apart | `[ pairtypes ]` | `Lennard-Jones 1-4` |
| Pair-specific Lennard-Jones (NBFIX) | `[ nonbond_params ]` | the table of the types |
| TIP3P with Lennard-Jones on its hydrogens | the parameters of the topology | nothing particular |

The control file of a CHARMM36 run:

```toml
[energy]
cutoff                 = 12.0
switch_distance        = 10.0
lennard_jones_modifier = "POWER_FORCE_SWITCH"
dispersion_correction  = "NONE"
electrostatics         = "PME"
```

## 2. What CHARMM computes

The CHARMM36 parameter files give their own nonbonded defaults, the same in
`par_all36m_prot.prm`, `par_all36_lipid.prm`, and `par_all36_na.prm`:
`vfswitch ctonnb 10.0 ctofnb 12.0 cutnb 14.0 e14fac 1.0`. The Lennard-Jones
force is switched off between 10 and 12 Å, there is no correction for the
dispersion beyond the cutoff, and the pairs three bonds apart are not
scaled; some types carry their own parameters for them instead. Amber force
fields are run differently: a plain cutoff of 8 to 10 Å with the
correction for the dispersion, and the pairs three bonds apart scaled
(1/1.2 and 1/2.0). A force field is fit with its truncation, so the
truncation is part of the force field; the reach of 12 Å also makes a
CHARMM system cost (12/9)³ ≈ 2.4 times the pairs of an Amber system at 9 Å.

Two particles at 3 to 12.5 Å in CHARMM 51b1 (ε = 0.238 kcal/mol,
R_min = 3.76 Å, from 10 to 12 Å; `scripts/validation/charmm/run.py`,
phase `pair`) settle what the documentation of CHARMM does not write out:

- **VFSWITCH is the force switch of Steinbach and Brooks**
  [[Steinbach1994]](references.md#steinbach1994), within 2.2 × 10⁻⁹ (the printed
  digits). Each inverse power r⁻ⁿ of A r⁻¹² − B r⁻⁶ is switched on its own:

  $$
  \Phi_n(r) = \begin{cases} r^{-n} - (r_s r_c)^{-n/2} & r \le r_s \\
  k_n \left(r^{-n/2} - r_c^{-n/2}\right)^2 & r_s < r < r_c \end{cases}
  \qquad k_n = \frac{r_c^{n/2}}{r_c^{n/2} - r_s^{n/2}}
  $$

  The force of each power is its own force times a switch that falls
  linearly in r^{n/2} from 1 at r_s to 0 at r_c, and the energy below r_s
  is shifted by a constant. CHARMM applies it to every nonbonded pair,
  those three bonds apart among them, with their own parameters.
- **The force switch of GROMACS is another function.** `vdw-modifier =
  force-switch` adds to the force of each power a cubic polynomial in
  r − r_s that makes the force and its slope vanish at r_c
  [[GromacsManual2025]](references.md#gromacsmanual2025). A rerun of GROMACS 2026.3 on the same two
  particles follows that polynomial; against VFSWITCH it differs by a
  constant below 10 Å (3.4 × 10⁻⁵ of the energy at 3 Å) and by up to 60%
  between 10 and 12 Å. The difference is small per pair but not per
  system: on the system of Section 5 the two switches differ by
  9.23 kcal/mol over the pairs that are not three bonds apart.
- **VSWITCH is the potential switch**
  S = (r_c² − r²)² (r_c² + 2r² − 3r_s²) / (r_c² − r_s²)³, a polynomial in
  r², within 2.8 × 10⁻⁹. It is not the potential switch of GROMACS, a
  polynomial in r (`truncation(switch)` of ops-m0.md, Section 4.8).
- **The Coulomb constant is 332.0716 kcal/mol Å/e²**: unit charges 10 Å
  apart give −33.20716 kcal/mol. MDIR keeps the constant of CODATA 2018,
  138.935457644 kJ/mol nm/e² or 332.0637 kcal/mol Å/e² (GROMACS rounds the
  same value to 138.935458), whatever the format it reads
  ([conventions.md](conventions.md), Section 4): its Coulomb energies are
  2.38 × 10⁻⁵ below CHARMM's, as they are 3.5 × 10⁻⁵ above sander's
  (332.0522).

## 3. The power force switch in MDIR

`lennard_jones_modifier = "POWER_FORCE_SWITCH"` gives the Lennard-Jones of a
topology the switch of Section 2, in the kernel of the pairs, and switches
the pairs three bonds apart as well (D121). It is not a kind of the
`truncation` attribute of the IR (ops-m0.md, Section 4.8): the attribute
applies to any kernel u(r), and this switch is defined only for a sum of
inverse powers, each switched on its own. The driver writes the kernel as
4ε (σ¹² Φ₁₂ − σ⁶ Φ₆) with s = σ/r and the powers of σ, so that no power of
r alone leaves the range of `f32`, and differentiation gives the forces.
`FORCE_SWITCH` stays the polynomial of GROMACS, and `POTENTIAL_SHIFT`
subtracts the value at the cutoff of each power; with a topology, each
modifies the Lennard-Jones only, not the direct sum of PME, and takes
`dispersion_correction = "NONE"`.

## 4. Reading the files of CHARMM, and converting them

```toml
[input]
format      = "CHARMM"               # or from the extension .psf
topology    = "system.psf"
coordinates = "system.crd"
parameters  = ["top_all36_prot.rtf", "par_all36m_prot.prm",
               "toppar_water_ions.str"]

[boundary]
type = "PERIODIC"
box  = [76.0, 40.0, 40.0]            # a CRD has no cell
```

A cell that is not orthorhombic is given by its lengths and angles, as
`crystal define` takes them, `box = [a, b, c, α, β, γ]`; CHARMM keeps it
in a symmetric frame, the rows of $G^{1/2}$ for the metric $G$, and MDIR
rotates the coordinates of the CRD into the lower-triangular frame of
docs/triclinic-m2.md (D123, D127).

`lib/Driver/Charmm.cpp` reads them as the documentation of CHARMM 51b1
describes them (`doc/io.info`, `doc/parmfile.info`, `doc/rtop.info`),
with the conventions that the documentation leaves open settled against
CHARMM's energies (D122):

- **The PSF** of the XPLOR kind, which CHARMM 51 writes by default and
  CHARMM-GUI writes (`PSF EXT CMAP XPLOR`): the atoms (segment, residue,
  names, the name of the type, charge, mass) by fields, the bonds, angles,
  dihedrals, impropers, the excluded pairs of `!NNB`, and the cross-terms
  of CMAP (`!NCRTERM`). A PSF whose types are numbers, whose meaning
  depends on the order the RTF was read in, is refused, as are lone pairs
  (`!NUMLP`) and the Drude model.
- **The CRD**, standard or extended (`count EXT`), by fields; the names of
  the atoms must be those of the PSF.
- **Files of topology, parameters, and streams**, read in the order given;
  what a later file defines replaces what an earlier one did. From an RTF
  only the masses; a PRM by its sections (`ATOMS`, `BONDS`, `ANGLES` with
  Urey–Bradley terms, `DIHEDRALS`, `IMPROPER`, `CMAP`, `NONBONDED` with
  the special 1-4 parameters and `E14FAC` in its options, `NBFIX` with
  optional 1-4 values); a stream file by its `read rtf card` and
  `read para card` blocks, skipping the other commands and stopping at the
  first `return`, which takes the first of two alternatives where CHARMM
  takes the one whose condition holds.
- **Wildcards.** Lines of the same dihedral types that follow one another
  make one set of terms; a later set of those types replaces it whole, and
  a set of the four types takes precedence over the set `X B C X`.
  Impropers try the four types, then `A X X D`, `X B C D`, `X B C X`, and
  `X X C D`, each in either direction; an improper of multiplicity 0 is
  harmonic, of another a periodic term.
- **Exclusions and pairs three bonds apart** from the graph of the bonds,
  as `NBXMOD 5`: the pairs one, two, and three bonds apart are excluded,
  those exactly three apart take their own Lennard-Jones (and NBFIX's 1-4
  values) and the Coulomb term times `E14FAC`, and the power force switch
  switches them too.
- **Units**: the energies of CHARMM are K (x − x0)² for bonds, angles,
  Urey–Bradley terms, and impropers; MDIR takes ½ k (x − x0)² with k = 2K.
  Rigid water is the residues `TIP3` by default, with the length H–H from
  the bond that TIP3P carries for SHAKE.

ParmEd 4 converts a PSF with its parameters to a topology of GROMACS, with
two defects found here:

- **It leaves out the special parameters of the pairs three bonds apart.**
  CP1, CP2, CP3, CT1, CT2, CT2A, CT3, N, NH1, and O have them in
  `par_all36m_prot.prm`. On the system of Section 5, 340 of the 812 pairs
  have such a type, and the Lennard-Jones of the pairs is 167.81 kcal/mol
  with the parameters of the other pairs against 43.90 with their own.
  The validation adds them as `[ pairtypes ]`.
- **Its `.gro` keeps 0.001 nm.** The bonds of 3900 flexible waters lose
  60 kcal/mol to the rounding; the validation writes the coordinates of
  the CRD with seven decimals, which the readers of GROMACS and MDIR both
  take from the width of the fields.

## 5. Validation

`scripts/validation/charmm/run.py` builds Trp-cage (NLYIQWLKDGGPSSGRPPPS)
in CHARMM36m from internal coordinates, packs 3900 TIP3P waters, 6 Na⁺,
and 7 Cl⁻ around it in a cell of 76 × 40 × 40 Å (12,017 particles),
minimizes it in CHARMM, and compares the energy of the written coordinates
term by term: CHARMM 51b1 (PME with κ = 0.34 Å⁻¹, grid 80 × 40 × 40,
order 6, VFSWITCH), GROMACS 2026.3 by a rerun (force-switch), and MDIR on
the CPU in double precision, with each switch. The parameter files are read
from `$CHARMM_TOPPAR`; nothing of CHARMM is copied.

| kcal/mol | CHARMM | MDIR, power force switch | relative | GROMACS | MDIR, force switch | relative |
|---|---|---|---|---|---|---|
| Bonds | 1839.518270 | 1839.518273 | 1.6e-9 | 1839.515117 | 1839.518273 | 1.7e-6 |
| Angles | 1296.684030 | 1296.684026 | −3.1e-9 | | | |
| Urey–Bradley | 4.367951 | 4.367951 | −5.3e-8 | | | |
| Angles and Urey–Bradley | | | | 1301.048031 | 1301.051977 | 3.0e-6 |
| Dihedrals | 183.561754 | 183.561754 | 0 | 183.561817 | 183.561754 | −3.5e-7 |
| Harmonic impropers | 1.246850 | 1.246850 | 4.8e-8 | 1.246834 | 1.246850 | 1.3e-5 |
| CMAP | 4.079922 | 4.079922 | 9.6e-8 | 4.079912 | 4.079922 | 2.5e-6 |
| Lennard-Jones, with 1-4 | 7920.937173 | 7920.935437 | −2.2e-7 | 7911.651418 | 7911.642465 | −1.1e-6 |
| Coulomb, all of it | −61630.970606 | −61629.506317 | 2.4e-5 | −61630.524105 | −61629.506317 | 1.7e-5 |

- The bonded terms agree with CHARMM to its printed digits.
- The Lennard-Jones agrees to the six digits of ε that ParmEd writes into
  the topology. An independent sum over all pairs in Python, with the
  parameters as CHARMM reads them, gives 7920.937170.
- The Coulomb energy differs from CHARMM's by the ratio of the constants,
  2.38 × 10⁻⁵ (Section 2); with it, by 2 × 10⁻⁷. GROMACS's is 1.7 × 10⁻⁵
  from both, in single precision with β from `ewald-rtol` = 7.9 × 10⁻⁹;
  not examined further.
- On the device in mixed precision with groups, the Lennard-Jones is
  7920.937481, and the bonded terms are within 2 × 10⁻⁶ of the CPU.
- From the files of CHARMM (the PSF, the CRD, and the three files of
  parameters), the Lennard-Jones is 7920.937169 against CHARMM's
  7920.937173 (5 × 10⁻¹⁰), with the parameters as CHARMM reads them, and
  every other term is that of the topology of GROMACS. Two POPC of
  CHARMM36 (the files of lipids appended to those of proteins) agree in
  every bonded term to the printed digits and in the Lennard-Jones to
  −6.334391 against −6.33439096; a synthetic system with wildcards of both
  kinds, a dihedral type of two terms, NBFIX, and a CMAP of 4 × 4 is
  `test/Driver/charmm.test`, its values those of CHARMM.
- From 50 ps of equilibration at 300 K (steps of 2 fs, SHAKE and SETTLE,
  mixed precision, 707 ns/day on one RTX 3090), 1 ns at constant energy
  drifts by +0.03 ± 0.23 kcal/mol/ns with the power force switch,
  −0.78 ± 0.23 with the force switch of GROMACS, and −0.92 ± 0.21 with a
  plain cutoff of the Lennard-Jones: the forces that differentiation gives
  are those of the energy. From the files of CHARMM, with TIP3 rigid by
  SETTLE, the drift is −1.19 ± 0.20 kcal/mol/ns at 706 ns/day.

The phase `hexagonal` of the script takes the 1417 waters of the build
within a hexagonal cell of CHARMM (36, 36, 38 Å, γ = 120°), in its frame,
minimizes them in CHARMM with PME, and evaluates the written coordinates:
MDIR, from the PSF and the CRD with `box = [36, 36, 38, 90, 90, 120]`,
agrees in the bonds and angles to the printed digits, in the
Lennard-Jones to 10⁻⁹, and in each term of the electrostatics, real
space, excluded pairs, reciprocal, and self, to 4 × 10⁻⁸ after the ratio
of the Coulomb constants; the same on the device in double precision,
and with the groups in mixed precision to 3 × 10⁻⁷ (D127).

`charmm27.ff` of GROMACS, through `scripts/validation/gromacs/run.sh`,
agrees with GROMACS within 4.3 × 10⁻⁶ in every term, angles with their
Urey–Bradley terms and dihedrals with the harmonic impropers among them.
`test/Driver/gromacs-charmm-terms.test` checks each term and each modifier
on a small topology against the formulas.

## 6. Not yet

| Item | Note |
|---|---|
| The cell from CHARMM-GUI's files | The box of a run from a PSF comes from `[boundary]`; CHARMM-GUI gives it in `step3_pbcsetup.str` and `sysinfo.dat` |
| Types given by numbers in a PSF | Their meaning depends on the order of the RTF; CHARMM 51 and CHARMM-GUI write names |
| VSWITCH | The potential switch of CHARMM (Section 2), for older inputs |
| Lone pairs of CGenFF, the Drude model | Virtual sites of other constructions, and polarization |
| The parameters of C36/LJ-PME | The mesh for the dispersion is in place (`lennard_jones = "PME"`, D162, validated on Trp-cage in CHARMM36m against GROMACS); the force field fitted to it has not been run |
