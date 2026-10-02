# 9. Correctness

A compiler that writes its own kernels must show that they compute the
model. This section collects the evidence, each item marked by its kind:
a check that runs with the tests (`lit`, 167 tests in `test/`), a value
that a test pins after it was compared once with an independent program
(the reference value is in the test's comment), or a measurement recorded
in a decision. Where the log says that an energy "changed by" a fraction,
it means $\lvert E_\text{last} - E_\text{first}\rvert/\lvert E_\text{first}\rvert$
between the first and the last row, not the largest excursion.

## 9.1 The terms and the forces against sander and GROMACS

*Table 9.1. The energy terms at the start against other programs. MDIR
uses the Coulomb constant of CODATA 2018 [[Tiesinga2021]](references.md#tiesinga2021), larger than
Amber's by a factor 1.0000346 (332.0637/332.0522); sander's electrostatic
terms are multiplied by that factor before they are compared.*

| System | Compared with | Result | Kind |
|---|---|---|---|
| Alanine dipeptide in TIP3P, ff14SB [[Maier2015]](references.md#maier2015), cutoff 9 Å, 1168 particles (`amber.test`) | sander (AmberTools 26) | Coulomb $-2333.9851$ against $-2333.9852$; bonds, angles, dihedrals, and both 1–4 terms to the last printed digit ($10^{-4}$ kcal/mol); Lennard–Jones with the dispersion correction 373.7997 against 373.7948, the difference being how the correction averages $C_6$ (Section 5.4) | Pinned |
| ACE-ALA-GLY-SER-NME in OPC [[Izadi2014]](references.md#izadi2014), ff19SB [[Tian2020]](references.md#tian2020) with CMAP, 2332 particles (`amber-ff19sb.test`) | sander | Coulomb $-2478.9494$ against $-2478.9495$; CMAP 0.096326 against 0.0963, and against an independent model 0.096326461457 | Pinned |
| Dipeptide in OPC with 387 extra points (`amber-opc.test`) | sander | Coulomb $-626.4365$ against $-626.4366$ | Pinned |
| Same system with PME, same $\beta$, grid, and order, `influence = "OPTIMAL"` (`amber-pme.test`) | sander | $-4136.14362$ against $-4136.14360$ kcal/mol | Pinned |
| Propane and water from a GROMACS topology, 224 particles, with macros, `#ifdef`, wildcards, and `nonbond_params` (`gromacs.test`) | GROMACS 2026.3 | Bonds, angles, dihedrals, both 1–4 terms, Lennard–Jones, and dispersion within $5\times10^{-6}$ relative, checked by a script at every run | `lit` |
| Pairs three bonds apart with `[ nonbond_params ]` and gen-pairs, and with `[ pairtypes ]` (`gromacs-nbfix-pairs.test`); impropers in the order of LEaP (`gromacs-leap-order.test`) | GROMACS 2026.3 | To the printed digits ($10^{-6}$ kcal/mol) | `lit` |
| The dipeptide from a topology converted to GROMACS format (`gromacs-dipeptide.test`) | MDIR's Amber reader | Eight terms within $2\times10^{-6}$ relative or $3\times10^{-4}$ kJ/mol | `lit` |
| amber99sb-ildn, 99sb, 03, 14sb; 99sb-ildn with TIP4P-Ew; amber19sb; charmm27 (`scripts/validation/gromacs`) | GROMACS | Within $5\times10^{-6}$; amber19sb within $3\times10^{-6}$, its CMAP within $9\times10^{-7}$; charmm27, with Urey–Bradley angles and harmonic impropers, within $4.3\times10^{-6}$ | Recorded |
| Urey–Bradley angles, harmonic impropers across $\pm\pi$, and each modifier of the Lennard-Jones of a topology (`gromacs-charmm-terms.test`) | The formulas; CHARMM 51b1 for the power force switch | To the printed digits ($10^{-6}$ kcal/mol) | `lit` |
| Trp-cage in CHARMM36m, 3900 TIP3P, 6 Na⁺, 7 Cl⁻, 12,017 particles, PME of the same $\beta$, grid, and order, VFSWITCH (`scripts/validation/charmm`) | CHARMM 51b1 | From the files of CHARMM (D122): bonded terms, Urey–Bradley, impropers, and CMAP within $10^{-7}$, Lennard-Jones within $5\times10^{-10}$; through a topology of GROMACS (D121), Lennard-Jones within $2.2\times10^{-7}$; Coulomb by the ratio of the constants, $2.38\times10^{-5}$ | Recorded |
| A synthetic PSF with wildcards of dihedrals and impropers, a dihedral of two terms, NBFIX, special 1-4 parameters, and CMAP of $4\times4$ (`charmm.test`) | CHARMM 51b1 | To the printed digits ($10^{-6}$ kcal/mol) | `lit` |
| The dipeptide in TIP3P from its GROMACS topology, PME of the same $\beta$, grid, and order (`scripts/validation/forces`) | GROMACS 2026.3 | Reciprocal energy within $2.6\times10^{-6}$; the rest of the Coulomb energy within $1.1\times10^{-6}$ of its largest terms | Recorded |
| The nine systems of the Amber benchmark suite, 23,558 to 1,067,095 particles, from the files of the suite as they are (two topologies in the format before Amber 7, D133), PME of the same $\beta$, even grid, and order, without the correction for the dispersion (`scripts/validation/suite`) | pmemd 26 on the CPU in double precision (JAC also sander) | Every term within $3.4\times10^{-8}$ relative; the forces within $4.0\times10^{-7}$ of the rms force, rms, and $7.4\times10^{-6}$ at most | Recorded |
| Ubiquitin in 5700 OPC waters, amber19sb.ff from `pdb2gmx`, 24,031 particles, PME of the same $\beta$, grid, and order (`scripts/validation/protein`) | GROMACS 2026.3 | Bonds, angles, dihedrals, CMAP, both 1–4 terms, Lennard–Jones, and dispersion within $1.7\times10^{-6}$ each; Coulomb within $1.9\times10^{-6}$ of GROMACS with tables of the Ewald correction ($6.6\times10^{-6}$ with its SIMD kernels, below) | Recorded |

The comparison of the suite found two things. The topology of Factor IX
that the benchmark had written again with a converter had lost 2283
dihedral terms, which the format before Amber 7 gives by negative
periodicities: its dihedrals were 1338.6 kcal/mol where pmemd takes 2259.2
from the file of the suite, so the rates of Factor IX before D133 were of
another model. And on a grid of an odd number of points the aliasing
factor of Amber's influence function takes another frequency at one
index (`docs/pme-m1.md`, Section 1.1): on Cellulose with 270 × 125 × 125
points the electrostatic energies differ by $1.6\times10^{-6}$, with
270 × 128 × 128 by $2\times10^{-9}$. MDIR chooses even grids.

**Forces on every particle** (`scripts/validation/forces/run.sh`). MDIR in
double precision writes the forces of the input coordinates into a
checkpoint after one step of $10^{-9}$ ps from zero velocities; sander
writes its own the same way, and GROMACS computes them again at the
positions of MDIR. With sander's charges scaled to MDIR's Coulomb
constant and its removal of the net force of PME turned off (`netfrc =
0`), the differences, relative to the rms force, are:

| System | Against | rms | Largest |
|---|---|---|---|
| Dipeptide in TIP3P, ff14SB, plain cutoff of 9 Å | sander | $2.0\times10^{-7}$ | $4.1\times10^{-6}$ |
| Dipeptide in OPC, PME of the same $\beta$, grid, and order | sander | $1.3\times10^{-7}$ | $2.5\times10^{-6}$ |
| Dipeptide in TIP3P from its GROMACS topology, the same PME | GROMACS, mixed precision | $6.2\times10^{-6}$ | $7.7\times10^{-5}$ |

The first two are the rounding of sander's forces, which it writes in
single precision; the third that of GROMACS in mixed precision (MDIR's
own mixed mode differs from its double precision by $5\times10^{-6}$ on
the same input).

**Ubiquitin against GROMACS** (`scripts/validation/protein/run.py`). The
system of Section 10.6, equilibrated by GROMACS, at the same positions in
both programs (the sites placed from their atoms on both sides), MDIR in
double precision on the CPU against a rerun of GROMACS. Two differences
were found and explained on the way. The impropers were 0.896 kcal/mol
above GROMACS's, by an amount that did not add up over subsets of the
lines of the topology. amber19sb.ff defines
`_FF_AMBER_LEAP_ATOM_REORDERING`, and grompp then puts the atoms of each
dihedral in the order that LEaP gives them: by the types of the matching
entry, a blank for a wildcard, it reverses a dihedral whose first type
sorts after its last, and it sorts the three outer atoms of every improper
after the first of its types, keeping the parameters found in the order
of the file. Which atoms are the outer ones changes the angle of an
improper, and whether an improper is reordered depends on the impropers
before it, which is why subsets did not add up. MDIR's reader now does the
same (`gromacs-leap-order.test`), and the dihedrals agree to
$1.5\times10^{-7}$. The Coulomb term then differed by $6.6\times10^{-6}$
(0.58 kcal/mol). On the waters alone GROMACS's short-range Coulomb term
is 0.554 kcal/mol below MDIR's with its default SIMD kernels, which
compute the Ewald correction by an analytical approximation in single
precision, and 0.121 below with its kernels with tables of the
correction; translating the whole system moves
GROMACS's short-range term by at most 0.02 kcal/mol. The approximation
makes the same error on the excluded pairs of every rigid water, and over
5700 waters the errors add up. The reciprocal sums agree to
$4\times10^{-6}$.

## 9.2 Particle mesh Ewald against a direct Ewald sum

`test/Driver/pme.test` runs three OPC waters at $\beta = 3.4705$ nm⁻¹
and computes the same terms with `Inputs/ewald_reference.py`, which sums
over the wave vectors instead of a grid, at every run. The direct sum,
the excluded pairs, and the self term agree to the digits printed
($10^{-6}$ kcal/mol); the reciprocal sum, 4.697241 against 4.6972413 on a
grid of 64 points with splines of order 8. On the default grid, 35 points
at order 4, the reciprocal sum is off by $7\times10^{-4}$ of itself, the
error of the method. The virial is checked against the derivative of
the energy with respect to a uniform scaling $\lambda$ of the cell and
the positions: $\operatorname{tr}\mathsf W$ against $-dU/d\lambda$ by
Richardson's central difference, to $3\times10^{-5}$ relative
(`check_virial.py`). On the ff19SB system the influence function of
Essmann et al. is $-1.8\times10^{-3}$ from the Ewald sum and the optimal
one $-2\times10^{-4}$ (recorded in `docs/pme-m1.md`).

## 9.3 Derivatives

- **Against closed forms.** Kernels generated by `md-differentiate` are
  run by `mlir-runner` and compared with closed forms of Lennard–Jones to
  $10^{-12}$ relative: energy, force, virial, and the derivative with
  respect to $\sigma$, for each truncation of Table 3.2
  (`test/Dialect/MD/Transforms/differentiate-values.mlir`). Bonds, angles
  in both forms, and dihedrals are compared with a reference in Python to
  $10^{-12}$ (`differentiate-tuples-values.mlir`). The reference's own
  forces are checked against finite differences with $h = 10^{-6}$ when
  the tests are generated.
- **Through the pipeline**, on the CPU, with OpenMP, and on the GPU:
  energies, forces, and virials of bonded and nonbonded terms and of
  tabulated potentials within $10^{-10}$ ($10^{-5}$ in the mixed mode);
  200 steps of velocity Verlet within $10^{-9}$
  (`test/Integration`).

## 9.4 Conservation of energy

Every test of dynamics bounds the change of the total or the conserved
energy over its run, and several record how it scales with the step,
which tells a correct integrator (second order) from a wrong force (no
convergence):

*Table 9.2. Conservation in the tests, and its convergence.*

| Test | System | Bound | Observed with the step |
|---|---|---|---|
| `argon.test` | 864 Lennard–Jones particles, 2000 × 5 fs | $10^{-4}$ | |
| `pme-settle.test` | Dipeptide in OPC, PME, SETTLE, 100 × 1 fs | $10^{-4}$ | $5.9\times10^{-5}$ at 1 fs, $7.8\times10^{-6}$ at 0.5 fs |
| `pme-gpu.test` | Same, 500 × 1 fs, GPU, double and mixed | $10^{-4}$ | |
| `settle.test` | 500 × 2 fs | $10^{-2}$ | 0.056, 0.016, 0.003 kcal/mol at 2, 1, 0.5 fs |
| `cmap.test` | Peptide in vacuum, 2000 × 0.1 fs | $10^{-3}$ | $1.1\times10^{-3}$, $2.9\times10^{-4}$, $7.4\times10^{-5}$ at 0.2, 0.1, 0.05 fs |
| `virtual-sites.test` | Three OPC waters, 1000 × 0.1 fs | $10^{-3}$ | $6.8\times10^{-4}$ at 0.1 fs, $1.7\times10^{-4}$ at 0.05 fs |
| `thermostat.test` | Mixture, NVT, 2000 × 4 fs | $10^{-4}$ (conserved energy) | |
| `barostat.test` | Mixture compressed from 12,487 to about 11,000 Å³ | $10^{-3}$; $10^{-4}$ with `EXACT` | |
| `barostat-rigid-gpu.test` | Dipeptide in OPC, rigid, NPT with $\tau_P = 0.5$ ps, 25,000 × 2 fs, deterministic | $10^{-2}$ (conserved energy) | $9.5\times10^{-4}$; $1.3\times10^{-2}$ with the count before D116 |

Halving the step divides the change by about four in each series, as a
method of second order should.

**Flexible water against sander** (`scripts/validation/nve`). The
dipeptide in TIP3P without constraints, particle mesh Ewald, 2 ps from the
same positions and velocities in both programs, MDIR in double precision.
The stretch of O–H has a period of about 9 fs, so at 0.5 fs, the step
usual for flexible water, the error of the integrator in each bond is some
$(\omega\Delta t)^2/8 = 1.5\%$ of its energy, and the comparison needs
smaller steps. The standard deviation of the total energy about a fitted
line, in kcal/mol, with its ratio to that of the step twice as long:

| Step | MDIR | sander | MDIR, terms shifted to 0 at the cutoff |
|---|---|---|---|
| 0.5 fs | 1.53 | 1.21 | 1.527 |
| 0.25 fs | 0.415 (3.7) | 0.52 (2.3) | 0.368 (4.16) |
| 0.125 fs | 0.143 (2.9) | 0.34 (1.6) | 0.092 (4.00) |
| 0.0625 fs | 0.165 (0.9) | 0.27 (1.2) | 0.023 (4.04) |

With the cutoff of 9 Å as Amber takes it, the energy of a pair jumps when
it crosses the cutoff, and both programs reach a floor that does not fall
with the step, MDIR at 0.15 kcal/mol and sander at 0.27 (whose leapfrog
reports a kinetic energy averaged over the half steps), with drifts of
$-0.2$ to $-0.5$ kcal/mol/ps in MDIR and $-0.2$ to $-0.95$ in sander. With
both terms shifted to 0 at the cutoff, which sander does not take, the
fluctuation of MDIR falls by 4.0 to 4.2 with each halving of the step, as
a method of second order should, and the drift is below 0.005 kcal/mol/ps.

**Over 2 ns.** The figure of merit for production is the drift over a
long run at the production step. JAC (23,558 atoms) at constant energy,
2 fs, SHAKE on the bonds of hydrogen and rigid water, in the mixed
precision of the suite, ran 2 ns at 722 ns/day. Its total energy fell by
5.7 kcal/mol by the first row of the log at 40 ps, in the first few ps
as finer logs show for both programs (Section 13), and then rose by 1.5 kcal/mol/ns, a
slope fitted to the rows after 40 ps, $2.6\times10^{-4}$ kJ/mol/ns per
atom; pmemd.cuda 26 (SPFP) on the same input rose by 5.9 kcal/mol/ns,
$1.0\times10^{-3}$ kJ/mol/ns per atom (Figure 7.1, D112). MDIR in double precision did not drift
over 0.4 ns. Section 7.3 tells how the first version of the mixed mode
drifted by 830 kcal/mol over the same run and how the cause was found.
Section 10 reports the change of the energy for every run of the suite
beside its rate.

## 9.5 Constraints, couplings, and random numbers

- **Constraints.** Every frame of the tests of SETTLE has O–H at
  0.87243313 Å and H–H at 1.37120510 Å within $5\times10^{-5}$ Å, in
  double precision (closed form) and in the mixed mode (M-SHAKE); every
  constrained bond of the tests of SHAKE is within $10^{-4}$ Å of its
  length, under velocity Verlet and leapfrog and through a compression of
  the barostat. The fused integration kernel (Section 8.2) is compared
  with the separate loops and with the CPU column by column to $10^{-9}$
  in double precision on 855 particles whose groups were placed apart
  in memory (`integration-gpu.test`).
- **Analytic one-bond option (D128).** The emitted kernel is checked on
  67 cases for squared bond length, mass-weighted displacement, and the
  fixed old correction direction. Cases cover equal and repartitioned
  masses, tiny corrections, near tangency, lattice image shifts, and the
  backward predictor's Newton fallback (`analytic-bonds.test`). The
  driver compares energy, pressure and virial logs with the default
  Newton solver on CPU and GPU in double and mixed precision. A hydrogen
  translated by a tilted lattice vector exercises the triclinic path,
  including GPU groups (`analytic-bonds-triclinic.test`). All nine Amber
  scale systems run with both options, including repartitioned masses at
  4 fs and the million-atom STMV case. The fused analytic path is also
  checked by memcheck, initcheck and racecheck. These checks establish
  equivalence on the tested inputs; the fallback retains the original
  fixed-iteration solver's limitations (Section 6.2).
- **Thermostat and barostat.** The factor of the thermostat, iterated
  over chains of 200,000 steps for four combinations of $N_f$ and of the
  decay $c$, has a mean of $K$ within 2% of its target and a variance
  within 5% of $2\bar K^2/N_f$; the drift of the strain of the barostat
  matches $2\ln(1 - f(P_0 - P)/2)$ to $10^{-15}$, and its noise over
  40,000 draws has the expected deviation within 2% (`test/Runtime/
  random.test`). These test the generators, not the distributions that a
  trajectory samples. Philox matches three known-answer tests of
  Random123 [[Salmon2011]](references.md#salmon2011).
- **Ensembles** (`scripts/validation/ensembles`). On 1039 rigid OPC waters
  with PME on the GPU in mixed precision, runs of 5 ns at 300 K and 306 K
  give mean kinetic energies within 0.9 standard errors of $N_fk_BT/2$
  and variances within 1.2 of $N_f(k_BT)^2/2$, and the whole
  distribution of the kinetic energy passes a one-sample
  Kolmogorov–Smirnov test against the gamma distribution of $N_f/2$ and
  $k_BT$ of the bath, not fitted ($D = 0.020$, $p = 0.26$ at 300 K and
  $D = 0.016$, $p = 0.61$ at 306 K, from 2500 and 2143 samples spaced by
  their statistical inefficiency), the test of [[Merz2018]](references.md#merz2018) that a
  distribution 10% too narrow fails at $p = 0.006$; the ratio of the
  distributions of the potential energy at the two temperatures has the
  slope $\beta_{300} - \beta_{306}$ within 0.17 standard errors, and that
  of the volumes at 1 and 300 atm the slope $-\beta\Delta P$ within 0.66
  [[Shirts2013]](references.md#shirts2013). At 1 atm the density, $0.99674 \pm 0.00031$ g/cm³, and
  the compressibility, $(4.71 \pm 0.26)\times10^{-5}$ /bar, are within
  one standard error of GROMACS with its own OPC.
- **Semi-isotropic coupling** (D119, `scripts/validation/barostat-semi`).
  On the 1039 OPC waters over 2 ns, isotropic, semi-isotropic, and
  semi-isotropic coupling with the height held give the density $0.99682
  \pm 0.00045$, $0.99664 \pm 0.00044$, and $0.99667 \pm 0.00054$ g/cm³
  and the compressibility $5.11 \pm 0.40$, $4.66 \pm 0.37$, and $4.78 \pm
  0.46\times10^{-5}$ /bar: a liquid has no shape, so the three must agree.
  Its volume has a distribution all the same, and the test of two
  pressures [[Shirts2013]](references.md#shirts2013) (1 and 300 atm, 5 ns each) gives the slope of
  $\ln P_2(V)/P_1(V)$ within 0.08 standard errors of $-\beta\Delta P$
  with semi-isotropic coupling and within 0.36 with the height held
  (0.66 with isotropic coupling): the test of [[Merz2018]](references.md#merz2018), which that
  paper applied to isotropic coupling only.
  On a bilayer of 126 POPC of Lipid21 [[Dickson2022]](references.md#dickson2022) in TIP3P against
  GROMACS on the same model, over 20 ns each, the area per lipid is
  $66.70 \pm 0.63$ Å² against $63.62 \pm 0.37$ with four to seven
  independent samples of it in each run, and the densities 1.0211 and
  1.0235 g/cm³. In one cell the two give the same difference of the
  lateral and normal pressures, $-5.4 \pm 3.7$ and $-5.4 \pm 4.4$ bar,
  which sets the area; their pressures differ by 43 bar, which sets the
  volume. Their virials of the forces agree per axis to $10^{-5}$ on the
  same positions, and GROMACS's pressure in that cell moves by 22 bar
  between LINCS of its default order and bonds of hydrogen left flexible,
  where MDIR's does not move. On TIP3P water in one cell the pressures of
  MDIR, GROMACS, and pmemd.cuda at 2 fs are 52.7, 34.4, and 61.6 bar and
  at 0.25 fs 45.1, 46.9, and 43.8 ($\pm 3$ to 4): the estimators of the
  pressure of the integrators, with the forces of the rigid waters in
  them, differ at the step of production and agree as it goes to zero.
  The difference of the densities is that, not the coupling.
- **Densities of the suite across programs** (`scripts/validation/suite/density.py`).
  JAC and Factor IX at 300 K and 1 bar for 1 ns at 2 fs from the files of
  the suite, as one model (D134); the mean volume over the second half,
  with the error of ten blocks, in Å³:

  | Program and barostat | JAC | Factor IX |
  |---|---|---|
  | MDIR, cell rescaling from the virial | 234,431 ± 95 | 911,547 ± 198 |
  | OpenMM 8.6.1, Monte Carlo | 234,479 ± 41 | 911,100 ± 210 |
  | pmemd.cuda 26, Monte Carlo | 234,929 ± 147 | 912,359 ± 193 |
  | pmemd.cuda 26, Berendsen from the virial | 234,682 ± 38 (299.5 K) | 912,388 ± 57 (299.7 K) |
  | GROMACS 2026.3, cell rescaling from the virial | 234,215 ± 73 | 910,469 ± 118 |

  MDIR agrees with the Monte Carlo barostat of OpenMM within 0.02% and
  0.05%; pmemd.cuda lies 0.1 to 0.2% above with either of its barostats
  and GROMACS 0.1% below, the order of their pressures at 2 fs above.
  Seeds of MDIR give 234,431 to 234,512. A Monte Carlo barostat accepts a
  scaling by the change of the energy, which jumps where a pair crosses
  the truncated cutoff, and the pressure of the virial does not see the
  jump [[Gomez2022]](references.md#gomez2022): without the correction for
  the dispersion, OpenMM gives JAC 237,223 ± 61 and MDIR 240,066 ± 96, 1.2%
  apart. With it the two agree, since for the $r^{-6}$ part the jump that
  the Monte Carlo barostat sees equals $E_\text{disp}/V$, the difference
  between the pressure of the correction, $2E_\text{disp}/V$, and the
  derivative of its energy.
- **Equivalences.** The trajectory does not depend on how steps are
  grouped into periods, on velocity Verlet against leapfrog with
  constraints, or on how the work of the barostat is counted: the tests
  diff the rows of step 2000.

## 9.6 Restarts and determinism

A checkpoint stores every value in 64 bits. Runs of 100 + 100 steps from
a checkpoint and of 200 steps end in states that are identical bit for
bit, every double of the positions, velocities, and forces compared by
`mdir checkpoint`: on the CPU with four threads, on the GPU, with the
thermostat, with the barostat at the default period and at a period of
one step, and with PME and rigid water (`*-restart.test`). In the
deterministic mode two runs on a GPU are bitwise identical, and so is a
run with the reciprocal sum on a second stream against the serial one
(`pme-gpu.test`, D87). The spatial sort changes the order of sums; runs
with and without it agree to about $10^{-10}$ (`reorder.test`).

## 9.7 Neighbor structures

- **Templates against all pairs.** The matrix of the host matches a
  reference in entry count, an order-independent checksum, and the
  longest row; the matrix of the device equals that of the host entry by
  entry, with no difference in five cases of 2000 particles. The build of
  groups is checked on 2000 particles with positions unwrapped up to
  three edges away, in seven cases including exclusions, a row of more
  excluded partners than a warp holds (D106), a position that is not a
  number (D107), a denser cube, a smaller reach, and a cell narrower than
  twice the reach and the extent of a group (D115): no pair within reach
  missing or duplicated, none beyond reach or in the wrong image, and the
  same 16,352 pairs as the matrix in the first (`neighbors-groups-gpu.mlir`).
- **Runs.** Over 100 steps in double precision the logs of the groups and
  of the matrix are identical, terms included, as are those of the dual
  list (11/9.3 Å) and the matrix, and those of the groups and the matrix
  on the argon–krypton mixture in its cell of 23.2 Å (`groups-gpu.test`). Under the barostat,
  the dual list matches one list over 1000 steps (Section 4.5).
- **The tests of validity.** Section 4.1 proves the test under scaling;
  `tiles-m1.md` proves the dual list's.

## 9.8 Robustness

The GPU tests run under compute-sanitizer with memcheck, initcheck, and
racecheck, with no error over 20 steps of the ff19SB system at constant
pressure with restraints, a run that exercises the loops over pairs, the
fused kernels, the disjoint sets of SETTLE and SHAKE, the virtual sites,
PME, the barostat, and the deferred reads of sums, and the same with the
groups and a dual list; and over 20 steps of a rhombic dodecahedron, which
exercises the kernels of a triclinic cell, with the matrix and with the
groups (D125, D126; `test/Sanitizer`). Run on the groups, they found two
defects of their build: the kernel of D115 read the box of a group past
the last before it tested the group, and the scan of a block let the lanes
of the first warp read a sum that another lane then overwrote, which
racecheck reports without a barrier between. Initcheck found that the
host copied back ten values of the total of the reciprocal sum, of which
the kernels write seven, the energy and the six components of the virial;
the copy now takes seven. Short runs of the nine systems of the Amber
suite check that every number of the log is finite (`test/Scale`).
A run of 500 ps on the CPU with OpenMP crashed as it wrote its last
checkpoint: the variables of the reductions of OpenMP were allocas inside
the loop of the steps, and the stack grew by 22 bytes a step until a call
at the end overflowed it. They are now allocated once (D117), and the
stack of 100,000 steps stays at its size after the compilation
(`hoist-allocas.mlir`).

## 9.9 The GPU against the CPU over long runs

From one state the rows of the logs of the GPU and of the CPU agree to
every printed digit for some hundreds of steps and then part, as the
dynamics amplifies the rounding of sums added in different orders: over
20 ps of the dipeptide in OPC in double precision, 500 steps of 1 fs, and
1.1 kcal/mol apart at 5 ps (`scripts/validation/gpu-cpu/run.sh`). What
must agree is then the distribution. The same system, equilibrated for
100 ps at 300 K and 1 atm on the GPU, ran 500 ps at constant volume on
each in double precision, 2 fs with SHAKE and SETTLE, stochastic velocity
rescaling every 10 steps with a seed of its own
(`scripts/validation/gpu-cpu/nvt.py`). With the errors of samples spaced
by their statistical inefficiency (65 and 133 independent samples of the
potential energy):

*Table 9.3. The GPU against the CPU at 300 K, 500 ps each.*

| Quantity | GPU | CPU | Difference |
|---|---|---|---|
| Mean potential energy, kcal/mol | $-4763.3 \pm 4.1$ | $-4759.2 \pm 2.6$ | $-0.84$ SE |
| Its standard deviation, kcal/mol | $32.6 \pm 2.9$ | $30.1 \pm 1.9$ | $+0.72$ SE |
| Mean temperature, K | $300.17 \pm 0.72$ | $301.03 \pm 0.48$ | $-1.00$ SE |
| Mean total energy, kcal/mol | $-4055.5 \pm 5.9$ | $-4049.4 \pm 3.6$ | $-0.88$ SE |

A first version of this comparison started from the file of tleap, whose
water is at about 0.6 g/cm³; at constant volume it relaxed over the whole
run, its potential energy fell by 26 to 30 kcal/mol from the first half to the
second on both, and only 21 and 23 independent samples remained. A run with one
seed for both shares the noise of the thermostat, and the errors above,
which treat the two as independent, would overstate the error of their
difference.

## 9.10 Triclinic cells

A triclinic cell (Sections 4.4 and 5.2) is checked against other programs
and against the paths of the orthorhombic cell:

- **The terms.** A rhombic dodecahedron of 403 TIP3P waters from GROMACS,
  whose tilts lie on the bounds of the reduced form, against a rerun of
  GROMACS 2026.3 with the same cutoff, grid, order, and $\beta$: bonds,
  angles, Lennard-Jones, and the dispersion correction within $10^{-6}$,
  Coulomb within $1.3\times10^{-5}$ of the sum and $2\times10^{-6}$ of its
  largest parts, the single precision of GROMACS (`triclinic.test`). A
  truncated octahedron of Amber (549 TIP3P, Na⁺, Cl⁻) agrees with sander
  in every term to the printed digits, the electrostatics after the ratio
  of the Coulomb constants (D123). 1417 TIP3P waters in a hexagonal cell
  of CHARMM (36, 36, 38 Å, $\gamma = 120°$), evaluated by CHARMM 51b1 in
  its symmetric frame and by MDIR from the PSF and the CRD with the cell
  given by its lengths and angles, agree in the bonds and angles to the
  printed digits, in the Lennard-Jones to $10^{-9}$, and in each part of
  the electrostatics, real space, excluded, reciprocal, and self, to
  $4\times10^{-8}$ after the ratio of the constants
  (`scripts/validation/charmm/run.py`, D127).
- **The device against the CPU.** On the dodecahedron, double precision
  on the device gives the terms, the virial, and 200 steps of the CPU to
  the printed digits; mixed precision, the deterministic mode, and order 6
  agree to $10^{-5}$ (`triclinic-gpu.test`, D125). The groups give the log
  of the matrix over 200 steps in double precision, with one list and
  with a dual list (D126).
- **The lists of the groups.** Every pair is checked against f64 in an
  octahedron, a dodecahedron, a hexagonal cell, and a cell narrow enough
  to need the images of D115: no pair within the reach missing or
  duplicated, none beyond it or in the wrong image
  (`neighbors-groups-triclinic-gpu.mlir`).
- **Conservation.** Amber's truncated octahedron in OPC conserves 10 ps
  at constant energy on the device as on the CPU ($6.3\times10^{-4}$
  against $5.8\times10^{-4}$), and with the groups as with the matrix
  (D125, D126).
- **The barostat** (D127). TIP3P water in a truncated octahedron of 1589
  waters and in a box of 1391, 400 ps at 1 bar and 300 K, has the
  densities $0.9850 \pm 0.0007$ and $0.9848 \pm 0.0008$ g/cm³ over the
  last 300 ps; the shape of the cell stays to the printed digits, and a
  run continued from a checkpoint gives the log of one that does not stop
  (`triclinic-npt.test`).
