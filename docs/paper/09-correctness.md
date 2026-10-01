# 9. Correctness

A compiler that writes its own kernels must show that they compute the
model. This section collects the evidence, each item marked by its kind:
a check that runs with the tests (`lit`, 146 tests in `test/`), a value
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
| The dipeptide from a topology converted to GROMACS format (`gromacs-dipeptide.test`) | MDIR's Amber reader | Eight terms within $2\times10^{-6}$ relative or $3\times10^{-4}$ kJ/mol | `lit` |
| amber99sb-ildn, 99sb, 03, 14sb; 99sb-ildn with TIP4P-Ew; amber19sb (`scripts/validation/gromacs`) | GROMACS | Within $5\times10^{-6}$; amber19sb within $3\times10^{-6}$, its CMAP within $9\times10^{-7}$ | Recorded |
| The dipeptide in TIP3P from its GROMACS topology, PME of the same $\beta$, grid, and order (`scripts/validation/forces`) | GROMACS 2026.3 | Reciprocal energy within $2.6\times10^{-6}$; the rest of the Coulomb energy within $1.1\times10^{-6}$ of its largest terms | Recorded |
| Ubiquitin in 5700 OPC waters, amber19sb.ff from `pdb2gmx`, 24,031 particles, PME of the same $\beta$, grid, and order (`scripts/validation/protein`) | GROMACS 2026.3 | Bonds, angles, dihedrals, CMAP, both 1–4 terms, Lennard–Jones, and dispersion within $1.7\times10^{-6}$ each; Coulomb within $1.9\times10^{-6}$ of GROMACS with tables of the Ewald correction ($6.6\times10^{-6}$ with its SIMD kernels, below) | Recorded |

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
  and variances within 1.2 of $N_f(k_BT)^2/2$; the ratio of the
  distributions of the potential energy at the two temperatures has the
  slope $\beta_{300} - \beta_{306}$ within 0.17 standard errors, and that
  of the volumes at 1 and 300 atm the slope $-\beta\Delta P$ within 0.66
  [[Shirts2013]](references.md#shirts2013). At 1 atm the density, $0.99674 \pm 0.00031$ g/cm³, and
  the compressibility, $(4.71 \pm 0.26)\times10^{-5}$ /bar, are within
  one standard error of GROMACS with its own OPC.
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
PME, the barostat, and the deferred reads of sums (`test/Sanitizer`). Short runs of the nine systems of the Amber
suite check that every number of the log is finite (`test/Scale`).

## 9.9 What is not yet verified

- The GPU against the CPU over long runs is a comparison of means: over
  20 ps of the dipeptide in OPC in double precision the rows of the two
  logs agree to every printed digit for 500 steps and then part as the
  dynamics amplifies the rounding of sums in different orders (1.1
  kcal/mol at 5 ps); the means of the potential energy and of the
  temperature over the last 16 ps differ by 1.8 and 1.6 standard errors
  (`scripts/validation/gpu-cpu`). A longer comparison, with errors from
  the autocorrelation, is still to do.
