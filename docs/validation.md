# Validation of the Physics

Status: kept current; last checked against the code on 2026-10-21. The
validation of particle mesh Ewald is in [pme.md](pme.md), Section 6, and
that of CHARMM in [charmm.md](charmm.md); the tables of the white paper
are in [Section 9](paper/09-correctness.md).

This document records how the terms, the integrators, and the ensembles
were compared with the reference engines and with scripts.

## 1. What is tested

| Test | Compared with | Tolerance |
|---|---|---|
| Energy, forces, and virial of each kind of bonded term, for chains of a few particles | A script that evaluates the definitions and checks its forces against finite differences | 1e-10 |
| The same with particles in line and with a dihedral of $\pi$ | The same | 1e-10 |
| A neighbor structure with exclusions | A search over all pairs | Exact |
| Lookup in a table, pairs three bonds apart | A script | 1e-10 |
| Particle mesh Ewald | An Ewald sum [[Ewald1921]](references.md#ewald1921) with many wave vectors, in a script | That of the parameters |
| SETTLE and SHAKE | A script; the constrained distances | That of the tolerance |
| The factor of the thermostat and of the barostat | A script with the same generator | 1e-12 |
| A run that continues from a checkpoint, with thermostat and barostat | The run that was not interrupted | Exact |
| Alanine dipeptide in water: the energy of each term at the start, and the forces | AmberTools and GROMACS, each from its own format | 1e-5, relative |
| The same system: the conservation of energy at constant energy, and temperature, pressure, and density at constant temperature and pressure | The engines, within the statistical error | |
| The JAC benchmark: run times | The engines on the same device | |

Measured as each part was built (to 2026-10-03):

| Part | Result |
|---|---|
| Tuple sets, internal coordinates, differentiation, loops over tuples: chains of particles with bonds, angles, and dihedrals | Energy, forces, and virial agree with a reference in double and mixed precision, on the CPU, with OpenMP, and on a GPU; 200 steps of velocity Verlet agree to 1e-9 |
| Exclusions in the neighbor build; pairs three bonds apart: chains with Lennard-Jones | Energy, forces, and virial agree with a reference on the CPU, with OpenMP, and on a GPU |
| The dipeptide in flexible water at constant energy, from the positions and velocities that sander takes | At 0.5 fs its energy fluctuates by 1.53 kcal/mol against sander's 1.21, and with the terms shifted at the cutoff the fluctuation falls by 4.0 to 4.2 with each halving of the step down to 0.0625 fs (`scripts/validation/nve`; white paper, Section 9.4) |
| Removal of the motion of the center of mass, random numbers, the thermostat (D67) | Philox 4×32-10 agrees with the known answers of Random123, and the factor of the thermostat samples the canonical distribution of the kinetic energy. On a mixture of Lennard-Jones, the conserved energy changes by 2.5 × 10⁻⁵ over 20000 steps at constant temperature, on the CPU and on a GPU; a trajectory is the same for any grouping of steps into loops and across a restart. A drift of every particle is removed to a momentum of 10⁻¹³ amu nm/ps. |
| Constraints: SETTLE, SHAKE, RATTLE, at 2 fs | Rigid water and rigid bonds of hydrogen keep their lengths to the precision of the trajectory; the target of D65 runs at 2 fs with PME and the thermostat (`test/Driver/ff19sb-gpu.test`) |
| The barostat, isotropic stochastic cell rescaling (D72) | 1394 OPC waters at 300 K and 1 bar come to 0.9977 g/cm³ over 300 ps, with a compressibility from the fluctuations of the volume of 4.1 × 10⁻⁵ /bar; a trajectory is the same for any grouping of steps into loops and across a restart; the target of D65 runs at constant pressure on a GPU (`test/Driver/barostat-ff19sb-gpu.test`) |
| CMAP, ff19SB | The terms of ACE-ALA-GLY-SER-NME with ff19SB in OPC agree with sander, and CMAP with an independent model to 12 digits; a peptide with amber19sb agrees with GROMACS. The energy of the peptide alone is conserved as the square of the time step. |

## 2. Comparisons with the engines

The terms of a run from a topology of GROMACS were compared with those of
GROMACS 2026.3 (mixed precision) for the peptide ALA-GLY-SER-LYS-ASP-TRP
in flexible water, built with `pdb2gmx` for each force field that GROMACS
ships (`scripts/validation/gromacs/run.sh`):

| Force field | Bonds, angles, dihedrals, pairs 1-4, Lennard-Jones, dispersion |
|---|---|
| amber99sb-ildn, amber99sb, amber03, amber14sb | Agree to 5 × 10⁻⁶ or better, each term |
| amber99sb-ildn with TIP4P-Ew, whose sites are virtual ([constraints.md](constraints.md), Section 2) | Agree to 4 × 10⁻⁶ or better, each term; the correction for the dispersion counts the sites among the particles in both |
| amber19sb, with CMAP ([tuples.md](tuples.md), Section 6) | Agree to 3 × 10⁻⁶ or better, each term; CMAP to 9 × 10⁻⁷ |
| charmm27, with Urey–Bradley angles (function 5), harmonic impropers (function 2), and CMAP (D121) | Agree to 4.3 × 10⁻⁶ or better, each term; angles with their Urey–Bradley terms to 3.5 × 10⁻⁷, dihedrals with the impropers to 1.8 × 10⁻⁷ |
| oplsaa | Agree to 4.3 × 10⁻⁶ or better, each term; the dihedrals of Ryckaert and Bellemans, terms given by an expression (D136), to 1.6 × 10⁻⁶ |
| gromos54a7, with the bonds and angles of function 2 of GROMOS, terms given by expressions (D136) | Agree to 6.5 × 10⁻⁶ or better, each term, but the angles of the flexible SPC water, 14 kJ/mol in all, to 2.1 × 10⁻⁵ |

Coulomb (SR) is compared with MDIR's reaction field at a permittivity of
78.5 (D140): `Coulomb`, `Coulomb excluded`, and `Coulomb self` agree with
it within 2.4 × 10⁻⁵ for every force field, and with OpenMM in double
precision within 1.5 × 10⁻⁹.

The two readers can be compared with each other when the same system is
in both formats. ParmEd converts a topology of Amber to the format of
GROMACS: the factors of the pairs three bonds apart become `fudgeQQ` and
`fudgeLJ`, and NBFIX pairs go to `[ nonbond_params ]`. A system with more
than one value of the factors does not convert faithfully. GROMACS
applies the NBFIX pairs to the pairs three bonds apart, as Amber does:
with gen-pairs, grompp generates them from the matrix of the nonbonded
parameters after `[ nonbond_params ]`, scales ε by `fudgeLJ`, and then
applies `[ pairtypes ]` as they are; the reader does the same, and the
three agree with a rerun of GROMACS 2026.3 (`gromacs-nbfix-pairs.test`).

**Forces on every particle** (2026-10-02, `scripts/validation/forces/run.sh`).
MDIR in double precision on the CPU writes the forces of the input
coordinates into a checkpoint after one step of $10^{-9}$ ps from zero
velocities; sander writes its forces the same way (`ntwf`, in single
precision), GROMACS by a rerun. For sander the charges of the topology
that MDIR reads are scaled by $\sqrt{332.0522173/332.0637133}$, its
Coulomb constant, and sander keeps the net force of particle mesh Ewald
(`netfrc = 0`; by default it removes it, which shifted every force by
the net force over the number of atoms, 3.5e-5 of the rms force). The
differences, relative to the rms force:

| System | Against | rms | Largest |
|---|---|---|---|
| Alanine dipeptide in TIP3P, ff14SB, a plain cutoff of 9 Å | sander | 2.0e-7 | 4.1e-6 |
| Alanine dipeptide in OPC, particle mesh Ewald of the same β, grid, and order | sander | 1.25e-7 | 2.5e-6 |
| The dipeptide in TIP3P from its topology of GROMACS, particle mesh Ewald of the same β, grid, and order | GROMACS 2026.3, mixed precision | 6.2e-6 | 7.7e-5 |

The differences with sander are those of its forces in single precision.
GROMACS computes again at the positions of MDIR after the step, which
SETTLE moves: the waters of the `.gro`, rounded to 0.001 nm, are not
rigid to 10⁻³ Å, and against the positions of the file the forces of the
waters differed by 2.6e-4. Its reciprocal energy agrees to 2.6e-6, and the
rest of its Coulomb energy to 1.1e-6 of its largest terms, the self
energy and the excluded pairs, which cancel to 3%; MDIR in mixed
precision on a device differs from MDIR in double precision by 5e-6
(rms) on the same input.

**Ubiquitin in OPC against GROMACS** (2026-10-02,
`scripts/validation/protein/run.py`). 1UBQ with amber19sb.ff and its OPC
from `pdb2gmx` (ff19SB with CMAP, 76 residues, neutral), 5700 waters,
24,031 particles in a triclinic cell of 5.4 × 5.6 × 6.0 nm, minimized and
then equilibrated by GROMACS 2026.3 for 200 ps at 300 K and 1 bar. The
terms of the equilibrated state, MDIR in double precision on the CPU
against a rerun of GROMACS (mixed precision) at the same positions, with
the sites placed from their atoms on both sides; a cutoff of 9 Å with no
shift, the correction for the dispersion, and particle mesh Ewald of the
same β (erfc(β r_c) = 1e-5, which both solve for), grid, and order:

| Term | MDIR, kcal/mol | Relative to GROMACS |
|---|---|---|
| Bonds | 209.2124 | −1.7e-6 |
| Angles | 620.1280 | 5.2e-7 |
| Dihedrals, proper and improper | 444.3518 | 1.5e-7 |
| CMAP | 57.2390 | 3.4e-7 |
| Lennard-Jones 1-4 | 272.9706 | −3.1e-7 |
| Coulomb 1-4 | 2875.2110 | 3.8e-9 |
| Lennard-Jones | 10207.9679 | −1.2e-6 |
| Dispersion | −540.0697 | 4.5e-7 |
| Coulomb, all but 1-4 | −87461.2462 | 6.6e-6; 1.9e-6 against tables of the Ewald correction |
| Total | −73314.2353 | 7.7e-6; 2.0e-6 against tables |

Two differences were found on the way. The impropers were 0.896 kcal/mol
above GROMACS, by an amount that did not add up over subsets of the lines
of the topology: grompp puts the atoms of dihedrals in the order of LEaP
when the force field asks for it ([driver.md](driver.md), Section 2.10), and the reader now does
too. The Coulomb term is the error of GROMACS's SIMD kernels, which
compute the Ewald correction by an analytical approximation in single
precision: its error is the same for the excluded pairs of every rigid
water, and over 5700 waters it adds up to 0.55 kcal/mol. The waters alone
show it: GROMACS's short-range Coulomb term is below MDIR's by 0.554
kcal/mol with its default kernels, 0.121 with tables of the correction
(`GMX_NBNXN_EWALD_TABLE`), and 0.136 with its plain C kernel; translating
the whole system moves GROMACS's
short-range Coulomb term by 0.02 kcal/mol at most. The reciprocal sums
agree to 4e-6 (592.8846 against 592.8822).

The rates on GPU 0 (an RTX 3090 at 300 W), 60,000 steps of 2 fs from the
equilibrated state with the second half timed, mixed precision in both:
MDIR with groups and the dual list (outer 12 Å, inner 9.6 Å, within 0.4%
of the best of 11 to 12 Å and 9.4 to 9.8 Å), SHAKE on the bonds of
hydrogen, and SETTLE; GROMACS with its nonbonded terms and PME on the GPU
and its update on the CPU, which virtual sites need, and its own pair
list (`verlet-buffer-tolerance` 0.005 kJ/mol/ps per atom):

| Ensemble | MDIR, ns/day | GROMACS, ns/day | MDIR / GROMACS | Energy changed by: MDIR | GROMACS |
|---|---|---|---|---|---|
| NVE | 605.2 ± 0.4 | 842.9 ± 2.9 with nstlist 80 (two repeats; the third lost 0.4 s to writing its output); 689.7 with the nstlist 10 that it keeps at constant energy | 71.8 ± 0.2% | 2.4e-5 to 4.4e-5 | 1.7e-4 to 2.0e-4 |
| NPT | 571.5 ± 0.7 | 865.5 ± 4.4; 771.8 at a tolerance of 5e-5 | 66.0 ± 0.4% | 1.6e-6 to 1.2e-4 (conserved energy) | 6.7e-3 to 7.1e-3 (conserved energy; 5.3e-3 to 6.9e-3 at tolerances of 5e-4 and 5e-5) |

The rates are means and sample standard deviations over three repeats
(2026-10-02, at D118, each program after the other); the other settings of GROMACS
are from one run each.

MDIR's conserved energy at constant pressure moved by −228 kcal/mol/ns
with the count of D92, the bias with rigid groups (below) as on 1039
waters (−234), and by 6.2e-5 to 1.4e-4 of its value over the 120 ps with
that of D116 (the row above; 543.5 ns/day in one run with the count of
D92). GROMACS's moves at
constant temperature without the barostat too, by +370 to +610
kcal/mol/ns (`nsttcouple` 1 or 25, τ_T 1 or 10 ps, `nstcalcenergy` 1 or
100), against −70 for its total energy at constant energy; not
investigated further.

The device time of a step (nsys, NVE; MDIR over 20,000 steps, GROMACS over
the 3528 that its profile captured; MDIR's kernels sorted by their kind):

| Part | MDIR, µs | GROMACS, µs |
|---|---|---|
| Loops over pairs | 91.6 (3 launches; the inner list of 9.6 Å) | 87.7 (one kernel; its list of 10.7 Å, built every 80 steps) |
| PME | 47.3 | 76.7, on a second stream, beside the nonbonded kernel |
| Loops over particles: the drift, the kicks, and the other loops over particles | 48.9 | The update on the CPU |
| Loops over tuples: bonded terms and excluded pairs 22.5, SETTLE 11.8, placement of the sites 4.9 | 39.3 | Bonded terms, SETTLE, and sites on the CPU; the excluded pairs in the nonbonded kernel |
| Builds and prunings | 34.6 | In the nonbonded kernel every 80 steps |
| Layout and reductions | | 18.3 |
| Total on the device | 261.8 on one stream | 182.8 on two |
| Wall time of a step | 310 | 214 |

The loop over pairs costs the same in both, and MDIR's PME is faster.
GROMACS runs beside that loop what MDIR runs after it: PME on a second
stream, and the bonded terms, the update, and the constraints on the 64
threads of the host; the host's time between MDIR's launches is 48 µs of
a step of 310. MDIR's second stream for the reciprocal sum (D81, D87)
gives 580.8 ns/day here against 584.0 without. On JAC (TIP3P, 8 Å)
MDIR's rate is 147% of pmemd.cuda's at constant energy, with the steps and
output of the suite's input (Section 10 of the white paper); on
this system, with four sites per water and a cutoff of 9 Å, its pair loop
and its list are as fast as GROMACS's, and the rest of the step is not.

**The ensembles** (2026-10-02, `scripts/validation/ensembles/run.py` and
`analyze.py`). A box of 1039 OPC waters from tleap, rigid (M-SHAKE in the
mixed mode), particle mesh Ewald, a cutoff of 9 Å with the correction for
the dispersion, 2 fs, groups and the dual list, the thermostat and the
barostat every 25 steps ($\tau_T$ = 1 ps, $\tau_P$ = 2 ps), 5 ns for each
run from one state equilibrated for 200 ps at 300 K and 1 atm, on the GPU
in mixed precision. Samples are spaced by their statistical inefficiency;
the slopes are maximum-likelihood estimates of $\ln P_2/P_1$ [[Shirts2013]](references.md#shirts2013).

| Test | MDIR | Expected | Standard errors |
|---|---|---|---|
| Mean kinetic energy at 300 K, kcal/mol | 1856.98 ± 0.66 | 1857.34 ($N_f k_BT/2$, $N_f$ = 6231) | −0.55 |
| Its variance at 300 K | 1103 ± 31 | 1107 ($N_f (k_BT)^2/2$) | −0.13 |
| Mean kinetic energy at 306 K | 1893.86 ± 0.72 | 1894.49 | −0.87 |
| Its variance at 306 K | 1194 ± 36 | 1152 | +1.16 |
| Slope of $\ln P_{306}(U)/P_{300}(U)$, mol/kcal | 0.03315 ± 0.0016 | 0.03289 ($\beta_{300} - \beta_{306}$) | +0.17 |
| Slope of $\ln P_{300\,\text{atm}}(V)/P_{1\,\text{atm}}(V)$, Å⁻³ | −0.00708 ± 0.00035 | −0.00731 ($-\beta\,\Delta P$) | +0.66 |

At 300 K and 1 atm the density is 0.99674 ± 0.00031 g/cm³ and the
compressibility (4.71 ± 0.26) × 10⁻⁵ /bar; GROMACS 2026.3 with its own
OPC (amber19sb.ff, its update on the CPU, which virtual sites need), the
same cutoff, PME, couplings, and length, gives 0.99698 ± 0.00030 g/cm³ and
(4.41 ± 0.23) × 10⁻⁵ /bar, 0.6 and 0.9 standard errors apart.

**The conserved energy at constant pressure drifted with rigid groups**
(found and fixed, D116). In the same runs the conserved energy at constant volume moved by
−0.16 kcal/mol/ns, and at constant pressure by −234 kcal/mol/ns, where
GROMACS's moved by −0.37. Runs of 100 ps from the same state: the default
count of the work, of Trotter type, −16.3 kcal/mol (−18.0 in double
precision, −22.1 with a period of 10 steps, −25.7 at 1 fs), the exact
count +1.3, the first-order count +209; over 20 ps a period of one step
drifts as much as one of 25 (−5.4 and −6.3). With flexible water at 0.5
fs, the Trotter count, the exact count, and constant volume drift alike.
So the Trotter count was biased when rigid groups scale by their centers:
a bias proportional to the time, independent of the period, as a term of
second order in the strain would be, about 2 × 10⁴ kcal/mol times the
sum of $(\ln\mu)^2$. Twice the internal kinetic energy, which the count
carries in $\mathcal W$ before and after the velocities are scaled,
accounts for a twelfth of it. The sampling was not in question (the test
of two pressures, the density, and the compressibility above); the count
that measures it was. The count took the virial of the step with twice
the internal kinetic energy, which is the virial of the groups only when
the forces of the constraints are those of one configuration; in the step
that scales, the constraints of the positions take the forces before the
scaling and those of the velocities the forces after it. Counted from the
virial of the groups of the evaluations (D116), the conserved energy of
the water box moves over 1 ns by +16.2 ± 11, +12.7 ± 13, and +12.0 ± 12
kcal/mol/ns at periods of 5, 25, and 100 steps (the exact count +3.7 ±
3.5), and that of ubiquitin in OPC over 0.6 ns by +11.4 ± 33, from −228.
The count changes nothing in the trajectory, so the tests of the
ensembles above stand.

**The GPU against the CPU over 20 ps** (`scripts/validation/gpu-cpu/run.sh`).
The dipeptide in OPC of `pme-settle.test`, 20,000 steps of 1 fs in double
precision on each from the same state: the rows of the two logs agree to
every printed digit for 500 steps, and then part as the dynamics amplifies
the rounding of sums added in different orders, by 1.1 kcal/mol at 5 ps.
Over the last 16 ps the mean potential energy, $-3807.9 \pm 2.5$ against
$-3801.8 \pm 2.5$ kcal/mol, and the mean temperature, $419.6 \pm 1.0$
against $417.2 \pm 1.0$ K (errors from ten blocks), differ by 1.8 and 1.6
standard errors; the input, heating from 300 K as it relaxes, conserves
its total energy to 1.0e-3 on the CPU and 7.7e-4 on the GPU.

**The GPU against the CPU at 300 K** (2026-10-02,
`scripts/validation/gpu-cpu/nvt.py`). The run above heats as it relaxes,
and the means of a run that is not stationary are not comparable. The
same system, equilibrated for 100 ps at 300 K and 1 atm on the GPU, then
500 ps at constant volume on each in double precision (2 fs, SHAKE and
SETTLE, the thermostat every 10 steps with a seed of its own, so that the
two are independent), errors from samples spaced by their statistical
inefficiency (65 and 133 independent samples of the potential energy):
the mean potential energy −4763.3 ± 4.1 against −4759.2 ± 2.6 kcal/mol
(−0.84 standard errors), its standard deviation 32.6 ± 2.9 against 30.1 ±
1.9 (+0.72), the mean temperature 300.17 ± 0.72 against 301.03 ± 0.48 K
(−1.00), the mean total energy −4055.5 ± 5.9 against −4049.4 ± 3.6
(−0.88). The first run of this comparison, at the volume of tleap
(about 0.6 g/cm³), relaxed over the whole 500 ps; its CPU run also
found D117, a crash at the end of 250,000 steps.

## 3. From the roadmap

What the roadmap held of these results, moved here when its rows became
state and next steps (#603).

### The goal of the first milestone

The order: leapfrog, then performance. The goal of performance for the
first milestone is at least the rate of pmemd.cuda (Amber 26, SPFP), in
mixed precision, on each system of the Amber suite (from 23,558 to
1,067,095 atoms, NVE and NPT) on the same GPU; the white paper begins when
it is reached. (Changed 2026-09-30 from 90% of GROMACS with CUDA:
pmemd.cuda keeps a neighbor list as MDIR does, with a skin and a test of
displacements, so the comparison measures the kernels; what GROMACS gains
with a thermostat comes from a list every 50 steps and dynamic pruning,
which is an item of its own.) On 2026-09-30, MDIR against pmemd.cuda:
JAC 84–85% (NVE), 63–64% (NPT); FactorIX 60%, 47%; Cellulose 49%, 40%;
STMV 38%.

The goal was reached on 2026-10-02 (D114): MDIR runs every system of the
suite at 102% to 127% of the rate of pmemd.cuda, at 103% to 132% after D118, and at 104% to 132% as one model with pmemd.cuda, with all the dihedral terms of Factor IX (D133, D134; white paper, Table 10.2).
The rates of the first comparison, kept for the record (2026-09-30,
ns/day, RTX 3090):

| System | Atoms | MDIR | GROMACS (CUDA) | pmemd.cuda (published) |
|---|---|---|---|---|
| jac_nve | 23,558 | 212 | 747 | 632 |
| jac_nve_4fs | 23,558 | 400 | 795 | 1197 |
| factorix_nve | 90,906 | 51 | 248 | 265 |
| cellulose_nve | 408,609 | 9 | 51 | 63 |
| stmv_npt_4fs | 1,067,095 | 5 | 40 | 39 |

### The barostat with positional restraints (D124)

Fixed 2026-10-02 (D124). Ubiquitin in OPC from the end of the tutorial's stage 3: with the restraints the cell stayed at 200,900 Å³, without them it went to 196,300. The references, scaled about the origin with the cell (D74), had shrunk with it (radius of gyration of the restrained atoms 11.33 Å against 11.73 for the file), and the restraints pushed outward. The center of the references now follows the cell and their offsets stay (`reference_scaling = "CENTER"`, the default): 196,262 ± 580 Å³ with the restraints, 196,293 ± 658 without; stage 3 of the tutorial ends at 195,700 Å³ instead of 201,000. The bilayer's stage 3, whose restraints are spread through the cell, takes `"ALL"`

### Physical validity

Done: the mean and the variance of the kinetic energy and the ensemble tests at two temperatures and two pressures on OPC water (white paper, Section 9.5), isotropic. Done (2026-10-02): the Kolmogorov–Smirnov test of the whole distribution of the kinetic energy against the gamma distribution of the bath, $p = 0.26$ and 0.61 at 300 and 306 K (`scripts/validation/ensembles/analyze.py`). Done (2026-10-02): the test at two pressures with semi-isotropic coupling, within 0.08 standard errors, and with the height held, within 0.36, on water, whose volume (not shape) has a distribution (white paper, Section 9.5). Next, in order: a test at two tensions, which needs a bilayer and runs of 100 ns or more; (3) frames of the velocities and equipartition between solute and solvent and among translation, rotation, and internal motion, where the constraints of mixed precision would show; (4) the fluctuation of the total energy at constant energy against the step, $\sigma_E \propto \Delta t^2$, with the forces switched to zero at the cutoff, since a plain cutoff breaks it. The `physical_validation` package of that paper takes the series as they are, through an adapter for the log and the frames of MDIR

### Other engines on MDBench

Preliminary (2026-10-02, one run each on GPU 1 beside another job, 30 ps at 1 fs; `scripts/benchmarks/mdbench/run.py`): 6n4o (Argonaute2 with miR-122, ff14SB and OL3 in TIP3P, 239,131 atoms, a cutoff of 8 Å), each engine with the input of MDBench and MDIR with its own couplings: MDIR 55.1 ns/day, GROMACS 2026.3 53.0 (over all steps, with its tuning of PME), pmemd.cuda 26 43.3, OpenMM 8.6.1 32.5 (mixed). Next: three repeats on GPU 0 alone, and 9naw (3,023,780 atoms). Done (2026-10-02): 2 ns of each with a frame every 10 ps (`run.py --frames 10000 --steps 2000000`), compared by `structure.py` with cpptraj: the radius of gyration of the complex is 30.9 to 31.3 Å in all four, the RMSF of the CA 0.77 to 0.85 Å, and the profiles of the RMSF by residue correlate between MDIR and each other engine (0.86 to 0.92) as the others do among themselves (0.79 to 0.95). MDIR's backbone RMSD over the second ns is the largest, 2.00 Å against 1.34 to 1.56, but over the 711 residues whose RMSF is below 1.2 Å in every engine it is 1.25 against 0.92 to 1.11: the rest is in the flexible loops, and one run of each cannot tell it from the spread between runs. Next: three runs of each with other seeds, MDIR and GROMACS first (both stochastic velocity rescaling). The volumes differ: over the second ns MDIR 2399.7 nm³, OpenMM 2402 (a Monte Carlo barostat, which takes energies, not the virial), pmemd.cuda 2432.6, GROMACS 2453.6 (the input of MDBench corrects the energy for the dispersion, not the pressure, `DispCorr = Ener`). At the cell of the restart, held fixed for 20 ps, MDIR's pressure is −133.1 ± 6.5 bar, GROMACS's with `EnerPres` −135.0 ± 6.7 (its correction −409.168 bar, MDIR's −409.167), and pmemd.cuda's +101.2 ± 8.2: pmemd.cuda is 234 bar above the others here, 1.3% in volume. On TIP3P water at 8 and 10 Å the two agree within 8 bar, and on ubiquitin in OPC within 67 ± 36; the 13,707 particles of the protein of 6n4o span 110 Å of a cell of 134 Å, and pmemd.cuda takes the virial of molecules. Not examined further

### Analysis tools on MDBench

Measured on 2026-10-02 on the 2 ns of 6n4o of MDBench (239,131 particles,
200 frames; `scripts/benchmarks/analysis/tools.py`): the same work, a fit
of the backbone of the protein, its RMSD per frame, the RMSF of each CA,
and the radius of gyration of its heavy atoms, read from the 574 MB DCD of
MDIR in the page cache. cpptraj [[Roe2013]](references.md#roe2013) took 0.5 s,
MDTraj [[McGibbon2015]](references.md#mcgibbon2015) 3.4 s, and MDAnalysis
[[MichaudAgrawal2011]](references.md#michaudagrawal2011) 6.9 s; the three agree to
5 × 10⁻⁵ Å, the rounding of frames in `f32`. GROMACS's `rms`, `rmsf`, and
`gyrate` took 8.8 s on its own XTC of the same length, three passes that
each decompress it. Simple observables run at the speed of reading the
file (1.1 GB/s for cpptraj here).

What went wrong in the comparisons of this work was never the arithmetic:

- `unwrap` of cpptraj from a reference in another image gave the RNA of
  6n4o an RMSD of 20 to 35 Å and the complex a radius of gyration of 64 Å,
  against 1.3 to 1.9 and 31, without a warning. MDIR writes the image of
  its sorted state, with the RNA 178 Å from the protein; the restart of
  MDBench has it in another; GROMACS writes whole molecules. Imaging each
  molecule next to the protein (`autoimage`) gave the right numbers.
- GROMACS writes a frame of step 0 and the others do not, so frames of the
  same index differ by one interval.
- The columns of `gmx energy` and of ParmEd's reporter for OpenMM come in
  their own order, not that of the request; they were misread three times.
- Definitions differ between programs: the weights of the radius of
  gyration, the reference of a fit, nm against Å.
