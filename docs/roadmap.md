# Roadmap

Status: 2026-10-03. The stages of the first milestone are in
[design-m1.md](design-m1.md), Section 18; the principles that the work
follows are in [principles.md](principles.md).

**Milestones** (D169):

| Milestone | Scope | Status |
|---|---|---|
| M1 | An all-atom protein in water with an Amber force field, on one node, CPU and GPU; at least the rate of pmemd.cuda on every system of the Amber suite | Done (D114, 2026-10-02); release in preparation |
| M2 | The Python API (Section 6) | Next |
| M3 | Learned potentials on one GPU: ML1, then the single-GPU stage of ML4 (Section 7) | Planned |
| M4 | Distributed execution: ML2, ML3, ML5, and distributed particle mesh Ewald (Section 7) | Planned |

Features of classical MD (Section 5) continue between milestones.

## 1. Robustness (under way)

| Item | State |
|---|---|
| Principles of development | Done ([principles.md](principles.md)) |
| Tests under compute-sanitizer; short runs of the Amber suite | Done (`test/Sanitizer`, `test/Scale`) |
| Kernels named after their op and line | Done |
| Reports of defects that can be reproduced: `mdir bug-report`, crash reproducers of the passes, versions, a template for issues | Done ([debugging.md](debugging.md)) |
| The set of each buffer in the type of the storage form; memory planning as a pass of its own | Design first, then implementation |
| A summary of the effects of each loop op, from which fusion decides | After the storage types |
| Copies between host and device ordered by async tokens in the IR | After the effects |
| A debug lowering that checks bounds and the residuals of iterative solvers | With the storage types |
| The barostat with positional restraints: a defect | Fixed 2026-10-02 (D124). Ubiquitin in OPC from the end of the tutorial's stage 3: with the restraints the cell stayed at 200,900 Å³, without them it went to 196,300. The references, scaled about the origin with the cell (D74), had shrunk with it (radius of gyration of the restrained atoms 11.33 Å against 11.73 for the file), and the restraints pushed outward. The center of the references now follows the cell and their offsets stay (`reference_scaling = "CENTER"`, the default): 196,262 ± 580 Å³ with the restraints, 196,293 ± 658 without; stage 3 of the tutorial ends at 195,700 Å³ instead of 201,000. The bilayer's stage 3, whose restraints are spread through the cell, takes `"ALL"` |
| The conserved energy of the Trotter count under a fast compression | Closed 2026-10-02, not a defect: the offset of half a step between the virials of the count and the scaling (D92), which the trajectory does not depend on. The peptide of D65 on a GPU in double precision, $\tau_P$ = 0.5 ps, conserved energy over the run: from an equilibrated cell, 20 ps, every count within ±0.07 kcal/mol/ps of 0 for periods of 1, 5, and 10 steps (at constant volume −0.001), the Trotter count fluctuating more (0.6 to 0.9 kcal/mol against 0.24); compressed from the file by 12 to 15% in 2 ps, the Trotter count below the exact one by 4.2 kcal/mol at 1 fs whatever the period, 2.1 at 0.5 fs, 1.9 at 0.25 fs with a period of 10 fs and 1.1 with one of 1 fs, while the run at constant volume from the same file drifts by −2.1 kcal/mol/ps of its own. A run that changes its volume fast is checked with `work = "EXACT"` |
| A build of the driver and runtime under AddressSanitizer and UndefinedBehaviorSanitizer | Done (`scripts/build-sanitized.sh`; the tests pass) |

## 2. The rest of the first milestone

| Stage | Work |
|---|---|
| M1e | Done (2026-10-02): the readers and renumbering; the format before Amber 7 and Amber's conventions (D133); flexible water at constant energy against sander, second order down to 0.0625 fs with the terms shifted at the cutoff (design-m1.md, Section 18) |
| M1f | Done: the terms and dynamics of the intermediate stage against AmberTools, GROMACS, and CHARMM (white paper, Section 9) |
| M1k | Done (2026-10-03): the Amber suite term by term at the start against pmemd 26 and sander (D133), conservation over runs, the volumes at constant pressure of JAC and Factor IX against OpenMM, pmemd.cuda, and GROMACS (within 0.05% of OpenMM's Monte Carlo barostat), and rates as one model with pmemd.cuda, 104% to 132% (D134; white paper, Sections 9 and 10) |
| Integrators | Done: leapfrog does what velocity Verlet does: constraints (SHAKE, SETTLE), virtual sites, the thermostats, the barostat, restraints (D76) |
| Comparison on a protein | Done (2026-10-02, `scripts/validation/protein`): ubiquitin in 5700 OPC waters with amber19sb.ff from `pdb2gmx` (ParmEd collapsed the 13 residue-specific maps of CMAP into one and was set aside). The terms agree with a rerun of GROMACS 2026.3 to 1.7e-6 each, Coulomb to 1.9e-6 against its tabulated kernels; the reader now puts the atoms of dihedrals in the order of LEaP, as grompp does. MDIR's rate is 70% of GROMACS's at constant energy and 64% at constant pressure: its loop over pairs is as fast and its PME faster, and the rest of the step runs in series on the device (white paper, Section 10.6) |
| The rest of the step against GROMACS | In part (D118): the flags in mapped memory of the host took the idle time of the device from 49 to 29 µs a step on ubiquitin (585 to 609 ns/day) and raised JAC by 6 to 7%. Next: the fused integration kernel (14 µs, 130 registers) and the kernel of the tuples across warps (13 µs in one block; spread over 47 blocks it gained what the smaller list of another run lost), which need the counters of Nsight Compute; the 0.75 µs between each of 32 launches (CUDA graphs) |
| Performance | Done: PME in `f32` (D78), positions converted once (D79), valid structures across scalings (D80), PME on a second stream (D81, off by default; D87 checks what runs beside it), the groups of the constraints as a disjoint union run at once (D83). Next: one kernel for the kick, drift, and constraints of each group; effect summaries of loops; host synchronization moved to the device (the test of validity and the decision to rebuild, one copy to the host per step now); CUDA graphs; the tile structure (D82, [tiles-m1.md](tiles-m1.md)), stages T1 to T4 |

| Barostat integrators | Done: the strain stepped in $\lambda = \sqrt V$ (eq. S7 of [[Bernetti2020]](references.md#bernetti2020)), which makes the exact work of D77 the paper's reversible integrator. Done: its Trotter integrator (SI Sec. V.C), the default, which needs no evaluation after a scaling (D92). The count of its work takes the virial of the groups of the evaluations (D116), which removed a drift of −230 kcal/mol/ns with rigid groups. Done: the ensemble test of two pressures (white paper, Section 9.5). Next: its effective energy (eq. S11) as a diagnostic, which the virial of the groups of D116 now defines the same before and after a scaling |
| CHARMM force fields | Done (2026-10-02, D121, [charmm-m1.md](charmm-m1.md)): from a topology of GROMACS, Urey–Bradley angles (function 5), harmonic impropers (function 2), and `lennard_jones_modifier = "POWER_FORCE_SWITCH"`, CHARMM's VFSWITCH [[Steinbach1994]](references.md#steinbach1994), which is not the force switch of GROMACS; NBFIX and the special 1-4 parameters were in place. Trp-cage in CHARMM36m agrees with CHARMM 51b1 to its printed digits in the bonded terms and to 2.2e-7 in the Lennard-Jones; charmm27.ff with GROMACS within 4.3e-6. Done (2026-10-02, D122): the files of CHARMM themselves, a PSF of the XPLOR kind, its CRD, and the files of topology, parameters, and streams; Lennard-Jones within 5e-10 of CHARMM on Trp-cage, two POPC of CHARMM36 to the printed digits (MDIR keeps the Coulomb constant of CODATA, 2.38e-5 below CHARMM's). Next: triclinic cells (F2; CHARMM-GUI's hexagonal membranes and octahedral solutes, in CHARMM's symmetric frame), the cell from CHARMM-GUI's files, VSWITCH, the lone pairs of CGenFF, and a bilayer of CHARMM-GUI at constant pressure against GROMACS and CHARMM |
| PME grid axes | TODO: the longest edge as the contiguous axis of the grid, along which the real-to-complex transform runs: cuFFT on an RTX 3090 takes 277 µs for the pair of transforms of 126 × 126 × 270 against 292 for 270 × 126 × 126 (Cellulose; 255 against 275 for 128 × 128 × 256). The spreading, the weights, the tables of D104, the products, and the gathering then take the axes permuted; nothing for a cubic cell (STMV) |
| Dual pair lists | Done (D114): an outer list of groups with a skin of 3 Å and an inner one pruned from it with 0.6 to 1 Å, whenever its test asks; 4 to 8 % on the Amber suite |
| Physical validity (Merz and Shirts 2018 [[Merz2018]](references.md#merz2018)) | Done: the mean and the variance of the kinetic energy and the ensemble tests at two temperatures and two pressures on OPC water (white paper, Section 9.5), isotropic. Done (2026-10-02): the Kolmogorov–Smirnov test of the whole distribution of the kinetic energy against the gamma distribution of the bath, $p = 0.26$ and 0.61 at 300 and 306 K (`scripts/validation/ensembles/analyze.py`). Done (2026-10-02): the test at two pressures with semi-isotropic coupling, within 0.08 standard errors, and with the height held, within 0.36, on water, whose volume (not shape) has a distribution (white paper, Section 9.5). Next, in order: a test at two tensions, which needs a bilayer and runs of 100 ns or more; (3) frames of the velocities and equipartition between solute and solvent and among translation, rotation, and internal motion, where the constraints of mixed precision would show; (4) the fluctuation of the total energy at constant energy against the step, $\sigma_E \propto \Delta t^2$, with the forces switched to zero at the cutoff, since a plain cutoff breaks it. The `physical_validation` package of that paper takes the series as they are, through an adapter for the log and the frames of MDIR |
| Periodic reordering of the state | TODO: the state is sorted in space only where a run begins (white paper, Section 4.2); the loops over pairs read the order of the last build (D86), but the gathers of positions and the scatters of forces follow the order of the state, which diffusion scrambles (water moves about 37 Å in 1 ns). Measured so far: JAC fell from 726 to 722 ns/day over 2 ns, 0.5%, then stayed flat (Section 10.5). First measure the loss over 10 ns of a large system (STMV, or 6n4o of MDBench at 239,131 atoms); if it exceeds about 1%, permute the state and renumber the relations every N builds, as GROMACS sorts its local state at each search |
| Pairs and lists, next | Measure first: the mean count of bits of the masks of the inner list, the share of the 16 pairs of an entry that are work (the cost of groups over single pairs), and whether a pruning is bound by its loads or its arithmetic. Then: in the pruning, a test of the box of the group that keeps or drops an entry whole when its candidate is within $R_\text{in} - a$ or beyond $R_\text{in} + a$ of the center, $a$ the half-diagonal of the box (about 4.7 Å for 16 atoms of water), before its 16 distances; by the volume of the shell, a third or more of the outer entries lie beyond, and a pruning is about 6% of a step of Cellulose. The forces of the loop over groups accumulated in 64-bit fixed point, as the grid of PME (D70), which makes groups deterministic (white paper, Section 13). Not now: lists of clusters along a Hilbert curve with indices stored as deltas in nibbles [[Thaler2026]](references.md#thaler2026), 3.6 bytes a particle against 12 for GROMACS, but on a GH200 no faster than GROMACS and slower than a full list; the loop over groups is bound by the pipe of loads and shuffles (92% of its peak on Cellulose, white paper, Section 10.2), not by the bytes of its list, and the lists fit (Cellulose: 153,517 blocks of 64 entries of 8 bytes, 190 bytes a particle). For many devices and systems of hundreds of millions |
| Other engines on MDBench | Preliminary (2026-10-02, one run each on GPU 1 beside another job, 30 ps at 1 fs; `scripts/benchmarks/mdbench/run.py`): 6n4o (Argonaute2 with miR-122, ff14SB and OL3 in TIP3P, 239,131 atoms, a cutoff of 8 Å), each engine with the input of MDBench and MDIR with its own couplings: MDIR 55.1 ns/day, GROMACS 2026.3 53.0 (over all steps, with its tuning of PME), pmemd.cuda 26 43.3, OpenMM 8.6.1 32.5 (mixed). Next: three repeats on GPU 0 alone, and 9naw (3,023,780 atoms). Done (2026-10-02): 2 ns of each with a frame every 10 ps (`run.py --frames 10000 --steps 2000000`), compared by `structure.py` with cpptraj: the radius of gyration of the complex is 30.9 to 31.3 Å in all four, the RMSF of the CA 0.77 to 0.85 Å, and the profiles of the RMSF by residue correlate between MDIR and each other engine (0.86 to 0.92) as the others do among themselves (0.79 to 0.95). MDIR's backbone RMSD over the second ns is the largest, 2.00 Å against 1.34 to 1.56, but over the 711 residues whose RMSF is below 1.2 Å in every engine it is 1.25 against 0.92 to 1.11: the rest is in the flexible loops, and one run of each cannot tell it from the spread between runs. Next: three runs of each with other seeds, MDIR and GROMACS first (both stochastic velocity rescaling). The volumes differ: over the second ns MDIR 2399.7 nm³, OpenMM 2402 (a Monte Carlo barostat, which takes energies, not the virial), pmemd.cuda 2432.6, GROMACS 2453.6 (the input of MDBench corrects the energy for the dispersion, not the pressure, `DispCorr = Ener`). At the cell of the restart, held fixed for 20 ps, MDIR's pressure is −133.1 ± 6.5 bar, GROMACS's with `EnerPres` −135.0 ± 6.7 (its correction −409.168 bar, MDIR's −409.167), and pmemd.cuda's +101.2 ± 8.2: pmemd.cuda is 234 bar above the others here, 1.3% in volume. On TIP3P water at 8 and 10 Å the two agree within 8 bar, and on ubiquitin in OPC within 67 ± 36; the 13,707 particles of the protein of 6n4o span 110 Å of a cell of 134 Å, and pmemd.cuda takes the virial of molecules. Not examined further |
| Semi-isotropic barostat | Done (D119): `coupling = "SEMI_ISOTROPIC"`, eqs. (9a, 9b) of [[Bernetti2020]](references.md#bernetti2020) with a surface tension and a held height, the kinetic energy, the virials, and the scaling by axis. Water samples the same volume as with isotropic coupling; on a POPC bilayer of Lipid21 the split of the pressure by axis agrees with GROMACS's in one cell. The pressure of that cell differs from GROMACS's by 43 bar (0.24% in density): on TIP3P water the pressures of MDIR, GROMACS, and pmemd.cuda at 2 fs are 52.7, 34.4, and 61.6 bar and agree near 45 bar at 0.25 fs, so it is the error of each integrator's estimator of the pressure at the step, not the coupling (D119). The area per lipid needs runs of 100 ns or more to compare to 1 Å²

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

## Documentation

| Item | State |
|---|---|
| Equations in the Markdown documents written in TeX (`$...$`, `$$...$$`, which GitHub renders) instead of Unicode text | Done (2026-10-02): the equations of every document of `docs/` are TeX; numbers, units, and the sizes of grids and tiles stay as text |
| A guide for contributors: building, the layers of the IR and where a feature goes, adding a term, a pass, or a lowering, the tiers of tests and the tools of debugging, the principles, how a change is reviewed | Done: Appendix B of the white paper, and `CONTRIBUTING.md` at the root |

## 3. White paper, after the first milestone

A paper that describes MDIR and what the first milestone shows, written
when the milestone ends and the rates reach those of pmemd.cuda across the
systems. *Written 2026-10-02:* [docs/paper/](paper/README.md), in
Markdown with equations in TeX, built as a PDF with pandoc and tectonic
(`scripts/paper/build-pdf.sh`); a change to a method, an algorithm, or an
implementation updates it. The performance of the suite is measured
against pmemd.cuda (the user, 2026-10-01); the comparison on a protein
(Section 10.6) is against GROMACS. The outline:

- A section of notation at the front (Done, Section 2): the symbols of positions,
  cells, images and their shifts, forces, the virial and its sign, units,
  and the types of the precision modes, fixed before the chapters and used
  in all of them.
- Every method that MDIR implements, with its equations as implemented
  (TODO: keep the design documents complete in TeX as features land).
- Usage (Done, 2026-10-03: Appendix C, [C-usage.md](paper/C-usage.md)):
  how a researcher runs MDIR, beside the reference of the control file
  (Appendix A): building and checking an installation, the stages of the
  standard pipeline as control files, inputs from Amber, GROMACS, and
  CHARMM, continuing a run on a cluster (U1 to U4), the choice of target
  and precision, terms given by expressions, implicit solvent, and
  failures, each command run at the commit of the paper. Its section on
  the outputs follows U12 when that lands.
- The design: the levels of the IR (`md`, `dyn`, `md_exec`), compilation of
  each run before it runs, the lowerings to CPUs and GPUs, the runtime.
- The neighbor algorithms in detail (Done, Section 4): the matrix (neighbors-m0.md)
  and the groups of 16 (groups-m1.md, D89 to D106): the compact order of
  the places, the frames that take no minimum image (D95), the masks and
  their ballots (D97), the queue of candidates and the ranges of partners
  (D99, D100), the excluded pairs in the masks and their fallback (D105,
  D106), the blocks of 64 entries and their pool, the exact test of
  validity under scaling (D80) and of the dual list with its proof
  (tiles-m1.md, Section 6), and why the tiles of D82 were set aside; with
  the measurements behind each choice.
- Correctness: agreement of the terms with sander and GROMACS, conservation
  of energy, the ensembles (the density of OPC water, the distributions of
  the thermostat and the barostat), the target of D65 (ff19SB in OPC).
- Performance: the Amber suite against GROMACS and pmemd.cuda, with the
  scripts that produce every number and figure.
- The way from double to mixed precision (Done, Section 7): what stays in `f64` and
  why (the differences of positions, D75; the sums), what moved to `f32`
  (the kernels of pairs, PME), and what each step cost and gained.
- Related work (Done, Section 11): every source of the list that the
  outline began with was read at its source on 2026-10-02.
  chemtrain-deploy, JAX MD, Reactant.jl, FFTc (also diva2:1757681),
  LAPIS (arXiv 2509.25605), the activity analysis of automatic
  differentiation on MLIR (doi:10.1145/3763125), mlip (arXiv 2505.22397),
  and MDcraft (arXiv 2511.22951) are cited. MLIR-AIR (arXiv 2510.14871,
  AI workloads on AMD NPUs) and SODA-OPT (high-level synthesis) target
  spatial hardware and concern neither neighbor lists nor molecular
  dynamics; they are not cited. The point the sources bear out: these
  compile tensor programs, linear algebra, or potentials, or wrap learned
  potentials in an engine, while MDIR has dialects of molecular dynamics
  itself (particle sets, neighbor structures, tuples, integrators) and
  lowers the whole step.
- The principles of development and the defects that led to them.
- A manual of the control file (Done, Appendix A, checked by
  `scripts/paper/check-appendix.sh`): every table and keyword, its
  units and default, and the combinations that are errors, checked
  against `mdir template` so that it follows the code.
- How to take part: how the code is laid out, how a new term, pass, or
  target is added and tested, and how defects are reported, so that a
  reader can begin to contribute.
- Limitations and the next milestones.

Every citation is checked against its source (docs/references.md), and the
figures are generated from the logs by scripts in the repository.

## 4. Usability (after the first milestone)

From a review of how a researcher meets MDIR (2026-10-02), checked against
the code. The order is that of the work; P0 is what a production run on a
cluster needs.

| # | Item | Priority | State and scope |
|---|---|---|---|
| U1 | `steps` as the total of a run, and `mdir run --continue` | P0 | Done (2026-10-02, D129): the checkpoint records the step its run began at, and `--continue` takes the steps that remain to it from the checkpoint of `[output]`, begins the run without one, and exits with 0 when it is complete, so the same command line resubmits a job until it is done; the intervals are checked against what remains. A run that begins from the checkpoint of another begins at its step, which keeps the random numbers of stages apart, so `steps` of a run that begins means what it did. The conserved energy continues across parts ([driver-m0.md](driver-m0.md), Section 2.7); since D[checkpoint-fingerprint] the checkpoint records what defined the run, `--continue` refuses other physics or coupling and names each change, and a run from another run's checkpoint evaluates its first forces where they differ |
| U2 | Output that continues with the run | P0 | Done (D130, D149): the checkpoint records the trajectory and its frames, and the part whose files the outputs go to; a continued run cuts the trajectory to its frames and the files of columns to its step, and appends, and appends the log file whole; it refuses a trajectory with fewer frames, other particles, or another interval, and a file of columns with other columns. `--no-append` writes the log, the files of columns, and the frames of the continued part to `<name>.partNNNN<ext>` |
| U3 | A stop on SIGTERM, SIGINT, or `--max-walltime` | P0 | Done (D131), without a change of the compiler: the function of the driver that writes a checkpoint ends the process after the write when a signal or the wall time asks for a stop, with the exit status 75 (`EX_TEMPFAIL`). The stop falls on a checkpoint, which keeps the continuation exact (R1); a second signal ends the run at once. The wall time leaves room for the longest interval between checkpoints so far |
| U4 | Keep the previous checkpoint | P0 | Done (D132): `<checkpoint>.prev` is a hard link to the one before, made before the rename that replaces it, so the name of the checkpoint always holds a complete state; without hard links a rename, which `--continue` detects; since D[checkpoint-format] the file is `fsync`ed before the rename and the directory after it, and the state carries its SHA-256, checked on every read |
| U5 | A directory for a run: the log in a file as well, protection against overwriting, a manifest | P1 | Done (D149, D168): `[output]` names the log and optional `manifest`; outputs use numbered backups on a fresh run. The manifest records versioned JSON Lines with build identity, input SHA-256 hashes including topology includes, resolved execution settings, device identity, warnings, and start/end events. Continuations append history, and `--no-append` uses the part file ([driver-m0.md](driver-m0.md), Section 2.8) |
| U6 | `mdir check` as a preflight | P1 | Done (2026-10-03, D151): the system and planned run (ensemble, step and length in ns, PME, constraints, target and precision), all outputs with their intervals and existing paths, warnings with recovery hints, and `--json` with structured errors and warnings. Reads without compiling, loading a checkpoint, or writing outputs |
| U7 | Errors that say what to do | P1 | Done (2026-10-03, D146): errors name the line, and enums list their values. Unknown keywords and table names suggest a unique nearest valid key within two edits, list the supported keys of the table, and point to `mdir template md` and `mdir template amber` |
| U8 | A quickstart first in the README, and `mdir doctor` | P1 | Done (D155): the README starts with a runnable argon quickstart; `doctor` reports the build, probes the CUDA driver and devices, and compiles and runs a small system on each requested target |
| U9 | Control files of the standard pipeline from `mdir template` | P2 | Done (2026-10-03, D147): `minimize`, `nvt`, `npt`, and `production` print the four stages of `examples/ala3` with placeholder input paths and their checkpoint chain. Embedded at build time; `md` and `amber` remain the reference templates, with no new command or aliases |
| U10 | Distribution | P2 | After the release of M1: an Apptainer or OCI image. A container needs the cubin of its kernels, not PTX that a driver older than the toolkit cannot compile |
| U11 | Many runs of one plan (as `-multidir`) | Later | With `ensemble` |
| U12 | Outputs as one system | P1 | Done (2026-10-03, D149; the user asked for it the same day): [output] names every file and its interval (`log`, `energy`, `pull`, which was `pull_coordinates`, `trajectory`, `checkpoint`); the files of columns share one form, a line of names, a line of units, one for each column, and rows of the step and values with six decimals; every file continues with the run (U2), and a run that is not continued keeps the outputs of an earlier one as backups, `#<name>.<n>#` (U5); `mdir check` lists each with its count and notes the backups (U6, D151). Not taken: intervals of their own for the files of columns, which the compiled schedule would need loops for |

Not taken: `target = "AUTO"` (a check would pass on one machine and fail
on another, and a large run could fall back to the CPU without a word; a
clear failure is better), runs without a number of steps (the counts of the
schedule are constants of the program), and checkpoints by the clock (they
would break the exact continuation).

## 5. Features for the purpose of MDIR

MDIR is for general MD, all-atom and coarse-grained, on workstations and
clusters ([decisions.md](decisions.md), Section 1, C1 and C2). What runs of that kind still lack, in the order of how often
they are needed:

| # | Feature | State |
|---|---|---|
| F1 | Langevin dynamics (the middle scheme, BAOAB) | Done (D135): `method = "LANGEVIN"` with `friction`, in every program of a step, with Philox in the kernel keyed by the step and the number of the particle; on the CPU and on a GPU, continued runs bitwise. Next: the energy that the friction and the noise exchange with the bath, summed in a field of each particle, for a conserved energy |
| F2 | Triclinic cells (the truncated octahedron and the rhombic dodecahedron of Amber and GROMACS, the hexagonal cells of CHARMM-GUI) | Done but the rates (D123, D125 to D127, [triclinic-m2.md](triclinic-m2.md)): read, reduced, and run on the CPU and on the device, with the matrix and the groups, at constant volume and pressure; validated against sander, GROMACS, and CHARMM 51b1 on a hexagonal cell. First rates (2026-10-02, RTX 3090, groups with a dual list, mixed precision): ubiquitin in OPC 12 Å beyond the protein, tleap's octahedron of 25,035 particles at 587 ns/day against its rectangular box of 26,031 at 581, 3% more time a particle for the minimum image and the lattice shifts, and only 4% fewer particles for this solute. Next: P5 (pmemd.cuda and GROMACS on the same systems, cells of the same least distance between images) |
| F3 | CHARMM force fields | In part: from a topology of GROMACS (D121) and from the files of CHARMM (D122), with triclinic cells (D127), validated against CHARMM 51b1. Particle mesh Ewald for the dispersion, `lennard_jones = "PME"` (D162), validated against OpenMM and GROMACS. Missing: VSWITCH, lone pairs, the Drude model |
| F4 | Outputs for analysis: frames of the velocities, XTC, the pressure tensor and the area in the log, observables in H5MD | In part: XTC (D141), with `trajectory_format` from the extension. Missing: frames of the velocities, which the test of equipartition needs, the pressure tensor and the area in the log, observables in H5MD |
| F5 | Restraints beyond positions: distance, angle, dihedral, flat-bottomed | Done (D136): terms over tuples given by expressions, [[energy.bond]], [[energy.angle]], and [[energy.dihedral]], as the custom forces of OpenMM; positional restraints keep their own table (D74). Over the centers of groups as well (D139), the restraints of pulling. A reference that moves with the time `t` and the coordinates, energies, and forces of the terms over centers written at every energy (D145). Terms of the absolute positions of single particles, with parameters of each, `[[energy.external]]` (D148) |
| F6 | Coarse-grained models: tabulated potentials, Martini from GROMACS topologies, DPD | Missing; part of the purpose from the start |
| F7 | Free energy and enhanced sampling: alchemical $\lambda$ with $dH/d\lambda$ (D2), collective variables, replica exchange | Not designed; neither [future-architecture-plan.md](future-architecture-plan.md) nor this roadmap has a design. The architecture names `ensemble` and replica exchange as a driver event (P4) |
| F8 | Learned potentials | Section 7 |
| F9 | The custom forces of OpenMM beyond tuples: pairs with a topology, tabulated functions, centers of groups (pulling), generalized Born | In progress: pair terms over the pairs of a topology, in its charges and Lennard-Jones parameters, with interaction groups (D137); tabulated functions of one argument, natural or periodic splines, in every expression (D138); terms over the centers of groups (D139), 0.21 to 0.24 ms a step on JAC with a bond and 0.26 with a dihedral since the sums of a block are added in one pass, a set of few tuples is reduced by one block, and the forces of a term are one field evaluated once per tuple (D150). The reaction field as GROMACS has it (D140); runs without a periodic cell, in a cell that no image reaches (D142). Intermediate fields, the derivative of the energy carried back through fields computed from the positions (D143), and generalized Born, OBC I and II, from a topology of Amber (D144), and HCT, salt, a cutoff of the descreening, and the radii of mbondi2 for a topology of GROMACS or CHARMM (D152). Terms over the triplets centered on each particle, the three-body term of Stillinger-Weber and mW water, `[[energy.triplet]]` without a topology on the CPU, its members written at every step from the neighbor matrix (D160); their GPU kernel, the selections `center` and `ends`, a topology, the exclusion of a far leg, and bond order (Tersoff) follow. Parameters of each particle, by masks of Amber, for the pair terms, the terms over tuples, and those of the positions, tabulated functions of two and three arguments and discrete ones, and terms over tuples of any length in their distances, angles, and dihedrals, as OpenMM's (D165). Next: the first reduction inside the loop of the sums (D150) and an open cell if a system needs one; later tables from files (the tables of GROMACS for F6) |
| F10 | Thermostats, integrators, and barostats beyond M1: a deterministic thermostat, Brownian dynamics, anisotropic and flexible cells, multiple time steps | In progress (D163): Nosé–Hoover chains (D163a), Brownian dynamics (D163b), anisotropic cell rescaling (D163c). Next, as a design only: multiple time steps (r-RESPA, [[Tuckerman1992]](references.md#tuckerman1992)), the energy split into a fast part (bonded terms, the direct sum within a short cutoff) and a slow part (the rest, the reciprocal sum), the slow forces kicking half an outer step at its ends and the fast ones integrated by velocity Verlet within it; in MDIR two `md.evaluate` of two `@energy` functions that the terms are partitioned into, a program of an outer step that loops over inner ones, and a check of the resonance of the outer step with the fastest motions, which limits it to about 4 fs with constraints; a flexible cell, with shear, needs the off-diagonal virial and the pressure tensor (F4), a decision of its own |

## 6. Python API (M2)

The design follows a reading of OpenMM's Python layer (2026-10-02,
`openmm/openmm` at 5ee2cba): its vocabulary and its reporters fit MDIR,
its copies and its silent caching do not. Both front ends, the control file
and Python, must produce the same IR and share one validation.

| Item | Design |
|---|---|
| Loading | `mdir.load_amber`, `mdir.load_gromacs`: topology, parameters, positions, cell; nothing about the run. Later an import of an OpenMM `System` for the supported subset, which reuses its force fields and its builders |
| Physics apart from execution | `System` (terms, cutoff, PME, constraints, custom potentials as expressions with per-particle and global parameters); the integrator and the ensemble; `Execution` (target, device, precision) as typed objects, not strings |
| An explicit compile | `mdir.compile(...)` returns an immutable program and its plan; changing the system afterwards marks it stale rather than being ignored. Parameters declared tunable are read from a buffer, so setting them does not recompile |
| Runs and reporters | `sim.run(n)` runs segments to the next report of any reporter (OpenMM's protocol), with the writers of the driver in C++ and the GIL released; a stop is polled between segments |
| State | `state()`: host copies in the order of the input, in the units of MD (nm, ps, kJ/mol, bar) as plain arrays. `view()`: DLPack tensors of the device in the order of the run with the index of each row, valid until the next run, on a stated stream, whose type is that of the buffer (`f32` forces in the mixed mode). Writes through a view advance the version of the state (P16) |
| Errors | Typed: input (file, line, term), compile (the diagnostic with its location), unsupported (feature, target), simulation (step, particle, quantity) |
| Checkpoints | The H5MD checkpoint of the driver, exact and portable, with the hashes of the model and the plan; one format, not two |

Still to decide for DLPack: the order of the particles when the run keeps them in the order of their positions (a permuted view, or the numbers of the particles alongside), the lifetime of a buffer that the caching allocator of the runtime owns, the stream on which a consumer may read, and the types of the mixed mode (forces in `f32`, the state in `f64`).

## 7. Learned potentials (M3) and distributed execution (M4)

D169 splits this section between two milestones: ML1 and the
single-GPU stage of ML4 close M3; ML2, ML3, ML5, and distributed particle
mesh Ewald close M4.

Classical MD remains a core use case. This extension preserves its existing
features, validation, and performance baselines, with no required MLIP
backend for classical runs. LJ/EAM are initial distributed validation
models; realistic molecular systems remain part of validation and the
long-term distributed scope.

Direction adopted in D166 and refined in D167 (2026-10-03); all milestones
below are planned, not implemented. [The architecture plan](future-architecture-plan.md#adopted-scope-and-first-deliverable)
holds the contracts and detailed sequence. The first deliverable is **one
existing local MLIP, validated on one GPU and then on two GPUs sharing one
physical system**. Independent trajectories on separate GPUs are useful
but do not validate this decomposition.

Start the optional metatomic adapter through the existing driver before
the distributed compiler and full Python API are complete. The first
scope is finite-range local interactions, a fixed cell, no constraints,
and one node. Keep the first-milestone classical MD paper's validation and
publication independent of this work. An adapter, distributed inference,
and compiler optimization are separate outcomes; none alone establishes
novelty or a speedup.

Use **Allegro → PaiNN → MACE** as the sequential support target, pinning
one existing artifact and its backend/adapter versions per architecture.
Allegro is the first local model; PaiNN introduces staged scalar/vector
message passing; MACE extends the validated stage contract. Compatibility
is checked per artifact, not promised for every checkpoint in a family.
Supporting these architectures does not by itself validate water-and-salt
properties; scientific validation is a separate application gate.

**PyTorch/JAX supplies neural-network AD and tensor execution.** MDIR
verifies particle dependencies, derivative ownership, communication,
lifetimes, and execution order. Begin with a whole-model energy/force/virial
call; staged execution later registers or reuses differentiable halo and
reverse-accumulation primitives while the external framework generates
backward. No new general reverse-mode engine is required. torch-mlir is an
optional later import route, not a dependency of ML1–ML3. Preserve semantic
contracts through external AD rather than reconstructing them from generic
tensor operations. See the [layer and model plan](future-architecture-plan.md#external-ad-and-execution-layers).

| Milestone | Work | Completion evidence |
|---|---|---|
| ML1 | One compatible Allegro artifact through the optional metatomic adapter and versioned potential/neighbor contract; retain external AD and tensor backend | On one GPU, energy, per-particle forces, and stress/virial agree with the original backend under declared tolerances; coordinate/cell finite differences and short NVE run; record artifact/backend identity, units, species, precision, copies, and synchronization; unsupported outputs rejected; classical builds still work without the adapter |
| ML2 | Dependency graph, ownership and periodic images, coverage and freshness; fixed two-domain LJ, then EAM and reverse contribution routing; CPU ranks before GPUs | Energy, forces, and virial agree with one domain; migration, rebuilds, boundaries, uneven/empty domains, and exactly-once accumulation tested; inspectable schedules and negative tests for missing transfers, stale intermediates, and invalid completion |
| ML3 | The ML1 local model on two GPUs using the ML2 ownership and transfer contract | Same artifact, weights, precision, and outputs as ML1; one/two-GPU agreement against the original backend including migration and short-run conservation; peak memory and full-step time recorded, without requiring a speedup; completes the first deliverable |
| ML4 | PaiNN with an opaque path and declared stages/differentiable communication, then MACE using the same contracts; external AD generates backward; semantic import is optional | Communication adjoint checks and same-model comparison of enlarged coordinate halos and per-layer feature exchange, including reverse derivatives and saved-value lifetimes; energy/force/virial agreement and measurements of bytes, messages, redundant work, memory, and full-step time |
| ML5 | Bounded legal-plan selection and measured overlap | Fixed-plan baselines and tuning cost reported; one/two/four-GPU strong and weak scaling; multiple nodes before a cluster claim; checked preconditions and an off switch for each optimization |

A four-GPU measurement follows ML3's one/two-GPU correctness gate; it is
not required to complete that first deliverable, nor is support for all
three architectures. ML5 can begin with the first validated staged model.
Distributed PME, global attention or charge solvers, constraints across
domains, training, arbitrary PyTorch/JAX import, and full-trajectory
differentiation remain outside it. The dependency vocabulary must leave
room for these cases without claiming support.

A persistent kernel cache and runnable compiled artifacts remain companion
work. Cache keys include the model/backend and code-affecting plan choices;
replanning that does not change generated code can reuse it. Preserve
artifact identity and restart contracts from ML1, initially with same-plan
continuation; changes of rank count require later validation.

To bring into line with the detailed plan: [architecture.md](architecture.md),
Section 6 still calls the plan an ordered list of stages, where the plan
makes the graph primary; its Section 4.3 names spherical harmonics as op
families and asks the halo to cover the receptive field of the energy,
where the plan is agnostic of the basis and separates the support of the
forces from that of the energy; its milestones (Section 12) have neither
EAM, the verifier, nor the adapter. [decisions.md](decisions.md), Section
7 still tags the dependency interface M2b, which A14 moved into M1.

## 8. Analysis

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

| # | Item | State |
|---|---|---|
| A1 | Frames correct by construction: molecules whole and the solute in one image, from the graph of bonds and constraints that MDIR holds; the step, the time, and the cell in each frame; the count of images of each molecule for an exact unwrapping (mean square displacements); velocities on request (F4) | Missing; with U2 |
| A2 | Observables of the run, compiled into the program and sharing its loops: radial distributions from the pair list, density profiles along an axis, the pressure tensor and the lateral pressure profile, order parameters, the interaction energy between two selections (protein and RNA), the distribution of the kinetic energy for the tests of [[Merz2018]](references.md#merz2018); summed in `f64`, with block averages and their errors | Missing. The interaction energy and the lateral pressure need the forces of the run, which a program of analysis has only through a rerun |
| A3 | `mdir analyze`: the same kernels over frames read from files: the energies of frames term by term (a rerun), observables of pairs on the device, the matrix of RMSDs between frames by QCP [[Theobald2005]](references.md#theobald2005) on the device | Missing |
| A4 | Exchange rather than reimplementation: frames to MDTraj, MDAnalysis, and PyTorch through the DLPack views of the Python API (Section 6); each observable of A2 and A3 validated against cpptraj and GROMACS as the terms are, its definition written down | Missing |

Not taken: reimplementing the actions of cpptraj. A device pays where pairs
are involved, where all pairs of frames are (clustering), or where the
frames never reach a file.
