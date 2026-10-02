# Roadmap

Status: 2026-10-02. The stages of the first milestone are in
[design-m1.md](design-m1.md), Section 18; the principles that the work
follows are in [principles.md](principles.md).

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
| A build of the driver and runtime under AddressSanitizer and UndefinedBehaviorSanitizer | Done (`scripts/build-sanitized.sh`; the tests pass) |

## 2. The rest of the first milestone

| Stage | Work |
|---|---|
| M1e | What remains of the readers and renumbering (design-m1.md, Section 18) |
| M1f | The terms and dynamics of the intermediate stage against AmberTools and GROMACS |
| M1k | The Amber suite against pmemd.cuda (published) and GROMACS 2026.3 with CUDA, on an RTX 3090: energies term by term against sander at the start, conservation and ensembles over runs, and rates |
| Integrators | Done: leapfrog does what velocity Verlet does: constraints (SHAKE, SETTLE), virtual sites, the thermostats, the barostat, restraints (D76) |
| Comparison on a protein | Done (2026-10-02, `scripts/validation/protein`): ubiquitin in 5700 OPC waters with amber19sb.ff from `pdb2gmx` (ParmEd collapsed the 13 residue-specific maps of CMAP into one and was set aside). The terms agree with a rerun of GROMACS 2026.3 to 1.7e-6 each, Coulomb to 1.9e-6 against its tabulated kernels; the reader now puts the atoms of dihedrals in the order of LEaP, as grompp does. MDIR's rate is 70% of GROMACS's at constant energy and 64% at constant pressure: its loop over pairs is as fast and its PME faster, and the rest of the step runs in series on the device (white paper, Section 10.6) |
| The rest of the step against GROMACS | In part (D118): the flags in mapped memory of the host took the idle time of the device from 49 to 29 µs a step on ubiquitin (585 to 609 ns/day) and raised JAC by 6 to 7%. Next: the fused integration kernel (14 µs, 130 registers) and the kernel of the tuples across warps (13 µs in one block; spread over 47 blocks it gained what the smaller list of another run lost), which need the counters of Nsight Compute; the 0.75 µs between each of 32 launches (CUDA graphs) |
| Performance | Done: PME in `f32` (D78), positions converted once (D79), valid structures across scalings (D80), PME on a second stream (D81, off by default; D87 checks what runs beside it), the groups of the constraints as a disjoint union run at once (D83). Next: one kernel for the kick, drift, and constraints of each group; effect summaries of loops; host synchronization moved to the device (the test of validity and the decision to rebuild, one copy to the host per step now); CUDA graphs; the tile structure (D82, [tiles-m1.md](tiles-m1.md)), stages T1 to T4 |

| Barostat integrators | Done: the strain stepped in $\lambda = \sqrt V$ (eq. S7 of [[Bernetti2020]](references.md#bernetti2020)), which makes the exact work of D77 the paper's reversible integrator. Done: its Trotter integrator (SI Sec. V.C), the default, which needs no evaluation after a scaling (D92). The count of its work takes the virial of the groups of the evaluations (D116), which removed a drift of −230 kcal/mol/ns with rigid groups. Done: the ensemble test of two pressures (white paper, Section 9.5). Next: its effective energy (eq. S11) as a diagnostic, which the virial of the groups of D116 now defines the same before and after a scaling |
| CHARMM force fields (later) | For CHARMM36 lipids and proteins: the switch of the Lennard-Jones force from 10 to 12 Å in runs from a topology, Urey–Bradley angles (function 5 of GROMACS), and NBFIX pairs |
| PME grid axes | TODO: the longest edge as the contiguous axis of the grid, along which the real-to-complex transform runs: cuFFT on an RTX 3090 takes 277 µs for the pair of transforms of 126 × 126 × 270 against 292 for 270 × 126 × 126 (Cellulose; 255 against 275 for 128 × 128 × 256). The spreading, the weights, the tables of D104, the products, and the gathering then take the axes permuted; nothing for a cubic cell (STMV) |
| Dual pair lists | Done (D114): an outer list of groups with a skin of 3 Å and an inner one pruned from it with 0.6 to 1 Å, whenever its test asks; 4 to 8 % on the Amber suite |
| Physical validity (Merz and Shirts 2018 [[Merz2018]](references.md#merz2018)) | Done: the mean and the variance of the kinetic energy and the ensemble tests at two temperatures and two pressures on OPC water (white paper, Section 9.5), isotropic. Done (2026-10-02): the Kolmogorov–Smirnov test of the whole distribution of the kinetic energy against the gamma distribution of the bath, $p = 0.26$ and 0.61 at 300 and 306 K (`scripts/validation/ensembles/analyze.py`). Done (2026-10-02): the test at two pressures with semi-isotropic coupling, within 0.08 standard errors, and with the height held, within 0.36, on water, whose volume (not shape) has a distribution (white paper, Section 9.5). Next, in order: a test at two tensions, which needs a bilayer and runs of 100 ns or more; (3) frames of the velocities and equipartition between solute and solvent and among translation, rotation, and internal motion, where the constraints of mixed precision would show; (4) the fluctuation of the total energy at constant energy against the step, $\sigma_E \propto \Delta t^2$, with the forces switched to zero at the cutoff, since a plain cutoff breaks it. The `physical_validation` package of that paper takes the series as they are, through an adapter for the log and the frames of MDIR |
| Periodic reordering of the state | TODO: the state is sorted in space only where a run begins (white paper, Section 4.2); the loops over pairs read the order of the last build (D86), but the gathers of positions and the scatters of forces follow the order of the state, which diffusion scrambles (water moves about 37 Å in 1 ns). Measured so far: JAC fell from 726 to 722 ns/day over 2 ns, 0.5%, then stayed flat (Section 10.5). First measure the loss over 10 ns of a large system (STMV, or 6n4o of MDBench at 239,131 atoms); if it exceeds about 1%, permute the state and renumber the relations every N builds, as GROMACS sorts its local state at each search |
| Pairs and lists, next | Measure first: the mean count of bits of the masks of the inner list, the share of the 16 pairs of an entry that are work (the cost of groups over single pairs), and whether a pruning is bound by its loads or its arithmetic. Then: in the pruning, a test of the box of the group that keeps or drops an entry whole when its candidate is within $R_\text{in} - a$ or beyond $R_\text{in} + a$ of the center, $a$ the half-diagonal of the box (about 4.7 Å for 16 atoms of water), before its 16 distances; by the volume of the shell, a third or more of the outer entries lie beyond, and a pruning is about 6% of a step of Cellulose. The forces of the loop over groups accumulated in 64-bit fixed point, as the grid of PME (D70), which makes groups deterministic (white paper, Section 13). Not now: lists of clusters along a Hilbert curve with indices stored as deltas in nibbles [[Thaler2026]](references.md#thaler2026), 3.6 bytes a particle against 12 for GROMACS, but on a GH200 no faster than GROMACS and slower than a full list; the loop over groups is bound by the pipe of loads and shuffles (92% of its peak on Cellulose, white paper, Section 10.2), not by the bytes of its list, and the lists fit (Cellulose: 153,517 blocks of 64 entries of 8 bytes, 190 bytes a particle). For many devices and systems of hundreds of millions |
| Other engines on MDBench | Preliminary (2026-10-02, one run each on GPU 1 beside another job, 30 ps at 1 fs; `scripts/benchmarks/mdbench/run.py`): 6n4o (Argonaute2 with miR-122, ff14SB and OL3 in TIP3P, 239,131 atoms, a cutoff of 8 Å), each engine with the input of MDBench and MDIR with its own couplings: MDIR 55.1 ns/day, GROMACS 2026.3 53.0 (over all steps, with its tuning of PME), pmemd.cuda 26 43.3, OpenMM 8.6.1 32.5 (mixed). Next: three repeats on GPU 0 alone, and 9naw (3,023,780 atoms) |
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
suite at 102% to 127% of the rate of pmemd.cuda, and at 103% to 132% after D118 (white paper, Table 10.2).
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

## 4. Later (TODO)

| Item | Notes |
|---|---|
| A Python API whose buffers follow DLPack | The state (positions, velocities, forces) and the fields shared with frameworks such as PyTorch and JAX without copies, through `__dlpack__` and `__dlpack_device__`, in both directions (for example forces from a learned potential into a step). To decide: the order of the particles when the run keeps them in the order of their positions (a permuted view, or the numbers of the particles alongside), the lifetime of a buffer that the caching allocator of the runtime owns, the stream on which a consumer may read, and the types of the mixed mode (forces in `f32`, the state in `f64`) |
