# Roadmap

Status: 2026-09-30. The stages of the first milestone are in
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
| Comparison on a protein | TODO: a protein of moderate size with ff19SB in OPC water (the target of D65, or larger), converted with ParmEd, run by GROMACS with CUDA and by MDIR: rates and agreement of the terms |
| Performance | Done: PME in `f32` (D78), positions converted once (D79), valid structures across scalings (D80), PME on a second stream (D81, off by default; D87 checks what runs beside it), the groups of the constraints as a disjoint union run at once (D83). Next: one kernel for the kick, drift, and constraints of each group; effect summaries of loops; host synchronization moved to the device (the test of validity and the decision to rebuild, one copy to the host per step now); CUDA graphs; the tile structure (D82, [tiles-m1.md](tiles-m1.md)), stages T1 to T4 |

| Barostat integrators | Done: the strain stepped in $\lambda = \sqrt V$ (eq. S7 of [[Bernetti2020]](references.md#bernetti2020)), which makes the exact work of D77 the paper's reversible integrator. Done: its Trotter integrator (SI Sec. V.C), the default, which needs no evaluation after a scaling (D92). Next: its effective energy (eq. S11) as a diagnostic, which with rigid groups needs the pressure of the same definition before and after a scaling; the ensemble test of two pressures (SI Fig. S6) for the white paper |
| CHARMM force fields (later) | For CHARMM36 lipids and proteins: the switch of the Lennard-Jones force from 10 to 12 Å in runs from a topology, Urey–Bradley angles (function 5 of GROMACS), and NBFIX pairs |
| PME grid axes | TODO: the longest edge as the contiguous axis of the grid, along which the real-to-complex transform runs: cuFFT on an RTX 3090 takes 277 µs for the pair of transforms of 126 × 126 × 270 against 292 for 270 × 126 × 126 (Cellulose; 255 against 275 for 128 × 128 × 256). The spreading, the weights, the tables of D104, the products, and the gathering then take the axes permuted; nothing for a cubic cell (STMV) |
| Dual pair lists | Done (D114): an outer list of groups with a skin of 3 Å and an inner one pruned from it with 0.6 to 1 Å, whenever its test asks; 4 to 8 % on the Amber suite |
| Semi-isotropic barostat | TODO (`coupling = "SEMI_ISOTROPIC"` stops with "not supported yet"). Eqs. (9a, 9b) and SI Secs. VI–VII of [[Bernetti2020]](references.md#bernetti2020): strains of the area and of the height from their own pressures and noises, a surface tension, and a frozen height; the kinetic energy per direction, scaling per axis in the Trotter step and in the exact work, the reference of restraints per axis. Validated on a POPC bilayer of Lipid21 from packmol-memgen against GROMACS, and on water, where it must sample what the isotropic barostat does |

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

The rates of the first comparison (2026-09-30, ns/day, RTX 3090):

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
| A guide for contributors: building, the layers of the IR and where a feature goes, adding a term, a pass, or a lowering, the tiers of tests and the tools of debugging, the principles, how a change is reviewed | With the white paper (Section 3), and as `CONTRIBUTING.md` |

## 3. White paper, after the first milestone

A paper that describes MDIR and what the first milestone shows, written
when the milestone ends and the rates reach those of pmemd.cuda across the
systems:

- A section of notation at the front (TODO): the symbols of positions,
  cells, images and their shifts, forces, the virial and its sign, units,
  and the types of the precision modes, fixed before the chapters and used
  in all of them.
- Every method that MDIR implements, with its equations as implemented
  (TODO: keep the design documents complete in TeX as features land).
- The design: the levels of the IR (`md`, `dyn`, `md_exec`), compilation of
  each run before it runs, the lowerings to CPUs and GPUs, the runtime.
- The neighbor algorithms in detail (TODO): the matrix (neighbors-m0.md)
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
- The way from double to mixed precision (TODO, if it makes a story): what stays in `f64` and
  why (the differences of positions, D75; the sums), what moved to `f32`
  (the kernels of pairs, PME), and what each step cost and gained.
- Related work (TODO: survey): compilers and MLIR in molecular dynamics.
  To look up, read at the source, and verify before any claim is cited
  (the list came from a summary whose claims and links are unchecked; some
  links did not match their topics):
  - chemtrain-deploy (arXiv 2506.04055): JAX potentials (MACE, Allegro,
    PaiNN) exported as StableHLO and run inside LAMMPS through XLA/PJRT
    on many GPUs.
  - JAX-MD: differentiable MD in JAX, compiled by XLA (through StableHLO).
  - Reactant.jl with EnzymeMLIR: Julia code traced to MLIR, with automatic
    differentiation at the level of MLIR (Enzyme).
  - FFTc (doi:10.1007/978-3-031-50684-0_16; arXiv 2308.00497): an MLIR
    dialect for FFTs, generating kernels for the hardware; bears on the
    transforms of PME, for which MDIR calls cuFFT.
  - Lapis, MLIR-AIR, SODA-OPT: sparse linear algebra and hardware co-design
    on MLIR; whether any concerns neighbor lists or MD is to be checked.
  - Also given without titles: arXiv 2505.22397, 2511.22951;
    doi:10.1145/3763125; DiVA diva2:1757681.
  The point to make, if the sources bear it out: these compile potentials
  or tensor programs (mostly machine-learned) through general ML
  compilers, while MDIR has dialects of MD itself (particle sets, neighbor
  structures, tuples, integrators) and lowers the whole step.
- The principles of development and the defects that led to them.
- A manual of the control file (TODO): every table and keyword, its
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
