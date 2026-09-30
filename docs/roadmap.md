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
| Performance | Host synchronization moved to the device (the test of validity and the decision to rebuild), fusion of the loops over particles, CUDA graphs, PME in single precision for the mixed mode (an experiment), the pair kernel |

| Barostat integrators | The strain stepped in λ = √V (eq. S7 of [[Bernetti2020]](references.md#bernetti2020)), which makes the exact work of D77 the paper's reversible integrator, with its effective energy (eq. S11) as a diagnostic; then its Trotter integrator (SI Sec. V.C), which needs no evaluation after a scaling; the ensemble test of two pressures (SI Fig. S6) for the white paper |
| CHARMM force fields (later) | For CHARMM36 lipids and proteins: the switch of the Lennard-Jones force from 10 to 12 Å in runs from a topology, Urey–Bradley angles (function 5 of GROMACS), and NBFIX pairs |
| Semi-isotropic barostat | Eqs. (9a, 9b) and SI Secs. VI–VII of [[Bernetti2020]](references.md#bernetti2020): strains of the area and of the height from their own pressures and noises, a surface tension, and a frozen height; the kinetic energy per direction, scaling per axis in the Trotter step and in the exact work, the reference of restraints per axis. Validated on a POPC bilayer of Lipid21 from packmol-memgen against GROMACS, and on water, where it must sample what the isotropic barostat does |

The order: leapfrog, then performance. The goal of performance for the
first milestone is 70% of the rate of GROMACS with CUDA, in mixed
precision, on each system of the Amber suite (from 23,558 to 1,067,095
atoms, NVE and NPT) and on the target of D65; the white paper begins when
it is reached.

The rates of the first comparison (2026-09-30, ns/day, RTX 3090):

| System | Atoms | MDIR | GROMACS (CUDA) | pmemd.cuda (published) |
|---|---|---|---|---|
| jac_nve | 23,558 | 212 | 747 | 632 |
| jac_nve_4fs | 23,558 | 400 | 795 | 1197 |
| factorix_nve | 90,906 | 51 | 248 | 265 |
| cellulose_nve | 408,609 | 9 | 51 | 63 |
| stmv_npt_4fs | 1,067,095 | 5 | 40 | 39 |

## 3. White paper, after the first milestone

A paper that describes MDIR and what the first milestone shows, written
when the milestone ends and the rates reach 70% of GROMACS across the
systems:

- The design: the levels of the IR (`md`, `dyn`, `md_exec`), compilation of
  each run before it runs, the lowerings to CPUs and GPUs, the runtime.
- Correctness: agreement of the terms with sander and GROMACS, conservation
  of energy, the ensembles (the density of OPC water, the distributions of
  the thermostat and the barostat), the target of D65 (ff19SB in OPC).
- Performance: the Amber suite against GROMACS and pmemd.cuda, with the
  scripts that produce every number and figure.
- The principles of development and the defects that led to them.
- Limitations and the next milestones.

Every citation is checked against its source (docs/references.md), and the
figures are generated from the logs by scripts in the repository.
