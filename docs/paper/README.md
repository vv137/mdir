# MDIR: A Compiler for Molecular Dynamics

*Minsoo Kim — white paper of the first milestone, October 2026*

## Abstract

MDIR compiles a run of molecular dynamics before it runs. The potential,
the integrator, the constraints, and the couplings of a run are written
in intermediate representations on MLIR: semantic dialects for what is
computed and how the state advances, an execution dialect of loops over
particles, pairs, and tuples, and a runtime dialect. Passes derive the
forces and the virial by differentiation, with exchange contracts that
make it legal to compute each pair once; carry neighbor structures across
steps under a test of validity that is exact under the scaling of a
barostat; fuse loops; assign a floating-point type to each value by its
role; and lower the result to OpenMP or to NVIDIA GPUs, specialized to
the topology and the parameters at hand. On one RTX 4090 in mixed
precision, MDIR runs the systems of the Amber GPU benchmark suite, from
23,558 to 1,067,095 atoms, at constant energy and at constant pressure,
with the steps and the output of the suite's own inputs: from 1277 ns/day
on JAC (23,558 atoms, 2 fs; 2406 ns/day at 4 fs) to 83 ns/day on STMV
(1,067,095 atoms, 4 fs). [Section 10](10-performance.md) measures it
against the target of the first milestone (D114), the rate of pmemd.cuda
26, with both programs under one protocol on the same device, and against
GROMACS 2026.3 on ubiquitin in OPC. The paper derives the methods as
they are implemented, among them groups of 16 particles that share a
list, a dual list pruned by its own exact test, and an analysis of a
drift of the energy that showed that a closed-form solver of rigid water
loses in single precision what an iterative one keeps.

## Contents

1. [Introduction](01-introduction.md)
2. [Notation](02-notation.md)
3. [Design](03-design.md): the levels of the IR, differentiation and
   exchange contracts, the passes, the lowerings, compilation at run time
4. [Neighbor structures](04-neighbors.md): the test of validity under
   scaling, the order of the particles, the matrix, groups of 16, the
   dual list
5. [Electrostatics](05-electrostatics.md): the Ewald splitting, the
   reciprocal sum with its forces and virial, PME on a device, the
   correction for dispersion
6. [Dynamics](06-dynamics.md): integrators, constraints by Newton's
   method and SETTLE, the thermostat, the barostat and the work of a
   scaling, what the log reports, random numbers
7. [Precision](07-precision.md): the roles of values, the approximations
   and their bounds, a drift of the energy and its removal
8. [Lowering to a GPU](08-gpu.md): loops to kernels, the integration
   kernel, the host and the device, where the time of a step goes
9. [Correctness](09-correctness.md)
10. [Performance](10-performance.md)
11. [Related work](11-related-work.md)
12. [Principles of development](12-development.md)
13. [Limitations and next steps](13-limitations.md)

Appendices: [A. The control file](A-control-file.md),
[B. Taking part](B-contributing.md), [C. Using MDIR](C-usage.md): an
installation and its check, inputs from Amber, GROMACS, and CHARMM, the
standard pipeline, target and precision, runs on a cluster, terms given
by expressions, implicit solvent, and what to do when something fails.
[References](references.md).

## Reproducing the paper

| What | How |
|---|---|
| The references | `scripts/paper/build-references.py` links the citations and writes `references.md`; `scripts/paper/verify-references.py docs/paper/references.md` checks every DOI against Crossref and DataCite |
| Tables 10.2 to 10.4 and Figure 10.1 | `scripts/paper/run-suite-repeats.sh MDIR PMEMD LOGS 3` (a search of the skins for each system first, cached in `LOGS/skins`, D281; then three repeats of MDIR and pmemd.cuda alternated under the protocol of `mdin.GPU`, `bench.py --suite-protocol`, D327), then `scripts/paper/suite-table.py LOGS` and `scripts/paper/plot-suite.py LOGS figures/suite.png`, with the suite prepared by `scripts/benchmarks/amber/bench.py prepare`. One run is `bench.py run --suite-protocol [--energies-only] [--record FILE]`; without `--suite-protocol`, `bench.py` runs as before D327. `bench.py run mdir --skin AUTO` writes the reaches as `"AUTO"` (`--prune-skin AUTO` the inner one alone), the search runs with `MDIR_SKIN_SEARCH=always`, the timed repeats take the cached choice, and `suite-table.py` adds the reaches and their source (#437) |
| Figure 7.1 | `scripts/paper/plot-energy.py LOGS figures/energy-jac.png`, from the logs of the runs of D112 |
| Appendix A | `scripts/paper/check-appendix.sh MDIR` checks that its example is what `mdir template amber` prints, and exits with 1 where it no longer agrees; without the `mdir` to ask, or with one that cannot be run, it prints its usage and exits with 2 (#209) |
| The PDF | `scripts/paper/build-pdf.sh OUT` writes each section to LaTeX with pandoc and builds `OUT/main.pdf` with tectonic, from the master file `tex/main.tex`; the two programs are on the path or named by `PANDOC` and `TECTONIC` ([environment.md](../environment.md#scripts)). The title page shows the compilation timestamp in UTC, refreshed on every build. |

A change to a method, an algorithm, or an implementation of MDIR updates
the sections of this paper that describe it.

## Related work

What the outline of the paper said of its sources, from the roadmap (#603):

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
