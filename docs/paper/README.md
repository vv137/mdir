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
the topology and the parameters at hand. On one RTX 3090 in mixed
precision, MDIR runs every system of the Amber GPU benchmark suite, from
23,558 to 1,067,095 atoms, at constant energy and at constant pressure,
at 103% to 132% of the rate of pmemd.cuda 26 on the same device; on
ubiquitin in OPC with a cutoff of 9 Å it reaches 66% to 72% of the rate
of GROMACS 2026.3, whose nonbonded kernel takes as long as MDIR's loop
over pairs but which runs PME, the bonded terms, and the update beside it.
Over
2 ns of the JAC system its total energy drifts by 1.5 kcal/mol/ns, against
5.9 for pmemd.cuda on the same input. The paper derives the methods as
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
[B. Taking part](B-contributing.md). [References](references.md).

## Reproducing the paper

| What | How |
|---|---|
| The references | `scripts/paper/build-references.py` links the citations and writes `references.md`; `scripts/paper/verify-references.py docs/paper/references.md` checks every DOI against Crossref and DataCite |
| Table 10.2 and Figure 10.1 | `scripts/paper/run-suite-repeats.sh MDIR PMEMD LOGS 3`, then `scripts/paper/suite-table.py LOGS` and `scripts/paper/plot-suite.py LOGS figures/suite.png`, with the suite prepared by `scripts/benchmarks/amber/bench.py prepare` |
| Figure 7.1 | `scripts/paper/plot-energy.py LOGS figures/energy-jac.png`, from the logs of the runs of D112 |
| Appendix A | `scripts/paper/check-appendix.sh MDIR` checks that its example is what `mdir template amber` prints |
| The PDF | `scripts/paper/build-pdf.sh OUT` writes each section to LaTeX with pandoc and builds `OUT/main.pdf` with tectonic, from the master file `tex/main.tex` |

A change to a method, an algorithm, or an implementation of MDIR updates
the sections of this paper that describe it.
