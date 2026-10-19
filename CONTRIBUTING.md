# Contributing to MDIR

The full guide is Appendix B of the white paper,
[docs/paper/B-contributing.md](docs/paper/B-contributing.md): building,
the tiers of tests, where a change goes, the tools of diagnosis, and how a
change is recorded and reviewed. This file is its summary.
[docs/workflow.md](docs/workflow.md) describes the process around a change:
roles, branches and draft pull requests, review, and merging.

## Build and test

Build LLVM and MLIR 23.1.2 with `scripts/build-llvm.sh` and, for
checkpoints, HDF5 with `scripts/build-hdf5.sh`; then configure and build
as in [README.md](README.md) and run

```sh
cmake --build build --target check-mdir
```

A change passes the default tier (`lit test`). A change to code that runs
on a GPU also passes it under compute-sanitizer (`lit -Dsanitize=1
test`), and a change to the neighbor structures, the kernels, or the
schedule also passes the short runs of the Amber suite
(`MDIR_BENCH_DIR=<dir> lit test/Scale`).

## What a change carries

- **A test** of what it changes, at the level where it changes it: the IR
  of a pass in `test/Dialect` or `test/Conversion`, a run of the driver in
  `test/Driver`. A value that a test pins after a comparison with another
  program names that program and the value it gave in the test's comment.
  A test that writes outputs writes them under `%t` and removes it first
  (`rm -rf %t && mkdir -p %t`); `test/Lit/test-outputs.test` checks it.
  A test that compares two runs in mixed precision within a tolerance,
  after a step, either compares where the positions of both are the same
  (at step 0, or after a step of $10^{-9}$ or $10^{-6}$ ps), or runs both
  where the sums have a fixed order (the CPU, or the deterministic mode,
  where a flip, if there is one, fails every run of a build and is found
  by the change that causes it), or has a tolerance sized for runs that
  have parted. Otherwise a
  coordinate that lies near a midpoint between two numbers of f32 can
  round to either one when the positions are converted for the kernels,
  and the forces of the two runs then differ by about $2k$ times its ulp:
  the test fails now and then, whatever the code it tests (#427, #428).
- **A switch and a comparison** for an optimization: it can be turned off,
  and a test compares both ways, bitwise where the sums have a fixed order.
- **A decision** in [docs/decisions.md](docs/decisions.md) for a choice
  that changes a method, a structure, or a default: what, why, and the
  measurements before and after. A branch names its decision by a label,
  `D[<label>]` (lowercase letters, digits, and dashes), wherever it would
  write the number; the number is given when the branch is merged, by
  `scripts/decision-number.sh assign <label>`, and
  `scripts/decision-number.sh check` refuses a label or a number given
  twice on `main`.
- **The documents and the white paper** updated with the code: the design
  documents in `docs/` describe what is implemented, and a change to a
  method, an algorithm, or an implementation updates the sections of
  `docs/paper/` that describe it.
- **References** by their keys in [docs/references.md](docs/references.md),
  each checked against its DOI (`scripts/paper/verify-references.py`).
  Code comments cite papers, not other programs.

The questions a review asks are the checklist of
[docs/principles.md](docs/principles.md).

## Reporting a defect

`mdir bug-report FILE -o DIR --run` collects the build, the machine, the
inputs with their hashes, and every stage of the compilation into one
directory; [docs/debugging.md](docs/debugging.md) tells how failures of
passes and of kernels are narrowed down.

## Language

Files in the repository are written in American English.

Equations in Markdown are TeX between `$...$` and `$$...$$`, and in a cell
of a table `\lvert` stands for `|`. A named operator is written
`\mathrm{erfc}`, `\mathrm{tr}`, and so on, not with `\operatorname`, which
GitHub does not render; where the operator is followed by a symbol, a thin
space keeps them apart: `\mathrm{tr}\,\mathsf W`.
