# Contributing to MDIR

The full guide is Appendix B of the white paper,
[docs/paper/B-contributing.md](docs/paper/B-contributing.md): building,
the tiers of tests, where a change goes, the tools of diagnosis, and how a
change is recorded and reviewed. This file is its summary.

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
- **A switch and a comparison** for an optimization: it can be turned off,
  and a test compares both ways, bitwise where the sums have a fixed order.
- **A decision** in [docs/decisions.md](docs/decisions.md) for a choice
  that changes a method, a structure, or a default: what, why, and the
  measurements before and after.
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
