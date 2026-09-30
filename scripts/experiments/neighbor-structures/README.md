# Neighbor structures on a device: prototypes and measurements

These are the experiments behind D82 (tile structure, on hold), D84
(determinism as a mode), and the warp search of the neighbor matrix
(docs/neighbors-m0.md, Section 2.5); docs/tiles-m1.md, Section 11, reads
them. They are standalone CUDA programs, not part of MDIR, so that a
layout can be measured in an afternoon before it is built into the
compiler.

| File | What it is |
|---|---|
| `prep.py` | Writes JAC (`jac_nve` of the Amber suite) as a flat binary: positions, charges, types, the Lennard-Jones tables, the excluded pairs |
| `pairs.cu` | The loop over pairs (Lennard-Jones from type tables and the direct sum of PME, in `f32`) with the neighbor matrix of MDIR (16 lanes per particle), with tiles of 8 × 8 and 8 × 4 (full lists), with 8 × 4 half lists (`f32` atomics and fixed point), and the pruning of a matrix from an outer reach. The lists are built on the host; only the kernels are timed, 1000 launches each. The forces of the variants are compared |
| `search.cu` | The search of a build of the matrix as the template of MDIR does it (cells, runs of cells): a thread per particle, a warp per particle with ballots (with and without the excluded pairs), and a block per cell with the runs in shared memory. The matrices are compared entry by entry |
| `run.sh` | Builds both, runs the sweeps, and writes `results/<date>-<device>.csv` |
| `plot.py` | Draws `pairs.png`, `search.png`, `model.png`, and `scan.png` into `results/` (matplotlib; the project uses the environment `~/opt/render`) |
| `results/reference.csv` | Numbers measured with nsys in MDIR and in GROMACS 2026.3, which the plots set beside the prototypes |
| `results/scan.csv` | The rate of MDIR on JAC against `pairlist_distance`, with the interval between builds |

```sh
MDIR_BENCH_DIR=~/opt/benchmarks/amber CUDA_VISIBLE_DEVICES=0 \
    NVCC=/usr/local/cuda-13.4/bin/nvcc ./run.sh
~/opt/render/bin/python plot.py results/<date>-<device>.csv
```

## What they showed (RTX 3090, 2026-09-30)

1. **At equal reach the matrix is faster than tiles** (`pairs.png`): at
   10 Å, 108 µs against 135 (8 × 4, full) and 146 to 149 (8 × 4, half);
   at 8.05 Å, 61 against 106 to 110. Of the slots of an 8 × 4 record, 38
   to 47 % hold a pair of the list; the kernel is bound by the arithmetic
   of the pairs, so the empty slots cost more than the contiguous loads
   save. The tile kernel of the prototype is twice as slow as that of
   GROMACS for the same list (about 50 µs at 8.05 Å), and the matrix at
   8.05 Å is within 20 % of it.
2. **The reach decides the loop, the build decides the reach.** GROMACS
   runs with a list of 8.05 Å kept for 10 steps, with a buffer estimated
   from a tolerance of the drift; MDIR keeps the exact test (D80) and needs
   a skin, so its list is 10 Å. `model.png` puts the loop of the
   prototype together with the builds at the measured intervals: with a
   build of 780 µs the optimum is 10 to 11 Å; with 150 µs it moves to 9 Å
   and the loop and the builds take about a third less.
3. **The search is bound by its tests, not by its loads** (`search.png`):
   a warp per particle is 25 to 40 % faster than a thread per particle
   with cells of half the reach or wider; a block per cell, which reads
   each run once for all the particles of the cell, is no faster. The
   excluded pairs, entered in the search, cost about 50 µs.
4. **In the app** the warp search with the excluded pairs took 612 µs a
   build (the prototype: 440) while it counted in `index`, 64 bits on the
   device, which took more than 64 registers and spilled; counting in
   `i32`, with the partners of the excluded pairs in registers, it takes
   430 µs. The kernel that marked the excluded pairs (180 µs) is gone.
   JAC went from 374 to 381 and then to 402 ns/day, and the best
   `pairlist_distance` moved from 10 to 9.5 Å (`scan.png`; its first curve
   was measured before D81 and D83, so the gap to it is not the search
   alone).

A build of 150 µs is open: the search is bound by its tests.
