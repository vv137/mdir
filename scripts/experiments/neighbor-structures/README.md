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
| `supercluster.cu` | The loop over pairs with two half lists against the matrix, on any system that `prep.py` writes: the cluster pair list of [Pall2013] as its GPU layout has it (super-clusters of 64 in 8 clusters of 8, a warp per super-cluster, the forces on i in registers and those on j summed with shuffles once per entry), and groups of 32 particles sharing a list of particles j, which turn around the warp one lane per step (the neighbor list that [SalomonFerrer2013] describes for pmemd, groups of 16 or 32). The forces are compared with the matrix |
| `spread.cu` | The atomic additions of the spreading of PME, order 4, on the grid of MDIR for a system: 4 threads a particle, a warp a particle in the order of the transform, and a warp a particle into bricks of 4 × 4 in x-y with a copy after, with `f32` and `i32` atomics; and the kernel of the weights, as a structure a particle and by component. The grids are compared with one summed on the host |
| `groups.cu` | The loop over groups of 16 against the matrix in the same places, for an order of the places (`cells`, `morton`, `zcurve`, `compact`) and a threshold of trivial acceptance |
| `build.cu` | The build of groups of 16 on a device (order by a counting sort on a Z curve, or the compact order from the host with `ORDER=compact`; boxes; the lists, a warp a group and a lane a candidate), compared entry by entry with lists built on the host |
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

## Large systems (RTX 3090, 2026-10-01)

Cellulose (408,609 atoms) against JAC, to see why MDIR falls further behind
pmemd.cuda as systems grow (JAC 84 %, Cellulose 49 %). `supercluster
<system.bin> <reach> <repeats>`; GPU 1, times of the kernels alone.

| Cellulose | reach 10 Å | reach 9 Å |
|---|---|---|
| Matrix, the order of `pairs.cu` (cells of half the reach) | 2013 µs | |
| Matrix, in the order of the clusters | 1694 µs | 1245 µs |
| Super-clusters, half list | 1752 to 1798 µs | 1590 µs |
| Groups of 32, half list | 1913 µs | 1633 µs |
| pmemd.cuda, nonbonded kernel (nsys, `kCalcPMEOrthoNBFrc16`) | | 886 µs |

1. **pmemd.cuda lists to 9 Å, not 10.** Its skin is 1 Å by default
   without MPI (`skinnb`, `mdin_ewald_dat.F90` of Amber 26), and it builds
   every 4.9 steps on Cellulose at about 1.1 ms a build. The suite's inputs
   for MDIR have `pairlist_distance = 10`.
2. **The matrix is bound by its index.** Its time follows its entries
   (1694 to 1245 µs as they go from 177 to 129 million, 4 bytes each), and a
   cheaper erfc (Abramowitz and Stegun 7.1.26, sharing the exponential of
   the force) changed nothing. The order of the particles matters: the same
   matrix in the order of the clusters (columns in x-y sorted by z, halved
   in x, y, z) is 16 % faster than in cells of half the reach.
3. **The half lists of these prototypes do not yet pay.** They compute
   each pair once, but at 9 to 10 Å only 18 to 26 % of their slots are
   within the cutoff, and a warp takes the costly path when any lane does.
   GROMACS runs the same layout with an inner list pruned to about the
   cutoff; its kernel on JAC (about 50 µs at 8.05 Å) is several times
   faster than these.
4. **Per particle, MDIR costs the same on JAC and Cellulose** (the loop
   over pairs about 5 ns, a build about 20 ns); pmemd.cuda gets cheaper per
   particle as systems grow, so its lead on JAC was hidden by its fixed
   costs.

On Cellulose, a step of MDIR takes 5.28 ms of the device against 2.75 for
pmemd.cuda: pairs 2102 against 886 µs, builds 1049 against 224 µs a step
(8.0 ms against 1.1 ms a build), bonded terms about 545 against 343, the
spreading of PME 553 against about 344, its gathering 339 against 79, and
the transforms about the same.

### The spreading of PME (2026-10-01)

`spread cell.bin` (Cellulose, grid 270 × 126 × 126, GPU 1), clearing and
copying included: 4 threads a particle 434 µs (`f32`) and 479 (`i32`); a
warp a particle 351 and 393; a warp a particle into bricks 257 and 278. The
weights of the particles take 142 µs written as a structure a particle and
52 by component; the whole, from the positions, 329 µs. In MDIR the three
kernels take 367 µs a step (docs/pme-m1.md).

### Groups of 16 (2026-10-01)

After reading how pmemd.cuda arranges its loop (groups of 16 particles
sharing a list of particles j, the two halves of a warp holding the same
16 and turning 16 of the 32 j each; an erfc from a table), `supercluster`
gained groups of 16 and, with `-DFAST_ERFC`, an erfc that shares the
exponential of the force (Abramowitz and Stegun 7.1.26). Cellulose, reach
9 Å, GPU 1:

| Loop | µs | Entries | Slots within the cutoff |
|---|---|---|---|
| Matrix | 1245 to 1267 | 128.7 million | |
| Groups of 32, erfcf | 1668 | 5.8 million | 23 % |
| Groups of 32, cheaper erfc | 1283 to 1300 | | |
| Groups of 16, cheaper erfc | 1013 | 8.8 million | 30 % |
| pmemd.cuda | 886 | | |

The cheaper erfc does not change the matrix, which is bound by its index,
and gains 22 % on the groups, which are bound by their arithmetic; leaving
out the minimum image gains 2 % more. A list of groups holds a fifteenth of
the entries of the matrix, which a build writes.

### The order of the groups, and their build (2026-10-01)

`groups cell.bin 9 <order> 0` (`-DFAST_ERFC`): the Z curve through cells of
1 to 4 particles gives boxes of 7.3 Å but 26.3 % of the slots within the
cutoff, and 1337 to 1377 µs, against 29.7 % and 1162 to 1213 µs in the
compact order: cutting a Z curve into sixteens puts particles across its
jumps into one group.

`build <system.bin> 9` (GPU 1), the lists of the groups:

| Order | Cellulose | JAC | Boxes over 20 Å (Cellulose) |
|---|---|---|---|
| Z curve (counting sort on the device) | 12.4 ms | 1.31 ms | 1272 of 25539 |
| Compact (from the host) | 2.44 ms | 0.22 ms | 2 of 26355 |

A lane takes a candidate and tests it against the 16 particles of the
group; a first version, which took the survivors two at a time (a
half-warp each) with a load of each from memory, took 53 to 59 ms on
Cellulose, and a box computed in the wrapped coordinates made a group
across the edge of the cell as large as the cell. The lists equal those of
the host but for 2 groups of Cellulose (pairs at the reach itself, to be
checked). The matrix of MDIR takes 8.0 ms a build of 10 Å on Cellulose.

The compact order on the device (`ORDER=gpu`: a counting sort by column in
x-y and bin of 0.1 Å along z, then a warp a chunk of 64 sorting by x and
each half by y with bitonic networks) gives groups as good as those of the
host (338.1 entries a group on Cellulose) in about 54 µs, with a scan and
an order by key besides. The lists then (Cellulose, reach 9 Å):

| Lists | Cellulose | JAC |
|---|---|---|
| A lane a candidate | 2.41 ms | 218 µs |
| The grid holding positions and places, not places alone | 2.38 ms | |
| Without the excluded pairs (to see their cost) | 1.42 ms | |
| The partners of a group sorted, a binary search a candidate | 1.76 ms | 157 µs |
| And the candidate relative to the center of the box, the group too | 1.48 ms | 116 µs |

A build of groups is then about 1.6 ms on Cellulose, against 8.0 ms for
the matrix of 10 Å. The lists equal those of the host but for 2 to 4
groups, at pairs at the reach itself.
