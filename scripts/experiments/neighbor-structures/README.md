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
| `octree.cu` | Forces without a neighbor list (D[octane]): the search in a region octree every step of [Toutouni2026] (T), with the update of the tree and the sort of the indices, and the same search on a uniform grid of cells (G); checks against all pairs, histograms of the levels, and dynamics of argon with the forces of the tree. It also writes argon in the layout of `prep.py` (`octree gen`) |
| `run.sh` | Builds both, runs the sweeps, and writes `results/<date>-<device>.csv`; with `OCTREE=1` also the runs of `octree.cu` into `results/<date>-octree-<device>.csv` |
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

Units of work (`groups.cu`, `-DFAST_ERFC`, compact, 9 Å): the list of a
group cut into units of 64, 128, or 256 entries, a warp a unit. JAC: 69.2,
76.9, 93.2 µs against 125.4 for a warp a group and 75.3 for the matrix.
Cellulose: 1109, 1097, 1118 µs against 1195 and 1200.

## Forces without a neighbor list: a search in an octree (RTX 3090, 2026-10-08)

`octree.cu` measures the method of Toutouni, Chakraborty, Tu, and Huang
(ICS '26, https://doi.org/10.1145/3797905.3807878) for D[octane]: no list
is stored, and every step a thread per atom searches a tree and computes
the forces of the pairs it finds, each pair twice, with no atomic addition.

| Variant | What runs every step |
|---|---|
| **T** | The tree of the paper. A region octree over the periodic cell; nodes with the bounds of their region, a pointer to the parent and one to the first child; leaves with a block of `MC` indices of atoms. A thread per atom, in the order of an array of indices sorted along a Morton curve, climbs from its leaf to the smallest ancestor whose region contains the query box of half-width $r_c$ (no skin), descends, pruning the nodes that the box does not meet, and sums the forces in registers. After the positions move, the atoms that left their leaf are moved (the *update*) |
| **G** | The same without a tree: the atoms are sorted into a uniform grid of cells (a counting sort on the device), and a thread or a warp per atom reads the runs of cells within the cutoff, as `search.cu` does for a build, and computes the forces |
| **M** | The loop over the neighbor matrix (`groups.cu`, 16 lanes a particle), and its build (`search.cu`, a warp a particle with the excluded pairs, cells of half the reach) divided by the steps between builds |
| **Gr** | The loop over groups of 16 (`groups.cu`, compact order, the best of the units of 64, 128, and 256 entries), and their build (`ORDER=gpu build`: the order, the positions and cells, the boxes, and the lists; its scan and its order by key are not timed) divided by the steps between builds |

All kernels are `f32` with the pair kernel of `groups.cu` and `-DFAST_ERFC`
(Lennard-Jones from the type tables and the direct sum of PME; plain
Lennard-Jones for argon, for which `groups.cu` is compiled with
`-DCUTOFF=10.122f -DLJ_ONLY`). The times are of the kernels alone, the
launches between two events, on GPU 0 in one session. M and Gr are the
existing programs, run in that session; their steps between builds are
MDIR's for JAC (4.4 at 9 Å, `results/scan.csv`) and Cellulose (4.0 at 9 Å,
Table 4.1 of the paper), and for argon the steps until an atom has moved
half of a skin of 1 Å in the dynamics of the prototype (57.7 at both
densities), which is not MDIR's test.

**Systems.** JAC (23,558 atoms, 62.23 Å) and Cellulose (408,609 atoms,
259.2 × 124.6 × 123.5 Å) from the Amber suite, $r_c = 8$ Å, a reach of 9 Å
for M and Gr. Argon, 1,000,000 atoms, $r_c = 3\sigma = 10.122$ Å, a reach
of 11.122 Å for M and Gr: the liquid (0.021 atoms/Å³, a cube of 362.46 Å)
and the density of the paper (9.07 × 10⁻⁴ atoms/Å³, a cube of 1033 Å). The
argon starts on a simple cubic lattice of 100³ sites with a uniform jitter
(0.2 Å and 3 Å), and is run by the prototype itself (velocity Verlet with
the forces of T, 1 fs, velocities scaled to 120 K every 10 steps for 3000
steps, then 1000 steps without): 90.0 pairs within the cutoff an atom in
the liquid, 3.70 at the density of the paper (2.51 on the lattice: it is
clustering). Uniform random positions (1,000,000 in 1033 Å) are used for
the histograms of the levels only.

**What the paper leaves open, and what the prototype does.**

| The paper | The prototype |
|---|---|
| "Adjust query bounds for PBC" (Algorithm 1, line 11), no more | A query box that leaves the cell is contained by no node but the root, so the climb ends at the root; the descent tests a node against the box and against its image on the other side. `sphere` below tests the minimum-image distance from the atom to the region instead, which the paper does not do |
| Nodes hold bounding boxes "that enclose all atoms contained in their subtree", and only the old and the new leaf are updated | The bounds are those of the region, which never change; tight boxes would need an update of the ancestors |
| A "register-resident traversal stack" | The children yet to visit at each level, 8 bits a level in one 64-bit register, and the parent pointer to return (an array indexed by a stack pointer is in local memory, not in registers); at most 8 levels |
| Removal and insertion in one kernel with atomic operations | Three kernels without a race: the leaf of each atom (climb, descend); a thread a leaf compacts its block; the atoms that moved append themselves with an atomic increment of the count |
| A leaf over its capacity `MC` (Eq. 8, buffer 1.1 to 1.5): not said | The atom goes to a list of overflow that every search reads in full, and is tried again at each update; nothing is dropped, and the overflows are counted. The timings use the capacity of the fullest leaf |
| Cubic cells | An axis half as long has one level fewer (Cellulose: 64 × 32 × 32 leaves at level 6) |
| Morton or Hilbert order | Morton, 10 bits an axis, `thrust::sort_by_key` |

### Exactness

Against all pairs on the host in `f64` (JAC, and argon of 4096 atoms in a
cell of 57.99 Å, 5.7 cutoffs wide, where the query box of 75.6 % of the
atoms leaves the cell; 58.3 % on JAC):

| Check | Result |
|---|---|
| The set of pairs within the cutoff: T at every level, with the box and with the sphere; G with cells of 1, 1/2, 1/3 of the cutoff | 4,930,214 (JAC) and 338,196 (argon) pairs counted from both ends: none missing, none added |
| T with leaves over their capacity (buffer 1.0: 348, 714, and 2505 atoms of JAC in the list of overflow at levels 2, 3, 4) | None missing, none added |
| T after 25 updates with steps 20 times as long (JAC: 16,321 to 63,785 changes of leaf, among them atoms across the faces of the cell; 16 atoms in overflow at level 4) | None missing, none added; every atom once in the leaf of its position or in the overflow |
| Forces, relative difference from the reference (root of the sum of squares) | JAC: T 7.5 × 10⁻⁷, G 7.2 to 7.5 × 10⁻⁷, with the excluded pairs 8.9 to 9.3 × 10⁻⁷; argon: 2.0 × 10⁻⁶ |
| Energy of the pairs | JAC 7.5 to 9.0 × 10⁻⁸ relative; argon 5 × 10⁻⁹ |
| Cellulose and argon of 1,000,000 atoms (no reference over all pairs): T against G | The same count of pairs for every atom; forces 4 × 10⁻⁸ to 1.9 × 10⁻⁷ |
| Total energy of argon over 3000 steps of 1 fs with the forces of T (the potential shifted to zero at the cutoff) | Liquid, 119.96 K: drift 2.3 × 10⁻⁵ $k_BT$ an atom a ns, range 10⁻⁶ kcal/mol an atom. Density of the paper: −1.4 × 10⁻⁴ $k_BT$ an atom a ns |

### Time a step

µs a step, the best setting of each variant. T: atoms in memory in the
order of a Morton curve (in brackets: in that of the input); G: the faster
of a thread and a warp an atom.

| | JAC | Cellulose | Argon, liquid | Argon, density of the paper |
|---|---|---|---|---|
| Pairs within the cutoff an atom | 209.3 | 221.0 | 90.0 | 3.70 |
| **T**: search and forces | 754 (902) | 7032 (7544) | 7033 (7596) | 2187 (2350) |
| T: update of the tree | 18 | 78 | 132 | 141 |
| T: Morton keys and sort of the indices | 81 | 294 | 446 | 446 |
| T: a step, sorting every step | 853 | 7404 | 7611 | 2774 |
| T: a step, never sorting again | 772 | 7110 | 7165 | 2328 |
| T: candidates tested an atom | 515 | 1403 | 660 | 35 |
| **G**: search and forces | 235 | 1997 | 1512 | 260 |
| G: sort into cells | 31 | 65 | 120 | 145 |
| G: a step | 266 | 2062 | 1632 | 405 |
| G: candidates tested an atom | 1854 | 864 | 348 | 25 |
| **M**: loop | 76 | 1235 | 1051 | 262 |
| M: build | 358 | 6601 | 10133 | 8079 |
| M: build over the steps between builds | 81 | 1650 | 176 | 140 |
| M: a step | 157 | 2885 | 1226 | 402 |
| M: entries an atom | 298 | 315 | 121 | 4.9 |
| **Gr**: loop | 70 | 1094 | 1139 | 201 |
| Gr: build | 206 | 2497 | 2548 | 3562 |
| Gr: build over the steps between builds | 47 | 624 | 44 | 62 |
| Gr: a step | 116 | 1718 | 1183 | 263 |
| Gr: slots an atom (each pair once) | 355 | 365 | 194 | 33 |
| With the excluded pairs: T, search and forces (input order) | 1118 | 9267 | | |
| With the excluded pairs: G, search and forces | 253 | 2449 | | |

In MDIR itself, Cellulose with groups at 9 Å takes 1041 µs of loop and 405 µs
of builds a step (Table 4.1 of the paper; not measured here): the build of
`build.cu` as it stands is slower than the build of MDIR.

The leaf level of T (search and forces, µs; pruning with the box / with the
sphere; atoms in the order of the input, then in Morton order; the last
column is the search with the excluded pairs, the atoms in the order of
the input):

| System | Leaf side | Atoms a leaf | Candidates an atom | Input order | Morton order | Update | Excluded pairs |
|---|---|---|---|---|---|---|---|
| JAC | 15.56 Å | 368 | 3074 / 2627 | 2564 / 1820 | 2168 / 1603 | 144 | 2170 |
| JAC | 7.78 Å | 46.0 | 1315 / 998 | 1337 / 1003 | 1038 / 803 | 30 | 1298 |
| JAC | 3.89 Å | 5.75 | 768 / 515 | 1737 / 902 | 1489 / 754 | 19 | 1118 |
| Cellulose | 16.20 Å | 399 | 3297 / 2830 | 13159 / 12151 | 12134 / 11398 | 333 | 13278 |
| Cellulose | 8.10 Å | 49.9 | 1403 / 1069 | 7544 / 7597 | 7032 / 7060 | 78 | 9267 |
| Cellulose | 4.05 Å | 6.23 | 818 / 549 | 10480 / 7791 | 9858 / 7295 | 57 | 9218 |
| Argon, liquid | 22.65 Å | 244 | 1657 / 1448 | 12325 / 13807 | 11638 / 13197 | 250 | |
| Argon, liquid | 11.33 Å | 30.5 | 660 / 514 | 7596 / 7698 | 7033 / 7151 | 136 | |
| Argon, liquid | 5.66 Å | 3.81 | 364 / 250 | 15791 / 9590 | 14866 / 9007 | 132 | |
| Argon, paper | 32.28 Å | 30.5 | 131 / 119 | 3692 / 3074 | 3215 / 2826 | 143 | |
| Argon, paper | 16.14 Å | 3.81 | 43 / 35 | 3546 / 2350 | 3326 / 2187 | 175 | |
| Argon, paper | 8.07 Å | 0.48 | 20 / 14 | 14404 / 9831 | 13475 / 8830 | 416 | |

G (search and forces, µs; a thread an atom / a warp an atom; then with the
excluded pairs; then the sort into cells; atoms in the order of the input):

| System | Cells of the cutoff | Candidates an atom | Thread / warp | Excluded pairs, thread / warp | Sort |
|---|---|---|---|---|---|
| JAC | 1 | 1854 | 499 / 235 | 631 / 253 | 31 |
| JAC | 1/2 | 872 | 344 / 325 | 422 / 310 | 27 |
| JAC | 1/3 | 664 | 329 / 523 | 411 / 540 | 27 |
| Cellulose | 1 | 1544 | 2614 / 4051 | 3218 / 4331 | 66 |
| Cellulose | 1/2 | 864 | 1997 / 5676 | 2449 / 5364 | 65 |
| Cellulose | 1/3 | 688 | 2089 / 8708 | 2490 / 8973 | 68 |
| Argon, liquid | 1 | 629 | 1725 / 4787 | | 120 |
| Argon, liquid | 1/2 | 348 | 1512 / 9888 | | 120 |
| Argon, liquid | 1/3 | 279 | 1899 / 18774 | | 151 |
| Argon, paper | 1 | 25 | 260 / 3567 | | 145 |
| Argon, paper | 1/2 | 14 | 451 / 8624 | | 424 |
| Argon, paper | 1/3 | 11 | 2173 / 15872 | | 1092 |

Further numbers of T:

- **The energy and virial** summed with the forces change the search by
  under 8 % (JAC 754 to 770 µs, Cellulose 7032 to 7550, liquid 7033 to
  7071).
- **The sort.** With the threads in the order of the atoms and not sorted,
  the search takes 1034 against 902 µs on JAC, 9595 against 7544 on
  Cellulose, and 10,990 against 7596 on the liquid (input order; with a
  random order of the atoms in memory, 101,119 against 10,784). With the
  atoms in Morton order in memory the sort of the indices changes nothing
  (754, 7070, 7065 µs without it). In the dynamics of argon the search with
  indices sorted 1, 10, 100, and 1000 steps before takes 7618, 7608, 7645,
  and 7882 µs (liquid) against 7670, 7664, 7647, and 7682 with the indices
  sorted at that step; 2347, 2360, 2353, 2388 against 2381, 2369, 2384,
  2392 at the density of the paper. Along fixed velocities (below), 100
  steps after the sort: Cellulose 9117 against 7826 µs, JAC 980 against 978.
- **The update**, on JAC at 3.89 Å: 7 µs to find the leaves, 7 to compact,
  5 to insert; on the liquid at 11.33 Å: 42, 74, 20.
- **The excluded pairs** are tested for a pair within the cutoff whose
  partner lies between the least and the greatest partner of the atom, by a
  scan of the partners. The cost depends on the order of the atoms: G with
  a thread an atom on JAC, 344 to 422 µs in the order of the input and 338
  to 524 in Morton order.
- **The capacity.** Eq. 8 with a buffer of 1.5 holds every atom only where
  a leaf has tens of atoms. Atoms over it: JAC at 3.89 Å, 137 in 106 leaves
  (the fullest leaf 13, `MC` = 9, a buffer of 2.26 for none); Cellulose at
  4.05 Å, 1286 (14 against 10); liquid at 5.66 Å, 1245 (8 against 6); the
  density of the paper at 16.14 Å, 26,981 (13 against 6), and at 8.07 Å,
  184,267 (6 against 1); uniform random positions at 16.14 Å, 41,468 (17
  against 6). With the list of overflow the pairs stay exact, and the
  search of JAC at 3.89 Å goes from 1737 to 2205 µs with 2505 atoms in it.

### Levels

The level of the smallest ancestor that contains the query box (the root is
level 0), as the fraction of the atoms. It does not depend on the leaf
level; the search starts (leaf level − this level) levels above the leaf.
*Periodic* is what the prototype does (and the kernel counts the same); *not
periodic* cuts the box at the faces of the cell, which is the model of
Eq. 1 to 7 of the paper.

| System, $D / r$ | | Root | 1 | 2 | 3 | 4 | 5 | Mean |
|---|---|---|---|---|---|---|---|---|
| Uniform random, 102.1 | periodic | 0.1134 | 0.1042 | 0.1840 | 0.2753 | 0.2710 | 0.0520 | 2.64 |
| | not periodic | 0.0578 | 0.1088 | 0.1921 | 0.2897 | 0.2910 | 0.0606 | 2.83 |
| Argon, paper, 102.1 | periodic | 0.1132 | 0.1043 | 0.1833 | 0.2757 | 0.2713 | 0.0521 | 2.64 |
| | not periodic | 0.0578 | 0.1086 | 0.1916 | 0.2900 | 0.2913 | 0.0607 | 2.83 |
| Argon, liquid, 35.8 | periodic | 0.2990 | 0.2328 | 0.2990 | 0.1680 | 0.0012 | | 1.34 |
| | not periodic | 0.1584 | 0.2650 | 0.3508 | 0.2216 | 0.0042 | | 1.65 |
| Cellulose, 32.4 × 15.6 × 15.4 | periodic | 0.3217 | 0.2649 | 0.2948 | 0.1186 | | | 1.21 |
| | not periodic | 0.0621 | 0.3315 | 0.3980 | 0.2076 | 0.0007 | | 1.75 |
| JAC, 7.78 | periodic | 0.8859 | 0.1141 | | | | | 0.11 |
| | not periodic | 0.5912 | 0.3944 | 0.0144 | | | | 0.42 |

The query box leaves the cell for 5.8 % of the atoms at the density of the
paper, 15.8 % in the liquid, 27.4 % on Cellulose, and 58.3 % on JAC.

The update: the fraction of the atoms that leave their leaf in a step, and,
for those, the levels climbed to the lowest common ancestor of the old and
the new leaf (an atom across a face of the cell climbs to the root). Argon:
the dynamics of the prototype, 3000 steps of 1 fs near 120 K. JAC and
Cellulose: 20 steps along fixed velocities drawn from the Maxwell-Boltzmann
distribution at 300 K, 2 fs, with a mass of 12 u for every atom (`prep.py`
writes no masses).

| System, leaf side | Leave in a step | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| Argon, liquid, 11.33 Å | 0.0334 % | 0.4984 | 0.2516 | 0.1250 | 0.0625 | 0.0624 | |
| Argon, paper, 16.14 Å | 0.0237 % | 0.4991 | 0.2509 | 0.1252 | 0.0604 | 0.0321 | 0.0323 |
| JAC, 7.78 Å | 0.279 % | 0.4958 | 0.2646 | 0.2395 | | | |
| JAC, 3.89 Å | 0.566 % | 0.5073 | 0.2443 | 0.1304 | 0.1180 | | |
| Cellulose, 8.10 Å | 0.278 % | 0.4991 | 0.2484 | 0.1267 | 0.1060 | 0.0198 | |
| Cellulose, 4.05 Å | 0.553 % | 0.4970 | 0.2511 | 0.1249 | 0.0638 | 0.0533 | 0.0100 |

### Memory

MB on the device beyond the positions, charges, types, and forces. T: the
nodes (36 bytes each), the counts and blocks at the capacity of the fullest
leaf, the leaf of each atom (old and new), the sorted indices and their
keys. G: the counts, starts, and cursors of the cells, and the sorted
copies of the positions, types, and indices. M: 4 bytes an entry and a
count (in brackets: rows of the width of the widest). Gr: 8 bytes an entry
and a start a group.

| | JAC | Cellulose | Argon, liquid | Argon, density of the paper |
|---|---|---|---|---|
| T, at the leaf level of the table of times | 0.77 | 9.37 | 23.5 | 41.5 |
| T, one level coarser | 0.54 | 9.04 | 22.2 | 23.6 |
| G, at the cells of the table of times | 0.66 | 12.2 | 32.3 | 40.7 |
| M at its reach | 28.2 (33.6) | 516 (662) | 488 (564) | 23.7 (68.0) |
| Gr at its reach | 4.0 | 71.4 | 89.6 | 12.0 |

### What the numbers show

1. **The search is exact** under periodic boundaries, with leaves over
   their capacity, and after partial updates of the tree, with the handling
   of the boundary and of the overflow that the prototype adds.
2. **A step of T takes 4.3 to 10.5 times a step of Gr and 2.6 to 6.9 times
   a step of M** (JAC 853 against 116 and 157 µs; Cellulose 7404 against
   1718 and 2885; liquid 7611 against 1183 and 1226; the density of the
   paper 2774 against 263 and 402), without the excluded pairs.
3. **Without the tree the same idea is 3.2 to 6.8 times faster**: G takes
   266, 2062, 1632, and 405 µs a step. A candidate of T costs more than one
   of G: its index is loaded from the block and its position from the array
   of the atoms, where G reads positions in a row (the kernels were not
   profiled). A step of G is 2.3 times a step of Gr on JAC, 1.2 on
   Cellulose, 1.4 on the liquid, and 1.5 at the density of the paper, and
   0.7 of a step of M on Cellulose with the build of `search.cu`.
4. **The update and the sort are small**: 18 to 141 µs and 81 to 446 µs
   against a search of 754 to 7033 µs; indices sorted 1000 steps before
   cost under 3 % in the dynamics of argon.
5. **The excluded pairs add 23 to 24 % to the search of T** (JAC 902 to
   1118 µs at 3.89 Å, Cellulose 7544 to 9267 at 8.10 Å, input order) and 8
   to 23 % to that of G (JAC 235 to 253 with a warp an atom, Cellulose 1997
   to 2449 with a thread an atom).
6. **The climb starts at the root for a large part of the atoms where the
   cell is a few cutoffs wide**: under periodic boundaries 88.6 % of the
   searches of JAC, 32.2 % of Cellulose, 29.9 % of the liquid, and 11.3 %
   at the density of the paper.
7. **T and G hold 0.5 to 42 MB where M holds 24 to 516 MB and Gr 4 to
   90 MB**; at the density of the paper Gr holds less than T.
