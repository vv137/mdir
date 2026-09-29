# Neighbor Structures for Milestone M0: Method

Status: implemented (2026-09-29).

This document describes how MDIR builds a neighbor structure and keeps it
valid, why it does so in this way, and how the times in it were measured.
The ops are specified in [ops-m0.md](ops-m0.md), Section 8.2.

## 1. The structure

A neighbor structure is a neighbor matrix: for each particle a row of the
particles within the reach, which is the cutoff plus the skin.

```text
count[i]            number of neighbors of particle i
index[i][0 .. W)    their indices; entries from count[i] on are unused
```

| Property | Value |
|---|---|
| Directed | If `j` is in the row of `i`, then `i` is in the row of `j`. |
| Complete | A row holds every particle within the reach. It may hold particles that are beyond the reach by less than the margin of Section 2.3; a loop over pairs tests the cutoff for every entry, so these contribute nothing. |
| Order of a row | The order of the cells that the search visits, and within a cell the order of the indices. The order does not depend on the threads, on the device, or on the run. |
| Overflow | `W` is a plan parameter. If a row is too narrow the run stops with a message. |

The order of a row decides the order in which a loop over pairs adds up
the forces on a particle, and with it the last bits of the forces.

## 2. The build

### 2.1 Steps

The build is a template in IR:
`lib/Runtime/Templates/NeighborsMatrix.mlir` for the host and
`lib/Runtime/Templates/NeighborsMatrixGPU.mlir` for a device. The compiler
adds the template to the module, where it is lowered with the rest of the
code. The two templates build the same matrix, entry by entry.

| Step | Work | On a device |
|---|---|---|
| 1 | Choose the width of the cells (Section 2.2) | On the host |
| 2 | For each particle: the position in the cell, in f32, and the cell | One thread per particle. The particles of a cell are counted with an atomic addition. |
| 3 | The offsets of the cells, from the counts | Three kernels: the sums of chunks of 256 cells, the offsets of the chunks in one thread, the offsets of the cells of each chunk |
| 4 | The particles in the order of the cells | One thread per particle takes the next slot of its cell, with an atomic addition |
| 5 | The particles of each cell in the order of their indices, and their positions in that order | One thread per cell: a sort by insertion, and a copy of the positions |
| 6 | The search: for each particle, test the particles of the cells within reach and fill the row (Sections 2.3 to 2.5) | One thread per particle, or one per row of cells of a particle |
| 7 | The largest count, and the counts limited to the width of a row | Two kernels: chunks of 256 particles, and one thread for the chunks |

Steps 2 to 4 are a counting sort. On the host it is sequential and keeps
the order of the indices, so step 5 is only the copy.

Step 3 was one thread on a device. With narrow cells there are as many
cells as particles, and one thread took longer than the search.

### 2.2 Cells and their width

Along an edge of the length `L` there are `n = floor(L / w)` cells, of the
width `L / n`, which is `w` or more. A search for the neighbors of a
particle visits the cells within reach of the cell of the particle:
`r = ceil(reach / (L / n))` cells on each side, or all cells, each once,
where these are fewer.

| Width `w` | Cells visited | Volume visited, in cubes of the reach |
|---|---|---|
| The reach | 3 × 3 × 3 | 27 |
| Half the reach | 5 × 5 × 5 | 15.6 |
| A third of the reach | 7 × 7 × 7 | 12.7 |
| The sphere that holds the neighbors | | 4.2 |

Narrow cells have fewer particles to test and more cells to visit. A
search reads the cells of a row as one run (Section 2.4), so what counts
is the number of rows. On the device that was measured, visiting a row
costs as much as testing 12 particles.

The build chooses the width when it runs, because the density is known
only then:

```text
for w in reach, reach / 2, reach / 3, each not below the least width:
    cost(w) = 12 · rows visited + particles tested
    particles tested = N · cells visited / cells
the width with the least cost is taken
```

| System | Particles in a cube of the reach | Width |
|---|---|---|
| Lennard-Jones liquid at the density 0.58, reach 2.2 | 6 | The reach |
| Liquid argon, reach 0.95 nm, in a cell of 3.5 nm | 18 | Half the reach: with cells of the reach there are 3 along an edge, and a search tests every particle |
| Water, reach 1.0 nm | 100 | Half the reach |

The least width is the plan parameter `cells` of `convert-md-to-md-exec`:
the number of cells that the reach may span. The default is 3.

Two factors keep rounding from hiding a neighbor:

| Factor | Where | Why |
|---|---|---|
| 1.0001 | The cells are that much wider than `w`. | With `L` a multiple of `w`, the ratio of the reach to the width of a cell would be a whole number, and rounding would decide whether one more cell is visited on each side. |
| 1.00005 | The reach, where the cells within reach are counted. | Rounding may put a particle that is at the edge of a cell into the cell next to it. |

### 2.3 The copy that the search works on

The search does not read the positions of the particles. It reads a copy
that step 2 and step 5 make for it:

| Property of the copy | Reason |
|---|---|
| The positions are in the cell, between 0 and `L`. | They have the same size whatever the particles have moved, so their rounding is bounded. |
| They are in f32. | On a device, arithmetic in f64 is several times slower. A position takes 12 bytes. |
| They are in the order of the cells. | The particles of a cell, and of a row of cells, are next to one another. The search reads memory in runs and does not look up a position through an index. |

The position in the cell is computed in the type of the positions and then
converted. A distance between two positions of the copy differs from the
distance of the particles by rounding. The search therefore takes a pair
that is within the reach plus a margin:

```text
margin = 3e-6 · (Lx + Ly + Lz)
```

| Source of the difference | Size |
|---|---|
| Conversion of a coordinate to f32 | Half a unit in the last place of `L`: 6e-8 `L` |
| A displacement, from two coordinates | 1.2e-7 `L` for each component |
| In the single mode, the position in the cell of a particle that is `b` cells away from the cell | Half a unit in the last place of `b L` |

The margin is 9e-6 `L` in a cubic cell. It covers positions in f64
always, and positions in f32 up to some 40 cells away from the cell.
Beyond that the positions in f32 have lost so many digits that the single
mode is not usable for other reasons.

A pair within the margin is beyond the cutoff by the skin. It contributes
nothing, and it does not change the order in which the other entries are
added up. For a reach of a tenth of `L` one entry in 4000 is such a pair.

The tests compare the matrix with a search over all pairs
(`test/Runtime/Inputs/neighbors_reference.py`): the entries are the pairs
within the reach plus the margin, exactly.

### 2.4 Rows of cells

The cells are numbered along x first. The cells of a row along x are next
to one another in the order of the cells, so the particles of the cells
`cx − r` to `cx + r` of a row are one run of the copy:

```text
for each of the rows within reach, along z and y:
    the run of the cells along x        one run, or two where the row
                                        goes around the edge of the cell
        for each particle of the run: test, and add to the row
```

The order of a row of the matrix is the one that a visit of cell after
cell gives.

### 2.5 The threads of the search on a device

A kernel takes as long as its slowest thread. With one thread for a
particle, a thread tests every particle of 9 to 49 rows of cells. A small
system leaves most of the device idle meanwhile.

The search is therefore split where the device has threads to spare: one
thread for each row of cells of a particle. The threads of a particle must
fill one row of the matrix in a fixed order, so the search runs twice:

| Kernel | Threads | Work |
|---|---|---|
| Count | One per particle and row of cells | Counts the neighbors in its row of cells |
| Places | One per particle | Turns the counts of the threads of the particle into their places in the row of the matrix, and stores the number of neighbors |
| Fill | One per particle and row of cells | Searches again and writes what it finds from its place on |

| | One thread for a particle | One for each row of cells |
|---|---|---|
| Time of the search | That of all rows of cells | That of one row of cells, twice, and one kernel more |
| Work | Once | Twice |
| Memory | | One number for each thread |

The search is split if it has no more threads than `split-limit`, an
option of `convert-md-exec-to-gpu`. The default is 131072, about the
threads that the device that was measured runs at once. With 25 rows of
cells that is 5000 particles.

The matrix is the same either way.

### 2.6 Plan parameters

| Parameter | Option | Default |
|---|---|---|
| The skin | `skin` of `convert-md-to-md-exec`; in a control file `pairlistdist − cutoffdist` | |
| The width of a row of the matrix | `width` of `convert-md-to-md-exec`; in a control file `neighbor_width` | Half as many again as a uniform density gives |
| The least width of the cells | `cells` of `convert-md-to-md-exec` | A third of the reach |
| The most threads of a split search | `split-limit` of `convert-md-exec-to-gpu` | 131072 |

## 3. The test of validity

A structure that was built at the positions `x_ref` is valid for the
positions `x` while

```text
max_i |x_i − x_ref,i| ≤ skin / 2
```

and the cell is the one it was built in. Two particles that are farther
apart than the reach at `x_ref` have then not come closer than the cutoff.

| Method | State |
|---|---|
| The thread that moves a particle in the drift tests it, and sets a flag if it is beyond the limit (D41). | Implemented; see [ops-m0.md](ops-m0.md), Section 9.6 |
| The largest displacement as a global maximum | Replaced. It is what a refresh does that is given no result of a test. |

Other schemes that were considered are in
[decisions.md](decisions.md), Section 5.6.

## 4. Measurements

### 4.1 How the times are measured

| Item | Method |
|---|---|
| System | A Lennard-Jones liquid at the density 0.58: particles on a cubic lattice with the spacing 1.2, each moved by up to 0.1 along each direction, at rest. Cutoff 2.0, skin 0.2, rows of 64 neighbors. |
| Integration | 200 steps of velocity Verlet with a time step of 0.004 |
| Time of a step | The program reads a clock before and after the loop over the steps. The time includes the builds and excludes the compilation and the start of the device. |
| Time of a kernel | The runtime library for devices, with `MDRT_PROFILE` and `MDRT_WAIT` set: the time from the launch of a kernel to its end. The host then waits after every kernel, so the sum of the kernels is more than the time of a step. |
| Repetition | The least of three runs |
| Machine | A host with 128 cores and an RTX 3090, CUDA toolkit 11.2 |
| Argon | `examples/argon.toml`: 864 atoms, cutoff 0.85 nm, reach 0.95 nm, 2000 steps of 5 fs, with `mdir-run` |

The liquid starts from a lattice, so its particles are in an order that
follows their positions. Section 5 has the times for another order.

### 4.2 The time of a build

Microseconds for a build, on the GPU in the mixed mode.

| Particles | Before | Now |
|---|---|---|
| 216 | 381 | 118 |
| 512 | 391 | 127 |
| 1728 | 425 | 143 |
| 4096 | 464 | 167 |
| 13824 | 623 | 223 |
| 32768 | 1195 | 279 |
| 110592 | 3301 | 480 |
| 262144 | 7145 | 846 |
| Argon, 864 | 963 | 117 |

"Before" is a build with cells of the reach, a search in the type of the
positions that looks up every position through an index, and one thread
for the offsets of the cells.

A structure is built every 18 steps or so. At 262144 particles a build now
takes as long as one step.

### 4.3 What the choices give

Microseconds for a build of the Lennard-Jones liquid, with the width of
the cells and the threads of the search set by hand.

| Particles | Reach | Half | Half, split | A third | A third, split |
|---|---|---|---|---|---|
| 216 | 141 | 148 | 119 | 169 | 125 |
| 1728 | 173 | 155 | 133 | 201 | 155 |
| 4096 | 191 | 178 | 163 | 250 | 213 |
| 13824 | 207 | 243 | 257 | 338 | 342 |
| 32768 | 304 | 357 | 427 | 409 | 531 |
| 110592 | 477 | 695 | 758 | 693 | 1077 |
| 262144 | 826 | 999 | 1469 | 1158 | 2833 |

| Observation | Consequence |
|---|---|
| At this density, 6 particles in a cube of the reach, cells of the reach are best for a large system. | The width is chosen from the density, not fixed (Section 2.2). |
| For a small system narrow cells are better even at this density: the cell has few cells along an edge, and they are wider than they need to be. | The cost counts the cells that there are in this cell. |
| A split search is faster up to 4096 particles and slower from 13824 on. | The limit of Section 2.5 |
| Before the cells of a row were read as one run, cells of half the reach took 1097 microseconds at 262144 particles, and cells of the reach 854. | Section 2.4 |
