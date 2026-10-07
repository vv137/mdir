# Neighbor Structures for Milestone M0: Method

Status: implemented (2026-09-29).

This document describes how MDIR builds a neighbor structure, keeps it
valid, and keeps the particles in an order in which a loop over pairs
reads little memory; why it does so in this way; and how the times in it
were measured.
The ops are specified in [ops-m0.md](ops-m0.md), Section 8.2.

## 1. The structure

A neighbor structure is a neighbor matrix: for each particle a row of the
particles within the reach, which is the cutoff plus the skin, as in the
neighbor list of Verlet [[Verlet1967]](references.md#verlet1967).

```text
count[i]            number of neighbors of particle i
index[i][0 .. W)    their indices; entries from count[i] on are unused
```

| Property | Value |
|---|---|
| Directed | If `j` is in the row of `i`, then `i` is in the row of `j`. |
| Complete | A row holds every particle within the reach. It may hold particles that are beyond the reach by less than the margin of Section 2.3; a loop over pairs tests the cutoff for every entry, so these contribute nothing. |
| Order of a row | The order of the cells that the search visits, and within a cell the order of the indices. The order does not depend on the threads, on the device, or on the run. |
| Overflow | `W` is the width a row has at first. A build that finds a row too narrow has the runtime make the rows a quarter wider than needed, and builds again. |

The order of a row decides the order in which a loop over pairs adds up
the forces on a particle, and with it the last bits of the forces.

On a device the matrix is kept in the order of the cells of its build
(D86): `order[p]` is the particle at the place `p`, the row `p` is that of
that particle, and its entries are places. A kernel of each step gathers
the positions and the fields that a loop over pairs reads into that order,
and the loop writes the forces of a particle at its place in the state.
The places of a cell are next to one another in memory, and the order is
new with every build.

## 2. The build

### 2.1 Steps

The build is a template in IR:
`lib/Runtime/Templates/NeighborsMatrix.mlir` for the host and
`lib/Runtime/Templates/NeighborsMatrixGPU.mlir` for a device. The compiler
adds the template to the module, where it is lowered with the rest of the
code. The two templates build the same matrix, entry by entry. Each has a
build of its own for a triclinic cell, `_triclinic`, which bins the
fractional coordinates (D123, D125).

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

The search bins the particles into cells [[Quentrec1973]](references.md#quentrec1973).
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

$$\text{margin} = 3 \times 10^{-6}\,(L_x + L_y + L_z).$$

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

A warp of 32 threads searches for one particle. Its lanes test 32
particles of a run of cells at once, and a ballot gives each neighbor its
place in the row: the number of lanes before it that found one, added to
the count so far. The row is the one that a thread for the particle would
fill, testing the particles one after the other, entry by entry. The
count goes on past the width of a row without writing, so that the build
can report how wide the row must be.

The excluded pairs are entered in the search: a neighbor that is an
excluded pair of the particle is written as the particle itself, which
the loops over pairs skip (Section 1). The excluded pairs of a particle
are few (the incidence structure of the relation `excluded`), and only
the neighbors that the ballot keeps are looked up.

| Search (JAC, 23,558 atoms, reach 10 Å, RTX 3090) | µs per build |
|---|---|
| A thread per particle, as before (2026-09-30) | 520 in the app, with 180 more for a kernel that marked the excluded pairs with a thread per entry of the matrix |
| A thread per row of cells of a particle, counted and then filled (the former split search) | Slower on JAC than a thread per particle |
| A warp per particle, in a prototype | 340 to 390; 440 with the excluded pairs |
| A warp per particle with the excluded pairs, in the app, counting in `index` (64 bits) | 612, and no kernel for the marks: 64 registers and spills |
| The same, counting in `i32`, the partners of the excluded pairs in registers (now) | 430 (the first build of a run: 653 before), 56 registers and no spills |
| A block per cell, the runs of cells in shared memory, in a prototype | 425 to 435 |

The block per cell reads a run once for all the particles of the cell,
and was no faster: the search is bound by the tests (1,700 candidates
per particle, 4 × 10⁷ for JAC), not by the loads. The prototypes and
their results are in
[scripts/experiments/neighbor-structures](../scripts/experiments/neighbor-structures/README.md).

The width of the cells is chosen for the warp: a row of cells costs one
step of 32 tests to visit, and its particles one step for every 32
(`mdrt_gpu_cell_width`). With a thread per particle the cost of a row was
that of 12 particles, and the cells were a third of the reach; for the
warp that width took 846 µs on JAC, half the reach 612.

The kernel counts in `i32`: the numbers of the particles, of the cells,
and of the entries of a row. With `index`, which is 64 bits on the
device, it needed more than 64 registers and spilled, and took 612 µs
where a CUDA program of the same search took 440. The partners of the
excluded pairs of a particle are held one per lane and compared by
shuffles, and only when a neighbor's number lies between the least and
the greatest of them; a particle with more than 32 compares the rest in
memory.

### 2.6 Plan parameters

| Parameter | Option | Default |
|---|---|---|
| The skin | `skin` of `convert-md-to-md-exec`; in a control file `pairlist_distance − cutoff` | |
| The width of a row of the matrix | `width` of `convert-md-to-md-exec`; in a control file `neighbor_capacity`, in Python `Execution.neighbor_capacity` | Half as many again as a uniform density gives; the driver estimates it from the start: half as many again as the most neighbors of a particle, plus 16, rounded up to four significant bits (D227) |
| The least width of the cells | `cells` of `convert-md-to-md-exec` | A third of the reach |

## 3. The test of validity

A structure that was built at the positions $\mathbf x^\text{ref}$ is valid for the
positions $\mathbf x$ while [[AllenTildesley2017]](references.md#allentildesley2017)

$$\max_i \lVert \mathbf x_i - \mathbf x^\text{ref}_i \rVert \le s / 2,$$

with $s$ the skin, and the cell is the one it was built in. Two particles that are farther
apart than the reach at $\mathbf x^\text{ref}$ have then not come closer than the cutoff.

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
| Argon | `examples/argon/argon.toml`: 864 atoms, cutoff 0.85 nm, reach 0.95 nm, 2000 steps of 5 fs, with `mdir run` |

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

## 5. The order of the particles

### 5.1 Why the order matters

A loop over pairs reads, for each particle, the positions of its
neighbors. If particles that are near one another in space are near one
another in memory, these are few places in memory. If not, every neighbor
is a place of its own, and for a large system most of them are not in the
cache.

The order of the input is what it is: that of a file, of molecules, of a
lattice. It also decays, because the particles of a liquid move away from
where they were.

### 5.2 Method

The particles are put in the order of their positions (P17, D44):

```mlir
%order = md_exec.spatial_order %x, %cell, %ids width(0.5)
%xs    = md_exec.permute %x, %order       // and every other field
```

| Item | Choice | Reason |
|---|---|---|
| The order | By cell, the cells numbered along x first, then y and z. Within a cell by the number of the particle. | A row of cells is a run in memory, as the search of a build reads it (Section 2.4). |
| The width of the cells | Half the reach | The neighbors of a particle are in 25 runs. |
| What is put in order | Every field of the particle set: the state, the masses, the parameters, the numbers of the particles | A loop reads the fields of a particle at one index. |
| When | Where the run begins, and where a segment begins | A segment begins with empty neighbor structures (R1), so no structure holds indices of the order before. |
| The number of a particle | Its place in the input. The run carries the numbers as a field and puts them in order with the others. | The files of a run are in the order of the input: a function of the host reads a field together with the numbers. |

The order is computed as a build computes the order of the cells
(Section 2.1, steps 2 to 5), with the sort of step 5 by the numbers of the
particles. A field in the order is written to a buffer of its own; the
buffer that held the field holds the next field that needs one.

**Exact restart.** The order is a function of the positions and of the
numbers of the particles. It does not depend on the order that the
particles are in. A run that continues from a checkpoint, whose particles
are in the order of the input, therefore arrives at the order of the run
that was not interrupted, and at its results, bit for bit. The tests
compare the states.

**Between two orders** the particles move. With the diffusion
coefficient of liquid argon, about 2e-3 nm²/ps, a particle moves 0.35 nm
in 10 ps, less than the width of a cell, and 1.1 nm in 100 ps. Measured on
JAC (RTX 3090, 2026-09-30), when the loops over pairs read the state in
this order: over 40 ps the loop over pairs went from 117 to 208 µs a step,
and a run of 200 ps ran 20 % slower than one of 40 ps. The loops over pairs
on a device therefore read their particles in the order of the cells of
the last build of their structure (Section 1, D86); the other kernels kept
their times.

| Alternative | Why not |
|---|---|
| A new order with every build of a neighbor structure | The build happens inside a step, where it is decided by a test. Every field that the loops carry would have to pass through the refresh. Every 18 steps is more often than the order decays. |
| A copy of the positions in the order of the cells, inside the neighbor structure, renewed in every step | One kernel more in every step, and the fields of the parameters would need copies as well. The state stays as it is, which is simpler for the files. |
| The order of a space-filling curve | Better for the rows of cells along z and y. Not measured. |

The control file has the keyword `spatial_order` in `[execution]`. The default
is `true`.

### 5.3 Measurements

The Lennard-Jones liquid of Section 4.1, with the particles on the
lattice in three ways:

| Order | Meaning |
|---|---|
| Lattice | Particle $i$ is on site $i$. The order follows the positions. |
| Scattered | Particle $i$ is on site $i p \bmod N$, with a number $p$ that has no divisor in common with $N$. Neighbors in space are far apart in memory. |
| Scattered, put in order | The scattered particles, put in the order of their positions before the first step |

Milliseconds per step on the GPU in the mixed mode:

| Particles | Lattice | Scattered | Scattered, put in order |
|---|---|---|---|
| 4096 | 0.0609 | 0.0616 | 0.0636 |
| 32768 | 0.175 | 0.175 | 0.165 |
| 110592 | 0.421 | 0.407 | 0.416 |
| 262144 | 0.819 | 1 | 0.828 |
| 1000000 | 2.95 | 4.11 | 2.97 |

On the host with 16 threads, in double precision:

| Particles | Lattice | Scattered | Scattered, put in order |
|---|---|---|---|
| 4096 | 0.175 | 0.193 | 0.184 |
| 32768 | 1.41 | 1.49 | 1.41 |
| 110592 | 4.7 | 5.17 | 4.69 |
| 262144 | 10.7 | 11.7 | 10.6 |
| 1000000 | 39.5 | 44.7 | 39.9 |

| Observation | Consequence |
|---|---|
| On the GPU in the mixed mode, a scattered order costs nothing up to 110592 particles, 22 percent at 262144, and 39 percent at a million. | The order matters for large systems. |
| Put in order, the scattered particles take the time of the lattice. | The order of the cells is as good as that of the lattice. |
| On the GPU in double precision the order changes the time by less than 1 percent. | There the arithmetic takes the time, not the memory. |
| On the host a scattered order costs 6 to 13 percent at every size. | |
