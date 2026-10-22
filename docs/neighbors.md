# Neighbor Structures for Milestone M0: Method

Status: implemented (2026-09-29) and kept current (last checked
2026-10-21). Section 6, the choice of the skins (D281), is of 2026-10.

This document describes how MDIR builds a neighbor structure, keeps it
valid, and keeps the particles in an order in which a loop over pairs
reads little memory; why it does so in this way; and how the times in it
were measured.
The ops are specified in [ops.md](ops.md), Section 8.2.

## 1. The structure

There are two kinds of neighbor structure (`kind` of
`md_exec.empty_neighbors`): the neighbor matrix of this document, on the
CPU and on a device, and on a device only the groups of 16 particles that
share a list, each pair once (D89, [neighbor-groups.md](neighbor-groups.md)), with an
optional dual list (D114). `neighbor_structure = "AUTO"`, the default,
takes the groups on a GPU for large systems and the matrix elsewhere
(D261). The test of validity (Section 3), the order of the particles
(Section 5), and the choice of the skins (Section 6) serve both.

The neighbor matrix holds for each particle a row of the
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
| Order of a row | The order of the cells that the search visits, and within a cell the order of the indices. The order does not depend on the threads or on the run. The host and a device choose the width of the cells with different costs (Sections 2.2 and 2.5), so the cells that a search visits, and with them the order of a row, can differ between the two. |
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
code. The two templates find the same neighbors and differ in what a row
holds: on a device the entries are places in the order of the cells
(Section 1) and an excluded pair is entered as the particle itself
(Section 2.5); on the host the entries are the indices of the particles,
the template takes no excluded pairs, and the lowering removes them from
the rows after the build and lowers the counts. Each has a
build of its own for a triclinic cell, `_triclinic`, which bins the
fractional coordinates (D123, D125).

| Step | Work | On a device |
|---|---|---|
| 1 | Choose the width of the cells (Section 2.2) | On the host |
| 2 | For each particle: the position in the cell, in f32, and the cell | One thread per particle. The particles of a cell are counted with an atomic addition. |
| 3 | The offsets of the cells, from the counts | Three kernels: the sums of chunks of 256 cells, the offsets of the chunks in one thread, the offsets of the cells of each chunk |
| 4 | The particles in the order of the cells | One thread per particle takes the next slot of its cell, with an atomic addition |
| 5 | The particles of each cell in the order of their indices, and their positions in that order | One thread per particle: its place within its cell is the number of the cell's particles with a smaller index, which it counts over the cell (D235); then a copy of the particles and of the positions in that order. A small system has few cells of many particles, which one thread per cell sorted by insertion in most of the time of a build |
| 6 | The search: for each particle, test the particles of the cells within reach and fill the row (Sections 2.3 to 2.5) | A warp of 32 threads per particle (Section 2.5) |
| 7 | The largest count, and the counts limited to the width of a row | In the kernel of the search: the first lane of a warp stores the limited count of its particle and raises the largest count with an atomic maximum, which does not depend on the order of the threads |

Steps 2 to 4 are a counting sort. On the host it is sequential and keeps
the order of the indices, so step 5 is only the copy.

Step 3 was one thread on a device. With narrow cells there are as many
cells as particles, and one thread took longer than the search.

Step 7 was two kernels of its own on a device, sums over chunks of 256
particles and one thread for the chunks; the search now gives the largest
count itself.

On a device, a position that is not a number, or whose largest coordinate
is beyond $10^{100}$, gets no cell and no place in step 2 (D107): the
search skips the place, the build counts such positions, and the caller
stops the run on a count that is not 0.

### 2.2 Cells and their width

The search bins the particles into cells [[Quentrec1973]](references.md#quentrec1973).
Along an edge of the length `L` there are `n = floor(L / w)` cells, of the
width `L / n`, which is `w` or more. A search for the neighbors of a
particle visits the cells within reach of the cell of the particle:
`r = ceil(reach / (L / n))` cells on each side, or all cells, each once,
where these are fewer. `n` is at least 1 and at most 256 along each axis
(#168, D225): a cell that is not a number, or one that a failed run has
blown up, then gives a count that is defined and fits memory. With 256
cells the cells are wider than `w`, which costs only time, since `r`
follows from the count. The largest system of the Amber suite takes at
most 77.

| Width `w` | Cells visited | Volume visited, in cubes of the reach |
|---|---|---|
| The reach | 3 × 3 × 3 | 27 |
| Half the reach | 5 × 5 × 5 | 15.6 |
| A third of the reach | 7 × 7 × 7 | 12.7 |
| The sphere that holds the neighbors | | 4.2 |

Narrow cells have fewer particles to test and more cells to visit. A
search reads the cells of a row as one run (Section 2.4), so what counts
is the number of rows. On the host, visiting a row costs as much as
testing 12 particles; a device counts in steps of a warp instead
(Section 2.5).

The build chooses the width when it runs, because the density is known
only then:

```text
for w in reach, reach / 2, reach / 3, each raised to the least width:
    particles tested = N · cells visited / cells
    host:    cost(w) = 12 · rows visited + particles tested
    device:  cost(w) = rows visited · 1.5 + particles tested / 32
the width with the least cost is taken; on a tie, the wider
```

The widths of the table below are examples from the time of writing. The
width that a build takes depends on the target, which has its own cost,
and on the box, through the number of cells that fit along an edge.

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

$$\text{margin} = 3 \times 10^{-6}\,(L_x + L_y + L_z),$$

and in a triclinic cell $3 \times 10^{-6}\,(a_x + b_y + c_z + \lvert b_x\rvert + \lvert c_x\rvert + \lvert c_y\rvert)$.

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
the loops over pairs skip (a loop takes only entries that are not its own particle). The excluded pairs of a particle
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
and a half steps of 32 tests, the visit and a last step that is half empty
on average, and its particles one step for every 32
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
| The skin | `skin` of `convert-md-to-md-exec`; in a control file `pairlist_distance − cutoff` | 0 for the pass; in a control file `pairlist_distance = "AUTO"`: for the matrix the cutoff plus 1.5 Å, which is not searched; for the groups the cutoff plus 3 Å, or the choice of a search (Section 6, D261, D281) |
| The width of a row of the matrix | `width` of `convert-md-to-md-exec`; in a control file `neighbor_capacity`, in Python `Execution.neighbor_capacity` | 64 for the pass; the driver always gives one, from `neighbor_capacity` or estimated from the start: half as many again as the most neighbors of a particle, plus 16, rounded up to four significant bits (D227). The most neighbors are counted exactly, on the host, with cells as wide as the reach over the threads of the process (D316). The lists of groups do not take their room from the width: their first build sizes them (D331, [neighbor-groups.md](neighbor-groups.md), Section 2) |
| The least width of the cells | `cells` of `convert-md-to-md-exec` | A third of the reach |

## 3. The test of validity

A structure that was built at the positions $\mathbf x^\text{ref}$ is valid for the
positions $\mathbf x$ while [[AllenTildesley2017]](references.md#allentildesley2017)

$$\max_i \lVert \mathbf x_i - \mathbf x^\text{ref}_i \rVert \le s / 2,$$

with $s$ the skin, in the cell it was built in. Two particles that are farther
apart than the reach at $\mathbf x^\text{ref}$ have then not come closer than the cutoff.

A cell that a barostat has scaled since the build does not end the
validity (D80, `md-exec-expose-validity`). With the edges $\mathbf L$ now
and $\mathbf L^\text{ref}$ at the build, $\mathbf m = \mathbf L \oslash
\mathbf L^\text{ref}$ per axis, the test compares each position with its
reference scaled as the cell was, against a limit that shrinks with the
least scale:

$$\max_i \lVert \mathbf x_i - \mathbf m \odot \mathbf x^\text{ref}_i \rVert \le \tfrac12 \max\bigl(\min(\mathbf m)\,(r_c + s) - r_c,\ 0\bigr),$$

which is $s/2$ in the cell of the build. Only the three edges of the
diagonal enter, also for a triclinic cell. With `rebuild_interval` the
structure is built every $n$ refreshes of the steps and not tested in
between (D88).

The potential of an output (`[output] observables`, the free-energy file, a
pull file) has the reach of the forces and shares the structure of the
steps (`md-exec-reuse-neighbors`, which leaves the cell out of what two
structures must have in common). It is evaluated at the state that the step
has just evaluated its forces at, so its refresh would repeat that of the
step: the same positions and cell, as values, and the same parameters. The
pass removes such a refresh, through a loop over states as well, and the
output reads the structure of the step (D237, #233): it
neither tests nor builds, and does not count for `rebuild_interval`. A
refresh at another state is not removed. Given the positions of one cell
and another cell (the outputs at a step where the barostat of Trotter type
scales, before the fix of #218), the test compared the positions with a
reference scaled for the wrong cell, built the structure where the steps
would not have, and recorded the wrong cell as $\mathbf L^\text{ref}$; the
steps then built at other steps and summed their forces in another order.
An output that is evaluated elsewhere than at the state of the step needs a
structure of its own; `output-independence*.test` compare the run with each
output against the run without it, bit for bit in the deterministic mode.

| Method | State |
|---|---|
| The thread that moves a particle in the drift tests it, and sets a flag if it is beyond the limit (D41). | Implemented; see [ops.md](ops.md), Section 9.6 |
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
| A split search is faster up to 4096 particles and slower from 13824 on. | The split search is no longer in the build: the search is a warp per particle (Section 2.5) |
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
%order = md_exec.spatial_order %x, %cell, %ids width(0.5)   // a length, half the reach
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

## 6. The choice of the skins

D281, issue #406. `pairlist_distance = "AUTO"` and
`pruned_distance = "AUTO"`, and the keys not given, let a run choose the
skin $s$ of its structure and the skin $s_\text{in}$ of the inner list of a
dual list (D114). A number fixes a skin, and is the switch off.

### 6.1 Why the run may choose them

The test of Section 3 is exact: a structure is rebuilt, and an inner list
pruned, when a particle has moved half its skin, so every pair within the
cutoff is in the list at every step whatever the skins. The skins change
which steps build and prune and the length of the lists, and so the speed
and the order of the sums, not the pairs. In the deterministic mode the
order of the entries of a row of the matrix follows its cells, and so its
reach: the water box of `pme-settle.test` at 10.0, 10.6, and 11.5 Å, 500
steps on an RTX 4090 in double precision, gives the energies of the log to
every digit and positions that differ by up to $1.8\times10^{-9}$ nm, also
with `spatial_order = false`. The deterministic mode therefore does not
search: a search chooses by timing, and the bits of a deterministic run
repeat only if its skins do (the maintainer's ruling on PR #407; sums independent of the reach are #409).

### 6.2 The cost of a step

A step of the groups with a dual list costs, per particle,

$$\frac{t_\text{step}}{N} = a + c_1\rho' R_\text{in}^3 + \frac{c_2\rho' R^3}{n_\text{prune}} + \frac{c_3 + c_4\rho' R^3}{n_\text{build}} + \frac{L}{N\,n_\text{build}},$$

with $\rho' = \rho / (0.1\ \text{Å}^{-3})$ the density relative to that of
water: the loop over the inner list, the pruning of the outer list every
$n_\text{prune}$ steps, and the build every $n_\text{build}$ steps with a
cost per particle and a fixed cost $L$ of its launches and its readback.
One list is the case $R_\text{in} = R$ without the pruning.

The intervals follow from the motion of the particles. A list is rebuilt
when the largest displacement since its build reaches half its skin. In
equilibrium each component of a velocity is Gaussian with the variance
$k_BT/m_i$, so over the $N$ particles the largest displacement after a time
$t$ is a few times the root mean square of the lightest; it grows as
$\sqrt{3k_BT/m}\,t$ while the velocities are correlated and more slowly
once they decorrelate, toward $\sqrt{6Dt}$. The inner skin, crossed within a
few steps, is in the first regime, $n_\text{prune} \approx
s_\text{in}/(2v_\text{max}\Delta t)$; the outer skin, crossed in tens of
steps, is between the two, and $n_\text{build}$ grows faster than linearly
with $s$. With the first regime, the inner skin that minimizes the cost
solves $3c_1R_\text{in}^2 = 2c_2R^3v_\text{max}\Delta t/s_\text{in}^2$, and
for $s_\text{in} \ll r_c$ it grows as $\sqrt{\Delta t}$: by $\sqrt 2$ from
2 to 4 fs, as the grid of #406 shows, where the default of D261 grows in
proportion to $\Delta t$.

The constants were fitted to the grid of #406: 480 runs of five systems of
the Amber suite on an RTX 4090, mixed precision, $r_c = 8$ Å, outer skins
of 1.5 to 6 Å and inner ones of 0.45 to 1.0 Å at 2 fs ($\times\Delta t/2$
fs), with the intervals that each run measured. Per particle,
$c_1 = 1.31$, $c_2 = 0.365$, $c_4 = 0.081$ ps per Å³, $c_3 = 2.12$ ns, and
$L = 74$ µs; the root mean square of the relative residuals is 1.0%.
Within each system $n_\text{build} \propto s^{1.57}$ and $n_\text{prune}
\propto s_\text{in}^{0.76}$ (the means of the five fits). On another
device the constants are used as they are: the model only narrows the
candidates, and the measurement decides.

The grid was measured before the loop over pairs pruned (D284,
[neighbor-groups.md](neighbor-groups.md), Section 6.2). Where it does, from 4096
groups on an RTX 4090, a pruning costs only a few percent of the loop
rather than $c_2\rho' R^3$, so the model overstates the cost of a short
inner skin, and the fastest $s_\text{in}$ may be shorter than it
predicts. The constants are kept: the candidates are measured, and a
refit belongs with a new grid.

### 6.3 The search

`mdir run` searches where all of these hold; elsewhere `"AUTO"` keeps the
default of D261 (cutoff + 3 Å and an inner list at cutoff + 0.6 Å ×
max(1, Δt / 2 fs) where `AUTO` takes the groups; cutoff + 1.5 Å for the
matrix and for `GROUPS` given), and the log says why:

- the target is a GPU, not in the deterministic mode;
- the structure is the groups (the matrix is not searched);
- not a minimization, no `rebuild_interval`, no `[[energy.model]]`;
- no entry of the cache for the key (Section 6.4);
- particles times steps of at least $5 \times 10^{12}$, about five hours on an RTX 4090
  at about 4 ns a particle and a step. `MDIR_SKIN_SEARCH=always` searches
  whatever the length, `never` never.

The search (`lib/Compiler/Skins.cpp`, `chooseSkins`):

1. **The reference**, the dual list of D261 for the skins that are chosen
   (also where `GROUPS` is given), is measured first. Its intervals of
   builds and prunings calibrate $n_\text{build}$ and $n_\text{prune}$ of
   the model to the system, which stands in for its temperature, its
   masses, its density, and its step.
2. **Two candidates of the model**: its best pair on a grid of outer skins
   of 1.5 to 6 Å by 0.5 Å and inner skins of 0.3 to 1.5 Å by 0.05 Å, and
   its best pair whose outer skin is at least 1 Å from that one and from
   the reference (0.2 Å of the inner skin where only it is chosen). The
   landscape is not smooth (JAC's rate falls from 4 to 5 Å and rises again
   to 6 Å, #406), so the second looks elsewhere.
3. **A climb** from the fastest so far, while it gains: steps of 0.5 Å of
   the outer skin away from the reference (toward the model's best where
   the reference is the fastest), then of 0.1 Å of the inner skin, up, or
   down if the first step up does not gain; at most seven candidates in
   all.

The candidates must pay for themselves: after the reference, the run
affords as many as fit, at the reference's cost of a compilation and its
steps each, within 2% of the time of the run at the reference's rate, at
most seven. With fewer than three it keeps the reference, and caches
nothing, so a longer run of the same key may still search. The
compilation grows with the system: 11 to 13 s a candidate for JAC, 17 to
22 s for Factor IX, 86 to 130 s for STMV on the node of an RTX 4090
(16 cores).

Each candidate is a simulation of its own (D196), compiled with its skins
as constants of its program, from the state the run begins with: two
warm-ups of 200 steps, then six blocks of about 0.5 s and at least 200
steps each, timed one by one; a part ends at a boundary that reads from the
device, so a block has waited for it. The time of a candidate is the median
of its blocks, and its error the standard error of their mean. A candidate
whose reach exceeds 0.999 of the least edge of the cell (D242) or 127 times
the cutoff is not tried; one that fails is dropped. The fastest is kept
unless its gain over the reference is within twice the standard error of
the difference, and at least 0.3%; then the reference is kept. The counts
of builds and prunings of the trials and their handler of failures are
reset before the run (`mdrtResetCounts`), so its log counts its own.

### 6.4 The cache and the record

The choice is stored as one JSON file a key, under `MDIR_SKIN_CACHE_DIR`,
by default `$XDG_CACHE_HOME/mdir/skins` (else `~/.cache/mdir/skins`);
`MDIR_SKIN_CACHE=off` neither reads nor writes it. The key is the text of
what the choice depends on: the version of MDIR, the particle count, a
hash of the composition (the types, masses, and charges), the cutoff, the
structure, each reach given as a number or `AUTO`, the name of the device,
the precision, the time step, the ensemble, its temperature, and whether
the loop over pairs prunes (`MDIR_FUSED_PRUNE`, D284; a key
of before misses once). A file
whose key differs is not taken, so a run of another step, device, or
system searches anew; a continuation, which has the key of the run it
continues, takes the choice of that run. Removing the directory, or an
entry, is always safe.

The log names the reaches and where they came from when a key was given
as `"AUTO"` or the reaches were cached or measured (`MDIR: skins: the
reach 14 Å, pruned to 8.8 Å (measured: ...)`), with a line for each
candidate. The manifest's `effective` has `pairlist_distance_source` and
`pruned_distance_source` (`given`, `default`, `cached`, or `measured`)
and `skin_search` (the source, the reason, the file of the cache, the
seconds, and each candidate's reaches, time, error, intervals, and the
model's prediction relative to the reference); `mdir check` prints
`skins:`, and `run.skins` with `--json`; `Program.plan` has the sources and
`skin_reason`. `mdir check` and `mdir.compile` measure nothing: they name
the default or the cache. In Python, `System.pairlist_distance` and
`System.pruned_distance` take `mdir.Auto`.

### 6.5 Measurements

The Amber suite on one RTX 4090 (`scripts/benchmarks/amber/bench.py`'s
control files, mixed precision, the groups given, $r_c = 8$ Å, twice the
suite's steps), in one job: the default of D261 (11 Å and an inner list at
8.6 Å, 9.2 Å at 4 fs), the best point of the grid of #406 where there is
one, and `"AUTO"` with `MDIR_SKIN_SEARCH=always`, in two rounds, the
second in reverse order, whose `"AUTO"` takes the cache. µs a step from the
second half of each run (the search is before it), the mean of the rounds;
reaches in Å.

| System | Atoms | Default | Grid's best | `"AUTO"` chose | `"AUTO"` | / best | / default | Search |
|---|---|---|---|---|---|---|---|---|
| JAC NVE, 2 fs | 23,558 | 136.7 | 131.4 (14 / 8.8) | 14 / 8.8 | 133.0 | 1.012 | 0.972 | 95 s |
| JAC NVE, 4 fs | 23,558 | 146.4 | 140.5 (14 / 8.9) | 14 / 8.8 | 140.9 | 1.003 | 0.962 | 94 s |
| JAC NPT, 2 fs | 23,558 | 150.4 | | 13.5 / 8.9 | 147.1 | | 0.978 | 189 s |
| JAC NPT, 4 fs | 23,558 | 154.9 | | 13.5 / 8.7 | 151.4 | | 0.977 | 186 s |
| Factor IX NVE | 90,906 | 335.9 | 332.9 (12.5 / 8.6) | 11 / 8.6 | 335.6 | 1.008 | 0.999 | 132 s |
| Factor IX NPT | 90,906 | 358.9 | | 11 / 8.6 | 358.7 | | 0.999 | 197 s |
| Cellulose NVE | 408,609 | 1458.8 | 1448.4 (11.5 / 8.45) | 11.5 / 8.65 | 1435.8 | 0.991 | 0.984 | 229 s |
| Cellulose NPT | 408,609 | 1519.1 | | 11.5 / 8.65 | 1498.7 | | 0.987 | 286 s |
| STMV NPT, 4 fs | 1,067,095 | 4731.0 | 4565.4 (12.5 / 8.9) | 12 / 8.7 | 4535.5 | 0.993 | 0.959 | 848 s |

`"AUTO"` is within 1% of the grid's best on four systems of five and
faster than it on two; on JAC at 2 fs it chose the grid's best point, and
the 1.2% is its first round, which runs right after the search: a run of
the same skins after 105 s of steps is 1.0% slower than one after a rest
(1320.5 against 1333.4 ns/day), from the device's heat, and the round that
took the cache was within 0.2%. A run that searched and one that took the
cache in a fresh process give the same rate (1334.1 against 1333.1
ns/day). On Factor IX no candidate gained beyond the noise, and the
default was kept, 0.8% from the grid's point at 12.5 Å. Against the
default the gain is 0 to 4.1%. The search costs 1.5 to 5 minutes, and 14
on STMV, whose compilation takes 86 to 130 s a candidate; a run affords
the candidates within 2% of its time (Section 6.3).

On an RTX PRO 6000 Blackwell (Max-Q), three rounds in one job, the search
chose other skins than on the RTX 4090 (Factor IX 13.5 / 8.7 Å, JAC
13.5 / 9 Å, STMV 12.5 / 8.8 Å), but the reference's blocks varied by up to several percent (its noise 2.9 to 7% on JAC), and the rounds of one setting by up to 4% (STMV with the cached skins: 3326 and 3456 µs), on a node of six devices that other users' jobs share. Where the noise covered the gain the search kept the default, as it should (JAC at 4 fs, JAC NPT at 2 fs, Cellulose NVE; on JAC at 4 fs 13.5 / 8.9 Å was 4.9% faster within a noise of the same size), so against the grid's best `"AUTO"` was 0.983 (JAC 2 fs), 1.032 (JAC 4 fs), 1.005 (Factor IX), 1.005 (Cellulose), 1.004 (STMV), the means of three rounds, and against the default 0.868 to 1.003. A comparison on that device wants a node that runs nothing else.
