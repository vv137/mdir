# Groups of Neighbors: Design and Algorithm

Status: accepted (2026-10-01), D89; kept current (last checked
2026-11-11). G1 done; G2 done in the default mode (`neighbor_structure =
"GROUPS"`), while the deterministic mode does not take the groups yet: the
control file refuses `GROUPS` with `deterministic`, and `AUTO` keeps the
matrix there; G3 done (D261): `neighbor_structure = "AUTO"`, the default,
takes the groups with a dual list (D114) on a GPU from 23,000 particles
(Section 6.3), and a run chooses the skins of both lists where they are
`"AUTO"` (D281, [neighbors.md](neighbors.md), Section 6). The
groups run in triclinic cells (D126), hold every image within a reach
below the least edge (D242), and prune the inner list in the loop over
pairs on large systems (D284). They replace the tile structure of D82
([neighbor-tiles.md](neighbor-tiles.md)), whose prototype was slower than
the neighbor matrix. The keys in brackets are those of
[references.md](references.md).

This document describes the second kind of neighbor structure on a
device, `groups`: sixteen particles that share one list of neighbors, over
which a loop computes each unordered pair once. The neighbor matrix
([neighbors.md](neighbors.md)) stays: on the CPU, for kernels that cannot
be computed once a pair, in the deterministic comparison of the two, and
on a device where it is faster.

**Sources.** The layout, groups of 16 particles that share one list of
the particles within the reach of any of them, a half list in which each
pair appears once, and a warp that walks a group's list with a mask of
interactions and exclusions for each entry, is the one published by
[[SalomonFerrer2013]](references.md#salomonferrer2013), Section 3.3. The
cluster pair lists of [[Pall2013]](references.md#pall2013) have the same
aim: particles close in space share one list, so that a list is read once
for several particles. What MDIR does in its own way is the order of the
particles (columns of a grid sorted along $z$, then bitonic sorts in $x$
and $y$; Section 2), the search (a grid of cells of half the reach whose
candidates are tested first against the bounding box of the group, then
against each member; Section 4), the frames that let the loop take no
minimum image (D95), the entries for every image within the reach (D115,
D242), the pool of blocks (D331), and the exact test of validity and its
dual list (D80, D114).

Sections 1 to 10 describe the algorithm as it is on `main`, with the
reasons for each choice and pointers to the measurements; Section 11 holds
the measurements, and Section 12 the history, the stages, and the open
questions. The white paper derives the same steps in Sections 4.4 and 4.5,
with the notation of its Section 2.

## 1. The algorithm in brief

### 1.1 Inputs and outputs

A build (`@mdrt_gpu_build_neighbors_groups`, written by the emitter of
`lib/Conversion/MDExecKernels/GroupsBuild.cpp`, `GroupsBuild.h`,
D404) takes:

| Input | Meaning |
|---|---|
| `x` | The positions of the $N$ particles in f64, as the state keeps them (unwrapped) |
| `box` | The cell: the edges $\mathbf L$, or the six numbers of a lower-triangular $H$ in the triclinic build (Section 5.1) |
| `reach` | The reach $R = r_c + s$ (`pairlist_distance`) |
| `excluded` | The excluded pairs of each particle, an incidence structure in ragged rows as for the matrix |

and writes, on the device:

| Output | Per | Meaning |
|---|---|---|
| `order`, `placeOf` | place, particle | The particle at place $p$, $\pi(p)$, or $-1$ for an empty place; the place of each particle |
| `shift` | place | $\mathbf S_p$: the whole cells that move the particle at place $p$, as it is kept, into the frame of its group (Section 3) |
| `entries`, `masks` | slot of a block | An entry $(q, \mathsf b, \mathbf e)$: the place $q$ in an i32; the mask $\mathsf b$ in bits 0 to 15 and the shift $\mathbf e$ in bits 16 to 27 (or 16 to 30) of another |
| `counts` | group | The number $E_g$ of entries of the list of group $g$ |
| `units`, `ordinals` | block | Block $b$ is block `ordinals[b]` of the list of group `units[b]` |
| `sizes` | build | On the host: the places, the largest $E_g$, the blocks, the groups whose excluded partners did not fit the memory of a warp (D106), the positions that are not numbers or are far outside the cell (D107, D176) |

The loop over pairs (Section 6.1) reads `order`, `shift`, the blocks, and
the positions and fields of the kernel gathered in the order of the
places. A dual list (Section 6.2) adds an inner list in buffers of its
own, pruned from these.

### 1.2 The data structures

```text
places:   | chunk 0 of column 0                 | chunk 1 of column 0 | ... | chunks of column 1 | ...
          | group 0 | group 1 | group 2 | group 3 | group 4 | ...
          | 16 places each; a place is a particle or empty (order = -1)

lists:    group g -> entries (q, mask, e), in blocks of 64 taken from one pool
pool:     | block 0       | block 1       | block 2       | ...
units:    | g=7           | g=2           | g=7           | ...      the group of each block
ordinals: | 0             | 0             | 1             | ...      its number in that group's list
entry:    entries[64 b + i] = q (i32)
          masks[64 b + i]   = bits 0-15: b_u, member u pairs with q
                              bits 16-27: e + 4, 4 bits an axis (x, y, z)
                              (triclinic: bits 16-30, e + 16, 5 bits a lattice vector)
shift:    bits 0-9, 10-19, 20-29: S_p + 512 along x, y, z
          (an empty place: 512 + 512*2^10 + 512*2^20 = 537395712, zero along each axis)
```

### 1.3 Invariants

Each build establishes these, and the loops, the pruning, and the tests
rely on them:

- **I1** (each pair once). For every unordered pair $\{i, j\}$ of distinct
  particles that is not excluded, and every image of it within $R$ at the
  build, exactly one bit of one entry is set: in the list of the group of
  the earlier of the two places, for the image that the entry takes
  (Sections 4 and 5). This holds for $R < \min_a L_a$ (Section 5.2); no bit
  is set for an excluded pair or an empty place. Bits may be set for pairs
  a little beyond $R$ (the rounding of f32, Section 7.2), never cleared
  for a pair within it.
- **I2** (fixed order). The entries of a group, and their order, do not
  depend on the threads: a warp builds a group's list from its cells in a
  fixed order (Section 4). Which blocks of the pool a list takes depends on
  the order in which the warps reach the pool's counter.
- **I3** (empty places). An empty place has `order` $= -1$, the shifts
  $\mathbf S_p = \mathbf 0$ and $\mathbf W_p = \mathbf 0$ (the fourth lane
  of its stored position is written, #413), and no entry names it; the
  loop gives it the fields of particle 0, which no bit reads.
- **I4** (frames). An entry is in the frame of its list's group: the
  position of $q$ in the frame of its own group plus $\mathbf e$ is the
  image of the entry, so the loop takes no minimum image (D95, Section 3).
- **I5** (dual list). The inner mask of an entry is a subset of its outer
  mask, and is always computed from the outer one (Section 6.2).
- **I6** (active units). The active units that a pruning lists are blocks
  of the structure, each holding at least one entry of the inner list,
  each once (Section 6.2).

### 1.4 One step

```text
refresh:                                       # md_exec.refresh_neighbors
  moved  <- the test of D80 against the build (Section 7.1)
  stale  <- the same test against the last pruning, with R_in (dual list only)
  if moved:  build (Sections 2 to 5), then prune if dual
  elif stale: prune (Section 6.2)
loop over pairs:                               # md_exec.pair_for over groups
  for each block (or active unit) of the lists: Section 6.1
```

The two tests are fused into the loop that moves the particles; a build
and a pruning are kernels of their own, except that a pruning may be done
by the first loop over pairs (D284, Section 6.2).

## 2. The compact order

The build sorts the particles into an order of *places* in which each run
of 16 places, a *group*, fills a small box. The steps, with
$\tilde{\mathbf x}_i$ the position wrapped into the cell and rounded to
f32:

1. **Columns.** The density is $\rho = N/V$, and the side of a column is
   $d_\text{col} = (64/\rho)^{1/3}$, the edge of a cube that holds 64
   particles on average. There are $N_x = \lfloor L_x/d_\text{col}\rfloor$
   columns along $x$ and $N_y$ along $y$, each clamped to $[1, 256]$, so
   $N_\text{col} = N_xN_y$ columns of sides $L_x/N_x$ and $L_y/N_y$, at
   least $d_\text{col}$ unless the cell is narrower.
2. **Bins along $z$.** Each column is cut into $N_z = \max(1, \lfloor
   2N/N_\text{col}\rfloor)$ bins along $z$, two for each particle of a
   column on average, so that a bin rarely holds more than a few particles.
3. **Counting sort.** The key of particle $i$ is $(k_y N_x + k_x)N_z +
   k_z$, with $k_a$ the column or bin of $\tilde x_{i,a}$, clamped to the
   grid. A count of each key, a scan, and a scatter put the particles in the
   order of the keys; each key's particles are then sorted by their number,
   so the order does not depend on the threads: one `md_alg.sort_by_key`
   (D407), whose key region also writes each particle's wrapped
   position (a slot of the particle) and counts the positions that have no
   key (an addition to a counter, D107). The places by cell of the grid of
   the candidates are a second one. The column and candidate grids are
   `md_alg.grid_plan`s, and the readers of both orders take them through
   `md_alg.bin_span` and `md_alg.bin_item`.
4. **Chunks.** Column $c$, with $n_c$ particles, is cut into
   $\lceil n_c/64\rceil$ chunks of 64 consecutive particles, from the bottom
   in $z$; the last is short. Chunk $t$ (counted over all columns in their
   order) takes the places $64t$ to $64t + 63$.
5. **Groups.** A warp sorts the $n \le 64$ particles of a chunk by $x$ with
   a bitonic network of 64 values (two a lane), ties broken by the number of
   the particle. The first $\lceil n/2\rceil$ form one half and the rest the
   other; each half is sorted by $y$ with a network of 32, and its first
   $\lceil n_h/2\rceil$ form one quarter and the rest the other. Quarter $k$
   takes the places $64t + 16k$ onward, so group $4t + k$ holds it; the
   places it leaves are empty (`order` $= -1$).

```text
column (side >= d_col)       chunk of 64 (height about d_col)     4 groups
   z                          sort by x, halve;                   +----+----+
   ^  +----------+            each half: sort by y, halve         | g2 | g3 |  y
   |  | chunk 2  |            ------------------------------>     +----+----+  ^
   |  | chunk 1  |                                                 | g0 | g1 |  |
   |  | chunk 0  |                                                 +----+----+  +--> x
      +----------+                                                 each about d_col/2 x d_col/2 x d_col
```

The halves and the quarters are uneven only by one particle: a short chunk
of $n$ particles spreads them over its four groups ($\lceil n/2\rceil$ and
$\lfloor n/2\rfloor$, then halved again), so a chunk of fewer than four
particles leaves whole groups empty, and a chunk of 64 gives four groups of
16. A column holds at most 63 empty places, in its last chunk.

```text
order(x~):                                           # Section 2, one build
  d_col <- (64 / rho)^(1/3)
  N_x, N_y <- clamp(floor(L / d_col), 1, 256);  N_z <- max(1, floor(2 N / (N_x N_y)))
  key_i <- (k_y N_x + k_x) N_z + k_z               # counting sort, then by number within a key
  for each column c, chunk t of it (a warp):
    s <- its <= 64 particles;  sort s by (x, number)
    for half h in (first ceil(|s|/2), rest):
      sort h by (y, number)
      quarter 0 <- first ceil(|h|/2), quarter 1 <- rest
      write each quarter k at places 64 t + 16 k + rank
```

**The shape of a group.** A chunk of 64 consecutive particles of a column
of cross section $d_\text{col}^2$ is about $d_\text{col}$ tall, so a group,
a quarter of it in $x$ and $y$, occupies a prism of about
$\tfrac12 d_\text{col} \times \tfrac12 d_\text{col} \times d_\text{col}$:
1:1:2, not a cube. At $\rho = 0.1$ Å$^{-3}$ (water), $d_\text{col} = 8.6$
Å, and the prism of $4.3 \times 4.3 \times 8.6$ Å has a half-diagonal of
$\tfrac{\sqrt6}{4}d_\text{col} = 5.3$ Å. A cube of the same volume would
reduce the volume that sets the length of a list (Section 8.4) by about 3%
at a reach of 9.5 Å.

**Why this order.** The cost of a list grows with the volume of the points
within $R$ of its group's box (Section 8.4), so the boxes must be small for
every group. The order of the cells of the matrix (cells of a third of the
reach, x first) gives boxes 12 Å long, and a Z curve cut into sixteens gives
boxes as long as the cell where it jumps (Section 1 of the README of
`scripts/experiments/neighbor-structures`). Sorting columns by $z$ and
chunks by $x$ and $y$ bounds every box by the column and the chunk, whatever
the positions. Section 11.1 measures the gain: the loop over groups gains
over the matrix only with this order and a cheaper erfc.

**Cost.** The counting sort is $O(N + N_\text{col}N_z)$ work, three scans
of the device among it; the networks are 21 stages for 64 values and 15
for 32, $O(N\log^2 64)$ in all, a warp a chunk.

## 3. Frames and boxes

**The positions of the places.** The build stores, for each place, the
wrapped position $\tilde{\mathbf x}_{\pi(p)}$ in f32, and counts the places
into a second grid (Section 4).

**Boxes.** A half-warp computes the box of each group: the members'
displacements from the first member in the minimum image,
$\mathbf r_u = \mathrm{img}(\tilde{\mathbf x}_{16g+u} -
\tilde{\mathbf x}_{16g})$, with $\mathrm{img}(\mathbf d) = \mathbf d -
\mathbf L\odot\mathrm{round}(\mathbf d\oslash\mathbf L)$, reduced by
shuffles to their least and greatest coordinates; the box is
$\tilde{\mathbf x}_{16g} + [\min_u \mathbf r_u, \max_u \mathbf r_u]$ in
f32, with center $\mathbf y_g$ and half-widths $\mathbf w_g$. An empty
member takes no part. Taken in the minimum image of the first member, a box
spans at most the cell along each axis, $w_{g,a} \le \tfrac12 L_a$, and a
group across the boundary of the cell has a small box.

**Frames** (D95). The *frame* of a group is its first member where it is
wrapped and the others at their minimum-image places relative to it,
$\tilde{\mathbf x}_{16g} + \mathbf r_u$. For each place the build stores
two shifts in whole cells, ten bits an axis from $-512$:

- $\mathbf S_p$ (`shift`): from the particle as it is kept, unwrapped,
  into the frame of its group, so that $\mathbf x_{\pi(p)} + \mathbf
  S_p\odot\mathbf L$ is the member in its frame;
- $\mathbf W_p$ (the fourth lane of the stored position): from its wrapped
  position to its frame, $-1$ to $1$ along each axis.

The loop never takes a minimum image: it adds $\mathbf S_p\odot\mathbf L$
to the stored position in f64 and converts the sum to f32 once (D101), and
each entry carries the cells $\mathbf e$ from its own group's frame to the
list's (Section 4). A frame anchored on the first particle as it is kept,
unwrapped, would let $\mathbf e$ grow with the drift of two groups apart
and overflow its four bits; anchored on the wrapped position, $\mathbf W_p
\in \{-1, 0, 1\}^3$ and $\mathbf e$ stays within $-4$ to $4$. The ten bits
of $\mathbf S_p$ hold a particle up to 511 cells from the cell; the run
stops long before (Section 9).

## 4. The search

A warp builds the list of one group; the lanes take candidates.

**Candidate cells.** The places are counted into a grid of
$N_{\text{cell},a} = \min(256, \max(3, \lfloor L_a/(R/2)\rfloor))$ cells
along each axis, of width $L_a/N_{\text{cell},a}$, at least half the reach
unless the cell is narrower than $\tfrac32 R$; each cell holds its places
sorted by place. With $R' = R + \delta_R$ the widened reach (Section 7.2),
the cells within the reach of the box of group $g$ are, along each axis,

$$
\Big\lfloor \frac{(y_{g,a} - w_{g,a} - R')\,N_{\text{cell},a}}{L_a}\Big\rfloor
\le k_a \le
\Big\lfloor \frac{(y_{g,a} + w_{g,a} + R')\,N_{\text{cell},a}}{L_a}\Big\rfloor,
$$

taken modulo $N_{\text{cell},a}$; a window wider than the grid is the whole
grid, so no cell is visited twice. The warp visits $z$, then $y$, then a
run along $x$ in one piece or two around the edge, in this fixed order.

**The half list.** A candidate at place $q < 16g$ belongs to an earlier
group, whose list holds the pair: it is dropped before any test.

**The box test, then the point tests** (D99). A lane takes a candidate,
its displacement from the center in the image nearest to it,
$\mathbf r_q = \mathrm{img}(\tilde{\mathbf x}_q - \mathbf y_g)$, and its
distance from the box,
$\lVert\max(\lvert\mathbf r_q\rvert - \mathbf w_g, \mathbf 0)\rVert$. A
candidate beyond $R'$ of the box is beyond $R'$ of every member and is
dropped. The others go to a queue of the warp in shared memory, in the
order of the lanes (a ballot and a rank); when the queue holds 32, each lane
takes one and tests it against the 16 members in a loop of its own, with
the members' displacements from the center, $\mathrm{img}(\tilde{\mathbf
x}_{16g+u} - \mathbf y_g)$, in shared memory. Bit $u$ is set when

$$
\lVert \mathbf r_q - \mathrm{img}(\tilde{\mathbf x}_{16g+u} - \mathbf y_g)\rVert^2 \le R'^2,
\qquad 16g+u \text{ not empty},
\qquad \text{and } q > 16g + u \text{ if } q < 16g + 16 .
$$

The last condition makes the pairs within a group appear once, with the
member at the earlier place. Testing where the candidates lay, a third of
the lanes worked (Section 11.2).

**Exclusions** (D100, D105, D106). Before the search the warp gathers the
excluded partners of its 16 members as keys $(\text{place} \ll 4) \mid u$,
each member in a lane of each half-warp, the first half taking its partners
at even positions of its row and the second those at odd ones (two lanes a
member), and sorts them with the bitonic network of the fewest values
that hold them (64, 128, or 256). A candidate outside the range of
the partners' places skips the search; otherwise a binary search finds the
first key at place $q$, and the walk over the keys of that place clears bit
$u$ of each. A group whose partners exceed the 256 keys of a warp's memory
reads the rows of the excluded pairs of the members that the candidate
pairs with instead (D106): exact, slower, and rare (Section 11.2).

**Entries.** A candidate with a bit left becomes the entry $(q, \mathsf b,
\mathbf e)$ with

$$
\mathbf e = \mathbf W_{16g} - \mathbf W_q - \mathbf n_q, \qquad
\mathbf n_q = \mathrm{round}\big((\tilde{\mathbf x}_q - \mathbf y_g)\oslash\mathbf L\big),
$$

the cells from the frame of $q$'s group to that of group $g$, for the image
nearest to the center of the box; $\mathbf e + 4$ goes in four bits an
axis, bits 16 to 27 of the mask.

**Blocks** (D331). The entries of a round are written in the order of the
lanes, after those of the rounds before. A list takes blocks of 64 entries
from one pool as it fills them: in a round, lane 0 takes from the pool's
counter, with one atomic addition, the block that the round begins and the
next one if the round runs into it, at most two, and names them in `units`
and `ordinals`. A block beyond the pool is counted but not written
(Section 8.1). The round is one `md_alg.append` (D407), whose
contract is this: the counter holds the full demand, the list's count
every entry, and nothing at or past the capacity is written.

```text
list(g):                                         # a warp; lane l
  y, w <- box of g;  m_u <- img(x~[16g+u] - y), u < 16       # shared memory
  P <- sorted keys (place << 4 | u) of the excluded partners of the members
  for cell in window(y, w, R') in the order z, y, x:          # Section 4
    for 32 places of the cell at a time, lane l takes q:
      near <- q >= 16 g  and  dist(img(x~[q] - y), box) <= R'
      push the near q onto the queue (ballot order)
      while the queue holds 32: take(32 of them)
  take(the rest)
  counts[g] <- entries written;  sizes[1] <- max(sizes[1], counts[g])

take(queue):                                     # lane l takes one q
  r <- img(x~[q] - y);  mask <- 0
  for u in 0..15:
    if real(16g+u) and |r - m_u|^2 <= R'^2 and not (q < 16g+16 and q <= 16g+u):
      mask |= 1 << u
  clear the bits of the excluded pairs (binary search of P for q)
  if mask != 0:
    e <- W[16g] - W[q] - round((x~[q] - y) / L)
    entry <- (q, mask | (e + 4) << 16)
  rank the entries by ballot; lane 0 takes at most 2 blocks; write
```

The code computes the distance from the box again when it takes the queue,
which this listing shows once.

**Why a fixed order.** A warp visits its cells in a fixed order and writes
its entries in the order of the lanes, so the list of a group, and the
order of its entries, are the same from run to run (I2). The design for
the deterministic mode (Section 6.1) rests on this; the blocks that a list
takes from the pool, and the order of the atomic additions of the loop, do
depend on the threads.

**Trivial acceptance**, taking a candidate near the box for the whole
group without testing its pairs, was measured and dropped (Section 11.1):
the build tests every pair against the reach.

## 5. Images

**One entry for each image within the reach** (D115). An entry takes one
image of its candidate, the one nearest to the center of the box. That is
the image of every pair of the entry within the reach as long as
$w_{g,a} + R' \le \tfrac12 L_a$ along each axis: a member is within
$w_{g,a}$ of the center, and an image within $R'$ of it is within
$w_{g,a} + R'$. A second kernel takes the groups with $2w_{g,a} + 2R' \ge
L_a$ along some axis, the only ones whose candidates can have another image
within the reach, scans their cells again, and continues each list from the
block where the first kernel left it (stored in column 6 of the box) and
its count. For each candidate it queues the other 26 images a cell away or
none along each axis,

$$
\mathbf r_q - \Delta\mathbf n\odot\mathbf L, \qquad \Delta\mathbf n \in
\{-1, 0, 1\}^3 \setminus \{\mathbf 0\},
$$

those within $R'$ of the box, with the place in bits 0 to 25 of the queue
and the code $9(\Delta n_z + 1) + 3(\Delta n_y + 1) + (\Delta n_x + 1)$ in
bits 26 to 30 (places are fewer than $2^{26}$). Each such image becomes an
entry with its own mask and its own shift, $\mathbf e$ less $\Delta\mathbf
n$. With edges longer than $2R'$, about $2R$, a pair has at most one image within the
reach, so the masks of a candidate's entries are disjoint and each pair is
in one list once; in a narrower cell a pair has a bit for each of its
images within the reach, and the loop, which tests the cutoff of each,
finds at most one within it, the edges being at least $2r_c$.

In a wide cell the second kernel is a launch of warps that return. Done in
the list kernel, the same code took 128 registers a thread instead of 64
and the build of Cellulose 30% more time (Section 11.2).

## 5.1 Triclinic cells (D126)

A triclinic cell ([triclinic.md](triclinic.md)) takes the build
`@mdrt_gpu_build_neighbors_groups_triclinic`, which the same emitter writes
with its parameter `triclinic`; only a module with a triclinic cell adds
it. The parameter is a choice of the emitter: each build has the code of
its cell and none of the other's (D404).

| Part | Orthorhombic | Triclinic |
|---|---|---|
| Wrapped positions | Into $[0, L)$ along each axis | Into the brick $[0,a_x)\times[0,b_y)\times[0,c_z)$, along c, then b, then a; the order, the columns, and the grid of the candidates are those of the brick |
| Minimum image | Per axis | One pass along c, b, and a (`@mdrt_gpu_groups_image_triclinic`), which gives the lattice shift $\mathbf n$ with it: the image in the brick, the nearest one within half of the least of $a_x, b_y, c_z$; the images of D115 take the others (Section 5.2) |
| Windows of the search | Per axis, wrapped per axis | A row of z $t_z$ cells beyond the grid is that of $t_z\mathbf c$ away: its window of y moves by $-t_z c_y$; a row of y $t_y$ beyond, by $t_y\mathbf b$: its window of x moves by $-t_z c_x - t_y b_x$, and by whole periods of a back into the grid, so that the run along x goes around the edge once at most. A range of z wider than the grid makes y and x whole, and one of y makes x whole: every cell once |
| Shifts of the frames and of the entries | Cells per axis, times the edges | Lattice vectors, times $H$ (`kernels::emitLatticeShift`): five multiply-adds once per entry |
| Shift of an entry in its mask | $(e + 4)$ in four bits an axis, bits 16 to 27 | $(e + 16)$ in five bits a vector, bits 16 to 30: the brick spans up to 3.5 periods of a in lattice coordinates |
| Images of D115 | The 27 codes a cell away or none along each axis | The 27 codes a lattice vector away: $\Delta n_c \in \{-1, 0, 1\}$, then y rounded again and $\Delta n_b$, then x and $\Delta n_a$; the queue holds the code in bits 26 to 30 and the place below |
| The widening $\delta_R$ | $3\times10^{-6}(L_x + L_y + L_z)$ | $3\times10^{-6}$ of the sum of the diagonal and the magnitudes of the three tilts |

`test/Runtime/neighbors-groups-triclinic-gpu.mlir` checks the lists of an
octahedron, a dodecahedron, a hexagonal cell, and a cell of D115 against
every image of every pair in f64.

## 5.2 How far the lists reach (D242)

**Claim.** The lists hold an entry bit for every image of every pair
within the reach $R$ while $R < \min_a L_a$, the least edge of an
orthorhombic cell or the least of $a_x, b_y, c_z$ of a triclinic one.

**Proof.** Let $\mathbf y$ be the center of the box of a group and
$\mathbf w$ its half-widths. (1) The members of a group are taken in the
minimum image of its first (per axis, or by the pass along $\mathbf c$,
$\mathbf b$, $\mathbf a$, which leaves each component within half of
$L_a$), so a box spans at most the cell along each axis: $w_a \le
\tfrac12 L_a$. (2) An image of a candidate within $R$ of a member of the
group is within $R$ of the box, so its displacement $\mathbf r$ from the
center has $\lvert r_a\rvert \le w_a + R$ along each axis. (3) The list
kernel takes the image nearest to the center, $\lvert r_a\rvert \le
\tfrac12 L_a$ (in a triclinic cell, the image of the pass), and the images
kernel those with $\Delta n_a \in \{-1, 0, 1\}$ cells more along each axis
(in a triclinic cell $\Delta n_c$, then $y$ rounded again and $\Delta n_b$,
then $x$ rounded again and $\Delta n_a$): every image with $\lvert
r_a\rvert \le \tfrac32 L_a$ along each axis, and no other image has that,
the next being at least $\tfrac32 L_a$ away along some axis. (4) The images
kernel runs for a group where $2(w_a + R) \ge L_a$ along some axis;
elsewhere (2) leaves only the nearest image. So the lists are complete
where $w_a + R < \tfrac32 L_a$ for every axis, which (1) gives for $R <
\min_a L_a$. $\blacksquare$

The bound is not sharp for a group whose box is small, but it is the one
that holds for every group whatever the positions (a sparse column makes a
box as wide as the cell), and a reach of the whole cell has no use. The
builds widen the reach by $3\times10^{-6}$ of the sum of the edges for the
rounding of f32, so the builder refuses `pairlist_distance` $> 0.999
\min_a L_a$ with the groups, a commit of a writable borrow does, and under
a barostat the runtime stops a run whose cell comes below
`pairlist_distance`/0.999 (`leastReachEdge` of `mdrtSetBox`), as it does
at twice the cutoff. The dual list is pruned from these lists with their
shifts and adds no image of its own.

Before D242 the orthorhombic images kernel took, along each axis, the
nearest image or the one on the other side of the boundary, 8
combinations: the images with $\lvert r_a\rvert \le L_a$ on the side of
$-\mathrm{sign}(r_a)$ only, complete while $w_a + R \le L_a$, which nothing
asked (#263). The triclinic kernel had the 27 codes, with a bound of half of
the least of $a_x, b_y, c_z$ on the reach, which the proof above replaces
by the whole of it.

`test/Runtime/neighbors-groups-gpu.mlir` and its triclinic twin count, for
each pair, the bits in the lists against the images within the reach in
f64 (up to two cells away), and compare the sum of the squared
displacements in the frames of the groups with that of those images, at
reaches up to 0.975 of the least edge; `group-reach-gpu.test` runs the
groups and the dual list in a cube and in a tilted cell under a barostat
with a reach of 0.77 to 0.89 of the edge against the sum over all images.

## 6. The loop and the dual list

### 6.1 The loop over pairs: each pair once

The loop implements Section 9.3 of [ops.md](ops.md): for each unordered
pair $\{i, j\}$ in the list and within the cutoff,

```text
a_i += k(i, j)
a_j += s · k(i, j)        s = +1 for a symmetric kernel, −1 for an antisymmetric one
S   += k_S(i, j)          weight 1 (the matrix counts each pair twice, weight ½)
```

This is legal only for kernels with an exchange contract
([ops.md](ops.md), Section 4.7): the forces of a potential are
antisymmetric (derived by differentiation), its energy and virial
symmetric. `md_exec.pair_for` carries the contract of each of its
destinations and sums (Section 10); a loop with a destination whose kernel
has none keeps the matrix. By I1 the loop then computes every pair within
the cutoff once: a pair within $r_c$ is within $R$ at the build while the
structure is valid (Section 7.1), so it has a bit, and a pair farther than
$R$ has none.

**The schedule** (`emitGroupPairKernel`, `lib/Conversion/MDExecKernels/`;
`lowerGroupPairFor`, `convert-md-exec-to-gpu`). A warp takes a block of a
group $g$, up to 64 entries in at most two rounds of 32. Lanes $u$ and
$u + 16$ hold member $u$, at place $16g+u$, with its position in its frame
and its fields. In a round, lane $\ell$ loads entry $32\cdot\text{round} +
\ell$: the position of $q$ in its frame plus $\mathbf e\odot\mathbf L$ (or
$\mathbf eH$), and its fields. The entries then turn within each half-warp,
for 16 steps, each lane taking the entry, its fields, and the values it has
received from the lane before it in its half: at step $k$ lane $u$ (of
either half) holds the entry loaded by lane $(u - k) \bmod 16$ of its half.
So each member meets the 16 entries of each half once, the 32 of the round
in its two lanes. After 16 steps every entry is back at its lane, which adds
the values it received to the entry's particle with one atomic addition for
each component. At the end of the block the two half-warps add their
members' values with a shuffle across 16 lanes, and the first half adds
them to the members' particles, once a block.

```text
pair_block(b):                         # a warp; lane l, u = l mod 16, h = l div 16
  g <- units[b];  mine <- position and fields of place 16g+u (in its frame)
  for each round of 32 entries of b:
    (q, mask) <- entry 32 round + l;  other <- position of q in its frame + e L, fields of q
    for v in 0..15: word_v <- ballot(bit v of mask)            # D97
    bits <- rotl((word_u >> 16 h) & 0xffff, 15 - u)            # bit 15 - k at step k
    recv <- 0
    for k in 0..15:
      d <- mine - other;  r2 <- (bit 15 of bits) ? |d|^2 : 4 r_c^2
      v <- kernel(r2, d, mine, other);  v <- (r2 < r_c^2) ? v : 0
      own += v;  recv += s v
      (other, recv) <- from lane (u - 1) mod 16 of half h;  bits <- bits << 1
    if q is a particle: atomic add recv to its destinations
  own += shuffle_xor(own, 16);  if h = 0 and 16g+u is a particle: atomic add own
  sums: weight 2 x (the op's weight of the sum over both orders)
```

The rotation is what lets the mask stay in its lane (D97): after $k$
shifts, bit 15 of lane $u$'s word is bit $(u - k) \bmod 16$ of its
half's ballot for member $u$, the pair of member $u$ with the entry it then
holds. A slot whose bit is clear is given $r^2 = 4r_c^2$, beyond the
cutoff, so that one comparison masks it: the kernel sees a finite distance,
its value is not taken, and the loop has no divergent branch. The loop
tests $r^2 < r_c^2$, strictly; the build tests $\le R'^2$.

**Its cost.** The kernel is bound by the pipe of loads and shuffles
(Nsight Compute, 92% of its peak on Cellulose), so what a pair reads and
turns is what it costs. A step shuffles each component of the entry's
position, of its fields, and of what it has received: 8 shuffles a pair on
the nonbonded kernel measured (D97), with two loads a pair (the radial
table of D94). The coefficients of Lennard–Jones at one entry of a type
table are one load of 8 bytes (D96), and the mask does not turn with the
entry. The kernel is evaluated for every slot, 16 an entry; the share of
them that hold a pair within the cutoff is $\zeta_c$ (Section 8.4).

**Warps and units.** The grid has a warp for each unit of work, as many
warps as particles at most, a warp taking the units $w$, $w + W$, ...; the
sums of a warp go to the scratch of a particle and are reduced as those of
the matrix. The positions and the fields that the kernel reads are gathered
in the order of the places first; an empty place takes those of particle
0, which no bit reads (I3). The buffers of the gather (and the frames of a
pruning check) take the room of the structure's places, not the number of
places that the last build occupied (D312, #471): that number moves by 64
from one build to the next, and the runtime's pool of device memory (not
the pool of blocks of the lists) reuses a block only for a request of the
same size. The room changes only when the structure grows.

**The scatter, measured and refused** (D291). A table of a block for the
values at the particles (`scatter(hash(...))`) in place of the atomic
additions was measured slower at every loop over pairs: 1.18 to 1.32 times
the step of the Amber suite over groups on an RTX 4090, and 2.06 times the
two-ended loops of EAM over a matrix ([device-primitives.md](device-primitives.md),
Section 2.7). `pair-hash=auto` refuses it; `pair-hash=on` takes it.

**Order of the sums.** The atomic additions make the sums depend on the
order of the threads. In the default mode they are additions in the type
of the destination (D84). The deterministic mode does not take groups
yet: the control file refuses `neighbor_structure = "GROUPS"` with
`deterministic`, and the lowering refuses the loop (stage G2). The design
for it is integer additions in fixed point (D70, [LeGrand2013]), whose sum
does not depend on the order: the value of an entry, summed over the 16
steps of a round in registers, would be converted and added, and the value
of a member converted and added for each block; the entries of a group are
in a fixed order (I2), so the sum of a round in registers would be the same
from run to run.

**Destinations.** A loop that adds each pair once adds into its
destination; it cannot overwrite it (the `overwrite` of
`md_exec.pair_for`). Its destination is zeros or the destination of the
term before (`md-exec-accumulate-destinations`), and the lowering clears a
destination that would be overwritten before the loop.

**Fusion.** A loop over groups is not a loop over the particles: it is not
fused with loops over tuples into runs of rows (`convert-md-exec-to-gpu`,
D83). Two loops over pairs fuse as before when both run over groups; a
fused loop whose destinations are not all symmetric or antisymmetric keeps
the matrix.

## 6.2 The dual list (D114)

`pruned_distance` in `[energy]` keeps a dual list: the structure of groups
built with the reach $R$ = `pairlist_distance` (the outer list) and an
inner list pruned from it with the reach $R_\text{in}$ = `pruned_distance`,
which the loops over pairs take. A structure that `md-exec-choose-neighbors`
leaves as a matrix, beside the groups of other loops, keeps one list: the
pass takes the reach and the test of the inner list from its refreshes
(D245; the loops of the derivative in the tunables of a Python model are
such loops).

**Validity.** With $\mathbf m = \mathbf L\oslash\mathbf L^\text{ref}$ and
$\mathbf m^p = \mathbf L\oslash\mathbf L^p$, the inner list holds every
pair within $r_c$ if

$$
2\max_i \lVert \mathbf x_i - \mathbf m\odot\mathbf x^\text{ref}_i\rVert \le
\min(\mathbf m)\,R - r_c
\quad\text{and}\quad
2\max_i \lVert \mathbf x_i - \mathbf m^p\odot\mathbf x^p_i\rVert \le
\min(\mathbf m^p)\,R_\text{in} - r_c .
$$

A pair that the inner list leaves out was left out by the outer list, and
the first condition keeps it beyond $r_c$ (Section 7.1), or the pruning
dropped it at $\mathbf x^p$ for being beyond $R_\text{in}$, and the second
condition, the theorem of Section 7.1 for a build at $\mathbf x^p$ with
$R_\text{in}$, keeps it beyond $r_c$. The proof does not depend on the
layout of the list ([neighbor-tiles.md](neighbor-tiles.md), Section 6;
white paper, Section 4.5).

**The tests.** Two loops over particles, fused with the loop that moves
them (`md-exec-expose-validity`): that of D80 against the configuration and
the cell of the build, and the same against those of the last pruning with
$R_\text{in}$ (`md_exec.reference_positions pruned`,
`md_exec.reference_cell pruned`). Each scales by its own cell, per axis; a
scaling has no shear, since the cell is orthorhombic or, triclinic, scaled
as $H\,\mathrm{diag}(\boldsymbol\mu)$ (I3 of [triclinic.md](triclinic.md)),
where $\mathbf m$ is the ratio of the diagonals. Before the first pruning
the cell of the inner list is not a number, so its test fails.

**The refresh.** Where the outer list is not valid, it is built and the
inner list pruned from it; where only the inner one is not, the inner one
is pruned again. A pruning sets the configuration and the cell of the inner
list only; those of the outer list change at a build only.

**The pruning, step by step** (`emitGroupsPrune`,
`lib/Conversion/MDExecToGPU/MDExecToGPU.cpp`):

1. *Frames and configuration.* A kernel moves the positions of all places
   into the frames of their groups in f32, as the loop takes them (D95,
   D101), and copies the positions, particle by particle in their order, to
   the configuration of the pruning $\mathbf x^p$ (D268); it clears the
   counts of the active units.
2. *Prune a group.* A warp for each group holds the 16 members, reads the
   outer entries of the group in their order, 32 at a time, and computes
   the inner bits of each,
   $\mathsf b^\text{in}_u = \mathsf b^\text{out}_u \wedge
   [\,r_{uq}^2 \le (R_\text{in} + \delta_R)^2\,]$, with $r_{uq}$ the
   distance of member $u$ and the entry in the frame of the group at the
   current positions and $\delta_R$ the widening of the build. Always from the
   outer mask (I5), so a pair dropped once comes back when it is near
   again; the outer entries and masks stay as the build left them.
3. *Compact.* The entries with a bit left are written to the front of the
   group's blocks, in their order, across the boundaries of the blocks: a
   ballot gives each lane its rank among the kept entries of its round,
   $\mathrm{popc}(\mathrm{ballot}(\text{keep}) \wedge (2^\ell - 1))$, and
   the entry goes to $k + \text{rank}$ with $k$ the entries kept so far.
   The block of each group's list by its number in the list (`blockOf`, a
   row of the blocks of the longest list for each group) is made at each
   build, not at each pruning (D268). Each block records how many entries
   it holds. The inner list thus fills its blocks as a list built with
   $R_\text{in}$ would.
4. *Active units.* The pruning lists the blocks that hold entries of the
   inner list, the first $\lceil k_g/64\rceil$ of each group $g$ that keeps
   $k_g$ entries, in the order of the blocks, with their number on the
   device (D268). It counts the active blocks in runs of 32 and of 1024
   blocks (atomic additions of integers, whose sums do not depend on the
   order), and a kernel with a warp for each run of 32 places each active
   block after those of the runs of 1024 before its own, of the runs of 32
   before its own in that run of 1024, and of the blocks before it in its
   run (a ballot). The loop over pairs takes these units (I6): its grid is
   that of all units, as the host knows no more, and a warp past their
   number does nothing. Each pruning clears the number first: a build that
   places no particle (D107; a run that has failed goes on to the end of its
   part) leaves no block, and no kernel counts them.

```text
prune(g):                                        # a warp
  k <- 0
  for each 32 outer entries of g, in order (lane l takes one):
    in <- out & { u : |x_u - (x_q + e L)|^2 <= (R_in + delta_R)^2 }
    keep <- in != 0;  rank <- popc(ballot(keep) & (2^l - 1))
    if keep: write (q, in | e) at entry k + rank of g's blocks
    k <- k + popc(ballot(keep))
  record the entries of each block;  k_g <- k
active units: the first ceil(k_g / 64) blocks of each g, in the order of the blocks
```

**The order.** The pruning keeps the order of the outer list and adds no
order of its own. The order of the outer list, the blocks that the build
takes with an atomic addition, and the atomic additions of the loop are as
without a dual list: the sums still depend on the order of the threads, and
the deterministic mode does not take groups. The loop takes the active
units in the order of the blocks, as it took all units before; a warp takes
other units than before, and its sums are added in another order.

**The frames.** The positions in the frames of the groups, in f32, are kept
with the structure. A loop over pairs that follows the refresh at the same
positions and in the same cell, with no op between that may write the
positions (by the memory effects of the ops and the aliases of the buffers,
`findSharedFrames`), and that computes in f32, takes them as they are when
the refresh pruned: they are the loop's own, to the bit (the same shift in
f64, added in f64, rounded to f32). Otherwise, and in double precision, the
loop moves the positions itself (D268).

**The pruning in the loop** (D284, #362). Where a refresh prunes, the first
loop over pairs that takes its result prunes for it, if it follows the
refresh in the same block at the same positions and in the same cell with
no op between that may write the positions (the conditions of the frames)
and no op between takes the refresh's result (`findPrunedInLoop`). The
refresh then runs only the kernel of step 1, and the loop runs one kernel
with a warp for each group: the warp prunes its group with the code of
step 2 and 3 (`emitPruneGroup`), waits for the stores of its other lanes
(`bar.warp.sync`), and computes the pairs of the entries it kept, read back
from the inner list as one stream of the group, 32 at a time, with the
round of Section 6.1 (`GroupStream`); the kernel of the active units
follows. The inner list is the pruning's to the bit. On the step itself the
warp takes the pairs within $r_c$ of the entries kept: since $r_c <
R_\text{in}$, an outer entry with a pair within $r_c$ at $\mathbf x^p$,
the configuration of the step, keeps the bit of that pair, so these are the
pairs of the loop over the fresh inner list; only the order of the
additions differs. On the steps that do not prune the loop takes the active
units as before.

**When the loop prunes.** The host decides at each pruning from the groups
of the last build (`takesFusedPruning`, one test for the refresh and the
loop): from 4096 groups, and from 20,480 where the kernels are compiled for
`sm_12x`; `MDIR_FUSED_PRUNE=off` never, `=always` at any number. The count
is that of the code, the places of the last build over 16, empty places
included, which is at least $N/16$ and at most $(N + 63N_\text{col})/16$
(Section 8.1); the tables of Section 11.7 give $N/16$. The reason: a warp a
group is slower where the groups do not fill the device, since the loop
otherwise has a warp for each active unit, several a group; the kernel
also holds 94 to 96 registers a thread against 72 for each of the two it
replaces (Section 11.7).

**The check.** With `MDIR_CHECK_PRUNING` set, each refresh counts (the loop
that prunes for it, after the loop), at the positions of its step, the
pairs within the cutoff in the outer list and in the inner list through the
active units, in f32 on the frames of that step; the inner list is a part
of the outer one (I5), so the two differ only where it misses a pair, and
the runtime stops the run. At its end the run reports the steps it checked
(`test/Driver/prune-check-gpu.test`; the option `check-pruning` of
`convert-md-exec-to-gpu`).

**The log.** `the inner lists were pruned N times, every k steps on
average`, the prunings that follow a build included, and `, M of them in
the loop over pairs` where the loop pruned.

## 6.3 The default: `AUTO` (D261)

`neighbor_structure = "AUTO"`, the default, takes the groups with a dual
list where they are faster and the matrix elsewhere. One function,
`resolveNeighborStructure`, resolves it once the system is read, for
`mdir run`, `mdir check`, and the Python model. It takes the groups when
every one of the following holds, in this order, and the matrix otherwise,
with the first reason that fails:

- the target is a GPU, outside the deterministic mode;
- no model in stages, which runs over the slots of the pairs of images
  that a matrix keeps (D294);
- no loop lacks an exchange contract: no implicit solvent, no derivative
  in the tunables;
- no `rebuild_interval`;
- at least 23,000 particles, virtual sites counted;
- the reach of the groups at most 0.999 of the least edge of the cell.

The groups take `pairlist_distance` = cutoff + 3 Å and `pruned_distance` =
cutoff + 0.6 Å × max(1, Δt / 2 fs) where those keys are not given; an
inner reach that is not below the outer one leaves a single list. The
matrix keeps cutoff + 1.5 Å. These are the defaults that `"AUTO"` begins
at; a long run measures candidates around them
([neighbors.md](neighbors.md#6-the-choice-of-the-skins), D281). Why
23,000 particles and these reaches: Section 11.8.

## 7. Validity and precision

### 7.1 When to build again (D80)

The structure is valid while

$$
2 \max_i \lVert \mathbf x_i - \mathbf m \odot \mathbf x^\text{ref}_i \rVert
\le \min(\mathbf m) R - r_c, \qquad \mathbf m = \mathbf L\oslash\mathbf L^\text{ref}.
$$

**Theorem.** Under this condition every pair within $r_c$ at $\mathbf x$ in
the cell $\mathbf L$ is in the lists.

*Proof.* Let $\{i, j\}$ be a pair that the lists leave out, so that every
image of it was farther than $R$ at the build (I1, Section 5.2). Scaling
each axis by $m_a$ maps the lattice of images of the old cell onto that of
the new one and stretches every displacement by at least $\min(\mathbf
m)$, so every image of the scaled references is more than $\min(\mathbf
m)R$ long. Each particle is within $d = \max_i\lVert\mathbf x_i - \mathbf
m\odot\mathbf x^\text{ref}_i\rVert$ of its scaled reference, so by the
triangle inequality every image of $\mathbf x_i - \mathbf x_j$ is longer
than $\min(\mathbf m)R - 2d \ge r_c$. $\blacksquare$

In a cell that has not changed this is the half-skin test $d \le s/2$; the
same holds in a triclinic cell, scaled only as $H\,\mathrm{diag}(\boldsymbol
\mu)$, with $\mathbf m$ the ratio of the diagonals (white paper, Section
4.1). The list is a superset of the pairs within the cutoff, and the cutoff
test of the loop removes the others, as for the matrix ([ops.md](ops.md),
Section 9.2). The test does not depend on the layout: it is the matrix's,
unchanged.

### 7.2 Tests in f32 and the widened reach

The build tests distances in f32 against $R' = R + \delta_R$, with
$\delta_R = 3\times10^{-6}(L_x + L_y + L_z)$ (in a triclinic cell, the sum
of the diagonal and the magnitudes of the three tilts), as the build of the
matrix does. With $\mathsf u = 2^{-24}$ the unit roundoff of f32, each
component of the difference in the point test of Section 4 carries:

| Source | Bound on the error of one component |
|---|---|
| The wrapped positions of the member and the candidate, rounded to f32 | $2\mathsf uL_a$ |
| $\tilde x - y_a$ for each of the two, with $\lvert\tilde x - y_a\rvert < \tfrac32 L_a$ (the center may lie half a cell outside) | $3\mathsf uL_a$ |
| The minimum image of each, whose result is at most about $\tfrac12 L_a$ | $\mathsf uL_a$ |
| The edge in that image: the product of $\mathrm{round}(\cdot)$, at most 1 in magnitude, and the edge rounded to f32 is exact, but that edge is within $\mathsf uL_a$ of $L_a$ | $2\mathsf uL_a$ |
| The shift $\Delta n_a$ times the edge in f32 that the images kernel subtracts (Section 5) | $\mathsf uL_a$ |
| The difference of the two displacements | $\mathsf u\lvert d_a\rvert$ |

The center $y_a$ itself cancels: both displacements are taken from the
same rounded value. So the computed difference vector is within
$9\mathsf u\lVert\mathbf L\rVert + \mathsf uR$ of the exact one between
the unrounded positions, and the sum of the squares and the comparison
with the rounded $R'^2$ add about $3\mathsf uR$ more on the distance. Near
the reach, with $R < \min_a L_a \le \tfrac13\sum_a L_a$, the distance is
computed within

$$
9\mathsf u\,\lVert\mathbf L\rVert + 4\mathsf uR \le 10.4\,\mathsf u
\sum_a L_a \approx 6.2\times10^{-7}\sum_a L_a
$$

of the exact one, about a fifth of $\delta_R$. The box test has errors of
the same order and the same margin. The image that is tested,
$\mathrm{round}(d_a/L_a)$ in f32, can differ from the nearest one only
where $\lvert d_a\rvert$ is within a few $\mathsf uL_a$ of $\tfrac12 L_a$;
for a pair within $R$ of a member that needs $w_{g,a} + R \ge \tfrac12 L_a
- 2\mathsf uL_a$, so $2w_{g,a} + 2R' > L_a$, and the images kernel takes
the other image (Section 5). So a pair within $R$ in f64 is always
kept, and a pair kept may be a little more than $\delta_R$ beyond $R$,
which the cutoff test of the loop removes (I1);
`test/Runtime/neighbors-groups-gpu.mlir` checks this against every pair in
f64. The pruning widens $R_\text{in}$ by the same $\delta_R$, for the same
reason. A run in a cell of 100 Å edges has $\delta_R = 0.9$ mÅ.

### 7.3 Positions in the loop

The loop reads positions in their frames, $\mathbf x_{\pi(p)} + \mathbf
S_p\odot\mathbf L$ summed in f64 and rounded to f32 once (D101): a position
in its frame is then as exact as one in the cell, however far the particle
has gone, and a displacement is one subtraction in the type of the kernel.
The template test places particles three cells out of the cell on either
side to check it.

## 8. Sizes, memory, and cost

### 8.1 Buffers and their growth

No size is fixed. The runtime holds the buffers (`mdrtGroupsCreate`,
`mdrtGroupsBuffer`): places for $N + \lfloor N/4\rfloor + 64$ at first, and
one block for each group. The places that a build takes are
$64\sum_c\lceil n_c/64\rceil \le N + 63N_\text{col}$: at most 63 empty
places a column. A build counts the places and the blocks it takes even
where they do not fit, and writes only those that fit. A build that finds a
buffer too small makes it a quarter larger than it needs (`mdrtGroupsGrow`)
and is made again, so no pair is ever left out; the run stops only if the
device has no memory left.

**The pool** (D331). The first build of a structure therefore sizes its
pool: $B_0 + \lfloor B_0/4\rfloor$ blocks for the $B_0$ that the
configuration at the start takes. The pool holds the sum over all the
groups, which follows the density and varies little from one build to the
next (white paper, Section 4.4); the first build counts it exactly, where
an estimate from $\bar n = \tfrac43\pi\rho R^3$ would need a factor that
depends on the shape of the groups (Section 8.4). A build that needs more
than the first build left, after a compression by the barostat for
example, grows the pool the same way; `mdir run` then logs how many times
the lists grew after the first build of their structure. The buffers that
a growth replaces go back to the driver, not to the runtime's pool of
device memory, since no later request takes their size (D312).

**Limits.** The queue of the images kernel holds a place in 26 bits, so the
places are fewer than $2^{26}$ (67 million); the place shift holds 511
cells; the columns and the cells of the grid are at most 256 a side.

**Lifetime.** A structure's buffers live until the program ends: the end of
`mdir run`, or the close of a Python simulation's activation (D215). The
evaluation at the start builds a structure that the steps do not continue
(D218). Since D330 (#471) the steps' structure takes the start's storage on
a device, with a reset, rather than storage of its own: the start's
structure is no longer used when the steps begin, and both have the same
shape. Their first build, at step 1, finds the lists with the room that the
start's first build sized (D331), so they grow only if they need more.
Section 11.5 measures both.

### 8.2 Memory

An entry takes 8 bytes (its place and its mask), a block 512; the pool
dominates. Per place the structure keeps `order` and `shift` (4 bytes
each), per particle `placeOf`, per group `counts`, and per block `units`
and `ordinals`; the boxes, the wrapped positions, the grid, and the keys
are scratch of the build. A dual list adds the inner entries, masks, the
counts of the blocks, the active units, and `blockOf`.

### 8.3 The cost of a build

A build is 12 kernels of the template and 7 calls of device primitives
(three scans, two scatters, two sorts of the bins;
[device-primitives.md](device-primitives.md)), and it ends by copying
`sizes` to the host and waiting for it: the fixed cost $L$ of the skin
model of [neighbors.md](neighbors.md), Section 6. The work is $O(N)$ for
the sorts and the boxes, and for each group:

- the places of its window, about $\rho\prod_a(2w_{g,a} + 2R + L_a/N_{\text{cell},a})$,
  each read once and half of them ($q \ge 16g$) tested against the box;
- 16 point tests for each candidate within the reach of the box, about
  $\tfrac12\rho V_R(\mathbf w_g)$ of them (Section 8.4).

On JAC ($R$ = 11 Å, $\rho \approx 0.098$ Å$^{-3}$, $d_\text{col} \approx
8.7$ Å, 11 cells of 5.66 Å a side), a group of the prism of Section 2 reads
about 3,600 places and tests about 1,800 against its box, of which about
700 go on to the 16 point tests: the box test saves more than half
of the point tests. These are estimates for a uniform density, not
measurements.

### 8.4 The length of a list and the slots of the loop

The list of group $g$ holds a candidate when some member is within $R$ of
it, that is the candidates in $\Omega_g$, the points within $R$ of some
member, at places after the group's. In a uniform fluid half of them are
later, so $E_g \approx \tfrac12\rho\lvert\Omega_g\rvert$. The pairs within
$R$ are $\tfrac12 N\bar n$, and the loop evaluates 16 slots an entry, so the
share of slots that hold a pair within $R$ is

$$
\zeta_R = \frac{\tfrac12 N\bar n}{16\sum_g E_g} \approx
\frac{\tfrac43\pi R^3}{\lvert\Omega_g\rvert} \qquad (N = 16 \times \text{groups}),
$$

and that within the cutoff $\zeta_c \approx \zeta_R(r_c/R)^3$. $\Omega_g$
lies within the points within $R$ of the box, whose volume for half-widths
$\mathbf w$ is that of the box, its six faces moved out by $R$, its twelve
edges rounded by quarter cylinders, and its eight corners by eighths of a
ball:

$$
V_R(\mathbf w) = 8w_xw_yw_z + 8(w_xw_y + w_yw_z + w_zw_x)R +
2\pi(w_x + w_y + w_z)R^2 + \tfrac43\pi R^3 ,
$$

so $\zeta_R \gtrsim \tfrac43\pi R^3/V_R(\mathbf w_g)$. For the prism of
Section 2, $\mathbf w = (\tfrac14, \tfrac14, \tfrac12)d_\text{col}$, this
bound is 0.34 at STMV ($R$ = 9.5 Å) and 0.39 at JAC ($R$ = 11 Å). The
measured pools bound $\sum_g E_g$ by 64 times the blocks, and so give
$\zeta_R \ge N\bar n/(2048B)$: at least 0.42 at STMV and 0.65 at JAC
(Section 11.5), 1.2 and 1.7 times the bound: the union of 16 balls is
well inside the box widened by $R$, the more so the longer the reach. The
bound still shows which terms set the length: at these reaches the quarter cylinders, which
grow with the sum of the edges, take about half of $V_R$. The shares of
slots within the cutoff measured on Cellulose, 23.8% in the order of cells
and 29.7% in the compact order (Section 11.1), are this $\zeta_c$.

The ratio of the mean list to $\bar n$ follows: $\bar E/\bar n \approx
\tfrac12\lvert\Omega_g\rvert/\tfrac43\pi R^3 = 1/(2\zeta_R)$, at most 1.2
at STMV and 0.77 at JAC by the blocks (Section 11.5), which is why the pool
is counted rather than estimated.

## 9. Robustness

- **Positions that are not numbers, or far outside the cell** (D107,
  D176). Such a position gets no key and no place, and is counted in the
  fifth size; the build completes, and the run stops with a message after
  it. A finite position whose wrap does not land in the cell (beyond about
  $10^{16}$ cells, where the wrap errs by more than a cell) is counted the
  same way, rather than sent outside the tables of the build.
- **Clamps.** The columns, the bins, and the cells of the grid are computed
  with `maxnumf` and `minnumf`, which take the other operand of one that is
  not a number, and clamped (columns 1 to 256, grid cells 3 to 256 a side),
  and a key is clamped to its table, so that every count is defined and fits
  memory whatever the cell (#168, D225). A wider cell costs only time.
- **Too many excluded partners** (D106). A group whose partners exceed the
  256 keys of a warp reads the rows of the excluded pairs instead, and is
  counted in the fourth size.
- **Room.** A buffer that is too small is counted, never overrun, and the
  build is made again (Section 8.1).

## 10. In the IR

- `md_exec.NeighborKind` has `groups`: `md_exec.empty_neighbors
  kind(groups)`, and the refresh builds that kind.
- `md_exec.pair_for` has `exchange`, one of `symmetric`, `antisymmetric`,
  `none` for each destination and each sum, from the `exchange` of the
  `md.gather_relation` or `md.sum_relation` it came from; the fusion of
  loops keeps them.
- The policy `(unique, atomic)` of `md_exec.pair_for`, which M0 declared
  and rejected, is legal over a structure of kind `groups`, with every
  destination symmetric or antisymmetric and every sum symmetric. The
  attribute `weights` of the op stays that of the sum over both orders of
  each pair (½ for a sum of pairs); the lowering doubles it, so that each
  pair counts with the weight 1 (Section 6.1).
- A pass, `md-exec-choose-neighbors` (option `kind=groups`), chooses the
  kind of each structure and the policy of its loops on a device (the kind
  stays `matrix` on the CPU).
- The type of a structure states what its build guarantees
  (D[neighbors-contract]): `!mdrt.neighbors<@atoms, unique, minimum,
  removed, stable(group, search)>` for groups and their dual list. That
  is each pair once (I1), in the frame of its group (I4), with no bit for
  an excluded pair, and the list of each group in the order of its search
  (I2, I5). `!mdrt.neighbors<@atoms>`, the matrix, is `directed, minimum,
  masked, stable(owner, search)`. `md-exec-choose-neighbors` gives the
  type to every value of a structure that it turns into groups. A
  structure whose values reach an op whose types the pass does not
  follow, such as an `scf.if`, keeps the matrix. Each loop over pairs
  states what it requires: its traversal and images, `grouped_by(owner)`
  under `owner_only`, and the order of the search for a `fixed_order`
  sum. The verifier refuses a loop whose requirement the type does not
  meet. The kind, the width, and the order of the places stay attributes
  of the build: they are cost choices.

### 10.1 Coverage across IR boundaries (D357, #572)

The neighbor structure's type identifies its particle set, while coverage
for a pair loop depends on the required cutoff, exact centers value, and
image convention. The coverage query follows SSA provenance through a
recognized `scf.for` loop only when its initial value and every yielded
value establish the required fact, including the zero-trip path. A private
function argument has the fact only when every known call site establishes
it with the corresponding actual arguments. Public, recursive, indirect,
or unrecognized boundaries leave the obligation unresolved and are
diagnosed. No boolean attribute substitutes for evidence. Current-state
freshness, that the structure is valid for the positions and cell that the
loop reads, is a separate obligation: a refresh tests it at run time (D80,
Section 7.1), and no open issue tracks a proof of it in the IR.

[D357](decisions/D357.md) compares this query with a contract in the
neighbor type. Provenance was chosen for coverage: the cutoff and the
exact centers are SSA facts and values, which a type cannot hold, and
unresolved required facts are errors. The layout contract of
D[neighbors-contract] is in the type instead: it is a closed lattice that
the build fixes, with no SSA subjects. The type also carries the images,
so the verifier refuses a loop whose images are not its structure's
before the query runs. Identity forwarding
of center subjects is checked, while body expressions, other loop forms,
function results, and CFG block arguments remain unrecognized boundaries
where the query has no proof. Private functions need at least one known
call site, and every use must be a direct call. The existing strict
distributed halo-source rule is preserved. No control-file key or neighbor
strategy changes.

## 11. Measurements

### 11.1 Why groups, and why the compact order

Measured on Cellulose (408,609 atoms, RTX 3090,
`scripts/experiments/neighbor-structures`, README):

| | Neighbor matrix | Groups of 16 |
|---|---|---|
| Entries of the list at a reach of 9 Å | 128.7 million, one per particle and neighbor | 8.8 million, one per group and neighbor |
| The loop over pairs, reach 9 Å | 1245 to 1267 µs, bound by the loads of its entries | 1013 to 1034 µs, bound by its arithmetic |
| A build, reach 10 Å | 8.0 ms in MDIR: 5.0 ms for the tests of the candidates, 1.6 for writing the matrix | |

A matrix writes and reads an entry for each particle and neighbor; each
pair is computed twice, once for each particle. Groups write an entry for
each group and neighbor, a sixteenth as many, and compute each pair once.

**What the gain rests on** (`groups.cu`, Cellulose, reach 9 Å, GPU 1,
2026-10-01; groups of 16 against the matrix in the same places):

| Order of the places | erfc | Slots within the cutoff | Groups | Matrix |
|---|---|---|---|---|
| Cells of a third of the reach, x first (the order of a build of the matrix) | `erfcf` | 23.8 % | 1751 µs | 1284 µs |
| The same | shares the exponential | 23.8 % | 1410 to 1416 µs | 1270 to 1305 µs |
| Compact: columns sorted by z, cut into chunks of 64, each halved by x and each half by y into groups of 16 (Section 2) | `erfcf` | 29.7 % | 1614 µs | 1236 µs |
| The same | shares the exponential | 29.7 % | 1162 to 1209 µs (996 to 1034 in `supercluster.cu`, not yet explained) | 1196 to 1331 µs |

So the loop over groups gains only with both a compact order of the places
and a cheaper erfc: with the exact `erfcf` it is slower than the matrix,
and with the order of the cells of the matrix its boxes are 12 Å long and
it is slower as well. Trivial acceptance within 0.75 R of the box (the
build as first written) made the loop 2.5 times slower in the order of
cells and changed little in the compact order: the build tests every pair
against the reach instead. The compact order makes the matrix faster too
(5 to 16 %). The gain of the build, a sixteenth of the entries and no test
of candidates far from a box, was measured in stage G1 (Section 11.4).

### 11.2 The build

Measured on the prototype (`build.cu`, reach 9 Å, GPU 1, 2026-10-01): the
lists of Cellulose in 1.48 ms and of JAC in 116 µs, with the order about
0.1 ms more, against 8.0 ms for a build of the matrix of 10 Å on Cellulose.
What it took: a lane a candidate, testing it against the 16 particles of
the group (taking the survivors two at a time took 53 ms); the partners of
the excluded pairs of a group sorted, and a binary search for each
candidate (a linear scan cost 1 ms of 2.4; a group with more partners than
the memory of its warp holds, 256, reads the rows of the excluded pairs
instead, D106); the candidate relative to the center of the box, once, and
the particles of the group too (1.76 to 1.48 ms). Testing the candidates
where they lay rather than through the queue of D99, a third of the lanes
worked. A group whose partners do not fit the 256 places of a warp's memory
is rare: Cellulose, the only system of the suite where it happens, has 1
such group in 255 builds under NVE and 7 in 253 under NPT.

The images in the list kernel (D115): the same code there took 128
registers a thread instead of 64, and a build of Cellulose 2259 µs instead
of 1735 in the template of that time, 30% more.

The three times of a build of Cellulose in this document come from
different builds and are not comparable: 1.48 ms is the prototype at 9 Å;
1735 µs the template at the time of D115; about 0.35 ms over its kernels
the template at 10 Å in the measurement of Section 11.4 (2026-10-01). No
later measurement of the build alone is recorded here.

### 11.3 The loop

With a warp for a whole group, JAC (1,568 groups) had too few warps: 125
µs against 75 for the matrix at 9 Å; with units of 64 entries, 69 µs, and
Cellulose 1109 µs against 1200 (`groups.cu`, 2026-10-01).

### 11.4 Measured in MDIR

Mixed precision, reach 10 Å (the benchmark settings), RTX 3090 at 300 W,
one run each, `nsys` (2026-10-01). The loop over pairs of a step (forces
only) and all the work of the device over the run:

| | Matrix | Groups | Matrix, D90 | Groups, D90 |
|---|---|---|---|---|
| Loop over pairs, Cellulose | 2134 µs | 2097 µs | 1856 µs | 1687 µs |
| Loop over pairs, JAC | 113 µs | 113 µs | 102 µs | 92 µs |
| The device over 500 steps of Cellulose | 2471 ms | 2081 ms | 2325 ms | 1876 ms |
| A build of Cellulose | about 1.0 ms in the kernel of the search, 10 builds per 100 steps | about 0.35 ms over its kernels | | |

With the exact `erfcf` the loops are even, as Section 11.1 found; with the
approximation of D90 the groups gain 9 % on Cellulose and 10 % on JAC. The
gain of the whole run comes mostly from the build. In double precision the
run of `test/Driver/groups-gpu.test` gives the energies of the matrix to
every digit of the log over 100 steps; JAC under NPT (400 steps) builds as
often, and its conserved energy drifts as much (−7.5 and −7.7 kcal/mol).

### 11.5 The pool and the lifetime

A fixed list of twice the width of a row stopped a run of Cellulose under
NPT when one group needed 2521 entries of 1248, where the longest list is
usually 740 to 900. Until D331 the pool started with as many entries a
group as a row of the matrix and one block more, from the particle with the
most neighbors with half as many again: 12 blocks a group at STMV, 1.84
times what the lists took; for Cellulose that pool was 280,929 blocks, 144
MB for the entries and the masks, against 510 MB for the fixed lists.
Cellulose takes at most 153,517 of those blocks (the reach of that run is
not recorded with it; at 9 Å a build takes 152,619, white paper, Section
4.5). A pool sized by D331 from 153,517 blocks would hold 191,896, 98 MB
(computed, not measured). A
run of `test/Driver/groups-gpu.test` that starts with almost no room grows
its buffers and gives the energies of the matrix to every digit.

Sized at the first build, STMV takes at most 433,376 of 541,603 blocks
(800,328 before) and JAC 9,664 of 12,046 (17,676 before), with no growth
after the first build. Sixty-four entries a block over $N/16$ bound the
mean list from above: at most 416 entries a group (6.5 blocks) against
$\bar n \approx 346$ at STMV ($R$ = 9.5 Å), and at most 420 against 545 at
JAC ($R$ = 11 Å). The same test compresses an argon and krypton mixture
from 18,991 to about 10,260 Å$^3$ under 5000 atm: its lists grow once after
the first build, and the energies are again those of the matrix to every
digit.

Each of the two structures of a run, before D330, held about 466 MiB at
STMV (1.07 million atoms): 413 MiB of lists and buffers of the runtime,
and 53 MiB of `placeOf` and the two reference configurations. Buffers of
the gather sized by the places that the last build occupied, before D312,
left a block in the runtime's pool for each new number, 783 MiB at STMV
after 1,000 steps.

### 11.6 The dual list

Measured (RTX 3090 at 300 W, mixed precision, one run each, ns/day):

| System | One list, 9 Å | Dual, outer / inner |
|---|---|---|
| JAC | 734 | 779 (11 / 8.6 Å) |
| JAC, 4 fs | 1352 | 1467 (11 / 9.0 Å) |
| FactorIX | 273 | 286 (11 / 8.6 Å) |
| Cellulose | 59.7 | 62.4 (12 / 8.6 Å) |
| STMV, 4 fs | 37.8 | 39.9 (11 / 9.0 Å) |

On Cellulose (11 / 9 Å) a pruning takes 347 µs in its main kernel (72
registers a thread) and 35 µs to gather the positions and index the
blocks, every 3.5 steps; the loop over the inner list takes 1060 µs,
against 1041 for a list built with 9 Å. The first version pruned each block
of the outer list into the same block: the inner list kept the blocks of
the outer one, partly filled, and the loop took 1320 µs; a pruning read the
positions in f64 and took 933 µs.

The block index made at each build rather than at each pruning (D268): on
the 103k TIP3P box, 7.6 µs every 2.4 steps become as much every 13.6.
After a pruning about 40% of the units of the outer list hold no entry of
the inner list. In the order of the groups instead of the order of the
blocks, the warps of the blocks of one group ran together, and the loop
took 576 µs a step on the 103k TIP3P box against 526 for all units in the
order of the blocks. The copy of the configuration in the kernel of the
frames: 13.0 µs on the 103k TIP3P box against 7.3 and 8.7 for the two
apart; written by place, where each particle's position is read anyway, the
scattered writes took 20.9 µs, and in the main kernel 17 µs more.

The active units, the index of the blocks made at the build, the copy of
the configuration in the kernel of the frames, and the frames shared with
the loop (D268, #357), against main at the same commit, RTX 4090, mixed
precision, the settings of
[scripts/benchmarks/rolling-prune](../scripts/benchmarks/rolling-prune)
(`steps.py`: frozen binaries, 6 rounds alternated in one job a system, the
first dropped; mean ± standard deviation of 5), µs a step:

| System | main | D268 | Difference |
|---|---|---|---|
| TIP3P, 103,292 waters (12 / 9.6 Å) | 1130.4 ± 1.5 | 1118.2 ± 1.9 | −12.2 |
| OPC, 103,292 waters (12 / 9.6 Å) | 1633.0 ± 2.1 | 1612.8 ± 2.2 | −20.2 |
| JAC (11 / 8.6 Å) | 137.0 ± 0.0 | 135.4 ± 0.5 | −1.6 |
| Ubiquitin in OPC (12 / 9.6 Å) | 182.2 ± 0.4 | 180.4 ± 0.5 | −1.8 |
| STMV, 4 fs (11 / 9.2 Å) | 4784.2 ± 8.9 | 4737.0 ± 12.6 | −47.2 |

On TIP3P (nsys, µs a step) the loops over pairs fall from 543.8 to 529.7
and the copies on the device from 4.3 to 1.0; a pruning takes 222 µs
against 224 (the frames and the configuration 13.0, the main kernel 206.0,
the active units 3.0, against 7.6 for the index of the blocks, 7.3 for the
frames, 200.6, and a copy of 8.7). The proposal expected about 25 µs a step
on TIP3P; the copy fused with the frames saves 3 µs of its 8.7, and the
counts of the active units cost about as much as the index of the blocks
saved. Counting the runs of 1024 blocks in a kernel of their own, rather
than with atomic additions in the main kernel, changed nothing (206.0 µs).
The forces at the start differ from main's no more than two runs of main
differ from each other (`forces.py`; on TIP3P at most 9.8 × 10⁻⁴ kJ/mol/nm
against 1.1 × 10⁻³, in double on JAC 1.4 × 10⁻¹² against 1.4 × 10⁻¹²).

### 11.7 The pruning in the loop

The pruning in the loop (D284, #362), against the pruning of the refresh in
the same binary (`MDIR_FUSED_PRUNE=always` and `off`), mixed precision, the
inputs of `scripts/benchmarks/rolling-prune` with Cellulose NVE of the
suite (11 / 8.6 Å) and TIP3P boxes of four sizes (12 / 9.6 Å),
[scripts/benchmarks/fused-prune](../scripts/benchmarks/fused-prune)
(`steps.py`: 4 rounds alternated in one job, the first dropped; mean of 3),
µs a step. The column of the groups is the sites over 16; the threshold
counts the places of a build over 16, empty places included, which is
larger by up to $63N_\text{col}/16$ (Section 6.2). For JAC, with 49 columns
of about 480 particles, that count, a multiple of 4 since the places are
a multiple of 64, lies between 1,476 and 1,664; the loop prototype of
Section 11.3 had 1,568.

| System | Groups (sites / 16) | RTX 4090: refresh | loop | RTX PRO 6000: refresh | loop |
|---|---|---|---|---|---|
| JAC | 1,472 | 136.0 | 137.7 | 116.3 | 124.0 |
| Ubiquitin in OPC | 1,626 | 188.7 | 193.7 | | |
| TIP3P, 27,234 sites | 1,702 | 155.0 | 157.0 | | |
| TIP3P, 52,686 sites | 3,292 | 233.7 | 233.7 | 173.0 | 178.7 |
| TIP3P, 103,158 sites | 6,447 | 405.0 | 388.3 | 291.3 | 294.3 |
| TIP3P, 203,490 sites | 12,718 | 741.0 | 707.0 | | |
| TIP3P, 309,876 sites | 19,367 | 1118.3 | 1050.3 | 896.3 | 898.3 |
| Cellulose | 25,538 | 1465.3 | 1375.7 | 1094.7 | 1093.0 |
| OPC, 413,168 sites | 25,823 | 1630.7 | 1528.3 | | |
| STMV, 4 fs | 66,693 | 4710.0 | 4458.0 | 3481.0 | 3441.3 |

A warp a group is slower where the groups do not fill the device: JAC
+1.2% a step on an RTX 4090. On the RTX 4090 the kernel that prunes and
computes takes, a call, the loop and 0 to 7% more (nsys: Cellulose 584.9
µs against 550.6 for the loop and 228.6 for the main kernel of the
pruning; TIP3P 309,876 sites 539.3 against 503.6 and 206.8; STMV 1638.4
against 1654.2 and 575.0). The reading that fits, not measured with
counters: the part of a warp that prunes is bound by memory, the part that
computes by the shuffles and the arithmetic, and warps of the two parts
run side by side on a multiprocessor. On STMV it takes less than the loop
alone; a warp a group adds the sums of its members once rather than once a
block, which may be why. It holds 94 to 96 registers a thread, against 72
for the loop and for the pruning. On the RTX PRO 6000 Blackwell Max-Q, at
its cap of 300 W, the kernel takes the sum of the two (Cellulose 591.2
against 415.0 and 167.5), so the loop that prunes gains at most 1%; that a
device bound by its power gains nothing from running the two parts side by
side is a hypothesis, not measured. Below about 3,300 groups a warp a group
leaves the device partly idle where the loop had a warp for each active
unit. The loop therefore prunes from 4096 groups, and on a device of
`sm_12x` from 20,480, where a step is equal from about 19,000 groups.

### 11.8 The default of D261

**The size.** The matrix at its default reach is set against the groups
at the reaches of Section 6.3, in mixed precision, µs a step: 6 rounds
alternated in one job, the first dropped. The standard errors are 0.0 to
0.5 µs on the RTX 4090 and up to 30 µs on the RTX PRO 6000 Blackwell, which
other jobs shared. The setup:

- 9 Å: PME at a tolerance of $10^{-5}$ with a spacing of at most 1 Å,
  SHAKE and SETTLE, NPT (V-rescale and C-rescale every 25 steps) from an
  equilibrated state;
- 8 Å: the Amber suite's control files, PME on their grids, at constant
  energy, except STMV at constant pressure.

| System | Particles | Cutoff (Å) | RTX 4090: matrix | groups | ratio | RTX PRO 6000: matrix | groups | ratio |
|---|---|---|---|---|---|---|---|---|
| Dipeptide in TIP3P | 1,168 | 9 | 75.2 | 88.0 | 1.170 | 91.6 | 102.6 | 1.120 |
| TIP3P box | 1,431 | 9 | 67.2 | 81.8 | 1.217 | 71.6 | 84.4 | 1.179 |
| TIP3P box | 2,841 | 9 | 71.6 | 98.8 | 1.380 | 71.2 | 96.4 | 1.354 |
| Ala3 in OPC (examples/ala3) | 4,905 | 9 | 105.0 | 109.4 | 1.042 | 122.2 | 150.4 | 1.231 |
| TIP3P box | 5,880 | 9 | 81.6 | 88.8 | 1.088 | 81.0 | 85.4 | 1.054 |
| TIP3P box | 8,922 | 9 | 90.2 | 100.4 | 1.113 | 88.2 | 90.6 | 1.027 |
| TIP3P box | 13,899 | 9 | 115.0 | 118.0 | 1.026 | 105.0 | 104.0 | 0.990 |
| TIP3P box | 16,767 | 9 | 121.0 | 122.8 | 1.015 | 113.6 | 107.0 | 0.942 |
| TIP3P box | 20,931 | 9 | 136.6 | 138.6 | 1.015 | 132.8 | 114.0 | 0.858 |
| JAC | 23,558 | 8 | 137.0 | 135.2 | 0.987 | 127.8 | 113.8 | 0.890 |
| JAC, 4 fs | 23,558 | 8 | 148.0 | 146.0 | 0.986 | 136.0 | 123.0 | 0.904 |
| TIP3P box | 23,598 | 9 | 150.0 | 145.2 | 0.968 | 152.0 | 128.2 | 0.843 |
| TIP3P box | 25,860 | 9 | 162.0 | 151.2 | 0.933 | 153.2 | 128.6 | 0.839 |
| Ubiquitin in OPC | 26,031 | 9 | 253.0 | 190.4 | 0.753 | 196.0 | 161.8 | 0.826 |
| TIP3P box | 27,234 | 9 | 171.0 | 157.0 | 0.918 | 162.4 | 136.6 | 0.841 |
| TIP3P box | 52,686 | 9 | 394.2 | 244.0 | 0.619 | 283.4 | 204.2 | 0.721 |
| OPC box | 61,324 | 9 | 575.2 | 291.6 | 0.507 | 419.2 | 225.6 | 0.538 |
| Factor IX | 90,906 | 8 | 538.8 | 329.2 | 0.611 | 410.8 | 241.6 | 0.588 |
| TIP3P box | 103,158 | 9 | 710.8 | 411.2 | 0.579 | 549.8 | 308.6 | 0.561 |
| TIP3P box | 203,490 | 9 | 1320.0 | 775.6 | 0.588 | 1108.8 | 624.4 | 0.563 |
| TIP3P box | 309,876 | 9 | 1970.8 | 1140.4 | 0.579 | 1679.4 | 985.0 | 0.587 |
| Cellulose | 408,609 | 8 | 2394.4 | 1490.0 | 0.622 | 1821.8 | 1116.0 | 0.613 |
| OPC box | 413,168 | 9 | 3350.2 | 1634.4 | 0.488 | 2729.2 | 1426.0 | 0.522 |
| STMV, 4 fs | 1,067,095 | 8 | 7500.2 | 4916.4 | 0.656 | 6216.4 | 3641.0 | 0.586 |

The groups take less time from 23,558 particles on the RTX 4090, and from
13,899 on the RTX PRO 6000; why the crossover moves with the device is not
analyzed here. The threshold is set by the device where the groups take
less time later, so that no system runs slower than on the matrix. On the
RTX PRO 6000, systems between about 14,000 and 23,000 particles therefore
keep the matrix, which is 1 to 14% slower there. At the same size, OPC
favors the groups more than TIP3P.

**The reaches.** The data below, on top of the measurements of #345,
support cutoff + 3 Å and cutoff + 0.6 Å at 2 fs. The inner skin grows in
proportion to the step, the rule of the Amber suite's script.

| System | RTX 4090 | RTX PRO 6000 |
|---|---|---|
| TIP3P, 103,158 sites, 9 Å | 12 / 9.6 Å 411.2; 11 / 9.6 Å 418.6; 13 / 9.6 Å 414.6; 12 / 10 Å 418.8 | 308.6; 322.0; 310.4; 314.6 |
| JAC, 8 Å, 2 fs | 11 / 8.6 Å 135.2; 11 / 9.0 Å 136.0; 10 / 8.6 Å 141.4; 12 / 8.6 Å 133.8 | 113.8; 113.0; 121.0; 111.4 |
| Factor IX, 8 Å, 2 fs | 11 / 8.6 Å 329.2; 11 / 9.0 Å 334.6; 10 / 8.6 Å 340.4; 12 / 8.6 Å 330.4 | 241.6; 245.6; 247.8; 240.0 |
| JAC, 4 fs | 11 / 8.6 Å 146.0, pruned every 1.9 steps; 11 / 9.2 Å 145.0, every 3.5 | 123.0; 119.6 |
| STMV, 4 fs | 11 / 8.6 Å 4916.4, pruned every 1.0 step; 11 / 9.2 Å 4807.6, every 2.1 | 3641.0; 3527.6 |

At 8 Å an outer reach of 12 Å is 1 to 2% faster than 11 Å on JAC, and on
Factor IX 0.4% slower on the RTX 4090 and 0.7% faster on the RTX PRO 6000.
The default keeps cutoff + 3 Å, the reach of the suite and of #345, which
is within 2% of the best on every system measured.

**The validation.** A run without the keys, on an RTX 4090, against
`GROUPS` given at these reaches and against the matrix:

| System | Particles | Without the keys | `GROUPS` given | `MATRIX` |
|---|---|---|---|---|
| TIP3P box | 309,876 | 1140.8 | 1130.2 | 1970.4 |
| Cellulose | 408,609 | 1489.2 | 1488.4 | 2395.6 |
| STMV, 4 fs | 1,067,095 | 4766.8 (11 / 9.2 Å) | 4860.0 (11 / 8.6 Å) | 7474.2 |
| JAC | 23,558 | 132.2 | 132.6 | 134.2 |
| Dipeptide | 1,168 | 72.4 | 85.0 | 72.6 |
| TIP3P box | 13,899 | 114.2 | 116.6 | 114.0 |
| TIP3P box | 20,931 | 136.2 | 137.0 | 136.2 |

On JAC, without the keys and at constant energy over 1 ns (11 / 8.6 Å),
the total energy changed by $1.7\times10^{-5}$ of its value: a fitted
drift of 1.9 kcal/mol/ns, against fluctuations of 1.9 kcal/mol.

## 12. History and status

### 12.1 Decisions

| Decision | What it set | Section |
|---|---|---|
| D80 | The exact test of validity under scaling | 7.1 |
| D82 | Tiles of 8 particles, superseded by the groups | [neighbor-tiles.md](neighbor-tiles.md) |
| D84, D70 | Atomic additions in the type of the destination; fixed point planned | 6.1 |
| D86 | The loops read the order of the places | 1.1 |
| D89 | Groups of 16 that share a list, each pair once | All |
| D90 | The cheaper erfc that the gain needs | 11.1, 11.4 |
| D94, D96, D97 | The radial table, one load for two coefficients, the mask by ballots | 6.1 |
| D95, D101 | Frames without a minimum image, shifts added in f64 | 3, 7.3 |
| D99 | The queue of the warp between the box test and the point tests | 4 |
| D100, D105, D106 | The excluded partners: range, sorted keys, rows for a large group | 4 |
| D107, D176, D225 | Positions that are not numbers or far away; buffers whatever the positions | 9 |
| D114 | The dual list | 6.2 |
| D115, D242 | An entry for each image within the reach; complete below the least edge | 5, 5.2 |
| D126 | Triclinic cells | 5.1 |
| D215, D218, D312, D330, D331 | Lifetime, the start, the room of the gather, shared storage, the pool | 8.1 |
| D245 | One list for a matrix beside the groups | 6.2 |
| D261 | `AUTO` | 6.3 |
| D268 | Active units, the block index at the build, the copy in the frames kernel, shared frames | 6.2 |
| D281 | The skins chosen by the run | [neighbors.md](neighbors.md) |
| D284 | The pruning in the loop | 6.2 |
| D291 | The table of a block, measured and refused | 6.1 |
| D294 | A model in stages keeps the matrix | 6.3 |
| D357 | Coverage across IR boundaries | 10.1 |

**Where the sections moved** (2026-11-11). This document had the structure
in Section 2, the loop in Section 3, what the loop reads in Section 4, the
build in Section 5, the IR in Section 6, the measurements of MDIR in 6.1,
the dual list in 6.2, the default in 6.3, the stages in 7, and the open
questions in 8. The numbers that other files cite keep their content:
5.1 (triclinic cells), 5.2 (how far the lists reach), 6.2 (the dual list),
6.3 (the default). The rest moved: the structure and its sizes to
Sections 1 and 8, the loop to 6.1, what the loop reads to 3 and 7.3, the
build to 2 to 5, the IR to 10, the measurements to 11.

### 12.2 Stages

| Stage | What | Checked by |
|---|---|---|
| G0 | The build prototyped standalone (`scripts/experiments/neighbor-structures`) against the build of the matrix | Its time on Cellulose and JAC, before it is written as a template |
| G1, done but the matrix in the compact order | The compact order of the places (for the matrix as well); the structure and its build on a device; the exchange contracts on `md_exec.pair_for` | The pairs of the list within the cutoff are those of the matrix, on JAC and Cellulose; the matrix in the compact order against the order of cells |
| G2, under way: the default mode and the erfc of D90 done; the deterministic mode to come | The loop over groups, each pair once, both modes; the cheaper erfc of the direct sum under `fast_math`, with the accuracy of the force near the cutoff checked (it is what lets the loop over groups gain, Section 11.1) | Forces, energies, and virials against the matrix (to the rounding; to the bit in the deterministic mode, where the matrix sums in fixed point too); conservation over runs |
| G3, done (D261) | Groups by default on a device for the loops that allow them: `neighbor_structure = "AUTO"`, the groups with a dual list from 23,000 particles, measured against the matrix on two kinds of device (Section 11.8) | The rates of the matrix and the groups across sizes; a run without the keys at the rate of `GROUPS` given |
| G4 | The reach of the list: a skin of 1 Å, once builds are cheap. Since then the dual list (D114) and the choice of the skins by the run (D281) set the reaches | Rates and intervals between builds |
| G5, under way: the table of D94 and the frames of D95 done | The arithmetic of the kernel (a table for the erfc, no minimum image, the order of the entries) | The prototype first |

### 12.3 Open questions

- The size of a group: 16 measured best among 16 and 32 on Cellulose; 8
  is untried.
- Why the same loop over groups of 16 measures 996 to 1034 µs in
  `supercluster.cu` and 1162 to 1209 in `groups.cu`.
- The size of a unit of work (64 and 128 entries measured; Section 11.3).
- The groups are 1:1:2 prisms (Section 2); a cube would shorten the lists
  by a few percent by Section 8.4, not measured.

## Next

From the roadmap (#603). Measure first: the mean count of bits of the
masks of the inner list, the share of the 16 pairs of an entry that are
work (the cost of groups over single pairs, $\zeta_c$ of Section 8.4), and
whether a pruning is bound by its loads or its arithmetic. Then: in the
pruning, a test of the box of the group that keeps or drops an entry whole
when its candidate is within $R_\text{in} - a$ or beyond $R_\text{in} + a$
of the center, $a$ the half-diagonal of the box (at most about 5.3 Å for
the prism of 16 atoms of water, Section 2), before its 16 distances; by the
volume of the shell, a third or more of the outer entries lie beyond, and a
pruning is about 6% of a step of Cellulose. The forces of the loop over
groups accumulated in 64-bit fixed point, as the grid of PME (D70), which
makes groups deterministic (white paper, Section 13).

Not now: lists of clusters along a Hilbert curve with indices stored as
deltas in nibbles [[Thaler2026]](references.md#thaler2026), 3.6 bytes a
particle, which on a GH200 were no faster than the lists the authors
compare them with and slower than a full list. The loop over groups is bound by the pipe of loads and shuffles
(92% of its peak on Cellulose, white paper, Section 10.2), not by the
bytes of its list, and the lists fit (Cellulose: 153,517 blocks of 64
entries of 8 bytes, about 190 bytes a particle). The bytes of a list may
matter for systems of hundreds of millions of particles across many
devices, which have not been measured.
