# Tile Neighbor Structure for M1: Design

Status: superseded (2026-10-01) by D89, groups of neighbors
([groups-m1.md](groups-m1.md)). Proposed (2026-09-30), D82; not implemented. A prototype
(Section 11) did not bear out the premise of stage T1 on an RTX 3090: at
equal reach the tile kernels were slower than the neighbor matrix. The
stages are on hold until the reach of the list and the cost of a build,
which the prototype found to decide the rate, are addressed. The
keys in brackets are those of [references.md](references.md).

This document describes a second kind of neighbor structure, `tiles`, for
the loops over pairs on a device. The structure of M0, the neighbor matrix
([neighbors-m0.md](neighbors-m0.md)), stays; the CPU keeps it, and a device
may keep it for small systems.

## 1. Why

On an RTX 3090 MDIR runs JAC (23,558 atoms) at 375 ns/day and GROMACS
2026.3 (CUDA) at 747; the gap grows with the system: FactorIX at 86
against 248, Cellulose at 15.6 against 51. A step of JAC spends, by nsys
(the measurements of D79, D81, and D83):

| Part | µs per step |
|---|---|
| The fused loop over pairs (Lennard-Jones, direct Coulomb, six loops) | 137–153 |
| Builds of the neighbor matrix, amortized (one build ≈ 600 µs every 9.3 steps) | 67 |
| Refresh of the matrix with the excluded pairs, amortized | 20 |
| The reciprocal sum of PME (spread, FFT, scale, gather) | ≈ 115, on a second stream (D81) |
| The constraints and the kicks and drifts | ≈ 60 (D83) |

The loop over pairs and the build are two thirds of the step, and both
scale with the number of pairs. Three properties of the matrix make them
expensive:

1. **Every pair is computed twice.** The matrix is directed (a row for each
   particle holds all its neighbors), and the loop computes the kernel for
   `(i, j)` and for `(j, i)`, each writing to its own particle (the
   owner-computes policy of [ops-m0.md](ops-m0.md), Section 9.2).
2. **Every neighbor is an index and a random load.** A row of JAC holds up
   to 736 indices; for each, the loop loads the index (4 bytes) and the
   position of the neighbor (12 bytes in `f32`) from wherever it is.
3. **The list keeps every pair within the reach for the whole lifetime.**
   With a cutoff of 8 Å and a reach of 10 Å, $(8/10)^3 = 51\,\%$ of the
   pairs of the list are within the cutoff; the rest cost a load and a
   test each.

## 2. What is borrowed, and from where

The tile structure is not a new algorithm. It is the cluster pair list of
GROMACS, and this document names what it takes from it:

| Element | Source |
|---|---|
| Particles in clusters of a fixed number, a list of pairs of clusters instead of pairs of particles; the particle list is the special case of clusters of one | [[Pall2013]](references.md#pall2013) |
| Clusters formed by binning the particles into columns in x and y, sorting each column along z, and cutting it into clusters; bounding boxes of the clusters for the search | [[Pall2013]](references.md#pall2013) |
| A mask of bits per pair of clusters, for the pairs that interact and for the excluded pairs | [[Pall2013]](references.md#pall2013) |
| On NVIDIA devices, a warp of 32 threads computing a pair of an i-cluster of 8 and a j-cluster of 4; eight i-clusters sharing one list of j-clusters | [[Pall2013]](references.md#pall2013) |
| The pruning of cluster pairs by the distances of their particles, on the device | [[Pall2013]](references.md#pall2013), [[Pall2020]](references.md#pall2020) |
| A dual pair list: an outer list with a long buffer, built rarely, and an inner list with a short one, pruned from the outer one often and cheaply, in a rolling fashion between force computations | [[Pall2020]](references.md#pall2020) |
| Accumulation of forces in fixed point with integer atomics, whose sum does not depend on the order of the threads | [[LeGrand2013]](references.md#legrand2013), as MDIR already does for the grid of PME (D70) |

The white paper and the design documents say so, with these citations,
wherever these parts appear.

## 3. What MDIR does differently

Three properties of MDIR constrain the borrowed design.

**The lifetime of a list is exact, not estimated.** GROMACS keeps a pair
list for a fixed number of steps and chooses the buffer from an estimate
of the energy drift that missing pairs cause, for a tolerance the user
sets (0.005 kJ/mol/ps per atom by default); it argues that a test of the
displacements asks for a rebuild at almost every step in a large system,
since the largest displacement grows with the number of particles
[[Pall2020]](references.md#pall2020). MDIR rebuilds when a particle has
moved half the skin (D41, D80), so that no pair within the cutoff is ever
missed. The tiles keep that, for both lists of the dual scheme
(Section 6). The argument of GROMACS is real and is measured in Section 9
(the interval between builds against the size of the system); a policy of
a fixed interval with a stated tolerance would be a separate choice of the
plan (the rebuild policy `interval` of decisions.md, Section 7, decided to be
an option only and never the default), not a change to this structure.

**The kernel of a pair is the semantic kernel.** A loop over pairs runs the
body that the `md` dialect wrote, with the distance, the displacement,
and two values of each gathered field (ops-m0.md, Section 8.2). The tile
structure changes the traversal, not the body: the same kernel runs for
the pairs of a tile pair that its mask selects and that are within the
cutoff. Whether a pair may be computed once (Section 5.2) is decided by
the exchange contract of the kernel, which the IR already carries
(`exchange(symmetric)`, `exchange(antisymmetric, derived)`; ops-m0.md,
Sections 4.7 and 9.3).

**The deterministic level is a mode (P6, D84).** A sum whose order the
threads decide is excluded in the deterministic mode, and allowed by
default. The full list writes each
particle from one thread in a fixed order, as the matrix does. A half list
adds the contribution of a pair to the second particle from another
thread: with floating-point atomics that breaks the level, in fixed point
it does not (Section 5.2).

## 4. The structure

### 4.1 Tiles

A tile is up to $b = 8$ particles that are close together. The build forms
tiles as [[Pall2013]](references.md#pall2013) forms clusters:

1. Bin the particles into columns of the plane x-y, of the spacing
   $a = (b / \rho)^{1/3}$ for the mean density $\rho$, so that $b$
   particles span about as much along z as a column is wide: 4.3 Å in
   water.
2. Sort the particles of each column by z.
3. Cut each column into tiles of $b$, the last one possibly partial.

A tile stores the indices of its particles (`i32`, −1 for the padding of a
partial tile) and its bounding box (in `f32`, rounded outward, so that a
box never excludes a pair it holds).

The particles are **not permuted** into the order of the tiles. MDIR
renumbers the members of every tuple set on the host when it permutes
(`md_exec.renumber`, D44), which is too slow to do at every build. Instead
the loop that converts the positions to the type of the kernel once per
evaluation (D79) writes them in the order of the tiles, gathering by the
indices of the tiles; the loop over pairs reads the positions of a tile as
$b$ consecutive entries, and writes the force of a particle once, at its
own index. Padding slots get a position far outside the cell, which the
cutoff test rejects. `md_exec.spatial_order` stays, for the locality of
the other loops.

### 4.2 The list of tile pairs

For each target tile $A$ the list holds the source tiles $B$ with at least
one pair within the reach $R$ at the build, in compressed sparse rows:

```text
row[A] .. row[A + 1]    the records of tile A
record:                 source tile B (i32), mask M_AB (64 bits)
```

$$
M_{AB}[u, v] = 1 \iff u, v \text{ are particles},\ (u, v) \text{ not excluded},\ u \ne v,\ \lVert \mathbf x_u - \mathbf x_v \rVert_\text{mi} \le R \text{ at the build.}
$$

The excluded pairs are cleared in the masks at the build, from the
relation `excluded`, instead of the per-entry marks of the matrix
(`md_exec.reset_neighbors ... exclude`). The build counts the records of
each tile, takes the prefix sum, and fills; it never truncates, and a list
that does not fit stops the run with the capacity it needed, as the matrix
does.

The minimum image of each pair is computed in the kernel, as now (D79),
rather than from a shift stored per tile pair as in
[[Pall2013]](references.md#pall2013); a shift per record is one of the
variants Section 9 measures.

## 5. The loop over a tile pair

### 5.1 Full list (owner computes)

A tile pair $(A, B)$ is in the rows of both $A$ and $B$. A warp computes
the row of one target tile: its 32 lanes are 8 targets $u$ × 4 lanes $q$,
and for each record the lane $(u, q)$ evaluates the sources $v = q$ and
$v = q + 4$ whose mask bit is set and which are within the cutoff. At the
end of the row the four lanes of a target add their sums with shuffles,
and one writes the force. No write conflicts, and the order of the sum of
a particle is fixed: the deterministic level holds.

The global sums take the weight ½, as for the directed matrix (ops-m0.md,
Section 9.2). Rows much longer than the mean are split across warps, with
a second, fixed-order sum over the parts.

### 5.2 Half list (each pair once)

A tile pair is stored once, in the row of the lower of the two tiles, and
the pair $(u, v)$ of a tile with itself once, for $u < v$. The kernel of
the pair gives the contribution to $u$; the exchange contract gives that to
$v$, $s\,k(u, v)$ with $s = +1$ for a symmetric and $s = -1$ for an
antisymmetric kernel (ops-m0.md, Section 9.3). The contributions to the
eight sources of a record are added across the lanes and written to the
forces of $B$ from the thread of $A$: a conflict, with two strategies.

| Strategy | Pair work | Determinism (P6) | Cost |
|---|---|---|---|
| Full list (5.1) | 2× | Deterministic | Twice the kernels |
| Half list, `f32` atomic additions | 1× | Not deterministic: the order of the additions to a particle depends on the threads; allowed by default (D84) | Atomics to global memory |
| Half list, fixed point (64-bit integers, as D70) | 1× | Deterministic | Integer atomics, a buffer of 64-bit integers, a conversion per particle per step |

A loop over pairs whose kernel has no exchange contract (`exchange(none)`)
takes the full list. The choice is a policy of `md_exec.pair_for`
(`policy(directed, owner_only)` now; `policy(once, fixed_point)` and
`policy(once, atomic)` for the half list, the last one not in the
deterministic mode, D84). This document does not choose between them;
Section 9 measures them.

## 6. Validity, with a dual list

Implemented for the groups of 16 (D114, [groups-m1.md](groups-m1.md),
Section 6.2); the tiles remain on hold.

A structure is valid for the positions $\mathbf x$ in the cell of edges
$\mathbf L$ when no pair within the cutoff $r_c$ is left out. D80 gives the
condition for a structure built at $\mathbf x^\text{ref}$ in the cell
$\mathbf L^\text{ref}$ with the reach $R$: with $\mathbf m = \mathbf L /
\mathbf L^\text{ref}$ per axis,

$$
2 \max_i \lVert \mathbf x_i - \mathbf m \odot \mathbf x^\text{ref}_i \rVert
\le \min(\mathbf m)\, R - r_c .
$$

It holds for tiles unchanged: a tile pair covers its particle pairs, and a
particle pair is left out only if it was farther than $R$ apart at the
build (a zero bit of a mask, or a tile pair not in the list).

The dual list adds an inner structure, pruned from the outer one at
$\mathbf x^{p}$ in the cell $\mathbf L^{p}$ with the reach $R_\text{in} =
r_c + s_\text{in} < R$: its masks keep the bits of the outer masks whose
pairs are within $R_\text{in}$ at $\mathbf x^{p}$. With $\mathbf m^{p} =
\mathbf L / \mathbf L^{p}$, the inner structure is valid when

$$
2 \max_i \lVert \mathbf x_i - \mathbf m \odot \mathbf x^\text{ref}_i \rVert
\le \min(\mathbf m)\, R - r_c
\quad\text{and}\quad
2 \max_i \lVert \mathbf x_i - \mathbf m^{p} \odot \mathbf x^{p}_i \rVert
\le \min(\mathbf m^{p})\, R_\text{in} - r_c .
$$

*Proof.* Let $(i, j)$ be a pair that the inner structure leaves out. Either
the outer structure left it out, and the first condition keeps it beyond
$r_c$ (D80); or the pruning dropped it, so that
$\lVert \mathbf x^{p}_i - \mathbf x^{p}_j \rVert_\text{mi} > R_\text{in}$.
Scaling by $\mathbf m^{p}$ moves the two apart by at least the factor
$\min(\mathbf m^{p})$, and each particle is within $d = \max_k \lVert
\mathbf x_k - \mathbf m^{p} \odot \mathbf x^{p}_k \rVert$ of its scaled
position, so $\lVert \mathbf x_i - \mathbf x_j \rVert_\text{mi} >
\min(\mathbf m^{p}) R_\text{in} - 2 d \ge r_c$ by the second condition. ∎

Two consequences follow: the inner list is pruned from the outer one,
never from the previous inner one, since a pair dropped once may come
back; and an inner
list pruned exactly to the cutoff ($s_\text{in} = 0$) is valid only at the
step of its pruning. Both maxima come from the loop that moves the
particles (D41) and share one flag.

When the first condition fails the outer structure is rebuilt (and the
inner one pruned from it); when only the second fails the inner structure
is pruned again, which reads the outer records and the current positions
and costs no search. GROMACS prunes on a schedule [[Pall2020]](references.md#pall2020);
MDIR prunes when the condition asks, and may prune earlier on a schedule
to keep the inner list short, which never makes it invalid.

## 7. In the IR

| Part | Now | With tiles |
|---|---|---|
| The structure | `md_exec.empty_neighbors kind(matrix) width(W)` | `kind(tiles) tile(8) capacity(C)`; `C` is the number of records, grown and rebuilt when a build needs more |
| The build and the test of validity | `md_exec.refresh_neighbors`; `md-exec-expose-validity` writes the test of D80 into the loop that moves the particles | Unchanged for the outer structure: the test is that of D80, whatever the kind |
| The excluded pairs | `md_exec.reset_neighbors ... exclude(...)` marks entries | The build clears the bits of the masks |
| The inner structure | — | `md_exec.prune_neighbors %outer, %x, %cell reach(R_in)`, a structure with a reference of its own; `md-exec-expose-validity` writes both tests, `md-exec-reuse-neighbors` refreshes the outer one or prunes the inner one as Section 6 says |
| The loop over pairs | `md_exec.pair_for %n ... policy(directed, owner_only)` | Unchanged in the IR; the lowering chooses the layout of Section 5 from the kind of the structure, as it chooses the lanes of a row now (`row-lanes`) |
| Each pair once | ops-m0.md, Section 9.3, not implemented | `policy(once, fixed_point)` or `policy(once, atomic)`, legal only with an exchange contract |
| The positions of the kernels | The conversion loop of D79 | The same loop, writing in the order of the tiles, with the indices of the tiles as a field it gathers by |

The tile structure is a device structure: the CPU lowering keeps the
matrix, and a module that asks for tiles on the CPU is refused with a
message, until a CPU layout exists.

## 8. The build on a device

| Step | Work |
|---|---|
| 1 | The columns: a counting sort of the particles by column, as the cells of the matrix are sorted now (neighbors-m0.md, Section 2.1) |
| 2 | Each column sorted by z: one thread per column, a sort by insertion (the columns hold tens of particles) |
| 3 | The tiles: indices and bounding boxes, one thread per tile |
| 4 | Count: for each target tile, the tiles of the columns within reach whose boxes are within $R$ and whose mask is not zero; a warp per target tile evaluates the 64 distances of a candidate at once |
| 5 | Offsets: the prefix sum of the counts (the chunked scan of the matrix build) |
| 6 | Fill: step 4 again, writing the records at their offsets |

Steps 4 and 6 do the same work twice rather than appending with a global
counter; if the second pass costs too much, a warp may keep its records
in pages of its own. The order of the records of a row is the order in
which the build visits the candidate tiles, fixed, so that the sum of a
particle has a fixed order (Section 5.1).

## 9. Plan and measurements

Each stage is measured before the next one begins, on JAC, FactorIX,
Cellulose, and STMV (the Amber suite, `scripts/benchmarks/amber`), one run
each (the benchmark policy of the project), with the times of the kernels
from nsys and the rate in ns/day, against the matrix and against GROMACS
2026.3 (CUDA, same settings).

| Stage | Content | What it answers |
|---|---|---|
| T1 | Tiles, full list, one list at the reach of now (10 Å for JAC) | The gain of contiguous loads and of fewer indices, alone |
| T2 | The half list in fixed point and with `f32` atomics | The price of determinism, and of the doubled work of T1 |
| T3 | The dual list: the outer one built as T1 or T2, the inner one pruned when Section 6 asks | The gain of fewer pairs outside the cutoff, and a longer outer lifetime |
| T4 | Variants: shifts stored per record, super-tiles of eight tiles that share one list | Whether the refinements of [[Pall2013]](references.md#pall2013) pay in MDIR |

With each stage the interval between builds (outer and inner) is logged
against the size of the system, which measures the argument of GROMACS
against displacement tests (Section 3).

## 10. Open questions

- The size of a tile on other devices (AMD wavefronts of 64) and whether
  the lowering chooses it from the target.
- Cells that are not orthorhombic: the bounding boxes and the columns
  assume an orthorhombic cell, as MDIR does now.
- Loops over one structure with different cutoffs (a cutoff below the
  reach of the list) are handled by the cutoff test, as now; a loop whose
  cutoff is much shorter would want a list of its own.
- A domain decomposition (M2c) keeps the structure per domain; the rows
  that need halo particles are those with records of halo tiles, which the
  build can classify, so that the other rows run while the halo arrives.

## 11. Prototype measurements (2026-09-30)

A standalone CUDA program
([scripts/experiments/neighbor-structures](../scripts/experiments/neighbor-structures/README.md),
with its results and plots) timed the loop over the pairs of JAC with lists built on the
host: Lennard-Jones from type tables and the direct sum of PME in `f32`,
the particles in the spatial order of MDIR, 1000 launches each, RTX 3090.

| Reach (Å) | Matrix, 16 lanes (µs) | Tiles 8 × 8, full (µs) | 8 × 4, full, a row over 4 warps (µs) | 8 × 4, half, `f32` atomics / fixed point (µs) |
|---|---|---|---|---|
| 8.05 | 60.6 | 142.1 | 111.0 | 112.1 / 106.7 |
| 8.5 | 70.2 | 148.1 | 116.2 | 118.4 / 112.9 |
| 10.0 | 108.3 | 171.2 | 136.7 | 159.3 / 148.3 |

The slots of the 8 × 4 records that hold a pair of the list are 38 to
47 %; of the entries of the matrix, 98 % are within the cutoff at 8.05 Å
and 51 % at 10 Å. A full list of tiles computes about twice the slots of
the matrix, and the kernel is bound by the arithmetic of the pairs, so the
contiguous loads do not pay for the empty slots. The half list halves the
records but adds the reduction and the atomic additions of the sources.

GROMACS 2026.3 on the same system (nsys, `-update gpu`): its nonbonded
kernel takes about 50 µs per step with `rlist` 8.05 Å and a half list, and
all its kernels 154 µs. The tile kernel of the prototype is twice as slow
as that of GROMACS for the same list, and the matrix with a list of
8.05 Å is within 20 % of it.

Pruning the matrix from an outer reach (a warp per row, compacted with a
ballot) took 87 to 96 µs from 10 Å, as much as it saves.

What separates MDIR from GROMACS on this system is therefore the reach of
the list (10 Å against 8.05 Å) and the cost of a build (about 600 µs), not
the layout. With the measured intervals between builds (every 3.0 steps at
8.6 Å, 9.3 at 10 Å), the time $K(R) + B / I(R)$ of the loop and the builds
is lowest at 10 Å for $B = 600$ µs, as measured; for $B = 150$ µs it moves
to about 9 Å and falls by about a third.
