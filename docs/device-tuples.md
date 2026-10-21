# The structures of the tuples built on the device (D257)

Issue #295; the design as ruled by the maintainer on PR #329 (all three questions as proposed). The
structure is stored in ragged rows and built on the device from the
members since D314 (#471, Sections 2 and 3). It builds on
the store of D253 (`runtime/mdrt_tuples.h`), which a Python simulation and
a frame evaluator keep across their activations and, with #316
(D255, #316), `mdir run` across its segments. Section
numbers below refer to this document.

## 1. What a start in a new order costs now

Where a program puts its particles in a new order (a segment of `mdir run`;
a start of an activation after steps; a commit of a borrow that moves a
particle to another cell of the order), every tuple set is renumbered and
its incidence structure built on the host, and on a GPU the structure is
uploaded, 16 MiB at a time (D253):

- the identities of the particles are read from the device once for each
  tuple set;
- the members in the order of the files are renumbered on the host,
  $m'_{t,s} = \mathrm{place}[m_{t,s}]$ with
  $\mathrm{place}[\mathrm{id}_p] = p$;
- the rows of the incidence structure are written on the host from an
  index of the tuples of each particle, a range of particles at a time,
  and each range is uploaded;
- outside the deterministic mode the renumbered members are uploaded as
  well, for the loops that evaluate each tuple once.

Measured (one RTX 4090, mixed precision, the settings of the Amber suite,
a segment of `mdir run`; D255): 4.5
ms on JAC, 15 ms on Factor IX, 78 ms on Cellulose, and 238 ms on STMV a
start in a new order. A Python simulation pays the same at a start after
steps (D253: 5.1, 17, and 98 ms with its evaluation on JAC, Factor IX, and
Cellulose, one RTX PRO 6000). A start in the order of the last one takes
what is kept and pays the comparison alone.

## 2. The observation that makes the build a gather

The incidence structure in a new order is the structure in the order of
the files with its rows permuted and its members relabeled. Let
$\mathrm{id}_p$ be the particle (in the order of the files) at place $p$,
and $\mathrm{place}$ its inverse. The row of place $p$ holds, in the order
of the tuples $t$ and of the slots $s$, every $(t, s)$ with
$m'_{t,s} = p$, that is with $m_{t,s} = \mathrm{id}_p$. So the row of
place $p$ lists the same $(t, s)$, in the same order, as the row of
particle $\mathrm{id}_p$ in the order of the files, and only the members
that it copies change, to $m'_{t,\cdot} = \mathrm{place}[m_{t,\cdot}]$.
The number of tuples of a row is the count of particle $\mathrm{id}_p$, so
the size of the structure does not depend on the order (D253 relies on
this already).

The index of the tuples of each particle in the order of the files (the
`MdrtTupleIndex` of `mdrt_tuples.h`: for particle $q$ the entries
$(t, s)$ from $\mathrm{first}_q$ to $\mathrm{first}_{q+1}$, in the order of
the tuples) is therefore a function of the members alone. It is built once
for a tuple set, on the host, where it is built now, and uploaded once. A
start in a new order then needs no count, no prefix sum, no atomic, and no
sort on the device: three loops in which every element is written by
exactly one thread.

**What is borrowed.** The structure is the compressed rows of a sparse
matrix (CSR, [[Saad2003]](references.md#saad2003), Chapter 3), the
incidence matrix of particles by tuples, built by distribution counting.
Until D314 the index above was built on the host
(`mdrtTupleIndexBuild`, D253), uploaded once for each tuple set, and the
rows of the dense structure gathered from it on the device. Since then the
device builds the ragged rows from the members directly (Section 3), with
the counting sort of the primitives of D296, and the index exists on the
host only, for the host's own builds.

## 3. The kernels

For each tuple set whose structures are to be built, the lowering
(`convert-md-exec-to-gpu`) launches, with the existing `launchOver`:

1. **place**, one thread a place $p$: $\mathrm{place}[\mathrm{id}_p] = p$
   (a scatter through a permutation: each element written once);
2. **renumber**, one thread an element of the members:
   $m'_i = \mathrm{place}[m_i]$, into the renumbered members on the
   device;
3. **rows** (D314, `buildOnDevice`), the structure in ragged
   rows ([ops.md](ops.md), `md_exec.build_incidence`;
   `include/mdir/Conversion/RaggedRows.h`) from the members $m'$ (or $m$
   in the order of the files), by `emitDeviceRaggedRows` and two kernels
   of its own:
   - a thread for each membership $j = a t + s$ adds 1 to the count of its
     place $m'_j$ (a device atomic), after a kernel clears the counts;
   - the exclusive scan of the counts gives the offsets $o_p$ of the rows
     (`emitDeviceScan`, three launches);
   - a thread for each membership takes the next slot of its place from
     the cursor (`emitCursorScatter`) and writes $j$ there;
   - a thread for each place sorts the $j$ of its row ascending
     (`emitSortBins`, an insertion sort; a row holds at most a few dozen);
   - a thread for each row writes its offset (and the first thread the
     number of rows, the first $a + 2$ threads the entry of zeros at the
     end), and a thread for each slot writes the entry of its $j$:
     $t = \lfloor j/a \rfloor$, $s = j \bmod a$, and $m'_{t,0..a-1}$.

   The counts, the cursor, the offsets, the slots, and the sums of the
   scan's blocks are scratch of the pool, freed after the build: $4(3N +
   M)$ bytes for $N$ places and $M$ memberships.

**Determinism.** The atomics of the count and of the scatter run in any
order, but the order within a row is fixed by the sort: ascending $j$ is
ascending $t$, and $s$ within one tuple, which is the order in which the
host's build (`mdrtTupleIndexBuild`, a loop over $t$ then $s$) fills a
row. So the structure does not depend on the order in which threads run,
and equals the host's element for element; `MDRT_CHECK_KEPT_TUPLES`
compares the whole buffer (Section 6). The same kernels run in and outside
the deterministic mode. A kernel of an integration run (D110) reads the
first entry of every particle's row and keeps it only where the row has
one: that read is the first entry of the next row or the entry of zeros at
the end, inside the buffer (#347).

**Where they come from.** The kernels are emitted by the lowering, as the
loops of the program are, not written in the runtime library, which is
compiled by the host's C compiler and has no kernels of its own; the
runtime keeps the memory and decides, as below, and the code launches.

## 4. With the store of D253

The store decides as it does now (I1 of
[jit-invariants.md](jit-invariants.md#what-a-simulation-keeps-across-activations-d253)):
a structure is taken only where its two inputs, the members in the order
of the files and the identities, equal those it was built from, element
for element; a structure that is not taken is built in the memory of an
entry that the code does not hold (I2). What changes is where the inputs
are compared and where the structure is built:

- **The members** are on the host, given by the entry, and are compared
  there with the kept copy, as now (`memcmp`; part of the 0.08 to 1.7 ms
  that D253 measured for both inputs on JAC, Factor IX, and Cellulose).
- **The identities** are on the device. Now they are read back once for
  each tuple set. Proposed: the store keeps one copy of the identities on
  the device, those of the last order it was asked for, and a number of
  that order. The code compares the identities that it hands the store
  with that copy in one kernel, which writes a flag on the device (every
  thread that finds a difference writes the same value, so the races are
  benign), and the store reads the flag, 4 bytes. This happens once for
  each value of the identities that the code holds, not once for each
  tuple set: the lowering emits the comparison once for each SSA value of
  identities, before the calls of the sets that use it. A segment of
  `mdir run` holds two: the anchors of the constraint groups are
  renumbered in the order before (`emitReorder`), from which the new order
  is computed, and every set then in the new order. If the identities
  differ, the copy is replaced on the device and the number of the order
  advances. An entry is taken only where its members are equal and its
  order is the current one, which is the same condition as an element-wise
  comparison of the identities with those it was built from, since the
  number advances whenever one identity differs. (An order that comes back
  after another is rebuilt: the comparison is with the last order only.
  That is conservative; no program alternates orders, but the anchors of a
  segment, renumbered in the order of the segment before, are compared
  with the order of that segment's sets and may then be taken. The counts
  of taken and built structures will be read from the prototype, not
  carried over from #327's tests.)
- **A start whose order did not change** takes what is kept, with no
  launch and no copy but the flag.
- **A start in a new order** launches the kernels of Section 3 into the
  memory of the entry; nothing is copied between the host and the device
  but the flag.
- **The call protocol** (one call before, one after, for each set): the
  code calls the store with the members and the identities and gets back
  the buffers of the entry (the members on the device, the renumbered
  members, the structure) and whether to build; it launches the
  kernels under an `scf.if` on that answer, then tells the store that the
  structure is built. **An entry becomes valid only at that second call**,
  never at the first: an activation that fails between the launches and
  the second call (D215's failure path) leaves the entry invalid, and the
  next start builds it again rather than take a half-written structure.
- **The lowering's side.** `md_exec.renumber` gives the call before, and
  `md_exec.build_incidence` of its result gives the launches and the call
  after; the pass carries the store's answer from the first op to the
  second in a map keyed by the result of the renumbering, as it carries
  `runtimeMembers` now. The result of `md_exec.renumber` on the host
  becomes a value that nothing reads. A program in the order of the input,
  whose members go to `md_exec.build_incidence` without a renumbering
  (`isGivenToFunction`, the frame evaluator), builds the structure of the
  order of the files through the same kernels with
  $\mathrm{place} = \mathrm{id} = $ the identity, once.
- **The copy of the members for the loops that evaluate each tuple once**
  (outside the deterministic mode) is the renumbered members on the device,
  which kernel 2 writes: no separate upload.
- **The structure of tuples found at run time** (the triplets of a
  three-body term) is built as now; it is not a tuple set of the topology.

**What stays on the host, and why.**

- The members in the order of the files: they are the program's input,
  and the comparison with the kept copy reads them there.
- Until D314, the index of the tuples of each particle in
  the order of the files, built on the host and uploaded once for each
  tuple set. The device now builds the rows from the members (Section 3),
  so the index and its $N + 1 + 2M$ i32 on the device are gone.
- The decisions of the store (which entry, take or build), and the check.
- The renumbered members are no longer needed on the host: the only
  consumers of `md_exec.renumber` on a GPU are `md_exec.build_incidence`
  and the copy for the loops over tuples once (`md-exec-assign-storage`
  keeps relations on the host for that reason alone). The pass that
  lowers them is the one that changes; the storage form does not.

## 5. Memory on the device

For each tuple set, new on the device and shared by its entries: the
members in the order of the files (4 bytes an element) and, until
D314, the index (8 bytes an element, and 4 bytes a particle
and one). For each entry, in
the deterministic mode, the renumbered members (outside it they are the
copy that exists now). For the store, one copy of the identities and one
array $\mathrm{place}$ (4 bytes a particle each), shared by the sets. From
the sizes of D253 (the five sets of the flexible systems): about 2.6 MB on
JAC, 8.5 MB on Factor IX, and 68 MB on Cellulose, against incidence
structures of 42.6, 164.4, and 696.3 MB. On the host the store no longer
needs the copy of the identities of each set and the renumbered members
(1.2 MB on JAC, 28 MB on Cellulose of the 48 MB of D253), nor, on a
device, the 16 MiB in which rows are written.

Measured with the prototype (the store's own count, `MDRT_PROFILE`; a
Python simulation of each system of the suite with PME, flexible, five
tuple sets, mixed precision, outside the deterministic mode), bytes on the
device:

| System | Atoms | Incidence structures | Renumbered members | Members and indexes of the sets | Order (identities and place) | New |
|---|---|---|---|---|---|---|
| JAC | 23,558 | 42,592,864 | 704,244 | 2,583,912 | 188,468 | 2.77 MB, 6.5% |
| Factor IX | 90,906 | 164,358,048 | 2,219,052 | 8,475,296 | 727,252 | 9.20 MB, 5.6% |
| Cellulose | 408,609 | 696,269,736 | 19,895,616 | 67,859,048 | 3,268,876 | 71.1 MB, 10.2% |
| STMV | 1,067,095 | 2,211,020,840 | 40,757,344 | 143,613,952 | 8,536,764 | 152.2 MB, 6.9% |

The renumbered members replace the copy that the loops over tuples once
read, of the same size, so what is new is the members and indexes of the
sets and the order (the last column, against the incidence structures).
In the deterministic mode, where there is no such copy, the renumbered
members are new as well.

## 5a. The cost of a start in a new order, measured

`scripts/benchmarks/activation/start.py --only moved` (20 steps and
`gradient()`, the steps subtracted; a start with its evaluation), `main`
(7bfa755) and this decision in turn six times in one job on one RTX 4090,
mixed precision; the least of five starts, ms, the medians of the five
rounds after the first:

| System | main | This decision | Of which the functions of the store |
|---|---|---|---|
| JAC | 5.29 | 1.98 | 3.3 → 0.07 |
| Factor IX | 18.1 | 5.67 | 12.2 → 0.17 |
| Cellulose | 96.9 | 26.2 | 70 → 1.8 |

`mdir run` with the settings of the Amber suite, the same way, s a run
(mean ± standard deviation of the five rounds after the first):

| System | Checkpoint every 1,000 steps, five segments | every 10,000, two | The suite's own run, one segment, ms a step in its second half |
|---|---|---|---|
| JAC | 0.784 ± 0.005 → 0.766 ± 0.005 | 2.810 → 2.806 | 0.1400 → 0.1398 |
| Factor IX | 1.970 ± 0.007 → 1.900 ± 0.007 | 7.092 → 7.080 | 0.3454 → 0.3454 |
| Cellulose | 8.724 ± 0.015 → 8.330 ± 0.038 | 31.41 → 31.26 | 1.515 → 1.509 |
| STMV | 27.41 ± 0.07 → 26.18 ± 0.03 | 100.76 → 100.15 | 4.810 → 4.813 |

What remains of a start on Cellulose is not the tuples (1.75 ms in the
functions of the store, with the comparison and the launches): it is the
evaluation, the arguments of the entry, and the rest of D253's list.

## 5b. Failures, the order of the stream, and the deterministic mode

- **A kernel that fails.** The device runtime ends the process at the
  first call that reports an error of the device (`check` in
  `mdrt_cuda.c`), with the name of the last kernel launched; nothing of
  the store survives the process. An activation that ends otherwise
  between the launches and `mdrtDeviceTuplesBuilt` (D215's boundaries:
  a part that is not numbers, a stop) leaves the entry invalid, since it
  becomes valid only at that call, and `mdrtTupleStoreBegin` does not
  change that: the next start in that order builds it again.
- **The order of the stream.** The comparison, the kernels of Section 3, the
  copy of the identities into the store (`cuMemcpyDtoDAsync`), and every
  kernel of the program run on the one stream of the runtime. The
  renumbered members that the loops over tuples once read are therefore
  complete before the first of those kernels, as the rows are before the
  loops over the incidence structure; no wait is needed.
- **The deterministic mode** has no loops that evaluate each tuple once
  (D84), so nothing reads the renumbered members there but the kernel of
  the rows, on the same stream. The kernels are the same in both modes.
  The count and the scatter of the rows use device atomics, whose order
  the sort of each row removes (Section 3); the flag of the comparison is
  written with the same value by every thread that writes it.

## 6. The check, and the CPU

- **`MDRT_CHECK_KEPT_TUPLES`** (D253) is reused: with it set, every
  structure that a start takes is also built on the host and compared with
  the one read back from the device, as now, and every structure that the
  kernels build is compared in the same way when the code says it is
  built. Since D314 the whole buffer is compared, the
  offsets and the entry of zeros included. A difference ends the program
  and names the structure. No new
  variable. Under the check the identities are read back from the device,
  since the host's build needs them; the claim of no copy is for a run
  without it.
- **The CPU target** is unchanged: its structures are on the host, where
  the store builds them in kept memory (D253).

## 7. Programs

The programs of `mdir run` (with #327, which opens the store) and of the
embedding front ends: a Python simulation and the simulation of a frame
evaluator. A frame evaluator keeps the particles in the order of the input
(D251); this design does not change that. Code that runs with no store
entered (no driver does) gets its structures built on the device in memory
of its own, and frees them.

## 8. Measurement plan

- **The cost of a start in a new order**, on one kind of device, `main`
  and this decision in turn in one job, at least five rounds, the first
  dropped: a segment of `mdir run` on JAC, Factor IX, Cellulose, and STMV
  (the marks of #327's table, and runs with a checkpoint every 1,000 and
  10,000 steps), and a start after steps of a Python simulation
  (`scripts/benchmarks/activation/start.py --only moved`, against #294's
  and D253's tables). Expected: the three kernels write the structure once
  (696 MB on Cellulose, about a millisecond at the bandwidth of an RTX
  4090) where the host now takes 78 ms.
- **The rate of a step** and the Amber suite's rates, not slower: the
  kernels of a step are not changed.
- **Device memory**, before and after, read from the store and from the
  device.

## 9. Validation plan

- **The structures against the host's, to the element**, under
  `MDRT_CHECK_KEPT_TUPLES`: every built and every taken structure, in the
  tests of the store (`python-kept-tuples*.test`, `run-kept-tuples*.test`)
  and on the four systems of the suite in runs of several segments.
- **Energies, forces, and the virial of every tuple term to the bit**
  against `main` in the deterministic mode, on the four systems of the
  suite, in mixed and double precision, a run and its continuation (the
  comparison of #327), and the sequence of starts of D253's test.
- **compute-sanitizer**, `memcheck` and `initcheck`, on the new kernels
  (the GPU test of the store), with the time it held the device.
- **The CPU target**: the suite's tests, unchanged.

## 10. Considered and not proposed

- **A count, a prefix sum, and a fill with atomics on the device** (the
  usual construction of compressed rows at every start): the fill by
  atomics puts the tuples of a row in the order in
  which threads run, and the rows would have to be sorted afterwards to
  equal the host's; with the index of Section 2 nothing is counted.
  *Taken by D314* (#471), with the sort of each row, when
  the rows became ragged: it builds from the primitives of D296 and needs
  no index on the device.
- **Sorting the pairs (particle, tuple, slot) by key on the device** at
  every start: correct and deterministic with a stable sort, but a sort of
  $T a$ keys where a gather suffices, and a sort that the lowering would
  have to emit.
- **Keeping the identities' comparison on the host**, reading them back
  once a start (1.6 MB on Cellulose) instead of once a set: simpler, and
  0.1 to 0.7 ms; it leaves a copy where the issue asks for none
  (ruled on PR #329: compared on the device).
- **Kernels in the runtime library** (PTX compiled with the toolkit at
  the build of MDIR): a second way of making kernels, which the build does
  not have; the lowering already emits every kernel.

## 11. The kernel of the tuples evaluated once: the sums of a block (D283)

Issue #408. Outside the deterministic mode a loop over tuples that may
share particles evaluates each tuple once, a thread a tuple, and the loops
of a step without global sums (the bonded terms, the 1-4 pairs, and the
correction of the excluded pairs of the reciprocal sum) are one kernel
(D84, D181). Until this decision each thread added every value of its tuple
to the destination of its member with an atomic of its own: three for the
force on each member.

The table is now emitted by the emitter that every site of the strategy
shares, `kernels::BlockHashTable` (D291,
[device-primitives.md](device-primitives.md), Section 2.6), with the same
generated code.

**What the kernel spent its time on.** Cellulose NVE (408,609 atoms,
1,700,480 tuples, of which 862,209 excluded pairs), mixed precision, one
RTX 4090, Nsight Systems, µs a step of the steady part of 3,000 steps:

| Set, as a kernel of its own | Tuples | µs | ps an atomic |
|---|---:|---:|---:|
| Excluded pairs | 862,209 | 89.9 | 17 |
| Dihedrals | 336,960 | 83.3 | 21 |
| 1-4 pairs | 272,256 | 33.5 | 21 |
| Angles | 177,228 | 33.1 | 21 |
| Bonds | 51,840 | 7.4 | 24 |
| The run as one kernel | 1,700,480 | 222.6 | |

Every set costs about the same per atomic addition, whether it reads
parameters (the dihedrals read three numbers in f64 a tuple) or none (the
excluded pairs). With plain stores in place of the atomics (wrong forces,
for timing alone) the kernel took 65 µs, and with a load, an addition, and
a store without an atomic 84 µs; on STMV (1,067,095 atoms) 444, 154, and
205 µs. About 70% of the kernel was its global atomics. The excluded pairs
evaluated by the rows of their particles instead (each pair twice, no
atomic) took 93 µs against 90: their cost is the atomics they share with
every set, not their arithmetic.

**How much the tuples of a block share.** In the order of the run, the
tuples of a set one after the other in the order of the files, a block of
128 tuples touches 0.237 distinct particles a member on Cellulose (the
dihedrals 0.13, the excluded pairs 0.27), 0.309 on JAC, and 0.269 on STMV;
the 32 tuples of a warp, matched one slot at a time, 0.48 on Cellulose.

**The kernel.** A block of 128 threads keeps a table of the particles of
its tuples in block memory: a key of 32 bits for each slot, −1 where it is
empty, and the sums of each destination of the run for each slot (three
numbers for a force). The table has 128 times the largest arity of the
run, rounded up to a power of two (512 slots and 8 KB for a run with
dihedrals and forces in f32), so a block has room for every member of its
tuples; a run whose destinations would take more than 48 KB keeps the
atomics. The threads empty the table, and after a barrier each evaluates
its tuple and, for each member, finds the slot of the particle: a
multiplicative hash of its number, then the next slots in turn
[[Knuth1998]](references.md#knuth1998) (Section 6.4), a compare-and-swap
claiming an empty slot (two threads that find one empty slot for the same
particle take it once; one for another particle tries the next). It adds
its values to the slot with an addition in block memory. After a second
barrier each slot that holds a particle adds its sums to each destination
with one global atomic for each component. The threads past the last
tuple of the run reach both barriers.

A search gives up after 8 slots, and the thread adds that value to the
destination with an atomic of its own. Without this bound the search grows
without limit where a block's tuples share few particles: one block of JAC
holds 494 particles in its 512 slots, and the kernel took 42 µs against
12 before the change; with the bound 13.2.

**What it changes, and what not.** The values reach the destination in
another order and partly summed in f32 within the block, so the forces
change by the rounding of their sums, as they change from one run to the
next with atomics. The deterministic mode has no loops evaluated once and
is unchanged. The loops with global sums (the steps that write energies)
keep their atomics, and so does every loop of a run that holds the forces
of a term over centers, whose block computes the sums of the centers
(D194). The compile takes up to 0.2 s longer (5.3 to 11 s on the suite).

**The strategy, its preconditions, and the choice.** The table is the
strategy `hash(S, P, L)` of the privatized scatter-accumulate, which
`md-exec-choose-scatter` sets on each loop that can take it:
$S$ = 512 slots for a run with dihedrals, `P` = 8 slots tried, and
$L$ = 100,000 tuples, below which the host launches the kernel with the
atomics (JAC, 55,000 tuples, ran 1.5% slower with the table). The
preconditions, the block memory and the occupancy, the measurements behind
the bound, and the switch `MDIR_TUPLE_SUMS` are in
[device-primitives.md](device-primitives.md), Section 2
(D285).

**Validated** (one RTX 4090, the settings of the Amber suite, one step;
the same binary with and without the switch; the force differences
relative to the RMS force):

| System | Mixed: block sums against atomics, RMS / largest | Mixed: atomics against atomics, RMS / largest | Double: RMS / largest | Deterministic mode against `main` |
|---|---|---|---|---|
| JAC | 2.6e-7 / 7.6e-6 | 9.8e-7 / 1.2e-4 | 3.8e-16 / 2.1e-14 | the same bits |
| Factor IX | 1.0e-6 / 2.2e-4 | 9.7e-7 / 2.2e-4 | 1.2e-15 / 2.8e-13 | the same bits |
| Cellulose | 8.8e-7 / 2.4e-4 | 1.1e-6 / 3.0e-4 | 1.0e-15 / 3.4e-13 | the same bits |

The potential energy at step 0 agrees to 1.4 × 10⁻¹⁰ of its value and the
virial to 10⁻⁸ in mixed precision, and to every printed digit in double.
STMV, after 25 steps (its interval of coupling), differs by 4.3 × 10⁻⁴ of
the RMS force between the two ways and by 4.2 × 10⁻⁴ between two runs
with atomics: the trajectories part as two runs with atomics do.

**Measured: the kernel** (the same binary with and without the switch,
alternated twice in one job, before the bound of 100,000 tuples; µs a step
of the tuple kernel from Nsight Systems, and ms a step):

| System | RTX 4090 (ada-003): atomics | Block sums | Step | RTX PRO 6000 (ada-004): atomics | Block sums | Step |
|---|---:|---:|---|---:|---:|---|
| JAC NVE | 12.1 | 13.2 | 0.14 → 0.14 | 6.8 | 6.8 (atomics, below the bound) | 0.12 → 0.12 |
| Factor IX NVE | 24.8 | 16.4 | 0.34 → 0.34 | 17.0 | 12.3 | 0.25 → 0.25 |
| Cellulose NVE | 220.8 | 84.4 | 1.49 → 1.36 | 151.5 | 81.0 | 1.19 → 1.15 |
| STMV NPT 4 fs | 439.1 | 189.4 | 4.89 to 5.10 → 4.65 | 254.1 | 160.3 | 3.88 → 3.82 |

**Measured: the Amber suite** (`scripts/benchmarks/amber/bench.py`, the
settings of the suite with a dual list of groups, skins 3 and 0.6 Å; `main`
and this decision alternated system by system in one job, three rounds,
the median in ns/day):

| System | RTX 4090 (ada-003): `main` | This decision | Ratio | RTX PRO 6000 (ada-004): `main` | This decision | Ratio |
|---|---:|---:|---:|---:|---:|---:|
| JAC NVE | 1311.8 | 1311.8 | 1.000 | 1506.8 | 1509.4 | 1.002 |
| JAC NVE 4 fs | 2451.1 | 2449.7 | 0.999 | 2822.3 | 2822.2 | 1.000 |
| JAC NPT | 1185.9 | 1185.8 | 1.000 | 1338.3 | 1339.5 | 1.001 |
| JAC NPT 4 fs | 2286.5 | 2286.1 | 1.000 | 2618.8 | 2618.8 | 1.000 |
| Factor IX NVE | 528.8 | 543.1 | 1.027 | 713.4 | 721.5 | 1.011 |
| Factor IX NPT | 495.8 | 508.3 | 1.025 | 660.2 | 669.1 | 1.013 |
| Cellulose NVE | 120.9 | 133.4 | 1.103 | 158.9 | 164.7 | 1.037 |
| Cellulose NPT | 116.8 | 127.2 | 1.089 | 152.5 | 157.7 | 1.034 |
| STMV NPT 4 fs | 74.5 | 76.6 | 1.028 | 99.3 | 101.3 | 1.020 |

STMV on the RTX 4090 varied in that job by 4% from round to round (other
jobs ran on the node); in a job before the bound, which does not change
STMV, it ran at 74.6 against 78.4 ns/day in three rounds within 0.3%
(1.051). On the RTX PRO 6000 the kernel of the tuples took less time than
on the RTX 4090 before the change (151 against 221 µs on Cellulose), and
the change gains less there.

**Considered and not done here.**

- *Parameters stored in f32.* Every load of a parameter of a tuple in f64
  is narrowed to f32 at once in the mixed mode, so parameters stored in
  f32 would give the same bits. Measured with the block sums by reading
  half the bytes (wrong values, the first launch of a run): Cellulose 85.5
  against 89.9 µs, STMV 196 against 200.5. About 4 µs a step on Cellulose,
  0.3% of a step, for a change of the arguments of every entry; not taken.
- *The excluded pairs in the loop over pairs.* Their cost was their
  atomics, which the block sums now share with the other sets; the masks of
  the groups clear an excluded pair and an empty place alike (neighbor-groups.md,
  Section 2), so the loop over pairs would need a second mask.
- *The tuples in the order of their first member.* A unit of up to 128
  particles would then leave 0.10 particles a member on Cellulose instead
  of the block's 0.237, but the order of the tuples is that of the files,
  which the energies of the terms and their fields follow. A follow-up.
