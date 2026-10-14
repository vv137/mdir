# The structures of the tuples built on the device (D[device-tuples], proposed)

Issue #295; the design, for review before it is implemented. It builds on
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
the width of the structure, the largest count, does not depend on the
order (D253 relies on this already).

The index of the tuples of each particle in the order of the files (the
`MdrtTupleIndex` of `mdrt_tuples.h`: for particle $q$ the entries
$(t, s)$ from $\mathrm{first}_q$ to $\mathrm{first}_{q+1}$, in the order of
the tuples) is therefore a function of the members alone. It is built once
for a tuple set, on the host, where it is built now, and uploaded once. A
start in a new order then needs no count, no prefix sum, no atomic, and no
sort on the device: three loops in which every element is written by
exactly one thread.

**What is borrowed.** Nothing new. The kernels are gathers and a scatter
through a permutation. The one construction of the usual kind is the
index itself, a count of the tuples of each particle, a prefix sum of the
counts, and a fill in the order of the tuples (the compressed rows of a
sparse matrix, built by distribution counting); it exists already on the
host (`mdrtTupleIndexBuild`, D253) and is not moved to the device.

## 3. The kernels

For each tuple set whose structures are to be built, the lowering
(`convert-md-exec-to-gpu`) launches, with the existing `launchOver`:

1. **place**, one thread a place $p$: $\mathrm{place}[\mathrm{id}_p] = p$
   (a scatter through a permutation: each element written once);
2. **renumber**, one thread an element of the members:
   $m'_i = \mathrm{place}[m_i]$, into the renumbered members on the
   device;
3. **rows**, one thread a pair (place $p$, slot $k$) over $N \times W_k$,
   $W_k$ the largest count (one launch over the product, the index
   decomposed in the kernel): with $q = \mathrm{id}_p$ and
   $c = \mathrm{first}_{q+1} - \mathrm{first}_q$, the thread of $k = 0$
   writes $c$ at the head of the row, and a thread with $k < c$ reads
   $(t, s)$ from entry $\mathrm{first}_q + k$ and writes $t$, $s$, and
   $m'_{t,0..a-1}$ at $1 + k(a + 2)$. Threads with $k \ge c$ write nothing,
   as the host writes nothing past the tuples of a row.

**Determinism.** No element is written by two threads and no element is
read after it is written in the same kernel, so the structure does not
depend on the order in which threads run: its rows hold the tuples of each
particle in the order of the tuples, as the host's do, and every element
that a loop over tuples reads equals the host's, element for element. The
same kernels run in and outside the deterministic mode.

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
- **A start in a new order** launches the three kernels of Section 3 into
  the memory of the entry; nothing is copied between the host and the
  device but the flag.
- **The call protocol** (one call before, one after, for each set): the
  code calls the store with the members and the identities and gets back
  the buffers of the entry (the members and the index on the device, the
  renumbered members, the structure) and whether to build; it launches the
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
- The index of the tuples of each particle in the order of the files: it
  is a function of the members alone, so it is kept once for each tuple
  set (not for each entry), built on the host as now, in $O(T a)$, and
  uploaded once with the members, for the whole simulation or run, and
  again only if the members change. Building it on the device would need a
  count, a prefix sum, and a fill in the order of the tuples, for something
  that is built once.
- The decisions of the store (which entry, take or build), and the check.
- The renumbered members are no longer needed on the host: the only
  consumers of `md_exec.renumber` on a GPU are `md_exec.build_incidence`
  and the copy for the loops over tuples once (`md-exec-assign-storage`
  keeps relations on the host for that reason alone). The pass that
  lowers them is the one that changes; the storage form does not.

## 5. Memory on the device

For each tuple set, new on the device and shared by its entries: the
members in the order of the files (4 bytes an element) and the index
(8 bytes an element, and 4 bytes a particle and one). For each entry, in
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
(b12ee87) and the prototype in turn six times in one job on one RTX 4090,
mixed precision; the least of five starts, ms, the medians of the five
rounds after the first:

| System | main | Prototype | Of which the functions of the store |
|---|---|---|---|
| JAC | 5.25 | 1.92 | 3.3 → 0.07 |
| Factor IX | 18.0 | 5.5 | 12.2 → 0.17 |
| Cellulose | 94.8 | 25.6 | 70.4 → 1.75 |

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
- **The order of the stream.** The comparison, the three kernels, the
  copy of the identities into the store (`cuMemcpyDtoDAsync`), and every
  kernel of the program run on the one stream of the runtime. The
  renumbered members that the loops over tuples once read are therefore
  complete before the first of those kernels, as the rows are before the
  loops over the incidence structure; no wait is needed.
- **The deterministic mode** has no loops that evaluate each tuple once
  (D84), so nothing reads the renumbered members there but the kernel of
  the rows, on the same stream. The kernels are the same in both modes,
  and none uses an atomic: the flag of the comparison is written with the
  same value by every thread that writes it.

## 6. The check, and the CPU

- **`MDRT_CHECK_KEPT_TUPLES`** (D253) is reused: with it set, every
  structure that a start takes is also built on the host and compared with
  the one read back from the device, as now, and every structure that the
  kernels build is compared in the same way when the code says it is
  built. A difference ends the program and names the structure. No new
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
- **Sorting the pairs (particle, tuple, slot) by key on the device** at
  every start: correct and deterministic with a stable sort, but a sort of
  $T a$ keys where a gather suffices, and a sort that the lowering would
  have to emit.
- **Keeping the identities' comparison on the host**, reading them back
  once a start (1.6 MB on Cellulose) instead of once a set: simpler, and
  0.1 to 0.7 ms; it leaves a copy where the issue asks for none
  (a question for the maintainer).
- **Kernels in the runtime library** (PTX compiled with the toolkit at
  the build of MDIR): a second way of making kernels, which the build does
  not have; the lowering already emits every kernel.
