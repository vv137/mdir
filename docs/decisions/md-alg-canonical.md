# D[md-alg-canonical]

Recorded at its merge.

*Draft of 2026-11-14: the design, for review before the implementation.*

**The always-legal transformations of the `md_alg` layer: removal of a
dead collection, merging of a collection computed twice, and one device
allocation shared by two buffers whose lifetimes do not overlap. They run
in the slot `md-alg-transform-collections` (D408) and read the shared
analysis (D409). None changes an order or a floating-point result, and
the PTX of every kernel stays that of the base: only the host's
allocations, frees, and the arguments of its launches change.** Issue
#876, part of #866 (item 4, materialization by reuse) and #558.

The maintainer's principle (2026-11-14): optimization does the
transformations that are always legal first. Semantic ones (sort
elision, anything that changes an order) come after the IR carries the
semantics of its consumers, and are out of scope here.

## What fires on the programs of `main`

The census was taken at the slot (`mdir emit --stage=lowered` with the
pipeline cut after `md-alg-transform-collections`) on the dodecahedron
with groups and with the matrix, JAC NVE with groups, and STMV with
groups.

| Candidate (#876) | Evidence at the slot | Fires |
|---|---|---|
| A scan or a sort of zero items | every count is a `memref.dim` or a value computed from the box | no |
| A `grid_plan` with constant extents | its extents come from the box (`vector.extract` of an argument) | no |
| Accessors of a constant bin | the only constants are `entry_size` and `column` of the ragged accessors, and launch sizes | no |
| Side outputs nobody reads | the `writes` of the groups build's sorts (the wrapped positions, the places) and the `adds` (the sizes, copied to the host) are all read | no |
| A dead collection | every collection is read, released after a read, or escapes | no |
| A collection computed twice | no two collection ops of one function have the same operands; the three `grid_plan`s of the groups build differ | no |
| Two buffers whose lifetimes do not overlap | the groups build: the seven buffers of its first sort are freed before the second sort's first use, but all fourteen are allocated at the top | **yes** |

So no fold is added (the rule of #876: only patterns that fire or that a
named later step needs). Dead-collection removal and merging fire on no
program of `main`; see the question below.

## 1. Dead collections

An `md_alg` op with memory effects is removed when:

- none of its results has a use;
- every collection it writes (D409, every role) has no use after it
  other than a release (`gpu.dealloc`, `memref.dealloc`) or a shape
  (`memref.dim`), and, where the op stands in a loop, no use before it
  either (D409's rule 6);
- no collection it writes escapes (D409's rules 3 and 4: an argument of
  the function, a call, a return, a store of the buffer, an unfollowed
  view, an op without `MemoryEffectOpInterface`);
- every buffer it writes that is not a collection of the analysis (a
  scan's scratch) is local, and is only released afterwards.

A read through a followed view (a subview) is a read of the root, so it
blocks the removal. An allocation whose only uses afterwards are its
release and the casts to it is removed with its release.

## 2. Common collections

Op B is removed when an op A before it in the same block:

- is the same op, with the same operands (the same SSA values), the same
  attributes, and equivalent regions (`OperationEquivalence`);
- reads none of the buffers it writes (an operand both read and written,
  such as the cursor of `scatter_to_bins` or the counts of a
  `sort_by_key` with `cleared`, makes the op not idempotent). The scratch
  of the execution group is exempt: the result does not depend on it
  (D406);
- and no op between A and B writes any operand of A or any alias of one
  (by `MemoryEffectOpInterface`; an op with unknown effects, a call, or a
  launch whose body writes a root of an operand counts as a write).

Then B recomputes what A wrote into the same buffers, and is erased. Two
ops that write the same values into different buffers are not merged
here: that is a buffer replacement, which needs the lifetimes of
Section 3, and is left for the materialization step (#866, item 4).

The walk is the pass's own, not `canonicalize` or `cse`: those would
also rewrite the bodies of the `gpu.launch` ops still in the module, and
the PTX must not change.

## 3. Buffer reuse by lifetime

### The lifetime model

At the slot the host's code is synchronous: `gpu.alloc`, `gpu.dealloc`,
and `gpu.launch` have no async tokens, and the runtime issues every launch,
copy, and free on one stream in the order of the host
(`runtime/mdrt_cuda.c`). Its pool hands a freed block to the next
allocation without waiting, which that order makes safe. The exceptions
are the async ops (`gpu.memcpy` and `gpu.wait` with tokens, such as the
copy of the groups build's sizes to the host) and the side stream of the
reciprocal sum (`mdrtSideBegin`, `mdrtSideEnd`, `mdrtSideJoin`; and
`mdrtUseStream`, `mdrtRestoreStream`).

The lifetime of a buffer is the interval of positions, in the pre-order
walk of D409, from its first use (any use of any alias of its root,
including the uses before its producer and in the bodies of launches) to
its release. A use in a loop extends to the end of the loop.

Buffer B reuses the allocation of buffer A when all of these hold:

1. both are `gpu.alloc` without an async token, in the same block, each
   with exactly one release (a `gpu.dealloc` without a token) in that
   block;
2. neither escapes (D409's rules 3 and 4);
3. the release of A is before the first use of B;
4. no use of A is an async op whose token is not waited (a `gpu.wait`
   with no result, through the chain of tokens) before the first use of
   B;
5. the function calls no function that changes the stream (the five
   above) and no external function that the pass does not know, anywhere.
   This is coarser than it needs to be, and it is on purpose: no program
   of `main` that has a reuse has such a call;
6. the element types and the ranks are the same, and the static sizes are
   equal.

When unsure, it does not reuse.

### The rewrite

One allocation takes the place of both, where A's stands. Its size is
the larger of the two (`arith.maxui` of the dynamic sizes). This requires
B's sizes to be defined before A's allocation, as in the groups build;
where the sizes are the same values, no maximum is computed. A and B
become `memref.reinterpret_cast`s of it to their own sizes, with offset 0
and unit strides, so every `memref.dim` and every argument of a launch
keeps its value except the pointer. A's release is erased, and B's
releases the shared allocation. The analysis runs again before the next
pair is taken, and each allocation is shared at most once per pass
(greedy, in the order of the allocations: no packing).

The groups build gives the pairs, in the order of the walk:

| First sort (freed) | Second sort (allocated) | Size |
|---|---|---|
| `key` (n) | `gkey` (capacity) | the larger: capacity |
| `keyCount` (keys) | `gcount` (cells) | the larger |
| `keyStart` (keys + 1) | `gstart` (cells + 1) | the larger |
| `keyCursor` (keys) | `gcursor` (cells) | the larger |
| `sorted` (n) | `gplace` (n) | n, no maximum |

`chunkBase` and `columnChunks` have no partner of their type left.

### The placement: the slot, not after `convert-md-alg-to-gpu`

- The SIMT lowering allocates and frees nothing (`MDAlgLowering.cpp` has
  no allocation): the buffers and their releases are the same before and
  after it.
- At the slot the analysis of D409 already follows the views to the
  roots, applies its escape rules, and places each use and release in the
  host's order. After the lowering the same facts would have to be
  recovered from the bodies of the launches.
- The candidates are the collections, which the analysis names; the
  buffers of `mdir_run`, with their own rules of release
  (`StageBodies.cpp`), are not touched.

The stream and token rules are those of the GPU's host code; at the slot
that code is already the GPU's, so they apply there.

## How the result is checked

- **The same device code.** By D406's method (`device-primitives.md`,
  Section 4.5): `mdir emit --stage=lowered` with `MDIR_GPU_BINARY=ptx
  MDIR_GPU_ARCH=sm_89 MDIR_COMPILE_CACHE=off MDIR_SKIN_CACHE=off
  MDIR_SKIN_SEARCH=never`, against the base, on D409's nine programs, the
  prolate kernel of PME on the dodecahedron, and STMV. The PTX of every
  kernel must be byte-identical; the host's code may differ only in its
  allocations, frees, and the arguments of its launches, and one
  program's host diff is shown.
- **Peak device memory**, measured the same way before and after: `mdir
  run` of 200 steps with `MDIR_DEVICE_MEMORY=2` on one RTX 5090, the
  peak in use and the peak with the pool, in bytes from the runtime's
  list of blocks at each peak (the dodecahedron's saving is below the
  0.1 MiB of the summary line). The programs: the dodecahedron with groups
  and with the matrix, JAC NVE with groups, and STMV with groups. At the
  base the peaks in use are 1.1, 3.1, 30.3, and 1,388.4 MiB; at STMV the
  groups build's buffers are at the peak (the blocks of n and of the
  capacity in i32 are in the list). The expected saving is about $n +
  \min(n, c)$ i32, which is 8.5 MB (0.6%) at STMV and 0.2 MB at JAC; the
  matrix programs have no pair. A chart in the PR.
- **Tests**, each pattern with a positive and a negative case:
  - dead collection: removed; kept for an escape (an argument, a call, a
    return), a read through a subview, a read in a launch, and a use
    before the producer in a loop;
  - merging: merged; kept for a write in between (an `md_alg` op, a
    launch, a store through a view), a different operand or attribute,
    and an operand both read and written;
  - reuse: shared with and without a maximum; kept for overlapping
    lifetimes, a use by a `gpu.memcpy` whose token is waited only after
    B's first use, a call to `mdrtSideBegin`, an escape, a second
    release, and a size that is defined after A's allocation.
- **Mutations**, each applied alone and reverted: a reuse that ignores the
  token (rule 4), a reuse that ignores the stream (rule 5), and a merge
  across an intervening write. Each must fail a test.
- **Sanitizer tier** (memcheck and racecheck) on a device with the CUDA
  toolkit, since the shared allocations change the device's memory use;
  one full suite at the final head; CI.
