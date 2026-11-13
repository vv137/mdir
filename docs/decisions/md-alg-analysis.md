# D[md-alg-analysis]

Recorded at its merge.

**One analysis of the collections of `md_alg`, shared by every later collection-level transformation, records for each collection that an `md_alg` op produces its ordering, access pattern, cardinality, effects and aliases, lifetime, and determinism requirement, and propagates the order that its consumers require back to its producer as a small specification value (membership, grouped by a key, sorted within a group by a key, stable, or the full layout), so that a sort is found redundant only when every consumer's requirement is met without it; any consumer that is not a declared `md_alg` reader requires the full layout. It changes no IR and no PTX.** Issue #866 (part 2, stage A of the maintainer's item 3), part of #558.

*Design draft for review; the sections marked open are questions for the
leader and the maintainer.*

Prior systems keep the collection-level form so that fusion and the
removal of intermediate buffers can be judged on it
[[Steuwer2017]](../references.md#steuwer2017),
[[Henriksen2017]](../references.md#henriksen2017). D408 kept the ops in
the module up to `md-alg-transform-collections`; this step gives that
place one analysis, which every transformation queries instead of judging
legality on its own (the maintainer's item 1 on #866). It works on the
memrefs of today (stage A): the ops declare the effect of each operand,
and the analysis follows producers to consumers through them. Nothing is
transformed.

## The collections

A collection is a buffer that an `md_alg` op writes, named by the op and
the role of the operand:

| Producer | Collections |
|---|---|
| `scan` | `start`, `cursor` |
| `scatter_to_bins` | `out` (the order), `cursor` |
| `sort_bins` | `items` in place, or `out` |
| `sort_by_key` | `keys`, `counts`, `start`, `cursor`, `out`, and the side outputs `writes`, `adds` |
| `append` | `into`, `units`, `ordinals`, the pool counter |

A buffer is followed through its views (`memref.memory_space_cast`,
`memref.cast`, `memref.subview`, `memref.reinterpret_cast`) to its root:
a `gpu.alloc` or `memref.alloc` of the function, or an argument of the
function. Every use of every alias of the root is a use of the collection.

For each collection the analysis records:

- **ordering**: what the producer guarantees (below), and what the
  consumers require;
- **access pattern** of each consumer: *streamed* (an `md_alg` op that
  reads the whole collection in index order: a scan of counts, a sort of
  bins), *gathered by key* (an accessor at a place that another
  collection gives: `bin_item`, `bin_span`, `ragged_entry`), or *random*
  (any other read, by index arithmetic);
- **cardinality**: *exact* (a constant), *bound* (at most an index value
  of the op: `n` items in the order of a counting sort, of which those
  with no bin are absent), or *run-time* (an index value: `bins + 1`
  offsets);
- **effects and aliases**: the producer's declared effect on the buffer
  (`MemoryEffectOpInterface`), the other writers, the aliases, and whether
  the buffer escapes (below);
- **lifetime**: the allocation, the first and the last use, and the
  release, by the position of each op in a pre-order walk of the
  function (the host's order of its launches); a use in a loop extends to
  the end of the loop;
- **determinism**: the producer's requirement (`getRequiredDeterminism`,
  D408).

Ordering and determinism are separate axes (the maintainer's item 1): the
order is what a consumer can observe of the positions, the determinism
what it can observe of the bits from run to run. The verdict on a sort
below is about the order alone; whether eliding it is allowed also
depends on the determinism requirement, which the transformation (#869)
combines with it.

## The order requirement

A requirement on a collection of items in bins is one of:

| Requirement | Meaning |
|---|---|
| `membership` | which items are in the collection |
| `grouped_by(G)` | the items of each group of `G` are contiguous, at the places that the offsets of `G` give |
| `sorted_within(G, K)` | and within each group ascending by the key `K`; ties in any order |
| `stable(G, K)` | and ties in the order of the items: every position fixed |
| `layout` | every element of the buffer, inside the bins or not, and every alias of it: what a reader by index arithmetic sees |

The key identities are part of the value: a group `G` is named by the
root of the offsets (the `start`) that define it, and a sort key `K` by
the root of its `by` buffer, or `item` for the item itself.
`grouped_by(cell)` and `grouped_by(class)` are different values.

They form a lattice, ordered by strength:

$$\mathrm{membership} \sqsubset \mathrm{grouped\_by}(G) \sqsubset \mathrm{sorted\_within}(G, K) \sqsubset \mathrm{stable}(G, K) \sqsubset \mathrm{layout}$$

where two requirements are comparable only if their keys agree. The
requirement of a collection is the join over its consumers:

- equal kinds and keys, or one weaker on a prefix of the other's keys
  (`grouped_by(G)` and `sorted_within(G, K)`): the stronger;
- keys that differ (`grouped_by(G_1)` and `grouped_by(G_2)`,
  `sorted_within(G, K_1)` and `sorted_within(G, K_2)`): `layout`. No
  requirement short of the layout states both, and the analysis does not
  know whether one key refines the other;
- no consumer: none (bottom).

What each producer guarantees for its order:

| Producer | Guarantees |
|---|---|
| `sort_by_key`, `order stable` | `stable(G, K)`, `G` its `start`, `K` its `by` or `item` |
| `sort_by_key`, `order unordered`; `scatter_to_bins` | `grouped_by(G)` |
| `sort_bins` | `stable(G, K)`, `G` its `start`, `K` its `by` or `item` |
| any other | `layout` of its own bytes, nothing about bins |

A requirement `r` is **met** by a guarantee `g` when `r ⊑ g`. **A sort is
redundant** (`sort_by_key` with `order stable`, or `sort_bins`) when the
join of its consumers' requirements on its result is met by what the
result would be without the sort: `grouped_by(G)` for a counting sort,
the guarantee of the input for a sort of bins. A requirement that the
sort does not meet either (a key that does not match) is reported as
such: the sort is not redundant, and not sufficient.

The requirement of each consumer:

| Consumer | Requires of the collection it reads |
|---|---|
| `bin_item %out[...]` with a declared requirement (open question 1) | that requirement |
| `bin_item` without one | `layout`: it reads the item at a place |
| `bin_span %start[...]` | `layout` of the offsets (they are not sorted; no sort acts on them) |
| `sort_bins` reading `items` by the item, or with `keys` (`stable_rank`, the items of a bin distinct) | `grouped_by(G)`: it orders each bin itself |
| `sort_bins` reading `items` with `by` | `stable(G, K_in)` of its input: ties keep the input's order |
| `scan`, `scatter_to_bins`, `sort_by_key` reading a buffer (`counts`, `keys`, `by`) | `layout`: they read it by index |
| `gpu.dealloc`, `memref.dealloc` | none: the end of the lifetime |
| a view (`memory_space_cast`, `cast`, `subview`, `reinterpret_cast`) | none of its own: its uses are followed |
| anything else | `layout` |

## The conservative rules

An unknown use never makes a sort look redundant. Each rule is a negative
test:

1. A read by any op that is not an `md_alg` op (a `memref.load` in a
   launch, a pair kernel that reads a row by index arithmetic) requires
   `layout`.
2. A write by an op other than the producer, which is not an `md_alg` op
   writing its declared output, requires `layout`.
3. The buffer escapes, and requires `layout`, when its root is an
   argument of the function, or an alias of it is an operand of a
   `func.call` or `func.return`, is stored to memory, has its address
   taken (`memref.extract_aligned_pointer_as_index`), or is forwarded by an
   op that is not one of the views above.
4. An op with no `MemoryEffectOpInterface`, or one whose effects on the
   buffer are unknown, is rule 3.
5. A use in a region that the walk cannot place (an op without the
   interface that holds regions) is rule 3.
6. Keys are compared by root; two different roots are different keys,
   even where the buffers hold the same values.

## Why not the sparse dataflow framework

On memrefs the order flows through memory: the producer writes a buffer,
and the consumers read it through other SSA values (views) and inside
the bodies of launches. MLIR's sparse backward analysis
(`SparseBackwardDataFlowAnalysis`) propagates lattices from the uses of
SSA values to the operands that define them, which is the flow of stage
B, where `sort_by_key` and `append` give a logical result: the same
lattice then becomes its lattice element. In stage A a walk from each
producer over the aliases of its buffer, classified by
`MemoryEffectOpInterface::getEffectsOnValue`, is the whole analysis. It is
an MLIR analysis (constructed from an `Operation *`, held by the
`AnalysisManager`), which the slot pass and the test pass query by
`getAnalysis`.

## Open questions

1. **How a reader declares less than the layout.** Today the only reader of
   the order of a counting sort is `bin_item`, which reads the item at a
   place; what the reader does with it is not in the IR. Without a
   declaration no reader requires less than `layout`, and no sort is ever
   redundant. Proposed: optional operands on `bin_item` that state what it
   observes, by key:
   `md_alg.bin_item %out[%place] observes grouped(%start)`,
   `observes sorted(%start, %by)`, `observes stable(%start, %by)`, or
   `observes membership`. Absent, it is `layout`, so every op of `main`
   keeps its meaning and no sort of `main` is redundant. The operands are
   SSA buffers, so that the key identity is the root, as in a producer.
   The alternative is a reader op of its own (`md_alg.bin_members`) that
   yields the items of a run of bins in no order; it is closer to stage B
   but is a new op with lowerings, for a step that transforms nothing.
2. The test pass is named `test-md-alg-print-collections`, by the
   convention of the test passes of `mdir-opt` (the issue's
   `-md-alg-print-collections`).

## What the programs of `main` give

Expected, from the IR at the slot of the dodecahedron with groups and
with the matrix (to be written in the test):

- `mdrt_gpu_matrix_sort`: the scatter writes its order into an argument
  (escape), and the stable rank's order is read by `memref.load` in a
  launch (rule 1): `layout`, not redundant.
- `mdrt_gpu_spatial_order`: the counting sort writes its order into an
  argument (escape): `layout`, not redundant.
- The groups build: its two counting sorts write local buffers read only
  by `bin_item`, which declares nothing: `layout`, not redundant. Their
  readers form groups of consecutive particles, so their order is
  observed: the result is the right one, not only a conservative one.

No sort of `main` is redundant.

## The same programs

The slot pass runs the analysis and changes nothing; it adds no attribute.
The lowered programs are compared byte for byte with `main` by the method
of D406 (docs/device-primitives.md, Section 4.5) on a subset of its 60
programs (to be listed).

## Tests

(To be written: the requirement kinds, a key that does not match, an
accessor-only consumer, a raw consumer, an escaping buffer, two consumers
whose requirements join, the programs of `main` at the slot, and the
mutations that the tests catch.)
