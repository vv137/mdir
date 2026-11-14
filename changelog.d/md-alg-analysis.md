### Added

- **The shared analysis of the `md_alg` collections** (D409, #866, part 2). For each buffer that an `md_alg` op produces, the analysis records:
  - its ordering;
  - its access pattern (streamed, gathered, or random);
  - its cardinality (exact, bound, or run-time);
  - its effects, aliases, and escapes;
  - its lifetime;
  - its determinism requirement.

  It propagates the order that the consumers require back to the producer: membership, grouped by a key, sorted within a group by a key, stable, or the full layout, with the key identity part of the value. A sort is found redundant only when every consumer's order is met without it and every consumer combines exactly or requires `unordered`. Any use that is not a declared reader (index arithmetic, a call, an escape, an op of unknown effects) requires the full layout. `md-alg-transform-collections` runs the analysis and changes nothing. `mdir-opt --test-md-alg-print-collections` prints it. On the programs of `main` no sort is redundant: each is read by index or escapes. The PTX is unchanged.
- **`md_alg.bin_members`**, a reader of the bins of a counting sort that states the order of its visit: `unordered`, `sorted`, or `stable`, by a key. Under `unordered`, its region combines integers alone and writes only slots of its item and additions to counters, which the verifier enforces. The serial lowering's option `reverse-unordered-visits` reverses such a visit, so that a test can show the result does not depend on the order. Both lowerings expand it into the loop over the bin through `md_alg.bin_span` and `md_alg.bin_item`. No program uses it yet.
