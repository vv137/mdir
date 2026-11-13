# D[md-alg-ir]

Recorded 2026-11-13.

**The collection ops of `md_alg` (D406, D407) survive as IR between passes: `convert-md-exec-to-gpu` leaves the ops of the neighbor matrix's template and of the groups build in the module, an empty pass `md-alg-transform-collections` holds the place of the collection-level transformations, and `convert-md-alg-to-gpu` lowers the ops after it; each op with an ordering or a floating-point aspect states a determinism requirement (`unordered`, `reproducible`, `bitwise`), which its verifier checks against its contract; and the execution parameters each op was lowered with are recorded in the IR, the run log, and the manifest. The PTX of every kernel stays that of the base.** Issue #866 (part 1), part of #558.

Prior systems need the collection-level form to survive between passes:
once the patterns are lowered to loops, the information that fusion and
the removal of intermediate buffers need is gone
[[Steuwer2017]](../references.md#steuwer2017),
[[Henriksen2017]](../references.md#henriksen2017). Until this step the ops
were created and lowered inside one pass, so no pass saw them.

## The cut

| Where the ops come from | Where they are lowered | Why |
|---|---|---|
| The neighbor matrix's template (`NeighborsMatrixGPU.mlir`: the scans, the scatters, the sorts within cells, the spatial order's `sort_by_key`, the grid plan, the ragged rows) | `convert-md-alg-to-gpu` | The template is parsed and instantiated for its type in `convert-md-exec-to-gpu`, so its ops are typed when they reach the module; nothing in the pass reads the template's code after it is added |
| The groups build (`GroupsBuild.cpp`: its two `sort_by_key`, four `append`, two `grid_plan`, and the readers of bins) | `convert-md-alg-to-gpu` | As the template: emitted into a module of its own and moved into the program's module whole |
| The templates of PME (Gaussian, triclinic, prolate: `wrap` and `grid_read` alone) | where the template is instantiated, in `convert-md-exec-to-gpu`, as before | After their ops are lowered the templates' loops over the order are unrolled and canonicalized, and the arrays of the splines promoted to values: that rewrite folds the accessors' arithmetic, so lowering them later would change the code. The accessors compute one value for one item and have no order and no collection: nothing for a collection-level pass to transform |
| The device build of ragged rows (`emitDeviceRaggedRows`, the `sort_by_key` of the incidence structures of D257) | where it is created, in `convert-md-exec-to-gpu`, as before | It is built in the program's own function, whose lowering then takes its launches: it names each kernel by a counter in the order of the lowering (D354), and moves constants and vectors into them. Lowered later, its kernels would have no names when the function's lowering names the next ones, and every later kernel would change its name |

The arithmetic that `convert-md-exec-to-gpu` finishes in its kernels after
they are written (the contraction of products and sums, the explicit fused
multiply-adds of the deterministic mode, #102, and the approximate
reciprocal of a division in f32, D98) also covers the regions of the
`md_alg` ops that are outside a kernel: the key region of a
`sort_by_key`, whose body becomes the key kernel. That region's
arithmetic was in a kernel when the walks ran before; without this, the
positions that the key kernels compute would lose their fused
multiply-adds. The SIMT lowering itself writes no floating-point
arithmetic in a kernel (the grid plan is host code).

The split of `Lowering` into neighbors and interactions (#558) is not
needed for the cut: the ops that survive come from code that the pass adds
whole (a template, the groups build), not from the lowering of the
program's functions.

## Pass order

```
convert-md-exec-to-gpu          the program's kernels; the matrix template's
                                and the groups build's md_alg ops stay
md-alg-transform-collections    empty: where fusion, the elision of a sort
                                whose order no consumer needs, and the
                                removal of intermediate buffers will go
convert-md-alg-to-gpu           each op replaced by its emitter's code where
                                it stands, its policy recorded
check-read-only-arguments
mdir-gpu-lower-to-nvvm
```

`md-alg-transform-collections` verifies the ops and changes nothing. The
transformations it will hold read the shared collection analysis of
#866 (ordering, access pattern, cardinality, effects, lifetime,
determinism), which comes first; none is implemented here.

## Determinism requirements

Each op with an ordering or a floating-point aspect takes an optional
attribute `determinism`, a requirement on the bits of its results:

- `unordered`: the results that its contract leaves free (the order of the
  items within a bin, the numbers of blocks) may differ from run to run;
- `reproducible`: every result is the same from run to run under a fixed
  policy (the execution parameters) and target;
- `bitwise`: every result is that of the serial definition, on every
  target and under every policy.

It is a separate axis from the ordering contract `order`
(the maintainer's comment on #866): an unordered collection can be
reproducible under a fixed schedule, and a sorted one can lose bitwise
equality through a reordered floating-point reduction. The tiers are
ordered, `unordered` < `reproducible` < `bitwise`; a consumer that needs
less states less, which lets a later pass change what the requirement
does not fix.

| Op | Strongest tier (the verifier refuses more) | Default (no attribute) | SIMT lowering |
|---|---|---|---|
| `scan` | `bitwise` (an integer scan, exact) | `bitwise` | meets every tier |
| `scatter_to_bins` | `reproducible`: `bitwise` contradicts its contract, which leaves the order within a bin free | `unordered` | `unordered` only: the cursor's atomics order the items |
| `sort_bins` | `bitwise` (a stable sort) | `bitwise` | meets every tier |
| `sort_by_key`, `order stable` | `bitwise` | `bitwise` | meets every tier |
| `sort_by_key`, `order unordered` | `reproducible` | `unordered` | `unordered` only |
| `append` | `reproducible`: `bitwise` contradicts its contract, which leaves the numbers of the blocks free | `unordered` | `unordered` only: concurrent lists take blocks in no fixed order |
| `grid_plan` | `bitwise` (one lowering, on the host, shared by both) | `bitwise` | meets every tier |

The accessors (`ragged_row`, `ragged_entry`, `wrap`, `grid_read`,
`bin_span`, `bin_item`) and the bins of a key region compute one value for
one item, exactly, and take no attribute.

The defaults are what each op's lowerings give today, so every program
keeps its meaning, and no op of `main` states the attribute. The
verifier refuses a requirement that the op's own contract rules out; the
SIMT lowering refuses one that its implementation cannot meet (a
deterministic scatter or append exists, for example the serial one, but
not on the device yet), as it refuses an op without execution parameters.
Where an op has more than one implementation that meets its requirement,
the cheapest is to be chosen; no op has more than one today.

## The recorded policy

`convert-md-alg-to-gpu` and the in-place lowerings of `convert-md-exec-to-gpu`
add, for each op with a determinism aspect that they lower, an entry to the
module's attribute `md_alg.policy`: the function, the op, its
requirement, its order, its strategies, and the constant sizes of its
launches' blocks. `mdir run` takes the attribute from the module once the
passes have run (so that `mdir emit --stage=lowered` prints the module of
the base), prints a line `MDIR: md_alg policy:` in the log, and writes the
entries to the manifest as `effective.md_alg_policy`
(`docs/run-manifest.schema.json`), next to the skin search's record
(D401). A test that compares two runs bitwise compares their policies
first.
