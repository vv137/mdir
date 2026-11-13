# D[md-alg]

Recorded at its merge.

**A target-neutral dialect `md_alg` holds the collection algorithms of the device lowering as typed ops with verifiers, each defined by a serial lowering to `scf` and lowered for a GPU by a SIMT lowering that calls the existing emitter; step 1 replaces the 22 device primitives that the templates and the groups build called by name (`func.call @mdir.primitive.<name>`, checked by hand in `ExpandPrimitives.cpp`) with seven ops, the warp-level helpers with upstream ops and typed emitter calls, and leaves the IR after `convert-md-exec-to-gpu` and the PTX unchanged.** Issue #861, part of #558 and #604.

## The layers

```
md_exec   execution skeletons and physics (unchanged)
  |
md_alg    what is computed: scan, scatter to bins, sort within bins,
          the accessors of ragged rows and of a periodic grid;
          later: histogram, select/compact, sort by key, segmented
          reduce, append with a capacity
  |  later: a policy pass fills in the execution model and its
  |  parameters from the target record (#433, #558 item 4)
  +-> serial : scf loops; the definition of each op and the test oracle
  +-> SIMT   : GPU, through the emitters of CompactionEmitter.h,
               SinglePassScan.cpp, RaggedRows.h, and GatherEmitter.h
  +-> SIMD   : CPU vector lanes (D402), later
  +-> dataflow : later; nothing in md_alg precludes it
```

An op's meaning names no warp, lane, or block. Its attributes state
semantics and contracts: the ordering of its result and what it reads and
writes. The execution parameters of the SIMT lowering (the strategy of a
scan or a sort, the scratch buffer, and the launch geometry) are an optional
group of the op that the result does not depend on: the serial lowering
ignores it, and the SIMT lowering requires it. Until the policy pass exists,
the caller gives them, as it gave them to the named primitives.

Warp-level operations are not `md_alg` ops (#861). The SIMT code of the
matrix template writes them as the upstream ops that the expansion emitted
(`gpu.ballot`, `arith.andi` and `math.ctpop`, `arith.shli` and
`arith.subi`); the groups build, C++ code, calls the typed emitters
(`emitWarpBallot`, `emitWarpRank`, `emitWarpCount`, `emitLanesBelow`,
`emitWarpScan`; there is no upstream subgroup scan). The block scan had no
caller in IR; it stays an emitter inside the device scans.

## The ops of step 1

| Op | Replaces | Contract (against the serial lowering) |
|---|---|---|
| `md_alg.scan` | `device_scan`, `device_scan_cursor`, `single_pass_scan`, `single_pass_scan_cursor`, `chunked_scan` | ordered integer: bitwise |
| `md_alg.scatter_to_bins` | `cursor_scatter` | unordered: each bin's items as a multiset, after a canonical sort; the cursor afterwards and every element outside the bins bitwise |
| `md_alg.sort_bins` | `sort_bins`, `sort_bins_by`, `rank_in_bins` | stable sort: bitwise |
| `md_alg.ragged_row`, `md_alg.ragged_entry` | `ragged_row`, `ragged_entry` | per item: bitwise |
| `md_alg.wrap` | `wrapped_grid_axis_index`, `wrapped_grid_axis_i32` | per item: bitwise |
| `md_alg.grid_read` | `grid_gather_index`, `grid_gather_i32`, `grid_gather_masked_i32` | per item: bitwise |

The 22 names become seven ops, five typed emitter calls or upstream ops of
the warp, and one emitter without a name (the block scan). A scan has one
definition whatever its strategy, and a sort within bins one definition
whether it sorts in place by insertion or into another buffer by the
stable rank.

The contracts that the issue lists and that no op of step 1 has: a
fixed-order reduction (bitwise) and a reordered floating-point reduction
(within $2\gamma_{n-1}\sum_k \lvert t_k\rvert$) come with the reductions,
and the contract of an overflow of a capacity (which entries are written,
what the counter holds afterwards so that a retry can size the buffer, and
what a caller may read) with the append. The scatter has no capacity: the
spans of the bins in `out` are a precondition, which a scan of the counts
of the keys gives.

## Pass order

The ops are created and lowered inside `convert-md-exec-to-gpu`, where the
calls were expanded: in each template after it is parsed and instantiated
for its types, and in the groups build after it is emitted. The SIMT
lowering walks the ops, verifies all of them before it lowers any, and
replaces each with its emitter's code where it stands, so that the module
after the pass is that of `main`. `convert-md-alg-to-gpu` (in place of
`expand-device-primitives`) and `convert-md-alg-to-scf` run the two
lowerings as passes, for the tests and for the split of #558, after which
`md_alg` is lowered by a pass of its own after the policy pass.

## What comes next

The value of the layer is that the stages of a domain algorithm become
visible in the IR. A neighbor build is a histogram, a scan, a scatter, a
sort within bins, a candidate search, and a compaction; a prune and the
interaction of pairs decompose similarly. Once they are `md_alg` ops, a pass
can remove intermediate buffers, skip a sort whose order is not needed, or
fuse stages. Step 1 is therefore not a renaming of the 22 calls: its ops are
the collection algorithms, typed and verified, with a serial definition.
The order of the steps:

1. the collection primitives called by name (this decision);
2. the append with a capacity and its overflow contract, the histogram, and
   select/compact, from the hand-written copies in the groups build and the
   prune;
3. the counting sort and the grid plan as compositions of these ops, used
   by the groups build, the matrix, the ragged rows, and #795;
4. the domain algorithms (build, prune, pair interaction) decomposed into
   collection ops, and the passes that remove buffers, skip sorts, and fuse
   stages;
5. the policy pass, and SIMD and dataflow lowerings.

## Validation

*(filled in with the results)*
