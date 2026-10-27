# D[async-copy-contract]

Recorded at its merge.

**Host-device copies have checked completion dependencies and source/destination borrows that forbid a dependent read, conflicting write, free, or reuse while the copy is in flight.**

This first slice of #567 keeps all copies, immediate waits, deferred
readback groups, runtime synchronization, and generated programs as they
are. The checker is read-only analysis state, called after storage
assignment, before GPU conversion, and after `keepReadbacksOnDevice`,
`deferReadbacks`, and stack-release insertion. The inspection pass is
`md-exec-check-transfers`.

Each transfer record retains the copy token, exact source/destination SSA
slices, their roots, allocation-generation identities, and destination
visibility domain. Root SSA identity denotes the generation in one
activation or loop iteration. The source is read-borrowed and the
destination write-borrowed until recognized completion. A free concerns
its entire generation, even through a slice disjoint from the copy.
Distinct allocations and the D87 alias query can prove different roots;
static byte envelopes can prove disjoint slices. Unknown required facts
produce an unsupported-contract diagnostic, never a safe-path assumption.

Only a blocking wait in the copy's block ends the borrows. Its dependencies
must include that copy's token, directly or through asynchronous waits or
dependent copies. A copy with an explicit dependency may access the earlier
copy's buffers after that predecessor completes; its own borrows are
checked independently. This async edge does not license a host access.
SSA dominance and an asynchronous wait alone are not host completion.
A synchronous copy completes at the copy operation. No completion through
an unresolved region, returned token, or recursive call is inferred.

The effect checks use complete D342 operand/row summaries, ordinary MLIR
memory effects, recursively checked regions, and checked function bodies
with actual arguments substituted. External functions need a valid D355
resource summary; a default-memory effect has no sufficient footprint.
The D355 parser is shared with program verification. Loop-carried aliases
inside a called function remain unsupported where substitution does not
establish their relationship to the caller's buffers. D337 set identity
is not alias separation. D334 and D338 describe the shared boundary.

The checker preserves D336's distinct copy completion, staging release,
and file commit. The same readiness/borrow meaning applies to #481 halo
events, but this slice does not change their implementation. Moving waits,
merging additional readbacks, pinned memory, CUDA graphs, or removing the
host-device `mgpuMemcpy` runtime-wide `finish()` require separate measured
changes. A token alone makes no overlap claim.

Focused positive tests cover immediate and synchronous copies, independent
work, source reads, disjoint static slices, asynchronous wait chains,
checked loop/function effects, and the actual deferred-readback conversion.
Negative tests cover destination reads, writes to either borrow, frees and
reuse, casts and subviews, token dominance without completion, unresolved
aliases, and accesses through loop/function boundaries. The checking pass
must leave the printed IR byte-identical. The corpus and full suite are
the final gates, with no numerical tolerance for a generated-code change.
Control keys, numerical methods, file formats, defaults, checkpoints, and
overwrite behavior are unchanged. This does not complete #567's scheduling
and runtime-overlap work.
