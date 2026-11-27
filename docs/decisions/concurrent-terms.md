# D[concurrent-terms]

Recorded at its merge.

**The force terms of a step may run on three device streams, chosen by
`[execution] term_streams`.** `SINGLE` (the default) issues every force term
on one stream. `RECIPROCAL` runs the reciprocal sums of PME on a second
stream, as `md-exec-assign-streams{reciprocal=true}` did (D87, D429).
`TERMS` also runs the loops over tuples (the bonded terms, the 1–4 pairs and
the exclusion corrections) on a third stream, beside the direct sum and the
reciprocal sum. The resolved value is a rule of the request, the target and
the mode (#952): the log line `MDIR: term_streams` and the manifest's
`resolved_choices.term_streams` give the request, the value, the reason,
the inputs and whether the choice changes rounding; `mdir pin` writes it.

**Destinations.** A loop over tuples can run beside the pairs only if the
two write disjoint memory. With `TERMS`,
`md-exec-accumulate-destinations{separate-tuples=true}` splits the chain of
force destinations that a loop over particles adds: the loops over pairs
accumulate into one buffer, the loops over tuples after them into another
from zeros, and the kick adds $F_\mathrm{pairs} + (F_{t_1} + F_{t_2} +
\dots)$ in place of $((F_\mathrm{pairs} + F_{t_1}) + F_{t_2}) + \dots$. The
order of each sum is fixed by the program; the streams do not decide it.
It differs from the single chain in the last bits, so `TERMS` records
`changes_rounding = true`. In the deterministic mode `TERMS` resolves to
`RECIPROCAL` and keeps the tuples in the destination of the pairs, so the
state of a deterministic run is the same to the bit for every value of the
key.

**Placement.** `md-exec-assign-streams{tuples=true}` marks a loop over
tuples without reductions (an ordinary step) if it can move up its block
past a loop over pairs: it moves past the ops it is independent of (D87),
after moving up first a loop over particles that stops it by writing what
it reads (the positions in f32). A loop that would not pass a loop over
pairs goes back where it stood, unmarked. The marked loops carry
`md_exec.lane = 2` and run on the third stream in the order of the program.
A loop that reduces values, which the host may read, stays on the first
stream; the lowering refuses one that is marked.

**Correctness.** D87's argument holds with more than one side op in a
window. The host issues ops in program order; each stream runs its work in
the order it was given; each `mdrtSideBegin` or `mdrtSide2Begin` records an
event on the first stream that the side stream waits for; `mdrtSideJoin`
makes the first stream wait for the last end event of both side streams.
Two ops marked for the same side stream are ordered by that stream. An op
between a side op $S$ and the next join that is not on the stream of $S$
may run at the same time as $S$, so it must be independent of $S$; `findJoin`
places the join before the first that is not, and `verifySide` checks every
window again at lowering, now with ops of the other side stream among
those checked. A run of loops over tuples, a run of rows, and the fused
integration kernel each stay on one stream: a marked loop does not join a
kernel with unmarked ones. The lowering of a marked loop issues its work
between the begin and the end of its stream and allocates nothing there
(P4 of D87), which it checks, and it must launch a kernel there.

**Graph capture.** The forks and the join are event records and waits on
streams. A capture of the first stream follows a fork into a side stream
and ends there at the join, so the windows of a step become parallel
branches of a captured graph, joined before the kick
(`[execution] step_graph`, D439). No host synchronization is added.

**Measured.** See the pull request of this decision; the default stays
`SINGLE`.
