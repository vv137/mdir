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
the inputs and `changes_rounding`; `mdir pin` writes the key, and Python
takes `Execution.term_streams`. On the CPU target every value resolves to
`SINGLE`. With `SINGLE` the lowered program is that of before (checked on
the Amber suite controls).

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
`RECIPROCAL` and keeps the tuples in the destination of the pairs, so a
deterministic run gives the same state to the bit for every value of the
key.

**Placement.** `md-exec-assign-streams{tuples=true}` marks a loop over
tuples without reductions (an ordinary step) if it can move up its block
past a loop over pairs: it moves past the ops it is independent of (D87),
after moving up first a loop over particles that stops it by writing what
it reads (the positions in f32). A loop that would not pass a loop over
pairs goes back where it stood, unmarked, and so do the ops moved for it.
The marked loops carry `md_exec.lane = 2` and run on the third stream in the
order of the program. A loop that reduces values, which the host may read,
stays on the first stream; the lowering refuses one that is marked. A
marked loop does not join a kernel with unmarked loops (a run of tuples, a
run of rows, the fused integration kernel of D110).

**Issue order.** The host issues the work of a step in program order, a
few microseconds a launch, so the marked ops at the top of a window delay
the issue of the loops over pairs. A placement that issued the side work
after the pairs instead, from an event recorded where it stood, was
measured and not kept: on JAC (RTX 4090, LAUNCH) it took the reciprocal
sum's overlap with the pairs from 21 µs a step to 3 (TOWERS) and from 8 to
0 (the default lists), and the step from 97.7 to 98.4 and from 101.8 to
113.2 µs in the trace, since kernels issued after the pair kernel wait for
its blocks to drain.

**Correctness.** D87's argument holds with two side streams and several
side ops in a window. The host issues ops in program order; each stream
runs its work in the order it was given. A side op $S$ starts after the
event that its begin (`mdrtSideBegin`, `mdrtSide2Begin`) records on the
first stream; the first stream waits at `mdrtSideJoin` for the last end
event of both side streams. So every op before $S$ runs before $S$, every
op from the join on runs after it, and only ops between the two may run at
the same time as $S$. Ops marked for the stream of $S$ are ordered by that
stream. Every other op in that interval, of the first stream or of the
other side stream, must be independent of $S$: `findJoin` places the join
before the first that is not, and `verifySide` checks the interval again
at lowering. Moving $S$ up past an independent op is the same lemma.

*Premise P1 for loops over tuples* (D87 asks that marking another op audit
its declared effects). A storage-form `md_exec.tuple_for` declares reads of
its incidence structure, its positions, its parameters and its `ins` (the
per-tuple tables among them), reads and writes of its `outs` (writes alone
where it overwrites), and its scratch; `storageEffects` adds a read of each
table that its kernel looks up. Its lowering reads, besides, the structures
of the tuples that the runtime keeps (D253) and the plans of bonded work
units, which no op of the ordinary step writes, and the block memory of its
kernel. It allocates nothing on the side stream, which the lowering checks
(P4), and it must issue a kernel there.

**Graph capture.** The begins, ends and the join are event records and
waits on streams, so a capture of the first stream follows them into the
side streams; a captured step (`[execution] step_graph`, D439) contains the
three chains as branches joined before the kick. The plain TOWERS graph
builder accepts these calls and refuses a window that crosses the end of
the drift phase, or that is not joined within the step. No host
synchronization is added.

**Measured** (RTX 4090, 2 fs, two jobs of three fresh processes per arm
in alternating order, ratio of the median ns/day to `main`; white paper,
Section 10.8). `RECIPROCAL`: JAC NVE 1.097, JAC NPT 1.069, Factor IX
0.96, Cellulose 0.94, STMV 0.90, TIP3P 10k 0.96, TIP3P 20k 1.07, and with
TOWERS and graph replay JAC NVT 1.14, TIP3P 10k 1.16, TIP3P 20k 1.16.
`TERMS`: 1.149, 1.093, 0.94, 0.86–0.87, 0.83, 1.16, 1.07, and 1.14, 1.17,
1.17. On one stream the bound of full concurrency is 32–56% below the
step (Table 10.10); in the graph traces the best schedule removed 33–48%
of the time between the step and the bound. On the large systems the
pair kernel fills the device and the kernels beside it slow it, and
TIP3P 10k with `TERMS` ran in two bands by process (#953). So the default
stays `SINGLE`; the three schedules are candidates for a timed choice
(#953).
