# D[groups-directed-outside]

Recorded 2026-11-21.

## Step 1 design

The existing builder and pruner retain their 16-member groups. Within a
group the pair mask selects each unordered pair once. The warp adds the
kernel value to its member and routes the exchange-signed value to the
other member in registers. Between groups both lists include the pair.
Each group computes only its own member's contribution. For symmetric
scalar sums, an in-group evaluation represents two ordered terms and an
outside evaluation represents one.

Each warp keeps the existing member totals in registers. At the end of a
block, a shuffle combines the two half-warps and one atomic add per
component writes each member's total. Several blocks can therefore
contribute to the same group. Scheme B removes the entry receipts and
their per-round global reductions, the cost identified in #912. Revisit
the block traversal only if measurements show that member atomics matter.

Compare one-step forces against the CPU oracle and two independent
baseline GPU evaluations. Time mixed precision against #922 on JAC,
Cellulose, and Large TIP3P in two interleaved runs with reversed order.
Report list occupancy, allocated list bytes, and entry load bytes per
step, distinguishing logical loads from measured memory traffic.
