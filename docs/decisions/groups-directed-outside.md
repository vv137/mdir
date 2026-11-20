# D[groups-directed-outside] Directed outside pairs in GROUPS

## Step 1 design

The existing builder and pruner retain their 16-member groups. Within a
group the pair mask selects each unordered pair once. The warp adds the
kernel value to its member and routes the exchange-signed value to the
other member in registers. Between groups both lists include the pair.
Each group computes only its own member's contribution. For symmetric
scalar sums, an in-group evaluation represents two ordered terms and an
outside evaluation represents one.

A warp must own the complete group before writing its member totals.
Groups can span several 64-entry blocks. The ordinary force pass must
therefore traverse all surviving blocks of its group, including partially
filled blocks. The fused pruning pass already has a group stream.
Member totals must preserve contributions from earlier force terms.

Compare one-step forces against the CPU oracle and two independent
baseline GPU evaluations. Time mixed precision against #922 on JAC,
Cellulose, and Large TIP3P in two interleaved runs with reversed order.
Report list occupancy, allocated list bytes, and entry load bytes per
step, distinguishing logical loads from measured memory traffic.
