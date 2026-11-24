# D[towers-tuner]

Recorded at its merge.

The existing height rule and launch trial selected the fastest height and
split on all eight validation systems. Two RTX 4090 timing runs pinned the
launches in opposite orders, covering four cells per large benchmark and
two on each small protein. A trial comparing all feasible heights agreed
with the old cell choice eight times. Setup therefore retains the D428
height rule and omits the extra cross-height trial.

The enlarged production reference covers 34 pinned launches. It adds two
Trp-cage shapes from a separate prolate PME sweep. Both use HALF, split 2,
one requested block per SM, and full record capacity. CHARMM36m uses
512 threads, panel 448, stripe 416, and 5,824 records. ff19SB + OPC uses
768 threads, panel 352, stripe 512, and 5,632 records.

The original Gaussian PME controls and a separate comparison with this
revision's six-point prolate PME window both favored the chosen protein
launches in each order. With prolate PME, the added shapes took 0.134 and
4.058 µs/step longer on CHARMM36m and ff19SB + OPC, respectively. The
earlier sweep used a different prolate implementation with an order-four
B-spline window. All measured protein launches had zero input-frame retries.

Across eight systems, the new trial chose the minimum two-order mean of
pair plus rebuild in eight of eight height/split cells and seven of eight
exact launches. Its mean and maximum losses were 0.021 and 0.169 µs/step.
The original selector matched eight cells and five launches, giving up
1.192 µs/step on average and 8.877 at most. When ranked by unprofiled
whole steps, both matched eight cells and six launches. The mean and
maximum losses were 0.144 and 0.992 µs/step for the new trial and 0.123
and 0.552 µs/step for the original selector.

Neither shape occurs in the automatic search. It proposes 128, 256, and
512 threads and derives the panel from the input column occupancy and
split. The stripe proposals use the full and warp-rounded half plate span,
subject to thread and shared-memory limits. At 512 threads and split 2,
CHARMM36m produced panel 384 with stripes 256 and 512. ff19SB + OPC
produced panel 512 with stripe 256 and panel 480 with stripe 512. Adding
768 threads alone would still leave the independent tile widths untested.

At the resolved height, each candidate rebuilds a private frame and runs
the moving force and cooperative finish. Pressure-coupled runs time the
fused virial. The finish recovers only on a failed certificate. Production
and trial share the rebuild row and the D431 exclusion grid.

After one measurement per feasible launch, the fastest five receive five
more measurements in shuffled order. If they all share a split, the fastest
candidate of the other feasible split also receives five measurements.
CUDA events bracket the rebuild and the pair kernels within each of nine
consecutive moving steps. The sample is the sum of the two mean intervals.
The smallest median sum wins. Every timed candidate records both intervals.
Each finalist and cell reports the two intervals from its median-sum sample.
An untimed force entry counts stripe halvings across tower blocks on one
rebuild of the input frame. Full worst-case capacity implies zero retries.
The search includes reduced record capacities because each record occupies
eight bytes of shared memory and can affect residency. Each split's record gives its best median.
The margin is the difference between the two split medians. Overlapping
interquartile ranges mark the margin as below noise. For a single split
both comparison fields are null. The log and manifest retain the timings,
winning control fields and trial cost. Explicit pins restrict their axes,
and a continuation uses the saved height.
