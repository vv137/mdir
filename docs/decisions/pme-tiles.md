# D[pme-tiles]

Recorded at its merge.

For the Gaussian cubic B-spline on the existing mesh, assign each particle
to a periodic cell. Pack the current positions and charges in cell order,
retaining the permutation to original atom IDs. Consume the cell offsets,
permutation and packed coordinates from #936 when their frame, cell and
coordinate conventions agree. Otherwise build the order locally on that
step. Account separately for this build and for packing and range checking.

One block accumulates a mesh patch per cell. Round each scaled stencil
contribution to nearest even before its first integer atomic add. Flush
each occupied patch word to its periodic global address. When a stencil
extends beyond the patch, deposit the entire stencil directly in the
global grid with the same rounding. Shared and global addition then give
the same grid for every block size and traversal order.

Let $S=2^e$, $Q$ be the largest absolute packed charge, $P$ the spline
order and $B$ the largest count of bases in any mesh voxel. Along one
axis the sum of the stencil weights at the left ends of their fractional
intervals is one. Replacing each with its maximum adds at most the
positive variation of the cardinal spline, bounded by one. The tensor
product therefore bounds the unrounded absolute contributions to a node
by $8BSQ$. At most $BP^3$ terms are rounded. Require
$B(8.01SQ+P^3)<2^{31}$ for signed 32-bit accumulation, allowing for
floating-point weight evaluation and rounding. The absolute bound also
covers intermediate sums. Check the occupancy of the current frame and
refresh the charge bound whenever charges change. Mixed precision uses $S=2^{23}$, $Q\le2$ and $B\le15$.
Check both inequalities on every evaluation before depositing any narrow
word. On failure, accumulate directly into i64 at the same scale. Double
precision uses i64 with $S=2^{40}$. A stencil visits each node at most
once on the supported meshes. With $N$ particles the wide certificate
requires $N(2S|q_i|+1)<2^{62}$ for every particle. The factor two covers
the evaluated weight product, the added unit covers rounding, and the
extra bit allows for evaluation of the bound in double precision. A
nonfinite charge or failed wide certificate stops the calculation before
the transform.

For cubic gathering, four neighboring lanes each contract one z slice.
Reduce the derivatives in a fixed shuffle tree. The producing lane
writes the three Cartesian components directly to the driver's force
array at the slot's original atom index. Compare integer grids bitwise
against a direct wide scatter, including forced patch escape. Test
gather reproducibility across block sizes with the same contraction order.

At setup, time the complete force-only transform sequence for the available
combinations of forward integer-conversion and inverse influence load
callbacks, including any explicit sweeps. Select by mesh dimensions,
precision, device and cuFFT version. Keep the explicit path available,
refresh influences on box changes, and retain the explicit energy and
virial reductions on observation steps. The first evaluation warms each available combination and measures three
complete transform pairs with CUDA events. The median decides the policy.
The per-axis tables and current inverse cell supply the inverse callback.
Fixed-order runs use the explicit transforms for reproducible rounding.
All these choices are internal. Benchmark each phase on the six equal-grid
frames of #933 and investigate any time above 1.15 times its reference
before proceeding.

The device force path passes three buffers from the frame producer to
the two mesh kernels. Cell offsets are i32 half-open ranges, with a final
sentinel equal to the particle count. Each slot has an i32 original atom
index and an aligned record of four mesh-precision values. Its first three
values are periodic coordinates in mesh units and its fourth is the
charge. The local producer groups the first wrapped stencil point on each
axis in cells eight grid points wide. Each cell contains eight four-point
subcells. Both indices vary fastest in z. A cell's particle range begins
at its first subcell offset and ends after its eighth subcell. Empty
subcells at the mesh boundary have equal consecutive offsets. A shared producer must use the same
frame, box, mesh placement and cell numbering. The order within one cell
is immaterial to the consumer.

For an eligible sum outside deterministic mode, setup warms both the current
scatter strategy and the integer tiled implementation, then alternates three
timed force evaluations of each. CUDA events enclose preparation, the cell
rebuild, transforms, solve and gather on the reciprocal stream. The lower
median wins. The process caches this choice by mesh, particle and listed
counts, mesh precision, geometry, scatter edge, device and cuFFT version.
The first observation calibrates using force-only evaluations before computing
its energy and virial. Deterministic evaluation always takes the integer path.
