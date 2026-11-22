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
refresh the charge bound whenever charges change. On certificate failure,
discard the narrow result and use a certified wide accumulation before
the transform. Select a scale for the precision being computed and check
the quantization error against the same-grid double oracle.

For cubic gathering, two or four neighboring lanes contract consecutive
z slices. Reduce the derivatives in a fixed shuffle tree and store an
aligned four-component result in original atom order, with one padding
component. Compare integer grids bitwise against a direct wide scatter,
including forced patch escape. Test gather reproducibility at fixed lane
mapping across block sizes. Different lane mappings have different
floating-point summation trees and require a fixed contraction order for
bitwise comparison.

At setup, time the complete force-only transform sequence for the available
combinations of forward integer-conversion and inverse influence load
callbacks, including any explicit sweeps. Select by mesh dimensions,
precision, device and cuFFT version. Keep the explicit path available,
refresh influences on box changes, and retain the explicit energy and
virial reductions on observation steps. Stage (a) uses explicit transforms.
All these choices are internal. Benchmark each phase on the six equal-grid
frames of #933 and investigate any time above 1.15 times its reference
before proceeding.

The device force path passes three buffers from the frame producer to
the two mesh kernels. Cell offsets are i32 half-open ranges, with a final
sentinel equal to the particle count. Each slot has an i32 original atom
index and an aligned record of four mesh-precision values. Its first three
values are periodic coordinates in mesh units and its fourth is the
charge. The local producer groups the first wrapped stencil point on each
axis in cells four grid points wide, with z varying fastest. The last
cell along an axis may be shorter. A shared producer must use the same
frame, box, mesh placement and cell numbering. The order within one cell
is immaterial to the consumer. Gathering scatters an aligned force record
by the slot's atom index. Conversion to the consumer's three Cartesian
components is included in the gather time.
