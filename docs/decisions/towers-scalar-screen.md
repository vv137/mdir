# D[towers-scalar-screen] TOWERS distance screens and warp votes

The optional `[execution] towers_shape.screen` selects `"AUTO"`, `"MMA"`,
or `"SCALAR"`. AUTO uses the matrix instruction at compute capability 7.5
and above. Older devices use the scalar screen. Preflight refuses an
explicit MMA request on an older device. The setup log and manifest record
the effective choice and reason. Both the setup trial and production kernels
use that choice for every launch at the selected height.

Both screens import the same lattice features and apply the same padded
threshold. A scalar lane evaluates the four-term dot product for its target
and packs the signs into the source mask. Section 4 of the paper gives the
no-false-negative bound.

The bounded signed 32-bit certification vote uses a subgroup sum.
Compilation for sm_80 and newer enables the NVVM redux instruction. Older
targets use five XOR shuffles and additions. The wide vote retains its
original overflow bound and accumulation. This also fixes #978 on sm_75.
