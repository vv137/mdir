# D[towers]

Recorded at its merge.

Particle tower and plate traversal: design for #931, proposed November 22, 2026.

## Selection and lowering

An explicit `[execution] neighbor_structure = "TOWERS"` requests a fresh
particle traversal at every direct-space evaluation. Keep the current AUTO
resolution in `resolveNeighborStructure` throughout qualification. Benchmark
AUTO changes separately on moving trajectories. The additional key is
`[execution] towers_unchecked = false`, applicable only to explicit TOWERS.

The strategy belongs on the neighbor structure and its consumers.
`md-exec-choose-neighbors` follows the structure through loops and branches.
Require the exchange contract of every consumer to permit unique traversal.
Specialize the pair region for force, energy and direct virial according to
its live outputs and effects. Lower the build and traversal with MLIR
builders beside the D426 matrix emitter. Describe reads of positions,
basis, topology and parameter tables, writes of packing scratch and status,
and updates of the requested destinations so the scheduler orders rebuild,
evaluation, recovery and consumption. The CPU reference executes the same
ownership rules and supplies observable IDs and counts for tests.

Generate the arithmetic from the pair region, including the indexed
$C_6,C_{12}$ table (with type-pair overrides), real-space Ewald and the
configured switching terms. Apply the existing unit and precision
conventions before fixed-point conversion. The replay clones the same
arithmetic with the same operand orientation and rounding. Reject a body
whose effects, exchange contract or exclusion correction cannot be
represented by these entries.

## Rebuild and unique ownership

Shaw's Neutral Territory interaction box takes its lateral coordinates from
the tower atom and its height from the plate atom. It may contain neither
endpoint. Shaw extends the tower by the cutoff in both axial directions.
The outer half plate has the thickness of the box and quarter-cylinder
edges. Rounding is part of the import region. TOWERS mirrors the lateral
orientation, stages whole cells, and breaks same-cell ties by sorted slot.
The lateral filter and conservative tensor screen reject excess staged
volume. Section 4.8 derives the stencil and distinguishes Shaw's construction
from the GPU implementation. [Shaw2005](../references.md#shaw2005)

Wrap positions into half-open fractional bins and retain the stable particle
ID. Count and insert, sort each cell on its locality key with ID as the
tie-breaker, pack position and type, and scatter the descriptors using the
inverse stencil. For small grids whose fixed-capacity packing fits in L2,
each sorting warp clears its count for the next insertion. Large grids use a
separate clear. Grow an overflowing allocation and rebuild from the same
coordinates before consuming any descriptor.

Use lateral cell widths at least the cutoff and a layer height at least the
cutoff or half the cutoff. Write $k=1$ or $2$ for the corresponding axial
reach. For different lateral columns, orient the source-to-target offset into
the four-cell negative lateral half shell. The source column and target layer
identify the owner. In one column, a target takes the preceding $k$ layers.
In one cell, the source slot must exceed the target slot. Disjoint panels
and source splits partition this order. Require distinct periodic offsets
(at least three lateral bins and $2k+1$ axial bins). Try a feasible geometry
before diagnosing an unsupported box.

For reduced triclinic cells use perpendicular face heights to choose the
fractional grid. A displacement inside the cutoff projects inside the
cutoff on each unit face normal, giving the same offset bounds. Precompute
Cartesian image shifts from the current basis. Project the target onto
each lateral face normal for conservative slab filtering. Closest-image
replay enumerates the integer intervals bounded by the candidate distance
divided by each face height. Renew geometry and setup when the basis changes.

## Screen, records and output

Stage the source panel and successive target stripes in a common Cartesian
image. Quantize only the screening operands. Choose a power-of-two scale
that bounds their integer magnitudes by 1024. The squared-distance
expansion then fits exactly in the 24-bit significand of the accumulator.
Enlarge the screening radius by the norm of the componentwise endpoint
error bounds, including packing and image arithmetic. Retain ambiguous
candidates. The generated pair region applies the strict physical cutoff.

Compact surviving masks into eight-byte records with a 32-bit source mask
and a packed target, source group and cumulative rank. Split the candidate
rank interval into contiguous ranges whose lengths differ by at most one.
Each worker finds its first record by prefix search and walks the set bits.
On capacity overflow shorten the stripe and retry before accumulating it.

In mixed precision the prototype rounds each force component once to Q18
and energy to Q32 in its native units (kJ/mol/nm and kJ/mol). Use opposite
integer words at the endpoints. Accumulate shared low residues and coarse
quotients with unit $2^{22}$. With at most 512 rounding residuals of at most
half a unit in a slot, the residual sum fits a signed word. Lift each slot
to the global 64-bit total before applying the final range test. Record
failed conversions and cancellation debits. Check the neighborhood and
exclusion count bound for global accumulation and the inclusive signed
output range after all signed corrections.

Evaluate the direct pair blindly, then cancel its exact rounded word for
each excluded pair and add the topology's physical correction. Account for
a canceled failed conversion in the certificate. Leave separately owned
1–4 terms with their existing evaluator. A failed certificate discards all
provisional outputs and replays physical pairs, skipping exclusions, into
wide two-limb accumulators. Report unrepresentable physical contributions
before a consumer can advance the simulation.

A force step instantiates only force arithmetic. An energy step reuses the
packing and launches a separate energy entry. A barostat step also requests
the direct tensor, including periodic image and replay corrections. Recovery
replaces every requested output together. Certification is the default.
With the unchecked opt-in, force certification during minimization,
in deterministic mode if admitted, for ages 0 through 99 after setup or
restart, and at subsequent multiples of 100. Unchecked modular output can
wrap between audits.

### Rectangular output entries

The compiler emits force alone by default. For checked mixed precision,
`energy-body` supplies a symmetric scalar direct energy region and the
corresponding exclusion correction when exclusions are present. The
traversal owns each pair once and consumes the unweighted pair energy.
It converts the mixed result to Q32 once per pair. Each warp reduces
signed two-limb integers before adding its total to the frame. The
certificate checks the signed range of the final sum and the net failed
conversion count. Physical replay skips excluded direct pairs and adds
their physical correction at any distance. The joint publication entry
commits both force and energy only after both recovery checks pass.

`direct-virial` uses the force region in a separate traversal. It skips
excluded direct pairs, evaluates their physical corrections and accumulates
the minimum-image displacement times the force on the target in binary64.
The six components are xx, xy, xz, yy, yz and zz, in kJ/mol. A nonfinite
component sets the shared fatal status before publication. Floating-point
addition order can change tensor bits across launch layouts. Deterministic
mode still requires its separate output and restart qualification.

## Setup and diagnostics

Select FULL or HALF from the atom count, the three perpendicular cell heights and the cutoff in every mode. For a rectangular box the heights are the box lengths.
Measure maximum source column and raw plate size at the selected height.
Try 4, 8 and 16 warps, one or two resident blocks and one or two disjoint
source ranges. Round panel and stripe sizes to warp multiples, cap the
panel at 16 warp groups and two worker spans, and respect the packed
prefix and target widths. Budget the certified tile, table and records
against queried per-block and per-SM shared memory and driver reservations.
Use read-only table access where staging cannot fit. Check actual occupancy.
At that fixed height, time nine rebuilt certified calls per launch trial and remeasure the five finalists
in five shuffled orders. Select by median. Stream longer columns and
retry smaller stripes as positions evolve. Re-select on device or box change.

For an explicit request, report the unsatisfied precondition in
`mdir check` and its JSON neighbor report. Diagnose incompatible list
pruning or rebuild intervals, unsupported pair regions, distributed or
rigid-unit traversal, and cells without a proven unique stencil.
Distinguish a static eligibility error from a frame-dependent rebuild retry.
Report the requested and effective structure and certification policy.
The kernel shape depends on live device and frame measurements.

## Acceptance

Commit and report each stage in order: rectangular rebuild and ownership,
screen and force loop, certificate and recovery, exclusions, energy and
virial, selector, then triclinic. Compare sorted bins and unordered pair
multiplicities against independent all-pairs enumeration, including moving
frames, boundary ties and reversed insertion. Require identical membership
and fixed-point words across launch shapes at fixed geometry. Force recovery
in a test. Compare physical outputs with the binary64 oracle in each admitted
precision, test every refusal and run the final suite at the rebased head.

For each available phase compare generated kernels on the same eight frames
and RTX 4090 with the matching stage 4n entry, geometry, table placement and
residency. Report rebuild clear/insert/sort-pack separately, eight active
block intervals, and uninstrumented kernel and complete-step times. Investigate
a phase over 1.15 times its baseline before advancing. Report moving whole-step
rates against main separately from these fixed-frame captures.

Explicit TOWERS requires mixed precision. In double precision, `mdir check`
must diagnose this requirement. Issue #937 tracks f64 predicates, pair
arithmetic and validation of the accumulation scale. Deterministic mode
remains ineligible until every requested output passes bitwise comparisons
across both launch shapes and restarts. Once qualified, it always uses
certified steps.

The direct compiler entry now emits the rectangular blind force stage from
one antisymmetric pair region. Its CPU implementation enumerates owned spans.
Its GPU implementation uses a conservative screen and distributes the set
bits of its mask records across the block. A record-capacity retry divides
the target stripe before any force is committed for that tile. A checked
direct entry counts failed narrow conversions, validates the final endpoints
and replays the whole frame into two-limb sums if necessary.
The consumer receives a force only after every atom passes. The rectangular
force now cancels excluded direct pairs and their conversion votes, adds
the physical correction and skips the excluded direct terms on wide replay.

`[execution] towers_shape` accepts `"AUTO"` or an inline table. A pin can
supply `threads`, `blocks_per_sm`, `stripe`, `panel`, `records`, `split`
and `cell_height`. Preflight derives omitted tile dimensions for a given
height. It reports the control and AUTO fields and checks device memory
limits. AUTO counts the towers at FULL by multiplying the three bin counts,
each the floor of the corresponding perpendicular face height divided by the cutoff.
Let $M$ be that count and $p=N/M$ the mean number of particles
per FULL tower. When both grids are supported, AUTO chooses HALF for
$M \le p^2/4$ and FULL otherwise. The number of independent towers
favors FULL, while more candidate pairs per tower favor HALF. The
largest value of $M/p^2$ favoring HALF and the smallest favoring FULL
on the rectangular calibration frames were 0.073 and 0.433. Their
arithmetic midpoint rounds to one quarter. On the minimized
311,971-atom octahedron the ratio was 0.212 and HALF was faster.
At setup the driver enumerates the feasible launch fields at that height,
including full and half target stripes and record budgets. It times nine
rebuilt certified force calls on private copies of the starting frame and
remeasures up to five finalists in five shuffled rounds. The lowest median
wins. The compile cache stores the launch by grid, device, pair specialization
and candidate set. A complete pin has a single candidate. Preflight lists
the proposals and checks the derived launch. The run log records the
selected launch and whether setup measured it or reused a cache entry.

At fixed height the force and energy words must be identical across all six
launch parameters. At different heights validation uses the physical force
tolerance against binary64. The strict cutoff operates on a cell-relative
binary32 displacement. At the cutoff rounding boundary a change of cell
origin can change membership by one pair and hence by its entire force.
The resolved height belongs in the run record and checkpoint.
Preflight compiles the mixed pair specialization when the toolchain and
device are available. Setup checks the compiled register count and occupancy
before the first step. As in 4n, the production compiler budget targets up to
two resident blocks. A higher residency pin must fit the compiled kernel.


The triclinic compiler entries take fractional positions and the nine row-major
Cartesian lattice components. Setup computes the face normals from cross
products and stores the Cartesian step for each bin axis. Packed points are
relative to the center of their bin in Cartesian coordinates. The lateral
tests project on the two independent normals. The physical exclusion uses
the closest lattice image even when component rounding chooses a longer
image. The fast path accepts an image shorter than half the minimum face
height. Otherwise each perpendicular height bounds the integer translations
to enumerate. Direct cancellation uses the same two staged endpoints as the
forward traversal.

The internal `finite-at-cutoff` specialization uses the rounded square of
the physical cutoff for pair regions known finite at the endpoint. The
generic mixed specialization retains the D159 guard. The setup trial can
establish this precondition for conventional Amber Gaussian PME and
Lennard-Jones with no additional pair expression. The floating-point direct
virial still requires its own bitwise qualification for deterministic runs.
