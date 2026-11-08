# Reciprocal grid composition: slice 2

D[reciprocal-grid], part of #452, implements the second slice of
[the approved design](reciprocal-primitives.md). D390 supplies the
assignment relation, structured spread/gather and common scalar window.

The scope is structured grid maps and spectral reductions, followed by
the explicitly selected rectangular Gaussian Coulomb chain. The existing
unnormalized FFT pairing and influence normalization are retained. The
energy remains a spectral reduction; it is not rewritten as a particle
sum. Self, background, excluded-pair and dispersion corrections retain
their existing ownership.

No ordinary PME caller selects the new pass. No control-file key,
environment variable, Python option, file format or overwrite behavior
changes. D389's forward tangent remains its two-sum construction.

Before admitting execution, the implementation must validate primal
energy, forces, charge fields and all strain components against independent
references and the existing route. The operation-by-operation precision
table, fixed comparison bounds and measured results will be recorded
here before a performance conclusion. Unsupported orders, windows,
influences, cells and derivative compositions receive named diagnostics.
No performance result is claimed by this development checkpoint.

The first structured-grid numerical fixture declares its bounds before
execution: absolute error at most $2\,10^{-12}$ plus relative error
$2\,10^{-12}$ for primal values, JVPs, VJPs and the weighted transpose
identity on small fixed inputs. It tests odd and even last-axis sizes;
only the latter has a Nyquist plane. Complex kernel operands use two real
components. Tensor cotangents use the declared grid pairing, while
uniform scalar and vector inputs use Euclidean pairing, so their VJPs
are weighted reductions rather than pointwise outputs.

## Grid foundation

`md.map_grid` and `md.sum_grid` define complete static domains and pure
straight-line kernels, with explicit uniform inputs. A half-domain tensor
does not certify boundary conjugacy; that proof remains necessary before
an inverse FFT. `md-differentiate-grid` expands explicit JVPs and VJPs
through the same kernel using the real scalar rules. `md-expand-grid-kernels`
provides the explicit tensor/loop reference form. It introduces no resident
execution or fused-strategy permission.

The initial focused run had three deterministic fixture setup failures:
two reused SSA names and one omitted linalg-to-loops pass. No numerical
comparison executed. One controlled correction and rerun passed all three
tests, including the compiled NumPy comparison on odd and even grids.
The compiled comparison checked 542 values, JVPs and VJPs; maximum absolute
error was $7.11\,10^{-15}$. The activity fixture initially omitted the
potential's required position/cell arguments: its first batch had one
pass and one failure. One correction and focused rerun passed 1/1.
The generated-dialect and fixture-cleanup checks passed 2/2.

## Precision record before chain admission

| Operation | Legacy Mixed | Legacy Double | Current slice-2 foundation |
|---|---|---|---|
| Position loads and fractional coordinates | f64, scaled local fraction narrowed to f32 | f64 | No assignment executor added yet |
| Window weights | f32 | f64 | D390 semantic f64 window unchanged |
| Assigned grid | f32 or deterministic i64 at scale $2^{40}$, converted to f32 | f64 or the same deterministic scale, converted to f64 | No grid assignment executor added yet |
| FFT | complex f32 | complex f64 | Existing typed value bridge unchanged |
| Influence map | f32 after f64 edge-table construction | f64 | Generic reference kernel uses f64; chain not admitted yet |
| Gathered potential and forces | f32 local arithmetic, requested field conversion | f64 | No gather executor added yet |
| Grid map arithmetic | f32 spectral products | f64 | f64 scalar or real components of complex f64 |
| Spectral reduction | local arithmetic follows execution precision; row and final sums f64 | f64 | Explicit sequential f64 reference sum |

This foundation admits no Mixed chain and makes no Mixed performance
comparison. The completed chain must record its actual types and bounds
before execution is admitted; a reference f64 chain must not be presented
as matching the legacy Mixed execution types.

The explicit stencil materialization and geometry fixture fixes its bounds
before execution: absolute $2\,10^{-11}$ plus relative $2\,10^{-12}$ for
primal assignment/interpolation and position/cell JVPs and VJPs, and
absolute $5\,10^{-8}$ for central differences. Its reference forms the
full assignment matrix independently from periodic images of the
truncated-power cardinal window. Zero amplitudes, signed inputs, wrapping
and both rectangular and skew cells are included. The development-only
`geometry=true` option on `md-differentiate-stencils` selects the new
geometry rules; the existing default named refusals remain unchanged.
The first compiled run passed all 1,056 primal/JVP/VJP values, with
maximum absolute error $1.84\,10^{-15}$ and central-difference discrepancy
$2.93\,10^{-10}$. The eight geometry request expansions and the existing
default refusal fixture passed 2/2. The execution form is a reference
traversal with explicit host buffers, not a new resident GPU strategy.

## Stopped development checkpoint

The selected-chain implementation is not admitted. Its first build had
one deterministic typed-initializer-list compile error; one correction
and controlled rebuild passed. The new selection fixture then failed
0/1 because the provenance checker used an integer-zero matcher for
floating matrix entries. One correction changed that matcher to the
floating-zero matcher. The controlled rerun also failed 0/1, now at the
activity check of the independent charge argument.

Development stopped under
[Ruling 13](https://github.com/vv137/mdir/pull/719#issuecomment-6576511036):
if the controlled rerun fails, stop. No chain numerical comparison or
final compatibility gate has run. Static inspection identifies a
non-owning `ValueRange` initialized from a temporary initializer list in
the activity loop as a lifetime hazard under C++17. The proposed
correction is an owning `SmallVector<Value>`; it has not been applied or
tested. The stopped source is retained for review. Ordinary PME and
D389 remain unchanged.

Final compatibility gates are a corpus against an archived PR base,
the order matrix, five bitwise checkpoints and both D309 shards at the
final clean head. The device sanitizer tier applies if device code
changes. Complete-chain numerical gates precede admission of the route.
