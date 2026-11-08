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

| Operation | Legacy Mixed | Legacy Double | Current f64 CPU reference |
|---|---|---|---|
| Position loads and fractional coordinates | f64, scaled local fraction narrowed to f32 | f64 | f64 host views and fractional coordinates |
| Window weights | f32 | f64 | f64 common scalar body |
| Assigned grid | f32 or deterministic i64 at scale $2^{40}$, converted to f32 | f64 or the same deterministic scale, converted to f64 | f64 ordered ideal sum, no quantized path |
| FFT | complex f32 | complex f64 | complex f64 through the existing CPU value bridge |
| Influence map | f32 after f64 edge-table construction | f64 | f64 explicit Gaussian scalar kernel |
| Gathered potential and forces | f32 local arithmetic, requested field conversion | f64 | f64 host sums and fields |
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

## Stopped checkpoint and authorized resumption

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
correction was an owning `SmallVector<Value>`. The maintainer authorized
that narrow change on #792. Its rebuild passed, and the one authorized
selection-test rerun passed 1/1. The classification rule is unchanged.
The earlier stopped source and both failure counts remain recorded for
review. Development resumes from this passed focused checkpoint;
the route is not admitted until its remaining gates pass. Ordinary PME
and D389 remain unchanged.

Before the first complete grid comparison, the independent dense-window
and direct-DFT fixture declares absolute $2\,10^{-9}$ plus relative
$2\,10^{-11}$ bounds on grid energy, forces and charge potential, and
absolute $2\,10^{-6}$ on all nine virial components against simultaneous
affine position/cell central differences. It checks the grid contribution
alone, without self, background or excluded-pair corrections. The initial
execution uses the explicit f64 host reference and the existing FFT bridge;
no Mixed or resident route is admitted by that test.

The first end-to-end fixture run failed before execution because it placed
raw buffer imports inside activity analysis. Moving the fixture to an
explicit `md.function` input boundary preserves independent field arguments;
the function is expanded before `md-inline`. The classification rule is
unchanged. The one controlled rerun also failed before execution: storage
assignment rejects the already-buffered `md_exec.linear_transform` emitted
by the existing FFT value bridge. Both runs had 0 passed and 1 failed;
no energy, force, charge or strain comparison executed.

Development is stopped again under Ruling 13. The proposed next change is
to let storage assignment clone the already-buffered linear transform
through its existing generic path, without relaxing field or ownership
checks. This change is not applied. The passing owning-range correction
and unchanged activity classification are retained.

The maintainer authorized admitting the already-buffered FFT through
`convertGeneric`. That narrow change rebuilt successfully. Its one
authorized fixture rerun failed 0/1 before numerical execution, now because
storage assignment encounters `md.lookup` inside an expanded grid loop.
The generic path's field/runtime-structure refusal, ownership and activity
classification are unchanged. Development stopped again as ruled.

The proposed next correction belongs in the new grid materializer:
obtain an explicit table buffer view, and materialize kernel lookups as
buffer loads before storage assignment. Do not broaden storage assignment
to admit arbitrary semantic lookup operations. That correction is not
applied, and no further fixture run has been made.

The maintainer authorized the table materializer and a narrower rule for
new-path compilation/lowering gaps. Uniform tables now expose explicit
buffer views and kernel lookups become buffer loads. The build passed.
Its affected-fixture run failed 0/1 at a new point: deallocation encountered
an unlowered `md_exec.particle_for` region. Correcting only the fixture's
pass order to lower execution loops before deallocation passed 1/1.
There was no repeated numerical comparison after a mismatch.

The complete CPU grid reference yielded energy 55.1125753975. Absolute
errors against the independent direct DFT and dense window were
$7.11\,10^{-14}$ for energy, $9.77\,10^{-14}$ for force components and
$1.85\,10^{-13}$ for charge potential. The maximum all-component virial
central-difference discrepancy was $4.47\,10^{-9}$. All predeclared bounds
passed. This remains a grid-only f64 CPU reference, not a Mixed or
resident GPU execution claim.

The evaluator-boundary fixture passed 1/1 by verifying the current named
refusal when this composition reaches `md.evaluate`. Integrating its
proved grid roots into the shared `md-differentiate` pass is the next
required step. It has not been changed: the revised ruling requires a
stop before shared-pass or refusal changes. The proposed integration
reuses the tested reverse graph for first derivatives of explicitly
selected Gaussian roots, preserves default and unsupported stencil
refusals, and leaves D389's forward tangent alone. Final compatibility,
admitted-precision and device gates remain outstanding.

The maintainer authorized that shared evaluator integration. The proved-root
dispatch now uses the existing graph proof, checks direct potential geometry
and closed intermediate uses, and applies the outer energy weight to the
reverse results. The build passed. Its one affected focused run had two
passes and two failures: existing bare-stencil virial refusals and the
existing tangent tests passed, while the two new evaluator fixtures hit
the same vector-type assertion in the direct amplitude-field derivative.
The negative mutation matrix stopped at its positive admission preflight;
its negative cases have not run.

Static inspection identifies the new scalar amplitude VJP's use of the
existing vector-only `scaleField` helper: its broadcast requires a vector,
but the charge-gradient element is f64. Development stopped as the ruling
requires after the failed affected run. The proposed correction is a scalar
weighted map confined to the new amplitude-VJP branch, following the
existing scalar reciprocal-field derivative pattern. The shared helper,
classification, tolerances and old refusals would remain unchanged. That
correction is not applied; no second focused run or numerical rerun has
been performed.

Final compatibility gates are a corpus against an archived PR base,
the order matrix, five bitwise checkpoints and both D309 shards at the
final clean head. The device sanitizer tier applies if device code
changes. Complete-chain numerical gates precede admission of the route.

The maintainer authorized the scalar weighted map and refined the rule to
allow corrections confined to code added by this PR. The amplitude-VJP
branch now uses its own scalar `MapParticles` kernel; the old vector helper
is unchanged. Its build passed. The focused batch had three passes and one
new-harness failure: the rectangular-proof mutation replaced a local SSA
name across function scopes. Restricting that replacement to the potential
passed the refusal fixture 1/1, including its positive control and all 19
negative mutations. The admitted evaluator, ordinary bare-stencil refusal
and existing tangent fixtures passed in the preceding batch.

A compiled evaluator comparison with outer weight 2.5 passed 1/1 against
the same independent direct-DFT and affine-strain oracle and predeclared
bounds. It checks weighted grid energy, forces, direct charge-field VJP
and every virial component. These results still admit no Mixed or resident
GPU route.

The new exact-input legacy Double CPU comparison has not reached numerical
execution. Its initial run failed 0/1 while the legacy template parser
loaded the LLVM dialect during pass execution. Adding the interface
attribute to the new fixture's printer declaration preloaded the dialect,
but the next run failed 0/1 at an unavailable C-interface printer symbol.
The attribute has now been moved to the fixture's defined entry point;
that last fixture correction is untested. No pre-existing compiler or
runtime code was changed. Development is stopped for a ruling on another
legacy-fixture run after these two failures. No numerical mismatch or
intermittent failure was retried, and final compatibility gates remain
pending.
