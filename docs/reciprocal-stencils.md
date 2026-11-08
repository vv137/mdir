# Particle-grid stencil primitives: slice 1

D[reciprocal-stencils], part of #452. The contracts and additional
requirements are approved by
[the ruling on #781](https://github.com/vv137/mdir/pull/781#issuecomment-6716738603).

## Semantic operations

**md.grid_stencil** constructs an indices-only relation from positions and
cell. Its type names the particle set, three periodic extents, support
order, and B-spline window. Orders 2 through 8 are admitted; each extent
must lie between the order and the signed i32 limit, and their product
must fit signed i64. Thus each particle has exactly $p^3$ distinct entries.
The relation is independent of amplitudes: zero charges retain their
cotangents.

For scaled fractional coordinate $u_a=K_a(xH^{-1})_a$, first reduce the
fractional coordinate modulo one. Entries have unwrapped indices
$b_a+j_a$, where $b_a=\lfloor u_a\rfloor-p+1$ and $0\le j_a<p$.
Grid indices are their nonnegative residues modulo $K_a$. The explicit
continuous kernel argument is the support distance $u_a-b_a-j_a$.
Unwrapped index arithmetic must use i64 before reduction; the i32 extent
bound does not license overflowing an i32 intermediate.
The cell must be finite, nonsingular and periodic in every direction
before execution; slice 1 declares this precondition without adding an
executor. Position and cell are not hidden in the relation's payload.

**md.scatter_stencil** and **md.gather_stencil** are structured pure
reductions. Their one-block weight kernel takes a vector of three f64
support distances and yields one f64 weight. It cannot capture dynamic
inputs, hold effects, or contain nested control flow. Proven immutable
literal constants may hoist during canonicalization. Positions and cell must be
the exact operands of the relation constructor, not merely values of
matching types. Particle-set identity and the full grid shape are checked.

Multiplication by the particle amplitude or grid value and summation over
entries belong to the operation's linear structure. Scatter defines a
fresh full grid initialized to zero; gather defines every particle row,
including zeros. Neither operation mutates an operand or aliases its
storage. Full coverage and ideal additive algebra cannot be overridden by
a list, a partial-coverage attribute or a quantizer. Device indexed writes,
fixed-point accumulation and ownership are execution contracts of later
slices, not effects claimed by these semantic operations.

**md.bspline_window** supplies one cardinal-window body shared by both
directions. **md-expand-stencil-windows** exposes its Cox-de Boor
arithmetic [deBoor2001](references.md#deboor2001). There is no independent
gather formula or handwritten derivative formula.

## Derivatives and explicit refusals

The declared linear transpose of scatter is gather, and conversely.
Both clone the same weight region. **md-differentiate-stencils** expands
explicit **md.stencil_jvp** and **md.stencil_vjp** requests. Operand numbers
are 0 (relation), 1 (positions), 2 (cell), and 3 (amplitudes or grid).
For operand 3, a JVP substitutes its tangent in the same structure; a VJP
uses the declared transpose and the supplied result cotangent.

Positions and cell are continuously active. In this slice, their
structured JVP/VJP requests deliberately fail with diagnostics naming
the operand and derivative direction. They are never silently zero,
including when the same operand also chooses support indices. The
adjoint interface marks them as kernel operands, not structural inputs.
The ordinary evaluator also refuses a virial composition containing these
operations when the energy's cell activity is active or unproven. Checking
only uses of the primary position field would miss a stencil built from
another field. Proven cell inactivity remains admissible; no continuous
dependency is discarded by omission.
Scalar expansion preserves diagnostic provenance on the potential as
**md.stencil_window_activity**, so lowering the scalar spelling or
canonicalizing a local JVP cannot erase this refusal boundary. The
attribute is a source location on **md.potential**, verified as such;
it grants no derivative or fusion capability.
Composed position, cell and influence derivatives are prerequisites of
the slice-2 chain and slice-3 products before those routes are admitted.

The scalar order-four window itself has JVPs and VJPs through the existing
scalar derivative rules applied to the common expanded body. Predicates
choose polynomial pieces; they do not differentiate integer wrapping.
Other orders have primal values but receive a named derivative refusal.
Higher-order composition and curvature gates remain prerequisites of
slice 3; this slice makes no HVP or complete reciprocal derivative claim.

## Precision against the legacy route

The new operations are reference semantic IR, like other **md** field
operations: floating values are f64 before execution precision is assigned.
Only the standalone scalar window is executable in this slice's CPU test.
There is no new Mixed or Double GPU execution route and no performance
conclusion or timing comparison. In particular, the f64 window reference
must not be compared in speed to legacy Mixed f32 arithmetic.

The legacy columns describe the rectangular device template
(**PMEGPU.mlir**) and its type specialization in **MDExecKernels.cpp**.

| Operation | Legacy Mixed | Legacy Double | Slice 1 |
|---|---|---|---|
| Positions as loaded | f64; fractional coordinates/cell in f64, scaled local fraction narrowed to f32 | f64 | Explicit f64 positions/cell; no device load |
| Window weights | f32 | f64 | f64 semantic kernel and standalone CPU reference |
| Assigned grid | f32; deterministic scatter uses i64 at scale $2^{40}$ then converts | f64; deterministic scatter uses the same i64 scale | Full f64 semantic tensor; no execution accumulator |
| FFT | complex f32 | complex f64 | Existing transform unchanged; no new chain |
| Influence and spectral product | f32 | f64 | No new influence operation in slice 1 |
| Gathered potential | f32 local arithmetic, then the requested field conversion | f64 | f64 semantic field; no device gather |
| Forces | f32 local arithmetic and force field | f64 | No new force route |
| Accumulators | Ordinary grid f32 atomics or deterministic i64; local gather f32; energy/virial row reductions and final sums f64 | Ordinary grid f64 atomics or deterministic i64; local gather f64; energy/virial sums f64 | Ideal f64 semantic sums only; no rounding, atomic, energy or virial strategy selected |

Every later implementation slice must replace semantic entries with its
actual execution types before a performance conclusion, and compare only
routes of matching types. Fixed-point rounding is not differentiated as
an ideal linear map.

## Validation contract and rollout

The independent reference is the truncated-power cardinal B-spline,
distinct from the compiled recurrence. Before the first comparison,
**test/Integration/stencil-window.py** fixes absolute bounds: $2\,10^{-13}$
for order-four values, $2\,10^{-12}$ for analytic derivatives and duality,
and $2\,10^{-9}$ for central differences. Additional primal orders use
$2\,10^{-10}$ because the independent power formula suffers cancellation.
The test compiles the actual common body, its JVP and its VJP to host
machine code. It checks support joins and outside support, rectangular
periodic enumeration, unique wrapped indices, partition of unity, signed
amplitudes, zero-charge cotangents and the assignment transpose identity.
The assignment sums are a test reference over compiled scalar weights,
not a production executor of the structured operations.

The focused comparison passed 53 order-four value/derivative cases,
102 additional primal cases across the other admitted orders and 384
periodic entries. Maximum absolute errors were $5.10\,10^{-16}$ for
order-four values, $5.00\,10^{-16}$ for JVPs, $6.83\,10^{-16}$ for VJPs,
$4.17\,10^{-17}$ for window duality, and $2.97\,10^{-11}$ for central
differences. The additional primal orders had maximum error
$2.09\,10^{-16}$; assignment duality had error $8.34\,10^{-17}$.

IR tests check both linear derivative directions, active position/cell
refusals for scatter and gather, full coverage, shape and particle
identity, exact geometry provenance, support order, captured values and
effects. The final-head compatibility gates are the module/lowered corpus,
order matrix, five bitwise checkpoints and full D309 suite. No device
emitter changes, so the conditional device-code sanitizer gate does not
apply. Timings, complete-chain NumPy/torch-pme comparisons, new-route
Mixed/Double GPU numerical comparisons, strain and higher derivatives
apply when the corresponding execution chain is admitted in later slices.

No ordinary PME builder emits these operations. The reciprocal sum, FFT
path and default strategies stay unchanged. #778's two-reciprocal-sum
forward tangent remains as it is; its possible migration is decided by
measurement in slice 3.
