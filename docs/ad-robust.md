# Robust differentiation

D182, issue #15, is implemented in separate reviewable parts:
activity analysis, scalar rule interfaces, numerical checking, singular
point conventions, and control-file term diagnostics.

## Activity analysis

`ActivityAnalysis` follows operands and values captured by particle, pair,
and tuple kernels. Each result is active, proven inactive, or unknown;
unknown carries a reason and is an error during differentiation. Active
means a path exists from the differentiation variable; it does not promise
that a derivative rule exists or that the derivative is nonzero.

The analysis is shared by scalar derivatives (including coordinates and
cell edges), positional intermediate fields, sum weights, and scalar
parameter derivatives. A kernel argument cannot affect values in its
lexically enclosing scope. Other arguments of a known function or kernel
are independent inputs. An argument of an unknown region is unknown.
Unknown effects and unknown regions never establish independence.

`mdir-opt input.mlir --md-analyze-activity=argument=2` reports all three
outcomes for each result in the body of each potential. This analysis
command reports unknowns without treating them as a successful derivative;
`--md-differentiate` rejects an unknown that a requested derivative needs.

The activity tests include scalar roots, captured parameters, propagation
through fields, and opaque operations. Existing numerical tests cover the
scalar, position, cell, and parameter paths that use the shared analysis.

## Scalar rule interface (D183)

`DerivativeOpInterface` dispatches scalar differentiation. External models
cover arithmetic, math, vector construction/extraction/broadcast, and table
lookup. Each model declares its operands differentiable or structural.
The derivative emitter consults those roles. Conditions, integer indices,
integer powers' exponents, and piecewise-constant operations are structural.
A successful null tangent means zero; failure is distinct.

Clients constructing a dialect registry call `registerDerivativeInterfaces`.
Both MDIR command-line tools register the models. Activity recognizes a
modeled scalar operation rather than trusting its dialect namespace.
`--md-check-derivative-coverage` checks every scalar operation in potentials,
including kernel operations that happen to be inactive at an evaluation.
It reports the missing operation and potential name.

## Numerical checking (D184)

Run `--md-check-derivatives` before differentiation. The pass instruments
existing `md.evaluate` operations; run the resulting module through the
ordinary CPU pipeline in double precision. It checks requested derivatives
only: include forces, virial, and `derivative(N)` requests to check them.
It evaluates the potential at the supplied coordinates, without replacing
them by synthetic coordinates or taking dynamics steps.

Every Cartesian coordinate is perturbed separately. Scalar parameters use
$h = \mathrm{step}\max(1, |x|)$, as do Cartesian coordinates. Forces are
compared with the negative central difference of energy. The virial is
$-\partial U/\partial\epsilon$ with coordinates and the cell strained
together; three stretches and three upper-triangular shears cover the six
independent components of a restricted-triclinic cell. The cell must be
constructed explicitly at the evaluation; an opaque cell argument is an
error. General cell orientations and the other three tensor components
are not independently checked.

The default `step=1e-5 atol=1e-6 rtol=1e-4` compares using
$|a-d| \le \mathrm{atol}+\mathrm{rtol}\max(|a|,|d|)$, where $d$ is
$(U(x+h)-U(x-h))/(2h)$. Choose tolerances for the scale of the potential;
near discontinuities a central difference can cross branches. Nonfinite
results fail. CPU storage and f64 in every precision role are required;
requests to narrow precision or use device storage fail before lowering.

Each perturbation owns a fresh buffer, never changed after field import.
It is exported before freeing it so storage does not put a freed buffer in
its pool. Failures print the potential, quantity, analytic and numerical
values, error, and tolerance to stderr, then abort execution. The checker
is expensive (two energies per coordinate) and intended for small
configurations. It does not alter the normal simulation pipeline.

`test/Integration/check-derivatives.test` runs reproducible random
expressions with seeds 15 and 161 and a deliberately wrong analytic force.
The existing Lennard-Jones, tuple, table, and generalized-Born integration
tests also check their actual configurations against energy differences.

## Singular and branch conventions (D185)

These are the reference semantics, including at ties:

| Point | Convention |
|---|---|
| Pair distance $r=0$ | The geometric conversion divides by $r$; no direction or analytic continuation is invented. An active force is undefined/nonfinite and a numerical check fails. Coincident particles are outside the supported coordinate-gradient domain. |
| `sqrt(0)` | An active tangent uses $x'/(2\sqrt{x})$ and is nonfinite at zero; an inactive argument has a proved zero tangent without a division. |
| `abs(0)` | Select the nonnegative branch, giving $x'$. |
| `min(a,b)` or `max(a,b)` at a tie | Select the second operand's derivative. |
| `select` on an active condition | Differentiate the selected value branch; no derivative of the condition or distributional contribution at its boundary is introduced. |
| `floor`, `ceil`, integer conversions, `step`, `delta` | Piecewise-constant derivative zero, including at a jump. |
| Nonperiodic continuous table endpoints | The endpoint is in range. Differentiate the first/last polynomial interval; strictly outside the range, the value and derivative are zero. The last interval is selected at the upper endpoint. |
| Periodic tables | Wrap to the half-open fundamental interval; use its spline derivative. |
| Discrete table half-integer boundaries | Nearest-point lookup using `floor(x+1/2)`, clamped to the valid indices; derivative zero even where the value jumps. |

A branch derivative at a discontinuity need not agree with a central
difference that crosses the discontinuity. The numerical checker reports
that disagreement; it does not reinterpret it as a proved zero. Domain
errors of logarithms, powers, angles, or collinear geometry likewise have
no invented continuation. Existing runtime checks diagnose nonfinite
simulation states, and the derivative checker rejects nonfinite comparisons.

`derivative-conventions.mlir` evaluates ties, active and inactive square
roots at zero, and coincident radial geometry directly. Table endpoint
conventions follow the interpolation in `Expression::emitSpline` and the
existing tabulated-function boundary tests.

## Control-file term diagnostics (D[ad-term-diagnostics])

Expression emission accepts the control-file term name and records it as a
named MLIR location on every emitted expression operation. Pair, tuple,
centroid, pull, triplet, and external terms supply their names. Quoting is
escaped for MLIR. Activity retains the operation that first made a proof
unknown, so a failure reported through an enclosing sum or field still
points to the named expression that caused it.

The driver regression injects a missing scalar rule into emitted IR and
checks that differentiation names the control-file term in the diagnostic.
This changes diagnostic attribution only; expressions and trajectories
retain their previous arithmetic and defaults.
