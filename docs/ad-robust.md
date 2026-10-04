# Robust differentiation

D[ad-robust], issue #15, is implemented in separate reviewable parts:
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

## Scalar rule interface (D[ad-rules])

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
