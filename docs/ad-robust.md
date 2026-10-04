# Robust differentiation

Design for D[ad-robust], issue #15.

Activity has three outcomes: active, proven inactive, and unknown. Unknown
is an error, never a zero derivative. The analysis follows operands and
values captured by kernels and is shared by differentiation with respect
to positions, cell variables, and scalar parameters.

Scalar rules use an op interface. External models cover arithmetic, math,
and vector operations; each model identifies differentiable and structural
operands. Coverage includes the operations emitted by custom expressions
and table interpolation. Unsupported active operations produce an error.

`--md-check-derivatives` instruments evaluations with central differences
in f64 at their supplied coordinates. The checker compares analytic
forces, virials, and requested parameter derivatives with energy changes.
It is intended for small diagnostic configurations, not production runs.

Validation includes activity regressions, rule coverage, numerical checks,
and a seeded random-expression test. The singular and branch conventions
will be stated here and in the specification and white paper.
