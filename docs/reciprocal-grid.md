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

Final compatibility gates are a corpus against an archived PR base,
the order matrix, five bitwise checkpoints and both D309 shards at the
final clean head. The device sanitizer tier applies if device code
changes. Complete-chain numerical gates precede admission of the route.
