### Changed

- Construct term fragments and custom-expression scalar kernels with typed `OpBuilder` operations at their shared potential insertion sites, preserving generated programs and loop fusion (D[builder-term-ir], #568, #708).
- Diagnose malformed builder assembly at its construction boundary instead of passing it to a later consumer (D[builder-term-ir], #706, #709).
